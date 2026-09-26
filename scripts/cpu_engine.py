#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The real patched `sglang-jax` engine on CPU, for the gates that need one.

    python3 scripts/cpu_engine.py DIR    # build the patched tree in DIR and print its path

The tree is `sglang-jax` at `SGL_COMMIT` (default eb061d8) with the three patches
`bootstrap_tpu_vm.sh` applies: `sglang-jax-877.patch` through `git am`, then
`steering-hook.patch` and `qwen3-steering-hook.patch` through `git apply`. A commit on top keeps
the new `layers/steering.py` in the tree. The engine imports from the tree's `python/` directory.

Where the tree comes from, first match wins:

- `SGLANG_JAX_TREE`: a tree this script built, used as it stands. `test_all.sh` builds one and
  sets it.
- `SGLANG_JAX_REPO`: a full `sglang-jax` clone that holds `SGL_COMMIT`. The build clones it with
  `--shared` into a temporary directory.
- neither: a blobless clone of `github.com/sgl-project/sglang-jax`, then the build.

The steering gates' rig, `steering/cpu_rig.py`, takes its tree from here too.

The engine's runtime imports are in `upstream/models/requirements.txt`. `require_engine` names
the ones this interpreter can't find before it builds anything.

`write_checkpoint` saves a tiny random-weight Qwen3 with `save_pretrained`, plus a byte-level BPE
tokenizer trained on a fixed word list, which `write_tokenizer` also writes alone for the gates
that need a real tokenizer and no model. `open_engine` starts the real Engine on it, on CPU, in
this process: the scheduler and the detokenizer run as threads, so a test calls `capture()`
directly and the engine never spawns a copy of the test. `reference_hidden_states` runs the same
checkpoint through `transformers`: in float32 for the reference a capture gets checked against,
and in bf16 for a noise floor.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
UPSTREAM_URL = "https://github.com/sgl-project/sglang-jax"
SGL_COMMIT = os.environ.get("SGL_COMMIT", "eb061d8")
PATCH_877 = "sglang-jax-877.patch"
STEERING_PATCHES = ("steering-hook.patch", "qwen3-steering-hook.patch")
STAMP = os.path.join(".git", "cpu-engine-patches")

# The engine's runtime imports that the other gates don't use, by module and by package name.
RUNTIME_IMPORTS = {
    "llguidance": "llguidance",
    "pathwaysutils": "pathwaysutils",
    "PIL": "pillow",
    "pybase64": "pybase64",
    "requests": "requests",
    "setproctitle": "setproctitle",
    "uvicorn": "uvicorn",
}
INSTALL = "uv pip install -r upstream/models/requirements.txt"

# The tiny checkpoint. Six blocks give six capture slots, 0 to 5.
NUM_LAYERS = 6
D_MODEL = 64
VOCAB = 512
CONTEXT = 2048

# A character no training text holds, so it stays one token. Its embedding row is scaled far past
# what float16 holds, the way a sink token's activations run large in a real model. Only a
# prompt that carries it gets the large rows.
MASSIVE_TEXT = "#"
MASSIVE_VALUE = 7.0e4

WORDS = (
    "the a of and to in is was for on that with as by at from his her it an were are which this be "
    "or has had not first one their its new after but who they have been also two more time can "
    "other when she there only into all during most over city world year may three river house "
    "music light stone water north south game line story field night power paper table window"
).split()

# The engine settings every CPU gate starts from. `enable_single_process` runs the scheduler
# and the detokenizer as threads in this process. The flash attention backend falls back to the
# native one on CPU and says so. `chunked_prefill_size` sits at one token bucket, so a batch
# longer than 64 tokens prefills over several passes, the path a long prompt takes on a chip.
# `device_indexes` puts the engine's one-device mesh on CPU device 0, so a process that forces
# several host devices for its own checks can still start it. `disable_overlap_schedule` keeps
# the forward pass on the scheduler thread: with overlap on, the worker thread can copy a device
# array to the host while the interpreter shuts down, and the process aborts with exit 134 after
# every check passed.
CPU_SETTINGS = {
    "device": "cpu",
    "dtype": "float32",
    "chunked_prefill_size": 64,
    "max_total_tokens": 4096,
    "enable_single_process": True,
    "disable_overlap_schedule": True,
    "device_indexes": [0],
    "log_level": "error",
    "random_seed": 0,
}

_tree = None


def _git(repo, *args, stdin=None):
    """Run git and raise with its whole output when it fails."""
    with open(stdin, "rb") if stdin else open(os.devnull, "rb") as fp:
        done = subprocess.run(["git", "-C", repo, *args], stdin=fp, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo} with exit {done.returncode}:\n"
            + done.stdout.decode(errors="replace")
            + done.stderr.decode(errors="replace")
        )
    return done.stdout.decode(errors="replace")


def patch_fingerprint() -> str:
    """SGL_COMMIT and the SHA-256 of each patch the tree takes, one per line."""
    lines = [f"commit {SGL_COMMIT}"]
    for name in (PATCH_877, *STEERING_PATCHES):
        with open(os.path.join(REPO_ROOT, "upstream", name), "rb") as fp:
            lines.append(f"{hashlib.sha256(fp.read()).hexdigest()}  {name}")
    return "\n".join(lines) + "\n"


def build_tree(dest: str, source: str | None = None) -> str:
    """Clone `source` (a local clone, or GitHub) into `dest`, check out SGL_COMMIT, apply the stack."""
    if os.path.exists(dest):
        raise FileExistsError(f"{dest} exists; build_tree writes a new tree")
    if source:
        clone = ["git", "clone", "--quiet", "--shared", source, dest]
    else:
        # Every commit, and file contents only for what the checkout needs.
        clone = ["git", "clone", "--quiet", "--filter=blob:none", UPSTREAM_URL, dest]
    done = subprocess.run(clone, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(clone)} failed:\n{done.stdout}{done.stderr}")
    _git(dest, "checkout", "--quiet", SGL_COMMIT)
    who = ["-c", "user.email=cpu-engine@local", "-c", "user.name=cpu-engine"]
    _git(dest, *who, "am", "--quiet", stdin=os.path.join(REPO_ROOT, "upstream", PATCH_877))
    for name in STEERING_PATCHES:
        _git(dest, "apply", os.path.join(REPO_ROOT, "upstream", name))
    _git(dest, "add", "-A")
    _git(dest, *who, "commit", "--quiet", "-m", "steering hooks")
    with open(os.path.join(dest, STAMP), "w") as fp:
        fp.write(patch_fingerprint())
    return dest


def stacked_tree() -> str:
    """The patched tree, found or built once per process, with its `python/` on `sys.path`."""
    global _tree
    if _tree is not None:
        return _tree
    tree = os.environ.get("SGLANG_JAX_TREE")
    if tree:
        stamp = os.path.join(tree, STAMP)
        if not os.path.exists(stamp):
            raise SystemExit(f"SGLANG_JAX_TREE={tree} has no {STAMP}; build it with "
                             f"python3 scripts/cpu_engine.py DIR")
        with open(stamp) as fp:
            if fp.read() != patch_fingerprint():
                raise SystemExit(f"SGLANG_JAX_TREE={tree} was built from other patches or another "
                                 f"commit; build a new one with python3 scripts/cpu_engine.py DIR")
    else:
        source = os.environ.get("SGLANG_JAX_REPO")
        if source and not os.path.isdir(os.path.join(source, ".git")):
            raise SystemExit(f"SGLANG_JAX_REPO={source} isn't a git clone")
        work = tempfile.mkdtemp(prefix="cpu-engine-")
        atexit.register(shutil.rmtree, work, True)
        print(f"building sglang-jax at {SGL_COMMIT} with the capture and steering patches, from "
              f"{source or UPSTREAM_URL}", flush=True)
        tree = build_tree(os.path.join(work, "sglang-jax"), source or None)
    sys.path.insert(0, os.path.join(tree, "python"))
    _tree = tree
    return tree


def require_imports():
    """Stop with the package names for the engine imports this interpreter can't find."""
    missing = [
        pkg for module, pkg in RUNTIME_IMPORTS.items() if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise SystemExit(
            f"the engine can't start without {', '.join(missing)}. Install what it needs into "
            f"this venv:\n  {INSTALL}"
        )


def require_engine():
    """Import the patched Engine, or stop with the packages it's missing."""
    require_imports()
    stacked_tree()
    try:
        from sgl_jax.srt.entrypoints.engine import Engine
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"the engine can't import {exc.name!r}. Install what it needs into this venv:\n"
            f"  {INSTALL}"
        ) from exc
    return Engine


def write_tokenizer(path: str):
    """The checkpoint's byte-level BPE tokenizer, trained on WORDS and saved to `path`.

    Returns the `tokenizers` object. Text outside WORDS falls back to byte tokens, so a character
    past ASCII spans several.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    os.makedirs(path, exist_ok=True)
    corpus = [" ".join(WORDS[i:] + WORDS[:i]) for i in range(len(WORDS))]
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    raw.train_from_iterator(corpus, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    tokenizer.save_pretrained(path)
    return raw


def write_checkpoint(path: str) -> str:
    """A tiny random-weight Qwen3 and its tokenizer, written with `save_pretrained`."""
    import numpy as np
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    raw = write_tokenizer(path)
    massive = raw.token_to_id(MASSIVE_TEXT)
    if massive is None or raw.encode(f"the {MASSIVE_TEXT} city").ids.count(massive) != 1:
        raise AssertionError(f"{MASSIVE_TEXT!r} doesn't tokenize to one token of its own")

    # head_dim stays at 128. The KV pool rounds head_dim up to a multiple of 128, and at eb061d8
    # the native attention backend hands o_proj that padded width, so a narrower head fails the
    # matmul on CPU.
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=D_MODEL,
        intermediate_size=2 * D_MODEL,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=CONTEXT,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=0,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                param.normal_(0.0, 0.1)
        signs = np.where(np.random.default_rng(0).random(D_MODEL) < 0.5, -1.0, 1.0)
        model.model.embed_tokens.weight[massive] = torch.tensor(MASSIVE_VALUE * signs)
    model.save_pretrained(path, safe_serialization=True)
    return path


def tokenize(model_path: str, texts) -> list:
    """Token ids for each text, with the tokenizer the engine loads from the same directory."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return [tokenizer(text)["input_ids"] for text in texts]


def open_engine(model_path: str, capture: bool = True, batch_size: int = 8,
                token_padding: int = 64, **overrides):
    """The real Engine on CPU, in this process, with the capture flag on unless `capture=False`.

    It goes through `capture_activations.open_engine` or `engine_settings`, the same way the
    scripts build it, with `CPU_SETTINGS` and `overrides` on top. Either way the payload log gets
    every keyword the Engine receives.
    """
    Engine = require_engine()
    sys.path.insert(0, HERE)
    import capture_activations

    settings = dict(CPU_SETTINGS, **overrides)
    if capture:
        engine = capture_activations.open_engine(
            model_path, batch_size=batch_size, token_padding=token_padding, **settings
        )
    else:
        payload = capture_activations.engine_settings(
            batch_size=batch_size, token_padding=token_padding, **settings
        )
        capture_activations.log_request(call="Engine", model_path=model_path, **payload)
        engine = Engine(model_path=model_path, **payload)
    # The engine reaps its child processes from a SIGCHLD handler. A single-process engine has
    # none, and the handler would reap a subprocess this process starts later, which then
    # reports exit 0 whatever happened. Put the default back.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    return engine


_references = {}


def reference_hidden_states(model_path: str, sequences, dtype: str = "float32") -> list:
    """`transformers` hidden states per sequence: `[seq, NUM_LAYERS, d_model]`, as float64.

    Entry `i` of `output_hidden_states` is the stream entering block `i`, which is capture slot
    `i`. The last entry is the final norm's output, which has no slot, so it's dropped. `dtype`
    picks the forward's precision: float32 for the reference, bfloat16 for a noise floor.
    """
    import numpy as np
    import torch
    import transformers

    key = (model_path, dtype)
    if key not in _references:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype)
        )
        model.eval()
        _references[key] = model
    model = _references[key]
    out = []
    with torch.no_grad():
        for ids in sequences:
            states = model(input_ids=torch.tensor([list(ids)]), output_hidden_states=True)
            stacked = [h[0].to(torch.float64).numpy() for h in states.hidden_states[:-1]]
            out.append(np.stack(stacked, axis=1))
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    source = os.environ.get("SGLANG_JAX_REPO")
    print(build_tree(os.path.abspath(argv[0]), source or None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
