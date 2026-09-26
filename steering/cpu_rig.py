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

"""Serve a tiny random Gemma 4 on CPU through the patched `sglang-jax`, for the engine gates.

    from cpu_rig import Rig
    rig = Rig(workdir)
    proc = rig.run("scripts/capture_activations.py", "--model-path", rig.model, ...)

`test_pick_feature.py` captures through it and `test_compare.py` steers through it. `Rig` takes
the patched `sglang-jax` tree from `scripts/cpu_engine.py`, the tree the `scripts/` engine gates
use, and each gate runs the repo's own command lines against that engine in a subprocess, the way
the README runs them on a TPU.

The model is Gemma 4 26B-A4B's text stack at four layers of 64 dim with random weights. Its config
is the checkpoint's own with the sizes cut down, so the engine takes the `gemma4` path the Run-it
chain takes, experts and shared MLP included. The tokenizer is the real one, fetched from the Hub
at a pinned revision, so the chat template and the BOS rules are the ones the chain meets.

Two things differ from the checkpoint. The CPU attention backend can't write the engine's
sliding-window KV pool, and one plain pool holds one head layout, so all four layers are full
attention and the engine runs with `disable_hybrid_swa_memory`. The backend also pads every head
to 128 and doesn't slice the padding back off, so the heads are 128 wide.

It needs the engine's runtime imports from `upstream/models/requirements.txt`, `git`, and network
access to the Hugging Face Hub. `scripts/cpu_engine.py` says where the tree comes from:
`SGLANG_JAX_TREE`, `SGLANG_JAX_REPO`, or a clone of GitHub.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load_shared():
    """`scripts/cpu_engine.py`, loaded from its path, whatever `sys.path` holds."""
    spec = importlib.util.spec_from_file_location(
        "scripts_cpu_engine", os.path.join(REPO, "scripts", "cpu_engine.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _load_shared()

CHECKPOINT = "google/gemma-4-26B-A4B-it"
CHECKPOINT_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

# The checkpoint's text config with these sizes swapped in. Every other field, from the RoPE
# parameters to the logit softcap and `attention_k_eq_v`, comes from the real config.json.
TINY = {
    "hidden_size": 64,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "layer_types": ["full_attention"] * 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "global_head_dim": 128,
    "num_global_key_value_heads": 1,
    "num_experts": 4,
    "top_k_experts": 2,
    "max_position_embeddings": 4096,
}
D_MODEL = TINY["hidden_size"]

# What `--engine-arg` sets for a CPU run. The rest of `capture_activations.engine_settings` stands.
CPU_ENGINE = {
    "device": "cpu",
    "disable_hybrid_swa_memory": True,
    "mem_fraction_static": 0.3,
    "max_total_tokens": 4096,
    "chunked_prefill_size": 256,
    "grammar_backend": "none",
    "disable_overlap_schedule": True,
    "random_seed": 0,
}
BATCH_SIZE = 4
TOKEN_PADDING = 256

# Capture prompts. Short enough that a batch of four fits the 256-token prefill bucket.
PROMPTS = [
    "The river froze early that winter, and the ferry stopped running.",
    "A good bread starter needs flour, water and a warm kitchen.",
    "The committee voted to delay the bridge repairs until spring.",
    "She tuned the violin before the audience took their seats.",
    "Most comets spend centuries in the cold far from the sun.",
    "The library extended its hours during the exam season.",
    "He planted tomatoes along the south wall of the garden.",
    "The new train line cut the commute to twenty minutes.",
    "Volcanic soil holds water well and grows strong vines.",
    "The museum opened a wing for early printing presses.",
    "A sudden storm grounded every flight out of the city.",
    "The recipe calls for two eggs and a pinch of salt.",
    "Engineers tested the dam gates before the rainy season.",
    "The chess club meets on Thursday evenings in the hall.",
    "Coral reefs grow slowly and recover slowly from heat.",
    "The old mill still turns when the river runs high.",
    "Her first novel follows a family across three wars.",
    "The factory switched to solar power last autumn.",
    "Owls hunt at night and rest in hollow trees by day.",
    "The mayor promised new lights for the harbor road.",
    "A healthy adult heart beats about seventy times a minute.",
    "The team rewrote the parser to handle nested quotes.",
    "Snow closed the mountain pass for most of February.",
    "The bakery sells out of rye loaves before noon.",
    "Glaciers carve deep valleys over thousands of years.",
    "The orchestra played the symphony without a conductor.",
    "Bees carry pollen home in baskets on their hind legs.",
    "The court ruled that the contract was never signed.",
    "Farmers rotate crops to keep the soil from wearing out.",
    "The satellite sends weather images every fifteen minutes.",
    "He fixed the leaking tap with a new rubber washer.",
    "The festival draws musicians from across the region.",
    "Salt water boils at a slightly higher temperature.",
    "The school added a course on reading old maps.",
    "Wolves returned to the valley after fifty years away.",
    "The printer jammed halfway through the annual report.",
    "Tea plants grow best on cool and misty hillsides.",
    "The archive holds letters from the city's founders.",
    "A lighthouse keeper once lived on that rocky island.",
    "The runners crossed the finish line in a steady rain.",
]


def tiny_gemma4(path: str, seed: int = 0) -> str:
    """Write the tiny checkpoint to `path` and return it.

    The weights come from `transformers`' own Gemma 4 text model at the tiny sizes, redrawn at a
    standard deviation of 0.1 so the stream isn't close to zero. `config.json` then takes the
    checkpoint's layout, a `gemma4` config holding a `text_config`, because that's the layout
    `sglang-jax` reads its head sizes from.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    def fetch(name):
        return hf_hub_download(CHECKPOINT, name, revision=CHECKPOINT_REVISION)

    with open(fetch("config.json")) as fp:
        real = json.load(fp)
    text = dict(real["text_config"], **TINY)
    text["dtype"] = "float32"
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(
        Gemma4TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
    )
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                param.normal_(0.0, 0.1)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    layout = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": real["model_type"],
        "text_config": text,
        "tie_word_embeddings": real["tie_word_embeddings"],
        "bos_token_id": text["bos_token_id"],
        "eos_token_id": real["eos_token_id"],
        "pad_token_id": text["pad_token_id"],
        "transformers_version": real["transformers_version"],
    }
    with open(os.path.join(path, "config.json"), "w") as fp:
        json.dump(layout, fp, indent=2)
    for name in TOKENIZER_FILES + ("generation_config.json",):
        shutil.copy(fetch(name), os.path.join(path, name))
    return path


def engine_args(**overrides) -> list:
    """`--engine-arg KEY=VALUE` pairs for a CPU run, each VALUE in the JSON the flag parses."""
    settings = dict(CPU_ENGINE, **overrides)
    args = []
    for key, value in settings.items():
        text = value if isinstance(value, str) else json.dumps(value)
        args += ["--engine-arg", f"{key}={text}"]
    return args


def results(stdout: str) -> list:
    """Every `RESULT {...}` line a repo script printed, as dicts."""
    prefix = "RESULT "
    return [
        json.loads(line[len(prefix):])
        for line in stdout.splitlines()
        if line.startswith(prefix + "{")
    ]


class Rig:
    """A patched checkout, the tiny checkpoint, and the environment that runs scripts on them."""

    def __init__(self, workdir: str):
        shared.require_imports()
        self.workdir = workdir
        self.checkout = shared.stacked_tree()
        self.model = tiny_gemma4(os.path.join(workdir, "tiny-gemma4"))
        env = dict(os.environ)
        paths = [os.path.join(self.checkout, "python")]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["JAX_PLATFORMS"] = "cpu"
        # One CPU device, whatever the calling gate forced for its own mesh.
        env.pop("XLA_FLAGS", None)
        # Every engine start here compiles the same few programs, so the later ones read them back.
        env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(workdir, "jax-cache")
        env["TOKENIZERS_PARALLELISM"] = "false"
        self.env = env

    def run(self, script: str, *args, timeout: float = 1800) -> subprocess.CompletedProcess:
        """Run a repo script with the venv's interpreter and print everything it wrote."""
        cmd = [sys.executable, os.path.join(REPO, script), *[str(a) for a in args]]
        proc = subprocess.run(
            cmd, cwd=REPO, env=self.env, capture_output=True, text=True, timeout=timeout
        )
        print(f"  $ {shlex.join([script, *[str(a) for a in args]])}")
        print(f"    exit {proc.returncode}")
        for name, text in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            for line in text.splitlines():
                print(f"    {name} | {line}")
        return proc
