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

"""Correctness gate for the Kimi K3 model patch and its capture hook.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 upstream/models/test_kimi_k3_model.py

Eleven checks. Every check runs the patched source, either pulled out of the
patched file by AST or imported from the patched checkout. Nothing here retypes
the implementation, and every architecture claim is measured against the
published `config.json`, safetensors headers and tensor bytes rather than
against a constant in this file.

1. Both patches apply to `sglang-jax` at `SGL_COMMIT` (default `eb061d8`) with
   `sglang-jax-877.patch` and both steering patches on it, the tree
   `scripts/bootstrap_tpu_vm.sh` builds, and every file they touch compiles.
2. MXFP4 dequantization and the sliced block reader, against an E2M1 decoder
   written from the bit fields.
3. The `situ` activation, against the published formula in float64 and against
   `SituAndMul` out of the modeling file the checkpoint repo ships.
4. `attn_res_mix`, against a float64 mixture written with unfolded weights and
   an explicit per-row softmax, and against the shipped `_apply_attn_res`.
5. The block-residual stack, decoder layer, model and capture hook together, on
   a data-sharded input under the engine's Explicit mesh, against a float64
   reference of the whole algorithm.
6. The KDA decay layout, against the `A_log` values the published checkpoint
   ships: one per head, padded with zeros to `head_dim`. Then the Mega KDA
   kernel on a per-head decay against a float64 recurrence.
7. The layer split and the state each half costs, through the patched config
   class reading the published `config.json`.
8. The name the registry has to hold and the config class that reaches the
   language tower, and an override of the layer count that has to reach the
   model and the state pools alike.
9. The model the loader builds: the runtime flags the server writes, every
   weight mapping against the published checkpoint's own shapes, and the MXFP4
   expert stacks read into the parameters `nnx.eval_shape` builds.
10. The served model. A tiny checkpoint in the published layout goes through
    the real `JAXModelLoader`, `ModelRunner`, startup precompile and the
    scheduler's batch code, and its logits after prefill and two decode steps
    match a float64 forward written from the published modeling file, at
    `--ep-size` 1 and 2 and at `--dp-size 2`. One `--ep-size 2` launch serves
    a five-layer file cut to four by `--json-model-override-args`, and two set
    `--ep-dispatch-algorithm` static and dynamic with two redundant experts.
    The default `--attention-backend fa` has to be refused off TPU by name, and
    `NativeAttention`'s TPU branch has to keep each data rank on its own KV.
11. The loader's other paths: a BF16 load with the router bias and `A_log`, a
    dummy load, `--model-layer-nums`, and a quantization config.

Every check carries mutants of the patched source, of the patched output, or of
the weights. Every mutant has to be caught. A mutant that slips through fails
the run, and a mutation operator that can't find its target fails the run as
well. A result that isn't finite fails every gate.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download.
`SGL_COMMIT` picks the commit, as it does for every gate. The published files
come from one pinned revision of the Hub repo, and each one has to match its
pinned SHA-256 before the test reads it. They're cached under `KIMI_K3_CACHE`,
or `~/.cache/kimi-k3-published/<revision>` when that isn't set. Check 10 runs
the engine's `ModelRunner`, which also imports
`pybase64` and `llguidance`. Without them check 10 fails and the run ends on the
`uv pip install` line, with the pins the checkout's `pyproject.toml` carries.
`LOG_PAYLOADS=1` prints the `ServerArgs` of every engine launch and every request
check 10 sends.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback
import urllib.request

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The KDA kernels are Pallas kernels. Off TPU they run in interpret mode.
os.environ.setdefault("PALLAS_INTERPRET", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

UPSTREAM = "https://github.com/sgl-project/sglang-jax"
HERE = os.path.dirname(os.path.abspath(__file__))

# The capture-hook suite carries `corrupt`, the check 1 control the gpt-oss,
# Nemotron 3 and steering suites import too.
sys.path.insert(0, os.path.join(HERE, os.pardir, "capture-hooks"))
from test_capture_hooks import corrupt  # noqa: E402

# The tree every patch here is a diff against: eb061d8 with the capture patch
# and both steering patches on it, which is what scripts/bootstrap_tpu_vm.sh
# builds and scripts/verify_patches.sh checks. The SGL_COMMIT environment
# variable moves the commit, as it does for every gate.
SGL_COMMIT = os.environ.get("SGL_COMMIT", "eb061d8")
CAPTURE_PATCH = os.path.join(HERE, "..", "sglang-jax-877.patch")
STEERING_PATCHES = [
    os.path.join(HERE, "..", "steering-hook.patch"),
    os.path.join(HERE, "..", "qwen3-steering-hook.patch"),
]

MODEL_PATCH = "kimi-k3-model.patch"
HOOK_PATCH = "kimi-k3-capture-hook.patch"

SRT = "python/sgl_jax/srt"
MODEL_FILE = f"{SRT}/models/kimi_k3.py"
MXFP4_FILE = f"{SRT}/utils/quantization/mxfp4.py"
ACTIVATION_FILE = f"{SRT}/layers/activation.py"
CONFIG_FILE = f"{SRT}/configs/kimi_linear.py"
HF_UTILS_FILE = f"{SRT}/hf_transformers_utils.py"
NATIVE_FILE = f"{SRT}/layers/attention/native_backend.py"
KIMI_LINEAR_MODEL_FILE = f"{SRT}/models/kimi_linear.py"
TOUCHED = [
    MODEL_FILE,
    MXFP4_FILE,
    ACTIVATION_FILE,
    f"{SRT}/layers/moe.py",
    CONFIG_FILE,
    HF_UTILS_FILE,
    f"{SRT}/utils/weight_utils.py",
    f"{SRT}/mem_cache/memory_pool.py",
    f"{SRT}/models/deepseek_v3.py",
    NATIVE_FILE,
    f"{SRT}/eplb/expert_location.py",
]
# Read for a control in check 8. The patch doesn't touch it.
UNTOUCHED = [KIMI_LINEAR_MODEL_FILE]

EPS = 1e-5

# float32 is the gate everywhere. It separates a defect from rounding by five
# orders of magnitude, and every one of these paths promotes to float32
# internally whatever the operands are.
F32_TOL = 1e-5
BF16_TOL = 5e-2
MUTANT_TOL = 1e-2
# bf16 rounding on the nine-layer stack reaches 2.1e-2, which is twice
# MUTANT_TOL, so the bf16 assertions carry a correlation floor as well. Rounding
# leaves the correlation above 0.9999; a mutant of detectable size doesn't.
BF16_MIN_CORR = 1.0 - 1e-4
# Check 10 serves in float32 with a float32 short-convolution state, so the
# engine's logits sit within rounding of the float64 forward. A weight that
# changes the model has to move them by MUTANT_TOL.
ENGINE_TOL = 1e-4

# The published checkpoint, at the Hub revision this test was checked against.
# `main` moves; a revision doesn't.
HF_MODEL = "moonshotai/Kimi-K3"
HF_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
HF_BASE = f"https://huggingface.co/{HF_MODEL}/resolve/{HF_REVISION}"
SHARD_TOTAL = 96
# Shard N + 1 holds layer N. Shard 94 holds the embedding, the final norm, the
# closing residual mixture and the head. Shards 95 and 96 hold the vision tower
# and the projector.
LAYER_SHARDS = {0: 1, 1: 2, 3: 4}
MODEL_LEVEL_SHARD = 94
VISION_SHARDS = (95, 96)
# A per-user cache keyed by the revision, unless KIMI_K3_CACHE names another.
CACHE = os.environ.get("KIMI_K3_CACHE") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
    "kimi-k3-published",
    HF_REVISION,
)
# SHA-256 of every file the test caches, read at HF_REVISION. A header is the
# JSON behind the 8-byte length at the front of a shard, and a tensor is its data
# bytes. The test execs code out of modeling_kimi_linear.py, so a cache that
# holds anything else stops the run before it reads a byte of it.
PINNED_SHA256 = {
    "config.json": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    "modeling_kimi_linear.py": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    "header-00001.json": "bfd47268eb1556d004cc0d8a8ce63935a73e812c87333cfa56d46312b1c692c8",
    "header-00002.json": "b91d17c110908e42f20e36949c0267fa43248364201a81a6db57dace3bf29776",
    "header-00004.json": "3ff389a539b7408ba7b99e9445efa1bc1e5f4bbba1e9402e2c494015add5a647",
    "header-00094.json": "f28a4f196abb0804a3f8a5b2c125f646f598359c5b92266442761216e47dd2bf",
    "header-00095.json": "0d8ce3215fcf775b96f313da0878ddea7cb4c1fbcbf3774088b9568b3a9d4ed9",
    "header-00096.json": "d48df5b82c7f4dbab9b9402c30d0391d17257e04d356079f07c0900c3214312b",
    "language_model.model.layers.0.self_attn.A_log.bin": (
        "c8ed0a50f5bde0f66fa1e0ca06a49108bd771fa410f4394b00958615f1d20bf9"
    ),
    "language_model.model.layers.1.self_attn.A_log.bin": (
        "e62123b255be20125cdcdd04bd0af0da3619845ef97cbce72993c4d1b126d072"
    ),
}

logger = logging.getLogger("test_kimi_k3_model")
# LOG_PAYLOADS=1, which the scripts/ tools read too, prints what check 10 hands
# the engine: every launch's ServerArgs and every request.
if os.environ.get("LOG_PAYLOADS", "").strip() not in ("", "0"):
    _payload_handler = logging.StreamHandler()
    _payload_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(_payload_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ----------------------------------------------------------------------------
# the published checkpoint
# ----------------------------------------------------------------------------


def _download(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "kimi-k3-patch-test"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read()


def _check_digest(name: str, data: bytes, where: str) -> None:
    """Refuse bytes that aren't the pinned revision's."""
    want = PINNED_SHA256.get(name)
    if want is None:
        raise KeyError(f"{name} has no pinned SHA-256")
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(
            f"{where} has SHA-256 {got}, not {want}, the {name} of {HF_MODEL} at "
            f"{HF_REVISION}. Delete it and rerun to fetch the pinned file."
        )


def _cached_bytes(name: str, fetch) -> bytes:
    """One published file out of `CACHE`, fetched on first use.

    A download has to match its pinned SHA-256 before it moves into place, and
    it moves only once it's whole, so a failed or wrong download leaves nothing
    for the next run to read. A cached copy is checked again on every read.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        data = fetch()
        _check_digest(name, data, f"the download of {name}")
        with open(path + ".partial", "wb") as fh:
            fh.write(data)
        os.replace(path + ".partial", path)
    with open(path, "rb") as fh:
        data = fh.read()
    _check_digest(name, data, f"the cached {path}")
    return data


def _cached(name: str, build):
    return json.loads(_cached_bytes(name, build))


def published_config() -> dict:
    """`config.json` as the loader reads it off the hub."""
    return _cached("config.json", lambda: _download(f"{HF_BASE}/config.json"))


def _shard_url(index: int) -> str:
    return f"{HF_BASE}/model-{index:05d}-of-{SHARD_TOTAL:06d}.safetensors"


def _shard_header_bytes(index: int) -> bytes:
    """One shard's header JSON. It sits behind an 8-byte length at the front."""

    def build() -> bytes:
        url = _shard_url(index)
        length = struct.unpack("<Q", _download(url, (0, 7)))[0]
        return _download(url, (8, 7 + length))

    return _cached_bytes(f"header-{index:05d}.json", build)


def shard_header(index: int) -> dict[str, dict]:
    """Every tensor in one safetensors shard, with its dtype and shape.

    Two range requests read the header without touching the tensor data.
    """
    header = json.loads(_shard_header_bytes(index))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def published_tensor(index: int, key: str) -> np.ndarray:
    """One F32 tensor's values out of a published shard, by one range request.

    The data section starts right after the header, and the header gives each
    tensor's offsets inside it.
    """
    header_length = len(_shard_header_bytes(index))
    entry = shard_header(index)[key]
    if entry["dtype"] != "F32":
        raise ValueError(f"{key} is {entry['dtype']}, and this reads F32 only")
    start, end = entry["data_offsets"]
    base = 8 + header_length
    data = _cached_bytes(
        f"{key}.bin", lambda: _download(_shard_url(index), (base + start, base + end - 1))
    )
    return np.frombuffer(data, dtype="<f4").reshape(entry["shape"])


def published_source(name: str) -> str:
    """One Python file the checkpoint repo ships, as text."""
    return _cached_bytes(name, lambda: _download(f"{HF_BASE}/{name}")).decode("utf-8")


def published_torch(names):
    """Named classes and functions out of the modeling file the repo ships.

    These are the reference the checkpoint's own authors wrote, so agreeing with
    them isn't agreeing with a second reading of the same paper.
    """
    import torch
    from torch import nn

    tree = ast.parse(published_source("modeling_kimi_linear.py"))
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in wanted}
    if missing:
        raise LookupError(f"modeling_kimi_linear.py holds no {sorted(missing)}")
    module = ast.Module(body=[copy.deepcopy(node) for node in wanted], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(module, "<published>", "exec"), namespace)  # noqa: S102
    return namespace


def published_shapes(shards) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for index in shards:
        shapes.update({key: tuple(value["shape"]) for key, value in shard_header(index).items()})
    return shapes


# ----------------------------------------------------------------------------
# checkout plumbing
# ----------------------------------------------------------------------------


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)


def get_checkout(workdir):
    """The tree the patches are diffs against, committed, ready to patch.

    `SGL_COMMIT` with `sglang-jax-877.patch` taken by `git am` and both steering
    patches applied on top, the stack `scripts/verify_patches.sh` builds. The
    steering patch adds a file, so the commit stages everything first and
    `git clean` can't take it back out.
    """
    src = os.environ.get("SGLANG_JAX_REPO")
    dest = os.path.join(workdir, "sglang-jax")
    if src and os.path.isdir(os.path.join(src, ".git")):
        args = ["git", "clone", "--quiet", "--shared", src, dest]
    else:
        args = ["git", "clone", "--quiet", UPSTREAM, dest]
    subprocess.run(args, check=True, capture_output=True)
    ident = ["-c", "user.email=kimi-k3-test@local", "-c", "user.name=kimi-k3-test"]
    steps = [
        ["checkout", "--quiet", SGL_COMMIT],
        [*ident, "am", "--quiet", CAPTURE_PATCH],
        *(["apply", patch] for patch in STEERING_PATCHES),
        ["add", "-A"],
        [*ident, "commit", "--quiet", "-m", "capture and steering"],
    ]
    for step in steps:
        done = git(dest, *step)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(step)} failed: {done.stderr.strip()}")
    return dest


def check_patches(repo, workdir):
    """Apply both patches in order, compile what they touch, refuse a corruption.

    The corrupted copy is checked against the same tree that just accepted the
    real patch, before the real patch lands. A tree that already carries the
    patch refuses the patch itself, so checking the corruption afterwards would
    report `refused` whatever `corrupt` did.
    """
    failures = 0
    for name in (MODEL_PATCH, HOOK_PATCH):
        patch = os.path.join(HERE, name)
        dry = git(repo, "apply", "--check", patch)
        ok = dry.returncode == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  applies")
        if not ok:
            print(f"      {dry.stderr.strip()}")
            return failures + 1

        with open(patch, encoding="utf-8") as fh:
            bad_text = corrupt(fh.read())
        bad_path = os.path.join(workdir, f"corrupt-{name}")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write(bad_text)
        refused = git(repo, "apply", "--check", bad_path).returncode != 0
        print(
            f"      control (one context line rewritten, same tree that took the patch): "
            f"{'refused' if refused else 'ACCEPTED'}"
        )
        if not refused:
            print("      FAIL: the control didn't fail, so --check reads nothing.")
            failures += 1

        applied = git(repo, "apply", patch)
        if applied.returncode != 0:
            print(f"      FAIL: apply failed after --check passed: {applied.stderr.strip()}")
            return failures + 1

    for rel in TOUCHED:
        compiled = subprocess.run(
            [sys.executable, "-m", "py_compile", os.path.join(repo, rel)],
            capture_output=True,
            text=True,
        )
        good = compiled.returncode == 0
        print(f"      py_compile {os.path.basename(rel)}: {'ok' if good else 'FAILED'}")
        if not good:
            print(f"      {compiled.stderr.strip()}")
            failures += 1
    return failures


def collect_sources(repo):
    """The text of every touched file, before and after both patches.

    `before` is what check 8 measures its controls against: a claim that the
    patch adds something is only worth anything if the same scan comes up empty
    on the unpatched tree.
    """
    before = {}
    for rel in TOUCHED + UNTOUCHED:
        path = os.path.join(repo, rel)
        before[rel] = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    for name in (MODEL_PATCH, HOOK_PATCH):
        applied = git(repo, "apply", os.path.join(HERE, name))
        if applied.returncode != 0:
            raise RuntimeError(f"{name} no longer applies: {applied.stderr.strip()}")
    out = {}
    for rel in TOUCHED + UNTOUCHED:
        with open(os.path.join(repo, rel), encoding="utf-8") as fh:
            out[rel] = fh.read()
    return out, before


# ----------------------------------------------------------------------------
# AST surgery: pull the patched code out and run it
# ----------------------------------------------------------------------------


def find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"class {name} not found")


def find_function(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return copy.deepcopy(node)
    raise LookupError(f"function {name} not found")


def find_method(src, cls_name, fn_name):
    for item in find_class(ast.parse(src), cls_name).body:
        if isinstance(item, ast.FunctionDef) and item.name == fn_name:
            return copy.deepcopy(item)
    raise LookupError(f"{cls_name}.{fn_name} not found")


def find_assign(src, name):
    """A module-level `NAME = ...` statement, as a one-statement module."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return copy.deepcopy(node)
    raise LookupError(f"assignment {name} not found")


def strip_annotations(fn):
    fn.returns = None
    fn.decorator_list = []
    args = fn.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        arg.annotation = None
    for arg in (args.vararg, args.kwarg):
        if arg is not None:
            arg.annotation = None
    return fn


def compile_nodes(nodes, extra=None):
    """Compile patched statements into a namespace holding only jax and numpy.

    Annotations come off the signatures, because they name types the rest of
    `sgl_jax` owns and this test doesn't import. Every statement body is the
    shipped text.
    """
    body = [
        strip_annotations(copy.deepcopy(n)) if isinstance(n, ast.FunctionDef) else n for n in nodes
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"jax": jax, "jnp": jnp, "np": np}
    namespace.update(extra or {})
    exec(compile(module, "<patched>", "exec"), namespace)  # noqa: S102
    return namespace


def run_mutant(build, run, want, tol=MUTANT_TOL):
    """Build a mutant of the patched source, run it, and say whether it moved.

    A mutation operator that can't find its target is a failure, not a catch.
    Renaming the thing it looks for would otherwise turn the whole mutation
    suite green while it tested nothing.
    """
    try:
        target = build()
    except LookupError as exc:
        return False, f"MUTATION OPERATOR FOUND NO TARGET: {exc}"
    try:
        got = run(target)
    except Exception as exc:  # noqa: BLE001 - any raise from a mutant is a catch
        return True, f"raised {type(exc).__name__}"
    rel = rel_error(got, want)
    return rel >= tol, f"rel={rel:.3e}"


def report_mutants(mutants, build, run, want, tol=MUTANT_TOL):
    failures = 0
    for label, mutate in mutants.items():
        caught, how = run_mutant(lambda m=mutate: build(m), run, want, tol)
        print(f"      control ({label}): {'caught' if caught else 'NOT DETECTED'} {how}")
        if not caught:
            failures += 1
    return failures


def rel_error(got, want):
    """Max abs error over the largest reference value.

    A shape mismatch or any value that isn't finite comes back as infinity. A
    NaN would fail `rel < tol` but pass `rel >= tol`, and `max()` drops it when
    it isn't the first argument, so no gate here ever sees one.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape or not (np.isfinite(got).all() and np.isfinite(want).all()):
        return float("inf")
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-12))


def correlation(got, want):
    got = np.asarray(got, np.float64).ravel()
    want = np.asarray(want, np.float64).ravel()
    if got.shape != want.shape:
        return float("nan")
    return float(np.corrcoef(got, want)[0, 1])


class Stub:
    """An object whose attributes are whatever the patched code reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __call__(self, *args, **kw):
        return self.__dict__["_call"](self, *args, **kw)


# ----------------------------------------------------------------------------
# 2. MXFP4
# ----------------------------------------------------------------------------


def ref_e2m1(code):
    """One E2M1 code, decoded from its bit fields.

    Sign in bit 3, exponent in bits 2 and 1, mantissa in bit 0. Exponent zero is
    the subnormal range, where the implicit leading bit is 0 and the exponent is
    the same as exponent one. Bias is 1.
    """
    sign = -1.0 if code & 0b1000 else 1.0
    exponent = (code >> 1) & 0b11
    mantissa = code & 0b1
    if exponent == 0:
        return sign * (mantissa * 0.5)
    return sign * (1.0 + 0.5 * mantissa) * 2.0 ** (exponent - 1)


def ref_dequant_mxfp4(packed, scale, group_size):
    """MXFP4 dequantization, one element at a time.

    Written from the format: low nibble first, one E8M0 exponent per group, and
    the scale is 2 to the power of the stored byte less 127. The loop is the
    point: it shares no reshape, no gather and no vectorized path with the
    implementation.
    """
    rows, byte_count = packed.shape
    out = np.zeros((rows, 2 * byte_count), dtype=np.float64)
    for r in range(rows):
        for b in range(byte_count):
            byte = int(packed[r, b])
            for half, code in enumerate((byte & 0x0F, (byte >> 4) & 0x0F)):
                column = 2 * b + half
                exponent = int(scale[r, column // group_size]) - 127
                out[r, column] = ref_e2m1(code) * (2.0**exponent)
    return out


def mxfp4_nodes(src):
    return [
        find_assign(src, "E2M1_VALUES"),
        find_assign(src, "MXFP4_EXPONENT_BIAS"),
        find_function(src, "unpack_nibbles"),
        find_function(src, "dequantize_mxfp4"),
    ]


def mxfp4_namespace(sources, mutate=None):
    nodes = mxfp4_nodes(sources[MXFP4_FILE])
    if mutate is not None:
        mutate(nodes)
    return compile_nodes(nodes)


def _node_named(nodes, name):
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise LookupError(f"{name} not among the compiled nodes")


def m_nibble_order(nodes):
    """The high nibble read first, so every pair of elements swaps."""
    fn = _node_named(nodes, "unpack_nibbles")
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node).startswith("np.stack"):
            node.args[0].elts = list(reversed(node.args[0].elts))
            return
    raise LookupError("no np.stack of the two nibbles")


def m_exponent_bias(nodes):
    """The E8M0 bias off by one, so every scale doubles."""
    node = _node_named(nodes, "MXFP4_EXPONENT_BIAS")
    if not isinstance(node.value, ast.Constant):
        raise LookupError("MXFP4_EXPONENT_BIAS isn't a literal")
    node.value = ast.Constant(value=node.value.value - 1)


def m_e2m1_subnormal(nodes):
    """The subnormal code decoded as if the leading bit were implicit."""
    node = _node_named(nodes, "E2M1_VALUES")
    literals = [n for n in ast.walk(node) if isinstance(n, ast.Constant) and n.value == 0.5]
    if not literals:
        raise LookupError("no 0.5 entry in E2M1_VALUES")
    for literal in literals:
        literal.value = 0.25


def m_scale_not_exponential(nodes):
    """The stored byte used as the scale instead of two to its power."""
    fn = _node_named(nodes, "dequantize_mxfp4")
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node).startswith("np.exp2"):
            node.func = ast.parse("np.asarray", mode="eval").body
            return
    raise LookupError("no np.exp2 in dequantize_mxfp4")


MXFP4_MUTANTS = {
    "high nibble read first": m_nibble_order,
    "E8M0 bias off by one": m_exponent_bias,
    "E2M1 subnormal given an implicit leading bit": m_e2m1_subnormal,
    "scale used linearly, not as a power of two": m_scale_not_exponential,
}


def check_mxfp4(sources):
    """The patched dequantizer against an independent bit-field decoder."""
    dequantize = mxfp4_namespace(sources)["dequantize_mxfp4"]

    rng = np.random.default_rng(11)
    rows, groups, group_size = 6, 5, 32
    columns = groups * group_size
    codes = rng.integers(0, 16, size=(rows, columns), dtype=np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    # Exponents around the bias, so the check sees scales above and below one.
    scale = rng.integers(120, 135, size=(rows, groups)).astype(np.uint8)

    got = dequantize(packed, scale, group_size=group_size)
    want = ref_dequant_mxfp4(packed, scale, group_size)
    rel = rel_error(got, want)
    print(
        f"      {rows}x{columns}, group {group_size}: "
        f"rel={rel:.3e} corr={correlation(got, want):.6f}"
    )
    failures = 0 if rel < F32_TOL else 1
    if failures:
        print("  [FAIL] dequantize_mxfp4 doesn't match the bit-field decoder")
    else:
        print("  [PASS] dequantize_mxfp4 matches the bit-field decoder")

    # A wrong count of scales has to raise rather than broadcast quietly.
    try:
        dequantize(packed, scale[:, :-1], group_size=group_size)
    except ValueError:
        print("      mismatched scale count raises ValueError")
    else:
        print("      FAIL: a short scale array was accepted")
        failures += 1

    # One nibble carries one element, so flipping it has to move one element.
    # rel_error divides by the largest value in the array and would hide that.
    one_flip = packed.copy()
    one_flip[0, 0] ^= 0x0F
    moved = int(np.count_nonzero(dequantize(one_flip, scale, group_size) != want))
    print(f"      control (one nibble flipped): {moved} element(s) moved")
    if moved == 0:
        failures += 1

    failures += report_mutants(
        MXFP4_MUTANTS,
        lambda mutate: mxfp4_namespace(sources, mutate=mutate)["dequantize_mxfp4"],
        lambda fn: fn(packed, scale, group_size=group_size),
        want,
    )
    failures += check_mxfp4_block(sources, group_size)
    return failures


def block_nodes(sources):
    return mxfp4_nodes(sources[MXFP4_FILE]) + [
        find_function(sources[MODEL_FILE], "read_mxfp4_block")
    ]


def check_mxfp4_block(sources, group_size):
    """`read_mxfp4_block` cuts the file, so the cut has to land where it says.

    The checkpoint stores `[out, in]` and `EPMoE` holds `[in, out]`. Both edges
    come off before anything decodes, and the input edge only cuts the file when
    it lands on a scale-group boundary. Both routes have to reach the same
    numbers as decoding everything and slicing afterwards. The tensors sit in a
    real safetensors file, opened the way `SequentialSafetensorManager` opens
    one, so the cuts go through the reader the loader uses.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    namespace = compile_nodes(block_nodes(sources))
    read_block = namespace["read_mxfp4_block"]

    rng = np.random.default_rng(17)
    out_dim, in_dim = 8, 256
    codes = rng.integers(0, 16, size=(out_dim, in_dim), dtype=np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    scale = rng.integers(120, 135, size=(out_dim, in_dim // group_size)).astype(np.uint8)
    scratch = tempfile.mkdtemp(prefix="kimi-k3-mxfp4-")
    path = os.path.join(scratch, "block.safetensors")
    save_file({"packed": packed, "scale": scale}, path)
    handle = safe_open(path, framework="np", device="cpu")
    try:
        return _check_mxfp4_block_cases(sources, read_block, handle, packed, scale, group_size)
    finally:
        del handle
        shutil.rmtree(scratch, ignore_errors=True)


def _check_mxfp4_block_cases(sources, read_block, handle, packed, scale, group_size):
    in_dim = 2 * packed.shape[1]
    whole = ref_dequant_mxfp4(packed, scale, group_size).T

    failures = 0
    cases = {
        "whole tensor": (slice(None), slice(None)),
        "output edge only": (slice(2, 5), slice(None)),
        "input edge on a group boundary": (slice(None), slice(32, 96)),
        "input edge off a group boundary": (slice(None), slice(5, 71)),
        "both edges": (slice(1, 4), slice(64, 160)),
    }
    for label, (out_slice, in_slice) in cases.items():
        got = read_block(
            handle,
            "packed",
            "scale",
            out_slice=out_slice,
            in_slice=in_slice,
            in_size=in_dim,
            group_size=group_size,
        )
        want = whole[in_slice, out_slice]
        rel = rel_error(got, want)
        print(f"      block ({label}): shape={tuple(got.shape)} rel={rel:.3e}")
        if rel >= F32_TOL:
            failures += 1

    # Control: the same 288 values in checkpoint order, laid into the block's
    # shape, the way a reader that skipped the transpose would hand them back.
    # Same shape and same values, so only the order can tell them apart.
    block = read_block(
        handle,
        "packed",
        "scale",
        out_slice=slice(1, 4),
        in_slice=slice(64, 160),
        in_size=in_dim,
        group_size=group_size,
    )
    untransposed = whole.T[slice(1, 4), slice(64, 160)].reshape(np.shape(block))
    rel = rel_error(block, untransposed)
    print(f"      control (block compared against the untransposed rectangle): rel={rel:.3e}")
    if rel < MUTANT_TOL:
        failures += 1

    # Every block mutant runs against both a group-aligned cut and one that
    # isn't, because the two take different routes through the function and a
    # mutant of one route is invisible to the other.
    both = (slice(32, 96), slice(5, 71))

    def read_both(fn):
        return np.concatenate(
            [
                np.asarray(
                    fn(
                        handle,
                        "packed",
                        "scale",
                        out_slice=slice(None),
                        in_slice=cut,
                        in_size=in_dim,
                        group_size=group_size,
                    ),
                    np.float64,
                ).ravel()
                for cut in both
            ]
        )

    failures += report_mutants(
        BLOCK_MUTANTS,
        lambda mutate: compile_nodes(_mutated_block_nodes(sources, mutate))["read_mxfp4_block"],
        read_both,
        np.concatenate([whole[cut, :].ravel() for cut in both]),
    )
    print(f"  [{'FAIL' if failures else 'PASS'}] read_mxfp4_block")
    return failures


def _mutated_block_nodes(sources, mutate):
    nodes = block_nodes(sources)
    mutate(_node_named(nodes, "read_mxfp4_block"))
    return nodes


def m_block_byte_offset(fn):
    """The packed cut taken in elements, not bytes, so it lands twice too far."""
    for node in ast.walk(fn):
        if isinstance(node, ast.BinOp) and ast.unparse(node) == "start // 2":
            node.right = ast.Constant(value=1)
            return
    raise LookupError("no `start // 2` in read_mxfp4_block")


def m_block_always_aligned(fn):
    """The unaligned route never taken, so an off-group cut reads the wrong bytes."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "aligned" for t in node.targets
        ):
            node.value = ast.Constant(value=True)
            return
    raise LookupError("no `aligned` assignment in read_mxfp4_block")


def m_block_no_transpose(fn):
    """The block returned in checkpoint order rather than `[in, out]`."""
    seen = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "T":
            node.attr = "base"
            seen += 1
    if seen == 0:
        raise LookupError("no transpose in read_mxfp4_block")


BLOCK_MUTANTS = {
    "packed cut taken in elements, not bytes": m_block_byte_offset,
    "every cut treated as group aligned": m_block_always_aligned,
    "block left in checkpoint order": m_block_no_transpose,
}


# ----------------------------------------------------------------------------
# 3. situ
# ----------------------------------------------------------------------------


def ref_situ(gate, up, beta, linear_beta):
    """`beta * tanh(g / beta) * sigmoid(g) * clip(up)`, in float64.

    The sigmoid is written out rather than called, so this shares no library
    path with the implementation.
    """
    gate = np.asarray(gate, np.float64)
    up = np.asarray(up, np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    activated = beta * np.tanh(gate / beta) * sigmoid
    if linear_beta is not None:
        up = linear_beta * np.tanh(up / linear_beta)
    return activated * up


def situ_function(sources, mutate=None):
    fn = find_function(sources[ACTIVATION_FILE], "situ_and_mul")
    if mutate is not None:
        mutate(fn)
    return compile_nodes([fn])["situ_and_mul"]


def _activated_assign(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "activated" for t in node.targets
        ):
            return node
    raise LookupError("no `activated` assignment in situ_and_mul")


def m_situ_no_sigmoid(fn):
    """The gate branch loses its sigmoid factor, leaving a bounded tanh."""
    node = _activated_assign(fn)
    if not isinstance(node.value, ast.BinOp):
        raise LookupError("`activated` isn't a product")
    node.value = node.value.left


def m_situ_no_beta(fn):
    """`beta` never multiplied back, so the gate saturates at 1 instead of 4."""
    node = _activated_assign(fn)
    inner = node.value.left
    if not isinstance(inner, ast.BinOp):
        raise LookupError("no `beta * tanh(...)` product in `activated`")
    node.value.left = inner.right


def m_situ_no_tanh(fn):
    """`tanh` dropped from the gate branch, so nothing bounds it."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node) == "jnp.tanh(gate_f32 / beta)":
            node.func = ast.parse("jnp.asarray", mode="eval").body
            return
    raise LookupError("no `jnp.tanh(gate_f32 / beta)` in situ_and_mul")


def m_situ_linear_unbounded(fn):
    """The linear branch left alone, so `linear_beta` does nothing."""
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "linear_beta" in ast.unparse(node.test):
            node.test = ast.Constant(value=False)
            return
    raise LookupError("no `linear_beta` test in situ_and_mul")


SITU_MUTANTS = {
    "sigmoid dropped from the gate branch": m_situ_no_sigmoid,
    "beta never multiplied back": m_situ_no_beta,
    "tanh dropped from the gate branch": m_situ_no_tanh,
    "linear branch left unbounded": m_situ_linear_unbounded,
}


def check_situ(sources):
    situ_and_mul = situ_function(sources)

    rng = np.random.default_rng(23)
    # Wide enough to cross both saturation knees: tanh(g / 4) bends near 4 and
    # tanh(u / 25) near 25.
    gate = rng.normal(size=(64, 128)) * 12.0
    up = rng.normal(size=(64, 128)) * 40.0
    config = published_config()["text_config"]
    beta = config["activation_situ_beta"]
    linear_beta = config["activation_situ_linear_beta"]

    failures = 0
    for dtype, tol, label in ((jnp.float32, F32_TOL, "f32"), (jnp.bfloat16, BF16_TOL, "bf16")):
        got = situ_and_mul(jnp.asarray(gate, dtype), jnp.asarray(up, dtype), beta, linear_beta)
        want = ref_situ(gate, up, beta, linear_beta)
        rel, corr = rel_error(got, want), correlation(got, want)
        print(
            f"      beta={beta} linear_beta={linear_beta}, {label}: "
            f"rel={rel:.3e} corr={corr:.6f}"
        )
        if rel >= tol or (dtype is jnp.bfloat16 and corr < BF16_MIN_CORR):
            failures += 1

    # The module the checkpoint repo ships, run on the same inputs.
    import torch

    situ_module = published_torch(["SituAndMul"])["SituAndMul"](beta, linear_beta)
    packed = torch.cat(
        [torch.tensor(gate, dtype=torch.float32), torch.tensor(up, dtype=torch.float32)], dim=-1
    )
    shipped = situ_module(packed).detach().numpy()
    rel = rel_error(
        situ_and_mul(jnp.asarray(gate, jnp.float32), jnp.asarray(up, jnp.float32), beta,
                     linear_beta),
        shipped,
    )
    print(f"      against SituAndMul from the checkpoint repo, f32: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1

    # linear_beta=None has to leave the up branch alone.
    got = situ_and_mul(jnp.asarray(gate, jnp.float32), jnp.asarray(up, jnp.float32), 1.0, None)
    rel = rel_error(got, ref_situ(gate, up, 1.0, None))
    print(f"      beta=1.0 linear_beta=None, f32: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1

    gate32, up32 = jnp.asarray(gate, jnp.float32), jnp.asarray(up, jnp.float32)
    want = ref_situ(gate, up, beta, linear_beta)
    controls = {
        "linear_beta dropped": rel_error(situ_and_mul(gate32, up32, beta, None), want),
        "gate and up swapped": rel_error(situ_and_mul(up32, gate32, beta, linear_beta), want),
        "beta set to 1.0": rel_error(situ_and_mul(gate32, up32, 1.0, linear_beta), want),
    }
    for label, rel in controls.items():
        caught = rel >= MUTANT_TOL
        print(f"      control ({label}): {'caught' if caught else 'NOT DETECTED'} rel={rel:.3e}")
        if not caught:
            failures += 1

    failures += report_mutants(
        SITU_MUTANTS,
        lambda mutate: situ_function(sources, mutate=mutate),
        lambda fn: fn(gate32, up32, beta, linear_beta),
        want,
    )

    print(f"  [{'FAIL' if failures else 'PASS'}] situ_and_mul")
    return failures


# ----------------------------------------------------------------------------
# 4 and 5. the block attention residual
# ----------------------------------------------------------------------------


def ref_mix(prefix_sum, stash, norm_scale, proj_weight, eps):
    """The softmax mixture, in float64, one row at a time.

    `norm_scale` and `proj_weight` stay separate here. The implementation folds
    them into one vector before the contraction, so this reaches the same number
    by a different operand order.
    """
    candidates = list(stash) + [prefix_sum]
    rows, hidden = prefix_sum.shape
    out = np.zeros((rows, hidden), np.float64)
    for r in range(rows):
        scores = np.zeros(len(candidates), np.float64)
        for c, candidate in enumerate(candidates):
            vector = np.asarray(candidate[r], np.float64)
            rms = np.sqrt(np.mean(vector**2) + eps)
            scores[c] = float(np.sum((vector / rms) * norm_scale * proj_weight))
        shifted = scores - scores.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        for c, candidate in enumerate(candidates):
            out[r] += probs[c] * np.asarray(candidate[r], np.float64)
    return out


def ref_rms(x, scale, eps):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * scale


class RefLayer:
    """float64 weights for one toy layer."""

    def __init__(self, rng, dim, is_kda):
        self.is_kda = is_kda
        self.w_attn = rng.normal(size=(dim, dim)) * 0.1
        self.w_mlp = rng.normal(size=(dim, dim)) * 0.1
        self.g_in = 1.0 + rng.normal(size=(dim,)) * 0.05
        self.g_post = 1.0 + rng.normal(size=(dim,)) * 0.05
        self.attn_norm = 1.0 + rng.normal(size=(dim,)) * 0.05
        self.attn_proj = rng.normal(size=(dim,)) * 0.3
        self.mlp_norm = 1.0 + rng.normal(size=(dim,)) * 0.05
        self.mlp_proj = rng.normal(size=(dim,)) * 0.3


def ref_stack(embed, layers, block_size, out_norm, out_proj, final_scale, eps, gate):
    """The whole block-residual algorithm, in float64.

    Written from the published `_forward_attn_residual` and the model loop: the
    prefix sum, the stash at every block boundary, the two mixtures per layer
    and the closing mixture. It builds no `[N, B, H]` tensor and runs no
    `concatenate`, so agreeing with it isn't agreeing with itself.
    """
    stream = np.asarray(embed, np.float64)
    stash = []
    captured = []
    for i, layer in enumerate(layers):
        if i in gate:
            captured.append(stream)
        read = (
            stream if not stash else ref_mix(stream, stash, layer.attn_norm, layer.attn_proj, eps)
        )
        if i % block_size == 0:
            stash = stash + [stream]
            running = None
        else:
            running = stream
        attn_out = np.tanh(ref_rms(read, layer.g_in, eps) @ layer.w_attn)
        running = attn_out if running is None else running + attn_out
        read = ref_mix(running, stash, layer.mlp_norm, layer.mlp_proj, eps)
        stream = running + np.tanh(ref_rms(read, layer.g_post, eps) @ layer.w_mlp)
    mixed = ref_mix(stream, stash, out_norm, out_proj, eps)
    return ref_rms(mixed, final_scale, eps), captured


def toy_norm(scale, dtype):
    """A stand-in RMSNorm matching `layers/layernorm.rmsnorm_forward`."""

    def call(self, x):
        x_f32 = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * jax.lax.rsqrt(mean2 + EPS)
        return (y * jnp.asarray(self.scale.value, jnp.float32)).astype(x.dtype)

    return Stub(_call=call, scale=Stub(value=jnp.asarray(scale, jnp.float32)), epsilon=EPS)


def build_layers(layer_call, ref_layers, dtype, block_size):
    """Patched decoder layers bound to toy attention and toy MLPs."""
    layers = []
    for i, ref in enumerate(ref_layers):
        w_attn = jnp.asarray(ref.w_attn, dtype)
        w_mlp = jnp.asarray(ref.w_mlp, dtype)

        def attn(self, positions, hidden, forward_batch, pool, w=w_attn, kda=ref.is_kda):
            out = jnp.tanh(hidden @ w).astype(hidden.dtype)
            state = (jnp.zeros((1,), jnp.int32), []) if kda else jnp.zeros((1,), jnp.int32)
            return out, state

        def mlp(self, hidden, w=w_mlp):
            return jnp.tanh(hidden @ w).astype(hidden.dtype)

        layers.append(
            Stub(
                _call=layer_call,
                layer_idx=i,
                rms_norm_eps=EPS,
                is_kda=ref.is_kda,
                is_moe_layer=False,
                attn_res_block_size=block_size,
                starts_block=i % block_size == 0,
                self_attn=Stub(_call=attn),
                mlp=Stub(_call=mlp),
                block_sparse_moe=None,
                input_layernorm=toy_norm(ref.g_in, dtype),
                post_attention_layernorm=toy_norm(ref.g_post, dtype),
                self_attention_res_norm=toy_norm(ref.attn_norm, dtype),
                mlp_res_norm=toy_norm(ref.mlp_norm, dtype),
                self_attention_res_proj=Stub(
                    weight=Stub(value=jnp.asarray(ref.attn_proj, jnp.float32).reshape(-1, 1))
                ),
                mlp_res_proj=Stub(
                    weight=Stub(value=jnp.asarray(ref.mlp_proj, jnp.float32).reshape(-1, 1))
                ),
            )
        )
    return layers


def build_model(model_call, layers, embed, out_norm, out_proj, final_scale, block_size, gate):
    return Stub(
        _call=model_call,
        layers=layers,
        layers_to_capture=list(gate),
        attn_res_block_size=block_size,
        embed_tokens=lambda ids: embed,
        norm=toy_norm(final_scale, embed.dtype),
        output_attn_res_norm=toy_norm(out_norm, embed.dtype),
        output_attn_res_proj=Stub(
            weight=Stub(value=jnp.asarray(out_proj, jnp.float32).reshape(-1, 1))
        ),
        config=Stub(rms_norm_eps=EPS),
    )


def patched_calls(sources, mutate=None):
    """The three pieces, each compiled under its own name."""
    src = sources[MODEL_FILE]
    mix_fn = find_function(src, "attn_res_mix")
    layer_fn = find_method(src, "KimiK3DecoderLayer", "__call__")
    model_fn = find_method(src, "KimiK3Model", "__call__")
    if mutate is not None:
        mutate(mix_fn, layer_fn, model_fn)

    mix = compile_nodes([mix_fn])["attn_res_mix"]
    layer_fn.name = "layer_call"
    model_fn.name = "model_call"
    ns = compile_nodes([layer_fn, model_fn], extra={"attn_res_mix": mix})
    return mix, ns["layer_call"], ns["model_call"]


def check_attn_res_mix(sources):
    """`attn_res_mix` on its own, against the unfolded float64 mixture."""
    mix, _, _ = patched_calls(sources)
    rng = np.random.default_rng(5)
    tokens, hidden, blocks = 48, 96, 4
    prefix = rng.normal(size=(tokens, hidden))
    stash = [rng.normal(size=(tokens, hidden)) for _ in range(blocks)]
    norm_scale = 1.0 + rng.normal(size=(hidden,)) * 0.1
    proj_weight = rng.normal(size=(hidden,)) * 0.4

    want = ref_mix(prefix, stash, norm_scale, proj_weight, EPS)
    failures = 0
    for dtype, tol, label in ((jnp.float32, F32_TOL, "f32"), (jnp.bfloat16, BF16_TOL, "bf16")):
        got = mix(
            jnp.asarray(prefix, dtype),
            jnp.asarray(np.stack(stash, axis=1), dtype),
            jnp.asarray(norm_scale, jnp.float32),
            jnp.asarray(proj_weight, jnp.float32).reshape(-1, 1),
            EPS,
        )
        rel, corr = rel_error(got, want), correlation(got, want)
        print(f"      {blocks} stashed + 1, {label}: rel={rel:.3e} corr={corr:.6f}")
        if rel >= tol or (dtype is jnp.bfloat16 and corr < BF16_MIN_CORR):
            failures += 1

    # The function the checkpoint repo ships, run on the same inputs.
    import torch

    apply_attn_res = published_torch(["_apply_attn_res"])["_apply_attn_res"]
    shipped = apply_attn_res(
        torch.tensor(prefix, dtype=torch.float32),
        torch.tensor(np.stack(stash, axis=1), dtype=torch.float32),
        Stub(weight=torch.tensor(proj_weight, dtype=torch.float32).reshape(1, -1)),
        Stub(weight=torch.tensor(norm_scale, dtype=torch.float32), variance_epsilon=EPS),
    )
    got = mix(
        jnp.asarray(prefix, jnp.float32),
        jnp.asarray(np.stack(stash, axis=1), jnp.float32),
        jnp.asarray(norm_scale, jnp.float32),
        jnp.asarray(proj_weight, jnp.float32).reshape(-1, 1),
        EPS,
    )
    rel = rel_error(got, shipped.detach().numpy())
    print(f"      against _apply_attn_res from the checkpoint repo, f32: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1

    # An empty stash has to be the identity: softmax over one score is 1.0.
    got = mix(
        jnp.asarray(prefix, jnp.float32),
        jnp.zeros((tokens, 0, hidden), jnp.float32),
        jnp.asarray(norm_scale, jnp.float32),
        jnp.asarray(proj_weight, jnp.float32).reshape(-1, 1),
        EPS,
    )
    rel = rel_error(got, prefix)
    print(f"      empty stash is the identity: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1

    prefix32 = jnp.asarray(prefix, jnp.float32)
    stack32 = jnp.asarray(np.stack(stash, axis=1), jnp.float32)
    norm32 = jnp.asarray(norm_scale, jnp.float32)
    proj32 = jnp.asarray(proj_weight, jnp.float32).reshape(-1, 1)
    controls = {
        "proj weight dropped from the score": rel_error(
            mix(prefix32, stack32, norm32, jnp.ones_like(proj32), EPS), want
        ),
        "norm weight dropped from the score": rel_error(
            mix(prefix32, stack32, jnp.ones_like(norm32), proj32, EPS), want
        ),
        "prefix sum left out of the stack": rel_error(
            mix(stack32[:, -1], stack32[:, :-1], norm32, proj32, EPS), want
        ),
        "one stash entry doubled": rel_error(
            mix(prefix32, stack32.at[:, 0].multiply(2.0), norm32, proj32, EPS), want
        ),
    }
    for label, rel in controls.items():
        caught = rel >= MUTANT_TOL
        print(f"      control ({label}): {'caught' if caught else 'NOT DETECTED'} rel={rel:.3e}")
        if not caught:
            failures += 1

    # Mutants of the function body itself. A permuted stash isn't one of them.
    # The mixture is a softmax over an unordered set, so a reordered stash gives
    # the same answer, and a control built on it would always pass.
    failures += report_mutants(
        MIX_MUTANTS,
        lambda mutate: patched_calls(sources, mutate=mutate)[0],
        lambda fn: fn(prefix32, stack32, norm32, proj32, EPS),
        want,
    )

    print(f"  [{'FAIL' if failures else 'PASS'}] attn_res_mix")
    return failures


# ----------------------------------------------------------------------------
# mutants of the patched stack
# ----------------------------------------------------------------------------


def find_if(fn, marker):
    """The first `if` in `fn` whose test text contains `marker`."""
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and marker in ast.unparse(node.test):
            return node
    raise LookupError(f"no `if` testing {marker!r}")


def m_mix_returns_normalized(mix_fn, layer_fn, model_fn):
    """The mixture averages the normalized vectors rather than the raw ones."""
    for node in ast.walk(mix_fn):
        if isinstance(node, ast.Call) and ast.unparse(node).startswith("jnp.einsum"):
            node.args[-1] = ast.Name(id="normed", ctx=ast.Load())
            return
    raise LookupError("no einsum in attn_res_mix")


def m_mix_skips_normalization(mix_fn, layer_fn, model_fn):
    """Scores read the raw vectors, so a long vector wins on length alone."""
    for node in ast.walk(mix_fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "normed" for t in node.targets
        ):
            node.value = ast.Name(id="stack_f32", ctx=ast.Load())
            return
    raise LookupError("no `normed` assignment in attn_res_mix")


MIX_MUTANTS = {
    "mixture averages the normalized vectors": m_mix_returns_normalized,
    "RMS normalization dropped before scoring": m_mix_skips_normalization,
}


def m_never_restart(mix_fn, layer_fn, model_fn):
    """`restart` never set, so the prefix sum never resets at a block boundary."""
    block = find_if(layer_fn, "self.starts_block")
    body = [
        s
        for s in block.body
        if not (
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "restart" for t in s.targets)
        )
    ]
    if len(body) == len(block.body):
        raise LookupError("no `restart` assignment under the block test")
    block.body = body


def m_always_restart(mix_fn, layer_fn, model_fn):
    """Every layer starts a block."""
    find_if(layer_fn, "self.starts_block").test = ast.Constant(value=True)


def m_stash_before_attention_mix(mix_fn, layer_fn, model_fn):
    """The attention mixture moved below the snapshot.

    A layer that starts a block then mixes in a copy of the prefix sum it's
    already mixing, so its own entry counts twice.
    """
    body = layer_fn.body
    mix = find_if(layer_fn, "block_residual.shape[1]")
    stash = find_if(layer_fn, "self.starts_block")
    body.pop(body.index(mix))
    body.insert(body.index(stash) + 1, mix)


def m_mlp_mix_uses_attention_weights(mix_fn, layer_fn, model_fn):
    """`mlp_res_*` swapped for `self_attention_res_*` in the second mixture."""
    seen = 0
    for node in ast.walk(layer_fn):
        if isinstance(node, ast.Attribute) and node.attr in ("mlp_res_norm", "mlp_res_proj"):
            node.attr = (
                "self_attention_res_norm" if "norm" in node.attr else "self_attention_res_proj"
            )
            seen += 1
    if seen == 0:
        raise LookupError("mlp_res weights not found")


def m_attention_mix_skipped(mix_fn, layer_fn, model_fn):
    """The first mixture never runs, so a layer reads the bare prefix sum."""
    find_if(layer_fn, "block_residual.shape[1]").test = ast.Constant(value=False)


def m_output_mix_skipped(mix_fn, layer_fn, model_fn):
    """The closing mixture never runs."""
    find_if(model_fn, "self.attn_res_block_size").test = ast.Constant(value=False)


def m_stash_holds_the_mixture(mix_fn, layer_fn, model_fn):
    """The stash takes the mixed value rather than the prefix sum."""
    block = find_if(layer_fn, "self.starts_block")
    for node in ast.walk(block):
        if isinstance(node, ast.Subscript) and ast.unparse(node).startswith("prefix_sum["):
            node.value = ast.Name(id="hidden_states", ctx=ast.Load())
            return
    raise LookupError("no prefix_sum slice in the stash")


def m_ffn_not_added(mix_fn, layer_fn, model_fn):
    """The FFN output dropped from the prefix sum."""
    for node in ast.walk(layer_fn):
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node).strip() == "prefix_sum = prefix_sum + ffn_out"
        ):
            node.value = ast.Name(id="prefix_sum", ctx=ast.Load())
            return
    raise LookupError("prefix sum FFN add not found")


def _capture_append(model_fn):
    for node in ast.walk(model_fn):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value).startswith("aux_hidden_states.append")
        ):
            return node
    raise LookupError("no aux_hidden_states.append in the layer loop")


def m_capture_after_the_layer(mix_fn, layer_fn, model_fn):
    """The capture moved below the layer call, so it returns the layer output."""
    _capture_append(model_fn)
    gate = None
    for node in ast.walk(model_fn):
        if isinstance(node, ast.If) and "layers_to_capture" in ast.unparse(node.test):
            gate = node
    if gate is None:
        raise LookupError("no layers_to_capture test in the layer loop")
    for node in ast.walk(model_fn):
        if isinstance(node, ast.For) and isinstance(node.body, list) and gate in node.body:
            node.body.remove(gate)
            node.body.insert(1, gate)
            return
    raise LookupError("the capture gate isn't a statement of the layer loop")


def m_capture_ignores_the_gate(mix_fn, layer_fn, model_fn):
    """Every layer captured whatever `layers_to_capture` holds."""
    for node in ast.walk(model_fn):
        if isinstance(node, ast.If) and "layers_to_capture" in ast.unparse(node.test):
            node.test = ast.Constant(value=True)
            return
    raise LookupError("no layers_to_capture test in the layer loop")


STACK_MUTANTS = {
    "prefix sum never restarts at a block boundary": m_never_restart,
    "every layer restarts the prefix sum": m_always_restart,
    "attention mixture moved below the snapshot": m_stash_before_attention_mix,
    "mlp mixture uses the attention weights": m_mlp_mix_uses_attention_weights,
    "attention mixture skipped": m_attention_mix_skipped,
    "closing mixture skipped": m_output_mix_skipped,
    "stash holds the mixture, not the prefix sum": m_stash_holds_the_mixture,
    "FFN output dropped from the prefix sum": m_ffn_not_added,
    "capture moved below the layer call": m_capture_after_the_layer,
    "capture ignores layers_to_capture": m_capture_ignores_the_gate,
}


def run_stack(calls, mesh, ref_layers, embed64, out_norm, out_proj, final_scale, dtype, gate,
              block_size):
    """Run the patched layer and model the way `ModelRunner._forward_raw` does.

    The embedding arrives sharded `P("data", None)`, which is what `Embed` hands
    back, and the jitted call runs under `jax.set_mesh` on a mesh with Explicit
    axes, which is what the scheduler builds. An Auto mesh without `set_mesh`
    lets a replicated operand meet a data-sharded one in a concatenate, and the
    server doesn't.
    """
    _, layer_call, model_call = calls
    embed = jax.device_put(jnp.asarray(embed64, dtype), NamedSharding(mesh, P("data", None)))
    layers = build_layers(layer_call, ref_layers, dtype, block_size)
    model = build_model(
        model_call, layers, embed, out_norm, out_proj, final_scale, block_size, gate
    )
    forward_batch = Stub(
        input_ids=jnp.zeros((embed.shape[0],), jnp.int32),
        positions=jnp.arange(embed.shape[0], dtype=jnp.int32),
        input_embedding=None,
        expert_location_metadata=None,
    )
    pools = Stub(recurrent_state_pool="recurrent", token_to_kv_pool="kv")
    with jax.set_mesh(mesh):
        return jax.jit(lambda: model_call(model, forward_batch, pools))()


def engine_mesh(data, tensor):
    """A `(data, tensor)` mesh with Explicit axes, as `create_device_mesh` builds it."""
    return jax.make_mesh(
        (data, tensor),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices()[: data * tensor],
    )


def flatten_result(result):
    """The final stream and every captured layer, as one array.

    One number then covers both halves, so a mutant confined to the capture
    hook can't report `no change`.
    """
    parts = [np.asarray(result[0], np.float64)]
    parts.extend(np.asarray(item, np.float64) for item in result[1])
    return np.concatenate(parts, axis=0)


def check_stack(sources, mesh):
    """The whole stack: prefix sum, stash, two mixtures per layer, closing mixture.

    `mesh` splits data two ways. The float32 run repeats on a one-way data axis,
    the shape `--dp-size 1` gives, because a replicated stash meets the
    embedding there too.
    """
    rng = np.random.default_rng(31)
    tokens, dim, num_layers, block_size = 64, 48, 9, 3
    ref_layers = [RefLayer(rng, dim, is_kda=(i % 4 != 3)) for i in range(num_layers)]
    embed64 = rng.normal(size=(tokens, dim))
    out_norm = 1.0 + rng.normal(size=(dim,)) * 0.05
    out_proj = rng.normal(size=(dim,)) * 0.3
    final_scale = 1.0 + rng.normal(size=(dim,)) * 0.05
    gate = list(range(num_layers))

    want_out, want_captured = ref_stack(
        embed64, ref_layers, block_size, out_norm, out_proj, final_scale, EPS, gate
    )
    want_all = np.concatenate([want_out, *want_captured], axis=0)

    def run(calls, dtype=jnp.float32, layers_to_capture=gate, on=mesh):
        return run_stack(
            calls, on, ref_layers, embed64, out_norm, out_proj, final_scale, dtype,
            layers_to_capture, block_size,
        )

    failures = 0
    runs = (
        (jnp.float32, F32_TOL, "f32", mesh),
        (jnp.bfloat16, BF16_TOL, "bf16", mesh),
        (jnp.float32, F32_TOL, "f32", engine_mesh(1, 8)),
    )
    for dtype, tol, label, on in runs:
        shape = f"data={on.shape['data']} tensor={on.shape['tensor']}"
        try:
            result = run(patched_calls(sources), dtype=dtype, on=on)
        except Exception as exc:  # noqa: BLE001 - the stack has to run on the server's mesh
            print(f"      {num_layers} layers on {shape}, {label}: RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:200]}")
            failures += 1
            continue
        rel = rel_error(result[0], want_out)
        corr = correlation(result[0], want_out)
        print(
            f"      {num_layers} layers, block {block_size}, {shape}, {label}: "
            f"rel={rel:.3e} corr={corr:.6f}"
        )
        if rel >= tol or (dtype is jnp.bfloat16 and corr < BF16_MIN_CORR):
            failures += 1
    if failures:
        # Every mutant would raise the same way and read as caught, so they
        # don't run on a stack that can't.
        print("  [FAIL] the block-residual stack doesn't run on the server's mesh")
        return failures + check_capture_entry(sources)

    result = run(patched_calls(sources))
    captured = result[1]
    if len(captured) != num_layers:
        print(f"      FAIL: captured {len(captured)} of {num_layers} layers")
        failures += 1
    else:
        print("      per layer, f32:")
        for i, (got, want) in enumerate(zip(captured, want_captured)):
            rel = rel_error(got, want)
            print(
                f"        layer {i:2d} {'KDA' if ref_layers[i].is_kda else 'MLA'}"
                f"{' block start' if i % block_size == 0 else '           '}"
                f"  max abs err={np.abs(np.asarray(got, np.float64) - want).max():.3e}"
                f"  rel={rel:.3e}  corr={correlation(got, want):.6f}"
            )
            if rel >= F32_TOL:
                failures += 1

    # The gate decides what lands, and an empty gate captures nothing.
    empty = run(patched_calls(sources), layers_to_capture=[])
    print(f"      empty layers_to_capture: {len(empty[1])} captured")
    if len(empty[1]) != 0:
        failures += 1

    # A gate of two layers captures those two and nothing else.
    picked = [1, 5]
    some = run(patched_calls(sources), layers_to_capture=picked)
    matched = len(some[1]) == len(picked) and all(
        rel_error(some[1][j], want_captured[i]) < F32_TOL for j, i in enumerate(picked)
    )
    print(f"      layers_to_capture={picked}: {len(some[1])} captured, "
          f"{'right layers' if matched else 'WRONG LAYERS'}")
    if not matched:
        failures += 1

    # The two pools have to come back on the right side of the split.
    kda_count = sum(1 for layer in ref_layers if layer.is_kda)
    got_kv = len(result[2])
    got_recurrent = len(result[3][0])
    print(
        f"      pools: {got_recurrent} recurrent, {got_kv} kv "
        f"(expected {kda_count} and {num_layers - kda_count})"
    )
    if got_recurrent != kda_count or got_kv != num_layers - kda_count:
        failures += 1

    print(f"  [{'FAIL' if failures else 'PASS'}] the block-residual stack runs as patched")

    # Every stack mutant runs twice, once capturing every layer and once
    # capturing two. A mutant that ignores `layers_to_capture` changes nothing
    # when the gate already holds every layer.
    want_picked = np.concatenate([want_out, *(want_captured[i] for i in picked)], axis=0)

    def run_both(calls):
        return np.concatenate(
            [
                flatten_result(run(calls)).ravel(),
                flatten_result(run(calls, layers_to_capture=picked)).ravel(),
            ]
        )

    failures += report_mutants(
        STACK_MUTANTS,
        lambda mutate: patched_calls(sources, mutate=mutate),
        run_both,
        np.concatenate([want_all.ravel(), want_picked.ravel()]),
    )
    failures += check_capture_entry(sources)
    return failures


# ----------------------------------------------------------------------------
# the entry-class half of the capture hook
# ----------------------------------------------------------------------------


def entry_call(sources, mutate=None):
    fn = find_method(sources[MODEL_FILE], "KimiK3ForCausalLM", "__call__")
    if mutate is not None:
        mutate(fn)
    fn.name = "entry_call"
    return compile_nodes([fn])["entry_call"]


UNSET = object()


def run_entry(call, capture: bool):
    """Drive `KimiK3ForCausalLM.__call__` with a recording logits processor."""
    aux = [jnp.full((2, 3), float(i)) for i in range(4)]
    seen = {}

    def processor(self, hidden, head, metadata, aux_hidden_states=UNSET):
        seen["aux"] = aux_hidden_states
        seen["head"] = head
        return "logits"

    model = Stub(
        _call=lambda self, forward_batch, pools: (
            jnp.zeros((2, 3)),
            list(aux),
            ["kv"],
            (["recurrent"], ["conv"]),
            ["topk"],
        ),
        embed_tokens="EMBED",
    )
    entry = Stub(
        model=model,
        config=Stub(tie_word_embeddings=False),
        lm_head="LM_HEAD",
        logits_processor=Stub(_call=processor),
        capture_aux_hidden_states=capture,
    )
    output = call(entry, Stub(), Stub(), Stub())
    return seen, output, aux


def m_entry_no_gate(fn):
    """The flag never consulted, so a plain request carries the capture."""
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "capture_aux_hidden_states" in ast.unparse(node.test):
            node.test = ast.Constant(value=False)
            return
    raise LookupError("no capture_aux_hidden_states test in the entry call")


def m_entry_drops_the_kwarg(fn):
    """`aux_hidden_states` never reaches the logits processor."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("logits_processor"):
            if not node.keywords:
                raise LookupError("the logits processor call carries no keyword")
            node.keywords = []
            return
    raise LookupError("no logits processor call in the entry call")


ENTRY_MUTANTS = {
    "the capture flag never consulted": m_entry_no_gate,
    "aux_hidden_states dropped from the logits processor call": m_entry_drops_the_kwarg,
}


def check_capture_entry(sources):
    """The half of the hook that lives on the entry class.

    The flag gates the capture and the list has to reach `LogitsProcessor` by
    keyword. Without the keyword the scheduler reshapes one layer of hidden
    states to 93 and the request fails.
    """
    failures = 0
    call = entry_call(sources)

    seen, output, aux = run_entry(call, capture=True)
    forwarded = isinstance(seen["aux"], list) and len(seen["aux"]) == len(aux)
    print(f"      capture on: logits processor receives {len(aux)} layers: "
          f"{'yes' if forwarded else 'NO'}")
    if not forwarded or seen["head"] != "LM_HEAD" or output[0] != "logits":
        failures += 1

    seen_off, _, _ = run_entry(call, capture=False)
    gated = seen_off["aux"] is None
    print(f"      capture off: logits processor receives {seen_off['aux']!r}: "
          f"{'gated' if gated else 'NOT GATED'}")
    if not gated:
        failures += 1

    for label, mutate in ENTRY_MUTANTS.items():
        try:
            mutant = entry_call(sources, mutate=mutate)
        except LookupError as exc:
            print(f"      control ({label}): MUTATION OPERATOR FOUND NO TARGET: {exc}")
            failures += 1
            continue
        try:
            observed, _, _ = run_entry(mutant, capture=False)
            caught = observed["aux"] is not None
            how = f"off-state forwards {type(observed['aux']).__name__}"
        except Exception as exc:  # noqa: BLE001 - any raise is a catch
            caught, how = True, f"raised {type(exc).__name__}"
        if not caught:
            observed_on, _, _ = run_entry(mutant, capture=True)
            caught = observed_on["aux"] is UNSET
            how = "on-state never passes the keyword" if caught else how
        print(f"      control ({label}): {'caught' if caught else 'NOT DETECTED'} {how}")
        if not caught:
            failures += 1

    print(f"  [{'FAIL' if failures else 'PASS'}] the capture hook on the entry class")
    return failures


# ----------------------------------------------------------------------------
# 6. the KDA decay layout
# ----------------------------------------------------------------------------


def ref_kda_recurrence(q, k, v, g, beta, decay, dt_bias, lengths, lower_bound, scale):
    """KDA over packed requests in float64, one token at a time.

    q and k are L2-normalized per head, the gate is `lower_bound * sigmoid(
    exp(A_log) * (g + dt_bias))`, and each request starts from a zero state:
    decay the state by the gate, write the delta-rule correction, read it back
    with the scaled query. `decay` is `[H, K]`. Kimi K3 reads one log decay per
    head, so each row of the table repeats one value.
    """
    heads, dim = decay.shape
    out = np.zeros(q.shape, np.float64)
    start = 0
    for length in lengths:
        state = np.zeros((heads, dim, v.shape[-1]))
        for t in range(start, start + length):
            q_t = q[t] / np.sqrt(np.sum(q[t] ** 2, axis=-1, keepdims=True) + 1e-6)
            k_t = k[t] / np.sqrt(np.sum(k[t] ** 2, axis=-1, keepdims=True) + 1e-6)
            gate = lower_bound / (1.0 + np.exp(-np.exp(decay) * (g[t] + dt_bias)))
            state = state * np.exp(gate)[:, :, None]
            delta = beta[t][:, None] * (v[t] - np.einsum("hk,hkv->hv", k_t, state))
            state = state + k_t[:, :, None] * delta[:, None, :]
            out[t] = np.einsum("hk,hkv->hv", q_t * scale, state)
        start += length
    return out


def per_head_table(values, heads, dim):
    """`[heads, dim]` log decays, each head's value across its channels."""
    return np.broadcast_to(np.asarray(values)[:heads, None], (heads, dim))


def per_channel_table(values, heads, dim):
    """The same vector read one value per channel, the reading this patch dropped."""
    return np.broadcast_to(np.asarray(values)[None, :dim], (heads, dim))


def check_mega_kda(repo, lower_bound):
    """The Mega KDA prefill kernel on one log decay per head.

    It's serving's default prefill kernel, and it takes the decay per head the
    way the backend hands over the layer's `[1, 1, H, 1]`. It takes BF16 only,
    and check 10 serves in float32, so this runs it here in Pallas interpret mode
    on two packed requests that share a 64-token tile, against the float64
    recurrence.
    """
    import ml_dtypes

    import_patched(repo)
    from sgl_jax.srt.kernels.kda.mega_kda import kda_forward_packed

    rng = np.random.default_rng(3)
    heads, dim, lengths = 4, 16, (40, 30)
    tokens = sum(lengths)

    def bf16(shape):
        return rng.normal(size=shape).astype(np.float32).astype(ml_dtypes.bfloat16)

    q, k, v, g = (bf16((tokens, heads, dim)) for _ in range(4))
    beta = (1.0 / (1.0 + np.exp(-rng.normal(size=(tokens, heads))))).astype(np.float32)
    decay = np.log(rng.uniform(1.0, 16.0, size=(heads,))).astype(np.float32)
    dt_bias = (rng.normal(size=(heads, dim)) * 0.5).astype(np.float32)
    scale = dim**-0.5

    def run(values):
        out, _ = kda_forward_packed(
            *(jnp.asarray(x)[None] for x in (q, k, v, g, beta)),
            cu_seqlens=jnp.asarray([0, lengths[0], tokens], jnp.int32),
            A_log=jnp.asarray(values),
            dt_bias=jnp.asarray(dt_bias),
            scale=scale,
            initial_state=jnp.zeros((len(lengths), heads, dim, dim), jnp.float32),
            lower_bound=lower_bound,
        )
        return np.asarray(out[0]).astype(np.float64)

    as64 = [np.asarray(x).astype(np.float64) for x in (q, k, v, g, beta)]
    want = ref_kda_recurrence(*as64, per_head_table(decay, heads, dim).astype(np.float64),
                              dt_bias.astype(np.float64), lengths, lower_bound, scale)
    failures = 0
    got = run(decay)
    rel, corr = rel_error(got, want), correlation(got, want)
    print(f"      Mega KDA on one log decay per head, bf16, two requests in one tile: "
          f"rel={rel:.3e} corr={corr:.6f}")
    if rel >= BF16_TOL or corr < BF16_MIN_CORR:
        failures += 1
    # Control: each head handed its neighbor's decay.
    rel = rel_error(run(np.roll(decay, 1)), want)
    print(f"      control (Mega KDA handed each head its neighbor's decay): rel={rel:.3e}")
    if rel < MUTANT_TOL:
        failures += 1
    return failures


def reference_a_log_declaration() -> str:
    """How the published `KimiDeltaAttention.__init__` builds `self.A_log`, as text."""
    tree = ast.parse(published_source("modeling_kimi_linear.py"))
    for node in ast.walk(find_class(tree, "KimiDeltaAttention")):
        if isinstance(node, ast.Assign) and any(
            ast.unparse(target) == "self.A_log" for target in node.targets
        ):
            return ast.unparse(node.value)
    raise LookupError("modeling_kimi_linear.py builds no self.A_log in KimiDeltaAttention")


def check_decay(repo):
    """What the published checkpoint ships for `A_log`, and how it's read.

    The header gives the shape and the tensor bytes give the layout: one log
    decay per head, then zeros up to `head_dim`. The reference modeling file
    sizes the parameter by the head count, FLA's kernels read it at
    `A_log + i_h`, and vLLM and SGLang keep its first `num_heads` entries. Check
    9 holds the mapping to that shape, check 10 serves it against a float64
    forward that reads it per head, and check 11 loads it.
    """
    failures = 0
    linear = published_config()["text_config"]["linear_attn_config"]
    num_heads, head_dim = linear["num_heads"], linear["head_dim"]
    lower_bound = linear["gate_lower_bound"]

    first = None
    for layer in (0, 1):
        index = LAYER_SHARDS[layer]
        key = f"language_model.model.layers.{layer}.self_attn.A_log"
        entry = shard_header(index)[key]
        values = published_tensor(index, key)
        live = int(np.count_nonzero(values[:num_heads]))
        padding = values[num_heads:]
        print(f"      layer {layer} A_log: {entry['dtype']} {entry['shape']}, "
              f"{live} nonzero of the first {num_heads}, then {padding.size} entries "
              f"of which {int(np.count_nonzero(padding))} are nonzero")
        if (tuple(entry["shape"]) != (head_dim,) or head_dim <= num_heads
                or live != num_heads or np.count_nonzero(padding)):
            print(f"      FAIL: A_log isn't {num_heads} per-head values padded with zeros "
                  f"to {head_dim}")
            failures += 1
        first = values if first is None else first

    declared = reference_a_log_declaration()
    per_head = "self.num_heads" in declared and "head_dim" not in declared
    print(f"      the reference KimiDeltaAttention builds self.A_log = {declared}: "
          f"{'one per head' if per_head else 'NOT ONE PER HEAD'}")
    if not per_head:
        failures += 1

    # Control: the published vector read per channel. The two readings have to
    # disagree, or no forward could tell the patch's reading from the wrong one.
    wrong = per_channel_table(first, num_heads, head_dim) != per_head_table(
        first, num_heads, head_dim
    )
    moved = int(np.count_nonzero(wrong))
    print(f"      control (layer 0's A_log read per channel): {moved:,} of "
          f"{num_heads * head_dim:,} (head, channel) decays differ from the per-head reading")
    if moved == 0:
        failures += 1

    failures += check_mega_kda(repo, lower_bound)
    print(f"  [{'FAIL' if failures else 'PASS'}] the KDA decay layout")
    return failures


# ----------------------------------------------------------------------------
# 7 and 8. the published config
# ----------------------------------------------------------------------------


def config_classes(sources, mutate=None):
    """The two config classes and the helpers the state pools read them through.

    `KimiLinearConfig`, `KimiK3Config`, `text_tower`, `_is_kimi_linear_config`
    and `get_kimi_linear_config`, compiled from the patched file.
    """
    from transformers.configuration_utils import PretrainedConfig

    tree = ast.parse(sources[CONFIG_FILE])
    nodes = [ast.parse("from __future__ import annotations").body[0]]
    for name in ("KimiLinearConfig", "KimiK3Config"):
        nodes.append(copy.deepcopy(find_class(tree, name)))
    for name in ("text_tower", "_is_kimi_linear_config", "get_kimi_linear_config"):
        nodes.append(find_function(sources[CONFIG_FILE], name))
    if mutate is not None:
        mutate(nodes)
    return compile_nodes(nodes, extra={"PretrainedConfig": PretrainedConfig, "Any": object})


def built_config(sources, mutate=None):
    namespace = config_classes(sources, mutate=mutate)
    return namespace, namespace["KimiK3Config"](**copy.deepcopy(published_config()))


def _class_node(nodes, name):
    for node in nodes:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"class {name} not among the compiled nodes")


def m_kda_layers_zero_based(nodes):
    """`kda_layers` read 0-based, which keeps the count and moves the layers."""
    for node in ast.walk(_class_node(nodes, "KimiLinearConfig")):
        if isinstance(node, ast.BinOp) and ast.unparse(node) == "layer_idx + 1":
            parent_fixed = ast.Name(id="layer_idx", ctx=ast.Load())
            node.left, node.op, node.right = parent_fixed, ast.Add(), ast.Constant(value=0)
            return
    raise LookupError("no `layer_idx + 1` in KimiLinearConfig")


def check_config(sources):
    """The layer split and what each half costs, read through the patched class."""
    failures = 0
    published = published_config()["text_config"]
    linear = published["linear_attn_config"]
    layers = published["num_hidden_layers"]
    block_size = published["attn_res_block_size"]
    heads, head_dim = linear["num_heads"], linear["head_dim"]
    conv_kernel = linear["short_conv_kernel_size"]

    _, config = built_config(sources)
    kda = [i for i in range(layers) if config.is_kda_layer(i)]
    mla = [i for i in range(layers) if not config.is_kda_layer(i)]
    print(f"  {len(kda)} KDA + {len(mla)} MLA over {layers} layers")
    if (len(kda), len(mla)) != (69, 24):
        print("  FAIL: the split doesn't match the published 69 and 24")
        failures += 1

    # The published lists are 1-based and cover the stack between them.
    want_kda = sorted(i - 1 for i in linear["kda_layers"])
    want_mla = sorted(i - 1 for i in linear["full_attn_layers"])
    if kda != want_kda or mla != want_mla:
        print("  FAIL: the split doesn't match the published kda_layers")
        failures += 1
    else:
        print("  every layer lands where the published kda_layers and full_attn_layers put it")
    if 0 not in kda or layers - 1 in kda:
        print("  FAIL: layer 0 must be KDA and the last layer must not be")
        failures += 1

    # Control: the same lists read 0-based keep the count and move the layers.
    try:
        _, mutant = built_config(sources, mutate=m_kda_layers_zero_based)
    except LookupError as exc:
        print(f"  control (0-based reading of kda_layers): MUTATION OPERATOR FOUND NO TARGET: {exc}")
        failures += 1
    else:
        moved = sorted(set(kda) ^ {i for i in range(layers) if mutant.is_kda_layer(i)})
        print(f"  control (0-based reading of kda_layers): {len(moved)} layers move")
        if not moved:
            print("  FAIL: the control didn't fail, so the indexing base is untested.")
            failures += 1

    boundaries = [i for i in range(layers) if i % block_size == 0]
    print(f"  block boundaries at {boundaries}, so the closing mixture spans "
          f"{len(boundaries) + 1} vectors")
    if len(boundaries) != 8:
        failures += 1

    state = len(kda) * heads * head_dim * head_dim * 4
    conv = len(kda) * (conv_kernel - 1) * 3 * heads * head_dim * 2
    kv_per_token = len(mla) * (published["kv_lora_rank"] + published["qk_rope_head_dim"]) * 2
    print(f"  recurrent state per request: {state / 2**20:,.0f} MiB float32")
    print(f"  conv state per request: {conv / 2**20:,.1f} MiB bfloat16")
    print(f"  MLA KV per token: {kv_per_token / 1024:,.0f} KiB bfloat16")
    print(f"  one request of recurrent state buys {state // kv_per_token:,} tokens of KV cache")
    if state // kv_per_token < 1000:
        print("  FAIL: the two pools are closer in size than the arithmetic says")
        failures += 1
    return failures


def entry_class_names(src):
    """Every architecture name `EntryClass` registers, read off the AST.

    The registry keys on the class name, so this is the set of strings a
    checkpoint's `architectures` can match.
    """
    tree = ast.parse(src)
    try:
        node = find_assign(src, "EntryClass").value
    except LookupError:
        return set()
    names = [node] if isinstance(node, ast.Name) else list(getattr(node, "elts", []))
    out = set()
    for item in names:
        if not isinstance(item, ast.Name):
            continue
        out.add(item.id)
        # A subclass registers its own name and answers to its base's behavior.
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef) and cls.name == item.id:
                out.add(cls.name)
    return out


def registered_config_types(src):
    """The config classes `_CONFIG_REGISTRY` builds itself from.

    The statement carries a type annotation, so this reads `AnnAssign` as well
    as `Assign`.
    """
    for node in ast.parse(src).body:
        target = getattr(node, "target", None)
        targets = [target] if target is not None else getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == "_CONFIG_REGISTRY" for t in targets):
            return {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    return set()


def m_config_text_wins_merge(nodes):
    """The text tower's keys win, so `architectures` becomes the inner name."""
    for node in ast.walk(_class_node(nodes, "KimiK3Config")):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "merged" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            node.value.values = list(reversed(node.value.values))
            return
    raise LookupError("no `merged` dict literal in KimiK3Config")


def m_config_drops_text(nodes):
    """The text tower never lands, so the language fields read defaults."""
    for node in ast.walk(_class_node(nodes, "KimiK3Config")):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "merged" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            node.value = ast.Call(
                func=ast.Name(id="dict", ctx=ast.Load()),
                args=[ast.Name(id="kwargs", ctx=ast.Load())],
                keywords=[],
            )
            return
    raise LookupError("no `merged` dict literal in KimiK3Config")


def m_config_separate_text_copy(nodes):
    """`text_config` held as its own `KimiLinearConfig`, the copy the patch once kept.

    An override then lands on one object while the pools read the other.
    """
    cls = _class_node(nodes, "KimiK3Config")
    prop = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "text_config"]
    init = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    if not prop or not init:
        raise LookupError("KimiK3Config has no text_config property or no __init__")
    cls.body.remove(prop[0])
    init[0].body.append(
        ast.parse('self.text_config = KimiLinearConfig(**{**text, "model_type": "kimi_linear"})')
        .body[0]
    )


CONFIG_MUTANTS = {
    "the text tower wins the merge": m_config_text_wins_merge,
    "the text tower never lands": m_config_drops_text,
}

# `--json-model-override-args` cuts the published stack to this many layers.
OVERRIDE_LAYERS = 4
OVERRIDES = (
    {"num_hidden_layers": OVERRIDE_LAYERS},
    {"text_config": {"num_hidden_layers": OVERRIDE_LAYERS}},
)


def override_views(namespace, overrides, apply_overrides, text_config_of):
    """Layer counts the model, `hf_text_config` and the pools read after an override.

    `apply_overrides` and `text_config_of` are the engine's own
    `apply_model_config_overrides` and `get_hf_text_config`.
    """
    config = namespace["KimiK3Config"](**copy.deepcopy(published_config()))
    apply_overrides(config, copy.deepcopy(overrides))
    pools = namespace["get_kimi_linear_config"](config)
    split = [i for i in range(config.num_hidden_layers) if not config.is_kda_layer(i)]
    return (
        config.num_hidden_layers,
        text_config_of(config).num_hidden_layers,
        pools.num_hidden_layers,
        pools.full_attention_layer_ids == split,
    )


def check_entry(sources, before, repo):
    """What the engine reads before it builds anything.

    Two facts decide whether the published checkpoint loads at all. The
    top-level `architectures` names the class the registry has to hold, and the
    top-level config carries no language-model field, so the config class has to
    reach `text_config` for every one of them. Then an override of the layer
    count has to reach every reader.
    """
    failures = 0
    published = published_config()
    architecture = published["architectures"][0]
    text_architecture = published["text_config"]["architectures"][0]

    names = entry_class_names(sources[MODEL_FILE])
    covered = architecture in names
    print(f"      EntryClass registers {sorted(names)}")
    print(f"      published architectures [{architecture}]: "
          f"{'covered' if covered else 'NOT COVERED'}")
    if not covered:
        failures += 1

    # Control: the name the text tower carries belongs to Kimi-Linear-48B, and
    # resolving K3 through it would serve the wrong model.
    print(f"      control (text tower's {text_architecture} used instead): "
          f"{'caught' if text_architecture not in names else 'NOT DETECTED'}")
    if text_architecture in names:
        failures += 1

    # Control: the model file is new, so the unpatched tree has no EntryClass.
    # An empty scan of an empty file proves nothing on its own, so the same
    # scanner also reads a model file the patch never touches and has to come
    # back with that file's own names.
    absent = before[MODEL_FILE] == ""
    neighbor = entry_class_names(sources[KIMI_LINEAR_MODEL_FILE])
    reads = bool(neighbor) and architecture not in neighbor
    print(f"      control (the model file on the unpatched tree): "
          f"{'absent' if absent else 'PRESENT'}; "
          f"the same scan on kimi_linear.py reads {sorted(neighbor)}")
    if not absent or not reads:
        failures += 1

    # Without an entry here nothing claims model_type "kimi_k3", so AutoConfig
    # refuses the checkpoint before any of the above runs.
    types = registered_config_types(sources[HF_UTILS_FILE])
    registered = "KimiK3Config" in types
    print(f"      _CONFIG_REGISTRY holds KimiK3Config: {'yes' if registered else 'NO'} "
          f"({len(types)} config classes)")
    if not registered:
        failures += 1

    # Control: the same scan on the unpatched tree has to find the other config classes
    # and not this one. An empty result means the scan reads nothing.
    stale_types = registered_config_types(before[HF_UTILS_FILE])
    caught = "KimiK3Config" not in stale_types and len(stale_types) > 1
    print(f"      control (same scan on the unpatched tree): {'caught' if caught else 'NOT DETECTED'} "
          f"({len(stale_types)} config classes, KimiK3Config absent: "
          f"{'KimiK3Config' not in stale_types})")
    if not caught:
        failures += 1

    namespace, config = built_config(sources)
    text = published["text_config"]
    checks = {
        "architectures": (list(config.architectures), published["architectures"]),
        "model_type": (config.model_type, published["model_type"]),
        "hidden_size": (config.hidden_size, text["hidden_size"]),
        "num_hidden_layers": (config.num_hidden_layers, text["num_hidden_layers"]),
        "hidden_act": (config.hidden_act, text["hidden_act"]),
        "attn_res_block_size": (config.attn_res_block_size, text["attn_res_block_size"]),
        "routed_expert_hidden_size": (
            config.routed_expert_hidden_size,
            text["routed_expert_hidden_size"],
        ),
        "activation_situ_beta": (config.activation_situ_beta, text["activation_situ_beta"]),
        "activation_situ_linear_beta": (
            config.activation_situ_linear_beta,
            text["activation_situ_linear_beta"],
        ),
        "latent_moe_use_norm": (config.latent_moe_use_norm, text["latent_moe_use_norm"]),
        "mla_use_output_gate": (config.mla_use_output_gate, text["mla_use_output_gate"]),
        "mla_use_nope": (config.mla_use_nope, text["mla_use_nope"]),
        "num_experts": (config.num_experts, text["num_experts"]),
        "num_experts_per_token": (config.num_experts_per_token, text["num_experts_per_token"]),
        "num_shared_experts": (config.num_shared_experts, text["num_shared_experts"]),
        "first_k_dense_replace": (config.first_k_dense_replace, text["first_k_dense_replace"]),
        "rms_norm_eps": (config.rms_norm_eps, text["rms_norm_eps"]),
        "vocab_size": (config.vocab_size, text["vocab_size"]),
        "text_config.num_attention_heads": (
            config.text_config.num_attention_heads,
            text["num_attention_heads"],
        ),
        "linear_attn_config": (config.linear_attn_config, text["linear_attn_config"]),
    }
    wrong = [f"{k}={got!r} want {want!r}" for k, (got, want) in checks.items() if got != want]
    print(f"      KimiK3Config answers {len(checks)} published fields: "
          f"{'all' if not wrong else 'MISSED ' + '; '.join(wrong)}")
    if wrong:
        failures += 1

    # Control: the top level alone carries no language model, so a config built
    # without reaching `text_config` can't answer for one.
    top_only = {k: v for k, v in published.items() if k not in ("text_config", "vision_config")}
    bare = namespace["KimiLinearConfig"](**copy.deepcopy(top_only))
    flat = bare.hidden_size == text["hidden_size"] and bare.linear_attn_config is not None
    print(f"      control (top level read as the whole config): "
          f"{'caught' if not flat else 'NOT DETECTED'} hidden_size={bare.hidden_size}")
    if flat:
        failures += 1

    want = (published["architectures"], text["hidden_size"], text["attn_res_block_size"])
    failures += report_mutants(
        CONFIG_MUTANTS,
        lambda mutate: built_config(sources, mutate=mutate)[1],
        lambda cfg: np.array(
            [
                float(list(cfg.architectures) == want[0]),
                float((cfg.hidden_size or 0) == want[1]),
                float((cfg.attn_res_block_size or 0) == want[2]),
            ]
        ),
        np.ones(3),
    )
    failures += check_overrides(sources, namespace, config, repo)

    print(f"  [{'FAIL' if failures else 'PASS'}] the published checkpoint resolves")
    return failures


def check_overrides(sources, namespace, config, repo):
    """One set of fields, which an override of the layer count has to reach.

    `ModelConfig` applies `--json-model-override-args` to the config the loader
    hands the model, and the pools size themselves from what
    `get_kimi_linear_config` returns. A second copy of the text fields would
    let the model build four layers while the pools expect 93, and the first
    forward would fail in `HybridLinearKVPool.replace_buffer`. Check 10 serves
    through that override.
    """
    import_patched(repo)
    from sgl_jax.srt.hf_transformers_utils import (
        apply_model_config_overrides,
        get_hf_text_config,
    )

    failures = 0
    same = config.text_config is config and namespace["get_kimi_linear_config"](config) is config
    print(f"      KimiK3Config is its own text_config and what the pools read: "
          f"{'yes' if same else 'NO'}")
    if not same:
        failures += 1
    for overrides in OVERRIDES:
        model, text, pools, split = override_views(
            namespace, overrides, apply_model_config_overrides, get_hf_text_config
        )
        ok = (model, text, pools) == (OVERRIDE_LAYERS,) * 3 and split
        print(f"      override {json.dumps(overrides)}: the model builds {model} layers, "
              f"hf_text_config says {text}, the pools {pools}"
              f"{'' if split else ', and the pools split the layers differently'}")
        if not ok:
            failures += 1

    def views(ns):
        return np.array(
            [
                count
                for overrides in OVERRIDES
                for count in override_views(
                    ns, overrides, apply_model_config_overrides, get_hf_text_config
                )[:3]
            ],
            np.float64,
        )

    failures += report_mutants(
        {"text_config held as a separate copy": m_config_separate_text_copy},
        lambda mutate: config_classes(sources, mutate=mutate),
        views,
        np.full(3 * len(OVERRIDES), OVERRIDE_LAYERS, np.float64),
    )
    return failures


# ----------------------------------------------------------------------------
# 9. the model the loader builds
# ----------------------------------------------------------------------------

# The server writes these onto the config object just before the loader builds
# the model, and the layers have to read them back.
RUNTIME_FLAGS = {
    "ep_size": 8,
    "moe_dp_size": 1,
    "ep_num_redundant_experts": 0,
    "moe_backend": "epmoe",
    "use_absorbed_mla": False,
    "enable_sequence_parallel": False,
}


def import_patched(repo):
    path = os.path.join(repo, "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    from sgl_jax.srt.configs.kimi_linear import KimiK3Config
    from sgl_jax.srt.models import kimi_k3

    return KimiK3Config, kimi_k3


def apply_mapping(shape, mapping):
    """The shape a checkpoint tensor has after the loader's transforms.

    `WeightLoader._process_and_assign_weight` transposes first, then narrows,
    reshapes and repeats, so this does the same in the same order.
    """
    if mapping.transpose_axes is not None:
        shape = tuple(shape[axis] for axis in mapping.transpose_axes)
    elif mapping.transpose:
        shape = (shape[1], shape[0])
    if mapping.narrow is not None:
        axis, length = mapping.narrow
        if shape[axis] < length:
            raise ValueError(f"narrow {tuple(mapping.narrow)} can't cut {shape[axis]} entries")
        shape = tuple(length if i == axis else dim for i, dim in enumerate(shape))
    if mapping.reshape is not None:
        if int(np.prod(shape)) != int(np.prod(mapping.reshape)):
            raise ValueError(f"reshape {tuple(mapping.reshape)} can't hold {int(np.prod(shape))}")
        shape = tuple(mapping.reshape)
    if mapping.repeat is not None:
        axis, times = mapping.repeat
        shape = tuple(dim * times if i == axis else dim for i, dim in enumerate(shape))
    return shape


def resolve_param(model, path):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return tuple(obj.get_raw_value().shape)


def compare_mappings(model, mappings, shapes, covered_layers, externally_loaded):
    """Every mapping against the published shapes, both directions."""
    wrong = []
    checked = 0
    for hf_key, mapping in mappings.items():
        if hf_key not in shapes:
            continue
        try:
            got = apply_mapping(shapes[hf_key], mapping)
        except ValueError as exc:
            wrong.append(f"{hf_key}: {exc}")
            continue
        want = resolve_param(model, mapping.target_path)
        checked += 1
        if got != want:
            wrong.append(f"{hf_key}: checkpoint gives {got}, the parameter is {want}")
        elif len(mapping.sharding) != len(got):
            wrong.append(f"{hf_key}: sharding {mapping.sharding} has the wrong rank for {got}")

    def layer_of(key):
        parts = key.split(".")
        if len(parts) > 4 and parts[2] == "layers" and parts[3].isdigit():
            return int(parts[3])
        return None

    absent = [
        key
        for key in mappings
        if key not in shapes and layer_of(key) in covered_layers | {None}
    ]
    uncovered = [key for key in shapes if key not in mappings and not externally_loaded(key)]
    return checked, wrong, absent, uncovered


def check_loader(repo, sources, workdir):
    """The model the loader builds, against the published checkpoint.

    `nnx.eval_shape` builds all 93 layers at full size without allocating a
    byte, so every parameter this model declares is the parameter the server
    would allocate. It runs under `jax.set_mesh` on the mesh
    `create_device_mesh` returns, which is how `JAXModelLoader` builds it. Then
    every weight mapping is measured against the shapes in the published
    safetensors headers, and the MXFP4 experts are read into parameters built
    the same way.
    """
    from flax import nnx

    failures = 0
    KimiK3Config, kimi_k3 = import_patched(repo)
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    config = KimiK3Config(**copy.deepcopy(published_config()))
    for name, value in RUNTIME_FLAGS.items():
        setattr(config, name, value)

    mesh = create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: kimi_k3.KimiK3ForCausalLM(config, mesh, jnp.bfloat16)
        )

    # The flags the server writes have to survive into the layers.
    moe_layer = next(layer for layer in model.model.layers if layer.is_moe_layer)
    mla_layer = next(layer for layer in model.model.layers if not layer.is_kda)
    got_ep = moe_layer.block_sparse_moe.experts.ep_size
    got_absorbed = mla_layer.self_attn.use_absorbed
    print(f"      --ep-size {RUNTIME_FLAGS['ep_size']} reaches EPMoE: ep_size={got_ep}; "
          f"use_absorbed_mla={RUNTIME_FLAGS['use_absorbed_mla']} reaches MLA: "
          f"use_absorbed={got_absorbed}")
    if got_ep != RUNTIME_FLAGS["ep_size"] or got_absorbed != RUNTIME_FLAGS["use_absorbed_mla"]:
        print("      FAIL: the model dropped a flag the server wrote")
        failures += 1
    experts_per_shard = config.num_experts // got_ep
    print(f"      {config.num_experts} experts split {experts_per_shard} to a shard")

    # What the layers took from the config, at the sizes the checkpoint ships.
    experts = moe_layer.block_sparse_moe.experts
    latent = moe_layer.block_sparse_moe
    text = published_config()["text_config"]
    built = {
        "EPMoE activation": (experts.activation, "situ"),
        "EPMoE situ_beta": (experts.situ_beta, text["activation_situ_beta"]),
        "EPMoE situ_linear_beta": (
            experts.situ_linear_beta,
            text["activation_situ_linear_beta"],
        ),
        "EPMoE hidden_size": (experts.hidden_size, text["routed_expert_hidden_size"]),
        "latent norm": (latent.routed_expert_norm is not None, text["latent_moe_use_norm"]),
        "MLA output gate": (mla_layer.self_attn.use_output_gate, text["mla_use_output_gate"]),
        "MLA has no rotary table": (mla_layer.self_attn.rotary_emb is None, text["mla_use_nope"]),
        "KDA full-rank gate": (
            model.model.layers[0].self_attn.use_full_rank_gate,
            text["linear_attn_config"]["use_full_rank_gate"],
        ),
        # The published header, not a constant here, says what the bias holds.
        "router bias dtype": (
            jnp.dtype(latent.gate.bias.get_value().dtype).name,
            {"F32": "float32", "BF16": "bfloat16"}[
                shard_header(LAYER_SHARDS[1])[
                    "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
                ]["dtype"]
            ],
        ),
    }
    off = [f"{k}={got!r} want {want!r}" for k, (got, want) in built.items() if got != want]
    print(f"      the layers carry {len(built)} published settings: "
          f"{'all' if not off else 'MISSED ' + '; '.join(off)}")
    if off:
        failures += 1

    failures += check_output_gate(kimi_k3)

    mappings = model._create_weight_mappings()
    covered_layers = set(LAYER_SHARDS)
    shards = list(LAYER_SHARDS.values()) + [MODEL_LEVEL_SHARD, *VISION_SHARDS]
    shapes = published_shapes(shards)
    checked, wrong, absent, uncovered = compare_mappings(
        model, mappings, shapes, covered_layers, kimi_k3.is_externally_loaded_key
    )
    print(f"      {len(mappings)} mappings, {len(shapes)} published tensors in shards "
          f"{shards}: {checked} compared")
    print(f"      shapes that disagree: {len(wrong)}")
    for item in wrong[:8]:
        print(f"        {item}")
    print(f"      mapped but absent from the checkpoint: {len(absent)} {absent[:4]}")
    print(f"      published tensors no mapping and no predicate covers: "
          f"{len(uncovered)} {uncovered[:4]}")
    failures += bool(wrong) + bool(absent) + bool(uncovered)

    vision = [key for key in shapes if key.startswith(("vision_tower.", "mm_projector."))]
    routed = [key for key in shapes if kimi_k3.is_routed_expert_key(key)]
    print(f"      the predicate accounts for {len(vision)} vision tensors and "
          f"{len(routed)} MXFP4 expert tensors")
    if not vision or not routed:
        failures += 1

    # Control: without the vision half of the predicate those tensors go
    # uncovered, which is what coverage validation raises on.
    _, _, _, left = compare_mappings(
        model, mappings, shapes, covered_layers, kimi_k3.is_routed_expert_key
    )
    print(f"      control (vision dropped from the predicate): "
          f"{'caught' if len(left) == len(vision) else 'NOT DETECTED'} "
          f"{len(left)} uncovered")
    if len(left) != len(vision):
        failures += 1

    # Controls: A_log read one value per channel, the reading this patch
    # dropped, and A_log reshaped to the head count with its padding left on.
    from sgl_jax.srt.utils.weight_utils import WeightMapping

    heads = config.linear_attn_config["num_heads"]
    head_dim = config.linear_attn_config["head_dim"]
    a_log_keys = [key for key in mappings if key.endswith(".A_log")]
    wrong_readings = {
        "A_log read per channel": {
            "sharding": ("tensor", None),
            "reshape": (1, head_dim),
            "repeat": (0, heads),
        },
        "A_log reshaped to the head count with its padding left on": {
            "sharding": (None, None, "tensor", None),
            "reshape": (1, 1, heads, 1),
        },
    }
    for label, fields in wrong_readings.items():
        if not a_log_keys:
            print(f"      control ({label}): NO A_log MAPPING TO PERTURB")
            failures += 1
            continue
        perturbed = dict(mappings)
        for key in a_log_keys:
            perturbed[key] = WeightMapping(
                target_path=mappings[key].target_path, transpose=False, **fields
            )
        _, bad, _, _ = compare_mappings(
            model, perturbed, shapes, covered_layers, kimi_k3.is_externally_loaded_key
        )
        print(f"      control ({label}): {'caught' if bad else 'NOT DETECTED'} {bad[:1]}")
        if not bad:
            failures += 1

    # The head loads with the sharding ParallelLMHead declares, which splits the
    # vocabulary over data and tensor, and with the padding it asks for.
    head_key = f"{kimi_k3.TEXT_PREFIX}.lm_head.weight"
    declared = model.lm_head.weight_mapping("lm_head.embedding")
    given = mappings.get(head_key)
    same = given is not None and (
        tuple(given.sharding), given.pad_width, given.target_path
    ) == (tuple(declared.sharding), declared.pad_width, declared.target_path)
    print(f"      lm_head mapping: sharding {None if given is None else tuple(given.sharding)}, "
          f"ParallelLMHead declares {tuple(declared.sharding)}: {'same' if same else 'DIFFERENT'}")
    if not same:
        failures += 1

    failures += check_expert_stack(kimi_k3, workdir)
    print(f"  [{'FAIL' if failures else 'PASS'}] the model the loader builds")
    return failures


def check_output_gate(kimi_k3):
    """`KimiK3MLAAttention._pre_o_proj`, the sigmoid gate on the MLA output.

    The gate reads the layer input rather than the attention result, so both the
    absorbed and the non-absorbed path reach it with the same shape.
    """
    failures = 0
    rng = np.random.default_rng(67)
    tokens, width = 12, 32
    attn_out = rng.normal(size=(tokens, width))
    hidden = rng.normal(size=(tokens, width))
    gate_out = rng.normal(size=(tokens, width))
    pre_o_proj = kimi_k3.KimiK3MLAAttention._pre_o_proj
    layer = Stub(
        use_output_gate=True,
        g_proj=Stub(_call=lambda self, x: (jnp.asarray(gate_out, jnp.float32), None)),
    )
    got = pre_o_proj(layer, jnp.asarray(attn_out, jnp.float32), jnp.asarray(hidden, jnp.float32))
    want = attn_out / (1.0 + np.exp(-gate_out))
    rel = rel_error(got, want)
    print(f"      MLA output gate: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1

    layer.use_output_gate = False
    ungated = pre_o_proj(
        layer, jnp.asarray(attn_out, jnp.float32), jnp.asarray(hidden, jnp.float32)
    )
    rel = rel_error(ungated, attn_out)
    print(f"      control (gate off passes the attention output through): rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1
    moved = rel_error(ungated, want)
    print(f"      control (gate off compared against the gated answer): rel={moved:.3e}")
    if moved < MUTANT_TOL:
        failures += 1
    return failures


def check_expert_stack(kimi_k3, workdir):
    """The MXFP4 expert stacks, read into the parameters the loader builds.

    `JAXModelLoader._get_model` builds the model under `nnx.eval_shape` and
    `jax.set_mesh`, so every EPMoE stack reaches `_load_routed_experts` as a
    ShapeDtypeStruct whose sharding names an AbstractMesh, which has no devices.
    This builds a tiny model the same way on the mesh `create_device_mesh`
    returns, writes its experts as MXFP4 into a real safetensors file, and runs
    `_load_routed_experts` through a real `WeightLoader` at three
    expert-parallel widths. The callback runs once per device, so every width
    cuts the expert, input and output axes differently. Then the
    physical-to-logical map goes in as a permutation, so a loader that indexed
    the checkpoint by physical slot would read the wrong expert.
    """
    from flax import nnx

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh
    from sgl_jax.srt.utils.weight_utils import SequentialSafetensorManager, WeightLoader

    failures = 0
    path = os.path.join(workdir, "tiny-experts")
    config, tensors = write_tiny_checkpoint(path)
    text = config["text_config"]
    mesh = create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
    dense = tiny_dense_experts(tensors, text)

    def build(ep_size):
        model_config = ModelConfig(model_path=path, dtype="float32", moe_backend="epmoe")
        write_runtime_flags(model_config.hf_config, ep_size=ep_size)
        with jax.set_mesh(mesh):
            model = nnx.eval_shape(
                lambda: kimi_k3.KimiK3ForCausalLM(model_config.hf_config, mesh, jnp.float32)
            )
        return model_config, model

    def want_stack(layer, param_name, order=None):
        hf_name = dict((name, hf) for hf, name in kimi_k3.EXPERT_MATRICES)[param_name]
        index = ("w1", "w3", "w2").index(hf_name)
        count = text["num_experts"]
        order = range(count) if order is None else order
        return np.stack([dense[layer][int(e)][index].T for e in order])

    for ep_size in (1, 2, 8):
        model_config, model = build(ep_size)
        loader = WeightLoader(model=model, model_config=model_config, mesh=mesh, dtype=jnp.float32)
        try:
            model._load_routed_experts(loader)
        except Exception as exc:  # noqa: BLE001 - the loader's own call has to work
            print(f"      ep_size={ep_size}: _load_routed_experts RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:160]}")
            failures += 1
            continue
        worst, layouts = 0.0, set()
        for layer_idx, layer in enumerate(model.model.layers):
            if not layer.is_moe_layer:
                continue
            for _, param_name in kimi_k3.EXPERT_MATRICES:
                got = getattr(layer.block_sparse_moe.experts, param_name).get_value()
                layouts.add(f"{param_name} {dict(got.sharding.mesh.shape)} {got.sharding.spec}")
                worst = max(worst, rel_error(np.asarray(got, np.float64),
                                             want_stack(layer_idx, param_name)))
        print(f"      ep_size={ep_size}: every routed stack read from MXFP4, rel={worst:.3e}")
        for layout in sorted(layouts):
            print(f"        {layout}")
        if worst >= F32_TOL:
            failures += 1

    # The expert-location map, as a permutation of the eight experts.
    model_config, model = build(2)
    loader = WeightLoader(model=model, model_config=model_config, mesh=mesh, dtype=jnp.float32)
    weight_info = loader.checkpoint_index()
    experts = model.model.layers[1].block_sparse_moe.experts
    target = experts.wi_0.get_value()
    source = f"{kimi_k3.TEXT_PREFIX}.model.layers.1.block_sparse_moe.experts"
    permutation = np.asarray(np.random.default_rng(53).permutation(text["num_experts"]))
    try:
        with SequentialSafetensorManager() as files:
            sharding = kimi_k3.expert_sharding(experts, target)
            permuted = model._stack_mxfp4_experts(
                weight_info, files, source, "w1", target, sharding, permutation
            )
            straight = model._stack_mxfp4_experts(
                weight_info, files, source, "w1", target, sharding
            )
    except Exception as exc:  # noqa: BLE001 - the stacker has to take the loader's parameter
        print(f"      experts permuted by the expert-location map: RAISED {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:160]}")
        return failures + 1
    rel = rel_error(np.asarray(permuted, np.float64), want_stack(1, "wi_0", permutation))
    print(f"      experts permuted by the expert-location map: rel={rel:.3e}")
    if rel >= F32_TOL:
        failures += 1
    # Control: the permuted read has to differ from the unpermuted one, or the
    # map went nowhere.
    moved = float(np.abs(np.asarray(straight, np.float64) - np.asarray(permuted, np.float64)).max())
    print(f"      control (the expert-location map ignored): max abs difference {moved:.3e}")
    if moved == 0.0:
        failures += 1
    return failures


# ----------------------------------------------------------------------------
# 10 and 11. the served model and the loader's other paths
# ----------------------------------------------------------------------------

# A four-layer Kimi K3 small enough to serve on CPU: a dense KDA layer, a routed
# KDA layer, a routed MLA layer and a routed KDA layer after it, with a block
# boundary every two layers, so the stash, both mixtures, the closing mixture
# and the MLA cache all run. Every field not named here is the published one.
TINY_TEXT = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "routed_expert_hidden_size": 32,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "num_hidden_layers": 4,
    "attn_res_block_size": 2,
    "vocab_size": 256,
    # Published at 1.0, where a dropped factor changes nothing.
    "routed_scaling_factor": 1.5,
    "max_position_embeddings": 2048,
}
TINY_LINEAR = {"num_heads": 8, "head_dim": 16, "kda_layers": [1, 2, 4], "full_attn_layers": [3]}
# A few names out of shards 95 and 96, so coverage validation sees the vision
# tower and the projector.
TINY_VISION = {
    "vision_tower.encoder.blocks.0.wqkv.weight": (12, 8),
    "vision_tower.encoder.blocks.0.norm0.weight": (8,),
    "mm_projector.proj.0.weight": (8, 8),
    "mm_projector.post_norm.weight": (8,),
}
# Two requests of different lengths share every prefill, so the KDA segments and
# the MLA masks both have to stop at a request boundary.
PROMPT_LENGTHS = (11, 6)
DECODE_STEPS = 2


def tiny_config() -> dict:
    """The published `config.json` with the language model cut down to size.

    The vision config goes: nothing in this file serves it, and the tiny
    checkpoint holds a handful of vision tensors only to exercise coverage.
    """
    config = copy.deepcopy(published_config())
    config["text_config"].update(TINY_TEXT)
    config["text_config"]["linear_attn_config"].update(TINY_LINEAR)
    config.pop("vision_config", None)
    for scope in (config, config["text_config"]):
        scope.update(pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config["media_placeholder_token_id"] = 3
    return config


def is_kda_layer(text, layer):
    return layer + 1 in text["linear_attn_config"]["kda_layers"]


def checkpoint_layout(text, layers=None) -> dict[str, tuple[tuple[int, ...], str]]:
    """Every language-model tensor as `(shape, dtype)`, in the published layout.

    Written from the modules `modeling_kimi_linear.py` declares: a torch
    `nn.Linear` stores `[out, in]`, a depthwise `ShortConvolution` stores
    `[D, 1, K]`, and the MXFP4 experts pack two codes per byte along the input.
    `check_tiny_layout` holds it to the published headers.
    """
    hidden = text["hidden_size"]
    lin = text["linear_attn_config"]
    heads, head_dim = lin["num_heads"], lin["head_dim"]
    proj = heads * head_dim
    conv = lin["short_conv_kernel_size"]
    mla_heads = text["num_attention_heads"]
    nope, rope, v_dim = text["qk_nope_head_dim"], text["qk_rope_head_dim"], text["v_head_dim"]
    q_rank, kv_rank = text["q_lora_rank"], text["kv_lora_rank"]
    latent, inter = text["routed_expert_hidden_size"], text["moe_intermediate_size"]
    shared = inter * text["num_shared_experts"]
    out = {}
    base = "language_model.model"
    for i in range(text["num_hidden_layers"]) if layers is None else layers:
        layer = f"{base}.layers.{i}"
        for name in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm",
                     "mlp_res_norm"):
            out[f"{layer}.{name}.weight"] = ((hidden,), "BF16")
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            out[f"{layer}.{name}.weight"] = ((1, hidden), "BF16")
        attn = f"{layer}.self_attn"
        if is_kda_layer(text, i):
            for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                out[f"{attn}.{name}.weight"] = ((proj, hidden), "BF16")
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                out[f"{attn}.{name}.weight"] = ((proj, 1, conv), "F32")
            out[f"{attn}.A_log"] = ((head_dim,), "F32")
            out[f"{attn}.dt_bias"] = ((proj,), "F32")
            out[f"{attn}.f_a_proj.weight"] = ((head_dim, hidden), "BF16")
            out[f"{attn}.f_b_proj.weight"] = ((proj, head_dim), "BF16")
            out[f"{attn}.b_proj.weight"] = ((heads, hidden), "BF16")
            out[f"{attn}.o_norm.weight"] = ((head_dim,), "F32")
            out[f"{attn}.o_proj.weight"] = ((hidden, proj), "BF16")
        else:
            out[f"{attn}.q_a_proj.weight"] = ((q_rank, hidden), "BF16")
            out[f"{attn}.q_a_layernorm.weight"] = ((q_rank,), "BF16")
            out[f"{attn}.q_b_proj.weight"] = ((mla_heads * (nope + rope), q_rank), "BF16")
            out[f"{attn}.kv_a_proj_with_mqa.weight"] = ((kv_rank + rope, hidden), "BF16")
            out[f"{attn}.kv_a_layernorm.weight"] = ((kv_rank,), "BF16")
            out[f"{attn}.kv_b_proj.weight"] = ((mla_heads * (nope + v_dim), kv_rank), "BF16")
            out[f"{attn}.o_proj.weight"] = ((hidden, mla_heads * v_dim), "BF16")
            out[f"{attn}.g_proj.weight"] = ((mla_heads * v_dim, hidden), "BF16")
        if i < text["first_k_dense_replace"]:
            dense = text["intermediate_size"]
            out[f"{layer}.mlp.gate_proj.weight"] = ((dense, hidden), "BF16")
            out[f"{layer}.mlp.up_proj.weight"] = ((dense, hidden), "BF16")
            out[f"{layer}.mlp.down_proj.weight"] = ((hidden, dense), "BF16")
            continue
        moe = f"{layer}.block_sparse_moe"
        out[f"{moe}.gate.weight"] = ((text["num_experts"], hidden), "BF16")
        out[f"{moe}.gate.e_score_correction_bias"] = ((text["num_experts"],), "F32")
        out[f"{moe}.routed_expert_down_proj.weight"] = ((latent, hidden), "BF16")
        out[f"{moe}.routed_expert_up_proj.weight"] = ((hidden, latent), "BF16")
        out[f"{moe}.routed_expert_norm.weight"] = ((latent,), "BF16")
        out[f"{moe}.shared_experts.gate_proj.weight"] = ((shared, hidden), "BF16")
        out[f"{moe}.shared_experts.up_proj.weight"] = ((shared, hidden), "BF16")
        out[f"{moe}.shared_experts.down_proj.weight"] = ((hidden, shared), "BF16")
        for e in range(text["num_experts"]):
            for name, (rows, cols) in (("w1", (inter, latent)), ("w3", (inter, latent)),
                                       ("w2", (latent, inter))):
                out[f"{moe}.experts.{e}.{name}.weight_packed"] = ((rows, cols // 2), "U8")
                out[f"{moe}.experts.{e}.{name}.weight_scale"] = ((rows, cols // 32), "U8")
    if layers is None:
        out[f"{base}.embed_tokens.weight"] = ((text["vocab_size"], hidden), "BF16")
        out[f"{base}.norm.weight"] = ((hidden,), "BF16")
        out[f"{base}.output_attn_res_norm.weight"] = ((hidden,), "BF16")
        out[f"{base}.output_attn_res_proj.weight"] = ((1, hidden), "BF16")
        out["language_model.lm_head.weight"] = ((text["vocab_size"], hidden), "BF16")
    return out


def check_tiny_layout():
    """The tiny checkpoint's layout, sized up to the published config, is the published one."""
    text = published_config()["text_config"]
    layout = checkpoint_layout(text, layers=sorted(LAYER_SHARDS))
    layout.update(checkpoint_layout(dict(text, num_hidden_layers=0)))
    headers = published_headers([*LAYER_SHARDS.values(), MODEL_LEVEL_SHARD])

    def disagreements(candidate):
        wrong = sorted(set(candidate) ^ set(headers))
        wrong += [key for key in candidate if key in headers and candidate[key] != headers[key]]
        return wrong

    wrong = disagreements(layout)
    print(f"      checkpoint_layout at the published size against shards "
          f"{[*LAYER_SHARDS.values(), MODEL_LEVEL_SHARD]}: {len(layout)} tensors, "
          f"{len(wrong)} disagree {wrong[:3]}")
    failures = 1 if wrong else 0
    # Control: the short convolutions written [D, K] instead of [D, 1, K].
    flattened = {
        key: ((shape[0], shape[2]), dtype) if len(shape) == 3 else (shape, dtype)
        for key, (shape, dtype) in layout.items()
    }
    caught = len(disagreements(flattened))
    print(f"      control (short convolutions written [D, K]): "
          f"{'caught' if caught else 'NOT DETECTED'} {caught} disagree")
    if not caught:
        failures += 1
    return failures


def published_headers(shards) -> dict[str, tuple[tuple[int, ...], str]]:
    out = {}
    for index in shards:
        out.update(
            {key: (tuple(value["shape"]), value["dtype"]) for key, value in shard_header(index).items()}
        )
    return out


def random_tensor(rng, key, shape, dtype, heads=None):
    """Weights in the published dtype, at scales that keep every layer's output order one.

    `A_log` holds one log decay for each of the `heads` KDA heads and zeros up to
    `head_dim`, the way check 6 reads the published file.
    """
    import ml_dtypes

    if dtype == "U8":
        if key.endswith("weight_scale"):
            # E8M0 exponents 118 to 124: scales from 2^-9 to 2^-3.
            return rng.integers(118, 125, size=shape).astype(np.uint8)
        return rng.integers(0, 256, size=shape).astype(np.uint8)
    if key.endswith("A_log"):
        value = np.log(rng.uniform(1.0, 16.0, size=shape))
        value[heads:] = 0.0
    elif key.endswith(("dt_bias", "conv1d.weight")):
        value = rng.normal(size=shape) * 0.5
    elif key.endswith("e_score_correction_bias"):
        value = rng.normal(size=shape) * 0.1
    elif len(shape) == 1:
        value = 1.0 + rng.normal(size=shape) * 0.1
    elif key.endswith("res_proj.weight"):
        value = rng.normal(size=shape) * 0.3
    else:
        value = rng.normal(size=shape) / np.sqrt(shape[-1])
    return value.astype(np.float32 if dtype == "F32" else ml_dtypes.bfloat16)


def write_tiny_checkpoint(path, seed=0, text_update=None, perturb=None, extra=None):
    """`config.json` and one safetensors file, the way the published repo lays them out."""
    from safetensors.numpy import save_file

    os.makedirs(path, exist_ok=True)
    config = tiny_config()
    config["text_config"].update(text_update or {})
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1)
    rng = np.random.default_rng(seed)
    tensors = {}
    heads = config["text_config"]["linear_attn_config"]["num_heads"]
    for key, (shape, dtype) in sorted(checkpoint_layout(config["text_config"]).items()):
        tensors[key] = random_tensor(rng, key, shape, dtype, heads)
    for key, shape in TINY_VISION.items():
        tensors[key] = random_tensor(rng, key, shape, "BF16")
    tensors.update(extra or {})
    if perturb is not None:
        perturb(tensors)
    save_file(tensors, os.path.join(path, "model.safetensors"))
    return config, tensors


def write_long_checkpoint(path, text):
    """The tiny checkpoint with one more MLA layer on the end of the file.

    `--json-model-override-args` cuts the fifth layer back off. The first four
    layers and the model-level tensors are the tiny checkpoint's own, so the
    model left after the cut is the one the other launches serve. The fifth layer
    draws from a random stream of its own.
    """
    layers = text["num_hidden_layers"]
    linear = text["linear_attn_config"]
    long_text = dict(text, num_hidden_layers=layers + 1)
    long_text["linear_attn_config"] = dict(
        linear, full_attn_layers=[*linear["full_attn_layers"], layers + 1]
    )
    rng = np.random.default_rng(1)
    extra = {
        key: random_tensor(rng, key, shape, dtype, linear["num_heads"])
        for key, (shape, dtype) in sorted(checkpoint_layout(long_text, layers=[layers]).items())
    }
    config, tensors = write_tiny_checkpoint(path, extra=extra)
    config["text_config"] = long_text
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1)
    return tensors


def tiny_dense_experts(tensors, text):
    """Every routed expert decoded by the bit-field reference: `{layer: [(w1, w3, w2)]}`."""
    out = {}
    for layer in range(text["num_hidden_layers"]):
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
        if f"{prefix}.0.w1.weight_packed" not in tensors:
            continue
        out[layer] = [
            tuple(
                ref_dequant_mxfp4(
                    tensors[f"{prefix}.{e}.{name}.weight_packed"],
                    tensors[f"{prefix}.{e}.{name}.weight_scale"],
                    32,
                )
                for name in ("w1", "w3", "w2")
            )
            for e in range(text["num_experts"])
        ]
    return out


# The float64 forward. Every function below is written from
# `modeling_kimi_linear.py` in the checkpoint repo, with the fla kernels it
# calls written out: `ShortConvolution` is a causal depthwise conv with a silu,
# `chunk_kda` is `ref_kda_recurrence`, and `FusedRMSNormGated` is an RMS norm
# times a sigmoid gate. Nothing here imports the patch.


def ref_short_conv(x, weight):
    """`ShortConvolution` on one sequence with no history: `weight` is `[D, 1, K]`."""
    kernel = weight[:, 0, :]
    width = kernel.shape[1]
    padded = np.concatenate([np.zeros((width - 1, x.shape[1])), x], axis=0)
    y = np.stack([np.sum(padded[t : t + width] * kernel.T, axis=0) for t in range(x.shape[0])])
    return y / (1.0 + np.exp(-y))


def ref_kda(w, prefix, h, text):
    lin = text["linear_attn_config"]
    heads, dim = lin["num_heads"], lin["head_dim"]
    tokens = h.shape[0]
    q, k, v = (
        ref_short_conv(h @ w[f"{prefix}.{n}_proj.weight"].T, w[f"{prefix}.{n}_conv1d.weight"])
        for n in "qkv"
    )
    shape = (tokens, heads, dim)
    raw = (h @ w[f"{prefix}.f_a_proj.weight"].T) @ w[f"{prefix}.f_b_proj.weight"].T
    beta = 1.0 / (1.0 + np.exp(-(h @ w[f"{prefix}.b_proj.weight"].T)))
    # One log decay per head: the first `heads` entries of the padded vector, as
    # `torch.empty(self.num_heads)` in the modeling file and `A_log + i_h` in
    # fla's kernels read it.
    decay = per_head_table(w[f"{prefix}.A_log"], heads, dim)
    out = ref_kda_recurrence(
        q.reshape(shape), k.reshape(shape), v.reshape(shape), raw.reshape(shape), beta, decay,
        w[f"{prefix}.dt_bias"].reshape(heads, dim), [tokens], lin["gate_lower_bound"],
        dim**-0.5,
    )
    output_gate = (h @ w[f"{prefix}.g_proj.weight"].T).reshape(tokens, heads, dim)
    out = ref_rms(out, w[f"{prefix}.o_norm.weight"], text["rms_norm_eps"])
    out = out / (1.0 + np.exp(-output_gate))
    return out.reshape(tokens, heads * dim) @ w[f"{prefix}.o_proj.weight"].T


def ref_mla(w, prefix, h, text):
    heads = text["num_attention_heads"]
    nope, rope, v_dim = text["qk_nope_head_dim"], text["qk_rope_head_dim"], text["v_head_dim"]
    tokens = h.shape[0]
    # `KimiRMSNorm` defaults to 1e-6, and neither LoRA norm is handed rms_norm_eps.
    q = ref_rms(h @ w[f"{prefix}.q_a_proj.weight"].T, w[f"{prefix}.q_a_layernorm.weight"], 1e-6)
    q = (q @ w[f"{prefix}.q_b_proj.weight"].T).reshape(tokens, heads, nope + rope)
    latent = h @ w[f"{prefix}.kv_a_proj_with_mqa.weight"].T
    kv_rank = text["kv_lora_rank"]
    kv = ref_rms(latent[:, :kv_rank], w[f"{prefix}.kv_a_layernorm.weight"], 1e-6)
    kv = (kv @ w[f"{prefix}.kv_b_proj.weight"].T).reshape(tokens, heads, nope + v_dim)
    # NoPE: the rope slice is shared across heads and never rotates.
    rot = np.broadcast_to(latent[:, None, kv_rank:], (tokens, heads, rope))
    key = np.concatenate([kv[..., :nope], rot], axis=-1)
    scores = np.einsum("qhd,khd->hqk", q, key) * (nope + rope) ** -0.5
    scores = np.where(np.tril(np.ones((tokens, tokens), bool))[None], scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", probs, kv[..., nope:]).reshape(tokens, heads * v_dim)
    out = out / (1.0 + np.exp(-(h @ w[f"{prefix}.g_proj.weight"].T)))
    return out @ w[f"{prefix}.o_proj.weight"].T


def ref_mlp(w, prefix, h, text):
    act = ref_situ(
        h @ w[f"{prefix}.gate_proj.weight"].T,
        h @ w[f"{prefix}.up_proj.weight"].T,
        text["activation_situ_beta"],
        text["activation_situ_linear_beta"],
    )
    return act @ w[f"{prefix}.down_proj.weight"].T


def ref_moe(w, prefix, h, text, experts):
    """`KimiSparseMoeBlock` with `KimiMoEGate`: sigmoid scores, bias for the choice only."""
    scores = 1.0 / (1.0 + np.exp(-(h @ w[f"{prefix}.gate.weight"].T)))
    choice = scores + w[f"{prefix}.gate.e_score_correction_bias"][None, :]
    top = text["num_experts_per_token"]
    chosen = np.argsort(-choice, axis=-1)[:, :top]
    weight = np.take_along_axis(scores, chosen, axis=-1)
    weight = weight / (weight.sum(axis=-1, keepdims=True) + 1e-20) * text["routed_scaling_factor"]
    latent = h @ w[f"{prefix}.routed_expert_down_proj.weight"].T
    routed = np.zeros_like(latent)
    beta, linear_beta = text["activation_situ_beta"], text["activation_situ_linear_beta"]
    for n in range(h.shape[0]):
        for j in range(top):
            w1, w3, w2 = experts[int(chosen[n, j])]
            act = ref_situ(latent[n] @ w1.T, latent[n] @ w3.T, beta, linear_beta)
            routed[n] += weight[n, j] * (act @ w2.T)
    routed = ref_rms(routed, w[f"{prefix}.routed_expert_norm.weight"], text["rms_norm_eps"])
    routed = routed @ w[f"{prefix}.routed_expert_up_proj.weight"].T
    return routed + ref_mlp(w, f"{prefix}.shared_experts", h, text)


def ref_forward(tensors, text, ids, streams=None):
    """Logits at every position of one sequence, float64, from the checkpoint tensors.

    `streams`, when given, collects the prefix sum entering each layer, which is
    what the capture hook returns.
    """
    w = {key: np.asarray(value).astype(np.float64) for key, value in tensors.items()
         if value.dtype != np.uint8}
    experts = tiny_dense_experts(tensors, text)
    eps = text["rms_norm_eps"]
    base = "language_model.model"
    stream = w[f"{base}.embed_tokens.weight"][np.asarray(ids)]
    stash = []
    for i in range(text["num_hidden_layers"]):
        layer = f"{base}.layers.{i}"
        if streams is not None:
            streams.append(stream)
        prefix_sum = stream
        read = prefix_sum
        if stash:
            read = ref_mix(prefix_sum, stash, w[f"{layer}.self_attention_res_norm.weight"],
                           w[f"{layer}.self_attention_res_proj.weight"].reshape(-1), eps)
        if i % text["attn_res_block_size"] == 0:
            stash = stash + [prefix_sum]
            prefix_sum = None
        h = ref_rms(read, w[f"{layer}.input_layernorm.weight"], eps)
        attend = ref_kda if is_kda_layer(text, i) else ref_mla
        attn = attend(w, f"{layer}.self_attn", h, text)
        prefix_sum = attn if prefix_sum is None else prefix_sum + attn
        read = ref_mix(prefix_sum, stash, w[f"{layer}.mlp_res_norm.weight"],
                       w[f"{layer}.mlp_res_proj.weight"].reshape(-1), eps)
        h = ref_rms(read, w[f"{layer}.post_attention_layernorm.weight"], eps)
        if i < text["first_k_dense_replace"]:
            ffn = ref_mlp(w, f"{layer}.mlp", h, text)
        else:
            ffn = ref_moe(w, f"{layer}.block_sparse_moe", h, text, experts[i])
        stream = prefix_sum + ffn
    mixed = ref_mix(stream, stash, w[f"{base}.output_attn_res_norm.weight"],
                    w[f"{base}.output_attn_res_proj.weight"].reshape(-1), eps)
    final = ref_rms(mixed, w[f"{base}.norm.weight"], eps)
    return final @ w["language_model.lm_head.weight"].T


def write_runtime_flags(hf_config, ep_size=1):
    """What `ModelRunner.load_model` writes onto the config before the loader runs."""
    hf_config.enable_dp_lm_head = False
    hf_config.ep_size = ep_size
    hf_config.moe_dp_size = 1
    hf_config.ep_num_redundant_experts = 0
    hf_config.moe_backend = "epmoe"
    hf_config.use_jax_allreduce_metadata = True
    hf_config.use_absorbed_mla = False
    hf_config.use_dsa_sparse = False
    hf_config.enable_sequence_parallel = False
    hf_config.vision_encoder_parallel = False
    hf_config.precompile_vision_patch_paddings = None


# The engine runs float32 end to end. The Mega KDA prefill kernel takes BF16
# only, so prefill runs the chunked KDA kernel, which the backend falls back to
# whenever a 64-token tile holds more than two requests, and the short
# convolution keeps its history in float32 rather than the default BF16.
ENGINE_ENV = {
    "SGLANG_JAX_KDA_PREFILL_KERNEL": "chunked",
    "SGLANG_JAX_CONV_STATE_DTYPE": "float32",
}


# `ModelRunner` imports these and check 11's loaders don't: the routed-experts
# capturer reads `pybase64`, and the sampler's grammar masks read `llguidance`.
ENGINE_PACKAGES = ("pybase64", "llguidance")


def missing_engine_packages() -> list[str]:
    missing = []
    for module in ENGINE_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def install_line(repo, packages) -> str:
    """`uv pip install` for `packages`, pinned the way the checkout's `pyproject.toml` pins them."""
    import tomllib

    from packaging.requirements import Requirement

    with open(os.path.join(repo, "python", "pyproject.toml"), "rb") as fh:
        dependencies = tomllib.load(fh)["project"]["dependencies"]
    pins = {Requirement(dep).name: dep for dep in dependencies}
    return "uv pip install " + " ".join(f"'{pins.get(name, name)}'" for name in packages)


def serve(ckpt, dp, ep, capture, attention_backend="native", overrides="{}", extra=None):
    """Load, precompile and serve two requests the way the scheduler does.

    The mesh is `Scheduler.__init__`'s call, the worker is the `ModelWorker` the
    scheduler builds, `run_precompile` is its startup precompile, and the tree
    cache comes from `build_kv_cache`, which is `init_memory_pool_and_cache`.
    Each step then goes through the same calls the scheduler's event loop makes
    on a non-overlap, non-speculative server: `PrefillAdder` and
    `ScheduleBatch.prepare_for_extend` for the prefill, `prepare_for_decode`
    for each decode step, `get_model_worker_batch` and
    `forward_batch_generation` to run it, and the output bookkeeping of
    `_extract_dp_output_ids` and `process_batch_result_*`. Requests go to data
    ranks in turn. `overrides` is `--json-model-override-args`, and `extra`
    holds any other server arguments the launch sets.

    Returns the prompts, each request's logits per step, the per-layer streams
    the capture hook returned when `capture` is set, the layer count the model
    built, and the shardings two loaded weights carry.
    """
    from sgl_jax.srt.eplb.expert_location import set_global_expert_location_metadata
    from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
    from sgl_jax.srt.managers.schedule_policy import PrefillAdder
    from sgl_jax.srt.managers.tp_worker import ModelWorker
    from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache
    from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sgl_jax.srt.sampling.sampling_params import SamplingParams
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    server_kwargs = {
        "model_path": ckpt,
        "device": "cpu",
        "tp_size": 8,
        "dp_size": dp,
        "ep_size": ep,
        "moe_backend": "epmoe",
        "dtype": "float32",
        "skip_tokenizer_init": True,
        "attention_backend": attention_backend,
        "page_size": 1,
        "disable_radix_cache": True,
        "max_running_requests": 4 * dp,
        "max_total_tokens": 512,
        "max_prefill_tokens": 64,
        "chunked_prefill_size": 64,
        "precompile_token_paddings": [16 * dp],
        "precompile_bs_paddings": [2 * dp],
        "disable_overlap_schedule": True,
        "random_seed": 0,
        "enable_return_hidden_states": capture,
        "json_model_override_args": overrides,
        **(extra or {}),
    }
    logger.info("ServerArgs for the served model: %s", json.dumps(server_kwargs))
    server_args = ServerArgs(**server_kwargs)
    # A server process starts with no expert-location map. A launch that sets a
    # dispatch algorithm leaves one in this process, and EPMoE and ForwardBatch
    # read it back.
    set_global_expert_location_metadata(None)
    mesh = create_device_mesh(ici_parallelism=[dp, 8 // dp], dcn_parallelism=[1, 1])
    worker = ModelWorker(server_args=server_args, mesh=mesh)
    worker.run_precompile()

    req_to_token_pool, allocator = worker.get_memory_pool()
    tree_cache = build_kv_cache(
        server_args=server_args,
        model_config=worker.model_config,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=server_args.page_size,
        is_hybrid=False,
        is_hybrid_recurrent=isinstance(req_to_token_pool, HybridReqToTokenPool),
        sliding_window_size=None,
        tp_size=server_args.tp_size,
        spec_algorithm=None,
        mesh=mesh,
    )
    vocab = worker.model_config.vocab_size
    rng = np.random.default_rng(7)
    prompts = {
        f"r{n}": rng.integers(4, vocab, size=length).tolist()
        for n, length in enumerate(PROMPT_LENGTHS)
    }
    reqs = []
    for n, (rid, ids) in enumerate(prompts.items()):
        params = SamplingParams(max_new_tokens=DECODE_STEPS + 1, temperature=0.0)
        params.normalize(None)
        req = Req(rid, "", list(ids), params, dp_rank=n % dp, eos_token_ids=set(),
                  vocab_size=vocab, return_hidden_states=capture)
        req.logprob_start_len = len(ids) - 1
        logger.info(
            "request %s: dp_rank=%d prompt_ids=%s max_new_tokens=%d temperature=0.0 "
            "return_hidden_states=%s logprob_start_len=%d",
            rid, n % dp, list(ids), DECODE_STEPS + 1, capture, req.logprob_start_len,
        )
        reqs.append(req)

    served = {req.rid: [] for req in reqs}
    captured = {req.rid: [] for req in reqs}

    def run_batch(batch):
        mwb = batch.get_model_worker_batch(
            *worker.get_precompile_paddings(), server_args.page_size, False
        )
        output, next_ids, _ = worker.forward_batch_generation(mwb, sampling_metadata=None)
        next_ids = np.asarray(jax.device_get(next_ids))
        logits = np.asarray(jax.device_get(output.next_token_logits), np.float64)
        states = None
        if output.hidden_states is not None:
            states = np.asarray(jax.device_get(output.hidden_states), np.float64)
        extend = batch.forward_mode.is_extend()
        rows_per_rank = len(mwb.input_ids) // dp if extend else mwb.per_dp_bs_size
        for rank, info in enumerate(batch.reqs_info):
            count = len(info.reqs) if info.reqs else 0
            start = rank * mwb.per_dp_bs_size
            # Scheduler._extract_dp_output_ids: prepare_for_decode reads these.
            info.output_ids = next_ids[start : start + count] if count else np.array([], np.int32)
            row = rank * rows_per_rank
            for i, req in enumerate(info.reqs or []):
                width = info.extend_lens[i] if extend else 1
                served[req.rid].append(logits[start + i])
                if states is not None:
                    captured[req.rid].append(states[row : row + width])
                row += width
                # process_batch_result_prefill / _decode.
                req.output_ids.append(int(next_ids[start + i]))
                req.check_finished()
                if extend and not req.finished():
                    tree_cache.cache_unfinished_req(req)

    running = ScheduleBatch.init_new(
        reqs=[[] for _ in range(dp)], req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator, tree_cache=tree_cache,
        model_config=worker.model_config, enable_overlap=False, dp_size=dp,
        spec_algorithm=None, mesh=mesh,
    )
    adder = PrefillAdder(
        server_args.page_size, tree_cache, allocator, running, 1.0,
        server_args.max_prefill_tokens, server_args.chunked_prefill_size, 0, dp_size=dp,
    )
    for req in reqs:
        req.init_next_round_input(tree_cache)
        adder.add_one_req(req)
    batch = ScheduleBatch.init_new(
        [adder.can_run_list.get(rank, []) for rank in range(dp)], req_to_token_pool,
        allocator, tree_cache, worker.model_config, False, dp,
        enable_custom_logit_processor=False, chunked_reqs=[None] * dp, mesh=mesh,
        spec_algorithm=None,
    )
    batch.prepare_for_extend()
    run_batch(batch)
    for _ in range(DECODE_STEPS):
        batch.filter_batch()
        if not batch.check_decode_mem():
            raise RuntimeError("the KV pool ran out during a two-step decode")
        batch.prepare_for_decode()
        run_batch(batch)

    model = worker.model_runner.model
    moe = next(layer for layer in model.model.layers if layer.is_moe_layer).block_sparse_moe
    stack = moe.experts.wi_0.get_value()
    return {
        "prompts": prompts,
        "outputs": {req.rid: list(req.output_ids) for req in reqs},
        "served": served,
        "captured": captured if capture else None,
        "layers": len(model.model.layers),
        "lm_head_spec": model.lm_head.embedding.get_value().sharding.spec,
        "lm_head_declared": P(*model.lm_head.kernel_axes),
        "expert_spec": stack.sharding.spec,
        "expert_mesh": dict(stack.sharding.mesh.shape),
    }


def compare_served(label, result, tensors, text, verbose=True):
    """Every step of every request against the float64 forward of its own sequence.

    Each request's decode steps feed back the tokens the engine chose, so the
    forward reads the same sequence the engine served. A request that came back
    with the wrong number of steps fails, and so does any value that isn't
    finite: `rel_error` turns it into infinity, and the gates test `rel <
    ENGINE_TOL`, which a NaN also fails. With `verbose` off only the summary
    line prints.
    """
    failures = 0
    worst, lowest = 0.0, 1.0
    steps = 1 + DECODE_STEPS
    for rid, prompt in result["prompts"].items():
        streams = []
        sequence = list(prompt) + result["outputs"][rid][:DECODE_STEPS]
        reference = ref_forward(tensors, text, sequence, streams)
        if len(result["served"][rid]) != steps:
            print(f"        {rid}: {len(result['served'][rid])} steps served, not {steps}")
            failures += 1
        for step, got in enumerate(result["served"][rid]):
            want = reference[len(prompt) - 1 + step]
            rel, corr = rel_error(got, want), correlation(got, want)
            worst, lowest = max(worst, rel), float(np.minimum(lowest, corr))
            if not rel < ENGINE_TOL:
                if verbose:
                    print(f"        {rid} step {step}: rel={rel:.3e} corr={corr:.6f}")
                failures += 1
        if result["captured"] is None:
            continue
        layers = text["num_hidden_layers"]
        if len(result["captured"][rid]) != steps:
            print(f"        {rid}: {len(result['captured'][rid])} captured steps, not {steps}")
            failures += 1
        for step, rows in enumerate(result["captured"][rid]):
            start = 0 if step == 0 else len(prompt) - 1 + step
            got = rows.reshape(rows.shape[0], layers, -1)
            for layer in range(layers):
                want = streams[layer][start : start + rows.shape[0]]
                rel = rel_error(got[:, layer], want)
                worst = max(worst, rel)
                if not rel < ENGINE_TOL:
                    if verbose:
                        print(f"        {rid} step {step} layer {layer} input: rel={rel:.3e}")
                    failures += 1
    what = "logits and every layer's input" if result["captured"] is not None else "logits"
    print(f"      {label}: {len(result['prompts'])} requests x {steps} steps, {what} against "
          f"float64: worst rel={worst:.3e}, lowest logit corr={lowest:.6f}")
    return failures


def poisoned(result):
    """The same result with every served logit and captured row set to NaN."""
    out = dict(result)
    out["served"] = {
        rid: [np.full(np.shape(step), np.nan) for step in steps]
        for rid, steps in result["served"].items()
    }
    if result["captured"] is not None:
        out["captured"] = {
            rid: [np.full(np.shape(rows), np.nan) for rows in steps]
            for rid, steps in result["captured"].items()
        }
    return out


def gate_count(result, text):
    """How many gates `compare_served` applies to a result: one per step and layer input."""
    layers = text["num_hidden_layers"] if result["captured"] is not None else 0
    return sum(len(steps) * (1 + layers) for steps in result["served"].values())


def check_served(repo, workdir, sources):
    """The tiny checkpoint through the real engine, against the float64 forward."""
    failures = check_tiny_layout()
    missing = missing_engine_packages()
    if missing:
        print(f"      the engine's ModelRunner imports {missing}, which this Python lacks: "
              f"{install_line(repo, missing)}")
        print("  [FAIL] the served model")
        return failures + 1
    saved = {key: os.environ.get(key) for key in ENGINE_ENV}
    os.environ.update(ENGINE_ENV)
    try:
        return failures + _check_served(repo, workdir, sources)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _check_served(repo, workdir, sources):
    import_patched(repo)
    from sgl_jax.srt.eplb.expert_location import set_global_expert_location_metadata

    failures = 0
    ckpt = os.path.join(workdir, "tiny-served")
    config, tensors = write_tiny_checkpoint(ckpt)
    text = config["text_config"]
    # The same model with one more MLA layer in the file, which
    # --json-model-override-args cuts back off. The model, the KV pool and the
    # recurrent pool all have to see four layers, or the first forward fails.
    long_ckpt = os.path.join(workdir, "tiny-served-long")
    long_tensors = write_long_checkpoint(long_ckpt, text)
    cut = json.dumps({"num_hidden_layers": text["num_hidden_layers"]})

    # With a dispatch algorithm set, TopK sends the router's logical expert ids
    # through the expert-location map, whose gathers read ids sharded over the
    # batch. Two redundant experts put copies in physical slots of their own.
    def eplb(algorithm):
        return {"ep_dispatch_algorithm": algorithm, "ep_num_redundant_experts": 2}

    runs = {}
    launches = (
        ("ep_size 1, dp_size 1, capture on", ckpt, tensors, text, 1, 1, True, "{}", None),
        (f"ep_size 2, dp_size 1, a five-layer file and --json-model-override-args '{cut}'",
         long_ckpt, long_tensors, text, 1, 2, False, cut, None),
        ("ep_size 2, dp_size 2", ckpt, tensors, text, 2, 2, False, "{}", None),
        ("ep_size 2, dp_size 1, --ep-dispatch-algorithm static, 2 redundant experts",
         ckpt, tensors, text, 1, 2, False, "{}", eplb("static")),
        ("ep_size 2, dp_size 1, --ep-dispatch-algorithm dynamic, 2 redundant experts",
         ckpt, tensors, text, 1, 2, False, "{}", eplb("dynamic")),
    )
    for label, path, weights, reference_text, dp, ep, capture, overrides, extra in launches:
        try:
            runs[label] = serve(path, dp, ep, capture, overrides=overrides, extra=extra)
        except Exception as exc:  # noqa: BLE001 - the engine has to load and serve
            print(f"      {label}: RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:240]}")
            failures += 1
            continue
        finally:
            # Each worker holds its own pools and compiled programs. The map an
            # EPLB launch left stays out of every model built after it.
            set_global_expert_location_metadata(None)
            gc.collect()
        result = runs[label]
        failures += compare_served(label, result, weights, reference_text)
        print(f"        {result['layers']} layers built; experts on mesh "
              f"{result['expert_mesh']} as {result['expert_spec']}; "
              f"lm_head {result['lm_head_spec']}, declared {result['lm_head_declared']}")
        if result["layers"] != reference_text["num_hidden_layers"]:
            print(f"        FAIL: the model has {result['layers']} layers, not "
                  f"{reference_text['num_hidden_layers']}")
            failures += 1
        if result["lm_head_spec"] != result["lm_head_declared"]:
            print("        FAIL: lm_head loaded with a sharding ParallelLMHead doesn't declare")
            failures += 1
    if not runs:
        print("  [FAIL] the served model")
        return failures

    base = runs.get("ep_size 1, dp_size 1, capture on")
    if base is not None:
        # Control: the gates on a result whose every served logit and captured
        # layer is NaN. Every one of them has to fail.
        label = "control (every served logit and captured layer set to NaN)"
        caught = compare_served(label, poisoned(base), tensors, text, verbose=False)
        gates = gate_count(base, text)
        print(f"        {caught} of {gates} gates failed")
        if caught != gates:
            failures += 1

        # Control: in the file, every layer 1 expert's w2 scale one exponent
        # higher on the first half of its output channels. routed_expert_norm
        # undoes a uniform scale, so only half the channels double. The served
        # logits have to move with the file and land on the float64 forward of
        # the new weights.
        def perturb(weights):
            for key in weights:
                if key.startswith("language_model.model.layers.1.") and key.endswith(
                        "w2.weight_scale"):
                    scaled = weights[key].copy()
                    scaled[: scaled.shape[0] // 2] += np.uint8(1)
                    weights[key] = scaled

        moved_ckpt = os.path.join(workdir, "tiny-served-moved")
        _, moved_tensors = write_tiny_checkpoint(moved_ckpt, perturb=perturb)
        label = "control (half of layer 1's w2 channels doubled in the file)"
        try:
            moved = serve(moved_ckpt, 1, 1, False)
        except Exception as exc:  # noqa: BLE001
            print(f"      {label}: RAISED {type(exc).__name__}")
            failures += 1
        else:
            first = moved["served"]["r0"][0]
            shift = rel_error(first, base["served"]["r0"][0])
            print(f"      {label}: served prefill logits move by rel={shift:.3e}")
            if shift < MUTANT_TOL:
                failures += 1
            failures += compare_served("the edited checkpoint", moved, moved_tensors, text)
        finally:
            gc.collect()

    # Off TPU only --attention-backend native serves. With the default fa a CPU
    # run swaps in NativeAttention but keeps the absorbed-MLA latent pool, which
    # NativeAttention can't write. The startup precompile has to refuse that
    # and name the flag that works.
    label = "the default --attention-backend fa on CPU"
    try:
        serve(ckpt, 1, 1, False, attention_backend="fa")
    except NotImplementedError as exc:
        message = str(exc).splitlines()[0]
        named = "--attention-backend native" in message
        print(f"      {label}: refused{'' if named else ' WITHOUT NAMING THE FLAG'}: "
              f"{message[:200]}")
        if not named:
            failures += 1
    except Exception as exc:  # noqa: BLE001 - any other raise leaves the user no way forward
        print(f"      {label}: RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        failures += 1
    else:
        print(f"      {label}: served without refusing")
        failures += 1
    finally:
        gc.collect()

    failures += check_native_tpu_dp(repo, sources)
    print(f"  [{'FAIL' if failures else 'PASS'}] the served model")
    return failures


def one_rank_native_attention(sources, native_backend):
    """`NativeAttention` with its `__call__` reading the batch as one rank.

    That's how the TPU branch read it before this patch: `ranks` came out 1
    whenever `is_tpu_runtime` said so.
    """
    fn = find_method(sources[NATIVE_FILE], "NativeAttention", "__call__")
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ranks" for target in node.targets
        ):
            node.value = ast.Constant(value=1)
            break
    else:
        raise LookupError("no `ranks` assignment in NativeAttention.__call__")
    namespace = compile_nodes([fn], extra=dict(vars(native_backend)))
    return type("OneRankNativeAttention", (native_backend.NativeAttention,),
                {"__call__": namespace["__call__"]})


def check_native_tpu_dp(repo, sources):
    """`NativeAttention`'s TPU branch under `--dp-size 2`, run on CPU.

    On TPU the pool's sharded KV-update kernel writes each data rank's rows
    into that rank's shard, so each rank has to attend over its own shard. The
    kernel is a Pallas TPU kernel, so it runs here under
    `pltpu.force_tpu_interpret_mode()`, and `is_tpu_runtime` answers True for
    the backend so it takes the TPU branch. The batch is laid out the way
    `ScheduleBatch` lays it out at `--dp-size 2`: each rank's tokens, requests
    and `cache_loc` in their own padded block, and each rank's slots counted
    from 1. The reference attends each request over its own tokens in float64.
    """
    from types import SimpleNamespace

    from jax.experimental.pallas import tpu as pltpu

    import_patched(repo)
    from sgl_jax.srt.layers.attention import native_backend
    from sgl_jax.srt.layers.radix_attention import AttentionType
    from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    dp, heads, dim = 2, 4, 128
    per_rank_lengths = ([3, 2], [4])
    tokens_per_rank, requests_per_rank = 16, 4
    total = tokens_per_rank * dp
    rng = np.random.default_rng(71)
    q, k, v = (np.zeros((total, heads, dim), np.float32) for _ in range(3))
    seq_lens = np.zeros(requests_per_rank * dp, np.int32)
    out_cache_loc = np.full(total, -1, np.int32)
    cache_loc = np.zeros(total, np.int32)
    rows = {rank: [] for rank in range(dp)}
    for rank, lengths in enumerate(per_rank_lengths):
        row, slot = rank * tokens_per_rank, 1
        for i, length in enumerate(lengths):
            seq_lens[rank * requests_per_rank + i] = length
            for _ in range(length):
                q[row], k[row], v[row] = (rng.normal(size=(heads, dim)) for _ in range(3))
                out_cache_loc[row] = slot
                rows[rank].append(row)
                row += 1
                slot += 1
        cache_loc[rank * tokens_per_rank : rank * tokens_per_rank + slot - 1] = np.arange(1, slot)

    want = np.zeros((total, heads, dim))
    for rank, lengths in enumerate(per_rank_lengths):
        start = rank * tokens_per_rank
        for length in lengths:
            qs, ks, vs = (x[start : start + length].astype(np.float64) for x in (q, k, v))
            logits = np.einsum("qhd,khd->qhk", qs, ks) / np.sqrt(dim)
            causal = np.tril(np.ones((length, length), bool))[:, None, :]
            logits = np.where(causal, logits, -np.inf)
            weights = np.exp(logits - logits.max(-1, keepdims=True))
            weights /= weights.sum(-1, keepdims=True)
            want[start : start + length] = np.einsum("qhk,khd->qhd", weights, vs)
            start += length

    mesh = create_device_mesh(ici_parallelism=[dp, 8 // dp], dcn_parallelism=[1, 1])
    layer = SimpleNamespace(
        layer_id=0, head_dim=dim, scaling=None, attn_type=AttentionType.DECODER,
        q_head_num=heads, kv_head_num=heads, sliding_window_size=None, softmax_dtype=None,
    )
    fields = {
        "seq_lens": seq_lens,
        "cache_loc": cache_loc,
        "extend_prefix_lens": np.zeros_like(seq_lens),
        "extend_seq_lens": seq_lens,
        "out_cache_loc": out_cache_loc,
    }

    def per_rank_error(backend_class):
        with jax.set_mesh(mesh), pltpu.force_tpu_interpret_mode():
            pool = MHATokenToKVPool(
                size=64, page_size=1, dtype=jnp.float32, head_num=heads, head_dim=dim,
                layer_num=1, mesh=mesh, dp_size=dp,
            )
            backend = backend_class(heads, heads, mesh)
            by_token = NamedSharding(mesh, P("data", "tensor", None))
            by_rank = NamedSharding(mesh, P("data"))
            arrays = {name: jax.device_put(value, by_rank) for name, value in fields.items()}

            @jax.jit
            def attend(q, k, v, arrays):
                batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, **arrays)
                return backend(q, k, v, layer, batch, pool)

            out, _ = attend(*(jax.device_put(x, by_token) for x in (q, k, v)), arrays)
        got = np.asarray(out, np.float64).reshape(total, heads, dim)
        return {rank: rel_error(got[picked], want[picked]) for rank, picked in rows.items()}

    failures = 0
    label = "NativeAttention's TPU branch at --dp-size 2, the TPU KV write in interpret mode"
    saved = native_backend.is_tpu_runtime
    native_backend.is_tpu_runtime = lambda mesh=None: True
    try:
        errors = per_rank_error(native_backend.NativeAttention)
        worst = max(errors.values())
        print(f"      {label}: rel per rank "
              f"{', '.join(f'{rank}: {error:.3e}' for rank, error in errors.items())}")
        if not worst < F32_TOL:
            failures += 1
        # Control: the batch read as one rank, as the TPU branch read it before.
        control = "control (the TPU branch reads the batch as one rank)"
        try:
            mutant = one_rank_native_attention(sources, native_backend)
        except LookupError as exc:
            print(f"      {control}: MUTATION OPERATOR FOUND NO TARGET: {exc}")
            failures += 1
        else:
            moved = per_rank_error(mutant)
            caught = max(moved.values()) >= MUTANT_TOL
            print(f"      {control}: {'caught' if caught else 'NOT DETECTED'} rel per rank "
                  f"{', '.join(f'{rank}: {error:.3e}' for rank, error in moved.items())}")
            if not caught:
                failures += 1
    finally:
        native_backend.is_tpu_runtime = saved
    return failures


def check_loader_paths(workdir):
    """A BF16 load with the router bias and `A_log`, a dummy load,
    `--model-layer-nums` and a quantization config.

    Every one runs through the real loaders on the mesh `create_device_mesh`
    returns.
    """
    import ml_dtypes
    from flax import nnx

    from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.model_loader.loader import get_model_loader
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    failures = 0
    mesh = create_device_mesh(ici_parallelism=[1, 8], dcn_parallelism=[1, 1])
    ckpt = os.path.join(workdir, "tiny-loader")
    config, tensors = write_tiny_checkpoint(ckpt)
    bare = os.path.join(workdir, "tiny-config-only")
    os.makedirs(bare, exist_ok=True)
    shutil.copy(os.path.join(ckpt, "config.json"), bare)

    def load(path, load_format=LoadFormat.JAX, abstract=False, **kwargs):
        model_config = ModelConfig(model_path=path, dtype="bfloat16", moe_backend="epmoe",
                                   **kwargs)
        write_runtime_flags(model_config.hf_config, ep_size=2)
        if abstract:
            model_config._abstract_mode = True
        loader = get_model_loader(LoadConfig(load_format=load_format), mesh)
        with jax.set_mesh(mesh):
            return loader.load_model(model_config=model_config)

    def attempt(label, fn):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - reported per case
            return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"

    # The router bias ships F32 and top-k picks on score plus bias, so a BF16
    # load has to keep it F32, bit for bit. Check 10 serves float32, where every
    # parameter is F32 anyway, so only a BF16 load can show a BF16 copy.
    bias_key = "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
    shipped = tensors[bias_key].astype(np.float64)
    model, error = attempt("bf16", lambda: load(ckpt))
    if error is not None:
        print(f"      BF16 load: RAISED {error}")
        failures += 1
    else:
        bias = model.model.layers[1].block_sparse_moe.gate.bias.get_value()
        diff = float(np.abs(np.asarray(bias, np.float64) - shipped).max())
        print(f"      BF16 load, router bias: checkpoint F32, loaded {jnp.dtype(bias.dtype).name}, "
              f"max difference {diff:.3e}")
        if bias.dtype != jnp.float32 or diff != 0.0:
            failures += 1
    # Control: the same values rounded through BF16 have to move, or bit
    # equality couldn't tell a BF16 copy from the F32 one.
    rounded = tensors[bias_key].astype(ml_dtypes.bfloat16).astype(np.float64)
    moved = float(np.abs(rounded - shipped).max())
    print(f"      control (the router bias rounded through BF16): max difference {moved:.3e}")
    if moved == 0.0:
        failures += 1

    # A_log: the load keeps the first num_heads entries of the file's padded
    # vector, one per head, F32 and bit for bit. What the padding holds never
    # reaches the parameter.
    heads = config["text_config"]["linear_attn_config"]["num_heads"]
    a_log_key = "language_model.model.layers.0.self_attn.A_log"
    want_a_log = tensors[a_log_key][:heads].astype(np.float64).reshape(1, 1, heads, 1)

    def loaded_a_log(loaded):
        return np.asarray(loaded.model.layers[0].self_attn.A_log.get_value())

    if model is not None:
        a_log = loaded_a_log(model)
        diff = rel_error(a_log, want_a_log)
        print(f"      BF16 load, layer 0 A_log: {a_log.shape} {a_log.dtype}, the file's first "
              f"{heads} of {tensors[a_log_key].size} entries: rel={diff:.3e}")
        if a_log.dtype != np.float32 or diff != 0.0:
            failures += 1

        def fill_padding(weights):
            for key in weights:
                if key.endswith(".A_log"):
                    filled = weights[key].copy()
                    filled[heads:] = 7.0
                    weights[key] = filled

        padded_ckpt = os.path.join(workdir, "tiny-loader-padding")
        write_tiny_checkpoint(padded_ckpt, perturb=fill_padding)
        padded, error = attempt("padding", lambda: load(padded_ckpt))
        if error is not None:
            print(f"      BF16 load with A_log's padding set to 7.0: RAISED {error}")
            failures += 1
        else:
            same = np.array_equal(loaded_a_log(padded), a_log)
            print(f"      BF16 load with A_log's padding set to 7.0: the parameter "
                  f"{'stays the same' if same else 'MOVES'}")
            if not same:
                failures += 1

    # A dummy load reads no checkpoint and keeps the experts where EPMoE puts
    # them: the loader's own fallback would lay each stack whole on every chip.
    cases = {
        "over the checkpoint": lambda: load(ckpt, LoadFormat.DUMMY),
        "config.json alone": lambda: load(bare, LoadFormat.DUMMY),
        "abstract, as AOT export runs it": lambda: load(bare, LoadFormat.DUMMY, abstract=True),
    }
    for label, fn in cases.items():
        model, error = attempt(label, fn)
        if error is not None:
            print(f"      dummy load, {label}: RAISED {error}")
            failures += 1
            continue
        stack = model.model.layers[1].block_sparse_moe.experts.wi_0.get_value()
        sharding = stack.sharding
        concrete = not isinstance(stack, jax.ShapeDtypeStruct)
        zero = bool(np.all(np.asarray(stack) == 0)) if concrete else True
        good = sharding.spec == P("expert", None, "tensor") and zero
        print(f"      dummy load, {label}: wi_0 {type(stack).__name__} "
              f"{dict(sharding.mesh.shape)} {sharding.spec}"
              f"{', all zero' if concrete and zero else ''}")
        if not good:
            failures += 1

    # --model-layer-nums keeps the first N layers. The rest of the file has to
    # count as accounted for, not as keys nobody maps.
    model, error = attempt("layer nums", lambda: load(ckpt, model_layer_nums=2))
    print(f"      --model-layer-nums 2: "
          f"{'RAISED ' + error if error else str(len(model.model.layers)) + ' layers loaded'}")
    if error or len(model.model.layers) != 2:
        failures += 1
    # Control: a stray key inside a served layer still stops the load.
    stray_ckpt = os.path.join(workdir, "tiny-stray")
    stray = {"language_model.model.layers.1.self_attn.stray.weight": np.zeros((4,), np.float32)}
    write_tiny_checkpoint(stray_ckpt, extra=stray)
    _, error = attempt("stray", lambda: load(stray_ckpt, model_layer_nums=2))
    caught = error is not None and "without a mapping" in error
    print(f"      control (an unmapped key in served layer 1): "
          f"{'caught' if caught else 'NOT DETECTED'} {error or ''}"[:220])
    if not caught:
        failures += 1

    # A quantization config. ModelConfig marks any --quantization-config-path
    # static on this checkpoint, and the model has to refuse that before the
    # loader rewires its linears. A config that isn't static has to reach EPMoE.
    # A built-in config, named the way a launch line names it.
    int8 = "int8.yaml"
    _, error = attempt("static", lambda: load(ckpt, quantization_config_path=int8))
    refused = error is not None and error.startswith("NotImplementedError")
    print(f"      --quantization-config-path int8.yaml: "
          f"{'refused' if refused else 'NOT REFUSED'} {error or ''}"[:220])
    if not refused:
        failures += 1

    from sgl_jax.srt.models import kimi_k3

    model_config = ModelConfig(model_path=ckpt, dtype="bfloat16", moe_backend="epmoe")
    hf = model_config.hf_config
    write_runtime_flags(hf, ep_size=2)
    for label, quantization, want in (
        ("int8.yaml as an online config", QuantizationConfig.from_path(int8), jnp.int8),
        ("the checkpoint's own compressed-tensors dict", hf.quantization_config, None),
    ):
        hf.quantization_config = quantization
        try:
            with jax.set_mesh(mesh):
                model = nnx.eval_shape(lambda: kimi_k3.KimiK3ForCausalLM(hf, mesh, jnp.bfloat16))
            got = model.model.layers[1].block_sparse_moe.experts.quantized_dtype
        except Exception as exc:  # noqa: BLE001
            got = f"RAISED {type(exc).__name__}: {exc}"
        print(f"      EPMoE given {label}: quantized_dtype={got}")
        if got != want:
            failures += 1

    print(f"  [{'FAIL' if failures else 'PASS'}] the loader's other paths")
    return failures


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAILED: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = engine_mesh(2, 4)

    workdir = tempfile.mkdtemp(prefix="kimi-k3-")
    missing = missing_engine_packages()
    try:
        repo = get_checkout(workdir)
        # Read the pins now: the checkout is gone by the time the summary prints.
        install = install_line(repo, missing) if missing else None
        base = git(repo, "rev-parse", "--short", "HEAD~2").stdout.strip()
        print(f"sglang-jax at {base} with the capture and steering patches on it")
        print(f"check 5 mesh: data={mesh.shape['data']} tensor={mesh.shape['tensor']} "
              f"on {devices[0].platform}")
        print(f"published reference: {HF_MODEL}")
        if install is not None:
            print(f"check 10 will fail: the engine needs {install}")
        print()

        print("1. patches apply and compile")
        failures = check_patches(repo, workdir)
        git(repo, "checkout", "--", ".")
        git(repo, "clean", "-qfd")
        sources, before = collect_sources(repo)

        checks = [
            ("2. MXFP4 dequantization", lambda: check_mxfp4(sources)),
            ("3. the situ activation", lambda: check_situ(sources)),
            ("4. attn_res_mix", lambda: check_attn_res_mix(sources)),
            ("5. the block-residual stack", lambda: check_stack(sources, mesh)),
            ("6. the KDA decay layout", lambda: check_decay(repo)),
            ("7. the published config", lambda: check_config(sources)),
            ("8. the published checkpoint resolves", lambda: check_entry(sources, before, repo)),
            ("9. the model the loader builds", lambda: check_loader(repo, sources, workdir)),
            ("10. the served model", lambda: check_served(repo, workdir, sources)),
            ("11. the loader's other paths", lambda: check_loader_paths(workdir)),
        ]
        for title, run in checks:
            print(f"\n{title}")
            try:
                failures += run()
            except Exception:  # noqa: BLE001 - a check that raises fails; the rest still run
                traceback.print_exc()
                print("  [FAIL] the check raised")
                failures += 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        if install is not None:
            # The last lines are what scripts/test_all.sh shows of a failing run.
            print(f"Check 10 needs {install}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
