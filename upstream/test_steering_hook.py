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

"""Correctness gate for `steering-hook.patch`.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 upstream/test_steering_hook.py

Fourteen checks. Every check runs the patched source, not a retyped copy of it.

1. The patch applies to a clean `sglang-jax` at `eb061d8`, every file it touches compiles, and
   it applies both before and after `sglang-jax-877.patch`. The qwen3 hook applies on top.
2. The patched files carry the parts of the hook, read out of their ASTs, and `Engine.generate`
   keeps every parameter it had where it was.
3. The patched `Gemma4Model.__call__` runs, under `jit`, on a sharded input, at float32 and
   bfloat16, against a float64 reference that steers token by token. Per-layer max absolute
   error and Pearson correlation are reported. Five mutants of the same patched source must
   fail.
4. The shipped `layers/steering.py` arithmetic matches a float64 reference on a batch that mixes
   static steering, conditional steering, a feature the bank doesn't hold, and tokens nobody
   steers. Six mutants must fail, and the same read at bfloat16 must move the fired set. A bank
   loads at the site it was fit on and is refused at any other, and a malformed bank is refused.
   A bank that holds a dead latent loads with a warning that names it.
5. The scheduler's per-token rules resolve positions and thresholds the way the request field
   documents. Each with a control.
6. Every key `_merge_steering` returns is a field of the real `ModelWorkerBatch`, against the
   dataclass field list read out of the patched source. The spread is `**_steering`, so a key
   with no field raises `TypeError` on every forward pass, steered or not.
7. `ForwardBatch.init_new` builds a steering batch whether or not a request asked for one, so
   the steered call and the unsteered call share a trace. The all-off arrays are placed once
   per token count, and the hook skips the bank gathers for them. Controls: the same two calls
   with `None` in place of the all-off batch take two traces, and a hook with no branch gathers
   on every batch.
8. The request validator rejects each malformed field, an unknown key and a threshold beside
   static mode. Control: for each, what the scheduler thread makes of the value if it gets
   through, which is a raise on the thread that serves every request, an `inf`, or an answer
   the request didn't ask for. Each control has to show its failure.
9. `apply_steering` traces under an explicit-sharding mesh, which is what the engine runs, not
   the auto mesh checks 3, 4, 7 and 14 use. Control: the same gather without `out_sharding` raises
   `ShardingTypeError` there.
10. The cache key a steered request gets names every field that decides what the hook does.
11. The real Engine, on CPU, with a tiny random Qwen3 and the radix cache on: a steered request
    and an unsteered one with the same prompt never share cached KV, two requests that steer
    the same way do, a client `extra_key` that names a steered namespace is refused, and the
    steering dict is logged where it crosses between components.
12. The real Engine refuses to start with `--enable-steering` and a speculative algorithm, and a
    server started without `--enable-steering` refuses a steered request.
13. On a mesh that spans two processes, `ForwardBatch.init_new` places the per-token arrays
    with no cross-process gather. Control: `jax.device_put` on the same arrays gathers.
14. The patched `QWen3Model.__call__` steers after the deepstack add, so a conditional probe
    reads the stream the next capture slot holds. Control: the hook moved above the add.

Checks 11 and 12 start the Engine, so they need the packages `sgl_jax` imports at startup, which
`models/requirements.txt` lists.

Every check carries negative controls. A control that passes fails the run. So does a mutation
that finds nothing to change, because a control that changed nothing tested nothing.

Point `SGLANG_JAX_REPO` at a checkout that holds `eb061d8` to skip the fetch.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "capture-hooks"))
from test_capture_hooks import (  # noqa: E402
    SGL_COMMIT,
    Stub,
    corrupt,
    find_class,
    find_method,
    get_checkout,
    git,
    is_self_attr,
    strip_annotations,
)

PATCH = os.path.join(HERE, "steering-hook.patch")
PATCH_877 = os.path.join(HERE, "sglang-jax-877.patch")
PATCH_QWEN3 = os.path.join(HERE, "qwen3-steering-hook.patch")

SRT = "python/sgl_jax/srt"
STEERING = f"{SRT}/layers/steering.py"
GEMMA4 = f"{SRT}/models/gemma4.py"
FORWARD_BATCH = f"{SRT}/model_executor/forward_batch_info.py"
MODEL_RUNNER = f"{SRT}/model_executor/model_runner.py"
SCHEDULE_BATCH = f"{SRT}/managers/schedule_batch.py"
IO_STRUCT = f"{SRT}/managers/io_struct.py"
SCHEDULER = f"{SRT}/managers/scheduler.py"
SERVER_ARGS = f"{SRT}/server_args.py"
TOKENIZER_MANAGER = f"{SRT}/managers/tokenizer_manager.py"
ENGINE = f"{SRT}/entrypoints/engine.py"
# qwen3-steering-hook.patch's one file.
QWEN3 = f"{SRT}/models/qwen3.py"
# The patch doesn't touch this file. ForwardBatch.init_new places arrays with its device_array.
JAX_UTILS = f"{SRT}/utils/jax_utils.py"
# The tiny random model and the Engine settings the engine checks share with the 877 patch's
# CPU test, which lives in the stacked tree.
CPU_TEST = "test/srt/test_return_hidden_states_cpu.py"

TOUCHED = [
    STEERING,
    GEMMA4,
    FORWARD_BATCH,
    MODEL_RUNNER,
    SCHEDULE_BATCH,
    IO_STRUCT,
    SCHEDULER,
    SERVER_ARGS,
    TOKENIZER_MANAGER,
    ENGINE,
]

AXIS = "data"
TOKENS = 64
DIM = 32
LAYERS = 6
STEER_LAYER = 2
BANK = 3
EPS = 1e-6

# The capture slot a bank in this file was fit on. Capture appends before `layer(...)` runs and
# the hook fires after it returns, so the flag that writes the same tensor is one less.
CAPTURE_SLOT = 20

# gemma4 defaults to bfloat16, so every run reports both. float32 is the gate, because it
# separates a defect from rounding.
F32_TOL = 1e-5
BF16_TOL = 5e-2
# Mutants run at float32, where the clean signal is around 1e-07.
MUTANT_TOL = 1e-2


# --------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------


def compile_function(node, namespace):
    """Compile one AST function into a callable, with the shipped body as written."""
    node = strip_annotations(copy.deepcopy(node))
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"<patched:{node.name}>", "exec"), namespace)  # noqa: S102
    return namespace[node.name]


def find_function(src, name):
    """A module-level `def name(...)`, by AST."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return copy.deepcopy(node)
    raise LookupError(f"{name} not found")


def mutated(node, mutate):
    """A deep copy of an AST node with `mutate` applied to it.

    An operator that can't find its target either raises or leaves the tree as it was, and the
    control would then run the unmutated source and report what that does. A copy that comes
    back unchanged raises LookupError, so every caller can count it as the failure it is.
    """
    before = ast.dump(node)
    node = copy.deepcopy(node)
    mutate(node)
    if ast.dump(node) == before:
        raise LookupError("the mutation changed nothing")
    return node


def load_module(src, name, mutate=None):
    """Exec a whole module's source into a fresh namespace, optionally mutating one function.

    A mutation that finds no target raises before anything runs. See `mutated`.
    """
    tree = ast.parse(src)
    if mutate is not None:
        tree = mutated(tree, mutate)
    ast.fix_missing_locations(tree)
    # A real module in sys.modules, because a dataclass resolves its string annotations through
    # sys.modules[cls.__module__].
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(tree, f"<patched:{name}>", "exec"), module.__dict__)  # noqa: S102
    return module.__dict__


def module_constants(src, *names):
    """The module-level `NAME = <literal>` assignments of a source, evaluated, by name.

    Holds only the names the source defines. A function that reads one the source lacks raises
    NameError when it runs.
    """
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                out[target.id] = ast.literal_eval(node.value)
    return out


def worse(worst, value):
    """The larger of two errors, with NaN as the worst error there is.

    `max(worst, nan)` keeps `worst`, so a fold that uses it drops a NaN layer and passes.
    """
    return float("inf") if np.isnan(value) else max(worst, value)


def collect_sources(repo):
    """The before and after text of every file the two steering patches touch, and jax_utils.py."""
    before = {}
    for rel in (*TOUCHED, QWEN3):
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            # layers/steering.py is new, so there's nothing to read before the patch.
            before[rel] = ""
            continue
        with open(path, encoding="utf-8") as fh:
            before[rel] = fh.read()
    with open(os.path.join(repo, JAX_UTILS), encoding="utf-8") as fh:
        jax_utils = fh.read()

    for patch in (PATCH, PATCH_QWEN3):
        applied = git(repo, "apply", patch)
        if applied.returncode != 0:
            raise RuntimeError(
                f"{os.path.basename(patch)} no longer applies: {applied.stderr.strip()}"
            )
    after = {}
    for rel in (*TOUCHED, QWEN3):
        with open(os.path.join(repo, rel), encoding="utf-8") as fh:
            after[rel] = fh.read()
    git(repo, "checkout", "--", ".")
    git(repo, "clean", "-qfd")
    return {"before": before, "after": after, "jax_utils": jax_utils}


def pearson(got, want):
    got = np.asarray(got, np.float64).reshape(-1)
    want = np.asarray(want, np.float64).reshape(-1)
    got = got - got.mean()
    want = want - want.mean()
    denom = np.linalg.norm(got) * np.linalg.norm(want)
    return float(got @ want / denom) if denom > 0 else 0.0


def max_abs(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape:
        return float("inf")
    return float(np.abs(got - want).max())


def relative(got, want):
    """Max absolute error against the largest value in the reference.

    The residual stream grows layer over layer, so an absolute tolerance would tighten on the
    first layer and loosen on the last. This is the number the tolerances gate on.
    """
    want = np.asarray(want, np.float64)
    return max_abs(got, want) / max(np.abs(want).max(), 1e-12)


# --------------------------------------------------------------------------------------
# 1. the patch applies
# --------------------------------------------------------------------------------------


def check_patch(repo, workdir):
    """Applies at `SGL_COMMIT`, compiles, stacks with 877, and refuses a corrupted copy."""
    failures = 0
    dry = git(repo, "apply", "--check", PATCH)
    ok = dry.returncode == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] steering-hook.patch applies at {SGL_COMMIT[:7]}")
    if not ok:
        print(f"      {dry.stderr.strip()}")
        return failures + 1

    git(repo, "apply", PATCH)
    for rel in TOUCHED:
        compiled = subprocess.run(
            [sys.executable, "-m", "py_compile", os.path.join(repo, rel)],
            capture_output=True,
            text=True,
        )
        good = compiled.returncode == 0
        print(f"      py_compile {os.path.basename(rel):<24} {'ok' if good else 'FAILED'}")
        if not good:
            print(f"      {compiled.stderr.strip()}")
            failures += 1

    # The qwen3 hook imports layers/steering.py, so it goes on top of this patch.
    qwen3 = git(repo, "apply", PATCH_QWEN3)
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", os.path.join(repo, QWEN3)],
        capture_output=True,
        text=True,
    )
    good = qwen3.returncode == 0 and compiled.returncode == 0
    print(
        f"      qwen3-steering-hook.patch on top, qwen3.py compiles:"
        f" {'ok' if good else 'FAILED'}"
    )
    if not good:
        print(f"      {qwen3.stderr.strip()} {compiled.stderr.strip()}")
        failures += 1

    # The flag and the reshape come from 877, and a run that steers and reads the per-layer
    # stream applies both. Neither patch may move a line the other reads.
    stacked = git(repo, "apply", "--check", PATCH_877)
    print(f"      877 applies on top: {'ok' if stacked.returncode == 0 else 'FAILED'}")
    failures += stacked.returncode != 0
    git(repo, "checkout", "--", ".")
    git(repo, "clean", "-qfd")

    git(repo, "apply", PATCH_877)
    reversed_order = git(repo, "apply", "--check", PATCH)
    print(f"      applies on top of 877: {'ok' if reversed_order.returncode == 0 else 'FAILED'}")
    failures += reversed_order.returncode != 0
    git(repo, "checkout", "--", ".")
    git(repo, "clean", "-qfd")

    with open(PATCH, encoding="utf-8") as fh:
        bad_text = corrupt(fh.read())
    bad_path = os.path.join(workdir, "corrupt-steering-hook.patch")
    with open(bad_path, "w", encoding="utf-8") as fh:
        fh.write(bad_text)
    bad = git(repo, "apply", "--check", bad_path)
    refused = bad.returncode != 0
    print(f"      control (one context line rewritten): {'refused' if refused else 'ACCEPTED'}")
    if not refused:
        print("      FAIL: the control didn't fail, so --check reads nothing.")
        failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 2. the parts of the hook
# --------------------------------------------------------------------------------------


def assigns_self_attr(node, attr):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(is_self_attr(t, attr) for t in sub.targets):
            return True
    return False


def calls(node, name):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == name:
            return True
    return False


def layer_loop(node):
    """The `for layer_id, layer in enumerate(self.layers)` loop."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.For):
            continue
        iter_ = sub.iter
        if not (isinstance(iter_, ast.Call) and isinstance(iter_.func, ast.Name)):
            continue
        if iter_.func.id == "enumerate" and is_self_attr(iter_.args[0], "layers"):
            return sub
    return None


def steer_gate_index(loop):
    """Index in `loop.body` of `if layer_id == self.steering_layer:`."""
    index_name = loop.target.elts[0].id
    for i, stmt in enumerate(loop.body):
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        if not (isinstance(test.left, ast.Name) and test.left.id == index_name):
            continue
        if is_self_attr(test.comparators[0], "steering_layer"):
            return i
    return None


def layer_call_index(loop):
    for i, stmt in enumerate(loop.body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == "layer":
                    return i
    return None


def flatten_index(src, attr):
    """Position of `self.<attr>` in the tuple `tree_flatten` builds."""
    fn = find_method(src, "ForwardBatch", "tree_flatten")
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Tuple) and any(is_self_attr(e, attr) for e in sub.elts):
            return [i for i, e in enumerate(sub.elts) if is_self_attr(e, attr)][0]
    return None


def unflatten_index(src, attr):
    """The `children[i]` that `tree_unflatten` assigns to `obj.<attr>`."""
    fn = find_method(src, "ForwardBatch", "tree_unflatten")
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Assign):
            continue
        target = sub.targets[0]
        if not (isinstance(target, ast.Attribute) and target.attr == attr):
            continue
        value = sub.value
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            if value.value.id == "children" and isinstance(value.slice, ast.Constant):
                return value.slice.value
    return None


def setup_under_draft_gate(src):
    """True when `_setup_steering()` runs only for a worker that isn't the draft model.

    Three speculative-decoding workers build a ModelRunner with `is_draft_worker=True` and the
    target's server_args. The draft model has no steering hook, so a `_setup_steering()` outside
    that gate raises and the server refuses to start whenever --enable-steering meets
    speculative decoding.
    """
    runner = find_class(ast.parse(src), "ModelRunner")
    if runner is None:
        return False
    called = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call) and is_self_attr(node.func, "_setup_steering")
    ]
    if not called:
        return False
    for node in ast.walk(runner):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        if not is_self_attr(test.operand, "is_draft_worker"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and is_self_attr(sub.func, "_setup_steering"):
                return True
    return False


def dataclass_fields(src, name):
    """The annotated field names of a dataclass, in declaration order.

    Deduplicated: `ModelWorkerBatch` upstream declares `capture_hidden_mode` twice, which
    Python folds into one field.
    """
    node = find_class(ast.parse(src), name)
    if node is None:
        return []
    return list(
        dict.fromkeys(
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        )
    )


def steering_block(src):
    """The statements `ForwardBatch.init_new` runs from `steering = None` to `obj = cls(...)`.

    Empty when init_new builds no steering batch, which is the unpatched tree.
    """
    fn = find_method(src, "ForwardBatch", "init_new")

    def index_of(name):
        for i, stmt in enumerate(fn.body):
            if isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], "id", None) == name:
                return i
        return None

    start, end = index_of("steering"), index_of("obj")
    if start is None or end is None:
        return []
    return fn.body[start:end]


def names_called(nodes):
    return {
        sub.func.id if isinstance(sub.func, ast.Name) else sub.func.attr
        for node in nodes
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, (ast.Name, ast.Attribute))
    }


def namespaces_steering(src):
    """True when the tokenizer manager appends `steering_cache_key(...)` to an `extra_key`."""
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Attribute) and t.attr == "extra_key" for t in node.targets):
            continue
        if "steering_cache_key" in names_called([node.value]):
            return True
    return False


def refuses_speculative_steering(src):
    """True when check_server_args raises on --enable-steering with a speculative algorithm."""
    try:
        fn = find_method(src, "ServerArgs", "check_server_args")
    except LookupError:
        return False
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "enable_steering" in test and "speculative_algorithm" in test:
            if any(isinstance(sub, ast.Raise) for stmt in node.body for sub in ast.walk(stmt)):
                return True
    return False


def inspect_sources(src):
    """Which parts of the hook the text carries."""
    gemma = find_class(ast.parse(src[GEMMA4]), "Gemma4Model")
    call = find_method(src[GEMMA4], "Gemma4Model", "__call__") if gemma else None
    loop = layer_loop(call) if call else None
    gate = steer_gate_index(loop) if loop else None
    call_at = layer_call_index(loop) if loop else None

    flat = flatten_index(src[FORWARD_BATCH], "steering")
    unflat = unflatten_index(src[FORWARD_BATCH], "steering")
    worker_fields = dataclass_fields(src[SCHEDULE_BATCH], "ModelWorkerBatch")

    parts = {
        "module": "def apply_steering" in src[STEERING]
        and "def conditional_steer" in src[STEERING]
        and "def static_steer" in src[STEERING],
        "site": gemma is not None and assigns_self_attr(gemma, "steering_layer"),
        "gate": gate is not None,
        "below": gate is not None and call_at is not None and gate > call_at,
        "hook": gate is not None and calls(loop.body[gate], "apply_steering"),
        "child": flat is not None,
        "unflatten": flat is not None and flat == unflat,
        "runner": "_setup_steering" in src[MODEL_RUNNER] and "steering_layer" in src[MODEL_RUNNER],
        "draft": setup_under_draft_gate(src[MODEL_RUNNER]),
        "flag": "--enable-steering" in src[SERVER_ARGS]
        and "--steering-bank" in src[SERVER_ARGS]
        and "--steering-layer" in src[SERVER_ARGS],
        "request": "steering" in src[IO_STRUCT] and "_validate_steering" in src[IO_STRUCT],
        "threaded": "steering=" in src[SCHEDULER],
        "per_token": "_merge_steering" in src[SCHEDULE_BATCH],
        "worker": all(
            key in worker_fields
            for key in ("steering_feature", "steering_alpha", "steering_threshold")
        ),
        "namespace": namespaces_steering(src[TOKENIZER_MANAGER]),
        "placement": "device_array" in names_called(steering_block(src[FORWARD_BATCH])),
        "no_spec": refuses_speculative_steering(src[SERVER_ARGS]),
    }
    return parts


def engine_params(src, method):
    """`Engine.<method>`'s parameters: the ones a caller can pass by position, then all of them."""
    engine = find_class(ast.parse(src), "Engine")
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    fns = [n for n in (engine.body if engine else []) if isinstance(n, defs) and n.name == method]
    if not fns:
        raise LookupError(f"Engine.{method} not found")
    fn = fns[0]
    positional = [arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args)]
    return positional, positional + [arg.arg for arg in fn.args.kwonlyargs]


def keeps_positions(before, after):
    """True when every parameter a caller could pass by position before still sits where it did."""
    return after[: len(before)] == before


def check_signature(sources):
    """`steering` joins Engine.generate without moving a parameter a caller passes by position."""
    failures = 0
    for method in ("generate", "async_generate"):
        before, _ = engine_params(sources["before"][ENGINE], method)
        after, every = engine_params(sources["after"][ENGINE], method)
        ok = keeps_positions(before, after) and "steering" in every
        print(
            f"  [{'PASS' if ok else 'FAIL'}] Engine.{method} takes steering and keeps"
            f" {len(before)} positional parameter(s) in place"
        )
        failures += not ok

        # Control: steering inserted ahead of return_routed_experts, as the patch first had it,
        # has to read as a moved parameter.
        moved = [name for name in after if name != "steering"]
        moved.insert(moved.index("return_routed_experts"), "steering")
        detected = not keeps_positions(before, moved)
        print(
            f"      control (steering ahead of return_routed_experts):"
            f" {'detected' if detected else 'NOT DETECTED'}"
        )
        if not detected:
            print("      FAIL: the control didn't fail, so this check detects nothing.")
            failures += 1
    return failures


def check_structure(sources):
    failures = 0
    after = inspect_sources(sources["after"])
    ok = all(after.values())
    for name, value in after.items():
        print(f"      {name:<10} {'y' if value else 'n'}")
    print(f"  [{'PASS' if ok else 'FAIL'}] the patched text carries every part")
    failures += not ok

    before = inspect_sources(sources["before"])
    stale = [name for name, value in before.items() if value]
    detected = not stale
    print(
        f"      control (unpatched text): "
        f"{'no hook present' if detected else 'ALREADY HAS ' + ' '.join(stale)}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so this check detects nothing.")
        failures += 1
    return failures + check_signature(sources)


# --------------------------------------------------------------------------------------
# 3. the patched model runs
# --------------------------------------------------------------------------------------


class ToyLayer:
    """A gemma4 decoder layer's shape: one residual stream in, one out, four return values."""

    def __init__(self, weights, dtype):
        w_attn, w_mlp, gain = weights
        self.w_attn = jnp.asarray(w_attn, dtype)
        self.w_mlp = jnp.asarray(w_mlp, dtype)
        self.gain = jnp.asarray(gain, jnp.float32)

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool, **kw):
        residual = hidden_states
        normed = rms_norm(hidden_states, self.gain)
        hidden_states = residual + jnp.asarray(normed @ self.w_attn, self.w_attn.dtype)
        residual = hidden_states
        normed = rms_norm(hidden_states, self.gain)
        outputs = residual + jnp.tanh(jnp.asarray(normed @ self.w_mlp, self.w_mlp.dtype))
        return outputs, jnp.zeros((1,), jnp.int32), [jnp.zeros((1,), jnp.int32)], None


def rms_norm(x, gain):
    """GemmaRMSNorm: reduce in float32, return the input dtype."""
    orig = x.dtype
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + EPS) * gain).astype(orig)


def reference_stack(embed64, weights64, steering, steer_layer, capture):
    """The same stack in float64, with the steering written out token by token.

    The loop over tokens builds no mask and gathers nothing, so it checks the masked select
    rather than restating it.
    """
    stream = embed64 * np.sqrt(DIM)
    out = []
    for i, (w_attn, w_mlp, gain) in enumerate(weights64):
        if i in capture:
            out.append(stream.copy())
        normed = stream / np.sqrt(np.mean(stream**2, axis=-1, keepdims=True) + EPS) * gain
        stream = stream + normed @ w_attn
        normed = stream / np.sqrt(np.mean(stream**2, axis=-1, keepdims=True) + EPS) * gain
        stream = stream + np.tanh(normed @ w_mlp)
        if i == steer_layer and steering is not None:
            stream = reference_steer(stream, steering)
    return np.stack(out), stream


def reference_steer(stream, steering):
    """`h -> h + alpha * v` on the tokens that pass, one token at a time, in float64."""
    vectors = np.asarray(steering.vectors, np.float64)
    probes = np.asarray(steering.probes, np.float64)
    slot = np.asarray(steering.slot)
    alpha = np.asarray(steering.alpha, np.float64)
    threshold = np.asarray(steering.threshold, np.float64)
    out = stream.copy()
    for token in range(stream.shape[0]):
        probe = probes[slot[token]]
        unit = probe / max(np.linalg.norm(probe), 1e-12)
        if float(stream[token] @ unit) > threshold[token]:
            out[token] = stream[token] + alpha[token] * vectors[slot[token]]
    return out


def build_steering(module, mesh, rng, dtype=np.float32):
    """A batch that mixes every case: static, conditional, unknown feature, and no steering."""
    vectors = rng.normal(size=(BANK, DIM))
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    probes = rng.normal(size=(BANK, DIM))
    features = np.array([7, 11, 40977], np.int32)
    thresholds = np.array([0.5, -np.inf, 0.0], np.float32)

    feature = np.full(TOKENS, -1, np.int32)
    alpha = np.zeros(TOKENS, np.float32)
    threshold = np.full(TOKENS, -np.inf, np.float32)

    feature[0:16] = 11  # static, every token
    alpha[0:16] = 2.0
    feature[16:32] = 7  # conditional, the bank's fitted threshold
    alpha[16:32] = 1.5
    threshold[16:32] = np.nan
    feature[32:40] = 40977  # conditional, a threshold the request set
    alpha[32:40] = 3.0
    threshold[32:40] = 0.25
    feature[40:44] = 12345  # not in the bank
    alpha[40:44] = 5.0
    # 44 onward steers nothing.

    bank = module["SteeringBank"](
        features=features,
        thresholds=thresholds,
        vectors=jax.device_put(jnp.asarray(vectors, jnp.float32), NamedSharding(mesh, P())),
        probes=jax.device_put(jnp.asarray(probes, jnp.float32), NamedSharding(mesh, P())),
    )
    return bank, feature, alpha, threshold


def steering_batch(bank, feature, alpha, threshold, token_sharding):
    """What `ForwardBatch.init_new` does with the scheduler's arrays, through the shipped code.

    The bank resolves them on the host, `device_array` from `utils/jax_utils.py` places them on
    the token axis, and the bank pairs them with its vectors and probes.
    """
    slot, alpha, threshold = bank.resolve(feature, alpha, threshold)
    slot, alpha, threshold = DEVICE_ARRAY((slot, alpha, threshold), sharding=token_sharding)
    return bank.batch(slot, alpha, threshold)


def run_model(sources, mesh, layers, embed, steering, capture, fn=None):
    """Run the patched `Gemma4Model.__call__`, or `fn` in its place, on stubs, under jit."""
    namespace = {
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "apply_steering": STEERING_MODULE["apply_steering"],
        "precision_tracer": Stub(
            jit_pure_callback_record=lambda *a, **k: jnp.zeros((1,), jnp.int32)
        ),
    }
    if fn is None:
        fn = find_method(sources["after"][GEMMA4], "Gemma4Model", "__call__")
    call = compile_function(fn, namespace)

    model = Stub(
        layers=layers,
        layers_to_capture=list(capture),
        steering_layer=STEER_LAYER,
        hidden_size=DIM,
        embed_tokens=lambda ids: embed,
        norm=Stub(_call=lambda s, x: rms_norm(x, jnp.ones((DIM,), jnp.float32))),
    )
    forward_batch = Stub(
        input_ids=jnp.zeros((TOKENS,), jnp.int32),
        positions=jnp.arange(TOKENS, dtype=jnp.int32),
        input_embedding=None,
        forward_mode=Stub(is_extend_or_draft_extend_or_mixed=lambda: False),
        steering=steering,
    )
    return jax.jit(lambda: call(model, forward_batch, "kv-pool"))()


def check_model(sources, mesh):
    failures = 0
    rng = np.random.default_rng(11)
    weights64 = [
        (
            rng.normal(size=(DIM, DIM)) * 0.1,
            rng.normal(size=(DIM, DIM)) * 0.1,
            1.0 + rng.normal(size=(DIM,)) * 0.05,
        )
        for _ in range(LAYERS)
    ]
    embed64 = rng.normal(size=(TOKENS, DIM))
    capture = list(range(LAYERS))

    bank, feature, alpha, threshold = build_steering(STEERING_MODULE, mesh, rng)
    token_sharding = NamedSharding(mesh, P(AXIS))
    steering = steering_batch(bank, feature, alpha, threshold, token_sharding)

    for dtype, tol, label in ((jnp.float32, F32_TOL, "f32"), (jnp.bfloat16, BF16_TOL, "bf16")):
        layers = [ToyLayer(w, dtype) for w in weights64]
        embed = jax.device_put(jnp.asarray(embed64, dtype), NamedSharding(mesh, P(AXIS, None)))
        result = run_model(sources, mesh, layers, embed, steering, capture)
        got = np.stack([np.asarray(h, np.float64) for h in result[1]])
        want, _ = reference_stack(embed64, weights64, steering, STEER_LAYER, capture)

        worst = 0.0
        for layer in range(LAYERS):
            err = max_abs(got[layer], want[layer])
            rel = relative(got[layer], want[layer])
            corr = pearson(got[layer], want[layer])
            worst = worse(worst, rel)
            mark = "steered" if layer > STEER_LAYER else "clean"
            print(
                f"      {label} layer {layer}: max_abs={err:.3e} rel={rel:.3e}"
                f" pearson={corr:.9f}  {mark}"
            )
        ok = worst < tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: worst layer rel={worst:.3e}")
        failures += not ok

    # The stream leaves the hook site in float32 even under a bfloat16 model.
    layers = [ToyLayer(w, jnp.bfloat16) for w in weights64]
    embed = jax.device_put(jnp.asarray(embed64, jnp.bfloat16), NamedSharding(mesh, P(AXIS, None)))
    result = run_model(sources, mesh, layers, embed, steering, capture)
    widened = [jnp.dtype(h.dtype).name for h in result[1]]
    ok = widened[: STEER_LAYER + 1] == ["bfloat16"] * (STEER_LAYER + 1) and set(
        widened[STEER_LAYER + 1 :]
    ) == {"float32"}
    print(f"  [{'PASS' if ok else 'FAIL'}] the stream widens at the hook and stays wide: {widened}")
    failures += not ok

    # Control: no steering at all leaves the stream alone and narrow.
    quiet = run_model(sources, mesh, layers, embed, None, capture)
    want, _ = reference_stack(embed64, weights64, None, STEER_LAYER, capture)
    got = np.stack([np.asarray(h, np.float64) for h in quiet[1]])
    rel = relative(got, want)
    narrow = {jnp.dtype(h.dtype).name for h in quiet[1]} == {"bfloat16"}
    print(
        f"      control (forward_batch.steering is None): rel={rel:.3e},"
        f" stream stays bfloat16={narrow}"
    )
    failures += not (rel < BF16_TOL and narrow)

    failures += check_model_mutants(sources, mesh, weights64, embed64, steering, capture)
    return failures


def m_steer_above(fn):
    """The hook moved above the layer call, so it lands on the previous layer's stream."""
    loop = layer_loop(fn)
    gate = loop.body.pop(steer_gate_index(loop))
    loop.body.insert(layer_call_index(loop), gate)


def m_gate_inverted(fn):
    loop = layer_loop(fn)
    loop.body[steer_gate_index(loop)].test.ops[0] = ast.NotEq()


def m_gate_removed(fn):
    """Every layer steers."""
    loop = layer_loop(fn)
    index = steer_gate_index(loop)
    gate = loop.body[index]
    loop.body[index : index + 1] = gate.body


def m_operand_dropped(fn):
    """`apply_steering(hidden_states, None)`, which is the hook wired to nothing."""
    loop = layer_loop(fn)
    gate = loop.body[steer_gate_index(loop)]
    for sub in ast.walk(gate):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "apply_steering":
            sub.args[1] = ast.Constant(value=None)


def m_hook_removed(fn):
    loop = layer_loop(fn)
    loop.body[steer_gate_index(loop)].body = [ast.Pass()]


MODEL_MUTANTS = {
    "hook moved above the layer call": m_steer_above,
    "gate inverted to !=": m_gate_inverted,
    "gate removed, every layer steers": m_gate_removed,
    "steering operand replaced by None": m_operand_dropped,
    "hook removed, gate kept": m_hook_removed,
}


def operator_missed(label, exc):
    """Report a mutation operator that found nothing to change. Returns 1, a failure."""
    print(
        f"      control ({label}): MUTATION OPERATOR FOUND NO TARGET,"
        f" {type(exc).__name__}: {exc}"
    )
    print("      FAIL: the mutation changed nothing, so this control tests nothing.")
    return 1


def check_model_mutants(sources, mesh, weights64, embed64, steering, capture):
    missed = 0
    want, _ = reference_stack(embed64, weights64, steering, STEER_LAYER, capture)
    layers = [ToyLayer(w, jnp.float32) for w in weights64]
    embed = jax.device_put(jnp.asarray(embed64, jnp.float32), NamedSharding(mesh, P(AXIS, None)))
    call = find_method(sources["after"][GEMMA4], "Gemma4Model", "__call__")
    for label, mutate in MODEL_MUTANTS.items():
        try:
            fn = mutated(call, mutate)
        except Exception as exc:  # noqa: BLE001 - an operator that can't apply tested nothing
            missed += operator_missed(label, exc)
            continue
        try:
            result = run_model(sources, mesh, layers, embed, steering, capture, fn=fn)
            got = np.stack([np.asarray(h, np.float64) for h in result[1]])
            rel = relative(got, want)
            # NaN is a broken model too, so only an error under the tolerance is a miss.
            caught = not rel < MUTANT_TOL
            how = f"rel={rel:.3e}" if caught else f"NOT DETECTED (rel={rel:.3e})"
        except Exception as exc:  # noqa: BLE001 - a mutant that raises is caught
            caught, how = True, f"raised {type(exc).__name__}"
        print(f"      control ({label}): {how}")
        if not caught:
            print("      FAIL: the control didn't fail, so this mutation is untested.")
            missed += 1
    return missed


# --------------------------------------------------------------------------------------
# 4. the shipped arithmetic
# --------------------------------------------------------------------------------------


def check_arithmetic(sources, mesh):
    failures = 0
    rng = np.random.default_rng(3)
    bank, feature, alpha, threshold = build_steering(STEERING_MODULE, mesh, rng)
    token_sharding = NamedSharding(mesh, P(AXIS))
    steering = steering_batch(bank, feature, alpha, threshold, token_sharding)

    # The host side: a request that named a feature the bank doesn't hold steers nothing, and
    # NaN picks up that feature's fitted threshold.
    slot = np.asarray(steering.slot)
    got_alpha = np.asarray(steering.alpha)
    got_threshold = np.asarray(steering.threshold)
    resolved = (
        np.array_equal(slot[0:16], np.ones(16, np.int32))
        and np.array_equal(got_threshold[0:16], np.full(16, -np.inf, np.float32))
        and np.array_equal(got_threshold[16:32], np.full(16, 0.5, np.float32))
        and np.array_equal(got_threshold[32:40], np.full(8, 0.25, np.float32))
        and np.array_equal(got_alpha[40:44], np.zeros(4, np.float32))
        and np.array_equal(got_threshold[40:44], np.full(4, np.inf, np.float32))
        and np.array_equal(got_alpha[44:], np.zeros(TOKENS - 44, np.float32))
    )
    print(
        f"  [{'PASS' if resolved else 'FAIL'}] the bank resolves features, fitted thresholds"
        f" and the unknown feature"
    )
    failures += not resolved

    # The device side, against the float64 reference.
    h64 = rng.normal(size=(TOKENS, DIM))
    for dtype, tol, label in ((jnp.float32, F32_TOL, "f32"), (jnp.bfloat16, BF16_TOL, "bf16")):
        h = jax.device_put(jnp.asarray(h64, dtype), NamedSharding(mesh, P(AXIS, None)))
        got = jax.jit(STEERING_MODULE["apply_steering"])(h, steering)
        want = reference_steer(np.asarray(h, np.float64), steering)
        err = max_abs(got, want)
        corr = pearson(got, want)
        ok = err < tol
        print(
            f"  [{'PASS' if ok else 'FAIL'}] apply_steering at {label}: max_abs={err:.3e}"
            f" pearson={corr:.9f}"
        )
        failures += not ok

    # The tokens that moved are the ones the predicate picked, and nothing else moved. An
    # unfired row comes back as the float32 widening of the input, bit for bit, so this reads
    # the array that went in rather than the float64 it was rounded from.
    h = jax.device_put(jnp.asarray(h64, jnp.float32), NamedSharding(mesh, P(AXIS, None)))
    got = np.asarray(jax.jit(STEERING_MODULE["apply_steering"])(h, steering), np.float64)
    moved = np.abs(got - np.asarray(h, np.float64)).max(axis=1) > 0
    fire = np.asarray(
        STEERING_MODULE["threshold_fire"](h, steering.probes[steering.slot], steering.threshold)
    )
    want_moved = fire & (np.asarray(steering.alpha) != 0)
    ok = np.array_equal(moved, want_moved) and moved[0:16].all() and not moved[44:].any()
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {int(moved.sum())} token(s) moved, and they are the"
        f" ones the predicate picked"
    )
    failures += not ok

    failures += check_bank_format(mesh)
    failures += check_module_mutants(sources, mesh, h64, steering)
    return failures


class Heard(logging.Handler):
    """Keeps the text of every warning a logger passes it."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def load_heard(path, mesh, width, steering_layer):
    """The shipped `SteeringBank.load`, plus the text of every warning it logged."""
    heard = Heard()
    logger = STEERING_MODULE["logger"]
    logger.addHandler(heard)
    try:
        bank = STEERING_MODULE["SteeringBank"].load(
            path, mesh, width, steering_layer=steering_layer
        )
    finally:
        logger.removeHandler(heard)
    return bank, heard.messages


def check_bank_format(mesh):
    """The file `steering/from_sae.py` writes is the file the server reads.

    Builds a bank out of a hand-made SAE, writes it with `save_bank`, and loads it with the
    shipped `SteeringBank.load`. The two sides agree on the keys, the dtypes and the order, or
    the server steers with the wrong row. The last feature never fires in calibration, so its
    threshold is `+inf`, and load has to name it.
    """
    sys.path.insert(0, os.path.join(HERE, os.pardir, "steering"))
    import from_sae as convert  # noqa: E402

    rng = np.random.default_rng(31)
    d_sae = 8
    w_dec = rng.normal(size=(d_sae, DIM)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=-1, keepdims=True)
    params = convert.SAEParams(
        w_enc=jnp.asarray(rng.normal(size=(DIM, d_sae)), jnp.float32),
        b_enc=jnp.asarray(rng.normal(size=(d_sae,)), jnp.float32),
        w_dec=jnp.asarray(w_dec),
        b_dec=jnp.zeros((DIM,), jnp.float32),
    )
    theta = jnp.asarray(np.r_[np.full(d_sae - 1, 0.4, np.float32), [np.inf]])
    features = [1, 5, d_sae - 1]
    bank = convert.build_bank(params, theta, features)

    # The site travels with the bank. `sae/train.py` records the capture slot, `from_sae.py`
    # converts it to the --steering-layer that writes the same tensor, and the two are one apart.
    site = {"capture_layer": CAPTURE_SLOT, "steering_layer": CAPTURE_SLOT - 1}
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "bank.npz")
        convert.save_bank(path, bank, meta=site)
        loaded, warned = load_heard(path, mesh, DIM, steering_layer=CAPTURE_SLOT - 1)

        same = (
            np.array_equal(loaded.features, np.asarray(features, np.int32))
            and max_abs(loaded.vectors, bank.vectors) == 0.0
            and max_abs(loaded.probes, bank.probes) == 0.0
            and np.array_equal(loaded.thresholds, bank.thresholds)
        )
        print(
            f"  [{'PASS' if same else 'FAIL'}] the server reads what from_sae.py writes:"
            f" {len(loaded.features)} feature(s), thresholds"
            f" {np.array2string(loaded.thresholds, precision=4)}"
        )
        failures = int(not same)

        # Static steering reads no threshold, so it still adds a dead latent's vector. The bank
        # loads, and the warning has to name the feature.
        dead = [message for message in warned if "dead latent" in message]
        named = len(dead) == 1 and f"[{d_sae - 1}]" in dead[0]
        print(
            f"  [{'PASS' if named else 'FAIL'}] load names the dead latent, feature {d_sae - 1}:"
            f" {dead}"
        )
        failures += not named

        # Every layer is the same width, so width can't catch a site mismatch. The recorded
        # site can, and the number it has to catch is the off-by-one itself: an SAE trained on
        # capture slot 20 belongs at --steering-layer 19, and 20 has to be refused.
        refused = []
        for wrong in (CAPTURE_SLOT, CAPTURE_SLOT - 2, 0):
            try:
                STEERING_MODULE["SteeringBank"].load(path, mesh, DIM, steering_layer=wrong)
            except ValueError:
                refused.append(wrong)
        caught = refused == [CAPTURE_SLOT, CAPTURE_SLOT - 2, 0]
        print(
            f"  [{'PASS' if caught else 'FAIL'}] a bank fit at capture slot {CAPTURE_SLOT}"
            f" loads at --steering-layer {CAPTURE_SLOT - 1} and is refused at"
            f" {[CAPTURE_SLOT, CAPTURE_SLOT - 2, 0]}: refused {refused}"
        )
        failures += not caught

    # Control: a bank with no recorded site loads at any layer, which is what a bank written
    # before the site traveled with it does. It has to warn rather than raise, or every such
    # bank stops working.
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "nosite.npz")
        convert.save_bank(path, bank)
        try:
            STEERING_MODULE["SteeringBank"].load(path, mesh, DIM, steering_layer=41)
        except ValueError as exc:
            print(f"      control (a bank with no recorded site): RAISED {exc}")
            print("      FAIL: a bank that records nothing has to load, with a warning.")
            failures += 1
        else:
            print("      control (a bank with no recorded site): loaded, unchecked")

    # Control: a bank of live latents loads with no dead-latent warning, so the warning above
    # comes from the +inf row.
    live = convert.build_bank(params, theta, features[:-1])
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "live.npz")
        convert.save_bank(path, live, meta=site)
        _, warned = load_heard(path, mesh, DIM, steering_layer=CAPTURE_SLOT - 1)
    if any("dead latent" in message for message in warned):
        print(f"      control (a bank of live latents): WARNED {warned}")
        print("      FAIL: the dead-latent warning fires on a bank that holds none.")
        failures += 1
    else:
        print("      control (a bank of live latents): no dead-latent warning")

    # Control: a bank as wide as the wrong model, and one missing a key, both refused.
    for name, build in (
        ("the wrong hidden size", lambda p: convert.save_bank(p, bank)),
        (
            "a missing key",
            lambda p: np.savez(p, features=bank.features, vectors=bank.vectors),
        ),
    ):
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "bad.npz")
            build(path)
            width = DIM + 1 if name == "the wrong hidden size" else DIM
            try:
                STEERING_MODULE["SteeringBank"].load(path, mesh, width)
            except (ValueError, KeyError) as exc:
                print(f"      control ({name}): refused, {type(exc).__name__}")
            else:
                print(f"      control ({name}): ACCEPTED")
                print("      FAIL: the control didn't fail, so load checks nothing.")
                failures += 1

    # Each of these has to fail at startup. They used to load and then stop the server from the
    # scheduler thread: a short thresholds array on the first request for a fitted threshold,
    # an empty bank on the all-off batch every forward pass builds, a NaN threshold in the
    # multi-host placement, and a non-finite vector in every steered row.
    fields = {
        "features": np.asarray(bank.features),
        "vectors": np.asarray(bank.vectors),
        "probes": np.asarray(bank.probes),
        "thresholds": np.asarray(bank.thresholds),
    }
    nan_vector = fields["vectors"].copy()
    nan_vector[0, 0] = np.nan
    malformed = {
        "a threshold short": dict(fields, thresholds=fields["thresholds"][:-1]),
        "no feature at all": {key: value[:0] for key, value in fields.items()},
        "a NaN threshold": dict(fields, thresholds=np.r_[fields["thresholds"][:-1], np.nan]),
        "a -inf threshold": dict(fields, thresholds=np.r_[fields["thresholds"][:-1], -np.inf]),
        "a NaN in a vector": dict(fields, vectors=nan_vector),
    }
    for name, arrays in malformed.items():
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "malformed.npz")
            np.savez(path, meta=np.array(json.dumps(site)), **arrays)
            try:
                STEERING_MODULE["SteeringBank"].load(
                    path, mesh, DIM, steering_layer=CAPTURE_SLOT - 1
                )
            except ValueError as exc:
                print(f"  [PASS] load refuses a bank with {name}: {exc}")
            else:
                print(f"  [FAIL] load accepted a bank with {name}")
                failures += 1
    return failures


def mutate_function(name, transform):
    """A module mutation that rewrites one function's body."""

    def apply(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                transform(node)
                return
        raise LookupError(f"{name} not found")

    return apply


def _drop_normalization(fn):
    """`unit = probe / scale` becomes `unit = probe`, so the probe scale leaks in."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign) and getattr(sub.targets[0], "id", None) == "unit":
            sub.value = ast.Name(id="probe", ctx=ast.Load())


def _flip_compare(fn):
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Compare) and isinstance(sub.ops[0], ast.Gt):
            sub.ops[0] = ast.Lt()


def _drop_alpha_axis(fn):
    """`alpha = alpha[:, None]` removed, so one token's alpha spreads over the whole row."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Subscript):
            if getattr(sub.targets[0], "id", None) == "alpha":
                sub.value = ast.Name(id="alpha", ctx=ast.Load())


def _one_slot(fn):
    """Every token gathers bank row 0.

    The gather is written `steering.vectors.at[slot].get(out_sharding=...)`, so the subscript
    hangs off `.at` and the bank name sits one attribute further in. Match the plain
    `steering.vectors[slot]` form too, so this bites whichever way the hook is written.
    """
    hit = 0
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Subscript) or not isinstance(sub.value, ast.Attribute):
            continue
        owner = sub.value
        if owner.attr == "at" and isinstance(owner.value, ast.Attribute):
            owner = owner.value
        if owner.attr in ("vectors", "probes"):
            sub.slice = ast.Constant(value=0)
            hit += 1
    if not hit:
        raise LookupError("no bank gather to mutate; apply_steering changed shape")


def _probe_is_vector(fn):
    """The probe gathered out of the vector table, which reads along the wrong direction."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign) and getattr(sub.targets[0], "id", None) == "probe":
            for inner in ast.walk(sub.value):
                if isinstance(inner, ast.Attribute) and inner.attr == "probes":
                    inner.attr = "vectors"


def _branch_call(fn):
    """The `lax.cond(...)` call in `apply_steering`."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == "cond":
                return sub
    raise LookupError("apply_steering has no lax.cond")


def _swap_branches(fn):
    """The branch that only widens runs when a token steers, and the steering when none does."""
    call = _branch_call(fn)
    call.args[1], call.args[2] = call.args[2], call.args[1]


def _drop_branch(fn):
    """`lax.cond(pred, steer, widen, h)` becomes `steer(h)`: every batch gathers and projects."""
    call = _branch_call(fn)
    steer, operand = call.args[1], call.args[3]
    call.func, call.args, call.keywords = steer, [operand], []


MODULE_MUTANTS = {
    "probe normalization dropped": mutate_function("projection", _drop_normalization),
    "threshold compare flipped": mutate_function("threshold_fire", _flip_compare),
    "per-token alpha not reshaped": mutate_function("static_steer", _drop_alpha_axis),
    "every token gathers bank row 0": mutate_function("apply_steering", _one_slot),
    "probe read from the vector table": mutate_function("apply_steering", _probe_is_vector),
    "the steering branch runs when nobody steers": mutate_function(
        "apply_steering", _swap_branches
    ),
}


def check_module_mutants(sources, mesh, h64, steering):
    missed = 0
    want = reference_steer(h64, steering)
    h = jax.device_put(jnp.asarray(h64, jnp.float32), NamedSharding(mesh, P(AXIS, None)))
    for label, mutate in MODULE_MUTANTS.items():
        try:
            module = load_module(sources["after"][STEERING], "mutant", mutate=mutate)
        except Exception as exc:  # noqa: BLE001 - an operator that can't apply tested nothing
            missed += operator_missed(label, exc)
            continue
        try:
            got = jax.jit(module["apply_steering"])(h, steering)
            err = max_abs(got, want)
            # NaN is a broken hook too, so only an error under the tolerance is a miss.
            caught = not err < MUTANT_TOL
            how = f"max_abs={err:.3e}" if caught else f"NOT DETECTED (max_abs={err:.3e})"
        except Exception as exc:  # noqa: BLE001 - a mutant that raises is caught
            caught, how = True, f"raised {type(exc).__name__}"
        print(f"      control ({label}): {how}")
        if not caught:
            print("      FAIL: the control didn't fail, so this mutation is untested.")
            missed += 1

    missed += check_precision(mesh)
    return missed


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def check_precision(mesh, tokens=4096):
    """The precision point: the same read at bfloat16 moves the fired set.

    A threshold at the median projection puts half the batch on each side, so the tokens that
    decide it sit within a rounding error of the bar. The float32 read has to agree with float64
    on every one of them, because a token that flips takes the whole `alpha * v` with it.
    """
    rng = np.random.default_rng(23)
    h64 = rng.normal(size=(tokens, DIM))
    probe64 = rng.normal(size=(DIM,))
    projections = h64 @ _unit(probe64)
    bar = float(np.median(projections))
    want = projections > bar

    h = jax.device_put(jnp.asarray(h64, jnp.float32), NamedSharding(mesh, P(AXIS, None)))
    probe = jnp.asarray(probe64, jnp.float32)
    fire32 = np.asarray(STEERING_MODULE["threshold_fire"](h, probe, jnp.float32(bar)))
    fire16 = np.asarray(
        STEERING_MODULE["threshold_fire"](
            h.astype(jnp.bfloat16), probe.astype(jnp.bfloat16), jnp.bfloat16(bar)
        )
    )
    exact = int((fire32 != want).sum())
    rounded = int((fire16 != want).sum())
    detected = rounded > exact and exact == 0
    print(
        f"      control (the read at bfloat16): {rounded} of {tokens} token(s) on the wrong"
        f" side against {exact} at float32 -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so the float32 read is untested.")
        return 1
    return 0


# --------------------------------------------------------------------------------------
# 5. the scheduler's per-token rules
# --------------------------------------------------------------------------------------


def check_scheduler(sources):
    """`_steering_token_rows` and `_steering_threshold`, pulled out of the patched source."""
    failures = 0
    src = sources["after"][SCHEDULE_BATCH]
    namespace = {"np": np}
    rows = compile_function(find_function(src, "_steering_token_rows"), namespace)
    threshold = compile_function(find_function(src, "_steering_threshold"), namespace)

    def window(fn, config, prefix=10, width=4):
        """One request's window: four tokens, starting at position 10."""
        return [int(r) for r in fn(config, prefix, width)]

    cases = [
        ("no positions steers the window", window(rows, {"feature": 1}), [0, 1, 2, 3]),
        (
            "positions map through the prefix",
            window(rows, {"feature": 1, "positions": [10, 13]}),
            [0, 3],
        ),
        (
            "a position in another chunk drops",
            window(rows, {"feature": 1, "positions": [3, 11, 99]}),
            [1],
        ),
        (
            "a repeated position steers once",
            window(rows, {"feature": 1, "positions": [11, 11]}),
            [1],
        ),
        ("static is -inf", threshold({"feature": 1}), float("-inf")),
        ("conditional is NaN", np.isnan(threshold({"feature": 1, "mode": "conditional"})), True),
        (
            "an explicit threshold wins",
            threshold({"feature": 1, "mode": "conditional", "threshold": 0.5}),
            0.5,
        ),
        ("a threshold without a mode wins too", threshold({"feature": 1, "threshold": -2.0}), -2.0),
    ]
    for name, got, want in cases:
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {got}")
        failures += not ok

    # A control compares against the value a defect would produce. NaN is never equal to itself,
    # so the "static read as conditional" control asks whether the result is NaN rather than
    # comparing to NaN, which would read "detected" either way.
    controls = [
        (
            "the prefix ignored",
            [int(r) for r in rows({"feature": 1, "positions": [10, 13]}, 0, 4)],
            [0, 3],
        ),
        (
            "mode misread as conditional",
            bool(np.isnan(threshold({"feature": 1, "mode": "static"}))),
            True,
        ),
    ]
    for name, got, wrong in controls:
        detected = got != wrong
        print(f"      control ({name}): {got} -> {'detected' if detected else 'NOT DETECTED'}")
        failures += not detected
    return failures


# --------------------------------------------------------------------------------------
# 6. the per-token fields reach the worker batch
# --------------------------------------------------------------------------------------


def worker_batch_stub(src):
    """A dataclass with `ModelWorkerBatch`'s real field list, read out of the patched source.

    `get_model_worker_batch` spreads `**_steering` into the `ModelWorkerBatch(...)` call, so a
    key with no matching field is a `TypeError` at construction. Building the real class here
    would drag in the whole engine, so this rebuilds its signature instead, which is the part
    the spread lands on.
    """
    names = dataclass_fields(src, "ModelWorkerBatch")
    if not names:
        raise LookupError("ModelWorkerBatch not found")
    return dataclasses.make_dataclass(
        "ModelWorkerBatchStub",
        [(name, object, dataclasses.field(default=None)) for name in names],
    )


def steering_keys(src):
    """The keys `_merge_steering` returns, read off both of its return statements."""
    fn = find_method(src, "ScheduleBatch", "_merge_steering")
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return sorted(keys)


def check_worker_batch(sources):
    """6. Every key the merge returns is a field of the batch the merge is spread into."""
    failures = 0
    src = sources["after"][SCHEDULE_BATCH]
    keys = steering_keys(src)
    expected = ["steering_alpha", "steering_feature", "steering_threshold"]
    ok = keys == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] _merge_steering returns {keys}")
    failures += not ok

    # `_merge_steering` returns all three keys on every path, including the one for a batch
    # nobody steers, so a missing field raises on every forward pass and not just a steered one.
    stub = worker_batch_stub(src)
    empty = dict.fromkeys(keys)
    filled = {key: np.zeros(4, np.float32) for key in keys}
    for label, payload in (("no request steers", empty), ("a request steers", filled)):
        try:
            stub(**payload)
        except TypeError as exc:
            print(f"  [FAIL] {label}: ModelWorkerBatch(**_steering) raised {exc}")
            failures += 1
        else:
            print(f"  [PASS] {label}: ModelWorkerBatch takes **_steering")

    # Control: the same spread against the unpatched field list has to raise, or this check
    # reads nothing.
    before = worker_batch_stub(sources["before"][SCHEDULE_BATCH])
    try:
        before(**empty)
    except TypeError as exc:
        print(f"      control (the unpatched field list): refused, {type(exc).__name__}")
    else:
        print("      control (the unpatched field list): ACCEPTED")
        print("      FAIL: the control didn't fail, so the field list is untested.")
        failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 7. one executable covers the steered and the unsteered batch
# --------------------------------------------------------------------------------------


def init_new_steering(src, device_array):
    """The steering block of `ForwardBatch.init_new`, as a callable.

    Takes the statements the patch adds, from `steering = None` down to the `obj = cls(...)`
    they feed, and runs them as written. Nothing is retyped, so the treedef this produces is
    the treedef the engine produces. `device_array` is the real one from `utils/jax_utils.py`,
    which the block places its arrays with.
    """
    block = steering_block(src)
    if not block:
        raise LookupError("init_new builds no steering batch")
    wrapper = ast.FunctionDef(
        name="_build_steering",
        args=ast.parse("def f(batch, model_runner): pass").body[0].args,
        body=[*copy.deepcopy(block), ast.Return(value=ast.Name(id="steering", ctx=ast.Load()))],
        decorator_list=[],
    )
    namespace = {
        "np": np,
        "NamedSharding": NamedSharding,
        "PartitionSpec": P,
        "device_array": device_array,
        "getattr": getattr,
        "len": len,
    }
    return compile_function(wrapper, namespace)


def check_one_executable(sources, mesh):
    """7. A batch nobody steers still carries a steering batch, so both share one trace."""
    failures = 0
    rng = np.random.default_rng(101)
    bank, feature, alpha, threshold = build_steering(STEERING_MODULE, mesh, rng)
    build = init_new_steering(sources["after"][FORWARD_BATCH], DEVICE_ARRAY)
    runner = Stub(steering_bank=bank, mesh=mesh)

    steered = build(
        Stub(
            input_ids=np.zeros(TOKENS, np.int32),
            steering_feature=feature,
            steering_alpha=alpha,
            steering_threshold=threshold,
        ),
        runner,
    )
    quiet = build(
        Stub(
            input_ids=np.zeros(TOKENS, np.int32),
            steering_feature=None,
            steering_alpha=None,
            steering_threshold=None,
        ),
        runner,
    )

    built = quiet is not None
    print(
        f"  [{'PASS' if built else 'FAIL'}] init_new builds a steering batch for a batch"
        f" nobody steers"
    )
    failures += not built
    if not built:
        return failures

    off = (
        np.array_equal(np.asarray(quiet.alpha), np.zeros(TOKENS, np.float32))
        and np.isposinf(np.asarray(quiet.threshold)).all()
        and int(np.asarray(quiet.slot).sum()) == 0
    )
    print(
        f"  [{'PASS' if off else 'FAIL'}] the all-off batch fires on no token:"
        f" alpha max={float(np.abs(np.asarray(quiet.alpha)).max())},"
        f" thresholds all +inf={bool(np.isposinf(np.asarray(quiet.threshold)).all())}"
    )
    failures += not off

    # The measurement: how many times the hook traces. Both batches are the same treedef with
    # the same avals, so the second call reuses the first call's executable. Each run gets its
    # own `jax.jit`, so the cache starts empty and the count is the count.
    h = jax.device_put(
        jnp.asarray(rng.normal(size=(TOKENS, DIM)), jnp.bfloat16),
        NamedSharding(mesh, P(AXIS, None)),
    )

    def trace_count(second):
        traces = []
        apply_steering = STEERING_MODULE["apply_steering"]

        @jax.jit
        def hook(stream, steering):
            traces.append(1)
            return apply_steering(stream, steering)

        first_out = hook(h, steered)
        second_out = hook(h, second)
        return len(traces), first_out, second_out

    count, hot, cold = trace_count(quiet)
    shared = count == 1
    print(
        f"  [{'PASS' if shared else 'FAIL'}] the steered and the unsteered batch share one"
        f" trace: {count}"
    )
    failures += not shared

    widened = jnp.dtype(hot.dtype).name == "float32" and jnp.dtype(cold.dtype).name == "float32"
    moved = int((np.asarray(cold, np.float64) != np.asarray(h, np.float64)).any(axis=1).sum())
    ok = widened and moved == 0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] both leave the stream float32 and the unsteered one"
        f" moves {moved} token(s)"
    )
    failures += not ok

    # Control: the form that skips the batch. `None` is a different treedef, so the same two
    # calls trace twice, and the second graph carries a bf16 stream where the first carries
    # float32. A precompiled bucket covers one of the two.
    split_count, _, narrow = trace_count(None)
    split = split_count == 2
    print(
        f"      control (None in place of the all-off batch): {split_count} traces,"
        f" stream comes back {jnp.dtype(narrow.dtype).name}"
        f" -> {'detected' if split else 'NOT DETECTED'}"
    )
    if not split:
        print("      FAIL: the control didn't fail, so the trace count reads nothing.")
        failures += 1

    # The all-off arrays are placed once per token count. The next batch of the same size gets
    # the same device arrays back, and a batch of another size gets its own.
    def quiet_batch(tokens):
        return build(
            Stub(
                input_ids=np.zeros(tokens, np.int32),
                steering_feature=None,
                steering_alpha=None,
                steering_threshold=None,
            ),
            runner,
        )

    again, other = quiet_batch(TOKENS), quiet_batch(TOKENS // 2)
    reused = all(getattr(again, f) is getattr(quiet, f) for f in ("slot", "alpha", "threshold"))
    print(f"  [{'PASS' if reused else 'FAIL'}] a second all-off batch reuses the placed arrays")
    failures += not reused
    separate = other.alpha is not quiet.alpha and other.alpha.shape == (TOKENS // 2,)
    print(
        f"      control (an all-off batch of {TOKENS // 2} tokens):"
        f" {'its own arrays' if separate else 'SHARED'}"
    )
    if not separate:
        print("      FAIL: the control didn't fail, so the reuse check reads nothing.")
        failures += 1

    # The hook's branch for a batch nobody steers gathers nothing out of the bank.
    gathers = branch_gathers(STEERING_MODULE["apply_steering"], h, quiet)
    ok = gathers == [0, 2]
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the branch a batch nobody steers takes skips the bank"
        f" gathers: gathers per branch {gathers}"
    )
    failures += not ok
    # Control: the hook with the branch taken out gathers on every batch.
    try:
        unbranched = load_module(
            sources["after"][STEERING],
            "sgl_steering_unbranched",
            mutate=mutate_function("apply_steering", _drop_branch),
        )
    except Exception as exc:  # noqa: BLE001 - an operator that can't apply tested nothing
        return failures + operator_missed("the hook with no branch", exc)
    flat = branch_gathers(unbranched["apply_steering"], h, quiet)
    detected = flat is None
    print(
        f"      control (the hook with no branch): {'no branch' if detected else flat}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so the branch check reads nothing.")
        failures += 1
    return failures


def count_primitive(jaxpr, name):
    """How many equations named `name` a jaxpr holds, the jaxprs in their params included."""
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    total = 0
    for eqn in jaxpr.eqns:
        total += eqn.primitive.name == name
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(getattr(item, "jaxpr", item), "eqns"):
                    total += count_primitive(item, name)
    return total


def branch_gathers(fn, h, steering):
    """Gathers in each branch of the first `cond` in fn's jaxpr, sorted. None with no `cond`."""

    def find(jaxpr):
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "cond":
                return eqn
            for value in eqn.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(getattr(item, "jaxpr", item), "eqns"):
                        found = find(item)
                        if found is not None:
                            return found
        return None

    cond = find(jax.make_jaxpr(fn)(h, steering))
    if cond is None:
        return None
    return sorted(count_primitive(branch, "gather") for branch in cond.params["branches"])


# --------------------------------------------------------------------------------------
# 8. the request validator
# --------------------------------------------------------------------------------------

# Each bad request, the field the validator's refusal has to name, and what the scheduler thread
# makes of it when nothing refuses it. The downstream control has to show that failure:
#   raises          the scheduler thread raises, which stops request handling for every caller
#   alpha inf       alpha lands as inf in float32, and inf * v takes a steered row to NaN
#   threshold inf   a finite threshold lands as inf, so it never fires or always does
#   fitted          the threshold comes out NaN, which the bank reads as "use the fitted one"
#   static          the request runs as static steering, whatever mode it asked for
#   conditional     a static request runs as conditional steering at its threshold
#   every token     the request steers every token of the window, not the positions it named
#   no token        the request steers no token and gets no error
#   no shift        the request steers with alpha 0, which moves nothing
#   another feature the feature id lands as an int the request didn't send
BAD_REQUESTS = [
    ("alpha is None", {"feature": 1, "alpha": None}, "alpha", "raises"),
    ("alpha is a string", {"feature": 1, "alpha": "strong"}, "alpha", "raises"),
    ("alpha isn't finite", {"feature": 1, "alpha": float("inf")}, "alpha", "alpha inf"),
    (
        "threshold is a string",
        {"feature": 1, "alpha": 1.0, "threshold": "hot"},
        "threshold",
        "raises",
    ),
    (
        "threshold is NaN",
        {"feature": 1, "alpha": 1.0, "threshold": float("nan")},
        "threshold",
        "fitted",
    ),
    ("mode isn't a string", {"feature": 1, "alpha": 1.0, "mode": 7}, "mode", "static"),
    (
        "positions isn't a list",
        {"feature": 1, "alpha": 1.0, "positions": 4},
        "positions",
        "raises",
    ),
    (
        "a position is a string",
        {"feature": 1, "alpha": 1.0, "positions": ["four"]},
        "positions",
        "raises",
    ),
    (
        "a position is negative",
        {"feature": 1, "alpha": 1.0, "positions": [-1]},
        "positions",
        "no token",
    ),
    ("feature is a float", {"feature": 1.5, "alpha": 1.0}, "feature", "another feature"),
    # The bank's feature ids are int32 and the positions int64. A larger value raises
    # OverflowError on the scheduler thread, which then signals the whole server to exit.
    ("feature is past int32", {"feature": 2**31, "alpha": 1.0}, "feature", "raises"),
    (
        "a position is past int64",
        {"feature": 1, "alpha": 1.0, "positions": [2**63]},
        "positions",
        "raises",
    ),
    ("feature is missing", {"alpha": 1.0}, "feature", "raises"),
    ("alpha is missing", {"feature": 1}, "alpha", "no shift"),
    # The scheduler stores alpha and the threshold as float32. A finite float64 past that range
    # turns into inf there.
    ("alpha is past float32", {"feature": 1, "alpha": 1e39}, "alpha", "alpha inf"),
    ("alpha is past float64", {"feature": 1, "alpha": 10**400}, "alpha", "raises"),
    (
        "threshold is past float32",
        {"feature": 1, "alpha": 1.0, "threshold": 1e39},
        "threshold",
        "threshold inf",
    ),
    (
        "threshold is below float32",
        {"feature": 1, "alpha": 1.0, "threshold": -1e39},
        "threshold",
        "threshold inf",
    ),
    # A misspelled key never reaches the field it meant, so the scheduler reads that field's
    # default.
    (
        "positions is misspelled",
        {"feature": 1, "alpha": 1.0, "position": [2]},
        "position",
        "every token",
    ),
    (
        "threshold is misspelled",
        {"feature": 1, "alpha": 1.0, "mode": "conditional", "treshold": 0.5},
        "treshold",
        "fitted",
    ),
    ("mode is misspelled", {"feature": 1, "alpha": 1.0, "Mode": "conditional"}, "Mode", "static"),
    (
        "a threshold beside mode static",
        {"feature": 1, "alpha": 1.0, "mode": "static", "threshold": 0.3},
        "static",
        "conditional",
    ),
]

WINDOW = 4


def shows_harm(harm, config, outcome):
    """True when what the scheduler thread made of a request is the failure `harm` names."""
    if harm == "raises":
        return isinstance(outcome, Exception)
    if isinstance(outcome, Exception):
        return False
    rows, feature, alpha, threshold = outcome
    return {
        "alpha inf": np.isinf(alpha),
        "threshold inf": np.isinf(threshold),
        "fitted": np.isnan(threshold),
        "static": threshold == -np.inf,
        "conditional": np.isfinite(threshold),
        "every token": rows == list(range(WINDOW)),
        "no token": rows == [],
        "no shift": alpha == 0.0,
        "another feature": feature != config.get("feature"),
    }[harm]


GOOD_REQUEST = {
    "feature": 1,
    "alpha": 1.5,
    "mode": "conditional",
    "threshold": 0.25,
    "positions": [0, 3],
}

# The edges of what a request may send. An infinite threshold means never fire or always fire,
# and float32 holds it as it is. A threshold with no mode means conditional steering.
EDGE_REQUESTS = [
    {"feature": 1, "alpha": 3e38, "threshold": float("-inf")},
    {"feature": 1, "alpha": -3e38, "mode": "conditional", "threshold": float("inf")},
    {"feature": 2**31 - 1, "alpha": 1, "positions": [2**63 - 1]},
    {"feature": 1, "alpha": 1.0, "threshold": 0.3},
    {"feature": 1, "alpha": 1.0, "mode": "STATIC"},
]


def check_validation(sources):
    """8. The front door rejects what the scheduler thread would raise on or misread."""
    failures = 0
    io_src = sources["after"][IO_STRUCT]
    schedule_src = sources["after"][SCHEDULE_BATCH]
    validate = compile_function(
        find_function(io_src, "_validate_steering"),
        {"np": np, **module_constants(io_src, "STEERING_FIELDS")},
    )
    rows = compile_function(find_function(schedule_src, "_steering_token_rows"), {"np": np})
    threshold = compile_function(find_function(schedule_src, "_steering_threshold"), {"np": np})

    def downstream(config):
        """What the scheduler thread makes of a request, on the thread that has nowhere to report.

        `_merge_steering` lays the request over a window, here WINDOW tokens from position 0: the
        rows it steers, the feature id in an int32 array, and alpha and the threshold in float32
        arrays. Returns (rows, feature, alpha, threshold) as those arrays hold them, or the
        exception the thread raises.
        """
        try:
            with np.errstate(over="ignore"):
                picked = [int(r) for r in rows(config, 0, WINDOW)]
                feature = int(np.int32(int(config["feature"])))
                alpha = float(np.float32(float(config.get("alpha", 0.0))))
                limit = float(np.float32(threshold(config)))
        except (TypeError, ValueError, OverflowError, KeyError) as exc:
            return exc
        return picked, feature, alpha, limit

    for name, config, field, harm in BAD_REQUESTS:
        try:
            validate(config)
        except ValueError as exc:
            rejected, why = field in str(exc), str(exc)
        else:
            rejected, why = False, "accepted"
        print(f"  [{'PASS' if rejected else 'FAIL'}] {name:<30} {why}")
        failures += not rejected

        # The control shows what the value costs if it gets past here. Either the scheduler
        # thread raises on it, which stops request handling for every caller, or it resolves to
        # something the request never asked for. A value that does neither isn't the defect
        # its entry names, so the control fails.
        outcome = downstream(config)
        if isinstance(outcome, Exception):
            described = f"raises {type(outcome).__name__}"
        else:
            picked, feature, alpha, limit = outcome
            described = (
                f"resolves to rows={picked} feature={feature} alpha={alpha} threshold={limit}"
            )
        shown = shows_harm(harm, config, outcome)
        print(
            f"      control (the scheduler thread, {harm}): {described}"
            f" -> {'shown' if shown else 'NOT SHOWN'}"
        )
        if not shown:
            print("      FAIL: the control didn't fail, so this refusal protects nothing it shows.")
            failures += 1

    try:
        validate(GOOD_REQUEST)
        validate(None)
        validate({"feature": 0, "alpha": 0.0})
        for config in EDGE_REQUESTS:
            validate(config)
    except Exception as exc:  # noqa: BLE001 - a valid request must not raise at all
        print(f"      control (a valid request): RAISED {type(exc).__name__}: {exc}")
        print("      FAIL: the validator rejects a request the API documents.")
        failures += 1
    else:
        print("      control (a valid request): accepted")
    return failures


# --------------------------------------------------------------------------------------


STEERING_MODULE = None
# `device_array` from the checkout's utils/jax_utils.py, which the patch leaves as it is.
DEVICE_ARRAY = None


def _drop_out_sharding(fn):
    """Gather the bank rows without naming the output sharding.

    This is how the hook was first written. It resolves fine under auto sharding, which is what
    checks 3, 4, 7 and 14 run, and raises under explicit sharding, which is what the
    engine runs. That gap shipped a server that loaded and then died on the first steered
    request, so this mutation is the control for check 9.
    """
    hit = 0
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not isinstance(func, ast.Attribute) or func.attr != "get":
            continue
        if not isinstance(func.value, ast.Subscript):
            continue
        owner = func.value.value
        if not isinstance(owner, ast.Attribute) or owner.attr != "at":
            continue
        sub.keywords = [k for k in sub.keywords if k.arg != "out_sharding"]
        hit += 1
    if not hit:
        raise LookupError("no `.at[...].get(out_sharding=)` to strip")


def check_explicit_sharding():
    """The engine traces under explicit sharding. Checks 3, 4, 7 and 14 use auto.

    `sgl_jax` runs its forward under a mesh whose axes are Explicit, where a gather with a
    replicated operand and token-sharded indices has no inferable output sharding. Run
    `apply_steering` there for real, and require the un-named form to fail in the same place.
    """
    failures = 0
    explicit = jax.make_mesh((8,), (AXIS,), axis_types=(jax.sharding.AxisType.Explicit,))
    rng = np.random.default_rng(11)

    with jax.set_mesh(explicit):
        bank, feature, alpha, threshold = build_steering(STEERING_MODULE, explicit, rng)
        steering = steering_batch(bank, feature, alpha, threshold, NamedSharding(explicit, P(AXIS)))
        h64 = rng.normal(size=(TOKENS, DIM))
        h = jax.device_put(jnp.asarray(h64, jnp.float32), NamedSharding(explicit, P(AXIS, None)))

        try:
            got = jax.jit(STEERING_MODULE["apply_steering"])(h, steering)
            err = max_abs(got, reference_steer(np.asarray(h, np.float64), steering))
            ok = err < F32_TOL
            spec = jax.typeof(got).sharding.spec
            print(
                f"  [{'PASS' if ok else 'FAIL'}] apply_steering traces under explicit sharding:"
                f" max_abs={err:.3e} out_spec={spec}"
            )
            failures += not ok
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [FAIL] apply_steering raised under explicit sharding:"
                f" {type(exc).__name__}: {exc}"
            )
            return failures + 1

        try:
            unnamed = load_module(
                SOURCES_AFTER[STEERING],
                "sgl_steering_noshard",
                mutate=mutate_function("apply_steering", _drop_out_sharding),
            )
        except Exception as exc:  # noqa: BLE001 - an operator that can't apply tested nothing
            return failures + operator_missed("out_sharding dropped", exc)
        try:
            jax.jit(unnamed["apply_steering"])(h, steering)
            print("      control (out_sharding dropped): NOT DETECTED")
            print("      FAIL: the control didn't fail, so this mutation is untested.")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"      control (out_sharding dropped): {type(exc).__name__} -> detected")

    return failures


# --------------------------------------------------------------------------------------
# 10. the cache key
# --------------------------------------------------------------------------------------


def _drop_positions(fn):
    """The key built without the positions, so two requests that steer different tokens share."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Dict):
            keep = [
                i
                for i, key in enumerate(sub.keys)
                if not (isinstance(key, ast.Constant) and key.value == "positions")
            ]
            sub.keys = [sub.keys[i] for i in keep]
            sub.values = [sub.values[i] for i in keep]


def cache_key_cases(key):
    """Pairs that must share a key, and pairs that must not. Returns (label, ok) per pair."""
    base = {"feature": 7, "alpha": 1.5}
    same = [
        ("an int alpha and its float", {"feature": 7, "alpha": 2}, {"feature": 7, "alpha": 2.0}),
        ("the default mode, spelled out", base, dict(base, mode="static")),
        ("mode in capitals", dict(base, mode="conditional"), dict(base, mode="CONDITIONAL")),
        (
            "positions in another order, one repeated",
            dict(base, positions=[3, 1]),
            dict(base, positions=[1, 3, 3]),
        ),
    ]
    different = [
        ("another feature", base, dict(base, feature=8)),
        ("another alpha", base, dict(base, alpha=1.25)),
        ("static against conditional", base, dict(base, mode="conditional")),
        ("another threshold", dict(base, threshold=0.25), dict(base, threshold=0.3)),
        ("a threshold against none", base, dict(base, threshold=0.25)),
        ("other positions", dict(base, positions=[1, 3]), dict(base, positions=[1])),
        ("positions against every token", base, dict(base, positions=[0, 1, 2])),
    ]
    out = [(f"same key: {label}", key(a) == key(b)) for label, a, b in same]
    out += [(f"another key: {label}", key(a) != key(b)) for label, a, b in different]
    return out


def check_cache_key(sources):
    """10. A steered request shares cached KV only with requests that steer the same way."""
    failures = 0
    src = sources["after"][IO_STRUCT]
    namespace = {"json": json, **module_constants(src, "STEERING_KEY_PREFIX")}
    key = compile_function(find_function(src, "steering_cache_key"), dict(namespace))
    for label, ok in cache_key_cases(key):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += not ok

    # The tokenizer manager refuses a client extra_key that holds the prefix. That rules out a
    # collision only if every key starts with the prefix and holds it nowhere else, so a client
    # key and a steering key can't join into another request's key.
    prefix = namespace.get("STEERING_KEY_PREFIX")
    if prefix is None:
        print("  [FAIL] io_struct.py names no STEERING_KEY_PREFIX for the tokenizer manager to refuse")
        return failures + 1
    configs = [
        {"feature": 7, "alpha": 1.5},
        {"feature": 2**31 - 1, "alpha": -3e38, "mode": "CONDITIONAL", "threshold": float("inf")},
        {"feature": 0, "alpha": 0, "mode": "static", "positions": [2**63 - 1, 0, 0]},
    ]

    def holds_prefix_once(k):
        return k.startswith(prefix) and prefix not in k[len(prefix) :]

    ok = all(holds_prefix_once(key(config)) for config in configs)
    print(f"  [{'PASS' if ok else 'FAIL'}] every key starts with {prefix!r} and holds it once")
    failures += not ok
    # Control: a key whose body repeats the prefix, as a mode string could if the validator
    # let one through, has to fail the same test.
    doubled = prefix + json.dumps({"mode": prefix})
    detected = not holds_prefix_once(doubled)
    print(
        f"      control (a key whose body holds the prefix):"
        f" {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so the prefix test reads nothing.")
        failures += 1

    # Control: the same cases against a key that forgets the positions have to fail.
    fn = find_function(src, "steering_cache_key")
    _drop_positions(fn)
    forgetful = compile_function(fn, dict(namespace))
    missed = [label for label, ok in cache_key_cases(forgetful) if not ok]
    print(f"      control (the key forgets the positions): {missed or 'NOT DETECTED'}")
    if not missed:
        print("      FAIL: the control didn't fail, so the key cases read nothing.")
        failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 11 and 12. the real Engine
# --------------------------------------------------------------------------------------


def build_stacked_tree(repo):
    """The tree a VM serves: 877 through `git am`, then the two steering patches through `git
    apply`, the way bootstrap_tpu_vm.sh and scripts/cpu_engine.py build it."""
    who = ["-c", "user.email=steering-test@local", "-c", "user.name=steering-test"]
    am = subprocess.run(
        ["git", "-C", repo, *who, "am", "--quiet", PATCH_877], capture_output=True, text=True
    )
    if am.returncode != 0:
        raise RuntimeError(f"sglang-jax-877.patch doesn't apply: {am.stdout}{am.stderr}")
    for patch in (PATCH, PATCH_QWEN3):
        applied = git(repo, "apply", patch)
        if applied.returncode != 0:
            raise RuntimeError(f"{os.path.basename(patch)} doesn't stack: {applied.stderr}")
    return repo


def cpu_test_module(tree):
    """The 877 patch's CPU test, for its tiny random model and its Engine settings."""
    spec = importlib.util.spec_from_file_location("cpu_test", os.path.join(tree, CPU_TEST))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_bank(path, dim, seed=0):
    """A one-feature bank at --steering-layer 1: feature 7 reads and writes one random direction."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    v /= np.linalg.norm(v)
    np.savez(
        path,
        features=np.array([7], np.int32),
        vectors=v[None, :],
        probes=v[None, :],
        thresholds=np.array([0.0], np.float32),
        meta=np.array(json.dumps({"steering_layer": 1, "capture_layer": 2})),
    )


def run_engine_child(scenario, tree, kwargs):
    """One scenario on the real Engine, in a child process. Returns (returncode, results, log)."""
    env = dict(os.environ)
    # One device, for tp_size 1. The eight this file forces for its own mesh would make the
    # engine's mesh refuse to build.
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=1")
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--engine-child",
            scenario,
            "--tree",
            tree,
            "--engine-args",
            json.dumps(kwargs),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    results = [
        json.loads(line[len("RESULT ") :])
        for line in proc.stdout.splitlines()
        if line.startswith("RESULT ")
    ]
    return proc.returncode, results, proc.stdout + proc.stderr


def engine_child(argv):
    """The child side of checks 11 and 12."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-child", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--engine-args", required=True)
    args = parser.parse_args(argv)
    kwargs = json.loads(args.engine_args)
    sys.path.insert(0, os.path.join(args.tree, "python"))

    def report(**fields):
        print("RESULT " + json.dumps(fields), flush=True)

    if args.engine_child == "check_args":
        from sgl_jax.srt.server_args import ServerArgs

        try:
            ServerArgs(**kwargs).check_server_args()
            report(case="check_args", error=None)
        except (ValueError, AssertionError) as exc:
            report(case="check_args", error=str(exc))
        return 0

    from sgl_jax.srt.entrypoints.engine import Engine
    from sgl_jax.srt.managers.io_struct import steering_cache_key

    try:
        engine = Engine(**kwargs)
    except ValueError as exc:
        report(case="refused_at_start", error=str(exc))
        return 0

    prompt = np.random.default_rng(1).integers(3, 512, size=130).tolist()
    sampling = {"max_new_tokens": 8, "temperature": 0.0, "ignore_eos": True}
    steer = {"feature": 7, "alpha": 40.0}
    # POST /generate to the real app in this process, from the 877 patch's CPU test.
    post_generate = cpu_test_module(args.tree).post_generate
    body = {"input_ids": prompt, "sampling_params": sampling}

    if args.engine_child == "flag_off":
        try:
            engine.generate(input_ids=prompt, sampling_params=sampling, steering=steer)
            refused = None
        except ValueError as exc:
            refused = str(exc)
        over_http = post_generate(engine, {**body, "steering": steer})
        after = engine.generate(input_ids=prompt, sampling_params=sampling)
        report(
            case="flag_off",
            refused=refused,
            http_status=over_http.status_code,
            http_body=over_http.text,
            served_after=len(after["output_ids"]),
        )
        engine.shutdown()
        return 0

    def run(label, steering):
        out = engine.generate(input_ids=prompt, sampling_params=sampling, steering=steering)
        report(
            case="radix",
            label=label,
            cached_tokens=int(out["meta_info"]["cached_tokens"]),
            output_ids=[int(t) for t in out["output_ids"]],
        )

    run("unsteered, fresh cache", None)
    engine.flush_cache()
    run("steered, fresh cache", steer)
    run("unsteered, after the steered request", None)
    run("steered again, after the steered request", steer)
    engine.flush_cache()
    run("unsteered, fresh cache again", None)
    run("steered, after an unsteered request", steer)

    # An unsteered request whose extra_key is the steered request's namespace would share its
    # cached KV. Engine.generate has no extra_key, so this goes through HTTP /generate.
    forged = post_generate(engine, {**body, "extra_key": steering_cache_key(steer)})
    plain = post_generate(engine, {**body, "extra_key": "tenant-a"})
    report(
        case="forged",
        status=forged.status_code,
        body=forged.text,
        control_status=plain.status_code,
    )
    engine.shutdown()
    return 0


def engine_settings(tree, workdir, **overrides):
    """The CPU test's Engine settings, pointed at a tiny Qwen3 and a bank at layer 1."""
    cpu_test = cpu_test_module(tree)
    model = os.path.join(workdir, "tiny-qwen3")
    bank = os.path.join(workdir, "bank_l1.npz")
    if not os.path.isdir(model):
        cpu_test.write_tiny_model(model, "qwen3")
        write_bank(bank, 128)
    settings = dict(
        enable_return_hidden_states=False,
        enable_steering=True,
        steering_bank=bank,
        steering_layer=1,
    )
    settings.update(overrides)
    return cpu_test.engine_args(model, **settings)


def check_engine_cache(tree, workdir):
    """11. The radix cache keeps steered and unsteered KV apart, on the real Engine."""
    failures = 0
    code, results, log = run_engine_child(
        "radix", tree, engine_settings(tree, workdir, log_level="debug")
    )
    got = {r["label"]: r for r in results if r.get("case") == "radix"}
    if len(got) != 6:
        print(f"  [FAIL] the engine run returned {len(got)} of 6 results, exit {code}. Log:")
        print(log)
        return 1

    fresh = got["unsteered, fresh cache"]
    steered = got["steered, fresh cache"]
    checks = [
        (
            "control: steering changes the output, so a leak between the two shows",
            steered["output_ids"] != fresh["output_ids"],
        ),
        (
            "an unsteered request after a steered one reuses none of its KV",
            got["unsteered, after the steered request"]["cached_tokens"] == 0
            and got["unsteered, after the steered request"]["output_ids"] == fresh["output_ids"],
        ),
        (
            "control: a second request that steers the same way reuses the cache",
            got["steered again, after the steered request"]["cached_tokens"] > 0
            and got["steered again, after the steered request"]["output_ids"]
            == steered["output_ids"],
        ),
        (
            "a steered request after an unsteered one reuses none of its KV",
            got["steered, after an unsteered request"]["cached_tokens"] == 0
            and got["steered, after an unsteered request"]["output_ids"] == steered["output_ids"],
        ),
    ]
    forged = [r for r in results if r.get("case") == "forged"]
    forged = forged[0] if forged else {}
    checks += [
        (
            "a client extra_key that names a steered namespace is refused",
            forged.get("status") == 400 and "steering:" in forged.get("body", ""),
        ),
        (
            "control: a client extra_key that names none is served",
            forged.get("control_status") == 200,
        ),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += not ok
    for label, r in got.items():
        print(f"      {label}: cached {r['cached_tokens']}, output {r['output_ids']}")
    print(f"      forged extra_key: {forged}")

    # The steering dict, logged in full where it crosses between components.
    crossings = {
        "the tokenizer manager sends it": "leaves the tokenizer manager",
        "the scheduler receives it": "reached the scheduler",
        "the scheduler lays it out per token": "over positions",
        "init_new resolves it against the bank": "token(s) steer; rows",
    }
    for label, marker in crossings.items():
        ok = marker in log and "'feature': 7" in log
        print(f"  [{'PASS' if ok else 'FAIL'}] debug log: {label}")
        failures += not ok
    if failures:
        print("      the engine log:")
        print(log)
    return failures


def check_engine_refusal(tree, workdir):
    """12. The real Engine refuses --enable-steering with a speculative algorithm, and a steered
    request on a server without --enable-steering."""
    failures = 0

    # A server without the flag holds no bank, so a steered request would come back unsteered.
    # It has to be refused where the caller sees it, through the Engine and through HTTP.
    code, results, log = run_engine_child(
        "flag_off", tree, engine_settings(tree, workdir, enable_steering=False)
    )
    found = [r for r in results if r.get("case") == "flag_off"]
    r = found[0] if found else {}
    ok = (
        "--enable-steering" in (r.get("refused") or "")
        and r.get("http_status") == 400
        and "--enable-steering" in r.get("http_body", "")
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a server without the flag refuses a steered request:"
        f" {r}"
    )
    failures += not ok
    # Control: the same server serves the same prompt unsteered, so the refusal is the steering
    # field's and not a server that can't serve.
    served = r.get("served_after") == 8
    print(f"      control (the same prompt without steering): served {r.get('served_after')}")
    if not served:
        print("      FAIL: the server served nothing, so its refusal says nothing about steering.")
        failures += 1
    if not (ok and served):
        print(f"      the engine exited {code}. Its log:")
        print(log)

    code, results, log = run_engine_child(
        "start", tree, engine_settings(tree, workdir, speculative_algorithm="EAGLE3")
    )
    refused = [r["error"] for r in results if r.get("case") == "refused_at_start"]
    ok = bool(refused) and "--enable-steering" in refused[0] and "--speculative" in refused[0]
    print(f"  [{'PASS' if ok else 'FAIL'}] the engine refuses to start: {refused or 'it started'}")
    if not ok:
        print(log)
    failures += not ok

    # Control: without --enable-steering the same speculative setting passes this check.
    code, results, log = run_engine_child(
        "check_args",
        tree,
        engine_settings(tree, workdir, enable_steering=False, speculative_algorithm="EAGLE3"),
    )
    errors = [r["error"] for r in results if r.get("case") == "check_args"]
    clean = len(errors) == 1 and "--enable-steering" not in (errors[0] or "")
    print(f"      control (the same setting without the flag): {errors or 'no result'}")
    if not clean:
        print("      FAIL: the control didn't fail, so the refusal reads nothing.")
        print(log)
        failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 13. placement on a mesh that spans two processes
# --------------------------------------------------------------------------------------


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def multihost_child(argv):
    """One of two processes over one 8-device mesh, 4 devices each, like a 2-host slice."""
    pid, port, tree = int(argv[1]), argv[2], argv[3]
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=2,
        process_id=pid,
        initialization_timeout=120,
    )
    sys.path.insert(0, os.path.join(tree, "python"))
    from jax.experimental import multihost_utils

    from sgl_jax.srt.layers.steering import SteeringBank
    from sgl_jax.srt.utils.jax_utils import device_array

    counts = {"gathers": 0}
    allgather = multihost_utils.process_allgather

    def counting(*args, **kwargs):
        counts["gathers"] += 1
        return allgather(*args, **kwargs)

    multihost_utils.process_allgather = counting

    mesh = Mesh(
        np.array(jax.devices()).reshape(1, 8),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    replicated = NamedSharding(mesh, P())
    rng = np.random.default_rng(0)
    bank = SteeringBank(
        features=np.array([10, 11, 12, 13], np.int32),
        thresholds=np.array([0.5, 0.25, np.inf, 0.1], np.float32),
        vectors=device_array(rng.normal(size=(4, 16)).astype(np.float32), sharding=replicated),
        probes=device_array(rng.normal(size=(4, 16)).astype(np.float32), sharding=replicated),
    )
    with open(os.path.join(tree, FORWARD_BATCH), encoding="utf-8") as fh:
        build = init_new_steering(fh.read(), device_array)
    runner = Stub(steering_bank=bank, mesh=mesh)

    tokens = 32
    feature = np.full(tokens, -1, np.int32)
    alpha = np.zeros(tokens, np.float32)
    threshold = np.full(tokens, np.inf, np.float32)
    feature[:8], alpha[:8], threshold[:8] = 11, 1.5, np.nan  # the fitted threshold, 0.25
    steered = Stub(
        input_ids=np.zeros(tokens, np.int32),
        steering_feature=feature,
        steering_alpha=alpha,
        steering_threshold=threshold,
    )
    quiet = Stub(
        input_ids=np.zeros(tokens, np.int32),
        steering_feature=None,
        steering_alpha=None,
        steering_threshold=None,
    )

    results = {"pid": pid, "case": "init_new"}
    for label, batch in (("steered", steered), ("nobody steers", quiet)):
        before = counts["gathers"]
        steering = build(batch, runner)
        jax.block_until_ready(steering.threshold)
        results[label] = {
            "gathers": counts["gathers"] - before,
            "threshold": np.asarray(steering.threshold.addressable_data(0))[:10].tolist(),
        }
    print("RESULT " + json.dumps(results), flush=True)

    # Control: the resolved arrays through jax.device_put onto the same sharding. device_put
    # compares every host's copy, and NaN never equals itself, so the fitted threshold is filled.
    slot = np.where(feature == 11, 1, 0).astype(np.int32)
    resolved = np.where(np.isnan(threshold), np.float32(0.25), threshold).astype(np.float32)
    before = counts["gathers"]
    placed = jax.device_put((slot, alpha, resolved), NamedSharding(mesh, P("data")))
    jax.block_until_ready(placed)
    control = {"pid": pid, "case": "device_put", "gathers": counts["gathers"] - before}
    print("RESULT " + json.dumps(control), flush=True)
    jax.distributed.shutdown()
    return 0


def run_coupled(argvs, env, workdir, timeout):
    """Run processes that wait on each other, and kill every one still alive at the end.

    Each writes to its own file, so no process blocks on a pipe the parent isn't reading while
    it waits on another. Returns (logs, timed_out).
    """
    files = [open(os.path.join(workdir, f"coupled-{i}.log"), "w+") for i in range(len(argvs))]
    procs = []
    timed_out = False
    try:
        for argv, fh in zip(argvs, files):
            procs.append(subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + timeout
        for proc in procs:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.1))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for proc in procs:
            proc.wait()
    logs = []
    for fh in files:
        fh.seek(0)
        logs.append(fh.read())
        fh.close()
    return logs, timed_out


def check_multihost(tree, workdir, timeout=600):
    """13. init_new places the per-token arrays with no cross-process gather."""
    port = free_port()
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
    )
    argvs = [
        [sys.executable, os.path.abspath(__file__), "--multihost-child", str(pid), str(port), tree]
        for pid in (0, 1)
    ]
    logs, timed_out = run_coupled(argvs, env, workdir, timeout)
    if timed_out:
        print(f"  [FAIL] the two processes ran past {timeout} s and were killed. They printed:")
        for log in logs:
            print(log)
        return 1
    results = {}
    for log in logs:
        for line in log.splitlines():
            if line.startswith("RESULT "):
                r = json.loads(line[len("RESULT ") :])
                results[(r["pid"], r["case"])] = r
    if len(results) != 4:
        print(f"  [FAIL] {len(results)} of 4 results came back. The two processes printed:")
        for log in logs:
            print(log)
        return 1

    failures = 0
    for pid in (0, 1):
        r = results[(pid, "init_new")]
        steered, quiet = r["steered"], r["nobody steers"]
        ok = steered["gathers"] == 0 and quiet["gathers"] == 0
        right = steered["threshold"][:8] == [0.25] * 8 and all(
            t == float("inf") for t in steered["threshold"][8:]
        )
        print(
            f"  [{'PASS' if ok and right else 'FAIL'}] process {pid}: init_new gathered"
            f" {steered['gathers']} time(s) for a steered batch and {quiet['gathers']} for an"
            f" all-off one; thresholds {steered['threshold']}"
        )
        failures += not (ok and right)
        gathers = results[(pid, "device_put")]["gathers"]
        detected = gathers > 0
        print(
            f"      control (jax.device_put on the same arrays): {gathers} gather(s)"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        if not detected:
            print("      FAIL: the control didn't fail, so the gather count reads nothing.")
            failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 14. the qwen3 hook and the deepstack add
# --------------------------------------------------------------------------------------

# The layers that take a deepstack plane, the way Qwen3-VL adds its vision features to its first
# layers, and the layer the hook sits after, which takes one.
DEEPSTACK = 2
QWEN3_STEER_LAYER = 1
QWEN3_FEATURE = 5


class ToyPairLayer:
    """A qwen3 decoder layer's shape: `(hidden, residual)` in and out, with the add deferred.

    The stream is `hidden + residual`, and residual is None on layer 0. The layer hands back
    its update as hidden and the stream it read as residual, so the next reader adds them.
    """

    def __init__(self, weights):
        w, gain = weights
        self.w = jnp.asarray(w, jnp.float32)
        self.gain = jnp.asarray(gain, jnp.float32)

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool, residual):
        stream = hidden_states if residual is None else hidden_states + residual
        update = jnp.tanh(rms_norm(stream, self.gain) @ self.w)
        return update, stream, jnp.zeros((1,), jnp.int32), [jnp.zeros((1,), jnp.int32)]


def reference_qwen3(embed64, weights64, deep64, steering, until=None):
    """The same stack in float64: each layer's update, its deepstack plane, then the hook.

    Returns the stream entering every layer, which is what capture reads. With `until`, it
    returns the stream leaving that layer before its deepstack add instead.
    """
    stream = np.asarray(embed64, np.float64)
    captured = []
    for i, (w, gain) in enumerate(weights64):
        captured.append(stream.copy())
        normed = stream / np.sqrt(np.mean(stream**2, axis=-1, keepdims=True) + EPS) * gain
        stream = stream + np.tanh(normed @ w)
        if i == until:
            return stream
        if i < deep64.shape[0]:
            stream = stream + deep64[i]
        if i == QWEN3_STEER_LAYER and steering is not None:
            stream = reference_steer(stream, steering)
    return np.stack(captured)


def run_qwen3(fn, mesh, embed64, weights64, deep64, steering):
    """Run a `QWen3Model.__call__` on stubs, under jit. Returns the captured layers, float64."""
    embed = jax.device_put(jnp.asarray(embed64, jnp.float32), NamedSharding(mesh, P(AXIS, None)))
    call = compile_function(
        fn,
        {
            "jax": jax,
            "jnp": jnp,
            "apply_steering": STEERING_MODULE["apply_steering"],
            "precision_tracer": Stub(
                jit_pure_callback_record=lambda *a, **k: jnp.zeros((1,), jnp.int32)
            ),
        },
    )
    model = Stub(
        layers=[ToyPairLayer(w) for w in weights64],
        layers_to_capture=list(range(LAYERS)),
        steering_layer=QWEN3_STEER_LAYER,
        embed_tokens=lambda ids: embed,
        norm=Stub(_call=lambda s, x: rms_norm(x, jnp.ones((DIM,), jnp.float32))),
    )
    forward_batch = Stub(
        input_ids=jnp.zeros((TOKENS,), jnp.int32),
        positions=jnp.arange(TOKENS, dtype=jnp.int32),
        input_embedding=None,
        deepstack_visual_embedding=jnp.asarray(deep64, jnp.float32),
        apply_for_deepstack=jnp.asarray(True),
        steering=steering,
    )
    result = jax.jit(lambda: call(model, forward_batch, "kv-pool"))()
    return np.stack([np.asarray(h, np.float64) for h in result[1]])


def deepstack_index(loop):
    """Index in `loop.body` of the `if` that adds the deepstack plane."""
    for i, stmt in enumerate(loop.body):
        if isinstance(stmt, ast.If) and "deepstack" in ast.dump(stmt.test):
            return i
    raise LookupError("no deepstack add in the layer loop")


def m_steer_before_deepstack(fn):
    """The hook moved above the deepstack add, where the patch first had it."""
    loop = layer_loop(fn)
    gate = loop.body.pop(steer_gate_index(loop))
    loop.body.insert(deepstack_index(loop), gate)


def check_qwen3_deepstack(sources, mesh):
    """14. The qwen3 hook steers after the deepstack add, on the stream capture reads next.

    Conditional steering reads a probe. The plane added to the image tokens at the hook's layer
    moves their projection past the threshold, and no token passes it before the add. So the
    hook fires on the image tokens only when it reads the stream after the add, the stream the
    next capture slot holds and an SAE trained on that slot saw.
    """
    failures = 0
    rng = np.random.default_rng(41)
    weights64 = [
        (rng.normal(size=(DIM, DIM)) * 0.1, 1.0 + rng.normal(size=(DIM,)) * 0.05)
        for _ in range(LAYERS)
    ]
    embed64 = rng.normal(size=(TOKENS, DIM))
    vector = _unit(rng.normal(size=DIM))
    probe = _unit(rng.normal(size=DIM))

    image = np.arange(TOKENS) < TOKENS // 2
    deep64 = np.zeros((DEEPSTACK, TOKENS, DIM))
    deep64[0, image] = rng.normal(size=(int(image.sum()), DIM)) * 0.1
    before_add = reference_qwen3(embed64, weights64, deep64, None, until=QWEN3_STEER_LAYER)
    projections = before_add @ probe
    bar = float(projections.max()) + 0.5
    deep64[QWEN3_STEER_LAYER, image] = (projections.max() - projections.min() + 1.0) * probe

    replicated = NamedSharding(mesh, P())
    bank = STEERING_MODULE["SteeringBank"](
        features=np.array([QWEN3_FEATURE], np.int32),
        thresholds=np.array([0.0], np.float32),
        vectors=jax.device_put(jnp.asarray(vector[None, :], jnp.float32), replicated),
        probes=jax.device_put(jnp.asarray(probe[None, :], jnp.float32), replicated),
    )
    steering = steering_batch(
        bank,
        np.full(TOKENS, QWEN3_FEATURE, np.int32),
        np.full(TOKENS, 3.0, np.float32),
        np.full(TOKENS, bar, np.float32),
        NamedSharding(mesh, P(AXIS)),
    )
    want = reference_qwen3(embed64, weights64, deep64, steering)
    quiet = reference_qwen3(embed64, weights64, deep64, None)
    call = find_method(sources["after"][QWEN3], "QWen3Model", "__call__")

    got = run_qwen3(call, mesh, embed64, weights64, deep64, steering)
    rel = relative(got, want)
    ok = rel < F32_TOL
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the qwen3 hook steers the stream after the deepstack"
        f" add: rel={rel:.3e}"
    )
    failures += not ok

    # Control: the steering moves the captured stream, so a hook that fired on nothing shows.
    moved = relative(quiet, want)
    detected = not moved < MUTANT_TOL
    print(
        f"      control (the same stack unsteered): rel={moved:.3e}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so this check reads nothing.")
        failures += 1

    # Control: the hook above the deepstack add reads the stream before it, where no token
    # passes the threshold.
    try:
        early = mutated(call, m_steer_before_deepstack)
    except Exception as exc:  # noqa: BLE001 - an operator that can't apply tested nothing
        return failures + operator_missed("the hook above the deepstack add", exc)
    try:
        rel = relative(run_qwen3(early, mesh, embed64, weights64, deep64, steering), want)
        detected = not rel < MUTANT_TOL
        how = f"rel={rel:.3e}"
    except Exception as exc:  # noqa: BLE001 - a mutant that raises is caught
        detected, how = True, f"raised {type(exc).__name__}"
    print(
        f"      control (the hook above the deepstack add): {how}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control didn't fail, so this mutation is untested.")
        failures += 1
    return failures


SOURCES_AFTER = {}


def main() -> int:
    global STEERING_MODULE, DEVICE_ARRAY
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAILED: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))

    workdir = tempfile.mkdtemp(prefix="steering-hook-")
    try:
        repo = get_checkout(workdir)
        head = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        print(f"sglang-jax at {head}")
        print(
            f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}   tokens={TOKENS}"
            f" dim={DIM} layers={LAYERS} steering_layer={STEER_LAYER}\n"
        )

        print("1. the patch applies and compiles")
        failures = check_patch(repo, workdir)
        sources = collect_sources(repo)

        STEERING_MODULE = load_module(sources["after"][STEERING], "sgl_steering")
        DEVICE_ARRAY = load_module(sources["jax_utils"], "sgl_jax_utils")["device_array"]
        SOURCES_AFTER.update(sources["after"])

        print("\n2. the parts of the hook")
        failures += check_structure(sources)

        print("\n3. the patched Gemma4Model runs")
        failures += check_model(sources, mesh)

        print("\n4. the shipped arithmetic")
        failures += check_arithmetic(sources, mesh)

        print("\n5. the scheduler's per-token rules")
        failures += check_scheduler(sources)

        print("\n6. the per-token fields reach the worker batch")
        failures += check_worker_batch(sources)

        print("\n7. one executable covers steered and unsteered")
        failures += check_one_executable(sources, mesh)

        print("\n8. the request validator")
        failures += check_validation(sources)

        print("\n9. the bank gather under explicit sharding")
        failures += check_explicit_sharding()

        print("\n10. the cache key")
        failures += check_cache_key(sources)

        tree = build_stacked_tree(repo)

        print("\n11. the radix cache on the real Engine")
        failures += check_engine_cache(tree, workdir)

        print("\n12. the real Engine refuses steering with speculative decoding")
        failures += check_engine_refusal(tree, workdir)

        print("\n13. placement on a mesh that spans two processes")
        failures += check_multihost(tree, workdir)

        print("\n14. the qwen3 hook and the deepstack add")
        failures += check_qwen3_deepstack(sources, mesh)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    if "--engine-child" in sys.argv:
        raise SystemExit(engine_child(sys.argv[1:]))
    if "--multihost-child" in sys.argv:
        raise SystemExit(multihost_child(sys.argv[1:]))
    raise SystemExit(main())
