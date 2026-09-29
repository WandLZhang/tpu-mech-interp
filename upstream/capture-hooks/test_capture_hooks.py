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

"""Correctness gate for the four capture-hook patches.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 upstream/capture-hooks/test_capture_hooks.py

Six checks. Every check runs the patched source, not a retyped copy of it. The hooks go on the
tree `scripts/bootstrap_tpu_vm.sh` builds, which is the tree the TPU VM serves: `sglang-jax` at
`SGL_COMMIT`, `sglang-jax-877.patch` through `git am`, then `steering-hook.patch` and
`qwen3-steering-hook.patch` through `git apply`.

1. Each patch applies to that tree, and every file it touches compiles.
2. Each patched file carries the parts of the hook, read out of its AST.
3. The patched `__call__` methods run, under `jit`, on a sharded input, at float32 and at
   bfloat16, against a float64 reference. A NaN capture fails. Nine mutants of the same patched
   source must fail, and a mutation operator that finds no target fails the run.
4. Without the hook, `--enable-return-hidden-states` refuses to start, on the real Engine.
5. `sglang-jax-877.patch` connects all four handoffs between `req.hidden_states` and
   `meta_info["hidden_states"]`.
6. `sglang-jax-877.patch` passes its own CPU test, `test/srt/test_return_hidden_states_cpu.py`,
   which runs the real Engine with a tiny random Qwen3, and the patched tree passes upstream's
   unit tests for the code the patch changes. Checks 4 and 6 need the packages `sgl_jax`
   imports at startup.

Every check carries a negative control. A control that passes fails the run.

`SGL_COMMIT` picks the `sglang-jax` commit, `eb061d8` by default, the way it does for
`scripts/cpu_engine.py`, `scripts/verify_patches.sh` and `scripts/bootstrap_tpu_vm.sh`. Point
`SGLANG_JAX_REPO` at a checkout that holds that commit to skip the fetch.
"""

from __future__ import annotations

import ast
import copy
import os
import shutil
import signal
import subprocess
import sys
import tempfile

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

UPSTREAM = "https://github.com/sgl-project/sglang-jax"
# The commit every patch in upstream/ targets. The SGL_COMMIT environment variable moves it.
SGL_COMMIT = os.environ.get("SGL_COMMIT") or "eb061d8b154056e0f07eef07cd6f98047dfcf4a7"
HERE = os.path.dirname(os.path.abspath(__file__))
AXIS = "token"

MODELS = "python/sgl_jax/srt/models"
KIMI_K25 = "python/sgl_jax/srt/multimodal/models/kimi_k25/kimi_k25_vl_generation.py"

MANAGERS = "python/sgl_jax/srt/managers"
MIXIN = f"{MANAGERS}/scheduler_output_processor_mixin.py"
DETOKENIZER = f"{MANAGERS}/detokenizer_manager.py"
TOKENIZER = f"{MANAGERS}/tokenizer_manager.py"
PATCH_877 = os.path.join(HERE, os.pardir, "sglang-jax-877.patch")
# The two patches bootstrap_tpu_vm.sh applies on top of 877, in its order.
STEERING_PATCHES = [
    os.path.join(HERE, os.pardir, "steering-hook.patch"),
    os.path.join(HERE, os.pardir, "qwen3-steering-hook.patch"),
]

# Every one of these models defaults to bfloat16, so the capture runs at both dtypes and reports
# both. float32 is the gate, because it separates a defect from rounding.
F32_TOL = 1e-5
BF16_TOL = 5e-2
# Mutants run at float32, where the clean signal is 2e-07. This sits five orders of magnitude
# above that. The smallest numeric miss any mutant produces, 2.7e-01, sits 27 times above it.
MUTANT_TOL = 1e-2

EPS = 1e-6

SPECS = [
    {
        "patch": "kimi-linear-capture-hook.patch",
        "serves": "Kimi-Linear-48B-A3B",
        "files": [f"{MODELS}/kimi_linear.py"],
        "inner": (f"{MODELS}/kimi_linear.py", "KimiModel"),
        "entries": [(f"{MODELS}/kimi_linear.py", "KimiLinearForCausalLM")],
        "nested": False,
        "layer_attrs": {"is_kda": False},
    },
    {
        "patch": "qwen3_5-capture-hook.patch",
        "serves": "Qwen3.5-35B-A3B",
        "files": [f"{MODELS}/qwen3_5.py"],
        "inner": (f"{MODELS}/qwen3_5.py", "Qwen3_5MoeModel"),
        "entries": [(f"{MODELS}/qwen3_5.py", "Qwen3_5MoeForConditionalGeneration")],
        "middle": (f"{MODELS}/qwen3_5.py", "Qwen3_5MoeForCausalLM"),
        "nested": True,
        "layer_attrs": {"is_full_attn": True},
    },
    {
        "patch": "deepseek-v3-capture-hook.patch",
        "serves": "DeepSeek V3, Kimi K2.5 VL",
        "files": [f"{MODELS}/deepseek_v3.py", KIMI_K25],
        "inner": (f"{MODELS}/deepseek_v3.py", "DeepseekV3Model"),
        "entries": [
            (f"{MODELS}/deepseek_v3.py", "DeepseekV3ForCausalLM"),
            (KIMI_K25, "KimiK25ForConditionalGeneration"),
        ],
        "nested": False,
        "layer_attrs": {},
    },
    {
        "patch": "glm4-moe-capture-hook.patch",
        "serves": "GLM-4.5, GLM-4.6",
        "files": [f"{MODELS}/glm4_moe.py"],
        "inner": (f"{MODELS}/glm4_moe.py", "Glm4MoeModel"),
        "entries": [(f"{MODELS}/glm4_moe.py", "Glm4MoeForCausalLM")],
        "nested": False,
        "layer_attrs": {},
    },
]


# --------------------------------------------------------------------------------------
# checkout plumbing
# --------------------------------------------------------------------------------------


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
    )


def run_git_steps(steps):
    """Run each git argument list in order. The first that fails raises with git's whole output."""
    for args in steps:
        done = subprocess.run(["git", *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed with exit {done.returncode}:\n"
                f"{done.stdout}{done.stderr}"
            )


def get_checkout(workdir):
    """A clean worktree at `SGL_COMMIT`, the commit the patches target.

    A local checkout is cloned with `--shared` and moved to `SGL_COMMIT`, whatever branch it
    has out, so it needs that commit in its history. Otherwise a blobless clone of upstream,
    moved the same way. Either way `SGL_COMMIT` can be a full or short hash or a branch name. A
    git step that fails raises with git's own error.
    """
    src = os.environ.get("SGLANG_JAX_REPO")
    dest = os.path.join(workdir, "sglang-jax")
    if src and os.path.isdir(os.path.join(src, ".git")):
        clone = ["clone", "-q", "--shared", src, dest]
    else:
        clone = ["clone", "-q", "--filter=blob:none", UPSTREAM, dest]
    run_git_steps([clone, ["-C", dest, "checkout", "-q", SGL_COMMIT]])
    return dest


def stack_patches(repo):
    """Build the tree `scripts/bootstrap_tpu_vm.sh` builds, in `repo`, a clean checkout.

    `sglang-jax-877.patch` goes in through `git am`, then both steering patches through
    `git apply`, in the bootstrap's order. A commit on top holds the files the steering patch
    creates, so `git checkout -- .` and `git clean -qfd` come back to this tree. Returns the
    commit the stack sits on. A git step that fails raises with git's own error.
    """
    head = git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed in {repo}: {head.stderr}")
    who = ["-c", "user.email=capture-hooks@local", "-c", "user.name=capture-hooks"]
    message = "sglang-jax-877.patch and the steering patches"
    run_git_steps(
        [
            ["-C", repo, *who, "am", "-q", PATCH_877],
            *(["-C", repo, "apply", patch] for patch in STEERING_PATCHES),
            ["-C", repo, "add", "-A"],
            ["-C", repo, *who, "commit", "-q", "-m", message],
        ]
    )
    return head.stdout.strip()


def read_files(repo, rels, rev=None):
    """{rel: text} for each path, from the worktree, or from commit `rev` when it's given."""
    out = {}
    for rel in rels:
        if rev is None:
            with open(os.path.join(repo, rel), encoding="utf-8") as fh:
                out[rel] = fh.read()
        else:
            shown = git(repo, "show", f"{rev}:{rel}")
            if shown.returncode != 0:
                raise RuntimeError(f"git show {rev}:{rel} failed: {shown.stderr}")
            out[rel] = shown.stdout
    return out


def patch_files(repo, patch):
    """Every path `patch` touches, as `git apply --numstat` lists them."""
    listed = git(repo, "apply", "--numstat", patch)
    if listed.returncode != 0:
        raise RuntimeError(f"git apply --numstat {patch} failed: {listed.stderr}")
    return [line.split("\t", 2)[2] for line in listed.stdout.splitlines() if line]


def kill_group(pgid):
    """SIGKILL every process left in group `pgid`. A group with no process left is fine."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_to_end(cmd, timeout, **kwargs):
    """`subprocess.run(cmd, capture_output=True, text=True)` that leaves no process behind.

    The 877 patch's CPU test starts each Engine in a child, and each Engine starts processes of
    its own. `subprocess.run` kills only the process it started when the timeout fires, so the
    rest would keep running. This starts `cmd` in a session of its own and kills that whole
    process group when `cmd` ends, on a timeout, on Ctrl-C, or on any other exception.
    """
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        **kwargs,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        finally:
            kill_group(proc.pid)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def corrupt(patch_text):
    """The same patch with one context line rewritten.

    `git apply --check` has to refuse this. If it accepts it, `--check` isn't reading the hunk
    bodies and check 1 proves nothing. Reapplying a patch to an already-patched tree doesn't
    work as a control, because git refuses any non-empty patch there whatever it contains.
    """
    lines = patch_text.splitlines(keepends=True)
    in_hunk = False
    for i, line in enumerate(lines):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and line.startswith(" ") and len(line.strip()) > 8:
            lines[i] = "         self.__control__ = None\n"
            return "".join(lines)
    raise AssertionError("no context line to corrupt")


def check_patches(repo, workdir):
    """Apply each patch to the stack, compile every file it touches, and refuse a corrupted copy.

    The files to compile come from the patch itself, through `git apply --numstat`, so a hunk
    in a file SPECS doesn't name still has to compile.
    """
    failures = 0
    for spec in SPECS:
        name = spec["patch"]
        patch = os.path.join(HERE, name)
        dry = git(repo, "apply", "--check", patch)
        ok = dry.returncode == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  applies to the stack")
        if not ok:
            print(f"      {dry.stderr.strip()}")
            failures += 1
            continue

        touched = patch_files(repo, patch)
        applied = git(repo, "apply", patch)
        if applied.returncode != 0:
            print(f"      FAIL: apply failed after --check passed: {applied.stderr.strip()}")
            failures += 1
            continue

        for rel in touched:
            if not rel.endswith(".py"):
                print(f"      {rel}: not Python, nothing to compile")
                continue
            path = os.path.join(repo, rel)
            compiled = subprocess.run(
                [sys.executable, "-m", "py_compile", path], capture_output=True, text=True
            )
            good = compiled.returncode == 0
            print(f"      py_compile {os.path.basename(rel)}: {'ok' if good else 'FAILED'}")
            if not good:
                print(f"      {compiled.stderr.strip()}")
                failures += 1

        git(repo, "checkout", "--", ".")
        git(repo, "clean", "-qfd")

        with open(patch, encoding="utf-8") as fh:
            bad_text = corrupt(fh.read())
        bad_path = os.path.join(workdir, f"corrupt-{name}")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write(bad_text)
        bad = git(repo, "apply", "--check", bad_path)
        refused = bad.returncode != 0
        print(f"      control (one context line rewritten): {'refused' if refused else 'ACCEPTED'}")
        if not refused:
            print("      FAIL: the control didn't fail, so --check reads nothing.")
            failures += 1
    return failures


def collect_sources(repo):
    """The text of every file each patch touches, on the stack and with the patch on top.

    Returns {patch_name: {"before": {rel: src}, "after": {rel: src}}}. Raises if a patch stops
    applying, so a rejected apply can't be reported as a missing hook.
    """
    out = {}
    for spec in SPECS:
        name = spec["patch"]
        patch = os.path.join(HERE, name)
        before = read_files(repo, spec["files"])

        applied = git(repo, "apply", patch)
        if applied.returncode != 0:
            raise RuntimeError(f"{name} no longer applies: {applied.stderr.strip()}")

        after = read_files(repo, spec["files"])
        git(repo, "checkout", "--", ".")
        git(repo, "clean", "-qfd")
        out[name] = {"before": before, "after": after}
    return out


# --------------------------------------------------------------------------------------
# AST surgery: pull the patched methods out and run them
# --------------------------------------------------------------------------------------


def find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_method(src, cls_name, fn_name):
    cls = find_class(ast.parse(src), cls_name)
    if cls is None:
        raise LookupError(f"class {cls_name} not found")
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == fn_name:
            return copy.deepcopy(item)
    raise LookupError(f"{cls_name}.{fn_name} not found")


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


def compile_method(fn):
    """Turn a patched method into a plain function taking `self` first.

    The body is the shipped text. Only the signature annotations come off, because they name
    types the rest of `sgl_jax` owns and this test doesn't import.
    """
    fn = strip_annotations(copy.deepcopy(fn))
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"jax": jax, "jnp": jnp, "np": np, "rmsnorm_forward": rmsnorm_forward}
    exec(compile(module, f"<patched:{fn.name}>", "exec"), namespace)  # noqa: S102
    return namespace[fn.name]


def is_self_attr(node, attr):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def capture_loop(node):
    """The `for layer_id, layer in enumerate(self.layers)` loop, or None."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.For):
            continue
        iter_ = sub.iter
        if not (isinstance(iter_, ast.Call) and isinstance(iter_.func, ast.Name)):
            continue
        if iter_.func.id != "enumerate" or len(iter_.args) != 1:
            continue
        if not is_self_attr(iter_.args[0], "layers"):
            continue
        if not (isinstance(sub.target, ast.Tuple) and len(sub.target.elts) == 2):
            continue
        if not isinstance(sub.target.elts[0], ast.Name):
            continue
        return sub
    return None


def gate_index(loop):
    """Index in `loop.body` of the `if layer_id in self.layers_to_capture:` block."""
    index_name = loop.target.elts[0].id
    for i, stmt in enumerate(loop.body):
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
            continue
        if not isinstance(test.ops[0], ast.In):
            continue
        if not (isinstance(test.left, ast.Name) and test.left.id == index_name):
            continue
        if is_self_attr(test.comparators[0], "layers_to_capture"):
            return i
    return None


def layer_call_index(loop):
    """Index in `loop.body` of the statement that invokes `layer(...)`."""
    for i, stmt in enumerate(loop.body):
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "layer"
            ):
                return i
    return None


def find_append(gate):
    for sub in ast.walk(gate):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "aux_hidden_states"
        ):
            return sub
    return None


def return_tuple(fn):
    """The last `return` of `fn`, if it returns a tuple."""
    found = None
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
            found = sub.value
    return found


def flag_gate_index(fn):
    """Index in `fn.body` of `if not self.capture_aux_hidden_states: ...`."""
    for i, stmt in enumerate(fn.body):
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            if is_self_attr(test.operand, "capture_aux_hidden_states"):
                return i
    return None


# --------------------------------------------------------------------------------------
# mutants of the patched source
# --------------------------------------------------------------------------------------


# Every operator finds its target through the locators below, and a locator that comes back empty
# raises LookupError. run_mutant counts that as a failure. An operator that quietly changes
# nothing would otherwise score a catch for a mutation it never made.


def need(found, what):
    """`found`, or LookupError naming `what` when a locator came back empty."""
    if found is None:
        raise LookupError(f"no {what}")
    return found


def target_gate(inner):
    """The capture loop and the index of its `if layer_id in self.layers_to_capture:` block."""
    loop = need(capture_loop(inner), "`for layer_id, layer in enumerate(self.layers)` loop")
    return loop, need(gate_index(loop), "`if layer_id in self.layers_to_capture:` block")


def target_append(inner):
    """The `aux_hidden_states.append(...)` call inside the gate."""
    loop, gi = target_gate(inner)
    return need(find_append(loop.body[gi]), "`aux_hidden_states.append(...)` in the gate")


def target_none_check(inner):
    """The append and its `hidden + residual if residual is not None else hidden` argument."""
    append = target_append(inner)
    if not (append.args and isinstance(append.args[0], ast.IfExp)):
        raise LookupError("no `... if residual is not None else ...` in the append")
    return append, append.args[0]


def target_returned_aux(inner):
    """The returned tuple and its `aux_hidden_states` elements."""
    tup = need(return_tuple(inner), "tuple return")
    aux = [e for e in tup.elts if isinstance(e, ast.Name) and e.id == "aux_hidden_states"]
    if not aux:
        raise LookupError("no aux_hidden_states in the returned tuple")
    return tup, aux


def m_append_after_layer(inner, entries):
    loop, gi = target_gate(inner)
    li = need(layer_call_index(loop), "`layer(...)` call in the loop")
    if gi > li:
        raise LookupError("the gate already sits below the `layer(...)` call")
    gate = loop.body.pop(gi)
    loop.body.insert(li, gate)


def m_gate_inverted(inner, entries):
    loop, gi = target_gate(inner)
    loop.body[gi].test.ops[0] = ast.NotIn()


def m_no_none_check(inner, entries):
    append, ifexp = target_none_check(inner)
    append.args[0] = ifexp.body


def m_none_check_inverted(inner, entries):
    """`residual if residual is None else hidden + residual`: the arms swapped."""
    _, ifexp = target_none_check(inner)
    ifexp.body, ifexp.orelse = ifexp.orelse, ifexp.body


def m_never_returned(inner, entries):
    tup, aux = target_returned_aux(inner)
    tup.elts = [e for e in tup.elts if e not in aux]


def m_returned_last(inner, entries):
    """`aux_hidden_states` moved to the end of the tuple, so the outer unpack silently swaps."""
    tup, aux = target_returned_aux(inner)
    rest = [e for e in tup.elts if e not in aux]
    if tup.elts == rest + aux:
        raise LookupError("aux_hidden_states is already last in the returned tuple")
    tup.elts = rest + aux


def m_flag_ignored(inner, entries):
    for fn in entries:
        i = need(flag_gate_index(fn), "`if not self.capture_aux_hidden_states:` in an entry class")
        fn.body.pop(i)


def m_keyword_dropped(inner, entries):
    for fn in entries:
        calls = [
            sub
            for sub in ast.walk(fn)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "logits_processor"
            and any(kw.arg == "aux_hidden_states" for kw in sub.keywords)
        ]
        if not calls:
            raise LookupError("no `logits_processor(..., aux_hidden_states=...)` in an entry class")
        for call in calls:
            call.keywords = [kw for kw in call.keywords if kw.arg != "aux_hidden_states"]


def m_list_never_filled(inner, entries):
    """The gate stays, the append goes."""
    target_append(inner)
    loop, gi = target_gate(inner)
    loop.body[gi].body = [ast.Pass()]


def m_capture_nan(inner, entries):
    """Every captured value times NaN. Check 3's comparisons have to flag it."""
    append = target_append(inner)
    nan = ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="nan", ctx=ast.Load())
    append.args[0] = ast.BinOp(left=append.args[0], op=ast.Mult(), right=nan)


MUTANTS = {
    "append moved below the layer call": m_append_after_layer,
    "gate inverted to `not in`": m_gate_inverted,
    "None check dropped, bare `h + residual`": m_no_none_check,
    "None check inverted": m_none_check_inverted,
    "aux dropped from the return tuple": m_never_returned,
    "aux moved to the end of the tuple": m_returned_last,
    "capture_aux_hidden_states gate removed": m_flag_ignored,
    "aux_hidden_states= keyword removed": m_keyword_dropped,
    "append removed, gate kept": m_list_never_filled,
}


# --------------------------------------------------------------------------------------
# stubs the patched methods run against
# --------------------------------------------------------------------------------------


def rmsnorm_forward(x, residual, weight, epsilon):
    """The upstream `layers/layernorm.py` contract: add in the input dtype, reduce in float32."""
    orig_dtype = x.dtype
    x_f32 = jnp.asarray(x, jnp.float32)
    if residual is not None:
        x_f32 = x_f32 + jnp.asarray(residual, jnp.float32)
        residual = x_f32.astype(orig_dtype)
    mean2 = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    y = x_f32 * jax.lax.rsqrt(mean2 + epsilon)
    output = (y * jnp.asarray(weight, jnp.float32)).astype(orig_dtype)
    return output if residual is None else (output, residual)


def fused_norm(x, residual, weight):
    """One pre-norm step of a decoder layer, matching `DeepseekV3DecoderLayer`.

    The add happens in the layer dtype. The reduction happens in float32, the way every RMSNorm
    in these four models does it. Returns `(normed, residual)`.
    """
    stream = x if residual is None else x + residual
    normed = rmsnorm_forward(stream, None, weight, EPS)
    return normed, stream


class Stub:
    """An object whose attributes are whatever the patched method reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __call__(self, *args, **kw):
        return self.__dict__["_call"](self, *args, **kw)


class ToyLayer:
    """A pre-norm block with the return contract every one of these models' layers uses.

    `residual` comes back as the stream after attention. `hidden` comes back as the MLP output.
    The stream entering the next layer is their sum, which is what the hook captures.
    """

    def __init__(self, weights, attrs):
        self.w_attn, self.w_mlp, self.g_in, self.g_post = weights
        self.__dict__.update(attrs)

    def __call__(self, positions, hidden, forward_batch, pool, residual, **kw):
        hidden, residual = fused_norm(hidden, residual, self.g_in)
        hidden = (hidden @ self.w_attn).astype(hidden.dtype)
        hidden, residual = fused_norm(hidden, residual, self.g_post)
        hidden = jnp.tanh(hidden @ self.w_mlp).astype(hidden.dtype)
        return hidden, residual, jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.int32)


LOGITS = "logits"


def make_weights(rng, num_layers, dim):
    """float64 weights. Every run casts down from these, so all dtypes share one reference."""
    return [
        (
            rng.normal(size=(dim, dim)) * 0.1,
            rng.normal(size=(dim, dim)) * 0.1,
            1.0 + rng.normal(size=(dim,)) * 0.05,
            1.0 + rng.normal(size=(dim,)) * 0.05,
        )
        for _ in range(num_layers)
    ]


def reference_stream(hidden64, weights64, gate):
    """The residual stream entering each gated layer, in float64, written as one running value.

    Deliberately not the `(hidden, residual)` split the models carry. If the reference replayed
    the same two operands in the same order, it would agree bit for bit at every dtype and the
    tolerance would measure nothing.
    """
    stream = hidden64
    out = []
    for i, (w_attn, w_mlp, g_in, g_post) in enumerate(weights64):
        if i in gate:
            out.append(stream)
        normed = stream / np.sqrt(np.mean(stream**2, axis=-1, keepdims=True) + EPS) * g_in
        stream = stream + normed @ w_attn
        normed = stream / np.sqrt(np.mean(stream**2, axis=-1, keepdims=True) + EPS) * g_post
        stream = stream + np.tanh(normed @ w_mlp)
    return np.stack(out) if out else np.zeros((0,) + hidden64.shape)


def build_batch(mesh, tokens, dim, dtype, embed64):
    embed = jax.device_put(
        jnp.asarray(embed64, dtype), NamedSharding(mesh, P(AXIS, None))
    )
    mask = jnp.ones((tokens,), dtype=bool)
    forward_batch = Stub(
        input_ids=jnp.zeros((tokens,), dtype=jnp.int32),
        positions=jnp.arange(tokens, dtype=jnp.int32),
        mrope_positions=None,
        input_embedding=None,
        expert_location_metadata=None,
        forward_mode=Stub(is_extend_or_draft_extend_or_mixed=lambda: False),
        get_token_valid_mask=lambda n: mask,
    )
    return forward_batch, embed


def build_model(spec, sources, dtype, weights64, mutate=None):
    """A runnable inner model and its entry classes, built from the patched text."""
    after = sources["after"]
    inner_file, inner_name = spec["inner"]
    inner_fn = find_method(after[inner_file], inner_name, "__call__")
    entry_fns = [find_method(after[f], c, "__call__") for f, c in spec["entries"]]

    if mutate is not None:
        mutate(inner_fn, entry_fns)

    inner_call = compile_method(inner_fn)
    entry_calls = [compile_method(fn) for fn in entry_fns]

    layers = [
        ToyLayer(
            tuple(jnp.asarray(w, dtype if w.ndim == 2 else jnp.float32) for w in ws),
            spec["layer_attrs"],
        )
        for ws in weights64
    ]
    return inner_call, entry_calls, layers


def run_inner(spec, inner_call, layers, embed, forward_batch, gate):
    inner = Stub(
        layers=layers,
        layers_to_capture=list(gate),
        embed_tokens=lambda ids: embed,
        norm=Stub(
            _call=lambda s, x: rmsnorm_forward(x, None, s.scale, s.epsilon),
            scale=jnp.ones(embed.shape[-1], jnp.float32),
            epsilon=EPS,
        ),
    )
    pool = Stub(token_to_kv_pool="kv-pool")
    result = jax.jit(lambda: inner_call(inner, forward_batch, pool))()
    return inner, result


def rel_error(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return float("inf")
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-12))


def within(rel, tol):
    """True only when `rel` is a number below `tol`.

    A NaN anywhere in a capture makes `rel_error` NaN, and NaN compares False with everything.
    So every gate reads `within(rel, tol)`, which fails a NaN capture, and none reads
    `rel >= tol`, which passes one.
    """
    return rel < tol


def aux_from(result):
    """`aux_hidden_states` is the second element of the inner model's tuple."""
    return result[1]


# --------------------------------------------------------------------------------------
# 2. the parts of the hook, read out of the AST
# --------------------------------------------------------------------------------------


def assigns_self_attr(node, attr):
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for target in sub.targets:
            if is_self_attr(target, attr):
                return True
    return False


def inspect_sources(spec, sources):
    """Which parts of the hook the text carries. Raises if a class named in SPECS is missing."""
    inner_file, inner_name = spec["inner"]
    tree = ast.parse(sources[inner_file])
    inner = find_class(tree, inner_name)
    if inner is None:
        raise LookupError(f"{inner_name} not found in {inner_file}")

    entries = []
    for rel, cls_name in spec["entries"]:
        cls = find_class(ast.parse(sources[rel]), cls_name)
        if cls is None:
            raise LookupError(f"{cls_name} not found in {rel}")
        entries.append(cls)

    loop = capture_loop(inner)
    gate = None if loop is None else gate_index(loop)
    call_at = None if loop is None else layer_call_index(loop)
    tup = return_tuple(find_method(sources[inner_file], inner_name, "__call__"))

    parts = {
        "list": assigns_self_attr(inner, "layers_to_capture"),
        "gate": gate is not None,
        "above": gate is not None and call_at is not None and gate < call_at,
        "returned": tup is not None
        and len(tup.elts) > 1
        and isinstance(tup.elts[1], ast.Name)
        and tup.elts[1].id == "aux_hidden_states",
        "flag": all(sets_flag(c, spec, sources) for c in entries),
        "threaded": all(threads_aux(c) for c in entries),
    }
    if spec["nested"]:
        parts["property"] = has_backbone_property(inner_file, spec, sources)
    return parts


def sets_flag(cls, spec, sources):
    """`capture_aux_hidden_states` on the class, or on a base class the patch also touches."""
    if assigns_self_attr(cls, "capture_aux_hidden_states"):
        return True
    bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    for rel, name in spec["entries"]:
        if name not in bases:
            continue
        base = find_class(ast.parse(sources[rel]), name)
        if base is not None and assigns_self_attr(base, "capture_aux_hidden_states"):
            return True
    return False


def threads_aux(cls):
    """Every `self.logits_processor(...)` in the class passes `aux_hidden_states`."""
    found = 0
    for sub in ast.walk(cls):
        if not isinstance(sub, ast.Call):
            continue
        if not (isinstance(sub.func, ast.Attribute) and sub.func.attr == "logits_processor"):
            continue
        found += 1
        if not any(kw.arg == "aux_hidden_states" for kw in sub.keywords):
            return False
    return found > 0


def has_backbone_property(inner_file, spec, sources):
    rel, cls_name = spec["entries"][0]
    cls = find_class(ast.parse(sources[rel]), cls_name)
    for item in cls.body:
        if not (isinstance(item, ast.FunctionDef) and item.name == "model"):
            continue
        names = [d.id for d in item.decorator_list if isinstance(d, ast.Name)]
        if "property" in names:
            return True
    return False


def check_structure(collected):
    failures = 0
    for spec in SPECS:
        name = spec["patch"]
        inner_name = spec["inner"][1]
        try:
            after = inspect_sources(spec, collected[name]["after"])
            before = inspect_sources(spec, collected[name]["before"])
        except LookupError as exc:
            print(f"  [FAIL] {inner_name}  {exc}")
            failures += 1
            continue

        ok = all(after.values())
        parts = " ".join(f"{k}={'y' if v else 'n'}" for k, v in after.items())
        print(f"  [{'PASS' if ok else 'FAIL'}] {inner_name}  {parts}")
        if not ok:
            failures += 1

        detected = not any(before.values())
        stale = " ".join(k for k, v in before.items() if v)
        print(
            f"      control (unpatched text): "
            f"{'no hook present' if detected else 'ALREADY HAS ' + stale}"
        )
        if not detected:
            print("      FAIL: the control didn't fail, so this check detects nothing.")
            failures += 1
    return failures


# --------------------------------------------------------------------------------------
# 3. the patched methods run
# --------------------------------------------------------------------------------------


def eagle3_gate(n):
    return [2, n // 2, n - 3]


def check_semantics(collected, mesh):
    failures = 0
    rng = np.random.default_rng(7)
    num_layers, tokens, dim = 12, 64, 32
    weights64 = make_weights(rng, num_layers, dim)
    embed64 = rng.normal(size=(tokens, dim))
    setup = (mesh, weights64, embed64, num_layers, tokens, dim)

    for spec in SPECS:
        name = spec["patch"]
        sources = collected[name]
        inner_name = spec["inner"][1]
        problems, _ = capture_problems(spec, sources, *setup)
        if spec["nested"]:
            problems += check_backbone_property(spec, sources)

        ok = not problems
        print(f"  [{'PASS' if ok else 'FAIL'}] {inner_name} runs as patched")
        for problem in problems:
            print(f"      {problem}")
        failures += 0 if ok else 1

        failures += check_nan_capture(spec, sources, *setup)
        failures += check_mutants(spec, sources, *setup)
    return failures


def capture_problems(
    spec, sources, mesh, weights64, embed64, num_layers, tokens, dim, mutate=None, verbose=True
):
    """Run the patched methods at each setting check 3 reads, and list what's wrong.

    Returns (problems, comparisons). comparisons holds a (label, rel, tol) for each captured
    stream checked against the float64 reference, and each one that isn't within tol is a
    problem too. `mutate` rewrites the patched source first, the way the NaN control does.
    """
    say = print if verbose else (lambda *args: None)
    all_layers = list(range(num_layers))
    problems, comparisons = [], []

    def compare(label, got, want, tol):
        rel = rel_error(got, want)
        comparisons.append((label, rel, tol))
        if not within(rel, tol):
            problems.append(f"{label} rel={rel:.3e}")
        return rel

    for dtype, tol, label in ((jnp.float32, F32_TOL, "f32"), (jnp.bfloat16, BF16_TOL, "bf16")):
        forward_batch, embed = build_batch(mesh, tokens, dim, dtype, embed64)
        inner_call, entry_calls, layers = build_model(spec, sources, dtype, weights64, mutate)
        _, result = run_inner(spec, inner_call, layers, embed, forward_batch, all_layers)
        want = reference_stream(embed64, weights64, all_layers)
        where = f"all {num_layers} layers, {label}"
        rel = compare(where, jnp.stack(aux_from(result)), want, tol)
        say(f"      {where}: rel={rel:.3e}  {'ok' if within(rel, tol) else 'FAILED'}")

    # The gate decides which layers land, and the default is to capture nothing.
    forward_batch, embed = build_batch(mesh, tokens, dim, jnp.float32, embed64)
    inner_call, entry_calls, layers = build_model(spec, sources, jnp.float32, weights64, mutate)

    _, result = run_inner(spec, inner_call, layers, embed, forward_batch, [])
    if len(aux_from(result)) != 0:
        problems.append("default gate captured something")
    say(f"      empty layers_to_capture: {len(aux_from(result))} captured")

    gate = eagle3_gate(num_layers)
    _, result = run_inner(spec, inner_call, layers, embed, forward_batch, gate)
    got = aux_from(result)
    # A capture with the wrong layer count has the wrong shape, which rel_error reads as inf.
    stacked = jnp.stack(got) if got else np.zeros((0, tokens, dim))
    want = reference_stream(embed64, weights64, gate)
    rel = compare(f"EAGLE3 gate {gate}", stacked, want, F32_TOL)
    say(f"      EAGLE3 gate {gate}: {len(got)} captured  rel={rel:.3e}")

    # The entry classes decide whether the list reaches the logits processor.
    for (_, cls_name), entry_call in zip(spec["entries"], entry_calls):
        args = (spec, sources, entry_call, inner_call, layers, embed, forward_batch)
        quiet = run_entry(*args, flag=False)["aux"]
        if quiet is not None:
            problems.append(f"{cls_name} passed aux with the flag off")
        loud = run_entry(*args, flag=True)["aux"]
        if loud is None or len(loud) != num_layers:
            problems.append(f"{cls_name} didn't forward {num_layers} layers")
        say(
            f"      {cls_name}: flag off -> {'None' if quiet is None else 'A LIST'},"
            f" flag on -> {'None' if loud is None else str(len(loud)) + ' layers'}"
        )
    return problems, comparisons


def check_nan_capture(spec, sources, mesh, weights64, embed64, num_layers, tokens, dim):
    """Control: every captured value times NaN has to fail each comparison above.

    rel_error of a NaN capture is NaN. A gate written `rel >= tol` passes it, because NaN
    compares False with everything. This control reruns the same comparisons on a mutant whose
    capture is all NaN.
    """
    _, comparisons = capture_problems(
        spec, sources, mesh, weights64, embed64, num_layers, tokens, dim,
        mutate=m_capture_nan, verbose=False,
    )
    passed = [label for label, rel, tol in comparisons if within(rel, tol)]
    flagged = len(comparisons) - len(passed)
    print(
        f"      control (every captured value times NaN): {flagged} of {len(comparisons)}"
        " comparisons fail"
    )
    if comparisons and not passed:
        return 0
    print(f"      FAIL: the control didn't fail, so a NaN capture passes: {passed}")
    return 1


def run_entry(spec, sources, entry_call, inner_call, layers, embed, forward_batch, flag):
    """Call a patched entry `__call__` and report what the logits processor received."""
    seen = {"aux": "not called"}

    def logits_processor(hidden, head, metadata, aux_hidden_states=None):
        seen["aux"] = aux_hidden_states
        return LOGITS

    inner = Stub(
        layers=layers,
        layers_to_capture=list(range(len(layers))),
        embed_tokens=lambda ids: embed,
        norm=Stub(
            _call=lambda s, x: rmsnorm_forward(x, None, s.scale, s.epsilon),
            scale=jnp.ones(embed.shape[-1], jnp.float32),
            epsilon=EPS,
        ),
        _call=lambda s, fb, pool: inner_call(s, fb, pool),
    )
    fields = {
        "capture_aux_hidden_states": flag,
        "logits_processor": logits_processor,
        "lm_head": "lm_head",
        "config": Stub(tie_word_embeddings=False),
        "tie_word_embeddings": False,
    }
    if spec["nested"]:
        after = sources["after"]
        middle_file, middle_cls = spec["middle"]
        middle_call = compile_method(find_method(after[middle_file], middle_cls, "__call__"))
        entry_file, entry_cls_name = spec["entries"][0]
        getter = compile_method(find_method(after[entry_file], entry_cls_name, "model"))
        fields["language_model"] = Stub(model=inner, _call=middle_call)
        entry_cls = type("EntryStub", (Stub,), {"model": property(getter)})
    else:
        fields["model"] = inner
        entry_cls = Stub

    entry = entry_cls(**fields)
    entry_call(entry, forward_batch, Stub(token_to_kv_pool="kv-pool"), "metadata")
    return seen


def check_backbone_property(spec, sources):
    """The `model` property is how `_setup_hidden_states_capture` reaches the backbone.

    `sglang-jax-877.patch` does `getattr(self.model, "model", None)` and then sets
    `layers_to_capture` on what it finds. Without the property that returns None, the list stays
    empty, and the run reports the final layer instead of every layer.
    """
    problems = []
    rel_path, cls_name = spec["entries"][0]
    cls = find_class(ast.parse(sources["after"][rel_path]), cls_name)
    prop_fn = None
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "model":
            prop_fn = item
    if prop_fn is None:
        return [f"{cls_name} has no `model` property"]

    getter = compile_method(prop_fn)
    backbone = Stub(layers_to_capture=[])
    entry = Stub(language_model=Stub(model=backbone))
    if getter(entry) is not backbone:
        problems.append("the `model` property doesn't return the backbone")

    entry_cls = type("EntryStub", (Stub,), {"model": property(getter)})
    live = entry_cls(language_model=Stub(model=backbone))
    # What the model runner does.
    found = getattr(live, "model", None)
    if found is None or not hasattr(found, "layers_to_capture"):
        problems.append("getattr(entry, 'model') doesn't reach layers_to_capture")
    else:
        found.layers_to_capture = [0, 1]
        if backbone.layers_to_capture != [0, 1]:
            problems.append("setting layers_to_capture through the property misses the backbone")
    # nnx.split walks instance attributes. A property isn't one, so the backbone stays single.
    if "model" in vars(live):
        problems.append("the property leaked a second instance reference to the backbone")
    if any(isinstance(item, ast.FunctionDef) and item.name == "model" and any(
        isinstance(d, ast.Attribute) and d.attr == "setter" for d in item.decorator_list
    ) for item in cls.body):
        problems.append("the `model` property has a setter, so it isn't read only")
    return problems


NO_TARGET = "MUTATION OPERATOR FOUND NO TARGET"


def check_mutants(spec, sources, mesh, weights64, embed64, num_layers, tokens, dim):
    """Mutants of the patched source. Every one has to be caught.

    Then each operator runs on the unpatched text, where the hook isn't, and has to report
    that it found no target. An operator that changes nothing there and raises nothing would
    score a catch on the patched text too, whenever its target moves.
    """
    setup = (mesh, weights64, embed64, num_layers, tokens, dim)
    missed = 0
    for label, mutate in MUTANTS.items():
        caught, how = run_mutant(spec, sources, *setup, mutate)
        print(f"      control ({label}): {how}")
        if not caught:
            print("      FAIL: the control didn't fail, so this mutation is untested.")
            missed += 1

    unpatched = {"after": sources["before"]}
    quiet = []
    for label, mutate in MUTANTS.items():
        caught, how = run_mutant(spec, unpatched, *setup, mutate)
        if caught or not how.startswith(NO_TARGET):
            quiet.append(f"{label}: {how}")
    print(
        f"      control (each operator on the unpatched text): {len(MUTANTS) - len(quiet)} of"
        f" {len(MUTANTS)} found no target"
    )
    for line in quiet:
        print(f"      FAIL: {line}")
    return missed + bool(quiet)


def run_mutant(spec, sources, mesh, weights64, embed64, num_layers, tokens, dim, mutate):
    """Apply one mutation and report whether the checks above would catch it.

    A mutation operator that can't find its target fails the run. Its locator raises
    LookupError, and a control built on it would test nothing.
    """
    all_layers = list(range(num_layers))
    forward_batch, embed = build_batch(mesh, tokens, dim, jnp.float32, embed64)
    try:
        inner_call, entry_calls, layers = build_model(
            spec, sources, jnp.float32, weights64, mutate=mutate
        )
    except LookupError as exc:
        return False, f"{NO_TARGET}: {exc}"

    try:
        _, result = run_inner(spec, inner_call, layers, embed, forward_batch, all_layers)
    except Exception as exc:  # noqa: BLE001 - any raise is a catch
        return True, f"raised {type(exc).__name__}"

    aux = aux_from(result)
    if not isinstance(aux, list):
        return True, "the second tuple element is no longer the capture list"
    if len(aux) != num_layers:
        return True, f"captured {len(aux)} of {num_layers} layers"

    want = reference_stream(embed64, weights64, all_layers)
    rel = rel_error(jnp.stack(aux), want)
    if not within(rel, MUTANT_TOL):
        return True, f"rel={rel:.3e}"

    # Still numerically right. The entry classes are the remaining surface.
    for (_, cls_name), entry_call in zip(spec["entries"], entry_calls):
        args = (spec, sources, entry_call, inner_call, layers, embed, forward_batch)
        try:
            off = run_entry(*args, flag=False)
            on = run_entry(*args, flag=True)
        except Exception as exc:  # noqa: BLE001
            return True, f"{cls_name} raised {type(exc).__name__}"
        if off["aux"] is not None:
            return True, f"{cls_name} leaked aux with the flag off"
        if on["aux"] is None or len(on["aux"]) != num_layers:
            return True, f"{cls_name} forwarded nothing with the flag on"
    return False, f"NOT DETECTED (rel={rel:.3e})"


# --------------------------------------------------------------------------------------
# 4. what happens with no hook at all
# --------------------------------------------------------------------------------------


def collect_delivery(repo, base):
    """The three files on the request's return path, at `base` and on the stack above it."""
    files = [MIXIN, DETOKENIZER, TOKENIZER]
    return {"before": read_files(repo, files, rev=base), "after": read_files(repo, files)}


def calls_named(tree, name):
    """Every `Call` node whose callee is `name`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                out.append(node)
            elif isinstance(func, ast.Attribute) and func.attr == name:
                out.append(node)
    return out


def delivery_links(sources):
    """The four handoffs that carry a request's hidden states from the scheduler to the caller.

    Each one is a plain structural read of the patched file. The scheduler can fill
    `req.hidden_states` and still deliver nothing, because the field the multimodal path uses
    stops at the scheduler.
    """
    mixin = ast.parse(sources[MIXIN])
    detok = ast.parse(sources[DETOKENIZER])
    tok = ast.parse(sources[TOKENIZER])

    def dumped(node):
        return ast.dump(node)

    # req.hidden_states reaches output_hidden_states either by a direct append, or through a
    # list the scheduler fills per request and then builds output_hidden_states from.
    wanted = dumped(ast.parse("req.hidden_states", mode="eval").body)
    feeders = {
        call.func.value.id
        for call in calls_named(mixin, "append")
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.args
        and any(dumped(sub) == wanted for sub in ast.walk(call.args[0]))
    }
    collected = "output_hidden_states" in feeders or any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "output_hidden_states" for t in node.targets)
        and any(isinstance(sub, ast.Name) and sub.id in feeders for sub in ast.walk(node.value))
        for node in ast.walk(mixin)
    )

    sent = any(
        any(
            isinstance(arg, ast.Name) and arg.id == "output_hidden_states"
            for arg in call.args
        )
        or any(
            kw.arg == "output_hidden_states" for kw in call.keywords
        )
        for call in calls_named(mixin, "BatchTokenIDOut")
    )

    forwarded = any(
        any(
            kw.arg == "output_hidden_states"
            and dumped(kw.value)
            == dumped(ast.parse("recv_obj.output_hidden_states", mode="eval").body)
            for kw in call.keywords
        )
        for call in calls_named(detok, "BatchStrOut")
    )

    surfaced = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "meta_info"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "hidden_states"
            for target in node.targets
        )
        and "output_hidden_states" in dumped(node.value)
        for node in ast.walk(tok)
    )

    return {
        "the scheduler puts req.hidden_states into output_hidden_states": collected,
        "BatchTokenIDOut carries output_hidden_states": sent,
        "the detokenizer forwards it into BatchStrOut": forwarded,
        "tokenizer_manager reads it into meta_info['hidden_states']": surfaced,
    }


def check_delivery(delivery):
    """5. The request path hands the caller what the scheduler collected.

    `req.hidden_states` reaches `meta_info["hidden_states"]` through four separate objects.
    Filling the list at one end says nothing about the other end, and the field the multimodal
    prompt-embed path reads is a different one that never leaves the scheduler.
    """
    failures = 0
    links = delivery_links(delivery["after"])
    for text, ok in links.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
        failures += not ok

    # Control: the unpatched tree already carries three of the four links, which is what makes
    # the fourth easy to miss. The patch owns that one, so the unpatched tree has to read as
    # broken there and whole everywhere else.
    plain = delivery_links(delivery["before"])
    owned = "the scheduler puts req.hidden_states into output_hidden_states"
    missing = [text for text, ok in plain.items() if not ok]
    detected = missing == [owned]
    print(f"      control (the unpatched tree, which breaks {missing or 'no link'})")
    if not detected:
        print("      FAIL: the control didn't fail, so this check reads nothing.")
        failures += 1
    return failures


# The 877 patch's own CPU test, run on the real Engine by checks 4 and 6. Every fix in the patch
# has a test here, so a test that goes missing fails the run rather than dropping its check.
CPU_TEST = "test/srt/test_return_hidden_states_cpu.py"
HOOKLESS_TEST = "test_model_without_the_hook_refuses_to_start"
ENGINE_TESTS = [
    "test_cached_prefix_returns_every_prompt_row",
    "test_mixed_batch_keeps_rows_with_their_request",
    "test_logprobs_beside_hidden_states",
    "test_http_generate_returns_hidden_states",
    "test_non_stream_request_gets_its_rows_once",
    "test_retraction_keeps_one_row_per_position",
    "test_retraction_with_rows_in_flight",
    "test_mixed_chunk_under_overlap_keeps_rows_in_place",
    "test_request_without_the_server_flag_is_refused",
    "test_settings_that_break_capture_refuse_to_start",
    "test_layer_filter_returns_the_named_slots",
    "test_layer_filter_refuses_bad_slot_lists",
    "test_layer_filter_command_line",
    "test_data_parallel_ranks_keep_their_rows",
    HOOKLESS_TEST,
]

# Upstream's own unit tests for the scheduler, batch and tokenizer manager code the 877 patch
# changes, the ones unittest runs with no extra packages. Check 6 runs them on the patched tree,
# by directory, because a module under test/srt can't import as test.srt.
UPSTREAM_TESTS = {
    "python/sgl_jax/test": [
        "test_scheduler_chunked_ownership",
        "test_scheduler_idle_check",
        "test_scheduler_retraction",
        "test_mixed_chunk_dp",
    ],
    "test/srt": [
        "test_tokenizer_manager_event",
        "test_prepare_for_extend_protected_len",
        "test_merge_cache_loc",
    ],
}
# The control's mutant reads server_args in process_batch_result_prefill on every batch, the way
# an earlier version of the patch did. Upstream's chunked-ownership tests drive that method with
# a stub scheduler that has no server_args, so they have to fail on it.
OUTPUT_PROCESSOR = "python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py"
EAGER_READ_AT = "        hidden_state_offset = 0\n"
EAGER_READ = EAGER_READ_AT + "        _ = self.server_args.enable_return_hidden_states\n"


def unittest_outcomes(text):
    """Map each test method `unittest -v` reported to the word it printed.

    "ok" means it passed, and a failed subtest wins over the rest.
    """
    outcomes = {}
    for line in text.splitlines():
        head, sep, word = line.strip().rpartition(" ... ")
        if not sep or not head.startswith("test_"):
            continue
        name = head.split(" ", 1)[0]
        if outcomes.get(name, "ok") == "ok":
            outcomes[name] = word.strip() or "no verdict"
    return outcomes


def run_upstream_tests(repo, env, modules=None):
    """Run upstream unit tests on the checkout as it stands.

    Returns (passed, ran, output). passed means every unittest process exited 0 and ran at
    least one test. modules maps a directory to its modules, UPSTREAM_TESTS by default.
    """
    passed, ran, output = True, 0, []
    for directory, names in (modules or UPSTREAM_TESTS).items():
        proc = run_to_end(
            [sys.executable, "-m", "unittest", "-v", *names],
            timeout=900,
            cwd=os.path.join(repo, directory),
            env=env,
        )
        counted = [line.split()[1] for line in proc.stderr.splitlines() if line.startswith("Ran ")]
        count = int(counted[-1]) if counted else 0
        passed = passed and proc.returncode == 0 and count > 0
        ran += count
        output.append(f"$ cd {directory} && python -m unittest -v {' '.join(names)}")
        output.append(proc.stdout + proc.stderr)
    return passed, ran, "\n".join(output)


def run_eager_read_control(repo, env):
    """Run the chunked-ownership tests on the patched tree with EAGER_READ in the method.

    Returns (failed, output). failed means they failed on the read the mutant added, and it's
    None when the mutant's anchor is missing.
    """
    path = os.path.join(repo, OUTPUT_PROCESSOR)
    with open(path) as fh:
        source = fh.read()
    if source.count(EAGER_READ_AT) != 1:
        return None, f"{OUTPUT_PROCESSOR} holds {source.count(EAGER_READ_AT)} copies of the anchor"
    with open(path, "w") as fh:
        fh.write(source.replace(EAGER_READ_AT, EAGER_READ))
    try:
        passed, _, output = run_upstream_tests(
            repo, env, {"python/sgl_jax/test": ["test_scheduler_chunked_ownership"]}
        )
    finally:
        with open(path, "w") as fh:
            fh.write(source)
    return not passed and "no attribute 'server_args'" in output, output


def run_engine_suite(repo):
    """Run the 877 patch's CPU test and upstream's unit tests on the stack, which holds 877.

    Returns a dict. outcomes maps a CPU test name to the word unittest printed for it, "ok"
    when it passed, with a failed subtest winning over the rest.
    """
    try:
        env = dict(os.environ)
        # The checkout goes ahead of the caller's PYTHONPATH and doesn't replace it, so packages
        # the caller put there still import.
        python_path = os.path.join(repo, "python")
        if env.get("PYTHONPATH"):
            python_path += os.pathsep + env["PYTHONPATH"]
        # The test's own children start the Engine at tp_size 1, which wants one device, not
        # the eight this file forces for its mesh.
        env.update(
            JAX_PLATFORMS="cpu",
            PYTHONPATH=python_path,
            XLA_FLAGS="--xla_force_host_platform_device_count=1",
        )
        proc = run_to_end([sys.executable, os.path.join(repo, CPU_TEST), "-v"], 3600, env=env)
        upstream_passed, upstream_ran, upstream_output = run_upstream_tests(repo, env)
        control_failed, control_output = run_eager_read_control(repo, env)
    finally:
        git(repo, "checkout", "--", ".")
        git(repo, "clean", "-qfd")

    return {
        "outcomes": unittest_outcomes(proc.stderr),
        "returncode": proc.returncode,
        "output": proc.stdout + proc.stderr,
        "upstream_passed": upstream_passed,
        "upstream_ran": upstream_ran,
        "upstream_output": upstream_output,
        "control_failed": control_failed,
        "control_output": control_output,
    }


def check_no_hook(engine):
    """4. Without the hook, `--enable-return-hidden-states` refuses to start.

    On a model with no `layers_to_capture`, `LogitsProcessor` would store the final normed
    hidden state, and the scheduler would cut it into per-layer slices that aren't layers. The
    877 patch's model runner refuses to start instead. The CPU test starts the real Engine on a
    hookless Qwen2 with the flag and requires that refusal. Its control starts the same
    checkpoint without the flag. Each of the four models here is hookless before its patch,
    which check 2's control reads.
    """
    outcomes = engine["outcomes"]
    outcome = outcomes.get(HOOKLESS_TEST, "no result")
    ok = outcome == "ok"
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a hookless Qwen2 refuses to start with the flag and"
        f" serves without it: {outcome}"
    )
    if not ok:
        print(f"      the CPU test exited {engine['returncode']}. Its output:")
        print(engine["output"])
    return int(not ok)


def check_engine_suite(engine):
    """6. `sglang-jax-877.patch` passes its CPU test on the real Engine, and upstream's tests.

    Upstream's unit tests for the code the patch changes have to pass on the patched tree too.
    Their control runs the chunked-ownership tests with process_batch_result_prefill reading
    server_args on every batch, which they have to catch.
    """
    outcomes, returncode = engine["outcomes"], engine["returncode"]
    failures = 0
    for name in ENGINE_TESTS:
        if name == HOOKLESS_TEST:
            continue
        outcome = outcomes.get(name, "no result")
        ok = outcome == "ok"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {outcome}")
        failures += not ok
    extra = sorted(set(outcomes) - set(ENGINE_TESTS))
    if extra:
        print(f"  [FAIL] the CPU test runs tests this gate doesn't list: {extra}")
        failures += 1
    if returncode != 0 and not failures and outcomes.get(HOOKLESS_TEST) == "ok":
        print(f"  [FAIL] the CPU test exited {returncode}")
        failures += 1
    if failures:
        print("      the CPU test's output:")
        print(engine["output"])

    files = sum(len(names) for names in UPSTREAM_TESTS.values())
    ok = engine["upstream_passed"]
    print(
        f"  [{'PASS' if ok else 'FAIL'}] upstream's unit tests for the scheduler, batch and"
        f" tokenizer manager code the patch changes: {files} files, {engine['upstream_ran']} tests"
    )
    if not ok:
        failures += 1
        print("      their output:")
        print(engine["upstream_output"])
    detected = engine["control_failed"] is True
    print("      control (process_batch_result_prefill reads server_args on every batch)")
    if not detected:
        print("      FAIL: the control didn't fail, so these tests read nothing. Its output:")
        print(engine["control_output"])
        failures += 1
    return failures


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAILED: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))

    workdir = tempfile.mkdtemp(prefix="capture-hooks-")
    try:
        repo = get_checkout(workdir)
        base = stack_patches(repo)
        at = git(repo, "rev-parse", "--short", base).stdout.strip()
        print(f"sglang-jax at {at}, then the stack scripts/bootstrap_tpu_vm.sh applies:")
        print("  sglang-jax-877.patch, steering-hook.patch, qwen3-steering-hook.patch")
        print(f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}\n")

        print("1. patches apply to the stack and compile")
        failures = check_patches(repo, workdir)
        collected = collect_sources(repo)
        delivery = collect_delivery(repo, base)
        engine = run_engine_suite(repo)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n2. the parts of the hook")
    failures += check_structure(collected)

    print("\n3. the patched methods run")
    failures += check_semantics(collected, mesh)

    print("\n4. no hook, no start")
    failures += check_no_hook(engine)

    print("\n5. the request path delivers what the scheduler collected")
    failures += check_delivery(delivery)

    print("\n6. sglang-jax-877.patch on the real Engine")
    failures += check_engine_suite(engine)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
