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

"""Gate for `glm5-fp8-accumulate.patch`: FP8 linears sum in float32 on GLM-5.3's serving tree.

Runs on CPU in child processes on 4 simulated devices. No TPU needed.

    python3 upstream/test_glm5_fp8_accumulate.py

GLM-5.3's second capture check on a v5p-64 (tp 32, 2026-09-27 08:48Z) failed layers 10 to 13 at
2.16 to 2.75 times the BF16 floor. Every FP8 linear runs the block-wise FP8 matmul, a Pallas TPU
kernel, and at `eb061d8` it keeps its accumulator in BF16: each 128-wide input block's float32
dot rounds to BF16, the block's float32 scale rounds to BF16, and the running sum rounds after
every block. `xla_quantized_matmul_local` then rounds each shard's partial sum to BF16 before a
row-parallel layer's all-reduce. A BF16 forward rounds the product once. The patch keeps the
accumulator and the scales in float32 and hands a row-parallel layer float32 partial sums to
reduce.

The children run the kernel itself, from each tree, in jax's TPU interpret mode; the tests for
the other GLM-5.3 patches swap it for a float32 XLA stand-in, which can't see the bug. The MLA
paged attention and the grouped matmul keep `test_glm5_tp_sharding.py`'s stand-ins, which sum in
float32 as the chip does for them (`EPMoE` asks gmm for a float32 accumulator). Every float32 dot
left at the default precision rounds its operands to BF16 the way a TPU runs it, through
`tpu_default_precision` from `models/test_nemotron_h_model.py`.

Checks:

1. The patch applies on the tree the GLM-5.3 row of `scripts/multihost_run.sh` builds
   (877, both steering patches, `multihost-hidden-states.patch`, `glm5-capture-hook.patch`,
   `glm5-tp-sharding.patch`), a corrupted copy doesn't, and every file it touches compiles.
2. The kernel on one GLM-5.3 `q_a_proj` slice (6,144 inputs, 48 blocks, non-power-of-two block
   scales, one input channel at 30 times the rest): its error against float64 over a BF16 matmul's
   error. The unpatched kernel sits at `KERNEL_BEFORE_MIN` or more, the patched one at
   `KERNEL_AFTER_MAX` or less. With power-of-two scales, which BF16 holds without rounding, and
   no outlier, the unpatched kernel still sits at `KERNEL_BEFORE_MIN` or more, and the patched one
   matches the BF16 matmul within 1%, since both round the same exact products once.
3. The reduction dtype, read from the traced program of `xla_quantized_matmul_local` under
   `shard_map` at a row-parallel spec: BF16 before the patch, float32 after.
4. The engine at --tp-size 4 --ep-size 4 on a tiny GLM-5.3 in the published FP8 layout, with
   non-power-of-two scales and two outlier channels in the embedding: each captured layer's median
   per-token error against transformers' float32 forward, over transformers' BF16 forward's
   (`check_capture.py`'s gate). The patched tree passes the gate at every layer. The unpatched
   tree's worst layer sits at `ENGINE_GAP_MIN` times the patched tree's worst or more.
5. Control: the patched tree on a copy with layer 1's `o_proj` scales at `CONTROL_SCALE` times
   has to fail the same gate against the unchanged checkpoint's forwards, so the gate can see an
   error of this size. In transformers' float32 forward that copy alone sits at 2.0 to 2.5 times
   the BF16 floor at 1.03.

The routed experts' scales are zero in every engine run. At --ep-size 4 the routed experts move a
few tokens by up to 9% on CPU (`test_glm5_tp_sharding.py` check 5), and they run the grouped
matmul, which the patch doesn't touch.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download. `LOG_PAYLOADS=1`
prints every engine reply, and `GLM5ACC_KEEP=1` keeps the trees, checkpoints and child logs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(HERE, "capture-hooks"))
sys.path.insert(0, HERE)

import test_glm5_tp_sharding as tp  # noqa: E402

PATCH = os.path.join(HERE, "glm5-fp8-accumulate.patch")
# The GLM-5.3 row of scripts/multihost_run.sh, in its order, before this patch.
BELOW = tp.BELOW + ("glm5-tp-sharding.patch",)
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS") == "1"
DEVICES = 4
TP = 4
BLOCK = 128
# GLM-5.3's q_a_proj takes 6,144 inputs. 512 of its 2,048 outputs keep the interpreter quick.
KERNEL_SHAPE = (64, 6144, 512)
# Layer 10's input on the chip run's reference: max |h| over RMS has a median of 31 per token.
OUTLIER = 30.0
KERNEL_BEFORE_MIN = 1.5
KERNEL_AFTER_MAX = 1.2
ENGINE_GAP_MIN = 1.3
FLOOR_FACTOR = 2.0
CONTROL_SCALE = 1.05
CONTROL_TENSOR = "model.layers.1.self_attn.o_proj.weight_scale_inv"
# hidden 1,024 gives q_a, kv_a and the shared expert's gate and up 8 input blocks each. The
# indexer keeps more tokens than a prompt holds, as GLM-5.3's 2,048 do in the chip run's prompts:
# the engine prefills dense MLA (DSA_PREFILL_SPARSE is off) and transformers masks to the top-k.
TINY = dict(tp.TINY, hidden_size=1024, index_topk=128)
# Embedding channels held at a fixed large value for every token, the way GLM-5.3's residual
# stream carries channel 4386 and a few others from layer 8 on.
OUTLIER_CHANNELS = {7: 1.0, 300: -0.6}
PROMPT_LENGTHS = (90, 37)

RESULTS = []
record = None


def _record(ok, label):
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)
    return ok


record = _record


# --------------------------------------------------------------------------------------------
# The tiny checkpoint
# --------------------------------------------------------------------------------------------


def quantize_e4m3(w, block=(BLOCK, BLOCK)):
    """float32 [out, in] -> (E4M3, float32 scale_inv per tile) with scale = amax / 448, as the
    published checkpoint stores it. The scales aren't powers of two, so a BF16 copy moves them."""
    import ml_dtypes

    out, inn = w.shape
    bo, bi = -(-out // block[0]), -(-inn // block[1])
    pad = np.zeros((bo * block[0], bi * block[1]), np.float32)
    pad[:out, :inn] = w
    tiles = pad.reshape(bo, block[0], bi, block[1])
    s = (np.maximum(np.abs(tiles).max(axis=(1, 3)), 1e-4) / 448.0).astype(np.float32)
    q = (tiles / s[:, None, :, None]).reshape(bo * block[0], bi * block[1])[:out, :inn]
    return np.clip(q, -448, 448).astype(ml_dtypes.float8_e4m3fn), s


def build_checkpoint(out, root, tokdir):
    """test_glm5_tp_sharding's builder at hidden 1,024, with published-style scales, the routed
    experts' scales at zero and the outlier channels in the embedding."""
    saved = (tp.TINY, tp.quantize_e4m3)
    tp.TINY, tp.quantize_e4m3 = TINY, quantize_e4m3
    try:
        tensors = tp.build_checkpoint(out, root, tokdir)
    finally:
        tp.TINY, tp.quantize_e4m3 = saved
    changed = dict(tensors)
    tag, emb = changed["model.embed_tokens.weight"]
    emb = emb.copy()
    for ch, value in OUTLIER_CHANNELS.items():
        emb[:, ch] = value
    changed["model.embed_tokens.weight"] = (tag, emb)
    for name, (tag, arr) in tensors.items():
        if ".mlp.experts." in name and name.endswith("weight_scale_inv"):
            changed[name] = (tag, np.zeros_like(arr))
    tp.write_safetensors(os.path.join(out, "model-00001-of-00001.safetensors"), changed)
    return changed


def control_checkpoint(src, dst, tensors):
    shutil.copytree(src, dst)
    changed = dict(tensors)
    tag, arr = changed[CONTROL_TENSOR]
    changed[CONTROL_TENSOR] = (tag, arr * np.float32(CONTROL_SCALE))
    tp.write_safetensors(os.path.join(dst, "model-00001-of-00001.safetensors"), changed)


# --------------------------------------------------------------------------------------------
# Children
# --------------------------------------------------------------------------------------------


def interpret_blockwise_kernel():
    """Run the tree's block-wise FP8 kernel in jax's TPU interpret mode on CPU.

    The kernel module reads `pl.pallas_call` when it traces, so a copy of `pl` whose
    `pallas_call` asks for interpret mode reaches that kernel alone. Returns the module.
    """
    import functools
    import types

    from jax.experimental.pallas import tpu as pltpu
    from sgl_jax.srt.kernels.quantized_matmul import blockwise_utils
    from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import blockwise_kernel

    pl = blockwise_kernel.pl
    proxy = types.SimpleNamespace(**{k: getattr(pl, k) for k in dir(pl) if not k.startswith("__")})
    proxy.pallas_call = functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams())
    blockwise_kernel.pl = proxy
    blockwise_utils._BLOCKWISE_KERNEL = blockwise_kernel.quantized_matmul_kernel
    blockwise_utils._TRIED_LOADING_BLOCKWISE_KERNEL = True
    return blockwise_kernel


def kernel_case(outlier, pow2, seed=0):
    """x [T, K] BF16, E4M3 weight, its scales, the float64 product and a BF16 matmul's product."""
    import jax.numpy as jnp
    import ml_dtypes

    t, k, n = KERNEL_SHAPE
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, k)).astype(np.float32)
    if outlier:
        x[:, 5] = OUTLIER
    w = (rng.standard_normal((n, k)) / math.sqrt(k)).astype(np.float32)
    tiles = w.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    s = (np.abs(tiles).max(axis=(1, 3)) / 448.0).astype(np.float32)
    if pow2:
        s = np.exp2(np.ceil(np.log2(s))).astype(np.float32)
    q = (tiles / s[:, None, :, None]).reshape(n, k).astype(ml_dtypes.float8_e4m3fn)
    wdq = q.astype(np.float32) * np.repeat(np.repeat(s, BLOCK, 0), BLOCK, 1)
    xb = np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))
    exact = xb.astype(np.float64) @ wdq.astype(np.float64).T
    w_bf16 = np.asarray(jnp.asarray(wdq, jnp.bfloat16).astype(jnp.float32))
    floor = np.asarray(jnp.asarray(xb @ w_bf16.T, jnp.bfloat16).astype(jnp.float32), np.float64)
    return x, q, s, exact, floor


def median_error(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if not np.all(np.isfinite(got)):
        return math.inf
    return float(np.median(np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)))


def kernel_child(tree, out_path):
    """Checks 2 and 3 on one tree: the kernel's errors and the row-parallel reduction dtype."""
    sys.path.insert(0, os.path.join(tree, "python"))
    import jax
    import jax.extend
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import get_safe_blockwise_tuned_value
    from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local

    bk = interpret_blockwise_kernel()
    t, k, n = KERNEL_SHAPE
    result = {}
    for label, outlier, pow2 in (("outlier", True, False), ("plain", False, True)):
        x, q, s, exact, floor = kernel_case(outlier, pow2)
        scale = jnp.asarray(np.repeat(s.T[:, None, :], BLOCK, axis=2))  # [in_blocks, 1, n]
        tuned = get_safe_blockwise_tuned_value(n_batch=t, n_out=n, n_in=k, x_q_dtype=jnp.bfloat16,
                                               w_q_dtype=jnp.float8_e4m3fn, block_size_in=BLOCK)
        got = bk.quantized_matmul_kernel(jnp.asarray(x, jnp.bfloat16), jnp.asarray(q), scale,
                                         block_size=BLOCK, tuned_value=tuned)
        result[label] = {"kernel": median_error(got, exact), "floor": median_error(floor, exact),
                         "out_dtype": str(got.dtype)}

    # The reduction a row-parallel layer runs, read off the traced program.
    mesh = Mesh(np.array(jax.devices()[:TP]), ("tensor",))
    x, q, s, _, _ = kernel_case(True, False)
    scale = jnp.asarray(np.repeat(s.T[:, None, :], BLOCK, axis=2))

    def local(xs, wq, ws):
        return xla_quantized_matmul_local(xs, wq, ws, quantize_activation=False,
                                          reduce_axis="tensor", weight_block_size=(BLOCK, BLOCK),
                                          allow_narrow_n_blockwise=True)

    fn = jax.shard_map(local, mesh=mesh, in_specs=(P(None, "tensor"), P(None, "tensor"),
                                                   P("tensor", None, None)),
                       out_specs=P(), check_vma=False)
    jaxpr = jax.make_jaxpr(fn)(jnp.asarray(x, jnp.bfloat16), jnp.asarray(q), scale)
    reductions = []

    def walk(jp):
        for eqn in jp.eqns:
            if eqn.primitive.name in ("psum", "psum2", "reduce_scatter", "psum_scatter",
                                      "all_reduce"):
                reductions.append((eqn.primitive.name, str(eqn.invars[0].aval.dtype)))
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jaxpr.jaxpr)
    result["reductions"] = reductions
    out = fn(jnp.asarray(x, jnp.bfloat16), jnp.asarray(q), scale)
    result["row_parallel_out_dtype"] = str(out.dtype)
    print("CHILD " + json.dumps(result), flush=True)
    with open(out_path, "w") as fp:
        json.dump(result, fp)
    return 0


def engine_child(tree, ckpt, out_path):
    """Check 4 and 5 on one tree: the engine's prefill capture at tp 4 under TPU arithmetic."""
    os.environ["SGLANG_JAX_TREE"] = tree
    import cpu_engine

    cpu_engine.stacked_tree()
    import capture_activations
    from jax._src import tpu_info

    sys.path.insert(0, os.path.join(HERE, "models"))
    from test_nemotron_h_model import tpu_default_precision

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V5P, 2)
    interpret_blockwise_kernel()
    tp.install_mla_standin()
    tp.install_gmm_standin()
    prompts = prompt_ids()
    try:
        with tpu_default_precision():
            engine = cpu_engine.open_engine(
                ckpt, capture=True, batch_size=4, token_padding=64, tp_size=TP, ep_size=TP,
                device_indexes=list(range(TP)), attention_backend="dsa_sparse",
                disable_radix_cache=True, context_length=tp.CONTEXT)
            try:
                outs = engine.generate(
                    input_ids=prompts, sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                    return_hidden_states=True)
            finally:
                engine.shutdown()
    except Exception as exc:
        traceback.print_exc()
        print("CHILD " + json.dumps({"raised": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0
    arrays, summary = {}, {"requests": []}
    for k, (ids, o) in enumerate(zip(prompts, outs)):
        if LOG_PAYLOADS:
            print("    reply:", json.dumps(capture_activations.reply_summary(o), default=str),
                  flush=True)
        hidden, prompt_rows = capture_activations.hidden_states_from_output(o)
        arrays[f"hidden{k}"] = hidden[:len(ids)]
        summary["requests"].append({"hidden": list(hidden.shape), "prompt_rows": prompt_rows})
    np.savez(out_path, **arrays)
    print("CHILD " + json.dumps(summary), flush=True)
    return 0


def prompt_ids():
    return [np.random.default_rng(7 + k).integers(3, 390, n).tolist()
            for k, n in enumerate(PROMPT_LENGTHS)]


def launch(args, log_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GLM5ACC_CHILD="1")
    flags = " ".join(f for f in env.get("XLA_FLAGS", "").split()
                     if not f.startswith("--xla_force_host_platform_device_count"))
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={DEVICES}".strip()
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), *args], env=env,
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log


def collect(job):
    result, out = tp.collect(job)
    return result, out


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def build_trees(workdir):
    """Check 1. The GLM-5.3 row's tree, and the same tree with the patch."""
    import cpu_engine
    from test_capture_hooks import corrupt

    print("check 1: the patch applies to the GLM-5.3 tree", flush=True)
    base = cpu_engine.build_tree(os.path.join(workdir, "base"),
                                 os.environ.get("SGLANG_JAX_REPO") or None)
    for name in BELOW:
        done = tp.git(base, "apply", os.path.join(HERE, name))
        if not record(done.returncode == 0, f"{name} applies to the stack"):
            print(done.stderr)
            return None, None
    path = os.path.join(base, tp.CPU_GUARD[0])
    with open(path) as fp:
        src = fp.read()
    if not record(src.count(tp.CPU_GUARD[1]) == 1,
                  f"the block-wise FP8 TPU check in {tp.CPU_GUARD[0]} is there to open for CPU"):
        return None, None
    with open(path, "w") as fp:
        fp.write(src.replace(tp.CPU_GUARD[1], tp.CPU_GUARD[2]))
    tp.git(base, "add", "-A")
    tp.git(base, "-c", "user.email=t@local", "-c", "user.name=t", "commit", "-qm", "glm5.3 row")
    fixed = os.path.join(workdir, "fixed")
    done = subprocess.run(["git", "clone", "-q", base, fixed], capture_output=True, text=True)
    if done.returncode:
        record(False, f"clone the base tree: {done.stderr}")
        return None, None
    shutil.copy(os.path.join(base, ".git", "cpu-engine-patches"),
                os.path.join(fixed, ".git", "cpu-engine-patches"))
    with open(PATCH) as fp:
        bad = os.path.join(workdir, "corrupt.patch")
        with open(bad, "w") as out:
            out.write(corrupt(fp.read()))
    record(tp.git(fixed, "apply", "--check", bad).returncode != 0,
           "control: a corrupted copy is refused")
    done = tp.git(fixed, "apply", PATCH)
    if not record(done.returncode == 0,
                  f"glm5-fp8-accumulate.patch applies on {' + '.join(BELOW)}"):
        print(done.stderr)
        return None, None
    touched = [l[6:].strip() for l in open(PATCH) if l.startswith("+++ b/")]
    for path in touched:
        done = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(fixed, path)],
                              capture_output=True, text=True)
        record(done.returncode == 0, f"{path} compiles")
    return base, fixed


def kernel_checks(results):
    print("check 2: the block-wise FP8 kernel on a GLM-5.3 q_a_proj slice, interpreted", flush=True)
    before, after = results["base"], results["fixed"]
    for label, res in (("before", before), ("after", after)):
        if "raised" in res:
            record(False, f"the kernel child {label} the patch ran: {res['raised'][:300]}")
            return
    rb = before["outlier"]["kernel"] / before["outlier"]["floor"]
    ra = after["outlier"]["kernel"] / after["outlier"]["floor"]
    record(rb >= KERNEL_BEFORE_MIN,
           f"before the patch, one input channel at {OUTLIER:g}x: kernel error "
           f"{before['outlier']['kernel']:.2e} is {rb:.2f}x a BF16 matmul's "
           f"{before['outlier']['floor']:.2e} (needs {KERNEL_BEFORE_MIN}x or more)")
    record(ra <= KERNEL_AFTER_MAX,
           f"after the patch: kernel error {after['outlier']['kernel']:.2e} is {ra:.2f}x the "
           f"BF16 matmul's (needs {KERNEL_AFTER_MAX}x or less)")
    pb = before["plain"]["kernel"] / before["plain"]["floor"]
    record(pb >= KERNEL_BEFORE_MIN,
           f"before the patch, power-of-two scales that BF16 holds exactly and no outlier: still "
           f"{pb:.2f}x the BF16 matmul, so the BF16 accumulator carries the error")
    pa = after["plain"]["kernel"] / after["plain"]["floor"]
    record(abs(pa - 1.0) <= 0.01,
           f"after the patch, power-of-two scales: {pa:.4f}x the BF16 matmul, which rounds the "
           f"same exact products once")
    dtypes = (before["outlier"]["out_dtype"], after["outlier"]["out_dtype"])
    record(dtypes == ("bfloat16", "bfloat16"),
           f"the kernel still returns x's dtype by default: {dtypes[0]} before, {dtypes[1]} after")
    print("check 3: the row-parallel reduction dtype, from the traced program", flush=True)
    record([d for _, d in before["reductions"]] == ["bfloat16"],
           f"before the patch the shards' partial sums reduce in BF16: {before['reductions']}")
    record([d for _, d in after["reductions"]] == ["float32"]
           and after["row_parallel_out_dtype"] == "bfloat16",
           f"after the patch they reduce in float32 and round once: {after['reductions']}, "
           f"output {after['row_parallel_out_dtype']}")


def engine_rows(npz_path, reference, floor):
    import check_capture

    data = np.load(npz_path)
    worst, rows_all = 0.0, []
    for k in range(len(PROMPT_LENGTHS)):
        rows, summary = check_capture.compare_layers(
            data[f"hidden{k}"], reference[k], 0.999, 3.0, floor[k], FLOOR_FACTOR)
        rows_all.append(rows)
        worst = max(worst, summary["worst_ratio"])
    return worst, rows_all


def describe(rows_all):
    parts = []
    for k, rows in enumerate(rows_all):
        parts.append(f"prompt {k}: " + ", ".join(
            f"L{r['layer']} {r['ratio']:.2f}" for r in rows if r["floor_token_error"] > 0))
    return "; ".join(parts)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="glm5acc-")
    try:
        root = tp.fetch_published()
        base, fixed = build_trees(workdir)
        if base is None:
            return 1
        import cpu_engine
        import check_capture

        tokdir = os.path.join(workdir, "tok")
        cpu_engine.write_tokenizer(tokdir)
        ckpt = os.path.join(workdir, "ckpt")
        tensors = build_checkpoint(ckpt, root, tokdir)
        ctrl = os.path.join(workdir, "ckpt-control")
        control_checkpoint(ckpt, ctrl, tensors)

        jobs = {}
        for label, tree in (("base", base), ("fixed", fixed)):
            out = os.path.join(workdir, f"kernel-{label}.json")
            jobs[("kernel", label)] = (launch(["kernel", tree, out], out[:-5] + ".log"), out)
        for label, tree, path in (("base", base, ckpt), ("fixed", fixed, ckpt),
                                  ("control", fixed, ctrl)):
            out = os.path.join(workdir, f"engine-{label}.npz")
            jobs[("engine", label)] = (launch(["engine", tree, path, out], out[:-4] + ".log"), out)

        # transformers' float32 forward and its BF16 forward, the two sides of the gate.
        batch = prompt_ids()
        reference = check_capture.reference_forward(ckpt, batch, "float32")
        floor = check_capture.reference_forward(ckpt, batch, "bfloat16")
        print(f"  transformers: float32 and BF16 forwards, {len(reference[0])} entries per prompt",
              flush=True)

        results = {}
        for key, (job, out) in jobs.items():
            print(f"  run: {key[0]} {key[1]}", flush=True)
            res, log = collect(job)
            results[key] = (res, log, out)

        kernel_checks({label: results[("kernel", label)][0] for label in ("base", "fixed")})

        print(f"check 4: the engine at --tp-size {TP} --ep-size {TP} against transformers",
              flush=True)
        worst = {}
        for label in ("base", "fixed", "control"):
            res, log, out = results[("engine", label)]
            if "raised" in res:
                record(False, f"the {label} engine run serves: {res['raised'][:300]}")
                print(log)
                continue
            worst[label], rows_all = engine_rows(out, reference, floor)
            print(f"    {label}: ratio to the BF16 floor per layer: {describe(rows_all)}",
                  flush=True)
        if "fixed" in worst:
            record(worst["fixed"] <= FLOOR_FACTOR,
                   f"after the patch every layer sits within {FLOOR_FACTOR:g}x the BF16 floor: "
                   f"worst {worst['fixed']:.2f}x")
        if "fixed" in worst and "base" in worst:
            record(worst["base"] >= ENGINE_GAP_MIN * worst["fixed"],
                   f"before the patch the worst layer sits at {worst['base']:.2f}x the floor, "
                   f"{worst['base'] / worst['fixed']:.2f}x the patched tree's worst (needs "
                   f"{ENGINE_GAP_MIN}x or more)")
        print("check 5: control", flush=True)
        if "control" in worst:
            record(worst["control"] > FLOOR_FACTOR,
                   f"the patched tree with {CONTROL_TENSOR} at {CONTROL_SCALE}x fails the gate: "
                   f"worst {worst['control']:.2f}x the floor")
    except Exception:
        traceback.print_exc()
        record(False, "the gate ran to the end")
    finally:
        if os.environ.get("GLM5ACC_KEEP") == "1":
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    failed = [label for ok, label in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if os.environ.get("GLM5ACC_CHILD") == "1":
        if sys.argv[1] == "kernel":
            raise SystemExit(kernel_child(sys.argv[2], sys.argv[3]))
        raise SystemExit(engine_child(sys.argv[2], sys.argv[3], sys.argv[4]))
    raise SystemExit(main())
