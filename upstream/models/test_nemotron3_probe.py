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

"""CPU gate for `nemotron3-probe.patch` and `check_capture.py --engine-layers`.

    python3 upstream/models/test_nemotron3_probe.py

The probe patch adds eight switches, `SGL_PROBE_TOPK`, `SGL_PROBE_GMM`, `SGL_PROBE_ROUTER`,
`SGL_PROBE_LATENT`, `SGL_PROBE_MOE`, `SGL_PROBE_SHARED`, `SGL_PROBE_MOE_OUT` and
`SGL_PROBE_MATMUL`, each off by default. Each swaps one piece of the LatentMoE path for a plain JAX
version, so an on-chip run per setting can show whether that piece moves the capture. On CPU in
float32 each switch has to give the default path's numbers. The TPU-only pieces they replace don't
run on CPU, so the checks on those read the pieces directly.

1. The probe patch applies on the Nemotron 3 model patch and its capture hook, and on top of
   `multihost-hidden-states.patch` as `scripts/multihost_run.sh` stacks it, and every file it
   touches compiles.
2. Each switch, the dense, shared and output switches together, and all of them at once, leave
   the tiny model's every capture slot, final state and logits within 1e-5 of the default path in
   float32: at tp=1, at tp=4, at tp=4 with the experts split four ways (`ep_size` 4), where each
   shard's grouped matmul starts at a group offset, and at tp=8 with ep_size 8. Control: with the
   switch on, a copy of its code that nudges its output by 1% moves the forward past 1e-5, so the
   switch reaches the code it names. `SGL_PROBE_MOE_OUT` changes nothing in float32, so its check
   runs in BF16: the slot after each MoE layer comes back float32 and moves by less than BF16's
   rounding, and the slot before stays put. `SGL_PROBE_MATMUL=highest` has to reach the shared
   expert and stop at gmm and the top-k, read from the setting each call sees.
3. `dense_gmm`, the `SGL_PROBE_GMM` stand-in, against `gmm` (the v1 kernel in interpret mode, the
   one a CPU runs) on 48 rows over 8 groups, at group offsets 0 and 4 with 4 local groups, in
   float32. Rows no local group owns come back zero. Control: one group's rows shifted by one.
4. The Pallas grouped top-k kernel a TPU runs, in interpret mode, against the plain JAX path
   `SGL_PROBE_TOPK=jnp` takes, on 256 tokens of Ultra's routing shape: 512 experts, top 22, one
   group, with the correction bias. The same expert set and the same weights per token.
   `SGL_PROBE_TOPK=jnp` turns the kernel off where the server arguments turn it on. Control:
   without the variable those arguments turn it on.
5. A misspelled value raises instead of running the default path.
6. `check_capture.py --engine-layers 3` on a tiny four-layer nemotron_h checkpoint through the real
   engine on CPU, against a reference npz of the whole model: the engine builds three layers, the
   capture holds three slots and the check passes on them. Control: a reference npz from the same
   checkpoint with layer 1's `fc2_latent_proj` scaled by 1.5 fails slot 2 and passes slots 0 and
   1, so the cut compares entry k with slot k. `--engine-layers 5`, past the reference's four
   layer inputs, stops before the engine starts.
7. `dense_experts`, the `SGL_PROBE_MOE=dense` body, summed over two shards, against a per-token
   gather of each token's top-k experts in float64, ungated squared ReLU and gated SiLU. Control:
   one routing id swapped.
8. Two requests packed to 128 tokens (384 routed rows at top 3) at tp=1, tp=4 ep=4 and tp=8 ep=8,
   against transformers one request at a time. `SGL_PROBE_MOE=dense` has to match everywhere; the
   default path and `SGL_PROBE_GMM=xla` are reported per request, which is where the GLM-5.3 CPU
   gate saw the second prompt move at --ep-size 4.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The model gate sets the 8-device CPU mesh before jax loads, and holds the `Harness` this reuses.
base = load_by_path("test_nemotron_h_model", os.path.join(HERE, "test_nemotron_h_model.py"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import check_capture  # noqa: E402
import cpu_engine  # noqa: E402

TOL = base.TOL
MODEL = os.path.join(HERE, "nemotron3-model.patch")
HOOK = os.path.join(HERE, "nemotron3-capture-hook.patch")
PROBE = os.path.join(HERE, "nemotron3-probe.patch")
MULTIHOST = os.path.join(REPO_ROOT, "upstream", "multihost-hidden-states.patch")

# One entry per on-chip probe setting, as the environment an engine start sets for it.
SWITCHES = {
    "topk=jnp": {"SGL_PROBE_TOPK": "jnp"},
    "gmm=xla": {"SGL_PROBE_GMM": "xla"},
    "gmm=xla32": {"SGL_PROBE_GMM": "xla32"},
    "router=f32highest": {"SGL_PROBE_ROUTER": "f32highest"},
    "latent=f32": {"SGL_PROBE_LATENT": "f32"},
    "moe=dense": {"SGL_PROBE_MOE": "dense"},
    "shared=f32": {"SGL_PROBE_SHARED": "f32"},
    "moe_out=f32": {"SGL_PROBE_MOE_OUT": "f32"},
    "matmul=highest": {"SGL_PROBE_MATMUL": "highest"},
    "moe=dense+shared=f32+moe_out=f32": {
        "SGL_PROBE_MOE": "dense",
        "SGL_PROBE_SHARED": "f32",
        "SGL_PROBE_MOE_OUT": "f32",
    },
    "all": {
        "SGL_PROBE_TOPK": "jnp",
        "SGL_PROBE_GMM": "xla32",
        "SGL_PROBE_ROUTER": "f32highest",
        "SGL_PROBE_LATENT": "f32",
        "SGL_PROBE_MOE": "dense",
        "SGL_PROBE_SHARED": "f32",
        "SGL_PROBE_MOE_OUT": "f32",
        "SGL_PROBE_MATMUL": "highest",
    },
}
PROBE_VARS = ("SGL_PROBE_TOPK", "SGL_PROBE_GMM", "SGL_PROBE_ROUTER", "SGL_PROBE_LATENT",
              "SGL_PROBE_MOE", "SGL_PROBE_SHARED", "SGL_PROBE_MOE_OUT", "SGL_PROBE_MATMUL")

# TINY on an eight-way tensor axis with the experts split eight ways, one a shard, the way Ultra
# serves at --tp-size 32 --ep-size 32 with 16 a shard. Eight Mamba-2 heads and eight KV heads, so
# every axis the tensor axis splits holds eight.
TINY_TP8 = dict(base.TINY, num_attention_heads=8, num_key_value_heads=8, mamba_num_heads=8,
                ep_size=8)

failures = []


def check(ok: bool, text: str):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    if not ok:
        failures.append(text)


def control(detected: bool, text: str):
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}", flush=True)
    if not detected:
        failures.append(f"control: {text}")


@contextlib.contextmanager
def probe_env(values: dict):
    """Clear every probe variable, set `values` for the block, and restore the old ones after."""
    saved = {name: os.environ.get(name) for name in PROBE_VARS}
    for name in PROBE_VARS:
        os.environ.pop(name, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def patched(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def forward(harness, ids):
    """Every capture slot, the final state and the logits, as one list of arrays."""
    streams, final, logits = harness.run(ids, [len(ids)], harness.new_pool())
    return list(streams) + [final, logits]


def gap_of(a, b) -> float:
    return base.worst_of(base.gap(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Check 2: every switch matches the default path in float32.
# ---------------------------------------------------------------------------


def nudges(moe_probe, gate_module):
    """Per switch, a context that nudges the code the switch runs by about 1%."""
    from sgl_jax.srt.models import nemotron_h

    def scaled(fn):
        return lambda *a, **k: fn(*a, **k) * 1.01

    moe_class = nemotron_h.NemotronHLatentMoE

    moe_probe_sigmoid = moe_probe.exact_sigmoid

    def topk_nudge():
        original = gate_module.TopK._biased_grouped_topk_jax

        def nudged(self, *a, **k):
            # TopK renormalizes the weights afterwards, so a uniform scale would cancel.
            weights, ids = original(self, *a, **k)
            return weights * (1.0 + 0.01 * jnp.arange(weights.shape[-1])), ids

        return patched(gate_module.TopK, "_biased_grouped_topk_jax", nudged)

    return {
        # On CPU the kernel is off either way, so the nudge goes on the path the switch picks.
        "topk=jnp": topk_nudge,
        "gmm=xla": lambda: patched(moe_probe, "dense_gmm", scaled(moe_probe.dense_gmm)),
        "gmm=xla32": lambda: patched(moe_probe, "dense_gmm", scaled(moe_probe.dense_gmm)),
        # A power, since a uniform scale of the scores cancels in the renormalization.
        "router=f32highest": lambda: patched(
            moe_probe, "exact_sigmoid", lambda x: moe_probe_sigmoid(x) ** 1.1),
        "latent=f32": lambda: patched(moe_probe, "highest_dot", scaled(moe_probe.highest_dot)),
        "moe=dense": lambda: patched(moe_probe, "dense_experts", scaled(moe_probe.dense_experts)),
        "shared=f32": lambda: patched(moe_class, "_probe_shared_f32",
                                      scaled(moe_class._probe_shared_f32)),
    }


def switches_match(repo, ids, label, **harness_kwargs):
    harness = base.Harness(repo, **harness_kwargs)
    from sgl_jax.srt.layers import gate as gate_module
    from sgl_jax.srt.layers import moe_probe

    with probe_env({}):
        default = forward(harness, ids)
    reference = base.reference_forward(harness.reference, ids)
    against_hf = gap_of(default[:-2], reference[0])
    check(against_hf <= TOL, f"{label}: the default path matches transformers, {against_hf:.3e}")
    table = nudges(moe_probe, gate_module)
    for name, env in SWITCHES.items():
        with probe_env(env):
            got = forward(harness, ids)
            error = gap_of(got, default)
            check(error <= TOL, f"{label}: {name} against the default path, max abs {error:.3e}")
            if name in table:
                with table[name]():
                    moved = gap_of(forward(harness, ids), default)
                control(moved > TOL,
                        f"{label}: {name} with its code nudged 1% moves the forward {moved:.3e}")


def moe_out_reaches_the_stream(repo, ids):
    """In BF16, `SGL_PROBE_MOE_OUT=f32` hands the capture a float32 slot after each MoE layer.

    The float32 `Harness` can't see this switch, since everything there is float32 already. In
    BF16 the slot leaving a MoE layer comes back float32 and moves off the default path's BF16
    slot by less than BF16's rounding, and the slot entering the first MoE layer stays the same.
    """
    harness = base.Harness(repo, dtype=jnp.bfloat16)
    with probe_env({}):
        default = forward(harness, ids)
    with probe_env({"SGL_PROBE_MOE_OUT": "f32"}):
        got = forward(harness, ids)
    moe_slots = [k + 1 for k, kind in enumerate(harness.cfg.layers_block_type) if kind == "moe"
                 and k + 1 < len(harness.cfg.layers_block_type)]
    dtypes = [str(got[k].dtype) for k in moe_slots]
    before = base.gap(np.asarray(got[1], np.float32), np.asarray(default[1], np.float32))
    moved = [base.gap(np.asarray(got[k], np.float32), np.asarray(default[k], np.float32))
             for k in moe_slots]
    scale = [float(np.abs(np.asarray(default[k], np.float32)).max()) for k in moe_slots]
    check(all(d == "float32" for d in dtypes) and before == 0.0
          and all(0.0 < m <= 2 ** -7 * sc for m, sc in zip(moved, scale)),
          f"BF16: moe_out=f32 makes slots {moe_slots} float32 ({dtypes}), moves them "
          f"{[f'{m:.2e}' for m in moved]} (under BF16 rounding of {[f'{x:.2f}' for x in scale]}), "
          f"and leaves slot 1 alone ({before:g})")
    control(str(default[moe_slots[0]].dtype) == "bfloat16",
            f"the default path's slot {moe_slots[0]} is {default[moe_slots[0]].dtype}")


def matmul_stops_at_the_kernels(repo, ids):
    """`SGL_PROBE_MATMUL=highest` reaches the block's own dots and not the Pallas calls.

    On a TPU the setting inside gmm's trace fails Mosaic ("Bad lhs type"). The CPU runs gmm in
    interpret mode, which accepts it, so this reads the setting each call sees instead: the
    shared expert's `__call__` has to see "highest", gmm and the top-k have to see nothing.
    """
    harness = base.Harness(repo)
    from sgl_jax.srt.layers import gate as gate_module
    from sgl_jax.srt.layers import moe as moe_module
    from sgl_jax.srt.models import nemotron_h

    seen = {"gmm": set(), "topk": set(), "shared": set()}

    def recording(key, fn):
        def wrapped(*a, **k):
            seen[key].add(jax.config.jax_default_matmul_precision)
            return fn(*a, **k)
        return wrapped

    def run(env):
        for key in seen:
            seen[key] = set()
        with probe_env(env), \
                patched(moe_module, "gmm", recording("gmm", moe_module.gmm)), \
                patched(gate_module.TopK, "__call__",
                        recording("topk", gate_module.TopK.__call__)), \
                patched(nemotron_h.NemotronHMLP, "__call__",
                        recording("shared", nemotron_h.NemotronHMLP.__call__)):
            forward(harness, ids)
        return {key: sorted(map(str, values)) for key, values in seen.items()}

    on = run({"SGL_PROBE_MATMUL": "highest"})
    check(on == {"gmm": ["None"], "topk": ["None"], "shared": ["highest"]},
          f"matmul=highest: the shared expert runs at highest, gmm and the top-k at the "
          f"default: {on}")
    off = run({})
    control(off["shared"] == ["None"], f"without the switch the shared expert sees {off['shared']}")


def dense_experts_matches_a_gather():
    """`dense_experts` over every shard against a per-token gather of the top-k experts, float64.

    The reference is the formulation the switch stands for: per token, take its top-k experts'
    matrices, run each, and add them with the top-k weights. 37 tokens, 8 experts, top 3, on two
    shards of 4 local experts, ungated squared ReLU and gated SiLU. Control: one routing id
    changed moves the sum.
    """
    from sgl_jax.srt.layers import moe_probe

    rng = np.random.default_rng(5)
    t, k, n, experts, top = 37, 16, 12, 8, 3
    x = rng.standard_normal((t, k))
    ids = np.stack([rng.choice(experts, top, replace=False) for _ in range(t)]).astype(np.int32)
    weights = rng.random((t, top))
    w0 = rng.standard_normal((experts, k, n)) * 0.3
    w1 = rng.standard_normal((experts, k, n)) * 0.3
    wo = rng.standard_normal((experts, n, k)) * 0.3

    def relu2(h0, h1):
        return np.square(np.maximum(h0, 0.0))

    def silu_gated(h0, h1):
        return h0 / (1.0 + np.exp(-h0)) * h1

    def gathered(act, gated, route):
        out = np.zeros((t, k))
        for i in range(t):
            for j in range(top):
                e = route[i, j]
                h1 = x[i] @ w1[e] if gated else None
                out[i] += weights[i, j] * (act(x[i] @ w0[e], h1) @ wo[e])
        return out

    def jax_act(name):
        def act(h0, h1):
            y = moe_probe.expert_activation(name, h0)
            return y if h1 is None else y * h1
        return act

    def sharded(name, gated, route):
        total = 0.0
        for first in (0, 4):
            part = moe_probe.dense_experts(
                jnp.asarray(x, jnp.float32), jnp.asarray(weights, jnp.float32),
                jnp.asarray(route), jnp.asarray(w0[first:first + 4], jnp.float32),
                jnp.asarray(w1[first:first + 4], jnp.float32) if gated else None,
                jnp.asarray(wo[first:first + 4], jnp.float32), jnp.asarray(first, jnp.int32),
                jax_act(name))
            total = total + np.asarray(part, np.float64)
        return total

    for name, act, gated in (("relu2", relu2, False), ("silu", silu_gated, True)):
        want = gathered(act, gated, ids)
        error = base.gap(sharded(name, gated, ids), want)
        check(error <= TOL, f"dense_experts, {name}{' gated' if gated else ''}, over two shards "
              f"against a per-token gather in float64: max abs {error:.3e}")
    other = ids.copy()
    other[3, 1] = next(e for e in range(experts) if e not in ids[3])
    moved = base.gap(sharded("relu2", False, other), gathered(relu2, False, ids))
    control(moved > TOL, f"token 3's second expert swapped moves the sum {moved:.3e}")


# The reproduction packs two requests of 80 and 48 tokens, 128 in all. At top 3 that routes 384
# rows into the experts, three whole 128-row tiles: the gmm v1 kernel a CPU runs refuses 192
# ("192 must be divisible by x-dimension tile size (128)", 2026-09-27), so 64 tokens can't run.
REPRO_LENS = [80, 48]


def per_request_reference(harness, ids, seq_lens):
    """The reference's layer inputs, final state and logits, run one request at a time."""
    layers, finals, logits = [], [], []
    start = 0
    for length in seq_lens:
        got = base.reference_forward(harness.reference, ids[start:start + length])
        layers.append(got[0])
        finals.append(got[1])
        logits.append(got[2])
        start += length
    stacked = [np.concatenate(group, axis=0) for group in zip(*layers)]
    return stacked + [np.concatenate(finals, axis=0), np.concatenate(logits, axis=0)]


def expert_parallel_repro(repo, rng_seed=7):
    """Two packed requests at 128 tokens against transformers, per request, per mesh and setting.

    The GLM-5.3 CPU gate saw the second prompt's tokens move by up to 9% at --ep-size 4 on the
    real engine (upstream/test_glm5_tp_sharding.py, check 5's note). This runs the same shape of
    batch through the tiny Nemotron 3 at tp 1, tp 4 with ep 4, and tp 8 with ep 8, with the
    default EPMoE path, with SGL_PROBE_GMM=xla (gmm swapped, dispatch kept), and with
    SGL_PROBE_MOE=dense (no dispatch). The dense path has to match transformers everywhere; the
    other two are reported per request.
    """
    rng = np.random.default_rng(rng_seed)
    ids = rng.integers(1, base.TINY["vocab_size"], sum(REPRO_LENS)).astype(np.int32)
    meshes = (("tp=1", dict()), ("tp=4 ep=4", dict(tp=4, config=dict(base.TINY_TP, ep_size=4))),
              ("tp=8 ep=8", dict(tp=8, config=TINY_TP8)))
    settings = (("default", {}), ("gmm=xla", {"SGL_PROBE_GMM": "xla"}),
                ("moe=dense", {"SGL_PROBE_MOE": "dense"}))
    split = REPRO_LENS[0]
    for label, kwargs in meshes:
        harness = base.Harness(repo, **kwargs)
        want = per_request_reference(harness, ids, REPRO_LENS)
        for name, env in settings:
            with probe_env(env):
                streams, final, logits = harness.run(ids, REPRO_LENS, harness.new_pool())
            got = list(streams) + [final, logits]
            first = base.worst_of(base.gap(a[:split], b[:split]) for a, b in zip(got, want))
            second = base.worst_of(base.gap(a[split:], b[split:]) for a, b in zip(got, want))
            text = (f"{label}, {name}, 128 tokens in two requests against transformers: request 0 "
                    f"max abs {first:.3e}, request 1 {second:.3e}")
            if name == "moe=dense":
                check(max(first, second) <= TOL, text)
            else:
                print(f"  note: {text}", flush=True)


# ---------------------------------------------------------------------------
# Check 3: dense_gmm against gmm.
# ---------------------------------------------------------------------------


def dense_gmm_matches():
    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
    from sgl_jax.srt.layers import moe_probe

    rng = np.random.default_rng(3)
    sizes = np.array([5, 0, 9, 3, 12, 7, 0, 12], np.int32)
    m, k, n = int(sizes.sum()), 128, 128
    lhs = jnp.asarray(rng.standard_normal((m, k)), jnp.float32)
    rhs_all = jnp.asarray(rng.standard_normal((8, k, n)) * 0.1, jnp.float32)
    for offset, local in ((0, 8), (0, 4), (4, 4)):
        rhs = rhs_all[offset:offset + local]
        want = gmm(lhs, rhs, jnp.asarray(sizes), preferred_element_type=jnp.float32,
                   group_offset=jnp.asarray(offset, jnp.int32), zero_initialize=True,
                   interpret=True)
        got = moe_probe.dense_gmm(lhs, rhs, jnp.asarray(sizes), jnp.asarray(offset, jnp.int32),
                                  jnp.float32)
        error = base.gap(got, want)
        start, end = int(sizes[:offset].sum()), int(sizes[:offset + local].sum())
        outside = float(np.abs(np.asarray(got)[np.r_[0:start, end:m]]).max(initial=0.0))
        check(error <= TOL and outside == 0.0,
              f"dense_gmm at group offset {offset} over {local} local groups against gmm: max abs "
              f"{error:.3e}, rows outside the local groups {outside:g}")
    shifted = sizes.copy()
    shifted[2] -= 1
    shifted[3] += 1
    moved = base.gap(moe_probe.dense_gmm(lhs, rhs_all, jnp.asarray(shifted), None, jnp.float32),
                     gmm(lhs, rhs_all, jnp.asarray(sizes), preferred_element_type=jnp.float32,
                         interpret=True))
    control(moved > TOL, f"one row moved from group 2 to group 3 lands at {moved:.3e}")


# ---------------------------------------------------------------------------
# Check 4: the Pallas grouped top-k against the JAX path.
# ---------------------------------------------------------------------------


def grouped_topk_matches():
    from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas
    from sgl_jax.srt.layers import gate as gate_module

    tokens, experts, top = 256, 512, 22
    rng = np.random.default_rng(11)
    logits = jax.nn.sigmoid(jnp.asarray(rng.standard_normal((tokens, experts)), jnp.float32))
    bias = jnp.asarray(rng.standard_normal(experts) * 0.1, jnp.float32)
    topk = gate_module.TopK(topk=top, renormalize=True, num_expert_group=1, topk_group=1,
                            routed_scaling_factor=5.0)
    want_w, want_ids = topk._biased_grouped_topk_jax(logits, bias)
    got_w, got_ids = grouped_topk_pallas(logits, bias, num_expert_group=1, topk_group=1,
                                         topk=top, interpret=True)
    order_w = np.argsort(np.asarray(want_ids), axis=1)
    order_g = np.argsort(np.asarray(got_ids), axis=1)
    same_sets = np.array_equal(np.take_along_axis(np.asarray(want_ids), order_w, 1),
                               np.take_along_axis(np.asarray(got_ids), order_g, 1))
    error = base.gap(np.take_along_axis(np.asarray(want_w), order_w, 1),
                     np.take_along_axis(np.asarray(got_w), order_g, 1))
    check(same_sets and error <= TOL,
          f"the Pallas grouped top-k in interpret mode picks the JAX path's experts for all "
          f"{tokens} tokens ({same_sets}) with weights {error:.3e} apart")

    on = types.SimpleNamespace(enable_topk_kernel=True, device="tpu")
    with patched(gate_module, "get_global_server_args", lambda: on):
        with probe_env({}):
            kernel_on = gate_module._topk_kernel_enabled()
        with probe_env({"SGL_PROBE_TOPK": "jnp"}):
            kernel_off = not gate_module._topk_kernel_enabled()
    check(kernel_off,
          "SGL_PROBE_TOPK=jnp turns the kernel off where the server arguments turn it on")
    control(kernel_on, "without the variable the same server arguments turn the kernel on")


# ---------------------------------------------------------------------------
# Check 5: a misspelled value raises.
# ---------------------------------------------------------------------------


def typo_raises():
    from sgl_jax.srt.layers import moe_probe

    with probe_env({"SGL_PROBE_GMM": "xal"}):
        try:
            moe_probe.probe("GMM")
            raised = False
        except ValueError:
            raised = True
    check(raised, "SGL_PROBE_GMM=xal raises")
    with probe_env({"SGL_PROBE_GMM": "XLA"}):
        check(moe_probe.active() == {"GMM": "xla"}, "SGL_PROBE_GMM=XLA reads as xla")
    with probe_env({}):
        control(moe_probe.active() == {}, "with nothing set no switch is on")


# ---------------------------------------------------------------------------
# Check 6: check_capture.py --engine-layers on the real engine.
# ---------------------------------------------------------------------------

# The tiny config with a vocabulary that holds cpu_engine's 400-token tokenizer, and top 2 of 8
# experts. The gmm v1 kernel a CPU runs takes a routed row count above 128 only in whole 128-row
# tiles, and the engine pads a prefill pass to 64 tokens, so 2 copies a token fill whole tiles
# where TINY's 3 leave 192 rows.
# TINY's `ME*E`, cut to `ME*`. The engine sizes its KV pool per attention layer, and a stack with
# none dies dividing by a zero cell size in `profile_max_num_token`, so the cut keeps layer 2.
# head_dim is 128: the KV pool rounds a head up to 128 wide, and at eb061d8 the native backend
# hands o_proj that padded width, the limit `cpu_engine.py`'s Qwen3 notes too.
ENGINE_TINY = dict(base.TINY, vocab_size=512, num_experts_per_tok=2, head_dim=128,
                   bos_token_id=None, eos_token_id=0, pad_token_id=0)
WORDS = cpu_engine.WORDS


def write_engine_checkpoint(path: str, fc2_scale: float = 1.0) -> str:
    """A tiny nemotron_h checkpoint under the published names, plus the CPU gates' tokenizer."""
    from safetensors.numpy import load_file, save_file

    from sgl_jax.srt.configs.nemotron_h import NemotronHConfig

    shell = base.Harness.__new__(base.Harness)
    shell.settings = ENGINE_TINY
    shell.cfg = NemotronHConfig(**ENGINE_TINY)
    base.write_checkpoint(shell, path, scale=0.1)
    if fc2_scale != 1.0:
        # Layer 1 of `ME*E` is the LatentMoE block.
        file = os.path.join(path, "model.safetensors")
        tensors = load_file(file)
        key = "backbone.layers.1.mixer.fc2_latent_proj.weight"
        tensors[key] = tensors[key] * fc2_scale
        save_file(tensors, file)
    cpu_engine.write_tokenizer(path)
    return path


def capture_command(model, prompts, *extra):
    return [
        sys.executable, os.path.join(REPO_ROOT, "scripts", "check_capture.py"),
        "--model-path", model, "--tp-size", "1", "--batch-size", "8", "--token-padding", "64",
        "--prompts-file", prompts, "--num-prompts", "1", "--chunked-prefill-size", "64",
        "--engine-arg", "device=cpu", "--engine-arg", "max_total_tokens=4096",
        "--engine-arg", "log_level=error", "--engine-arg", "disable_overlap_schedule=true",
        "--engine-arg", "random_seed=0", "--engine-arg", "disable_radix_cache=true",
        *extra,
    ]


def results_of(stdout: str) -> list:
    return [json.loads(line[len("RESULT "):]) for line in stdout.splitlines()
            if line.startswith("RESULT ")]


def engine_layers_cut(repo, workdir):
    if os.path.join(repo, "python") not in sys.path:
        sys.path.insert(0, os.path.join(repo, "python"))
    rng = np.random.default_rng(1)
    prompts = os.path.join(workdir, "prompts.txt")
    with open(prompts, "w") as fp:
        lines = [" ".join(rng.choice(WORDS) for _ in range(n)) for n in (14, 22, 25, 20)]
        fp.write("\n".join(lines) + "\n")
    model = write_engine_checkpoint(os.path.join(workdir, "nemo-tiny"))
    other = write_engine_checkpoint(os.path.join(workdir, "nemo-tiny-scaled"), fc2_scale=1.5)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(repo, "python"), os.path.join(REPO_ROOT, "scripts")]), JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)
    for name in PROBE_VARS:
        env.pop(name, None)

    def run(args, timeout=1800):
        done = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
        return done

    npz = os.path.join(workdir, "ref.npz")
    scaled_npz = os.path.join(workdir, "ref-scaled.npz")
    for source, path in ((model, npz), (other, scaled_npz)):
        done = run(capture_command(source, prompts, "--reference-only", path))
        check(done.returncode == 0 and os.path.exists(path),
              f"--reference-only writes {os.path.basename(path)} from {os.path.basename(source)}, "
              f"exit {done.returncode}")
        if done.returncode != 0:
            print("      its stdout:\n" + done.stdout + "\n      its stderr:\n" + done.stderr)
            return
    entries = len(np.load(npz)["reference0"])
    check(entries == 5, f"the whole-model reference holds {entries} entries: 4 layer inputs and "
          "the final norm's output")

    cut = run(capture_command(model, prompts, "--reference-npz", npz, "--engine-layers", "3"))
    results = results_of(cut.stdout)
    ok = (cut.returncode == 0 and "All checks passed." in cut.stdout and len(results) == 2
          and all(r["layers"] == 3 and len(r["ratios"]) == 3 for r in results))
    check(ok, f"--engine-layers 3 passes on slots 0 to 2 of both prompts, exit {cut.returncode}, "
          f"ratios {[r['ratios'] for r in results]}")
    if not ok:
        print("      its stdout:\n" + cut.stdout + "\n      its stderr:\n" + cut.stderr)
    shapes = [line for line in cut.stdout.splitlines() if line.startswith("engine capture: ")]
    check(len(shapes) == 1 and shapes[0].count(", 3, 64)") == 2,
          f"the engine built three layers: {shapes}")

    wrong = run(capture_command(model, prompts, "--reference-npz", scaled_npz,
                                "--engine-layers", "3"))
    results = results_of(wrong.stdout)
    failed = [sorted(r["failed_layers"].get("median", [])) for r in results]
    control(wrong.returncode != 0 and len(results) == 2 and all(f == [2] for f in failed),
            f"a reference with layer 1 scaled fails slot 2 alone: failed {failed}, exit "
            f"{wrong.returncode}")
    if not (len(results) == 2 and all(f == [2] for f in failed)):
        print("      its stdout:\n" + wrong.stdout + "\n      its stderr:\n" + wrong.stderr)

    past = run(capture_command(model, prompts, "--reference-npz", npz, "--engine-layers", "5"),
               timeout=600)
    control(past.returncode != 0 and "holds 5 entries" in past.stderr
            and "engine capture" not in past.stdout,
            f"--engine-layers 5 stops before the engine, exit {past.returncode}")

    override = check_capture.engine_layer_override('{"a": 1}', 3)
    check(json.loads(override) == {"a": 1, "num_hidden_layers": 3},
          f"an override passed with --engine-arg keeps its keys: {override}")
    try:
        check_capture.engine_layer_override({"num_hidden_layers": 4}, 3)
        clash = False
    except ValueError:
        clash = True
    control(clash, "an override that sets another num_hidden_layers raises")


def main():
    with tempfile.TemporaryDirectory() as workdir:
        print("check 1: the probe patch applies on the model patch and its hook, and compiles")
        repo = base.prepare_repo(workdir, patches=[MODEL, HOOK, PROBE])
        with tempfile.TemporaryDirectory() as second:
            base.prepare_repo(second, patches=[MODEL, HOOK, MULTIHOST, PROBE])
        check(True, "the probe patch applies after multihost-hidden-states.patch too")

        ids = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59], np.int32)
        print("check 2: every switch against the default path in float32")
        switches_match(repo, ids, "tp=1")
        switches_match(repo, ids, "tp=4", tp=4, config=base.TINY_TP)
        switches_match(repo, ids, "tp=4 ep=4", tp=4, config=dict(base.TINY_TP, ep_size=4))
        switches_match(repo, ids, "tp=8 ep=8", tp=8, config=TINY_TP8)
        moe_out_reaches_the_stream(repo, ids)
        matmul_stops_at_the_kernels(repo, ids)

        print("check 3: dense_gmm against gmm")
        dense_gmm_matches()

        print("check 4: the Pallas grouped top-k against the JAX path")
        grouped_topk_matches()

        print("check 5: a misspelled value raises")
        typo_raises()

        print("check 6: check_capture.py --engine-layers on the real engine")
        engine_layers_cut(repo, workdir)

        print("check 7: dense_experts against a per-token gather of the top-k experts")
        dense_experts_matches_a_gather()

        print("check 8: two packed requests at 128 tokens, per mesh, against transformers")
        expert_parallel_repro(repo)

    if failures:
        print(f"{len(failures)} check(s) failed:")
        for text in failures:
            print(f"  {text}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
