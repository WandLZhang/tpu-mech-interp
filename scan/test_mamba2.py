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

"""Correctness gate for the chunked Mamba-2 selective scan.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 scan/test_mamba2.py

Every state case compares two things against one float32 token-by-token
reference, `mamba2.sequential_reference` fed the same `dt`: the chunk pairs
folded on a single device, and the same pairs run through the sharded path in
`affine_scan.py`. The output case, the discretization, the conv and the cache
transpose each compare against a float64 numpy oracle written from the
published mixer rather than from this repo. Then it runs negative controls that
each break one step. A control must fail; if one passes, this file fails
itself.

A control has to clear the same bar the implementation has to pass, so
`CONTROL_MIN` equals `REL_TOL`.

`A` and `dt_bias` are drawn the way the reference initializer draws them, so
`A` spans `1 .. mamba_num_heads` and `dt` spans the `time_step` range. Heads at
the top of that range forget a chunk of state completely and heads at the
bottom keep nearly all of it. That spread is why the error metric takes a
denominator per head: one global denominator hides a small head's error behind
a large head's magnitude.

`errors` runs in float64, and CPU `dot_general` is true float32, so this file
can't measure what TPU precision costs. It reads the jaxpr instead and fails on
any `dot_general` that leaves the precision open. What it can measure is the
float32 `exp` and `log1p` that softplus runs through, and `run_discretize`
prints both for the platform it runs on.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testing import force_host_devices, sharded_states, too_few_devices  # noqa: E402

# Before jax starts a backend.
force_host_devices(8)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from affine_scan import (  # noqa: E402
    compose_local,
    sequential_reference as chunk_reference,
)
from mamba2 import (  # noqa: E402
    NEMOTRON_3_SUPER,
    as_cache_layout,
    causal_conv,
    chunk_outputs,
    chunk_pairs,
    conv_halo,
    discretize,
    entering_states,
    from_cache_layout,
    pad_to_chunks,
    sequential_reference,
)

AXIS = "ctx"
# One bound follows the platform for both kinds of reference: the float32 token-by-token
# recurrence the state cases use, and the float64 oracles the rest use. softplus runs through
# float32 `exp` and `log1p`, and `run_discretize` prints what each costs on the platform: about
# 1e-7 on CPU. On a TPU v5e, dt lands at 2.2e-4 against its float64 oracle before any scan runs.
# The matmuls aren't the cause: they all ask for HIGHEST.
_BACKEND = jax.default_backend()
REL_TOL = {"cpu": 1e-4}.get(_BACKEND, 5e-4)
CONTROL_MIN = REL_TOL
# 8 shards against one device, same platform, same arithmetic reassociated. This bound doesn't
# follow the platform, so a looser REL_TOL reaches accuracy and never the sharding.
SHARD_TOL = 1e-5
# How much worse a BF16 state must come out. Both sides are measured against the float32
# token-by-token reference, so the FP32 side carries the platform's floor and the gap narrows
# where that floor is higher.
BF16_COST = {"cpu": 1e3}.get(_BACKEND, 1e2)


def make_problem(seed, tokens, heads, groups, state_size, head_dim, cfg=NEMOTRON_3_SUPER):
    """One layer's SSM inputs, drawn the way the initializer draws them.

    `x`, `B` and `C` come out of the input projection in BF16. `dt_bias`, `A_log`
    and `D` are stored parameters. The state is float32.

    The reference sets `A = arange(1, mamba_num_heads + 1)` and stores
    `log(A)`, so a case with fewer heads spreads them over that same range and
    still spans the decay the model spans. `dt_bias` is the inverse softplus of
    `exp(U(log time_step_min, log time_step_max))`, again the reference draw,
    and `dt_raw` is a unit scale projection output. Per token `dt` reaches the
    floor over part of the sequence, which is the regime the layer runs in.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(tokens, heads, head_dim)).astype(np.float32) * 0.5
    b = rng.normal(size=(tokens, groups, state_size)).astype(np.float32) * 0.5
    c = rng.normal(size=(tokens, groups, state_size)).astype(np.float32) * 0.5
    d_skip = rng.normal(size=(heads,)).astype(np.float32)

    a = np.linspace(1.0, cfg.mamba_num_heads, heads)
    a_log = np.log(a)

    dt_target = np.exp(
        rng.uniform(math.log(cfg.time_step_min), math.log(cfg.time_step_max), size=heads)
    )
    dt_bias = dt_target + np.log(-np.expm1(-dt_target))
    dt_raw = rng.normal(size=(tokens, heads)).astype(np.float32)

    h0 = rng.normal(size=(heads, state_size, head_dim)).astype(np.float32) * 0.1

    return dict(
        x=jnp.asarray(x, dtype=jnp.bfloat16),
        b=jnp.asarray(b, dtype=jnp.bfloat16),
        c=jnp.asarray(c, dtype=jnp.bfloat16),
        dt_raw=jnp.asarray(dt_raw),
        dt_bias=jnp.asarray(dt_bias, dtype=jnp.float32),
        a_log=jnp.asarray(a_log, dtype=jnp.float32),
        d=jnp.asarray(d_skip),
        h0=jnp.asarray(h0),
    )


# --------------------------------------------------------------------------
# Oracles. Each one is a float64 numpy transcription of the published mixer,
# written from the definition and calling nothing in `mamba2.py`.
# --------------------------------------------------------------------------


def discretize_oracle(dt_raw, dt_bias, a_log, limit):
    """softplus(dt + bias), clamped to `limit`, times `A = -exp(A_log)`."""
    z = np.asarray(dt_raw, np.float64) + np.asarray(dt_bias, np.float64)
    dt = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    dt = np.clip(dt, limit[0], limit[1])
    a = -np.exp(np.asarray(a_log, np.float64))
    return dt, dt * a


def recurrence_oracle(x, b, c, dt, log_decay, h0, d=None):
    """The whole layer, one token at a time, in float64.

    Returns the state after every token and the output of every token.
    """
    xf = np.asarray(x, np.float64)
    heads = xf.shape[1]
    groups = b.shape[1]
    bf = np.repeat(np.asarray(b, np.float64), heads // groups, axis=1)
    cf = np.repeat(np.asarray(c, np.float64), heads // groups, axis=1)
    dtf = np.asarray(dt, np.float64)
    decay = np.exp(np.asarray(log_decay, np.float64))

    h = np.asarray(h0, np.float64).copy()
    states, outputs = [], []
    for t in range(xf.shape[0]):
        xd = xf[t] * dtf[t][:, None]
        h = decay[t][:, None, None] * h + bf[t][:, :, None] * xd[:, None, :]
        y = np.einsum("hn,hnp->hp", cf[t], h)
        if d is not None:
            y = y + np.asarray(d, np.float64)[:, None] * xf[t]
        states.append(h.copy())
        outputs.append(y)
    return np.stack(states), np.stack(outputs)


def conv_oracle(x, weight, bias, prefix=None):
    """Depthwise causal conv from the definition: tap `k-1` reads token `t`."""
    xf = np.asarray(x, np.float64)
    wf = np.asarray(weight, np.float64)
    bf = np.asarray(bias, np.float64)
    kernel, dim = wf.shape
    left = np.zeros((kernel - 1, dim)) if prefix is None else np.asarray(prefix, np.float64)
    padded = np.concatenate([left, xf], axis=0)

    out = np.zeros((xf.shape[0], dim))
    for t in range(xf.shape[0]):
        acc = bf.copy()
        for back in range(kernel):
            acc = acc + wf[kernel - 1 - back] * padded[kernel - 1 + t - back]
        out[t] = acc
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def errors(got, want, head_axis=None):
    """Max absolute error, worst per-head relative error, and cosine.

    With `head_axis` set, every head gets its own denominator, so a slow head
    whose state peaks two orders below the fastest head still reports its own
    error. The denominator has a floor at 1e-6 of the global magnitude, which
    keeps a head that holds nothing from dividing by near zero.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    diff = np.abs(got - want)
    max_abs = float(diff.max())
    scale = max(float(np.abs(want).max()), 1e-12)

    if head_axis is None:
        rel = max_abs / scale
    else:
        heads = want.shape[head_axis]
        per_diff = np.moveaxis(diff, head_axis, 0).reshape(heads, -1).max(axis=1)
        per_want = np.moveaxis(np.abs(want), head_axis, 0).reshape(heads, -1).max(axis=1)
        rel = float(np.max(per_diff / np.maximum(per_want, 1e-6 * scale)))

    gf, wf = got.ravel(), want.ravel()
    cos = float(gf @ wf / max(np.linalg.norm(gf) * np.linalg.norm(wf), 1e-12))
    return max_abs, rel, cos


class Report:
    """Counts failures and prints one line each."""

    def __init__(self):
        self.failures = 0

    def check(self, ok, label, detail=""):
        self.failures += 0 if ok else 1
        print(f"      [{'PASS' if ok else 'FAIL'}] {label}   {detail}")

    def measure(self, label, got, want, head_axis=None, tol=REL_TOL):
        max_abs, rel, cos = errors(got, want, head_axis)
        self.check(rel < tol, label, f"max_abs={max_abs:.3e}  rel={rel:.3e}  cos={cos:.8f}")
        return rel

    def control(self, label, got, want, head_axis=None, bound=None):
        _, rel, cos = errors(got, want, head_axis)
        detected = rel > (CONTROL_MIN if bound is None else bound)
        print(
            f"      control ({label}): rel={rel:.3e} cos={cos:.6f}"
            f"  -> {'detected' if detected else 'NOT DETECTED'}"
        )
        if not detected:
            print("      FAIL: the control did not fail, so this test detects nothing.")
            self.failures += 1

    def control_raises(self, label, fn):
        try:
            fn()
        except ValueError as exc:
            print(f"      control ({label}): raised ValueError, {exc}  -> detected")
            return
        print(f"      control ({label}): returned  -> NOT DETECTED")
        print("      FAIL: the control did not fail, so this test detects nothing.")
        self.failures += 1


# --------------------------------------------------------------------------
# Broken variants, each one written out rather than patched into the source.
# --------------------------------------------------------------------------


def broken_pairs(x, b, dt, log_decay, chunk_size, mode):
    """Chunk pairs with one step of the fold broken.

    `flat` drops the intra-chunk decay, so every token in a chunk contributes
    with full weight. `inclusive` sums the log decay through token `t` instead
    of stopping before it, the off-by-one a segment sum invites.
    """
    tokens, heads, head_dim = x.shape
    state_size = b.shape[-1]
    chunks = tokens // chunk_size

    xf = np.asarray(x.astype(jnp.float32))
    bf = np.asarray(b.astype(jnp.float32))
    dtf = np.asarray(dt)
    gf = np.asarray(log_decay).reshape(chunks, chunk_size, heads)

    xd = (xf * dtf[..., None]).reshape(chunks, chunk_size, heads, head_dim)
    bh = np.repeat(bf, heads // b.shape[1], axis=1).reshape(chunks, chunk_size, heads, state_size)

    cum = np.cumsum(gf, axis=1)
    total = cum[:, -1, :]
    if mode == "flat":
        decay = np.ones_like(cum)
    elif mode == "inclusive":
        decay = np.exp(total[:, None, :] - cum + gf)
    else:
        raise ValueError(mode)

    a_chunks = np.exp(total)[..., None, None] * np.eye(state_size, dtype=np.float32)
    b_chunks = np.einsum("clhn,clhp->chnp", bh * decay[..., None], xd)
    return jnp.asarray(a_chunks), jnp.asarray(b_chunks)


def broken_outputs(x, b, c, dt, log_decay, h_entering, chunk_size, d, mode):
    """Chunk outputs with one term of the assembly broken.

    `acausal` drops the causal mask, so a token reads the tokens after it.
    `no_skip` drops the `D` term. `no_inherited` drops the state the chunk
    receives, which is the term that carries a sequence shard.
    """
    tokens, heads, head_dim = x.shape
    state_size = b.shape[-1]
    chunks = tokens // chunk_size

    xf = np.asarray(x.astype(jnp.float32), np.float64)
    dtf = np.asarray(dt, np.float64)
    bh = np.repeat(np.asarray(b.astype(jnp.float32), np.float64), heads // b.shape[1], axis=1)
    ch = np.repeat(np.asarray(c.astype(jnp.float32), np.float64), heads // c.shape[1], axis=1)

    xd = (xf * dtf[..., None]).reshape(chunks, chunk_size, heads, head_dim)
    bh = bh.reshape(chunks, chunk_size, heads, state_size)
    ch = ch.reshape(chunks, chunk_size, heads, state_size)

    cum = np.cumsum(np.asarray(log_decay, np.float64).reshape(chunks, chunk_size, heads), axis=1)
    gap = cum[:, :, None, :] - cum[:, None, :, :]
    causal = np.tril(np.ones((chunk_size, chunk_size), bool))[None, :, :, None]
    if mode == "acausal":
        decay = np.exp(np.minimum(gap, 0.0))
    else:
        decay = np.where(causal, np.exp(np.where(causal, gap, -np.inf)), 0.0)

    score = np.einsum("clhn,cshn->clsh", ch, bh)
    y = np.einsum("clsh,cshp->clhp", score * decay, xd)
    if mode != "no_inherited":
        inherited = ch * np.exp(cum)[..., None]
        y = y + np.einsum("clhn,chnp->clhp", inherited, np.asarray(h_entering, np.float64))
    y = y.reshape(tokens, heads, head_dim)
    if mode != "no_skip":
        y = y + np.asarray(d, np.float64)[None, :, None] * xf
    return y


# --------------------------------------------------------------------------
# The sharded path
# --------------------------------------------------------------------------


def sharded(a, b, h0, mesh, break_chain=False):
    """`_testing.sharded_states` on this file's mesh axis.

    `break_chain` is the negative control: every shard starts from `h0`, so
    any shard after the first inherits nothing.
    """
    return sharded_states(a, b, h0, mesh, AXIS, break_chain=break_chain)


def transcendental_errors(z):
    """float32 `exp` and `log1p` on this platform, against float64 on the same inputs.

    softplus(z) is max(z, 0) + log1p(exp(-|z|)), so `exp` runs on -|z| and
    `log1p` on what comes back. Each op gets float32 inputs, and the float64
    reference reads those same float32 values, so the figure is the op's own
    error. Each is the worst relative difference over the draw.
    """
    x32 = -np.abs(np.asarray(z, np.float64)).astype(np.float32)
    want_exp = np.exp(x32.astype(np.float64))
    got_exp = np.asarray(jnp.exp(jnp.asarray(x32)), np.float64)
    u32 = want_exp.astype(np.float32)
    want_log1p = np.log1p(u32.astype(np.float64))
    got_log1p = np.asarray(jnp.log1p(jnp.asarray(u32)), np.float64)
    # A value that underflows to zero has no relative error to read.
    live = want_log1p > 0
    exp_rel = float(np.max(np.abs(got_exp - want_exp)[live] / want_exp[live]))
    log1p_rel = float(np.max(np.abs(got_log1p - want_log1p)[live] / want_log1p[live]))
    return exp_rel, log1p_rel


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


def run_discretize(report):
    """`discretize` against the mixer, including both ends of the clamp."""
    print("  discretize")
    cfg = NEMOTRON_3_SUPER
    p = make_problem(21, 256, 8, 2, 16, 8)
    # Both published sizes leave `mamba_dt_limit` at (0.0, inf), where neither
    # end binds, so the clamp itself is checked against a finite pair.
    if cfg.time_step_limit != (0.0, math.inf):
        raise AssertionError(f"NEMOTRON_3_SUPER clamps dt to {cfg.time_step_limit}")
    limit = (cfg.time_step_min, math.inf)

    dt, log_decay = discretize(p["dt_raw"], p["dt_bias"], p["a_log"], limit)
    want_dt, want_g = discretize_oracle(p["dt_raw"], p["dt_bias"], p["a_log"], limit)

    floored = float(np.mean(np.asarray(want_dt) <= limit[0]))
    print(f"      floor moves {100 * floored:.2f}% of the (token, head) steps")
    report.check(floored > 0.005, "the floor is live on this draw", f"{100 * floored:.2f}%")

    # The two float32 ops softplus runs through, on the arguments this draw
    # hands them. Their error is what the platform bound has to cover.
    exp_rel, log1p_rel = transcendental_errors(
        np.asarray(p["dt_raw"], np.float64) + np.asarray(p["dt_bias"], np.float64)
    )
    report.check(
        exp_rel < REL_TOL and log1p_rel < REL_TOL,
        f"float32 exp and log1p on {_BACKEND} sit under the {REL_TOL:g} bound",
        f"exp rel={exp_rel:.2e}  log1p rel={log1p_rel:.2e}",
    )

    report.measure("dt        ", dt, want_dt, head_axis=1)
    report.measure("log decay ", log_decay, want_g, head_axis=1)
    report.check(dt.dtype == jnp.float32, "dt is float32", str(dt.dtype))

    # A config that ships a finite upper limit gets it.
    capped = (limit[0], 0.02)
    dt_cap, _ = discretize(p["dt_raw"], p["dt_bias"], p["a_log"], capped)
    want_cap, _ = discretize_oracle(p["dt_raw"], p["dt_bias"], p["a_log"], capped)
    clipped = float(np.mean(np.asarray(want_dt) >= capped[1]))
    report.measure("dt, capped", dt_cap, want_cap, head_axis=1)
    report.check(clipped > 0.005, "the cap is live on this draw", f"{100 * clipped:.2f}%")

    no_floor = discretize(p["dt_raw"], p["dt_bias"], p["a_log"], (0.0, math.inf))[0]
    no_bias = discretize(p["dt_raw"], jnp.zeros_like(p["dt_bias"]), p["a_log"], limit)[0]
    report.control("floor dropped", no_floor, want_dt, 1)
    report.control("cap ignored  ", dt, want_cap, 1)
    report.control("bias dropped ", no_bias, want_dt, 1)
    report.control("A not negated", -log_decay, want_g, 1)


def run_case(report, name, seed, chunks, chunk_size, heads, groups, state_size, head_dim, mesh):
    """Fold, shard, and break."""
    tokens = chunks * chunk_size
    p = make_problem(seed, tokens, heads, groups, state_size, head_dim)
    x, b, h0 = p["x"], p["b"], p["h0"]
    dt, log_decay = discretize(
        p["dt_raw"], p["dt_bias"], p["a_log"], NEMOTRON_3_SUPER.time_step_limit
    )

    want = sequential_reference(x, b, dt, log_decay, h0, stride=chunk_size)
    log_total = np.asarray(log_decay, np.float64).reshape(chunks, chunk_size, heads).sum(1)
    keep = np.exp(log_total)
    per_head = keep.mean(axis=0)
    widest = float(np.abs(log_total).max())

    print(
        f"  {name}  chunks={chunks:<3} chunk={chunk_size:<4} heads={heads} groups={groups}"
        f" N={state_size} P={head_dim}  chunk decay {per_head.min():.2e}..{per_head.max():.3f}"
    )

    # `chunk_terms` forms the intra-chunk decay as `exp(log_total - cum)`, a
    # difference of two running sums. The tail increments stay near 0.1 while
    # `log_total` runs to tens or hundreds, so the last tokens of a chunk come
    # out of a canceled subtraction. This pins the draw in that regime: a
    # narrower `A` or `dt` range would hide the cancellation rather than fix
    # it. The measurements below then carry the error it costs.
    report.check(
        widest > 20.0 and per_head.min() < 1e-3,
        "the wide-gate regime is live",
        f"|log_total| up to {widest:.0f}, slowest head keeps {per_head.max():.3f}"
        f" and fastest keeps {per_head.min():.2e}",
    )

    a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, chunk_size)
    one_device = chunk_reference(a_chunks, b_chunks, h0)
    rel_f32 = report.measure("chunk pairs, one device", one_device, want, 1)
    got_shards = sharded(a_chunks, b_chunks, h0, mesh)
    report.measure("chunk pairs, 8 shards  ", got_shards, want, 1)
    report.measure("8 shards match one device", got_shards, one_device, 1, tol=SHARD_TOL)

    report.control(
        "intra-chunk decay dropped",
        sharded(*broken_pairs(x, b, dt, log_decay, chunk_size, "flat"), h0, mesh),
        want,
        1,
    )
    report.control(
        "segment sum off by one   ",
        sharded(*broken_pairs(x, b, dt, log_decay, chunk_size, "inclusive"), h0, mesh),
        want,
        1,
    )
    broken_chain = sharded(a_chunks, b_chunks, h0, mesh, break_chain=True)
    report.control("cross-device chain broken", broken_chain, want, 1)
    # The shard check's control: the same break against one device, at SHARD_TOL.
    report.control("broken chain vs one device", broken_chain, one_device, 1, bound=SHARD_TOL)

    # mamba_ssm_cache_dtype is float32 in a BF16 model. Hold the state in BF16 and the same fold
    # loses three orders of magnitude on CPU, and two on a TPU v5e, where the FP32 side it's
    # measured against already carries the transcendental floor.
    a_bf, b_bf = chunk_pairs(x, b, dt, log_decay, chunk_size, state_dtype=jnp.bfloat16)
    _, rel_bf, _ = errors(chunk_reference(a_bf, b_bf, h0.astype(jnp.bfloat16)), want, 1)
    orders = {1e3: "three", 1e2: "two"}.get(BF16_COST, f"{math.log10(BF16_COST):.0f}")
    report.check(
        rel_bf > BF16_COST * rel_f32,
        f"bf16 state costs at least {orders} orders",
        f"bf16 rel={rel_bf:.3e} vs fp32 rel={rel_f32:.3e}, ratio {rel_bf / rel_f32:.1e}",
    )


def run_outputs(report, mesh):
    """The per-token output, which is what the chunk states are for."""
    chunks, chunk_size, heads, groups, state_size, head_dim = 8, 16, 8, 2, 16, 8
    tokens = chunks * chunk_size
    print(f"  layer output  tokens={tokens} chunk={chunk_size} heads={heads}")

    p = make_problem(33, tokens, heads, groups, state_size, head_dim)
    x, b, c, h0, d = p["x"], p["b"], p["c"], p["h0"], p["d"]
    dt, log_decay = discretize(
        p["dt_raw"], p["dt_bias"], p["a_log"], NEMOTRON_3_SUPER.time_step_limit
    )

    _, want = recurrence_oracle(
        x.astype(jnp.float32),
        b.astype(jnp.float32),
        c.astype(jnp.float32),
        dt,
        log_decay,
        h0,
        d,
    )

    a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, chunk_size)
    states = sharded(a_chunks, b_chunks, h0, mesh)
    h_entering = entering_states(states, h0)
    got = chunk_outputs(x, b, c, dt, log_decay, h_entering, chunk_size, d=d)

    report.measure("y from the sharded states", got, want, head_axis=1)
    report.check(got.shape == (tokens, heads, head_dim), "y is [T, H, P]", str(got.shape))
    report.check(got.dtype == jnp.float32, "y is float32", str(got.dtype))

    for mode, label in (
        ("acausal", "causal mask dropped      "),
        ("no_skip", "D skip dropped           "),
        ("no_inherited", "inherited state dropped  "),
    ):
        report.control(
            label,
            broken_outputs(x, b, c, dt, log_decay, h_entering, chunk_size, d, mode),
            want,
            1,
        )


def run_padding(report, mesh):
    """A sequence that doesn't fill its last chunk, folded and sharded."""
    chunk_size, heads, groups, state_size, head_dim = 16, 4, 2, 8, 4
    devices = mesh.shape[AXIS]
    tokens = 16 * 5 + 7
    print(f"  padding  tokens={tokens} chunk={chunk_size} devices={devices}")

    p = make_problem(11, tokens, heads, groups, state_size, head_dim)
    x, b, h0 = p["x"], p["b"], p["h0"]
    dt, log_decay = discretize(
        p["dt_raw"], p["dt_bias"], p["a_log"], NEMOTRON_3_SUPER.time_step_limit
    )
    want = sequential_reference(x, b, dt, log_decay, h0, stride=chunk_size)

    # One device: pad to whole chunks only.
    xp, bp, dtp, gp, real = pad_to_chunks(x, b, dt, log_decay, chunk_size)
    a_chunks, b_chunks = chunk_pairs(xp, bp, dtp, gp, chunk_size)
    got = chunk_reference(a_chunks, b_chunks, h0)
    report.check(
        got.shape[0] == want.shape[0],
        "one chunk boundary state per kept token",
        f"{got.shape[0]} vs {want.shape[0]}",
    )
    report.check(
        real == tokens,
        "pad_to_chunks returns the token count before padding",
        f"{real} of {xp.shape[0]}",
    )
    report.measure(f"padding, one device ({xp.shape[0]} tokens)", got, want, head_axis=1)

    # Sharded: the chunk count also has to divide over the mesh.
    xs, bs, dts, gs, real_s = pad_to_chunks(x, b, dt, log_decay, chunk_size, num_devices=devices)
    a_s, b_s = chunk_pairs(xs, bs, dts, gs, chunk_size)
    report.check(
        a_s.shape[0] % devices == 0 and real_s == tokens,
        "padded chunk count divides over the mesh",
        f"{a_s.shape[0]} chunks over {devices} devices",
    )
    # A stride of zero has no chunk count to divide, so both ends have to refuse it.
    report.control_raises(
        "no devices", lambda: pad_to_chunks(x, b, dt, log_decay, chunk_size, num_devices=0)
    )
    report.control_raises("chunks of zero tokens", lambda: pad_to_chunks(x, b, dt, log_decay, 0))
    got_s = sharded(a_s, b_s, h0, mesh)
    report.measure(
        f"padding, {devices} shards ({xs.shape[0]} tokens)",
        got_s[: want.shape[0]],
        want,
        head_axis=1,
    )
    tail = got_s[want.shape[0] - 1 :]
    report.measure(
        f"the {tail.shape[0]} padded chunks hold the last real state",
        tail,
        np.broadcast_to(np.asarray(want[-1], np.float64), tail.shape),
        head_axis=1,
    )

    # Zero chunks fold to the identity. `shard_map` never hands one shard zero
    # chunks while another holds some, which is why the pad above rounds up to
    # the mesh, so this covers an empty sequence.
    empty = compose_local(a_s[:0], b_s[:0])
    eye = jnp.broadcast_to(jnp.eye(state_size, dtype=a_s.dtype), empty[0].shape)
    report.check(
        bool(jnp.all(empty[0] == eye)) and bool(jnp.all(empty[1] == 0)),
        "zero chunks fold to the identity",
    )

    def left_pad(arr, pad):
        widths = [(pad, 0)] + [(0, 0)] * (arr.ndim - 1)
        return jnp.pad(arr, widths)

    pad = xp.shape[0] - tokens
    a_bad, b_bad = chunk_pairs(
        left_pad(x, pad), left_pad(b, pad), left_pad(dt, pad), left_pad(log_decay, pad), chunk_size
    )
    report.control("left padded", chunk_reference(a_bad, b_bad, h0), want, 1)


def run_conv(report):
    """The depthwise conv in front of the SSM, split over 8 shards."""
    kernel = NEMOTRON_3_SUPER.conv_kernel
    shards, per, dim = 8, 32, 12
    print(f"  conv  kernel={kernel} shards={shards} tokens={shards * per}")

    rng = np.random.default_rng(5)
    x32 = rng.normal(size=(shards * per, dim)).astype(np.float32)
    w32 = rng.normal(size=(kernel, dim)).astype(np.float32)
    bias32 = rng.normal(size=(dim,)).astype(np.float32)
    x = jnp.asarray(x32)
    w = jnp.asarray(w32)
    bias = jnp.asarray(bias32)

    want = conv_oracle(x32, w32, bias32)
    report.measure("conv, one piece", causal_conv(x, w, bias), want, head_axis=1)

    pieces = [x[i * per : (i + 1) * per] for i in range(shards)]

    def run(with_halo):
        out = []
        for i, piece in enumerate(pieces):
            prefix = None if (i == 0 or not with_halo) else conv_halo(pieces[i - 1], kernel)
            out.append(causal_conv(piece, w, bias, prefix))
        return jnp.concatenate(out, axis=0)

    report.measure("conv, 8 shards ", run(True), want, head_axis=1)
    report.control("halo dropped ", run(False), want, 1)
    report.control("taps reversed", causal_conv(x, w[::-1], bias), want, 1)

    # The dtype contract: a BF16 checkpoint hands over BF16, and the four tap
    # accumulation has to run in float32 anyway.
    xb = x.astype(jnp.bfloat16)
    wb = w.astype(jnp.bfloat16)
    bias_b = bias.astype(jnp.bfloat16)
    got_bf = causal_conv(xb, wb, bias_b)
    want_bf = conv_oracle(
        np.asarray(xb, np.float32), np.asarray(wb, np.float32), np.asarray(bias_b, np.float32)
    )
    report.check(got_bf.dtype == jnp.float32, "bf16 input returns float32", str(got_bf.dtype))
    report.measure("conv, bf16 input", got_bf, want_bf, head_axis=1)

    padded = np.concatenate([np.zeros((kernel - 1, dim), np.float32), np.asarray(xb, np.float32)])
    acc = jnp.zeros((x.shape[0], dim), jnp.bfloat16)
    for i in range(kernel):
        window = jnp.asarray(padded[i : i + x.shape[0]], jnp.bfloat16)
        acc = (acc + window * wb[i]).astype(jnp.bfloat16)
    report.control("bf16 accumulation", (acc + bias_b).astype(jnp.float32), want_bf, 1)

    # A shard shorter than the halo can't fill its neighbor's window.
    report.control_raises("short shard", lambda: conv_halo(x[: kernel - 2], kernel))
    report.control_raises(
        "short prefix", lambda: causal_conv(x, w, bias, jnp.zeros((kernel - 2, dim), jnp.float32))
    )


def run_cache_layout(report, mesh):
    """The state in the layout the reference cache stores.

    The oracle writes the recurrence in `[H, N, P]` and the transpose turns it
    into `[H, P, N]`, so a missing transpose still has a shape that compares
    when `N` equals `P`. The control drops it and has to fail.
    """
    tokens, heads, groups, state_size, head_dim = 24, 4, 2, 8, 8
    print(f"  cache layout  [N,P]->[P,N]  N={state_size} P={head_dim}")

    p = make_problem(7, tokens, heads, groups, state_size, head_dim)
    x, b, c, h0 = p["x"], p["b"], p["c"], p["h0"]
    dt, log_decay = discretize(
        p["dt_raw"], p["dt_bias"], p["a_log"], NEMOTRON_3_SUPER.time_step_limit
    )

    states, _ = recurrence_oracle(
        x.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32), dt, log_decay, h0
    )
    want = np.swapaxes(states[-1], -1, -2)

    a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, 8)
    state = chunk_reference(a_chunks, b_chunks, h0)[-1]

    report.measure("cache layout", as_cache_layout(state), want, head_axis=0)
    report.control("transpose dropped", state, want, 0)

    round_trip = float(np.abs(np.asarray(from_cache_layout(as_cache_layout(state)) - state)).max())
    report.check(round_trip == 0.0, "cache layout round trip", f"max_abs={round_trip:.3e}")


def _dot_precisions(fn, *args):
    """Every `dot_general` precision in the jaxpr, scan and shard_map bodies too."""
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                found.append(eqn.params.get("precision"))
            for value in eqn.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    inner = getattr(item, "jaxpr", item)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def run_precision(report, mesh):
    """Every float32 contraction has to ask for full precision.

    A TPU `dot_general` at the default precision rounds both operands to BF16,
    which hands back the accuracy the float32 cache dtype pays for. CPU runs
    true float32 either way, so this reads the jaxpr rather than the numbers.
    """
    chunk_size, heads, groups, state_size, head_dim = 16, 8, 2, 16, 8
    tokens = 8 * chunk_size
    print("  matmul precision")

    p = make_problem(41, tokens, heads, groups, state_size, head_dim)
    dt, log_decay = discretize(
        p["dt_raw"], p["dt_bias"], p["a_log"], NEMOTRON_3_SUPER.time_step_limit
    )

    def whole_path(x, b, c, dt, log_decay, h0, d):
        a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, chunk_size)
        states = sharded(a_chunks, b_chunks, h0, mesh)
        y = chunk_outputs(
            x, b, c, dt, log_decay, entering_states(states, h0), chunk_size, d=d
        )
        return states, y, causal_conv(x[:, 0, :], jnp.ones((4, head_dim)), jnp.zeros(head_dim))

    found = _dot_precisions(
        whole_path, p["x"], p["b"], p["c"], dt, log_decay, p["h0"], p["d"]
    )
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    open_ended = [f for f in found if f != highest]
    report.check(
        len(found) > 0 and not open_ended,
        "every dot_general asks for HIGHEST",
        f"{len(found)} contractions, {len(open_ended)} left open",
    )

    loose = _dot_precisions(
        lambda u, v: jnp.einsum("ij,jk->ik", u, v), jnp.ones((4, 4)), jnp.ones((4, 4))
    )
    detected = any(f != highest for f in loose)
    print(
        f"      control (precision left open): {loose}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this test detects nothing.")
        report.failures += 1


def run_config(report):
    """The `Mamba2Config` byte counts against the shapes the cache allocates."""
    print("  config")
    cfg = NEMOTRON_3_SUPER

    report.check(
        cfg.conv_halo == cfg.conv_kernel - 1,
        "the halo is conv_kernel - 1 tokens",
        f"{cfg.conv_halo}",
    )
    report.check(
        cfg.conv_dim == cfg.mamba_num_heads * cfg.mamba_head_dim
        + 2 * cfg.n_groups * cfg.ssm_state_size,
        "conv_dim covers the SSM input, B and C",
        f"{cfg.conv_dim}",
    )
    # Super projects B and C per group, 8 groups over 128 heads. The served
    # copy of this config reads the ratio to slice each shard's groups.
    report.check(
        cfg.heads_per_group == 16 and cfg.heads_per_group * cfg.n_groups == cfg.mamba_num_heads,
        "heads that share one B row",
        f"{cfg.heads_per_group}",
    )

    # One float32 SSM state per head, and a conv slot holding the tokens before
    # the current one. The pool allocates `[conv_dim, conv_kernel - 1]`, so a
    # count over `conv_kernel` over-reports by `conv_dim` floats per layer.
    ssm = cfg.mamba_num_heads * cfg.mamba_head_dim * cfg.ssm_state_size * 4
    conv = cfg.conv_dim * (cfg.conv_kernel - 1) * 4
    report.check(
        cfg.state_bytes_per_head * cfg.mamba_num_heads == ssm,
        "the SSM slot is heads x head_dim x state in float32",
        f"{ssm / 1024**2:.0f} MiB per layer",
    )
    report.check(
        cfg.conv_state_bytes == conv,
        "the conv slot holds conv_kernel - 1 tokens",
        f"{conv / 1024:.0f} KiB per layer",
    )
    report.check(
        cfg.state_bytes_per_layer == ssm + conv,
        "the layer total adds the two slots",
        f"{(ssm + conv) / 1024**2:.2f} MiB per layer",
    )

    off_by_one = cfg.conv_dim * cfg.conv_kernel * 4
    report.check(
        cfg.conv_state_bytes != off_by_one,
        "the conv count is not the full kernel width",
        f"{off_by_one / 1024:.0f} KiB would be the wrong answer",
    )
    half = cfg.mamba_num_heads * cfg.mamba_head_dim * cfg.mamba_head_dim * 4
    report.check(
        cfg.state_bytes_per_head * cfg.mamba_num_heads != half,
        "the SSM slot is not square",
        f"a square slot holds {half / ssm:.0%} of it",
    )
    bf16 = ssm // 2
    report.check(
        cfg.state_bytes_per_head * cfg.mamba_num_heads != bf16,
        "the state is float32, not the weight dtype",
        f"BF16 would be {bf16 / 1024**2:.0f} MiB",
    )


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(too_few_devices(devices, 8))
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))
    cfg = NEMOTRON_3_SUPER
    print(f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}")
    print(
        f"config: N={cfg.ssm_state_size} P={cfg.mamba_head_dim} H={cfg.mamba_num_heads}"
        f" G={cfg.n_groups} chunk={cfg.chunk_size} conv_kernel={cfg.conv_kernel}"
        f" dt in {cfg.time_step_limit}"
        f"  state={cfg.state_bytes_per_head * cfg.mamba_num_heads / 1024:.0f} KiB per layer\n"
    )

    report = Report()
    run_discretize(report)
    print()

    cases = [
        # (name, seed, chunks, chunk_size, heads, groups, N, P)
        ("small        ", 1, 16, 16, 2, 1, 8, 4),
        ("model shapes ", 2, 8, cfg.chunk_size, 8, 2, cfg.ssm_state_size, cfg.mamba_head_dim),
        ("many chunks  ", 3, 32, 64, 4, 2, 32, 16),
    ]
    for name, seed, chunks, chunk_size, heads, groups, state_size, head_dim in cases:
        run_case(report, name, seed, chunks, chunk_size, heads, groups, state_size, head_dim, mesh)
        print()

    run_outputs(report, mesh)
    print()
    run_padding(report, mesh)
    print()
    run_conv(report)
    print()
    run_cache_layout(report, mesh)
    print()
    run_config(report)
    print()
    run_precision(report, mesh)

    print()
    if report.failures:
        print(f"FAILED: {report.failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
