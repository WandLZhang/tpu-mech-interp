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

"""Correctness gate for the chunked Kimi Delta Attention fold.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 scan/test_kda.py

`run_gate` checks `gate_log` against the gate the served code computes: a
float64 transcription written in this file, six values worked by hand, and
`Glm5NextTextForgetGate` and `KimiLinearForgetGate` from `transformers` when
that's installed. It covers every `A_log` layout, and its controls are the
clamped softplus gate, a per-channel `A_log` read down the head axis, and the
bounded gate graded against Kimi-Linear. The bound has no default, so a
`gate_log` call or a `KDAConfig` that leaves it out has to fail.

Every fold case compares three things against one float32 token-by-token
reference fed the same gate: the chunk pairs folded on a single device, the
factored replay that never builds `A`, and the same pairs run through the
sharded path in `affine_scan.py`. Then it runs negative controls that each
break one step of the fold. A control must fail; if one passes, this file fails
itself.

Error is measured per head. A global denominator lets a wrong head with a small
state hide behind a right head with a large one.

The gate is tempered in the first three cases so a chunk keeps part of its
incoming state. On a wide gate a single chunk forgets everything, the
cross-device chain carries nothing, and a broken chain looks correct. The
`split gate` case gets both at once: half the key channels sit on the gate
bound and half stay open, so `c_L` reaches -320 while the state still survives
the chunk and `A` still carries it.

`run_wide_gate` covers the whole-chunk forget on its own. It draws `A_log` from
`log(Uniform(1e-9, 16))`, checks the fold and the triangular system stay finite
where the summed log gate reaches several hundred negative, and runs three
controls: the factored `exp(c_t) * exp(-c_s)` form, the softplus gate in place
of the bounded one, and the state accumulated in bfloat16.

`run_positive_gate` feeds a gate that isn't sign-constrained. `run_padding`
checks every chunk boundary of a padded sequence, not only the last, and pads
a sequence shorter than the mesh out to one chunk per device.
`run_precision` reads the jaxpr, which is the only way a CPU run can see a
`dot_general` that would drop to bfloat16 on the MXU.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testing import (  # noqa: E402
    force_host_devices,
    sharded_states,
    skip_reason,
    too_few_devices,
)

# Before jax starts a backend.
force_host_devices(8)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from affine_scan import (  # noqa: E402
    compose_local,
    replay_local,
    sequential_reference as chunk_reference,
)
from kda import (  # noqa: E402
    GATE_LOWER_BOUND,
    GLM_5_3_FLASH,
    KIMI_K3,
    STATE_DTYPE,
    KDAConfig,
    apply_factored,
    chunk_factors,
    chunk_pairs,
    gate_log,
    intra_chunk_system,
    pad_to_chunks,
    pair_decay,
    sequential_reference,
)

AXIS = "ctx"
# The bound against the float32 token-by-token reference follows the platform. The gate runs
# through float32 `exp`, and `test_mamba2.py` prints what `exp` and `log1p` cost on the platform:
# about 1e-7 on CPU. A v5e run put these chunk states at 9.7e-6 to 2.0e-4. The matmuls aren't the
# cause: they all ask for HIGHEST, and the triangular solve measures 9.9e-8.
_BACKEND = jax.default_backend()
REL_TOL = {"cpu": 1e-4}.get(_BACKEND, 5e-4)
CONTROL_MIN = 1e-3
# 8 shards against one device, same platform, same arithmetic reassociated. This bound doesn't
# follow the platform, so a looser REL_TOL reaches accuracy and never the sharding.
SHARD_TOL = 1e-5
# `gate_log` against the float64 gate. CPU float32 lands near 1e-7. Elsewhere it takes REL_TOL,
# which covers the transcendental error above.
GATE_TOL = {"cpu": 1e-6}.get(_BACKEND, REL_TOL)
# Every contraction on the path has to ask for this. See `loose_precision_dots`.
HIGHEST = (lax.Precision.HIGHEST, lax.Precision.HIGHEST)


def _sigmoid(z):
    """The logistic function in float64, split by sign so neither half overflows."""
    z = np.asarray(z, np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def served_gate(a_raw, dt_bias, table, lower_bound):
    """The KDA gate the served code computes, in float64 numpy.

    `table` is `A_log` laid out to broadcast against `[H, K]`. With a bound
    it's `lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`. At `eb061d8`,
    sglang-jax computes that in `kda_gate_chunk_cumsum` for chunked prefill,
    in the Mega KDA kernel, and in `_fused_kda_gate` for decode, and
    `Glm5NextTextForgetGate` computes it in `transformers`. Without a bound
    it's `-exp(A_log) * softplus(a + dt_bias)`. The sigmoid and the softplus
    are written out, so nothing here calls `kda.py` or jax.
    """
    x = np.asarray(a_raw, np.float64) + np.asarray(dt_bias, np.float64)
    scale = np.exp(np.asarray(table, np.float64))
    if lower_bound is None:
        return -scale * (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))
    return lower_bound * _sigmoid(scale * x)


def clamped_softplus_gate(a_raw, dt_bias, table, lower_bound):
    """The softplus gate clamped at the bound, a control for `run_gate`.

    It matches the bounded gate only where both saturate.
    """
    return np.maximum(served_gate(a_raw, dt_bias, table, None), lower_bound)


def temper_shift(x, scale, target, lower_bound=GATE_LOWER_BOUND):
    """The shift that moves the mean bounded gate over `x` to `target`.

    The bounded gate falls as the shift grows, so bisection finds it.
    """
    low, high = -80.0, 80.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if served_gate(x + mid, 0.0, math.log(scale), lower_bound).mean() > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def make_problem(
    seed,
    tokens,
    chunk_size,
    heads,
    key_dim,
    value_dim,
    chunk_decay=None,
    split=False,
):
    """One KDA layer's inputs, in the dtypes a BF16 checkpoint hands over.

    `k`, `v`, `beta` and the gate projection all come out of projections in
    BF16. KDA L2 normalizes `k` per head and `beta` is a sigmoid, so both
    arrive bounded. `A_log` and `dt_bias` are stored parameters and stay
    float32.

    `chunk_decay` of `None` draws `A_log` from `log(Uniform(1e-9, 16))` and
    `dt_bias` from a unit normal. That puts the bounded gate on its bound for
    part of every chunk and the softplus gate near -97 on the strongest heads.
    A float instead tempers the bounded gate so the mean decay over one chunk
    lands near that value, spread across heads: the top of the range keeps
    most of its state across a chunk, the bottom keeps a quarter. It sets
    `A_log` to zero, where `transformers` starts a bounded KDA gate, and
    shifts each head's `dt_bias` until the mean log gate hits its target.

    `split` splits the key channels in two. The first half lands on the gate
    bound, the second half is tempered and carries the state across the chunk.
    That's the case where both `A` and the cross-chunk chain matter at a gate
    large enough to underflow.
    """
    rng = np.random.default_rng(seed)

    k = rng.normal(size=(tokens, heads, key_dim)).astype(np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
    v = rng.normal(size=(tokens, heads, value_dim)).astype(np.float32) * 0.5
    beta = 1.0 / (1.0 + np.exp(-rng.normal(size=(tokens, heads)).astype(np.float32)))

    dt_bias = rng.normal(size=(heads, key_dim)).astype(np.float32)
    a_raw = rng.normal(size=(tokens, heads, key_dim)).astype(np.float32)
    # Round the projection to BF16 here, so the tempering reads the draw the gate reads.
    a_raw = np.asarray(jnp.asarray(a_raw, jnp.bfloat16).astype(jnp.float32))

    open_channels = slice(None)
    if split:
        half = key_dim // 2
        dt_bias[:, :half] += 40.0
        open_channels = slice(half, None)
        chunk_decay = 0.5 if chunk_decay is None else chunk_decay

    if chunk_decay is None:
        a_log = np.log(rng.uniform(1e-9, 16.0, size=heads)).astype(np.float32)
    else:
        a_log = np.zeros(heads, np.float32)
        spread = np.linspace(2.0, 0.5, heads)
        target = np.log(chunk_decay) * spread / chunk_size
        for h in range(heads):
            x = a_raw[:, h, open_channels] + dt_bias[h, open_channels]
            dt_bias[h, open_channels] += temper_shift(x, math.exp(a_log[h]), target[h])

    h0 = rng.normal(size=(heads, key_dim, value_dim)).astype(np.float32) * 0.1

    return (
        jnp.asarray(k, dtype=jnp.bfloat16),
        jnp.asarray(v, dtype=jnp.bfloat16),
        jnp.asarray(beta, dtype=jnp.bfloat16),
        jnp.asarray(a_raw, dtype=jnp.bfloat16),
        jnp.asarray(dt_bias),
        jnp.asarray(a_log),
        jnp.asarray(h0),
    )


def broken_pairs(k, v, beta, g, chunk_size, mode):
    """Chunk pairs with one step of the fold broken.

    `none` breaks nothing and has to reproduce `chunk_pairs`. Without that the
    other modes prove nothing, because a difference against the reference could
    come from this function rather than from the step it removes.

    `no_intra_decay` drops `exp(c_t - c_s)` from the triangular system, so a
    token reads earlier writes at full weight. `no_correction` sets `w` to
    zero, so the chunk writes its values without subtracting what the incoming
    state already holds. `carry_off_by_one` counts a token's own gate twice on
    the way out of the chunk, the off-by-one a cumulative sum invites.
    `naive_pair` builds `exp(c_t - c_s)` as `exp(c_t) * exp(-c_s)`.
    `clamped_pair` clamps `c_t - c_s` at zero instead of masking the square by
    position, which is the older form of `pair_decay`.

    Everything here runs in float32 so `naive_pair` overflows the way the real
    path would.
    """
    tokens, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    chunks = tokens // chunk_size
    length = chunk_size

    kf = np.asarray(k.astype(jnp.float32)).reshape(chunks, length, heads, key_dim)
    vf = np.asarray(v.astype(jnp.float32)).reshape(chunks, length, heads, value_dim)
    bf = np.asarray(beta.astype(jnp.float32)).reshape(chunks, length, heads)
    gf = np.asarray(g.astype(jnp.float32)).reshape(chunks, length, heads, key_dim)

    c = np.cumsum(gf, axis=1)
    log_last = c[:, -1]

    k_hat = kf * np.exp(c)
    carry = log_last[:, None] - c
    if mode == "carry_off_by_one":
        carry = carry + gf
    k_g = kf * np.exp(carry)

    strict = np.tril(np.ones((length, length), dtype=np.float32), -1)
    diff = c[:, :, None] - c[:, None, :]
    if mode == "no_intra_decay":
        decay = np.ones((chunks, length, length, heads, key_dim), dtype=np.float32)
    elif mode == "naive_pair":
        decay = np.exp(c)[:, :, None] * np.exp(-c)[:, None, :]
    elif mode == "clamped_pair":
        decay = np.exp(np.minimum(diff, 0.0))
    else:
        mask = strict.astype(bool)[None, :, :, None, None]
        decay = np.where(mask, np.exp(np.where(mask, diff, 0.0)), 0.0)

    pair = np.einsum("ctshk,cthk,cshk->chts", decay, kf, kf)
    beta_h = np.swapaxes(bf, 1, 2)[..., None]
    lower = np.eye(length, dtype=np.float32) + beta_h * pair * strict

    rhs = np.concatenate(
        [beta_h * np.swapaxes(vf, 1, 2), beta_h * np.swapaxes(k_hat, 1, 2)], axis=-1
    )
    solved = np.linalg.solve(lower, rhs).astype(np.float32)
    u = solved[..., :value_dim]
    w = solved[..., value_dim:]
    if mode == "no_correction":
        w = np.zeros_like(w)

    k_g = np.swapaxes(k_g, 1, 2)
    eye = np.eye(key_dim, dtype=np.float32)
    a_chunks = np.exp(log_last)[..., None] * eye - np.einsum("chlk,chlj->chkj", k_g, w)
    b_chunks = np.einsum("chlk,chlv->chkv", k_g, u)
    return jnp.asarray(a_chunks), jnp.asarray(b_chunks)


def sharded(a, b, h0, mesh, break_chain=False):
    """`_testing.sharded_states` on this file's mesh axis.

    `break_chain` is the negative control: every shard starts from `h0`, so
    any shard after the first inherits nothing.
    """
    return sharded_states(a, b, h0, mesh, AXIS, break_chain=break_chain)


def factored_replay(k, v, beta, g, h0, chunk_size):
    """Replay every chunk through `apply_factored`, never building `A`."""
    log_last, k_g, w, u = chunk_factors(k, v, beta, g, chunk_size)
    h = h0
    out = []
    for c in range(log_last.shape[0]):
        h = apply_factored(log_last[c], k_g[c], w[c], u[c], h)
        out.append(h)
    return jnp.stack(out)


def errors(got, want):
    """Max absolute error, worst per-head relative error, and cosine.

    The relative figure divides each head by its own largest entry. A shared
    denominator hides a head whose state is small against the largest head in
    the layer, and on the wide gate the worst head reads 1.5x the global
    figure.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if not np.isfinite(got).all():
        return np.inf, np.inf, 0.0
    max_abs = np.abs(got - want).max()
    if want.ndim == 4:
        # [states, heads, K, V]
        axes = (0, 2, 3)
        per_head = np.abs(got - want).max(axis=axes)
        denom = np.maximum(np.abs(want).max(axis=axes), 1e-12)
        rel = float((per_head / denom).max())
    else:
        rel = max_abs / max(np.abs(want).max(), 1e-12)
    gf, wf = got.ravel(), want.ravel()
    cos = float(gf @ wf / max(np.linalg.norm(gf) * np.linalg.norm(wf), 1e-12))
    return max_abs, rel, cos


def report(label, got, want, tol=None):
    """Print one comparison. Returns 1 if it failed."""
    max_abs, rel, cos = errors(got, want)
    ok = rel < (REL_TOL if tol is None else tol)
    print(
        f"      [{'PASS' if ok else 'FAIL'}] {label}"
        f"   max_abs={max_abs:.3e}  rel={rel:.3e}  cos={cos:.8f}"
    )
    return 0 if ok else 1


def report_control(label, got, want, bound=None):
    """Print one control. Returns 1 if the control didn't fail."""
    _, rel, cos = errors(got, want)
    detected = rel > (CONTROL_MIN if bound is None else bound)
    print(
        f"      control ({label}): rel={rel:.3e} cos={cos:.6f}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this test detects nothing.")
        return 1
    return 0


def glm5_forget_gate(a_raw, dt_bias, a_log_heads, lower_bound):
    """`Glm5NextTextForgetGate` from `transformers` on these inputs.

    The module's two projections are swapped for identities, so `a_raw` goes
    straight into the forward the GLM-5.3-Flash model runs. `A_log` is one
    value per head there.

    Returns:
      The gate and None, or None and the reason it can't run: `transformers`
      or `torch` missing, or a `transformers` without the GLM-5.3-Flash code.
    """
    try:
        import torch
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextForgetGate,
        )
    except ImportError as exc:
        return None, skip_reason("glm5_next", exc, needs_torch=True)
    tokens, heads, channels = a_raw.shape
    config = Glm5NextTextConfig(
        hidden_size=heads * channels,
        linear_head_dim=channels,
        linear_num_heads=heads,
        linear_lower_bound=lower_bound,
    )
    module = Glm5NextTextForgetGate(config)
    module.f_a_proj = torch.nn.Identity()
    module.f_b_proj = torch.nn.Identity()
    with torch.no_grad():
        module.dt_bias.copy_(torch.from_numpy(np.ascontiguousarray(dt_bias).reshape(-1)))
        module.A_log.copy_(torch.from_numpy(np.ascontiguousarray(a_log_heads)))
        hidden = torch.from_numpy(np.ascontiguousarray(a_raw).reshape(1, tokens, -1))
        out = module(hidden)
    return out.numpy().reshape(tokens, heads, channels), None


def kimi_linear_forget_gate(a_raw, dt_bias, a_log_heads, linear_attn_config):
    """`KimiLinearForgetGate` from `transformers` on these inputs.

    That module runs the softplus gate and reads no bound at all. `A_log` is
    `[1, 1, H, 1]` there. The projections are swapped for identities, as in
    `glm5_forget_gate`.

    Returns:
      The gate and None, or None and the reason it can't run: `transformers`
      or `torch` missing, or a `transformers` without the Kimi-Linear code.
    """
    try:
        import torch
        from transformers.models.kimi_linear.configuration_kimi_linear import (
            KimiLinearConfig,
        )
        from transformers.models.kimi_linear.modeling_kimi_linear import (
            KimiLinearForgetGate,
        )
    except ImportError as exc:
        return None, skip_reason("kimi_linear", exc, needs_torch=True)
    tokens, heads, channels = a_raw.shape
    config = KimiLinearConfig(
        hidden_size=heads * channels, linear_attn_config=dict(linear_attn_config)
    )
    module = KimiLinearForgetGate(config)
    module.f_a_proj = torch.nn.Identity()
    module.f_b_proj = torch.nn.Identity()
    with torch.no_grad():
        module.dt_bias.copy_(torch.from_numpy(np.ascontiguousarray(dt_bias).reshape(-1)))
        module.A_log.copy_(
            torch.from_numpy(np.ascontiguousarray(a_log_heads).reshape(1, 1, heads, 1))
        )
        hidden = torch.from_numpy(np.ascontiguousarray(a_raw).reshape(1, tokens, -1))
        out = module(hidden)
    return out.numpy().reshape(tokens, heads, channels), None


def run_gate():
    """`gate_log` against the gate the served code computes.

    Three references, none of which calls `gate_log`: values worked by hand, a
    float64 transcription of the served formula, and the GLM-5.3-Flash and
    Kimi-Linear forget gates in `transformers` when it's installed. The draws
    cover every `A_log` layout, with the bound and without it.
    """
    print("  gate")
    failures = 0
    bound = GATE_LOWER_BOUND

    # The bound has no default. A default of -5.0 hands a caller who leaves it
    # out the bounded gate, -2.5 at zero, where Kimi-Linear sets no bound and
    # serves the softplus gate, -0.69. So leaving it out has to fail.
    try:
        left_out = gate_log(
            jnp.zeros((1, 1, 1), jnp.float32),
            jnp.zeros((1, 1), jnp.float32),
            jnp.zeros((1, 1), jnp.float32),
        )
    except TypeError:
        print("      [PASS] gate_log refuses a call that leaves the bound out")
    else:
        print(
            "      [FAIL] gate_log ran with the bound left out and returned"
            f" {float(left_out[0, 0, 0]):.3f}"
        )
        failures += 1
    try:
        unbound = KDAConfig(
            num_heads=KIMI_K3.num_heads,
            head_dim=KIMI_K3.head_dim,
            short_conv_kernel_size=KIMI_K3.short_conv_kernel_size,
            num_hidden_layers=KIMI_K3.num_hidden_layers,
            num_kda_layers=KIMI_K3.num_kda_layers,
        )
    except TypeError:
        print("      [PASS] KDAConfig refuses a config that leaves gate_lower_bound out")
    else:
        print(f"      [FAIL] KDAConfig filled in gate_lower_bound={unbound.gate_lower_bound}")
        failures += 1

    # sigmoid(0) = 1/2, sigmoid(ln 3) = 3/4 and softplus(0) = ln 2, so each gate
    # below is plain arithmetic.
    ln2, ln3 = math.log(2.0), math.log(3.0)
    by_hand = (
        # (a + dt_bias, A_log, bound, gate)
        (0.0, 0.0, bound, -2.5),
        (ln3, 0.0, bound, -3.75),
        (-ln3, 0.0, bound, -1.25),
        (ln3 / 2, ln2, bound, -3.75),
        (0.0, 0.0, None, -ln2),
        (0.0, ln2, None, -2 * ln2),
    )
    hand_errors = []
    for x, a_log, lower, want in by_hand:
        got = gate_log(
            jnp.full((1, 1, 1), x, jnp.float32),
            jnp.zeros((1, 1), jnp.float32),
            jnp.full((1, 1), a_log, jnp.float32),
            lower,
        )
        hand_errors.append(abs(float(got[0, 0, 0]) - want) / abs(want))
    # `np.max` carries a NaN through, where a running `max(worst, rel)` keeps
    # the finite side and a NaN gate would pass.
    worst = float(np.max(hand_errors))
    ok = bool(worst < GATE_TOL)
    print(f"      [{'PASS' if ok else 'FAIL'}] six gates worked by hand   rel={worst:.3e}")
    failures += 0 if ok else 1

    rng = np.random.default_rng(17)
    tokens, heads, channels = 64, 4, 8
    a_raw = (rng.normal(size=(tokens, heads, channels)) * 2.0).astype(np.float32)
    dt_bias = rng.normal(size=(heads, channels)).astype(np.float32)
    per_head = np.log(rng.uniform(0.1, 16.0, size=heads)).astype(np.float32)
    per_channel = np.log(rng.uniform(0.1, 16.0, size=channels)).astype(np.float32)
    full = np.log(rng.uniform(0.1, 16.0, size=(heads, channels))).astype(np.float32)
    layouts = (
        ("[H]   ", per_head, per_head[:, None]),
        ("[K]   ", per_channel, per_channel[None, :]),
        ("[H, K]", full, full),
        ("[1, 1, H, 1]", per_head.reshape(1, 1, heads, 1), per_head[:, None]),
    )
    for lower in (bound, None):
        name = "bounded " if lower is not None else "softplus"
        for label, a_log, table in layouts:
            got = gate_log(jnp.asarray(a_raw), jnp.asarray(dt_bias), jnp.asarray(a_log), lower)
            failures += report(
                f"{name} gate, A_log {label}",
                got,
                served_gate(a_raw, dt_bias, table, lower),
                tol=GATE_TOL,
            )

    # NEGATIVE CONTROL: the softplus gate clamped at the bound, graded the
    # same way. It agrees with the bounded gate only where both saturate.
    want = served_gate(a_raw, dt_bias, per_head[:, None], bound)
    failures += report_control(
        "softplus gate clamped at the bound",
        clamped_softplus_gate(a_raw, dt_bias, per_head[:, None], bound),
        want,
        bound=GATE_TOL,
    )

    # With as many heads as channels a flat A_log could run along either axis.
    square = 8
    a_sq = (rng.normal(size=(tokens, square, square)) * 2.0).astype(np.float32)
    dt_sq = rng.normal(size=(square, square)).astype(np.float32)
    vector = np.log(rng.uniform(0.1, 16.0, size=square)).astype(np.float32)
    try:
        gate_log(jnp.asarray(a_sq), jnp.asarray(dt_sq), jnp.asarray(vector), bound)
    except ValueError:
        print("      [PASS] a flat A_log is refused when heads equal channels")
    else:
        print("      [FAIL] a flat A_log was read as one layout when heads equal channels")
        failures += 1
    for label, shaped, table in (
        ("[1, K]", vector[None, :], vector[None, :]),
        ("[H, 1]", vector[:, None], vector[:, None]),
    ):
        got = gate_log(jnp.asarray(a_sq), jnp.asarray(dt_sq), jnp.asarray(shaped), bound)
        failures += report(
            f"bounded gate, A_log {label} at H == K",
            got,
            served_gate(a_sq, dt_sq, table, bound),
            tol=GATE_TOL,
        )
    # NEGATIVE CONTROL: the per-channel vector read down the head axis, which
    # is how a layout test that tries the head count first reads it.
    failures += report_control(
        "per-channel A_log read per head",
        served_gate(a_sq, dt_sq, vector[:, None], bound),
        served_gate(a_sq, dt_sq, vector[None, :], bound),
        bound=GATE_TOL,
    )

    for lower in (bound, None):
        name = "bounded " if lower is not None else "softplus"
        modeling, missing = glm5_forget_gate(a_raw, dt_bias, per_head, lower)
        if missing:
            print(f"      skipped: {name} gate against Glm5NextTextForgetGate, because {missing}")
            continue
        got = gate_log(jnp.asarray(a_raw), jnp.asarray(dt_bias), jnp.asarray(per_head), lower)
        failures += report(
            f"{name} gate against Glm5NextTextForgetGate", got, modeling, tol=GATE_TOL
        )

    # Kimi-Linear's `linear_attn_config` carries these keys and no bound. Read
    # it the way sglang-jax does, `.get("gate_lower_bound")`, and the gate has
    # to match the module that serves that config.
    linear_attn_config = {"head_dim": channels, "num_heads": heads, "short_conv_kernel_size": 4}
    modeling, missing = kimi_linear_forget_gate(a_raw, dt_bias, per_head, linear_attn_config)
    if missing:
        print(f"      skipped: softplus gate against KimiLinearForgetGate, because {missing}")
    else:
        got = gate_log(
            jnp.asarray(a_raw),
            jnp.asarray(dt_bias),
            jnp.asarray(per_head.reshape(1, 1, heads, 1)),
            linear_attn_config.get("gate_lower_bound"),
        )
        failures += report(
            "softplus gate against KimiLinearForgetGate", got, modeling, tol=GATE_TOL
        )
        # NEGATIVE CONTROL: the bounded gate, which a default of -5.0 would
        # have handed a caller who left the Kimi-Linear bound out.
        failures += report_control(
            "bounded gate against KimiLinearForgetGate",
            gate_log(
                jnp.asarray(a_raw),
                jnp.asarray(dt_bias),
                jnp.asarray(per_head.reshape(1, 1, heads, 1)),
                bound,
            ),
            modeling,
            bound=GATE_TOL,
        )
    return failures


def run_case(
    name, seed, chunks, chunk_size, heads, key_dim, value_dim, mesh, split=False
):
    """Fold, replay, shard, and break. Returns the number of failed checks."""
    tokens = chunks * chunk_size
    k, v, beta, a_raw, dt_bias, a_log, h0 = make_problem(
        seed,
        tokens,
        chunk_size,
        heads,
        key_dim,
        value_dim,
        chunk_decay=None if split else 0.5,
        split=split,
    )
    g = gate_log(a_raw, dt_bias, a_log, GATE_LOWER_BOUND)

    want = sequential_reference(k, v, beta, g, h0, stride=chunk_size)
    c_last = np.asarray(g).reshape(chunks, chunk_size, heads, key_dim).sum(1)
    keep = float(np.exp(c_last).mean())

    print(
        f"  {name}  chunks={chunks:<3} chunk={chunk_size:<3} heads={heads}"
        f" K={key_dim} V={value_dim}  mean chunk decay={keep:.3f}"
        f"  gate min={float(g.min()):.2f}  summed min={float(c_last.min()):.1f}"
    )

    failures = 0
    a_chunks, b_chunks = chunk_pairs(k, v, beta, g, chunk_size)

    # A head that forgets everything has A = 0, and then a broken chain looks
    # correct. Every case has to carry something across a chunk.
    carried = float(np.abs(np.asarray(a_chunks)).max())
    if carried < 1e-3:
        print(f"      FAIL: max|A| is {carried:.3e}, so no chunk carries its state.")
        failures += 1

    one_device = chunk_reference(a_chunks, b_chunks, h0)
    failures += report("chunk pairs, one device", one_device, want)
    failures += report(
        "factored replay        ",
        factored_replay(k, v, beta, g, h0, chunk_size),
        want,
    )
    got_shards = sharded(a_chunks, b_chunks, h0, mesh)
    failures += report("chunk pairs, 8 shards  ", got_shards, want)
    failures += report("8 shards match one device", got_shards, one_device, tol=SHARD_TOL)

    # The controls only attribute a failure if the unbroken build agrees with
    # the implementation first.
    a_ref, b_ref = broken_pairs(k, v, beta, g, chunk_size, "none")
    failures += report(
        "control build, nothing broken",
        chunk_reference(a_ref, b_ref, h0),
        want,
    )

    controls = [
        ("intra-chunk decay dropped", broken_pairs(k, v, beta, g, chunk_size, "no_intra_decay")),
        ("delta correction dropped ", broken_pairs(k, v, beta, g, chunk_size, "no_correction")),
        ("carry-out off by one     ", broken_pairs(k, v, beta, g, chunk_size, "carry_off_by_one")),
        ("cross-device chain broken", (a_chunks, b_chunks)),
    ]
    for i, (label, (a_bad, b_bad)) in enumerate(controls):
        last = i == len(controls) - 1
        got = sharded(a_bad, b_bad, h0, mesh, break_chain=last)
        failures += report_control(label, got, want)
        if last:
            # The shard check's control: the same break against one device, at SHARD_TOL.
            failures += report_control(
                "broken chain vs one device", got, one_device, bound=SHARD_TOL
            )

    return failures


def run_wide_gate(seed, chunks, chunk_size, heads, key_dim, value_dim):
    """A gate wide enough to forget a whole chunk.

    `A_log` comes from `log(Uniform(1e-9, 16))` and `dt_bias` from a unit
    normal. The softplus gate then reaches about -97 on one token. The bounded
    gate stays above its bound, and a channel that sits on the bound for a
    whole chunk still sums to -320. Heads at that end forget their whole
    chunk, which is the right answer and not an overflow.
    """
    tokens = chunks * chunk_size
    k, v, beta, a_raw, dt_bias, a_log, h0 = make_problem(
        seed, tokens, chunk_size, heads, key_dim, value_dim
    )
    g_soft = gate_log(a_raw, dt_bias, a_log, None)
    g = gate_log(a_raw, dt_bias, a_log, GATE_LOWER_BOUND)
    c_last = np.asarray(g).reshape(chunks, chunk_size, heads, key_dim).sum(1)
    g_np = np.asarray(g)
    on_bound = float((g_np <= GATE_LOWER_BOUND + 1e-3).mean())
    moved = float((np.abs(g_np - np.asarray(g_soft)) > 1e-3).mean())

    print(
        f"  wide gate  chunks={chunks} chunk={chunk_size} heads={heads}"
        f" K={key_dim} V={value_dim}"
    )
    print(
        f"      softplus gate min={float(g_soft.min()):.2f} per token. Bounded gate:"
        f" {on_bound * 100:.0f}% of entries on the bound, {moved * 100:.1f}% more than"
        f" 1e-3 off the softplus gate, summed min={float(c_last.min()):.1f} per chunk"
    )

    failures = 0
    if float(g_soft.min()) > -70.0:
        print("      FAIL: the softplus gate never got large, so this case tests nothing.")
        failures += 1
    if not (g_np.min() >= GATE_LOWER_BOUND and g_np.max() <= 0.0 and on_bound > 0.0):
        print(
            f"      FAIL: the bounded gate spans {g_np.min()} to {g_np.max()},"
            f" not the bound to zero."
        )
        failures += 1

    # The triangular system is where a positive exponent would land. Check it
    # directly: solve_triangular reads only the lower half, so a NaN in the
    # upper half never reaches the output on CPU and would ship to the MXU.
    k_c = k[:chunk_size].astype(STATE_DTYPE)
    for label, gate in (("bounded", g), ("softplus", g_soft)):
        c = jnp.cumsum(gate[:chunk_size].astype(STATE_DTYPE), axis=0)
        decay = pair_decay(c)
        system = intra_chunk_system(k_c, beta[:chunk_size].astype(STATE_DTYPE), c)
        upper = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool))
        clean = bool(
            jnp.isfinite(decay).all()
            and jnp.isfinite(system).all()
            and (decay[upper] == 0).all()
        )
        # `solve_triangular` takes no precision argument, so the solve is the
        # one op on this path that runs at whatever XLA picks. A well
        # conditioned system is what makes that acceptable.
        cond = (
            float(np.linalg.cond(np.asarray(system, dtype=np.float64)).max())
            if clean
            else np.inf
        )
        well_conditioned = cond < 10.0
        ok = clean and well_conditioned
        print(
            f"      [{'PASS' if ok else 'FAIL'}] I + M finite, {label} gate"
            f"   max cond={cond:.2f}"
        )
        failures += 0 if ok else 1

    a_chunks, b_chunks = chunk_pairs(k, v, beta, g, chunk_size)
    finite = bool(jnp.isfinite(a_chunks).all() and jnp.isfinite(b_chunks).all())
    print(f"      [{'PASS' if finite else 'FAIL'}] (A, B) finite at the wide gate")
    failures += 0 if finite else 1

    want = sequential_reference(k, v, beta, g, h0, stride=chunk_size)
    failures += report(
        "chunk pairs, one device", chunk_reference(a_chunks, b_chunks, h0), want
    )
    failures += report(
        "factored replay        ",
        factored_replay(k, v, beta, g, h0, chunk_size),
        want,
    )

    # Control 1: the factorization, not a broken step. exp(c_t - c_s) built as
    # exp(c_t) * exp(-c_s) is the same number on paper.
    with np.errstate(over="ignore", invalid="ignore"):
        a_bad, b_bad = broken_pairs(k, v, beta, g, chunk_size, "naive_pair")
    bad_finite = bool(jnp.isfinite(a_bad).all() and jnp.isfinite(b_bad).all())
    print(
        f"      control (exp(c_t) * exp(-c_s)): finite={bad_finite}"
        f"  -> {'NOT DETECTED' if bad_finite else 'detected'}"
    )
    if bad_finite:
        print("      FAIL: the control stayed finite, so this test detects nothing.")
        failures += 1

    # Control 2: the softplus gate in place of the bounded one. The config sets
    # a bound, so a fold fed the softplus gate computes a different function.
    a_soft, b_soft = chunk_pairs(k, v, beta, g_soft, chunk_size)
    failures += report_control(
        "softplus gate, no bound  ",
        chunk_reference(a_soft, b_soft, h0),
        want,
    )

    # Control 3: the state accumulated in bfloat16. c_L reaches -320, where one
    # bfloat16 step is 2, so the answer stays finite and is wrong.
    a_bf, b_bf = chunk_pairs(k, v, beta, g, chunk_size, state_dtype=jnp.bfloat16)
    failures += report_control(
        "bfloat16 accumulation    ",
        chunk_reference(a_bf.astype(STATE_DTYPE), b_bf.astype(STATE_DTYPE), h0),
        want,
    )

    return failures


def run_positive_gate():
    """A gate that isn't sign-constrained.

    `chunk_pairs` takes `g` as a plain array, so a port whose gate can rise
    above zero reaches it. The pairwise decay masks the square by position, so
    the fold stays right. Clamping `c_t - c_s` at zero instead is a no-op while
    `c` falls and a silent wrong answer once it rises, which is the control.
    """
    chunk_size, chunks, heads, key_dim, value_dim = 16, 8, 4, 8, 8
    tokens = chunks * chunk_size
    k, v, beta, a_raw, dt_bias, a_log, h0 = make_problem(
        5, tokens, chunk_size, heads, key_dim, value_dim, chunk_decay=0.5
    )
    # Push part of every head's gate above zero, so `c` rises and falls.
    g = gate_log(a_raw, dt_bias, a_log, GATE_LOWER_BOUND) + 0.05
    positive = float((np.asarray(g) > 0).mean())

    want = sequential_reference(k, v, beta, g, h0, stride=chunk_size)
    a_chunks, b_chunks = chunk_pairs(k, v, beta, g, chunk_size)

    print(f"  positive gate  {positive * 100:.0f}% of entries above zero")
    failures = report(
        "chunk pairs, one device", chunk_reference(a_chunks, b_chunks, h0), want
    )
    failures += report(
        "factored replay        ",
        factored_replay(k, v, beta, g, h0, chunk_size),
        want,
    )
    a_bad, b_bad = broken_pairs(k, v, beta, g, chunk_size, "clamped_pair")
    failures += report_control(
        "pair decay clamped at 0  ", chunk_reference(a_bad, b_bad, h0), want
    )
    return failures


def run_padding(mesh):
    """A sequence that doesn't fill its last chunk, and one shorter than the mesh.

    Checks every chunk boundary, not only the last one. The final state is
    right under left padding too, because a padded token writes nothing. Only
    the interior boundaries say the padding went on the correct end.
    """
    chunk_size, heads, key_dim, value_dim = 16, 4, 8, 8
    chunks = 8
    tokens = chunk_size * (chunks - 1) + 9
    k, v, beta, a_raw, dt_bias, a_log, h0 = make_problem(
        11, tokens, chunk_size, heads, key_dim, value_dim, chunk_decay=0.5
    )
    g = gate_log(a_raw, dt_bias, a_log, GATE_LOWER_BOUND)

    per_token = sequential_reference(k, v, beta, g, h0)
    # The boundary of every full chunk, then the last real token for the chunk
    # the padding fills out.
    edges = [chunk_size * (i + 1) - 1 for i in range(chunks - 1)] + [tokens - 1]
    want = jnp.stack([per_token[t] for t in edges])

    kp, vp, bp, gp, real = pad_to_chunks(k, v, beta, g, chunk_size)
    a_chunks, b_chunks = chunk_pairs(kp, vp, bp, gp, chunk_size)

    print(f"  padding  tokens={real} -> {kp.shape[0]}  boundaries={len(edges)}")
    failures = 0
    if real != tokens:
        print(f"      FAIL: pad_to_chunks reported {real} real tokens, not {tokens}.")
        failures += 1
    if a_chunks.shape[0] != chunks:
        print(f"      FAIL: padded to {a_chunks.shape[0]} chunks, not {chunks}.")
        failures += 1

    failures += report(
        "chunk pairs, one device", chunk_reference(a_chunks, b_chunks, h0), want
    )
    failures += report(
        "factored replay        ",
        factored_replay(kp, vp, bp, gp, h0, chunk_size),
        want,
    )
    failures += report(
        "chunk pairs, 8 shards  ", sharded(a_chunks, b_chunks, h0, mesh), want
    )

    # NEGATIVE CONTROL: the same padding on the left end. The last state still
    # comes out right, because a padded token writes nothing, and the interior
    # boundaries don't.
    pad = kp.shape[0] - tokens

    def left(arr):
        return jnp.pad(arr, [(pad, 0)] + [(0, 0)] * (arr.ndim - 1))

    a_left, b_left = chunk_pairs(left(k), left(v), left(beta), left(g), chunk_size)
    failures += report_control(
        "left padded              ", chunk_reference(a_left, b_left, h0), want
    )

    # A sequence shorter than the mesh. `shard_map` gives every device the same
    # number of chunks, so `num_devices` pads out to one chunk per device, and a
    # chunk that holds only padding folds to the identity pair.
    devices = mesh.shape[AXIS]
    short = 2 * chunk_size + 5
    held = -(-short // chunk_size)
    ks, vs, bs, gs, real_short = pad_to_chunks(
        k[:short], v[:short], beta[:short], g[:short], chunk_size, num_devices=devices
    )
    a_s, b_s = chunk_pairs(ks, vs, bs, gs, chunk_size)
    eye = np.broadcast_to(np.eye(key_dim, dtype=np.float32), a_s[held:].shape)
    identity = bool(
        np.array_equal(np.asarray(a_s[held:]), eye) and not np.asarray(b_s[held:]).any()
    )
    ok = a_s.shape[0] == devices and real_short == short and identity
    print(
        f"      [{'PASS' if ok else 'FAIL'}] {short} tokens pad to {a_s.shape[0]} chunks"
        f" over {devices} devices, and the {a_s.shape[0] - held} past the sequence are"
        f" identity pairs"
    )
    failures += 0 if ok else 1
    short_edges = [chunk_size * (i + 1) - 1 for i in range(held - 1)] + [short - 1]
    want_short = jnp.stack([per_token[t] for t in short_edges])
    got_short = sharded(a_s, b_s, h0, mesh)
    failures += report(
        f"{short} tokens, {devices} shards", got_short[:held], want_short
    )
    failures += report(
        "padded chunks hold the last real state",
        got_short[held:],
        jnp.broadcast_to(want_short[-1], got_short[held:].shape),
    )

    # A whole number of chunks comes back untouched, with its own token count.
    same = pad_to_chunks(k[:chunk_size], v[:chunk_size], beta[:chunk_size],
                         g[:chunk_size], chunk_size)
    if same[0].shape[0] != chunk_size or same[4] != chunk_size:
        print("      FAIL: pad_to_chunks padded a sequence that already fit.")
        failures += 1

    # The stride has to divide the token count, or the states line up against
    # chunks that cover different tokens.
    try:
        sequential_reference(k, v, beta, g, h0, stride=chunk_size)
        print("      FAIL: a stride that doesn't divide the tokens was accepted.")
        failures += 1
    except ValueError:
        print("      [PASS] a stride that doesn't divide the tokens is rejected")

    return failures


def run_config():
    """The numbers the sizing prints, checked against the shapes they come from.

    A KDA layer holds two slots per sequence: the float32 recurrent state, and
    the BF16 short-conv windows on `q`, `k` and `v` that sglang-jax's
    `RecurrentStatePool` keeps beside it. `state_bytes` adds both. The figures
    are literals, written out from the published shapes.
    """
    print("  config")
    failures = 0
    expected = {
        # halo, recurrent per head, recurrent per layer, conv per layer,
        # both per layer, both over every KDA layer
        "Kimi-K3": (KIMI_K3, 3, 65536, 6291456, 221184, 6512640, 449372160),
        "GLM-5.3-Flash": (GLM_5_3_FLASH, 3, 65536, 4194304, 147456, 4341760, 147619840),
    }
    for label, (cfg, *want) in expected.items():
        got = [
            cfg.conv_halo,
            cfg.state_bytes_per_head,
            cfg.recurrent_state_bytes_per_layer,
            cfg.conv_state_bytes,
            cfg.state_bytes_per_layer,
            cfg.state_bytes,
        ]
        recurrent = cfg.recurrent_state_bytes_per_layer * cfg.num_kda_layers
        conv = cfg.conv_state_bytes * cfg.num_kda_layers
        ok = got == want
        print(
            f"      [{'PASS' if ok else 'FAIL'}] {label:14s} halo={got[0]}"
            f" state={got[5] / 1024**2:.1f} MiB per sequence, recurrent"
            f" {recurrent / 1024**2:.1f} and conv {conv / 1024**2:.1f}"
        )
        if not ok:
            print(f"      FAIL: got {got}, want {want}")
            failures += 1

    # The three ways a budget goes wrong, written out for Kimi K3. The first is
    # the recurrent-only figure this file printed before the conv windows.
    cfg = KIMI_K3
    channels = 3 * cfg.num_heads * cfg.head_dim
    wrong = {
        "the total counts the conv windows": 434110464,
        "the conv windows hold kernel - 1 tokens, not the kernel": channels * 4 * 2,
        "the conv windows are BF16, not float32": channels * 3 * 4,
    }
    got = {
        "the total counts the conv windows": cfg.state_bytes,
        "the conv windows hold kernel - 1 tokens, not the kernel": cfg.conv_state_bytes,
        "the conv windows are BF16, not float32": cfg.conv_state_bytes,
    }
    for label, bad in wrong.items():
        ok = got[label] != bad
        print(f"      [{'PASS' if ok else 'FAIL'}] {label}   {bad} would be the wrong answer")
        failures += 0 if ok else 1
    return failures


def run_cost_model():
    """The two crossovers the module docstring quotes.

    Replay: `apply_factored` costs `2 L K V`, `A @ h` costs `K^2 V`, so the
    factors win while `2 L < K`. Memory: the factors hold `K + 2 L K + L V`
    floats, the pair holds `K^2 + K V`.
    """
    print("  cost model")
    failures = 0

    def flops(length, key_dim, value_dim):
        return 2 * length * key_dim * value_dim, key_dim * key_dim * value_dim

    def floats(length, key_dim, value_dim):
        return (
            key_dim + 2 * length * key_dim + length * value_dim,
            key_dim * key_dim + key_dim * value_dim,
        )

    for label, cfg in (("Kimi-K3", KIMI_K3), ("GLM-5.3-Flash", GLM_5_3_FLASH)):
        k_dim = cfg.head_dim
        length = cfg.chunk_size
        fac_flops, dense_flops = flops(length, k_dim, k_dim)
        fac_floats, dense_floats = floats(length, k_dim, k_dim)
        # Half the head dimension is where the replay cost crosses, and both
        # shipped configs sit on it.
        on_the_line = fac_flops == dense_flops and length == k_dim // 2
        smaller = fac_floats < dense_floats
        ok = on_the_line and smaller
        print(
            f"      [{'PASS' if ok else 'FAIL'}] {label:14s} L={length} K={k_dim}"
            f"  replay {fac_flops / dense_flops:.2f}x  floats"
            f" {fac_floats} vs {dense_floats}"
        )
        failures += 0 if ok else 1

    # The replay crossover is L < K/2, not L < K.
    k_dim = 128
    below = flops(k_dim // 2 - 1, k_dim, k_dim)
    above = flops(k_dim - 1, k_dim, k_dim)
    ok = below[0] < below[1] and above[0] > above[1]
    print(f"      [{'PASS' if ok else 'FAIL'}] replay crossover sits at half of K")
    failures += 0 if ok else 1

    # The last chunk length whose factors still fit in less than the pair.
    crossover = max(
        length for length in range(1, 4 * k_dim) if floats(length, k_dim, k_dim)[0]
        < floats(length, k_dim, k_dim)[1]
    )
    ok = crossover == 84
    print(f"      [{'PASS' if ok else 'FAIL'}] factors stay smaller up to L={crossover}")
    failures += 0 if ok else 1
    return failures


def _sub_jaxprs(value):
    """Every jaxpr nested inside one equation parameter."""
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        return [value.jaxpr]
    if hasattr(value, "eqns"):
        return [value]
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_sub_jaxprs(item))
        return found
    return []


def loose_precision_dots(fn, *args):
    """Every `dot_general` that asks for less than `HIGHEST` on both operands.

    A float32 `dot_general` at the default precision rounds both operands to
    bfloat16 on the MXU, and `HIGH` keeps only part of the float32 mantissa.
    So an unset precision fails, and so does an explicit `DEFAULT`, `HIGH`,
    `'bfloat16'` or an algorithm preset. Every contraction in this repo spells
    it `HIGHEST`, so an algorithm preset fails even where it's full float32.
    CPU ignores the flag, so the only way a CPU run can see the difference is
    to read it out of the jaxpr.

    Returns:
      `(path, precision)` for each one, where `path` names the nested jaxprs.
    """
    found = []

    def walk(jaxpr, path):
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            here = f"{path}/{name}"
            precision = eqn.params.get("precision")
            if name == "dot_general" and precision != HIGHEST:
                found.append((here, precision))
            for value in eqn.params.values():
                for sub in _sub_jaxprs(value):
                    walk(sub, here)

    walk(jax.make_jaxpr(fn)(*args).jaxpr, "")
    return found


def run_precision(mesh):
    """Every contraction on the KDA path has to ask for `HIGHEST`."""
    chunk_size, chunks, heads, key_dim, value_dim = 16, 8, 2, 8, 8
    tokens = chunks * chunk_size
    k, v, beta, a_raw, dt_bias, a_log, h0 = make_problem(
        13, tokens, chunk_size, heads, key_dim, value_dim, chunk_decay=0.5
    )
    g = gate_log(a_raw, dt_bias, a_log, GATE_LOWER_BOUND)
    a_chunks, b_chunks = chunk_pairs(k, v, beta, g, chunk_size)

    checks = {
        "chunk_pairs": (lambda: chunk_pairs(k, v, beta, g, chunk_size)),
        "chunk_factors": (lambda: chunk_factors(k, v, beta, g, chunk_size)),
        "apply_factored": (
            lambda: factored_replay(k, v, beta, g, h0, chunk_size)
        ),
        "sequential_reference": (
            lambda: sequential_reference(k, v, beta, g, h0, stride=chunk_size)
        ),
        "compose_local": (lambda: compose_local(a_chunks, b_chunks)),
        "replay_local": (lambda: replay_local(a_chunks, b_chunks, h0)),
        "sharded path": (lambda: sharded(a_chunks, b_chunks, h0, mesh)),
    }

    print("  precision")
    failures = 0
    for label, fn in checks.items():
        loose_dots = loose_precision_dots(fn)
        ok = not loose_dots
        print(f"      [{'PASS' if ok else 'FAIL'}] {label} asks for HIGHEST")
        for path, precision in loose_dots:
            print(f"          dot_general at {path} asks for {precision}")
        failures += 0 if ok else 1

    # NEGATIVE CONTROLS: the check has to be able to fail, so run one dot for
    # each way of asking for less than HIGHEST.
    loose_asks = (
        ("no precision", {}),
        ("Precision.DEFAULT", {"precision": lax.Precision.DEFAULT}),
        ("Precision.HIGH", {"precision": lax.Precision.HIGH}),
        ("'bfloat16'", {"precision": "bfloat16"}),
        ("DotAlgorithmPreset.BF16_BF16_F32", {"precision": lax.DotAlgorithmPreset.BF16_BF16_F32}),
    )
    for label, kwargs in loose_asks:

        def loose(kwargs=kwargs):
            return jnp.einsum("chlk,chlv->chkv", a_chunks, b_chunks, **kwargs)

        if loose_precision_dots(loose):
            print(f"      control (dot with {label}): detected")
        else:
            print(f"      FAIL: the jaxpr walk found nothing in a dot with {label}.")
            failures += 1

    # And a dot that asks for HIGHEST has to pass, or the controls prove nothing.
    def tight():
        return jnp.einsum(
            "chlk,chlv->chkv", a_chunks, b_chunks, precision=lax.Precision.HIGHEST
        )

    if loose_precision_dots(tight):
        print("      FAIL: the jaxpr walk flagged a dot that asks for HIGHEST.")
        failures += 1
    else:
        print("      [PASS] a dot that asks for HIGHEST passes")
    return failures


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(too_few_devices(devices, 8))
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))
    print(f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}")
    for label, cfg in (("Kimi-K3      ", KIMI_K3), ("GLM-5.3-Flash", GLM_5_3_FLASH)):
        print(
            f"config: {label} H={cfg.num_heads} head_dim={cfg.head_dim}"
            f" chunk={cfg.chunk_size} conv_kernel={cfg.short_conv_kernel_size}"
            f" gate_bound={cfg.gate_lower_bound}"
            f"  {cfg.num_kda_layers}/{cfg.num_hidden_layers} layers KDA"
            f"  state={cfg.state_bytes / 1024**2:.1f} MiB per sequence, conv windows included"
        )
    print()

    cases = [
        # (name, seed, chunks, chunk_size, heads, K, V, split)
        ("small           ", 1, 16, 16, 2, 8, 8, False),
        ("model shapes    ", 2, 8, KIMI_K3.chunk_size, 4, KIMI_K3.head_dim,
         KIMI_K3.head_dim, False),
        ("many chunks     ", 3, 32, 32, 4, 32, 32, False),
        ("split gate      ", 4, 8, KIMI_K3.chunk_size, 4, 32, 32, True),
    ]

    failures = run_gate()
    print()
    for name, seed, chunks, chunk_size, heads, key_dim, value_dim, split in cases:
        failures += run_case(
            name, seed, chunks, chunk_size, heads, key_dim, value_dim, mesh, split
        )
        print()

    failures += run_wide_gate(36, 8, KIMI_K3.chunk_size, 4, 32, 32)
    print()
    failures += run_positive_gate()
    print()
    failures += run_padding(mesh)
    print()
    failures += run_config()
    print()
    failures += run_cost_model()
    print()
    failures += run_precision(mesh)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
