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

"""Chunked Kimi Delta Attention, expressed as affine chunk pairs.

`affine_scan.py` shards a recurrence of the form `h' = A @ h + B` across a
sequence axis. This file produces the `(A, B)` pairs for a KDA layer, one pair
per chunk, so that scan can carry the recurrent state.

KDA runs a gated delta rule. Per head the state `h` is `[K, V]`, and one token
updates it with a per-channel decay and a rank-one delta write:

    g_t     = gate_lower_bound * sigmoid(exp(A_log) * (a_t + dt_bias))
    alpha_t = exp(g_t)                                 g_t <= 0, one per key channel
    u_t     = beta_t * (v_t - (diag(alpha_t) @ h)^T k_t)
    h_t     = diag(alpha_t) @ h_{t-1} + k_t u_t^T

That `g_t` is the gate every served KDA path computes once the config sets
`gate_lower_bound`, and both published KDA models set it to -5.0. The sigmoid
holds `g_t` between the bound and zero. It isn't the softplus gate
`-exp(A_log) * softplus(a_t + dt_bias)` with a clamp under it. The two agree
only where both saturate. At `a_t + dt_bias = 0` and `A_log = 0` the bounded
gate is -2.5 and the softplus gate is -0.69. `gate_log` computes the bounded
gate, or the softplus gate when the bound is `None`. The bound has no default.
sglang-jax reads a config without `gate_lower_bound` as no bound, and
`transformers` fills in -5.0 for GLM-5.3-Flash, so the caller picks.

`u_t` reads the state, so every token in a chunk depends on the tokens before
it. A Mamba-2 chunk sums independent terms; this one solves a triangular
system. Unrolling still lands on an affine map.

## The fold

Write `c_t` for the cumulative log gate through token `t`, so `c_t = sum_{s<=t}
g_s` and `c_0 = 0`. Unrolling the recurrence to the end of the chunk gives

    h_L = diag(exp(c_L)) @ h_0 + K_g^T @ U

where `K_g` holds `k_t * exp(c_L - c_t)`, the key carried to the chunk end with
the decay it still owes, and `U` holds the `u_t`. Expanding `u_t` the same way
gives a triangular system in `U`:

    (I + M) @ U = diag(beta) @ (V - K_hat @ h_0)
    M[t, s]     = beta_t * sum_d k_t[d] k_s[d] exp(c_t[d] - c_s[d])   for s < t
    K_hat[t]    = k_t * exp(c_t)

`I + M` is unit lower triangular, so one triangular solve splits `U` into a
part that depends on `h_0` and a part that doesn't:

    u = solve(I + M, beta * V)          [L, V]
    w = solve(I + M, beta * K_hat)      [L, K]
    U = u - w @ h_0

Substitute and the chunk is affine:

    h_L = (diag(exp(c_L)) - K_g^T @ w) @ h_0 + K_g^T @ u
    A   = diag(exp(c_L)) - K_g^T @ w
    B   = K_g^T @ u

`u` and `w` never touch `h_0`, so a whole chunk is one triangular solve and two
contractions. Only the chunk-to-chunk step stays sequential, and that step is
what `affine_scan.py` takes over.

## Staying finite over a wide gate

The bounded gate never drops below `gate_lower_bound`, so over a 64-token chunk
`c_L` reaches -320 on a channel that sits on the bound. The softplus gate has no
bound, and with `A_log` drawn from `log(Uniform(1e-9, 16))` one token reaches
about -97. That's fine in float32 until something exponentiates its negation.

With a non-positive gate every exponent this file forms is at most zero:

* `exp(c_t)` and `exp(c_L - c_t)` run over `t <= L`, so both exponents are
  non-positive and both underflow to zero.
* `exp(c_t[d] - c_s[d])` is only ever needed for `s < t`. `pair_decay` masks
  the rest of the square by position, before and after the exponential, so the
  half nobody reads never becomes an infinity that a following multiply by
  zero turns into a NaN.

The factored form `exp(c_t) * exp(-c_s)` is the same number on paper and a NaN
in float32: the first factor underflows to zero and the second overflows to
infinity. `test_kda.py` runs that version as a control.

This rests on the accumulation dtype. `c_L` reaches -320, where one bfloat16
step is 2, so `state_dtype=jnp.bfloat16` returns a finite answer that's wrong
by tens of percent.

It doesn't rest on the sign of the gate. `pair_decay` picks its triangle by
position, not by the sign of `c_t - c_s`, so a gate that rises folds correctly
too. Past `c_t = 88` the float32 `exp` overflows, which is loud.

A head that forgets its whole chunk comes out as `A = 0`. That's the right
answer, not an overflow.

## Precision

Every contraction here asks for `Precision.HIGHEST`. A float32 `dot_general`
at the default precision rounds both operands to bfloat16 before the MXU
multiplies them, which takes this fold from 8e-07 to 3e-03 relative error and
undoes the float32 state dtype. CPU always runs true float32, so a CPU test
can't see the difference. `test_kda.py` reads the jaxpr instead and fails on
any `dot_general` that leaves the precision open.

One op stays out of reach. `solve_triangular` lowers to `triangular_solve`,
which takes no precision parameter, so the 64x64 solve per chunk per head runs
at whatever XLA picks. The system is `I + M` with `cond(I + M)` between 1.0 and
2.3 on the shapes here, so it doesn't amplify what it's handed.

## Shapes

The state is `[..., K, V]`, `A` is `[..., K, K]`, `B` is `[..., K, V]`, the
same as `affine_scan.py`. For KDA the leading axis is heads, and `K` and `V`
are both `head_dim`. Layer inputs are `[T, H, K]` for keys and gates, `[T, H,
V]` for values and `[T, H]` for beta. Chunked arrays put the chunk axis first.

`A` is a diagonal plus a rank-L update, so `chunk_factors` returns the factors
and `chunk_pairs` materializes the dense `[..., K, K]` the generic scan takes.
The factors hold `K + 2 L K + L V` floats against `K^2 + K V` for the pair, so
at `L = 64` and `K = V = 128` they hold 24,704 floats against 32,768. They stay
the smaller of the two up to `L = 84` at that head dimension, and tie at 85.

Replay is the other way around. `apply_factored` costs `2 L K V` against the
`K^2 V` of `A @ h`, so the factors are cheaper only while `L < K / 2`. Both
shipped configs pair `chunk_size = 64` with `head_dim = 128`, where the two
cost the same.

The short depthwise conv in front of the projections is the same op Mamba-2
runs, so `mamba2.causal_conv` and `mamba2.conv_halo` cover the shard halo.

Shapes for the two KDA models are in `KIMI_K3` and `GLM_5_3_FLASH`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular

from affine_scan import pad_chunks

__all__ = [
    "CONV_STATE_DTYPE",
    "GATE_LOWER_BOUND",
    "GLM_5_3_FLASH",
    "KIMI_K3",
    "PRECISION",
    "STATE_DTYPE",
    "KDAConfig",
    "apply_factored",
    "chunk_factors",
    "chunk_pairs",
    "decay_table",
    "gate_log",
    "intra_chunk_system",
    "pad_to_chunks",
    "pair_decay",
    "sequential_reference",
]

# The recurrent state stays float32 in a BF16 model, the same way
# `mamba_ssm_cache_dtype` does.
STATE_DTYPE = jnp.float32

# The short-conv windows sit beside the recurrent state in sglang-jax's
# `RecurrentStatePool`, which keeps them in BF16 unless
# `SGLANG_JAX_CONV_STATE_DTYPE` says otherwise.
CONV_STATE_DTYPE = jnp.bfloat16

# Keep float32 contractions in float32 on the MXU. See the module docstring.
PRECISION = lax.Precision.HIGHEST

# `gate_lower_bound` from `linear_attn_config`. Both published KDA models set
# this value.
GATE_LOWER_BOUND = -5.0


@dataclasses.dataclass(frozen=True)
class KDAConfig:
    """The fields of a KDA layer config that the sharded scan reads.

    `num_heads`, `head_dim`, `short_conv_kernel_size` and `gate_lower_bound`
    come from `linear_attn_config` in the model's `config.json`. `chunk_size`
    doesn't. It's the kernel's chunk length, and 64 is what the reference
    kernel uses.

    `gate_lower_bound` has no default, and neither does the bound `gate_log`
    takes. A number picks the bounded sigmoid gate and `None` picks the
    softplus gate, and the two are different functions. A config without the
    key reads as `None` in `kimi_linear.py` at sglang-jax `eb061d8` and in the
    Kimi K3 port in `upstream/models/`, so Kimi-Linear serves the softplus
    gate. `Glm5NextTextConfig` in `transformers` fills in -5.0 instead.
    """

    num_heads: int
    head_dim: int
    short_conv_kernel_size: int
    num_hidden_layers: int
    num_kda_layers: int
    gate_lower_bound: float | None
    chunk_size: int = 64

    @property
    def conv_halo(self) -> int:
        """Tokens a sequence shard needs from the shard before it.

        The depthwise causal conv in front of the projections reads
        `short_conv_kernel_size` tokens. A shard owns all but the first
        `short_conv_kernel_size - 1` of them, so it receives that many from its
        left neighbor.
        """
        return self.short_conv_kernel_size - 1

    @property
    def state_bytes_per_head(self) -> int:
        """One head's float32 recurrent state."""
        return self.head_dim * self.head_dim * jnp.dtype(STATE_DTYPE).itemsize

    @property
    def recurrent_state_bytes_per_layer(self) -> int:
        """One layer's recurrent state over every head, per sequence."""
        return self.state_bytes_per_head * self.num_heads

    @property
    def conv_state_bytes(self) -> int:
        """One layer's short-conv windows, per sequence.

        A depthwise causal conv runs on each of `q`, `k` and `v`, so the pool
        holds `[3 * num_heads * head_dim, short_conv_kernel_size - 1]` beside
        the recurrent state, in `CONV_STATE_DTYPE`. The current token arrives
        with the request, so each window keeps the tokens before it and no
        more. A budget built from the recurrent state alone leaves this out.
        """
        channels = 3 * self.num_heads * self.head_dim
        return channels * self.conv_halo * jnp.dtype(CONV_STATE_DTYPE).itemsize

    @property
    def state_bytes_per_layer(self) -> int:
        """One layer's whole state, per sequence: the recurrent state plus the conv windows.

        Neither term grows with the sequence length. `mamba2.Mamba2Config`
        adds its two slots the same way.
        """
        return self.recurrent_state_bytes_per_layer + self.conv_state_bytes

    @property
    def state_bytes(self) -> int:
        """Every KDA layer's state, conv windows included, for one sequence."""
        return self.state_bytes_per_layer * self.num_kda_layers


# moonshotai/Kimi-K3. 69 of 93 layers are KDA, the other 24 are gated MLA.
KIMI_K3 = KDAConfig(
    num_heads=96,
    head_dim=128,
    short_conv_kernel_size=4,
    num_hidden_layers=93,
    num_kda_layers=69,
    gate_lower_bound=GATE_LOWER_BOUND,
)

# zai-org/GLM-5.3-Flash. 34 of 45 layers are KDA, the other 11 are sparse MLA.
GLM_5_3_FLASH = KDAConfig(
    num_heads=64,
    head_dim=128,
    short_conv_kernel_size=4,
    num_hidden_layers=45,
    num_kda_layers=34,
    gate_lower_bound=GATE_LOWER_BOUND,
)


def decay_table(a_log: jnp.ndarray, heads: int, channels: int) -> jnp.ndarray:
    """`A_log` as a float32 table that broadcasts against `[heads, channels]`.

    A flat vector is read by its length. `[H]` is one value per head and `[K]`
    one per channel. A flat vector can't say which axis it runs along when
    `heads == channels`, so that case raises. A shape with two or more axes
    says it itself, so `[H, K]`, `[H, 1]`, `[1, K]` and Kimi-Linear's
    `[1, 1, H, 1]` broadcast as they stand. Kimi K3's flat `[128]` is 96
    per-head values padded with zeros to `head_dim`, so slice it to
    `[:num_heads]` first, or this reads it one value per channel.

    Raises:
      ValueError: if `a_log` fits neither layout, if a flat `a_log` could run
        along either axis, or if `a_log` is a scalar.
    """
    a_log = jnp.asarray(a_log).astype(STATE_DTYPE)
    if a_log.ndim == 0:
        raise ValueError(
            "a_log is a scalar. Pass one value per head, one per channel, or a "
            f"table that broadcasts to [{heads}, {channels}]"
        )
    if a_log.ndim == 1:
        size = a_log.shape[0]
        if size == heads == channels and size > 1:
            raise ValueError(
                f"a_log holds {size} values and the layer has {heads} heads of "
                f"{channels} channels, so a flat vector could run along either "
                "axis. Pass it as [H, 1] for one value per head or [1, K] for one "
                "per channel."
            )
        if size == heads:
            return a_log[:, None]
        if size == channels:
            return a_log[None, :]
        raise ValueError(
            f"a_log holds {size} values, which is neither {heads} heads "
            f"nor {channels} channels"
        )
    table = None
    if all(d == 1 for d in a_log.shape[:-2]):
        table = a_log.reshape(a_log.shape[-2:])
    if table is None or any(
        d not in (1, want) for d, want in zip(table.shape, (heads, channels))
    ):
        raise ValueError(
            f"a_log has shape {tuple(a_log.shape)}, which doesn't broadcast to "
            f"[{heads}, {channels}]"
        )
    return table


def gate_log(
    a_raw: jnp.ndarray,
    dt_bias: jnp.ndarray,
    a_log: jnp.ndarray,
    lower_bound: float | None,
) -> jnp.ndarray:
    """Turn the gate projection and the stored parameters into a log gate.

    With a bound, `g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`.
    That's what the sglang-jax KDA kernels compute at `eb061d8`, prefill and
    decode, and what `Glm5NextTextForgetGate` computes in `transformers`.
    Without one, `g = -exp(A_log) * softplus(a + dt_bias)`, the gate the same
    code runs when the config sets no bound. `dt_bias` is one value per (head,
    channel), so the result varies per token, per head and per key channel.

    Every KDA checkpoint here stores `A_log` one value per head, in three
    shapes. Kimi-Linear-48B ships `[1, 1, H, 1]`, and the GLM-5.3-Flash forget
    gate in `transformers` holds `[H]`. Kimi K3 ships a flat `[128]`, which is
    `head_dim` long: its 96 per-head values, then 32 zeros. A flat vector of
    `head_dim` values reads as one per channel, so pass Kimi K3's as
    `A_log[:num_heads]`. `decay_table` reads every layout.

    Args:
      a_raw: [T, H, K] the `f_b_proj` output, reshaped to heads.
      dt_bias: [H, K]
      a_log: the stored `A_log`, in any layout `decay_table` reads.
      lower_bound: `gate_lower_bound` from the layer config, or `None` for a
        config that sets no bound. `None` runs the softplus gate. There's no
        default, because a wrong one computes the other gate with no error.
        `KDAConfig` says how each stack reads a config without the key.

    Returns:
      [T, H, K] float32 log gate, at most zero. With a bound it lies between
      `lower_bound` and zero.

    Raises:
      ValueError: if `decay_table` refuses `a_log`, or if `lower_bound` is
        above zero.
    """
    if lower_bound is not None and lower_bound > 0.0:
        raise ValueError(f"gate_lower_bound must be at most zero, got {lower_bound}")
    heads, channels = a_raw.shape[-2], a_raw.shape[-1]
    rate = jnp.exp(decay_table(a_log, heads, channels))
    shifted = a_raw.astype(STATE_DTYPE) + dt_bias.astype(STATE_DTYPE)
    if lower_bound is None:
        return -rate * jax.nn.softplus(shifted)
    return jnp.asarray(lower_bound, STATE_DTYPE) * jax.nn.sigmoid(rate * shifted)


def pad_to_chunks(
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    g: jnp.ndarray,
    chunk_size: int,
    num_devices: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Right pad the token axis up to a whole number of chunks.

    Padding carries `k = 0`, `beta = 0` and `g = 0`. A padded token writes
    nothing, reads nothing and multiplies the decay by one, so the state at the
    end of a padded chunk is the state at the last real token, and a chunk
    that holds only padding folds to the identity pair.

    `shard_map` needs the chunk count to divide over the mesh, so pass
    `num_devices` and the pad rounds up to `chunk_size * num_devices`. A
    3-chunk prompt on 8 chips then pads to 8 chunks, one per chip. Leave it at
    1 for the single-device path. `affine_scan.pad_chunks` does the padding,
    the same one `mamba2.pad_to_chunks` runs.

    Returns:
      The four padded arrays and the token count before padding. Keep that
      count: the chunk boundary the padding invents lands on a token the
      sequence doesn't have, and only the caller knows which states are real.

    Raises:
      ValueError: `chunk_size` or `num_devices` is below 1.
    """
    (k, v, beta, g), tokens = pad_chunks((k, v, beta, g), chunk_size, num_devices)
    return k, v, beta, g, tokens


def pair_decay(c: jnp.ndarray) -> jnp.ndarray:
    """`exp(c_t - c_s)` on the strictly lower triangle, zero everywhere else.

    Args:
      c: [L, H, K] cumulative log gate.

    Returns:
      [L, L, H, K] indexed `[t, s, head, channel]`.

    Only `s < t` is ever read. The mask picks that triangle by position, on
    both sides of the exponential, so what the rest of the square holds never
    matters. On the first chunk of the wide gate `test_kda.py` draws, a
    quarter of the unmasked upper half overflows to infinity, and a later
    multiply by zero turns that into a NaN.

    Masking by position rather than clamping the difference also keeps the
    lower triangle right when `c` isn't monotone. A clamp at zero is a no-op
    while every gate entry is non-positive and a silent wrong answer when one
    isn't.
    """
    length = c.shape[0]
    strict = jnp.tril(jnp.ones((length, length), dtype=bool), -1)[:, :, None, None]
    diff = c[:, None, :, :] - c[None, :, :, :]
    return jnp.where(strict, jnp.exp(jnp.where(strict, diff, 0.0)), 0.0)


def intra_chunk_system(
    k_c: jnp.ndarray,
    beta_c: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """The unit lower triangular `I + M` one chunk solves against.

    `M[t, s] = beta_t * sum_d k_t[d] k_s[d] exp(c_t[d] - c_s[d])` for `s < t`,
    and zero elsewhere.

    Args:
      k_c: [L, H, K]
      beta_c: [L, H]
      c: [L, H, K] cumulative log gate.

    Returns:
      [H, L, L]
    """
    length = k_c.shape[0]
    pair = jnp.einsum(
        "tshk,thk,shk->hts", pair_decay(c), k_c, k_c, precision=PRECISION
    )
    beta_h = jnp.swapaxes(beta_c, 0, 1)[..., None]
    return jnp.eye(length, dtype=pair.dtype) + beta_h * pair


def _chunk_block(
    k_c: jnp.ndarray,
    v_c: jnp.ndarray,
    beta_c: jnp.ndarray,
    g_c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fold one chunk. All inputs are already float32.

    Args:
      k_c: [L, H, K]
      v_c: [L, H, V]
      beta_c: [L, H]
      g_c: [L, H, K]

    Returns:
      log_last: [H, K] the summed log gate, `c_L`.
      k_g: [H, L, K] keys carried to the chunk end.
      w: [H, L, K] the part of `U` that reads the incoming state.
      u: [H, L, V] the part of `U` that doesn't.
    """
    value_dim = v_c.shape[-1]

    c = jnp.cumsum(g_c, axis=0)
    log_last = c[-1]

    k_hat = k_c * jnp.exp(c)
    k_g = k_c * jnp.exp(log_last[None] - c)

    lower = intra_chunk_system(k_c, beta_c, c)
    beta_h = jnp.swapaxes(beta_c, 0, 1)[..., None]

    # One solve for both right hand sides. u takes the values, w takes the
    # decayed keys, and neither touches the incoming state.
    rhs = jnp.concatenate(
        [beta_h * jnp.swapaxes(v_c, 0, 1), beta_h * jnp.swapaxes(k_hat, 0, 1)],
        axis=-1,
    )
    solved = solve_triangular(lower, rhs, lower=True)
    u = solved[..., :value_dim]
    w = solved[..., value_dim:]

    return log_last, jnp.swapaxes(k_g, 0, 1), w, u


def chunk_factors(
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    g: jnp.ndarray,
    chunk_size: int,
    state_dtype: jnp.dtype = STATE_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fold each chunk into the factors of its affine map.

    Args:
      k: [T, H, K] keys, L2 normalized per head by the layer.
      v: [T, H, V]
      beta: [T, H] the delta-rule write strength, in (0, 1).
      g: [T, H, K] log gate from `gate_log`, at most zero. A gate that rises
        folds correctly too, up to where `exp` overflows.
      chunk_size: the kernel's chunk length.
      state_dtype: accumulation dtype. Leave it at float32. bfloat16 returns a
        finite answer that's wrong by tens of percent once `c_L` runs past a
        few hundred negative.

    Returns:
      log_last: [C, H, K]
      k_g: [C, H, L, K]
      w: [C, H, L, K]
      u: [C, H, L, V]

    `lax.map` runs the chunks one at a time. The pairwise decay is `[L, L, H,
    K]`, which is the largest array in the fold, and mapping holds one chunk of
    it instead of all `C`.
    """
    tokens, heads, key_dim = k.shape
    if tokens % chunk_size:
        raise ValueError(f"{tokens} tokens don't divide into chunks of {chunk_size}")
    chunks = tokens // chunk_size
    value_dim = v.shape[-1]

    k_c = k.astype(state_dtype).reshape(chunks, chunk_size, heads, key_dim)
    v_c = v.astype(state_dtype).reshape(chunks, chunk_size, heads, value_dim)
    beta_c = beta.astype(state_dtype).reshape(chunks, chunk_size, heads)
    g_c = g.astype(state_dtype).reshape(chunks, chunk_size, heads, key_dim)

    def body(chunk):
        return _chunk_block(*chunk)

    return lax.map(body, (k_c, v_c, beta_c, g_c))


def chunk_pairs(
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    g: jnp.ndarray,
    chunk_size: int,
    state_dtype: jnp.dtype = STATE_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Affine `(A, B)` pairs for `affine_scan`, one per chunk.

    Returns:
      a_chunks: [C, H, K, K]
      b_chunks: [C, H, K, V]

    Feed these straight to `compose_local`, `incoming_state` and
    `replay_local`.
    """
    log_last, k_g, w, u = chunk_factors(k, v, beta, g, chunk_size, state_dtype)
    key_dim = k_g.shape[-1]
    eye = jnp.eye(key_dim, dtype=log_last.dtype)
    a_chunks = jnp.exp(log_last)[..., :, None] * eye - jnp.einsum(
        "chlk,chlj->chkj", k_g, w, precision=PRECISION
    )
    b_chunks = jnp.einsum("chlk,chlv->chkv", k_g, u, precision=PRECISION)
    return a_chunks, b_chunks


def apply_factored(
    log_last: jnp.ndarray,
    k_g: jnp.ndarray,
    w: jnp.ndarray,
    u: jnp.ndarray,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """Apply one chunk's affine map without building `A`.

    `h' = exp(c_L) * h + k_g^T @ (u - w @ h)`. Costs `2 L K V` against the
    `K^2 V` the dense form costs, so it's the cheaper replay while `2 L < K`,
    meaning a chunk shorter than half the head dimension. Both shipped configs
    pair `chunk_size = 64` with `head_dim = 128`, where the two cost the same
    and only the smaller footprint of the factors separates them.

    Args:
      log_last: [..., K]
      k_g: [..., L, K]
      w: [..., L, K]
      u: [..., L, V]
      h: [..., K, V]

    Returns:
      [..., K, V]
    """
    read = jnp.einsum("...lk,...kv->...lv", w, h, precision=PRECISION)
    write = jnp.einsum("...lk,...lv->...kv", k_g, u - read, precision=PRECISION)
    return jnp.exp(log_last)[..., :, None] * h + write


def sequential_reference(
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    g: jnp.ndarray,
    h_init: jnp.ndarray,
    state_dtype: jnp.dtype = STATE_DTYPE,
    stride: int = 1,
) -> jnp.ndarray:
    """The gated delta rule applied one token at a time.

    This is the thing the chunked form has to match. It builds no chunk, no
    triangular system and no composition, so it checks the reformulation rather
    than restating it.

    Args:
      k: [T, H, K]
      v: [T, H, V]
      beta: [T, H]
      g: [T, H, K]
      h_init: [H, K, V]
      state_dtype: accumulation dtype.
      stride: keep every `stride`-th state. Pass `chunk_size` to get the state
        at each chunk boundary, which is what the chunked form returns. The
        token count has to be a whole number of strides, so a padded sequence
        goes through `pad_to_chunks` first.

    Returns:
      [T // stride, H, K, V] state after each kept token.

    Raises:
      ValueError: if `stride` doesn't divide the token count. The tail would
        otherwise run and then get dropped, and the caller would line up states
        against chunks that cover different tokens.
    """
    tokens = k.shape[0]
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if tokens % stride:
        raise ValueError(f"{tokens} tokens don't divide into strides of {stride}")
    kf = k.astype(state_dtype)
    vf = v.astype(state_dtype)
    bf = beta.astype(state_dtype)
    alpha = jnp.exp(g.astype(state_dtype))

    h = h_init.astype(state_dtype)
    out = []
    for t in range(tokens):
        h = alpha[t][..., None] * h
        read = jnp.einsum("hk,hkv->hv", kf[t], h, precision=PRECISION)
        write = bf[t][:, None] * (vf[t] - read)
        h = h + kf[t][..., None] * write[:, None, :]
        if (t + 1) % stride == 0:
            out.append(h)
    return jnp.stack(out)
