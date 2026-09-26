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

"""Chunked Mamba-2 selective scan, expressed as affine chunk pairs.

`affine_scan.py` shards a recurrence of the form `h' = A @ h + B` across a
sequence axis. This file produces the `(A, B)` pairs for a Mamba-2 layer, one
pair per chunk, so that scan can carry the SSM state.

The Mamba-2 SSD recurrence is affine in the state. Per token, per head:

    h_t = exp(dt_t * A_h) * h_{t-1} + B_t (dt_t x_t)^T

`A_h` is one negative scalar per head, so the decay is a scalar. Fold a chunk
of L tokens and the affine map falls out:

    g_t     = dt_t * A_h                     log decay of one token
    cum_t   = sum_{s <= t} g_s               running sum inside the chunk
    A_chunk = exp(cum_{L-1}) * I             one scalar times the identity
    B_chunk = sum_t exp(cum_{L-1} - cum_t) B_t (dt_t x_t)^T

`exp(cum_{L-1} - cum_t)` is the decay that still applies to token `t` by the
end of the chunk. The sum runs over `t` with no dependence between terms, so a
chunk is one einsum. Only the chunk-to-chunk step is sequential, and that step
is what `affine_scan.py` takes over.

Shapes follow `affine_scan.py`: the state is `[..., K, V]`, `A` is
`[..., K, K]`, `B` is `[..., K, V]`. For Mamba-2 the leading axis is heads,
`K` is `ssm_state_size` and `V` is `mamba_head_dim`. Chunked arrays put the
chunk axis first.

Note the state index comes first here. The reference cache stores the same
numbers as `[heads, mamba_head_dim, ssm_state_size]`, so it's the transpose.
`as_cache_layout` and `from_cache_layout` convert.

`chunk_outputs` turns the chunk boundary states into the per-token layer
output. It adds the `C` projection against the state each chunk inherits, the
causal intra-chunk block, and the `D` skip. `y_t` is what the gated norm and
the output projection consume.

`mamba_ssm_cache_dtype` is float32 even though the weights ship in BF16, so
every function here that does arithmetic upcasts its inputs and returns
float32. `state_dtype` overrides that, which is how the test measures what BF16
accumulation costs. The three functions that only move data, `conv_halo`,
`as_cache_layout` and `from_cache_layout`, pass the dtype through.

Every matmul and einsum here asks for `Precision.HIGHEST`. A float32
`dot_general` at the default precision rounds both operands to BF16 before the
MXU multiplies them, which hands back the accuracy the float32 cache dtype
pays for. CPU always runs true float32, so a CPU test can't see the difference.
`test_mamba2.py` reads the jaxpr instead and fails on any `dot_general` that
leaves the precision open.

`A_chunk` is a scalar times the identity. `chunk_terms` returns that scalar,
one number per chunk per head. `chunk_pairs` materializes the dense
`[..., K, K]` form the generic scan takes, which holds K squared numbers for
the same content: 16,384 times as many at K=128. At the 262,144 token context
that's 16 GiB of `a_chunks` per layer against 1 MiB for the scalar form, so a
caller that can consume the scalar should take `chunk_terms`.

Shapes for Nemotron 3 Super 120B-A12B are in `NEMOTRON_3_SUPER`.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax import lax

# `entering_states` lives beside `replay_local`, whose output it shifts. It
# stays in `__all__` here because `chunk_outputs` reads its result.
from affine_scan import entering_states, pad_chunks

__all__ = [
    "Mamba2Config",
    "NEMOTRON_3_SUPER",
    "PRECISION",
    "STATE_DTYPE",
    "as_cache_layout",
    "causal_conv",
    "chunk_outputs",
    "chunk_pairs",
    "chunk_terms",
    "conv_halo",
    "discretize",
    "entering_states",
    "from_cache_layout",
    "pad_to_chunks",
    "sequential_reference",
]

# `mamba_ssm_cache_dtype`. The state stays float32 in a BF16 model.
STATE_DTYPE = jnp.float32

# Keep float32 contractions in float32 on the MXU. See the module docstring.
PRECISION = lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True)
class Mamba2Config:
    """The fields of a Mamba-2 layer config that the sharded scan reads.

    `time_step_min` and `time_step_max` are the range the `dt_bias`
    initializer draws over. The runtime clamp is a separate pair,
    `time_step_limit`, which `config.json` spells `mamba_dt_limit`. Both
    published Nemotron 3 sizes leave it at the `(0.0, inf)` default, where
    neither end binds. The Mamba-2 CUDA kernels, Megatron-LM and vLLM apply
    that pair, and so does this file.

    The torch path of `NemotronHMamba2Mixer` in `transformers` 5.17 doesn't.
    It builds its own pair, `(time_step_min, inf)`, and floors prefill `dt` at
    0.001. NVIDIA's torch path in the checkpoint repo floors prefill and decode
    alike. Against either one this file differs on every step whose softplus
    lands under 0.001.
    """

    ssm_state_size: int
    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    time_step_min: float = 1e-3
    time_step_max: float = 0.1
    time_step_limit: tuple[float, float] = (0.0, math.inf)

    @property
    def heads_per_group(self) -> int:
        """Heads that share one `B` row. `B` is projected per group."""
        return self.mamba_num_heads // self.n_groups

    @property
    def conv_halo(self) -> int:
        """Tokens a sequence shard needs from the shard before it.

        The depthwise causal conv in front of the SSM reads `conv_kernel`
        tokens. A shard owns all but the first `conv_kernel - 1` of them, so
        it receives that many from its left neighbor.
        """
        return self.conv_kernel - 1

    @property
    def state_bytes_per_head(self) -> int:
        """One head's float32 SSM state."""
        return self.ssm_state_size * self.mamba_head_dim * 4

    @property
    def conv_dim(self) -> int:
        """Channels the depthwise conv runs over.

        The input projection hands the conv three concatenated pieces: the SSM
        input at `mamba_num_heads * mamba_head_dim` channels, then `B` and `C`
        at `n_groups * ssm_state_size` each.
        """
        return (
            self.mamba_num_heads * self.mamba_head_dim
            + 2 * self.n_groups * self.ssm_state_size
        )

    @property
    def conv_state_bytes(self) -> int:
        """One layer's float32 conv state, per sequence.

        The cache holds `[conv_dim, conv_kernel - 1]` beside the SSM state and
        in the same dtype, so decode can finish a window that started in the
        previous step. The current token arrives with the request, so the slot
        carries the `conv_kernel - 1` before it and no more. A cache manager
        that allocates only the SSM slot leaves this one out.
        """
        return self.conv_dim * (self.conv_kernel - 1) * 4

    @property
    def state_bytes_per_layer(self) -> int:
        """One layer's whole recurrent state, per sequence.

        The SSM state over every head, plus the conv state. Neither term grows
        with the sequence length.
        """
        return self.mamba_num_heads * self.state_bytes_per_head + self.conv_state_bytes


NEMOTRON_3_SUPER = Mamba2Config(
    ssm_state_size=128,
    mamba_num_heads=128,
    mamba_head_dim=64,
    n_groups=8,
    conv_kernel=4,
    chunk_size=128,
)


def discretize(
    dt_raw: jnp.ndarray,
    dt_bias: jnp.ndarray,
    a_log: jnp.ndarray,
    time_step_limit: tuple[float, float] = (0.0, math.inf),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Turn the raw `dt` projection and `A_log` into a step and a log decay.

    Follows the Mamba-2 CUDA kernels: add the bias, softplus, clamp to
    `time_step_limit`, then multiply by `A = -exp(A_log)`. Both ends of the
    clamp apply, so a config with a finite upper limit gets it.
    `Mamba2Config` says which reference paths clamp elsewhere.

    Args:
      dt_raw: [T, H] the `dt` slice of the input projection.
      dt_bias: [H]
      a_log: [H] the stored log of `-A`.
      time_step_limit: `(low, high)` clamp applied after the softplus,
        read from `mamba_dt_limit` in `config.json`.

    Returns:
      dt: [T, H] float32 time step.
      log_decay: [T, H] float32 `dt * A`, which is at most zero.
    """
    low, high = time_step_limit
    dt = jax.nn.softplus(dt_raw.astype(STATE_DTYPE) + dt_bias.astype(STATE_DTYPE))
    dt = jnp.clip(dt, low, high)
    a = -jnp.exp(a_log.astype(STATE_DTYPE))
    return dt, dt * a


def _to_heads(b: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    """Expand a per-group `B` to per-head. [T, G, N] -> [T, H, N].

    Head `h` reads group `h // heads_per_group`, so the repeat is contiguous.
    """
    groups = b.shape[-2]
    if num_heads % groups:
        raise ValueError(f"{num_heads} heads don't divide into {groups} groups")
    return jnp.repeat(b, num_heads // groups, axis=-2)


def pad_to_chunks(
    x: jnp.ndarray,
    b: jnp.ndarray,
    dt: jnp.ndarray,
    log_decay: jnp.ndarray,
    chunk_size: int,
    num_devices: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Right pad the token axis up to a whole number of chunks.

    Padding carries `dt = 0` and `log_decay = 0`, so a padded token adds
    nothing to `B_chunk` and multiplies `A_chunk` by one. The state at the end
    of a padded chunk is the state at the last real token.

    A sequence shard also needs the chunk count to divide over the mesh, so
    pass `num_devices` and the pad rounds up to `chunk_size * num_devices`. A
    2,000 token prompt at `chunk_size=128` on 16 chips pads to 2,048, which is
    16 chunks, one per chip. Leave it at 1 for the single-device path.
    `affine_scan.pad_chunks` does the padding, the same one `kda.pad_to_chunks`
    runs.

    Returns:
      The four padded arrays and the token count before padding, which says
      which chunk boundaries are real.

    Raises:
      ValueError: `chunk_size` or `num_devices` is below 1.
    """
    (x, b, dt, log_decay), tokens = pad_chunks((x, b, dt, log_decay), chunk_size, num_devices)
    return x, b, dt, log_decay, tokens


def chunk_terms(
    x: jnp.ndarray,
    b: jnp.ndarray,
    dt: jnp.ndarray,
    log_decay: jnp.ndarray,
    chunk_size: int,
    state_dtype: jnp.dtype = STATE_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fold each chunk into a log decay and a state increment.

    Args:
      x: [T, H, P] the SSM input, before the `dt` scaling.
      b: [T, G, N] per group.
      dt: [T, H]
      log_decay: [T, H] `dt * A` from `discretize`.
      chunk_size: `chunk_size` from the model config.
      state_dtype: accumulation dtype.

    Returns:
      log_total: [C, H] the summed log decay of each chunk. `A_chunk` is
        `exp(log_total)` times the identity.
      b_chunks: [C, H, N, P] the state each chunk adds on its own.
    """
    tokens, heads, head_dim = x.shape
    state_size = b.shape[-1]
    if tokens % chunk_size:
        raise ValueError(f"{tokens} tokens don't divide into chunks of {chunk_size}")
    chunks = tokens // chunk_size

    xd = x.astype(state_dtype) * dt.astype(state_dtype)[..., None]
    xd = xd.reshape(chunks, chunk_size, heads, head_dim)
    bh = _to_heads(b.astype(state_dtype), heads)
    bh = bh.reshape(chunks, chunk_size, heads, state_size)

    g = log_decay.astype(state_dtype).reshape(chunks, chunk_size, heads)
    cum = jnp.cumsum(g, axis=1)
    log_total = cum[:, -1, :]
    # Decay still owed to token t when the chunk ends. The largest exponent is
    # zero, at the last token, so this can't overflow.
    decay = jnp.exp(log_total[:, None, :] - cum)

    b_chunks = jnp.einsum(
        "clhn,clhp->chnp", bh * decay[..., None], xd, precision=PRECISION
    )
    return log_total, b_chunks


def chunk_pairs(
    x: jnp.ndarray,
    b: jnp.ndarray,
    dt: jnp.ndarray,
    log_decay: jnp.ndarray,
    chunk_size: int,
    state_dtype: jnp.dtype = STATE_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Affine `(A, B)` pairs for `affine_scan`, one per chunk.

    `a_chunks` is a scalar times the identity, written out in full because the
    generic scan takes a matrix. It holds K squared numbers per chunk per head
    where `chunk_terms` holds one, so prefer `chunk_terms` when the caller can
    apply the scalar itself.

    Returns:
      a_chunks: [C, H, N, N]
      b_chunks: [C, H, N, P]

    Feed these straight to `compose_local`, `incoming_state` and
    `replay_local`.
    """
    log_total, b_chunks = chunk_terms(x, b, dt, log_decay, chunk_size, state_dtype)
    state_size = b_chunks.shape[-2]
    eye = jnp.eye(state_size, dtype=b_chunks.dtype)
    a_chunks = jnp.exp(log_total)[..., None, None] * eye
    return a_chunks, b_chunks


def chunk_outputs(
    x: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    dt: jnp.ndarray,
    log_decay: jnp.ndarray,
    h_entering: jnp.ndarray,
    chunk_size: int,
    d: jnp.ndarray | None = None,
    state_dtype: jnp.dtype = STATE_DTYPE,
) -> jnp.ndarray:
    """Per-token layer output from each chunk's incoming state.

    The chunk pairs carry the state across shards. This turns that state into
    `y_t`, the thing the gated norm and the output projection read. Three terms
    add up, and the first two are the off-diagonal and diagonal blocks of the
    SSD form:

        y_t = C_t . h_t + D_h x_t
            = exp(cum_t) C_t . h_in
              + sum_{s <= t} exp(cum_t - cum_s) (C_t . B_s) dt_s x_s
              + D_h x_t

    `h_in` is the state entering the chunk that holds token `t`, so a sequence
    shard needs nothing from its neighbors beyond the pair `affine_scan.py`
    already carries. Inside a chunk the sum over `s` is one masked einsum.

    Args:
      x: [T, H, P] the SSM input, before the `dt` scaling. `D` skips this, and
        the second term scales it by `dt`.
      b: [T, G, N] per group.
      c: [T, G, N] per group, the same projection layout as `b`.
      dt: [T, H]
      log_decay: [T, H] `dt * A` from `discretize`.
      h_entering: [C, H, N, P] state entering each chunk, from
        `entering_states`.
      chunk_size: `chunk_size` from the model config.
      d: [H] the skip weight. `None` drops the skip.
      state_dtype: accumulation dtype.

    Returns:
      [T, H, P] the mixer output before the gated norm.
    """
    tokens, heads, head_dim = x.shape
    state_size = b.shape[-1]
    if tokens % chunk_size:
        raise ValueError(f"{tokens} tokens don't divide into chunks of {chunk_size}")
    chunks = tokens // chunk_size

    xd = x.astype(state_dtype) * dt.astype(state_dtype)[..., None]
    xd = xd.reshape(chunks, chunk_size, heads, head_dim)
    bh = _to_heads(b.astype(state_dtype), heads)
    bh = bh.reshape(chunks, chunk_size, heads, state_size)
    ch = _to_heads(c.astype(state_dtype), heads)
    ch = ch.reshape(chunks, chunk_size, heads, state_size)

    g = log_decay.astype(state_dtype).reshape(chunks, chunk_size, heads)
    cum = jnp.cumsum(g, axis=1)

    # Decay from source token s to target token t, zero where s is after t.
    # Mask the exponent rather than the result, because cum_t - cum_s is
    # positive above the diagonal and exp would overflow.
    gap = cum[:, :, None, :] - cum[:, None, :, :]
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), bool))[None, :, :, None]
    decay = jnp.exp(jnp.where(causal, gap, -jnp.inf))

    score = jnp.einsum("clhn,cshn->clsh", ch, bh, precision=PRECISION)
    y = jnp.einsum("clsh,cshp->clhp", score * decay, xd, precision=PRECISION)

    inherited = ch * jnp.exp(cum)[..., None]
    y += jnp.einsum(
        "clhn,chnp->clhp", inherited, h_entering.astype(state_dtype), precision=PRECISION
    )

    y = y.reshape(tokens, heads, head_dim)
    if d is not None:
        y += d.astype(state_dtype)[None, :, None] * x.astype(state_dtype)
    return y


def sequential_reference(
    x: jnp.ndarray,
    b: jnp.ndarray,
    dt: jnp.ndarray,
    log_decay: jnp.ndarray,
    h_init: jnp.ndarray,
    state_dtype: jnp.dtype = STATE_DTYPE,
    stride: int = 1,
) -> jnp.ndarray:
    """The recurrence applied one token at a time.

    This is the thing the chunked form has to match. It builds no chunk and no
    composition, so it checks the reformulation rather than restating it.

    Args:
      stride: keep every `stride`-th state, and always keep the last one. Pass
        `chunk_size` to get the state at each chunk boundary, which is what the
        chunked form returns. A sequence that doesn't fill its last chunk still
        lines up, because `pad_to_chunks` leaves the state at the last real
        token sitting at the end of the padded chunk. The default keeps all T
        states, which is large at real head counts.

    Returns:
      [ceil(T / stride), H, N, P] state after each kept token.
    """
    tokens, heads, _ = x.shape
    if tokens == 0:
        raise ValueError("sequential_reference needs at least one token")
    xd = x.astype(state_dtype) * dt.astype(state_dtype)[..., None]
    bh = _to_heads(b.astype(state_dtype), heads)
    decay = jnp.exp(log_decay.astype(state_dtype))

    h = h_init.astype(state_dtype)
    out = []
    for t in range(tokens):
        h = decay[t][:, None, None] * h + bh[t][..., None] * xd[t][:, None, :]
        if (t + 1) % stride == 0 or t == tokens - 1:
            out.append(h)
    return jnp.stack(out)


def as_cache_layout(h: jnp.ndarray) -> jnp.ndarray:
    """[..., N, P] -> [..., P, N], the layout the reference cache stores."""
    return jnp.swapaxes(h, -1, -2)


def from_cache_layout(h: jnp.ndarray) -> jnp.ndarray:
    """[..., P, N] -> [..., N, P], the layout the affine scan wants."""
    return jnp.swapaxes(h, -1, -2)


def conv_halo(x: jnp.ndarray, conv_kernel: int) -> jnp.ndarray:
    """The tail this shard sends to the shard on its right.

    Moves data and doesn't accumulate, so the dtype passes through.

    Args:
      x: [T, D] tokens this shard owns. A shard has to hold at least
        `conv_kernel - 1` tokens, or its tail can't fill the neighbor's window.
      conv_kernel: `conv_kernel` from the model config.

    Returns:
      [conv_kernel - 1, D]
    """
    if conv_kernel <= 1:
        return x[:0]
    halo = conv_kernel - 1
    if x.shape[0] < halo:
        raise ValueError(
            f"a shard of {x.shape[0]} tokens can't send a {halo} token halo;"
            f" split the sequence so every shard holds at least {halo} tokens"
        )
    return x[-halo:]


def causal_conv(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    prefix: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Depthwise causal conv over the token axis, the op in front of the SSM.

    The projection hands over BF16, and a BF16 four tap accumulation loses
    about three digits, so this upcasts and returns float32 like the rest of
    the file.

    Args:
      x: [T, D]
      weight: [conv_kernel, D]
      bias: [D]
      prefix: [conv_kernel - 1, D] tokens from the left neighbor. Zeros mean
        this shard holds the start of the sequence.

    Returns:
      [T, D] float32
    """
    kernel, dim = weight.shape
    tokens = x.shape[0]
    x = x.astype(STATE_DTYPE)
    weight = weight.astype(STATE_DTYPE)
    bias = bias.astype(STATE_DTYPE)
    if prefix is None:
        prefix = jnp.zeros((kernel - 1, dim), STATE_DTYPE)
    elif prefix.shape[0] != kernel - 1:
        raise ValueError(
            f"a kernel of {kernel} needs a {kernel - 1} token prefix,"
            f" got {prefix.shape[0]}"
        )
    padded = jnp.concatenate([prefix.astype(STATE_DTYPE), x], axis=0)
    windows = jnp.stack([padded[i : i + tokens] for i in range(kernel)])
    return jnp.einsum("ktd,kd->td", windows, weight, precision=PRECISION) + bias
