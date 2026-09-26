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

"""Sequence-sharded affine scan for recurrent attention layers.

Nemotron 3 (Mamba-2), Kimi K3 (KDA) and GLM-5.3-Flash (KDA) all carry a
recurrent state that a paged-attention cache manager doesn't model, and all
three update it with a scan that a sequence shard can't split.

The way out is that the inter-chunk recurrence is affine in the state:

    h' = A @ h + B

Affine maps compose associatively:

    (A_r, B_r) o (A_l, B_l) = (A_r @ A_l,  A_r @ B_l + B_r)

so the scan becomes three steps. Each device folds its local chunks into one
(A, B) pair, the pairs compose across devices, then each device replays locally
from the state it inherits. Each device sends ceil(log2 D) + 1 pairs per layer
over D devices: one per prefix round, then one for the closing shift. A pair
holds a K x K and a K x V matrix per head, so it grows with the head count and
not with the sequence.

This is the approach that went upstream in MaxText as
https://github.com/AI-Hypercomputer/maxtext/pull/4968 for GatedDeltaNet.

Three implementation choices:

1. `compose_local` uses `lax.scan`, not `lax.associative_scan`. The associative
   version materializes the running composition at every chunk.
2. `incoming_state` composes with a Hillis-Steele prefix scan over `ppermute`,
   not an all-gather. The gather sends D - 1 pairs per device per layer, and
   the prefix scan sends ceil(log2 D) + 1: 4 against 7 at D = 8.
3. Both scan bodies are wrapped in `jax.checkpoint`. Without it the autodiff
   residuals stack per chunk.

Every matmul here asks for `Precision.HIGHEST`. A float32 `dot_general` at the
default precision rounds both operands to BF16 before the MXU multiplies them,
and a fold that composes one pair per chunk carries that rounding the length of
the sequence. CPU always runs true float32, so a CPU test can't see the
difference and reads the jaxpr instead.

`shard_map` gives every device the same number of chunks, so the chunk count
has to divide over the mesh. A sequence shorter than that pads with whole
chunks that fold to the identity pair. `pad_chunks` does the padding, and
`mamba2.pad_to_chunks` and `kda.pad_to_chunks` call it with the inputs their
layer folds. `compose_local` also folds zero chunks to the identity pair, which
covers an empty sequence.

Both scans run under `shard_map`'s default `check_vma=True`. A `lax.scan` carry
has to keep one type, and inside `shard_map` that type records which mesh axes
a value varies over. The chunks vary over the sequence axis, and the identity
pair `compose_local` starts from doesn't, nor does an initial state passed in
replicated. `_vary_like` casts the starting carry to vary where the chunks do.

Shapes throughout: state `h` is [..., K, V], `A` is [..., K, K], `B` is
[..., K, V]. Leading dimensions are batch and heads and are carried untouched.
Chunked arrays put the chunk axis first: `[C, ..., K, K]`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

# Keep float32 contractions in float32 on the MXU. See the module docstring.
PRECISION = lax.Precision.HIGHEST

__all__ = [
    "PRECISION",
    "compose",
    "identity_pair",
    "compose_local",
    "entering_states",
    "incoming_state",
    "pad_chunks",
    "replay_local",
    "sequential_reference",
]


def compose(left: tuple, right: tuple) -> tuple:
    """Compose two affine maps. `left` is applied first.

    h -> right(left(h)) = A_r @ (A_l @ h + B_l) + B_r
    """
    a_l, b_l = left
    a_r, b_r = right
    return (
        jnp.matmul(a_r, a_l, precision=PRECISION),
        jnp.matmul(a_r, b_l, precision=PRECISION) + b_r,
    )


def identity_pair(a_like, b_like) -> tuple:
    """The identity affine map, shaped to match one chunk's (A, B).

    Reads only `shape` and `dtype`, so a `jax.ShapeDtypeStruct` works.
    """
    k = a_like.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(k, dtype=a_like.dtype), a_like.shape)
    return (eye, jnp.zeros(b_like.shape, b_like.dtype))


def _varying_axes(x) -> frozenset:
    """Mesh axes `x` varies over inside `shard_map`. Empty outside it."""
    typeof = getattr(jax, "typeof", None)
    if typeof is None:
        return frozenset()
    aval = typeof(x)
    manual = getattr(aval, "manual_axis_type", None)
    if manual is not None:
        return frozenset(manual.varying)
    return frozenset(getattr(aval, "vma", ()))


def _vary_like(x, *likes):
    """Cast `x` to vary over every mesh axis one of `likes` varies over.

    A `lax.scan` carry has to come out of the body with the type it went in
    with. Under `check_vma=True` that type includes the mesh axes a value
    varies over, and a body that mixes in a chunk adds the chunk's axes. So
    the starting carry has to carry them already. Outside `shard_map` nothing
    varies and `x` comes back as it went in.
    """
    missing = frozenset().union(*(_varying_axes(v) for v in likes)) - _varying_axes(x)
    if not missing:
        return x
    axes = tuple(sorted(missing, key=str))
    if hasattr(lax, "pcast"):
        return lax.pcast(x, axes, to="varying")
    return lax.pvary(x, axes)


def compose_local(a_chunks: jnp.ndarray, b_chunks: jnp.ndarray) -> tuple:
    """Fold this device's chunks into a single affine map.

    Args:
      a_chunks: [C, ..., K, K]
      b_chunks: [C, ..., K, V]

    Returns:
      (A, B) for the whole local shard, shapes [..., K, K] and [..., K, V].

    Uses `lax.scan` rather than `associative_scan`, and checkpoints the body.
    Zero chunks fold to the identity pair.
    """

    @jax.checkpoint
    def body(carry, chunk):
        return compose(carry, chunk), None

    # Built from the shapes, not from `a_chunks[0]`, so C == 0 works.
    init = identity_pair(
        jax.ShapeDtypeStruct(a_chunks.shape[1:], a_chunks.dtype),
        jax.ShapeDtypeStruct(b_chunks.shape[1:], b_chunks.dtype),
    )
    init = tuple(_vary_like(part, a_chunks, b_chunks) for part in init)
    (a_total, b_total), _ = lax.scan(body, init, (a_chunks, b_chunks))
    return a_total, b_total


def incoming_state(
    a_local: jnp.ndarray,
    b_local: jnp.ndarray,
    h_init: jnp.ndarray,
    axis_name: str,
) -> jnp.ndarray:
    """State this device inherits from every device before it in the sequence.

    Runs inside `shard_map` over `axis_name`. Composes the per-device affine
    maps with a Hillis-Steele prefix scan over `lax.ppermute`: ceil(log2 D)
    rounds, then one shift, each sending one (A, B) pair. That's O(log D)
    pairs per device, where an all-gather sends O(D).

    Args:
      a_local: [..., K, K] this device's folded A.
      b_local: [..., K, V] this device's folded B.
      h_init:  [..., K, V] state entering the whole sequence.
      axis_name: the mesh axis the sequence is sharded over.

    Returns:
      [..., K, V] state entering this device's first chunk. Device 0 gets
      `h_init` unchanged.
    """
    num_devices = lax.axis_size(axis_name)
    index = lax.axis_index(axis_name)

    a_incl, b_incl = a_local, b_local

    offset = 1
    while offset < num_devices:
        perm = [(src, (src + offset) % num_devices) for src in range(num_devices)]
        a_recv = lax.ppermute(a_incl, axis_name, perm)
        b_recv = lax.ppermute(b_incl, axis_name, perm)

        # Devices with index < offset wrapped around and received a pair that
        # isn't upstream of them. They keep what they have.
        valid = index >= offset
        a_comp, b_comp = compose((a_recv, b_recv), (a_incl, b_incl))
        a_incl = jnp.where(valid, a_comp, a_incl)
        b_incl = jnp.where(valid, b_comp, b_incl)

        offset *= 2

    # Inclusive prefix covers devices 0..index. Shift right by one to get the
    # exclusive prefix, which is what enters this device.
    shift = [(src, (src + 1) % num_devices) for src in range(num_devices)]
    a_excl = lax.ppermute(a_incl, axis_name, shift)
    b_excl = lax.ppermute(b_incl, axis_name, shift)

    first = index == 0
    a_id, b_id = identity_pair(a_local, b_local)
    a_excl = jnp.where(first, a_id, a_excl)
    b_excl = jnp.where(first, b_id, b_excl)

    return jnp.matmul(a_excl, h_init, precision=PRECISION) + b_excl


def replay_local(
    a_chunks: jnp.ndarray,
    b_chunks: jnp.ndarray,
    h_in: jnp.ndarray,
) -> jnp.ndarray:
    """Apply this device's chunks in order, starting from `h_in`.

    `h_in` can come in replicated, one state every shard shares. It's cast
    to vary where the chunks do before the scan starts.

    Returns:
      [C, ..., K, V] state after each local chunk.
    """

    @jax.checkpoint
    def body(h, chunk):
        a, b = chunk
        h_next = jnp.matmul(a, h, precision=PRECISION) + b
        return h_next, h_next

    h_in = _vary_like(h_in, a_chunks, b_chunks)
    _, states = lax.scan(body, h_in, (a_chunks, b_chunks))
    return states


def entering_states(states: jnp.ndarray, h_in: jnp.ndarray) -> jnp.ndarray:
    """Shift chunk-end states so index `c` holds the state entering chunk `c`.

    `replay_local` returns the state after each chunk. A layer's per-token
    output reads the state before each chunk, which is the same list shifted
    right with the shard's incoming state in front.

    Args:
      states: [C, ...] state after each local chunk.
      h_in: [...] state entering the first local chunk.

    Returns:
      [C, ...] in the dtype of `states`.

    Concatenate first and drop the last row after. Slicing `states[:-1]`
    first hands `jnp.concatenate` a zero-length array whenever a shard holds
    one chunk, and a zero-size operand crashes the CPU backend inside
    `shard_map`.
    """
    return jnp.concatenate([h_in[None].astype(states.dtype), states], axis=0)[:-1]


def pad_chunks(arrays, chunk_size: int, num_devices: int = 1) -> tuple[tuple, int]:
    """Right pad each array's token axis to whole chunks that divide over the mesh.

    `shard_map` gives every device the same number of chunks, so the pad rounds
    the token count up to `chunk_size * num_devices`. Zeros go on the end.
    `mamba2.pad_to_chunks` and `kda.pad_to_chunks` say why zero is the identity
    for the inputs their layer folds.

    Args:
      arrays: arrays that share a leading token axis.
      chunk_size: tokens per chunk.
      num_devices: devices the chunks shard over. 1 is the single-device path.

    Returns:
      The padded arrays in order, and the token count before padding. Keep that
      count: a chunk boundary the padding invents lands on a token the sequence
      doesn't have, and only the caller knows which states are real.

    Raises:
      ValueError: `chunk_size` or `num_devices` is below 1.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    if num_devices < 1:
        raise ValueError(f"num_devices must be at least 1, got {num_devices}")
    tokens = arrays[0].shape[0]
    pad = (-tokens) % (chunk_size * num_devices)
    if pad == 0:
        return tuple(arrays), tokens

    def tail(arr):
        widths = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
        return jnp.pad(arr, widths)

    return tuple(tail(arr) for arr in arrays), tokens


def sequential_reference(
    a_chunks: jnp.ndarray,
    b_chunks: jnp.ndarray,
    h_init: jnp.ndarray,
) -> jnp.ndarray:
    """The recurrence applied chunk by chunk on one device.

    This is the thing the sharded path has to match. It never builds a
    composition, so it checks the reformulation rather than restating it.
    """
    h = h_init
    out = []
    for i in range(a_chunks.shape[0]):
        h = jnp.matmul(a_chunks[i], h, precision=PRECISION) + b_chunks[i]
        out.append(h)
    return jnp.stack(out)
