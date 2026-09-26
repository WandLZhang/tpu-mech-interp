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

"""Causal activation steering inside a compiled JAX forward pass.

Steering adds a direction to the residual stream part way through the model:

    h -> h + alpha * v

Serving has to keep the intervention inside the one program the engine already
compiled. A prefill graph for a 32-chip slice takes minutes to compile, and a
serving engine holds one executable per (batch bucket, sequence bucket).
Anything that makes the steering decision in Python multiplies that set.

So both modes here take their decision as a device array:

1. Static steering picks positions with a boolean mask of shape [T]. The mask
   is an operand, not a trace-time constant, so one executable covers every
   choice of steered positions and every value of alpha. Build the mask inside
   the jit from a padded positions array and the positions stay operands too.
2. Conditional steering reads the activation, applies a predicate, and steers
   the tokens that pass. `conditional_steer` writes that as a masked select.
   `conditional_steer_cond` wraps the same update in `lax.cond`, which skips
   the add when no token in the batch passes.

`lax.cond` traces both branches once and picks between them at run time, so it
costs one program, not one per decision. The Python form,
`python_branch_steer`, is here as the thing to measure against: it reads the
mask on the host, so a new set of positions is a new trace and a new compile.
Each of the three device-side modes is one program across requests. The three
modes are three different programs.

Precision. The engine serves a BF16 residual stream. A BF16 add throws the
steering vector away: at a per-element residual RMS of 300 with alpha = 1 and
5,376 dim, 99% of the delta's components come back bit-identical, and the shift
that lands has 0.18 of the norm you asked for. So the hook widens it to float32,
adds there, and leaves it widened for the rest of the stack. Rounding back to
BF16 after the add loses the same 99%, so the widening has to persist. Reads
work the same way: `projection` normalizes and contracts in float32, which is
what makes the probe scale invariant and keeps the fired set stable. Every
matmul asks for `HIGHEST` precision, because a float32 matmul at TPU's default
runs a single BF16 pass.

Shapes: `h` is [T, D] over flattened tokens, matching how a continuous-batching
engine lays out the residual stream. `v` and the probe are [D]. Masks are [T].
The hook site (which layer) is a Python int and compiles in. Everything that
changes per request stays on device.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

__all__ = [
    "STEER_DTYPE",
    "PRECISION",
    "mask_from_positions",
    "static_steer",
    "applied_cosine",
    "projection",
    "threshold_fire",
    "conditional_steer",
    "conditional_steer_cond",
    "python_branch_steer",
    "block",
    "check_hook_layer",
    "run_forward",
    "sequential_reference",
]

# The hook runs in float32 inside a BF16 model, the same way `STATE_DTYPE` in
# `scan/kda.py` holds the recurrent state.
STEER_DTYPE = jnp.float32

# A float32 matmul at TPU's default precision runs one BF16 pass, which puts a
# read of the residual stream back at BF16 error. Ask for the float32 result.
PRECISION = lax.Precision.HIGHEST


def mask_from_positions(positions: jnp.ndarray, num_tokens: int) -> jnp.ndarray:
    """Boolean [T] mask from an array of token positions.

    Args:
      positions: [P] int array. Pad with -1 to hold the length fixed; -1 never
        matches a real position, so the padding steers nothing. A repeated
        position sets one mask bit, so it steers once.
      num_tokens: T.

    Returns:
      [T] bool. Built on device from an operand, so changing which positions
      get steered doesn't retrace. Call this inside the jit and the positions
      array is an operand too, which is what the padding is for.
    """
    positions = jnp.asarray(positions)
    index = jnp.arange(num_tokens)
    return jnp.any(index[None, :] == positions[:, None], axis=0)


def static_steer(
    h: jnp.ndarray,
    v: jnp.ndarray,
    alpha: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Add `alpha * v` to the masked rows of the residual stream.

    Widens `h` to float32 and returns float32. A BF16 add drops most of the
    delta, so the stream stays wide from here to the end of the stack.

    Args:
      h: [T, D] residual stream, any float dtype.
      v: [D] steering direction.
      alpha: scalar. Pass a device array to keep it out of the signature.
      mask: [T] bool, which tokens to steer.

    Returns:
      [T, D] float32. Unmasked rows are the float32 widening of `h` bit for
      bit. Masked rows always take the add, so `alpha == 0` leaves them
      numerically unchanged but normalizes -0.0 to +0.0.
    """
    wide = h.astype(STEER_DTYPE)
    delta = jnp.asarray(alpha, STEER_DTYPE) * v.astype(STEER_DTYPE)
    return jnp.where(mask[:, None], wide + delta[None, :], wide)


def applied_cosine(
    h_before: jnp.ndarray,
    h_after: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Per-token cosine between the shift the hook landed and `v`.

    Report this when the stream is narrow. It's how you see a steering vector
    that rounded away instead of landing.

    Args:
      h_before: [T, D] stream at the hook site without the intervention.
      h_after: [T, D] stream at the hook site with it.
      v: [D] the direction you asked for.

    Returns:
      [T] float32 in [-1, 1]. Rows that didn't move return 0.
    """
    delta = h_after.astype(STEER_DTYPE) - h_before.astype(STEER_DTYPE)
    v = v.astype(STEER_DTYPE)
    num = jnp.dot(delta, v, precision=PRECISION)
    den = jnp.linalg.norm(delta, axis=-1) * jnp.linalg.norm(v)
    return jnp.where(den > 0, num / jnp.maximum(den, 1e-30), 0.0)


def projection(h: jnp.ndarray, probe: jnp.ndarray) -> jnp.ndarray:
    """Read each token's residual stream along `probe`.

    Normalizes and contracts in float32. Normalizing in `h.dtype` breaks the
    scale invariance below: at BF16 a probe rescaled by 10 moves the projection
    by 1.3e-2 against a projection standard deviation of 0.98, which flips
    tokens across a threshold.

    Args:
      h: [T, D].
      probe: [D] read direction, normalized here so the threshold means the
        same thing whatever scale the probe arrives at.

    Returns:
      [T] float32.
    """
    h = h.astype(STEER_DTYPE)
    probe = probe.astype(STEER_DTYPE)
    scale = jnp.maximum(jnp.linalg.norm(probe), jnp.asarray(1e-12, STEER_DTYPE))
    return jnp.dot(h, probe / scale, precision=PRECISION)


def threshold_fire(
    h: jnp.ndarray,
    probe: jnp.ndarray,
    threshold: jnp.ndarray,
    gate: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Which tokens the predicate selects.

    Compares in float32. A BF16 compare puts 512 tokens on 409 distinct values,
    so ties break arbitrarily under a strict `>` and the fired set drifts from
    the float64 answer. A token that flips moves the output by the whole
    `alpha * v`, not by a rounding-scale amount.

    Args:
      h: [T, D].
      probe: [D] read direction.
      threshold: scalar. Tokens with a projection above it fire.
      gate: optional [T] bool, a hard mask applied on top. Use it to hold the
        intervention off padding, or off the prompt.

    Returns:
      [T] bool.
    """
    fire = projection(h, probe) > jnp.asarray(threshold, STEER_DTYPE)
    if gate is not None:
        fire = jnp.logical_and(fire, gate)
    return fire


def conditional_steer(
    h: jnp.ndarray,
    v: jnp.ndarray,
    alpha: jnp.ndarray,
    probe: jnp.ndarray,
    threshold: jnp.ndarray,
    gate: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Steer the tokens that pass the predicate. Masked select, no branch.

    Returns:
      (h_out [T, D] float32, fire [T] bool).
    """
    fire = threshold_fire(h, probe, threshold, gate)
    return static_steer(h, v, alpha, fire), fire


def conditional_steer_cond(
    h: jnp.ndarray,
    v: jnp.ndarray,
    alpha: jnp.ndarray,
    probe: jnp.ndarray,
    threshold: jnp.ndarray,
    gate: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Same result as `conditional_steer`, with `lax.cond` around the update.

    The branch is whole-batch: when no token fires, the update is skipped. Both
    branches trace once into the same executable, so the decision happens at
    run time and the program count stays at one. The lowered program carries a
    `stablehlo.case`, which the masked-select form doesn't. The saving is real
    only when batches that fire nothing are common, because the branch still
    costs a device-side reduction over [T].

    Returns:
      (h_out [T, D] float32, fire [T] bool).
    """
    fire = threshold_fire(h, probe, threshold, gate)

    def apply(operands):
        h_in, fire_in = operands
        return static_steer(h_in, v, alpha, fire_in)

    def skip(operands):
        return operands[0].astype(STEER_DTYPE)

    return lax.cond(jnp.any(fire), apply, skip, (h, fire)), fire


def python_branch_steer(
    h: jnp.ndarray,
    v: jnp.ndarray,
    alpha: jnp.ndarray,
    positions: tuple[int, ...],
) -> jnp.ndarray:
    """The form to avoid. `positions` is a Python tuple read at trace time.

    Every distinct set of positions is a distinct trace and a distinct
    executable. The test measures that against the masked form.

    Matches `static_steer` value for value, including the dtype, the
    steer-once rule for a repeated position, and the -1 padding, which steers
    nothing. So the only difference the test reads is the program count.
    """
    out = h.astype(STEER_DTYPE)
    delta = jnp.asarray(alpha, STEER_DTYPE) * v.astype(STEER_DTYPE)
    for pos in dict.fromkeys(positions):
        # `mask_from_positions` matches only 0..T-1. An index of -1 would wrap to
        # the last row.
        if 0 <= pos < out.shape[0]:
            out = out.at[pos].add(delta)
    return out


def block(layer_params: tuple, h: jnp.ndarray) -> jnp.ndarray:
    """One residual block, standing in for a decoder layer.

    Small enough to read, shaped like the thing a hook sits inside: the output
    is the input plus a nonlinear update, so a steering vector added here
    propagates through every later layer.
    """
    w_in, w_out = layer_params
    inner = jnp.tanh(jnp.dot(h, w_in.astype(h.dtype), precision=PRECISION))
    return h + jnp.dot(inner, w_out.astype(h.dtype), precision=PRECISION)


def check_hook_layer(hook_layer: int, num_layers: int) -> int:
    """Reject a hook site that no layer index matches.

    An out-of-range site silently runs the whole stack unsteered, so it has to
    be an error. Layers count from 0, which makes `len(params)` the natural
    off-by-one on a 60-layer model.
    """
    if not isinstance(hook_layer, int) or isinstance(hook_layer, bool):
        raise TypeError(f"hook_layer must be an int, got {type(hook_layer).__name__}")
    if not 0 <= hook_layer < num_layers:
        raise ValueError(
            f"hook_layer {hook_layer} is outside 0..{num_layers - 1} for a"
            f" {num_layers}-layer stack"
        )
    return hook_layer


def run_forward(
    params: list,
    h_in: jnp.ndarray,
    hook=None,
    hook_layer: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the stack, applying `hook` to the residual stream after one layer.

    Args:
      params: list of (w_in, w_out) per layer.
      h_in: [T, D] embedded tokens.
      hook: callable [T, D] -> [T, D], or None.
      hook_layer: which layer the hook sits after, counting from 0. A Python
        int, so the site compiles in. The positions and the strength don't.
        Outside `0..len(params) - 1` raises.

    Returns:
      (h_out [T, D], h_hook [T, D]) where `h_hook` is the residual stream right
      after the hook site. Capturing it is what lets a test read the shift
      without unwinding the rest of the stack.

    A steering hook returns float32, so with a BF16 `h_in` the stack runs BF16
    up to the hook site and float32 after it.
    """
    check_hook_layer(hook_layer, len(params))
    h = h_in
    h_hook = h_in
    for i, layer in enumerate(params):
        h = block(layer, h)
        if i == hook_layer:
            if hook is not None:
                h = hook(h)
            h_hook = h
    return h, h_hook


def sequential_reference(
    params: list,
    h_in,
    v,
    alpha: float,
    positions,
    hook_layer: int = 0,
):
    """The forward pass and the steering written out token by token in NumPy.

    The ground truth. It builds no mask and calls nothing in this module, so it
    checks the masked form rather than restating it. A repeated position steers
    once, and a position outside 0..T-1, such as the -1 padding, steers
    nothing, matching the mask.
    """
    import numpy as np

    if not 0 <= hook_layer < len(params):
        raise ValueError(
            f"hook_layer {hook_layer} is outside 0..{len(params) - 1} for a"
            f" {len(params)}-layer stack"
        )
    h = np.asarray(h_in, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h_hook = h
    for i, (w_in, w_out) in enumerate(params):
        w_in = np.asarray(w_in, dtype=np.float64)
        w_out = np.asarray(w_out, dtype=np.float64)
        h = h + np.tanh(h @ w_in) @ w_out
        if i == hook_layer:
            h = h.copy()
            for pos in dict.fromkeys(int(p) for p in positions):
                if 0 <= pos < h.shape[0]:
                    h[pos] = h[pos] + alpha * v
            h_hook = h
    return h, h_hook
