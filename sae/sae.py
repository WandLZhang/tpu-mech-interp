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

"""Sparse autoencoder for TPU. Train with BatchTopK, serve with JumpReLU.

This follows the method in Appendix B of the Gemma Scope 2 technical report. An SAE is two
matmuls with a sparsifying activation in between:

    f(x) = sigma(W_enc (x - b_dec) + b_enc)
    xhat = W_dec f + b_dec

BatchTopK keeps the `k * batch` largest pre-activations across the whole batch and zeros the
rest, so the active count is fixed per batch rather than per token. A fixed active count lets XLA
compile a sparse decoder with static shapes, which is where the TPU speedup comes from. The
alternative, an L1 or L0 penalty, gives a count that moves every step.

BatchTopK needs the whole batch to decide what fires, so it can't run at inference on one token.
The fix is a conversion: fit one JumpReLU threshold per latent from the activations BatchTopK
kept during training, then serve

    f(x) = JumpReLU_theta(W_enc x + b_enc) = z * (z >= theta)

which is elementwise and needs no top-k at all. `fit_jumprelu_thresholds` does the fit and
`fold_pre_encoder_bias` folds `b_dec` into `b_enc` so inference skips the subtraction too.

Four implementation choices:

1. The selection is an exact top-k by default. `batch_topk` calls `lax.top_k` at
   `recall_target=1.0` and `lax.approx_max_k` below it. The approximate kernel partitions instead
   of sorting, which is faster on a large flat array, and it only exists on TPU. The JumpReLU
   conversion needs an exact selection, because the window in `fit_jumprelu_thresholds` reads the
   kept set as everything at or above the cutoff. So `threshold_stats` runs an exact selection
   whatever `recall_target` says. Both kernels index the flattened `[batch, d_sae]` array in
   int32. `lax.top_k` caps one selection at 2**31 pre-activations and `lax.approx_max_k` at
   2**31 - 1, so a 1M dictionary trains at a batch of 2,048 at most with the exact kernel and
   2,047 with the approximate one. `selection_limit` gives the cap, and `batch_topk` refuses a
   batch past it.
2. The decoder gathers `w_dec` rows at the sparse indices and segment-sums them. A dense
   `f @ w_dec` would multiply through a matrix that's `1 - k/d_sae` zeros.
3. The dictionary shards along the latent axis. Every parameter except `b_dec` carries that axis,
   so the encoder matmul runs per shard. Given the mesh, `decode_sparse` runs in a `shard_map`.
   Each shard gathers the decoder rows it holds and segment-sums them into a `[batch, d_model]`
   partial sum, and one all-reduce adds the partial sums. A training step on the mesh then holds
   four collectives. The pre-activations all-gather, because a global BatchTopK compares every
   latent against every other. The decoder's partial sums all-reduce, and on the way back so do
   the `[k * batch]` gradient of the kept values and the `[batch, d_model]` input gradient that
   reaches `b_dec`. Without the mesh, XLA all-reduces the whole `[k * batch, d_model]` block of
   gathered rows instead, `k` times the bytes. Each shard still gathers a `[k * batch, d_model]`
   block, with a zero row for every selection another shard holds. Selecting per shard drops the
   all-gather, which is what `batch_topk_count(num_shards=S)` is for, at the price of latents in
   different shards never competing.
4. `w_dec` rows stay unit norm, and the part of their gradient parallel to the row is projected
   out before the Adam update. Without the projection, Adam spends steps growing a row that
   renormalization then shrinks back.

Every matmul asks for `PRECISION`, which is `HIGHEST`. At TPU's default precision a float32 matmul
runs one BF16 pass, so a float32 `dtype` would compute its pre-activations at BF16 error, and the
thresholds `fit_jumprelu_thresholds` fits would disagree with the float32 probe the steering hook
reads. A bfloat16 `dtype` runs one pass either way.

Shapes: activations `x` are `[batch, d_model]`, pre-activations `z` are `[batch, d_sae]`,
`w_enc` is `[d_model, d_sae]`, `w_dec` is `[d_sae, d_model]`. Latent vectors are rows of `w_dec`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

__all__ = [
    "LATENT_AXIS",
    "PRECISION",
    "SAEConfig",
    "SAEParams",
    "init_params",
    "cast_params",
    "param_specs",
    "param_shardings",
    "shard_params",
    "latent_shards",
    "encode_pre",
    "batch_topk_count",
    "selection_limit",
    "check_selection",
    "batch_topk",
    "scatter_dense",
    "decode_sparse",
    "decode_dense",
    "jump_relu",
    "forward_batchtopk",
    "forward_jumprelu",
    "reconstruction_loss",
    "normalize_decoder",
    "project_decoder_grad",
    "fold_pre_encoder_bias",
    "ThresholdStats",
    "threshold_stats",
    "calibration_stats",
    "fit_jumprelu_thresholds",
]

# The mesh axis the dictionary shards over. Model parallelism only, as in the report.
LATENT_AXIS = "latent"

# A float32 matmul at TPU's default precision runs one BF16 pass. Every matmul here asks for the
# float32 result, the precision `steering/steering.py` reads the stream at.
PRECISION = lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True)
class SAEConfig:
    """Everything that fixes the shape of the model.

    Args:
      d_model: width of the activation the SAE reads.
      expansion_factor: dictionary size as a multiple of `d_model`.
      k: target active latents per token. BatchTopK enforces `k * batch` per batch.
      recall_target: selection recall. 1.0 runs an exact `lax.top_k`. Below 1.0 runs
        `lax.approx_max_k`, which is faster on TPU and can miss members of the true top set.
      dtype: compute dtype for the forward pass. Reduced precision here is free, because the
        master weights and the optimizer state stay in `param_dtype`.
      param_dtype: dtype of the parameters, and therefore of the Adam moments. Keep this at
        float32. At bfloat16 the second moment increments by 0.001 of itself per step, which
        falls under half an ulp and stalls, and a peak learning rate of 7e-5 is a little over
        one ulp of a unit-norm decoder element.
    """

    d_model: int
    expansion_factor: int
    k: int
    recall_target: float = 1.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor

    def __post_init__(self):
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.expansion_factor < 1:
            raise ValueError(f"expansion_factor must be positive, got {self.expansion_factor}")
        if not 1 <= self.k <= self.d_sae:
            raise ValueError(f"k must be in [1, {self.d_sae}], got {self.k}")
        if not 0.0 < self.recall_target <= 1.0:
            raise ValueError(f"recall_target must be in (0, 1], got {self.recall_target}")


class SAEParams(NamedTuple):
    """Trainable state. A NamedTuple, so optax and `jax.tree` walk it directly."""

    w_enc: jnp.ndarray  # [d_model, d_sae]
    b_enc: jnp.ndarray  # [d_sae]
    w_dec: jnp.ndarray  # [d_sae, d_model]
    b_dec: jnp.ndarray  # [d_model]


def init_params(key: jax.Array, cfg: SAEConfig) -> SAEParams:
    """He-uniform decoder rescaled to unit rows, encoder as its transpose, zero biases.

    The two matrices start tied and then move apart. Nothing re-ties them.
    """
    limit = jnp.sqrt(6.0 / cfg.d_sae)
    w_dec = jax.random.uniform(
        key, (cfg.d_sae, cfg.d_model), dtype=cfg.param_dtype, minval=-limit, maxval=limit
    )
    w_dec = _unit_rows(w_dec)
    return SAEParams(
        w_enc=w_dec.T.copy(),
        b_enc=jnp.zeros((cfg.d_sae,), cfg.param_dtype),
        w_dec=w_dec,
        b_dec=jnp.zeros((cfg.d_model,), cfg.param_dtype),
    )


def cast_params(params: SAEParams, dtype: jnp.dtype) -> SAEParams:
    """Every parameter in one dtype.

    The forward pass casts to `cfg.dtype` through this, which keeps the master weights and the
    Adam moments at `cfg.param_dtype`. The cast transposes to a cast on the way back, so gradients
    arrive in the parameter dtype.
    """
    return jax.tree.map(lambda a: a.astype(dtype), params)


def _unit_rows(w: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
    return w / jnp.maximum(norm, 1e-8)


# --- Sharding -------------------------------------------------------------------------------


def param_specs() -> SAEParams:
    """Partition spec per parameter. Everything with a latent axis splits over the mesh.

    `b_dec` is the only replicated array. It lives on the `d_model` axis, so each shard holds a
    copy and the decoder sum adds it once after the cross-shard reduction.
    """
    return SAEParams(
        w_enc=P(None, LATENT_AXIS),
        b_enc=P(LATENT_AXIS),
        w_dec=P(LATENT_AXIS, None),
        b_dec=P(),
    )


def param_shardings(mesh) -> SAEParams:
    """`param_specs` bound to a mesh, ready for `jax.jit` in_shardings or `device_put`."""
    return SAEParams(*[NamedSharding(mesh, spec) for spec in param_specs()])


def shard_params(params: SAEParams, mesh) -> SAEParams:
    """Lay a parameter set out across `mesh` along the latent axis."""
    return jax.device_put(params, param_shardings(mesh))


def latent_shards(mesh) -> int:
    """How many shards `mesh` splits the dictionary into. 1 for no mesh."""
    if mesh is None:
        return 1
    if LATENT_AXIS not in mesh.shape:
        raise ValueError(f"the mesh has axes {tuple(mesh.shape)}, none of them {LATENT_AXIS!r}")
    return int(mesh.shape[LATENT_AXIS])


# --- Encoder --------------------------------------------------------------------------------


def encode_pre(
    params: SAEParams, x: jnp.ndarray, subtract_pre_bias: bool = False
) -> jnp.ndarray:
    """Pre-activations `[batch, d_sae]`, before any sparsity.

    Training subtracts `b_dec` from the input first. `fold_pre_encoder_bias` removes that step
    for inference by pushing the subtraction into `b_enc`. The matmul runs at `PRECISION`, so the
    pre-activations a threshold gets fit on are the float32 ones the steering probe reads.
    """
    if subtract_pre_bias:
        x = x - params.b_dec
    return jnp.matmul(x, params.w_enc, precision=PRECISION) + params.b_enc


def batch_topk_count(k: int, batch: int, d_sae: int, num_shards: int = 1) -> int:
    """How many latents BatchTopK keeps.

    `k * batch` across the whole batch. Inside a `shard_map` over the latent axis each shard
    selects its own `k * batch / num_shards`, which keeps the selection shard-local. Latents in
    different shards then never compete, which the report calls out as the cost of that choice.

    The per-shard count has to divide evenly. A floor here would shrink the global keep without
    saying so: at k=3, batch=5 and 8 shards it gives 1 per shard, so 8 latents survive where 15
    were asked for.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    total = min(k * batch, batch * d_sae)
    if total % num_shards:
        raise ValueError(
            f"k={k} batch={batch} keeps {total} latents, which {num_shards} shards don't divide"
        )
    count = total // num_shards
    if count < 1:
        raise ValueError(f"k={k} batch={batch} over {num_shards} shards leaves nothing to keep")
    return count


def selection_limit(recall_target: float = 1.0) -> int:
    """The most pre-activations one BatchTopK selection can cover, `batch * d_sae`.

    Both kernels return int32 indices into the flattened array. `lax.top_k` covers 2**31
    elements, indices 0 to 2**31 - 1. `lax.approx_max_k` differentiates through a gather that
    adds the array's length to an int32 index, so its training step stops one element short.
    """
    return 2**31 if recall_target >= 1.0 else 2**31 - 1


def check_selection(batch: int, d_sae: int, recall_target: float = 1.0) -> None:
    """Refuse a BatchTopK selection that its int32 indices can't address.

    Without this, `lax.top_k` raises at trace time and `lax.approx_max_k` raises an int32
    `OverflowError` from its gradient, neither of which names the batch that caused it.
    """
    limit = selection_limit(recall_target)
    if batch * d_sae > limit:
        kernel = "lax.top_k" if recall_target >= 1.0 else "lax.approx_max_k"
        raise ValueError(
            f"BatchTopK over a batch of {batch:,} at d_sae={d_sae:,} selects from "
            f"{batch * d_sae:,} pre-activations, and {kernel} indexes them in int32, which "
            f"reaches {limit:,}. Use a batch of {limit // d_sae:,} or fewer at this width."
        )


def batch_topk(
    pre_acts: jnp.ndarray, count: int, recall_target: float = 1.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Keep the `count` largest positive pre-activations across the whole batch.

    At `recall_target=1.0` this is `lax.top_k`, which is exact on every backend, so a CPU run and
    a TPU run keep the same set. Below 1.0 it's `lax.approx_max_k`, which partitions instead of
    sorting. That kernel only exists on TPU; XLA lowers it to sort-and-slice everywhere else, so a
    CPU run of an approximate selection still returns the exact set and tells you nothing about
    what the TPU will do.

    Args:
      pre_acts: `[batch, d_sae]`.
      count: number of latents to keep, from `batch_topk_count`.
      recall_target: 1.0 for an exact selection, lower for the TPU partition kernel.

    Returns:
      `(values, indices)`, both `[count]`. `indices` index the flattened `[batch * d_sae]` array.
      Values that came out non-positive are clamped to zero, so they decode to nothing and the
      shape stays static.

    Raises:
      ValueError: `batch * d_sae` is past `selection_limit`. The shapes are static, so this fires
        at trace time, on the first step.
    """
    check_selection(pre_acts.shape[0], int(np.prod(pre_acts.shape[1:])), recall_target)
    flat = pre_acts.reshape(-1)
    if recall_target >= 1.0:
        values, indices = lax.top_k(flat, count)
    else:
        values, indices = lax.approx_max_k(flat, count, recall_target=recall_target)
    return jnp.maximum(values, 0.0), indices


def scatter_dense(
    values: jnp.ndarray, indices: jnp.ndarray, batch: int, d_sae: int
) -> jnp.ndarray:
    """Sparse `(values, indices)` back to a dense `[batch, d_sae]` code.

    Only for reading metrics and for comparing against the JumpReLU path. The decoder never
    needs it.
    """
    flat = jnp.zeros((batch * d_sae,), values.dtype).at[indices].set(values)
    return flat.reshape(batch, d_sae)


# --- Decoder --------------------------------------------------------------------------------


def _gather_sum(
    w_dec: jnp.ndarray,
    values: jnp.ndarray,
    token: jnp.ndarray,
    latent: jnp.ndarray,
    batch: int,
) -> jnp.ndarray:
    """`values[i] * w_dec[latent[i]]`, summed into row `token[i]` of a `[batch, d_model]` block."""
    return jax.ops.segment_sum(values[:, None] * w_dec[latent], token, num_segments=batch)


def _shard_local_sum(
    w_dec: jnp.ndarray, values: jnp.ndarray, indices: jnp.ndarray, batch: int, mesh
) -> jnp.ndarray:
    """The sparse decode as a `shard_map` over the latent axis, without `b_dec`.

    Each shard holds `d_sae / S` decoder rows. It zeroes the selected values that index another
    shard's rows, gathers and segment-sums the rest into a `[batch, d_model]` partial sum, and a
    `psum` adds the `S` partial sums. The one collective moves `[batch, d_model]`.
    """
    d_sae = w_dec.shape[0]
    shards = latent_shards(mesh)
    if d_sae % shards:
        raise ValueError(f"d_sae={d_sae} doesn't split into {shards} shards on {LATENT_AXIS!r}")
    width = d_sae // shards

    def body(w_local, vals, idx):
        local = idx % d_sae - lax.axis_index(LATENT_AXIS) * width
        mine = (local >= 0) & (local < width)
        partial = _gather_sum(
            w_local, jnp.where(mine, vals, 0), idx // d_sae, jnp.where(mine, local, 0), batch
        )
        return lax.psum(partial, LATENT_AXIS)

    return jax.shard_map(
        body, mesh=mesh, in_specs=(P(LATENT_AXIS, None), P(), P()), out_specs=P()
    )(w_dec, values, indices)


def decode_sparse(
    params: SAEParams,
    values: jnp.ndarray,
    indices: jnp.ndarray,
    batch: int,
    add_bias: bool = True,
    mesh=None,
) -> jnp.ndarray:
    """Reconstruct from sparse latents. Gather decoder rows, sum within each token.

    Args:
      values: `[count]` latent magnitudes.
      indices: `[count]` indices into the flattened `[batch, d_sae]` code.
      batch: rows in the output.
      add_bias: add `b_dec`, once, after the sum over shards.
      mesh: the mesh `w_dec` shards over. With more than one device on `LATENT_AXIS`, the decode
        runs shard-local in a `shard_map` and all-reduces a `[batch, d_model]` partial sum. With
        none, XLA partitions the plain gather on its own, and on a sharded `w_dec` that
        all-reduces the whole `[count, d_model]` block of gathered rows.

    Returns:
      `[batch, d_model]`.

    Cost is `count * d_model` multiply-adds against `batch * d_sae * d_model` for the dense form.
    At k=100 and a 256k dictionary that's a factor of 2,560.
    """
    if latent_shards(mesh) > 1:
        summed = _shard_local_sum(params.w_dec, values, indices, batch, mesh)
    else:
        d_sae = params.w_dec.shape[0]
        summed = _gather_sum(params.w_dec, values, indices // d_sae, indices % d_sae, batch)
    return summed + params.b_dec if add_bias else summed


def decode_dense(params: SAEParams, codes: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct from a dense `[batch, d_sae]` code."""
    return jnp.matmul(codes, params.w_dec, precision=PRECISION) + params.b_dec


# --- Activations ------------------------------------------------------------------------------


def jump_relu(pre_acts: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Pass a pre-activation through if it clears its own threshold, else zero it.

    Magnitude above the threshold is untouched, so this is a gate and not a shrinkage.

    The comparison is inclusive. A fitted threshold can land on a value BatchTopK kept, and a
    strict `>` would then drop it. In float32 that's one pre-activation in three million, but a
    bfloat16 forward pass holds 256 values per octave, so thousands of pre-activations per batch
    share the cutoff's bucket and 6% of firing latents come out with a threshold equal to their
    own smallest kept value.
    """
    return jnp.where(pre_acts >= theta, pre_acts, 0.0)


def forward_batchtopk(
    params: SAEParams,
    cfg: SAEConfig,
    x: jnp.ndarray,
    subtract_pre_bias: bool = True,
    mesh=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Training forward pass.

    Runs in `cfg.dtype`. The parameters arrive in `cfg.param_dtype` and get cast on the way in,
    so the master weights and the Adam moments keep their own precision. `mesh` is the mesh the
    parameters shard over, which `decode_sparse` needs to decode shard-local.

    Returns:
      `(recon, values, indices)`. `recon` is `[batch, d_model]`.
    """
    batch = x.shape[0]
    compute = cast_params(params, cfg.dtype)
    pre = encode_pre(compute, x.astype(cfg.dtype), subtract_pre_bias=subtract_pre_bias)
    count = batch_topk_count(cfg.k, batch, cfg.d_sae)
    values, indices = batch_topk(pre, count, cfg.recall_target)
    return decode_sparse(compute, values, indices, batch, mesh=mesh), values, indices


def forward_jumprelu(
    params: SAEParams, x: jnp.ndarray, theta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inference forward pass. No top-k, no pre-encoder bias, one threshold per latent.

    Call `fold_pre_encoder_bias` on `params` first.

    Returns:
      `(recon, codes)` with `codes` dense `[batch, d_sae]` and mostly zero.
    """
    codes = jump_relu(encode_pre(params, x), theta)
    return decode_dense(params, codes), codes


def reconstruction_loss(
    params: SAEParams,
    cfg: SAEConfig,
    x: jnp.ndarray,
    subtract_pre_bias: bool = True,
    mesh=None,
) -> tuple[jnp.ndarray, dict]:
    """Mean squared error, and the metrics worth logging alongside it.

    BatchTopK enforces sparsity by construction, so there's no sparsity term. The whole loss is
    the reconstruction error.

    Args:
      params: trainable state.
      cfg: model shape.
      x: `[batch, d_model]` activations.
      subtract_pre_bias: subtract `b_dec` before the encoder, which is what training does. Pass
        `False` for parameters that have been through `fold_pre_encoder_bias`, because their
        `b_enc` already carries `-b_dec @ w_enc` and subtracting again applies it twice.
      mesh: the mesh the parameters shard over, for the shard-local decode.

    Returns:
      `(mse, metrics)`. The reduction runs in float32 whatever `cfg.dtype` says.
    """
    recon, values, indices = forward_batchtopk(
        params, cfg, x, subtract_pre_bias=subtract_pre_bias, mesh=mesh
    )
    x = x.astype(jnp.float32)
    err = x - recon.astype(jnp.float32)
    mse = jnp.mean(jnp.sum(err * err, axis=-1))

    variance = jnp.mean(jnp.sum((x - jnp.mean(x, axis=0)) ** 2, axis=-1))
    live = jnp.zeros((cfg.d_sae,), jnp.bool_).at[indices % cfg.d_sae].max(values > 0)
    metrics = {
        "mse": mse,
        "fvu": mse / jnp.maximum(variance, 1e-8),
        "l0": jnp.sum(values > 0) / x.shape[0],
        "live_fraction": jnp.mean(live.astype(jnp.float32)),
    }
    return mse, metrics


# --- Constraints on the decoder ---------------------------------------------------------------


def normalize_decoder(params: SAEParams) -> SAEParams:
    """Rescale every latent vector back to unit norm. Run it after every update."""
    return params._replace(w_dec=_unit_rows(params.w_dec))


def project_decoder_grad(grads: SAEParams, params: SAEParams) -> SAEParams:
    """Remove the component of the decoder gradient parallel to each unit-norm latent vector.

    That component only changes a row's length, and renormalization undoes it. Leaving it in
    feeds Adam a signal it can't act on, which distorts the second-moment estimate for the
    directions that do move.
    """
    rows = _unit_rows(params.w_dec)
    parallel = jnp.sum(grads.w_dec * rows, axis=-1, keepdims=True) * rows
    return grads._replace(w_dec=grads.w_dec - parallel)


def fold_pre_encoder_bias(params: SAEParams) -> SAEParams:
    """Push the `x - b_dec` subtraction into `b_enc`.

    `W_enc (x - b_dec) + b_enc == W_enc x + (b_enc - W_enc b_dec)`, so pre-activations come out
    identical and a fitted threshold vector carries over unchanged.
    """
    shift = jnp.matmul(params.b_dec, params.w_enc, precision=PRECISION)
    return params._replace(b_enc=params.b_enc - shift)


# --- BatchTopK to JumpReLU --------------------------------------------------------------------


class ThresholdStats(NamedTuple):
    """What one calibration batch says about where the thresholds belong.

    `kept_min` is `[d_sae]`, the smallest value BatchTopK kept for each latent, and `inf` for a
    latent that didn't fire. `cutoff` is the smallest value it kept anywhere in the batch, which
    is the bar it held every latent to.
    """

    kept_min: jnp.ndarray
    cutoff: jnp.ndarray


def threshold_stats(pre_acts: jnp.ndarray, cfg: SAEConfig) -> ThresholdStats:
    """Run BatchTopK on one batch and report where its decision boundary fell.

    The selection here is always exact, whatever `cfg.recall_target` says, and the reduction runs
    in float32. Both are load-bearing for the fit. `fit_jumprelu_thresholds` reads the window
    `[cutoff, kept_min_i]` as empty of occurrences of latent `i`, which holds only when the kept
    set is everything at or above the cutoff. An approximate selector admits values below the
    cutoff and skips values above it, so a threshold fitted inside that window drops activations
    BatchTopK kept.
    """
    pre_acts = pre_acts.astype(jnp.float32)
    batch = pre_acts.shape[0]
    count = batch_topk_count(cfg.k, batch, cfg.d_sae)
    values, indices = batch_topk(pre_acts, count, recall_target=1.0)
    kept = jnp.where(values > 0, values, jnp.inf)
    kept_min = jnp.full((cfg.d_sae,), jnp.inf, jnp.float32).at[indices % cfg.d_sae].min(kept)
    return ThresholdStats(kept_min=kept_min, cutoff=jnp.min(kept))


@functools.partial(jax.jit, static_argnames=("cfg", "subtract_pre_bias"))
def calibration_stats(
    params: SAEParams, x: jnp.ndarray, cfg: SAEConfig, subtract_pre_bias: bool = False
) -> ThresholdStats:
    """What `fit_jumprelu_thresholds` reads off one calibration batch, as one compiled program."""
    return threshold_stats(encode_pre(params, x, subtract_pre_bias=subtract_pre_bias), cfg)


def fit_jumprelu_thresholds(
    params: SAEParams,
    cfg: SAEConfig,
    batches: Iterable[jnp.ndarray],
    subtract_pre_bias: bool = False,
) -> jnp.ndarray:
    """Fit one JumpReLU threshold per latent from a calibration stream.

    Each batch gives a window for latent `i`. The bottom is the cutoff BatchTopK applied to the
    whole batch, since the latent cleared it. The top is the smallest value BatchTopK kept for
    that latent, since anything under that got dropped. The threshold goes in the middle, as the
    geometric mean of the two.

    Over batches, the cutoff is averaged and the kept minimum is taken as a median. A latent that
    fires rarely and hard has a long right tail on its kept minimum, and the median ignores it.
    A latent that never fires gets `inf`, so JumpReLU holds it off.

    Feed this the activation distribution the SAE trained on. A few dozen batches is enough.

    The reduction holds a `[batches, d_sae]` float32 array on the host, which is 256 MiB for a 1M
    dictionary over 64 batches. The masking and the median build temporaries of the same shape on
    top of that, so budget between two and three times the stored array. Measured with
    `tracemalloc`, that case peaks at 593 MiB.

    Args:
      params: trained parameters. Fold the pre-encoder bias first, or pass
        `subtract_pre_bias=True` to match the training-time encoder.
      cfg: the config the SAE trained under. `k` has to match.
      batches: iterable of `[batch, d_model]` activation batches.
      subtract_pre_bias: subtract `b_dec` before the encoder.

    Returns:
      `[d_sae]` float32 thresholds. Float32 whatever `cfg.dtype` is, because a bfloat16 threshold
      snaps onto the same coarse grid as the pre-activations it gates.
    """

    kept_min = []
    cutoff = []
    for x in batches:
        batch_stats = calibration_stats(params, x, cfg, subtract_pre_bias)
        kept_min.append(np.asarray(batch_stats.kept_min, dtype=np.float32))
        cutoff.append(float(batch_stats.cutoff))
    if not kept_min:
        raise ValueError("the calibration stream was empty, so no threshold can be fit")

    kept_min = np.stack(kept_min)
    fired = np.isfinite(kept_min)
    seen = fired.sum(axis=0)
    live = seen > 0

    theta = np.full((cfg.d_sae,), np.inf, dtype=np.float32)
    if live.any():
        bar = np.where(fired, np.asarray(cutoff, np.float32)[:, None], np.float32(0.0)).sum(
            axis=0
        ) / np.maximum(seen, 1)
        margin = np.nanmedian(
            np.where(fired[:, live], kept_min[:, live], np.float32(np.nan)), axis=0
        )
        theta[live] = np.sqrt(bar[live] * margin)
    return jnp.asarray(theta, jnp.float32)
