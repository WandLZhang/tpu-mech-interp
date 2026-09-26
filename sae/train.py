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

"""Train a BatchTopK sparse autoencoder with optax, then convert it to JumpReLU.

Five steps, in this order:

1. Measure one scalar `c` from the first 16 batches so `E[||x/c||^2] == 1`. Raw activation norms
   move over orders of magnitude between layers and sites, so a fixed scale is what lets one
   learning rate work everywhere.
2. Train on `x/c` with Adam, starting on those same 16 batches. The loss is reconstruction error
   alone. After every update, project the parallel part out of the decoder gradient and
   renormalize the latent vectors.
3. Fold `c` back into the parameters, so inference reads raw activations.
4. Fold the pre-encoder bias into `b_enc`, so inference skips a subtraction.
5. Fit one JumpReLU threshold per latent on a calibration stream, so inference skips the top-k.

What ships is a parameter set plus a threshold vector. Serving is two matmuls and a comparison.

Hyperparameters follow the Gemma Scope 2 report: learning rate 7e-5, cosine warmup from 0.1 of
that over 1,000 steps, Adam with betas (0, 0.999) and eps 1e-8, batch 4,096. Beta1 of zero means
Adam keeps no momentum.

Run it against a capture. `scripts/capture_activations.py` writes a manifest beside its shards,
and `--manifest` reads it for the shard list in capture order, for `--d-model`, and for a
`--layer` that names the capture slot. The checkpoint records that slot as `capture_layer`, which
`steering/from_sae.py` turns into the `--steering-layer` a server takes:

    python3 sae/train.py --manifest caps/manifest.json --layer 20 \
        --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz

`--activations` takes a glob of shards from anywhere, `[tokens, d_model]` or
`[tokens, layers, d_model]`. On a 3-D shard `--layer` picks a position on the layer axis, which
is the capture slot only when the capture kept every slot, so name the slot with
`--capture-layer`:

    python3 sae/train.py --activations 'caps/*.npy' --layer 1 --capture-layer 20 \
        --d-model 5376 --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz

Without `--capture-layer` the checkpoint records no slot, and `from_sae.py` leaves the
`--steering-layer` to you. A 2-D shard has no layer axis, so there `--layer` can only mean the
slot, and the checkpoint records it.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import math
import os
import sys
import time
from typing import Callable, Iterable, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sae as sae_lib  # noqa: E402
from sae import LATENT_AXIS, SAEConfig, SAEParams  # noqa: E402

__all__ = [
    "SCALE_BATCHES",
    "TrainConfig",
    "TrainResult",
    "batches_needed",
    "warmup_schedule",
    "project_decoder_updates",
    "build_optimizer",
    "make_step",
    "make_mesh",
    "fit_input_scale",
    "unscale_params",
    "train",
    "activation_stream",
    "manifest_inputs",
    "manifest_slot",
    "glob_slot",
    "save",
    "step_flops",
    "peak_flops_per_chip",
]

# BF16 dense peak per chip from the Cloud TPU pages, keyed by `jax.Device.device_kind`. MFU reads
# against this whatever dtype the SAE trains in, which is the convention the Gemma Scope report uses.
PEAK_BF16_FLOPS = {
    "TPU v4": 275e12,
    "TPU v5 lite": 197e12,
    "TPU v5": 459e12,
    "TPU v6 lite": 918e12,
}


def step_flops(batch_size: int, d_model: int, d_sae: int, k: int) -> int:
    """FLOPs for one training step: the encoder matmul and the sparse decoder, both directions.

    The encoder is the one dense product, `[batch, d_model] x [d_model, d_sae]`, 2 * B * d * m
    forward. Its backward pass runs two more of that size: the weight gradient, and the input
    gradient that reaches `b_dec` through the pre-encoder subtraction. The decoder gathers the
    `k * B` kept rows and sums them, 2 * k * B * d forward and twice that backward, so a step is
    6 * B * d * (m + k). The top-k, the loss and the optimizer are left out, so MFU from this
    reads slightly low.
    """
    return 6 * batch_size * d_model * (d_sae + k)


def peak_flops_per_chip() -> float | None:
    """The BF16 peak of the current device, or None off TPU and on chips not in the table."""
    return PEAK_BF16_FLOPS.get(jax.devices()[0].device_kind)


# The scale fit reads this many batches off the front of the stream, and training starts on them.
SCALE_BATCHES = 16


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Optimizer settings. Defaults are the ones in the report."""

    steps: int = 100_000
    batch_size: int = 4_096
    learning_rate: float = 7e-5
    warmup_steps: int = 1_000
    warmup_floor: float = 0.1
    adam_b1: float = 0.0
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    calibration_batches: int = 64
    log_every: int = 100
    seed: int = 0


class TrainResult(NamedTuple):
    params: SAEParams
    threshold: jnp.ndarray
    scale: float
    history: list


def batches_needed(train_cfg: TrainConfig) -> int:
    """How many batches `train` reads off its stream.

    Training starts on the `SCALE_BATCHES` the scale fit read, so the steps take
    `max(steps, SCALE_BATCHES)` batches, and calibration reads its own after them.
    """
    return max(train_cfg.steps, SCALE_BATCHES) + train_cfg.calibration_batches


def warmup_schedule(
    peak: float, warmup_steps: int, floor: float = 0.1
) -> optax.Schedule:
    """Cosine ramp from `floor * peak` to `peak` over `warmup_steps`, then hold.

    Holding rather than decaying keeps a run resumable. An SAE trains on a stream, so the step
    count is a budget and not a fixed horizon.
    """
    if warmup_steps < 1:
        return optax.constant_schedule(peak)

    def schedule(step):
        t = jnp.clip(step / warmup_steps, 0.0, 1.0)
        shape = 0.5 * (1.0 - jnp.cos(math.pi * t))
        return peak * (floor + (1.0 - floor) * shape)

    return schedule


def project_decoder_updates() -> optax.GradientTransformation:
    """Strip the length-changing part of the decoder gradient before Adam sees it."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("project_decoder_updates needs params; chain it inside optax.chain")
        return sae_lib.project_decoder_grad(updates, params), state

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer(train_cfg: TrainConfig) -> optax.GradientTransformation:
    """Gradient projection first, then Adam on what's left.

    Adam sizes its moments from the parameters, so `SAEConfig.param_dtype` sets the precision of
    the second moment. Keep it at float32. `cfg.dtype` still picks the forward-pass precision.
    """
    return optax.chain(
        project_decoder_updates(),
        optax.adam(
            learning_rate=warmup_schedule(
                train_cfg.learning_rate, train_cfg.warmup_steps, train_cfg.warmup_floor
            ),
            b1=train_cfg.adam_b1,
            b2=train_cfg.adam_b2,
            eps=train_cfg.adam_eps,
        ),
    )


def make_step(
    cfg: SAEConfig, optimizer: optax.GradientTransformation, mesh: Mesh | None = None
) -> Callable:
    """Build the jitted update. Returns `(params, opt_state, metrics)`.

    Parameters keep whatever sharding they arrive with, so the encoder matmul runs per shard on
    the latent axis. Pass the mesh they shard over and the decoder gathers and segment-sums per
    shard too, in a `shard_map`. The selection then all-gathers the pre-activations and the decode
    all-reduces a `[batch, d_model]` partial sum. The backward pass all-reduces the `[k * batch]`
    gradient of the kept values and the `[batch, d_model]` input gradient that reaches `b_dec`.
    Without the mesh the decode all-reduces every gathered row, `[k * batch, d_model]`.
    """

    def step(params, opt_state, x):
        (loss, metrics), grads = jax.value_and_grad(
            sae_lib.reconstruction_loss, has_aux=True
        )(params, cfg, x, mesh=mesh)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = sae_lib.normalize_decoder(params)
        metrics = dict(metrics, loss=loss)
        return params, opt_state, metrics

    return jax.jit(step, donate_argnums=(0, 1))


def make_mesh(cfg: SAEConfig, devices=None) -> Mesh:
    """One-dimensional mesh over the latent axis, sized to what divides the dictionary.

    The report uses model parallelism only. Data stays replicated, so every device sees the whole
    batch and the selection compares every latent against every other.
    """
    devices = list(devices if devices is not None else jax.devices())
    if not devices:
        raise ValueError("make_mesh needs at least one device, got an empty list")
    count = len(devices)
    while count > 1 and cfg.d_sae % count != 0:
        count -= 1
    return Mesh(np.array(devices[:count]), (LATENT_AXIS,))


def fit_input_scale(batches: Iterable[np.ndarray]) -> float:
    """One scalar `c` with `E[||x/c||^2] == 1` over the sample.

    Raises:
      ValueError: the sample is empty, all zero, or holds a NaN or an infinity. A NaN makes `c`
        NaN, and an infinity makes it infinite, which scales every batch to zero.
    """
    total = 0.0
    count = 0
    for x in batches:
        x = np.asarray(x, dtype=np.float64)
        total += float(np.sum(x * x))
        count += x.shape[0]
    if count == 0:
        raise ValueError("no activations to measure a scale from")
    mean_sq_norm = total / count
    if not math.isfinite(mean_sq_norm):
        raise ValueError(
            f"the scale sample holds a non-finite activation: its mean squared norm is "
            f"{mean_sq_norm}"
        )
    if mean_sq_norm <= 0.0:
        raise ValueError("activations are all zero, so there's no scale to fit")
    return math.sqrt(mean_sq_norm)


def unscale_params(params: SAEParams, scale: float) -> SAEParams:
    """Rewrite parameters trained on `x/c` to read raw activations and emit raw reconstructions.

    Pre-activations come out unchanged, so a threshold vector fit either side of this transfers.
    Latent vectors stop being unit norm. They carry activation units now.
    """
    return SAEParams(
        w_enc=params.w_enc / scale,
        b_enc=params.b_enc,
        w_dec=params.w_dec * scale,
        b_dec=params.b_dec * scale,
    )


def train(
    cfg: SAEConfig,
    train_cfg: TrainConfig,
    stream: Iterator[np.ndarray],
    mesh: Mesh | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Run the five steps and return parameters ready to serve.

    Args:
      cfg: model shape. `d_model`, `expansion_factor` and `k` all come from here.
      train_cfg: optimizer settings.
      stream: iterator of `[batch_size, d_model]` activation batches, `batches_needed(train_cfg)`
        of them. The scale fit reads the first `SCALE_BATCHES`, training starts on those same
        batches, and calibration reads the batches after the last one training used.
      mesh: where the dictionary lives. Built from `jax.devices()` when absent.
      log: line sink.

    Returns:
      `TrainResult(params, threshold, scale, history)`.

    Raises:
      ValueError: a batch holds a NaN or an infinity. The scale fit and calibration check their
        own batches. A training batch with one turns the parameters to NaN, the loss at the next
        logged step shows it, and the last step always logs.
    """
    key = jax.random.key(train_cfg.seed)
    mesh = mesh if mesh is not None else make_mesh(cfg)
    log(f"mesh: {LATENT_AXIS}={mesh.shape[LATENT_AXIS]} on {jax.devices()[0].platform}")
    log(f"config: {dataclasses.asdict(cfg) | {'d_sae': cfg.d_sae}}")
    log(f"train: {dataclasses.asdict(train_cfg)}")

    scale_sample = list(itertools.islice(stream, SCALE_BATCHES))
    if not scale_sample:
        raise ValueError("the activation stream was empty")
    scale = fit_input_scale(scale_sample)
    log(f"input scale: {scale:.6f}")

    params = sae_lib.shard_params(sae_lib.init_params(key, cfg), mesh)
    optimizer = build_optimizer(train_cfg)
    opt_state = optimizer.init(params)
    step_fn = make_step(cfg, optimizer, mesh)

    history = []
    completed = 0
    started = time.time()
    # MFU covers the steps between two log lines, so the first step's compile stays out of it.
    peak = peak_flops_per_chip()
    chips = mesh.devices.size
    flops = step_flops(train_cfg.batch_size, cfg.d_model, cfg.d_sae, cfg.k)
    last_log = None
    scaled = (
        jnp.asarray(x, cfg.param_dtype) / scale for x in itertools.chain(scale_sample, stream)
    )
    for step, x in enumerate(itertools.islice(scaled, train_cfg.steps)):
        if x.shape != (train_cfg.batch_size, cfg.d_model):
            raise ValueError(
                f"step {step} got a batch of {x.shape}, "
                f"expected {(train_cfg.batch_size, cfg.d_model)}"
            )
        params, opt_state, metrics = step_fn(params, opt_state, x)
        completed = step + 1
        if step % train_cfg.log_every == 0 or step == train_cfg.steps - 1:
            payload = {k: float(v) for k, v in metrics.items()}  # float() waits for the device
            now = time.time()
            payload["step"] = step
            payload["elapsed_s"] = round(now - started, 2)
            if last_log is not None and peak and now > last_log[1]:
                steps, secs = step - last_log[0], now - last_log[1]
                payload["mfu"] = round(flops * steps / secs / (chips * peak), 4)
            last_log = (step, now)
            history.append(payload)
            log(json.dumps(payload))
            if not math.isfinite(payload["loss"]):
                raise ValueError(
                    f"step {step}: the loss is {payload['loss']}, so a batch at or before this "
                    f"step holds a non-finite activation and the parameters carry it now"
                )

    if completed == 0:
        raise ValueError(f"the stream ran dry before step 0 of {train_cfg.steps}")
    if completed < train_cfg.steps:
        log(f"stream ended after {completed} of {train_cfg.steps} steps")

    params = sae_lib.fold_pre_encoder_bias(unscale_params(params, scale))

    calibration = list(itertools.islice(stream, train_cfg.calibration_batches))
    if not calibration:
        raise ValueError("the stream ran dry before calibration, so no threshold can be fit")
    if len(calibration) < train_cfg.calibration_batches:
        log(f"calibration got {len(calibration)} of {train_cfg.calibration_batches} batches")
    nonfinite = [i for i, x in enumerate(calibration) if not np.isfinite(np.asarray(x)).all()]
    if nonfinite:
        raise ValueError(f"calibration batch(es) {nonfinite} hold a non-finite activation")
    calibration = [jnp.asarray(x, cfg.param_dtype) for x in calibration]
    threshold = sae_lib.fit_jumprelu_thresholds(params, cfg, calibration)
    live = int(jnp.sum(jnp.isfinite(threshold)))
    finite = threshold[jnp.isfinite(threshold)]
    log(
        json.dumps(
            {
                "calibration_batches": len(calibration),
                "live_latents": live,
                "dead_latents": cfg.d_sae - live,
                "threshold_min": float(jnp.min(finite)) if live else None,
                "threshold_median": float(jnp.median(finite)) if live else None,
                "threshold_max": float(jnp.max(finite)) if live else None,
            }
        )
    )
    return TrainResult(params=params, threshold=threshold, scale=scale, history=history)


# --- Activation stream --------------------------------------------------------------------------


def activation_stream(
    paths: list,
    batch_size: int,
    layer: int | None = None,
    shuffle_bytes: int = 4 << 30,
    shuffle_tokens: int | None = None,
    seed: int = 0,
    passes: int = 1,
) -> Iterator[np.ndarray]:
    """Batches of activations read from capture shards, shuffled across shards.

    Capture writes tokens in sequence order, and neighboring tokens correlate hard. An SAE
    trained on that order learns the order. The shuffle buffer draws each batch from a random
    position in a large pool, which is the same trick the Gemma Scope pipeline uses.

    The pool is one preallocated float32 array, filled a slice at a time straight off the mmap and
    shuffled in place. Sizing it in bytes rather than tokens is what keeps it the same size at
    every `d_model`: 1M tokens at `d_model=5376` is 21.0 GiB of float32, and building that pool
    by concatenating a list of arrays holds two copies at once for 42.0 GiB.

    Args:
      paths: `.npy` shards, `[tokens, d_model]` or `[tokens, layers, d_model]`.
      batch_size: tokens per batch.
      layer: which layer to slice out of a three-dimensional shard.
      shuffle_bytes: pool size in bytes. The token count follows from `d_model`.
      shuffle_tokens: pool size in tokens, which overrides `shuffle_bytes` when given.
      seed: buffer draw order.
      passes: how many times to walk the shard list.

    Yields:
      `[batch_size, d_model]` float32 arrays. A partial final batch is dropped.
    """
    if not paths:
        raise ValueError("no activation shards to read")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    rng = np.random.default_rng(seed)
    pool = None
    held = 0

    def emit():
        """Shuffle everything held, hand out whole batches, move the remainder to the front."""
        nonlocal held
        rng.shuffle(pool[:held])
        take = (held // batch_size) * batch_size
        for start in range(0, take, batch_size):
            yield pool[start : start + batch_size].copy()
        pool[: held - take] = pool[take:held]
        held -= take

    for _ in range(passes):
        for path in paths:
            shard = np.load(path, mmap_mode="r")
            if shard.ndim == 3:
                if layer is None:
                    raise ValueError(f"{path} is {shard.shape}; pass --layer to pick a slice")
                shard = shard[:, layer, :]
            elif shard.ndim != 2:
                raise ValueError(f"{path} has shape {shard.shape}, expected 2 or 3 dimensions")
            d_model = shard.shape[1]
            if pool is None:
                capacity = (
                    shuffle_tokens
                    if shuffle_tokens is not None
                    else max(batch_size, shuffle_bytes // (d_model * 4))
                )
                pool = np.empty((capacity, d_model), dtype=np.float32)
            if shard.shape[1] != pool.shape[1]:
                raise ValueError(
                    f"{path} is {shard.shape[1]} wide, the pool is {pool.shape[1]}"
                )
            read = 0
            while read < shard.shape[0]:
                room = pool.shape[0] - held
                if room == 0:
                    yield from emit()
                    room = pool.shape[0] - held
                    if room == 0:
                        raise ValueError(
                            f"the pool holds {pool.shape[0]} tokens, which batch_size="
                            f"{batch_size} can't drain"
                        )
                chunk = min(room, shard.shape[0] - read)
                pool[held : held + chunk] = shard[read : read + chunk]
                held += chunk
                read += chunk
    if pool is not None and held >= batch_size:
        yield from emit()


def manifest_inputs(path: str, layer: int | None) -> tuple:
    """Shard paths, `d_model` and a shard-axis index, read from a capture manifest.

    `scripts/capture_activations.py` owns the format, so this defers to it. The import sits
    inside the function, which keeps `sae/` importable on its own.

    Args:
      path: the manifest, or the directory holding it.
      layer: a model layer index. The manifest maps it to its position on the shard's layer axis.

    Returns:
      `(paths, d_model, axis_index)`. `axis_index` is None for a two-dimensional shard.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, os.pardir, "scripts"))
    import capture_activations as capture  # noqa: E402

    manifest = capture.read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"no capture manifest at {path}")
    paths = capture.manifest_shards(path)
    if not paths:
        raise ValueError(f"{path} lists no shards")
    axis = capture.layer_index(manifest, layer) if layer is not None else None
    if axis is None and manifest.get("layer_axis"):
        raise ValueError(f"{path} holds layers {manifest['layers']}; pass --layer to pick one")
    return paths, int(manifest["d_model"]), axis


def manifest_slot(path: str, layer: int | None) -> int | None:
    """The capture slot a `--manifest` run trains on: `layer`, or the only slot the capture kept."""
    if layer is not None:
        return layer
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, os.pardir, "scripts"))
    import capture_activations as capture  # noqa: E402

    layers = (capture.read_manifest(path) or {}).get("layers") or []
    return int(layers[0]) if len(layers) == 1 else None


def glob_slot(first_shard: str, layer: int | None, capture_layer: int | None) -> int | None:
    """The capture slot an `--activations` run trains on, or None when nothing names it.

    On a 3-D shard `layer` is a position on the layer axis, which says nothing about the slot, so
    only `capture_layer` counts. A 2-D shard has no layer axis, so `layer` names the slot there.
    """
    if np.load(first_shard, mmap_mode="r").ndim == 3 or layer is None:
        return capture_layer
    if capture_layer is not None and capture_layer != layer:
        raise ValueError(
            f"{first_shard} is 2-D, so --layer {layer} names the slot, and --capture-layer "
            f"{capture_layer} names another"
        )
    return layer


def save(
    path: str, cfg: SAEConfig, result: TrainResult, capture_layer: int | None = None
) -> str:
    """Write parameters, thresholds and the config to one `.npz`. Returns the path written.

    `capture_layer` is the slot the training shards came from, which `steering/from_sae.py`
    turns into the `--steering-layer` a server takes. Capture slot `k` is the stream entering
    block `k`, and the steering hook writes the stream leaving the block it's given, so the two
    numbers are one apart. `np.savez` adds `.npz` to a name that lacks it, so the path returned
    carries it too.
    """
    path = path if path.endswith(".npz") else path + ".npz"
    np.savez(
        path,
        w_enc=np.asarray(result.params.w_enc),
        b_enc=np.asarray(result.params.b_enc),
        w_dec=np.asarray(result.params.w_dec),
        b_dec=np.asarray(result.params.b_dec),
        threshold=np.asarray(result.threshold),
        config=json.dumps(
            {
                "d_model": cfg.d_model,
                "expansion_factor": cfg.expansion_factor,
                "d_sae": cfg.d_sae,
                "k": cfg.k,
                "recall_target": cfg.recall_target,
                "input_scale": result.scale,
                "activation": "jumprelu",
                "capture_layer": capture_layer,
            }
        ),
    )
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--activations", help="glob of .npy capture shards")
    source.add_argument(
        "--manifest", help="capture manifest, which carries the shard list and --d-model"
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="the capture slot with --manifest; a position on a 3-D shard's layer axis with "
        "--activations",
    )
    ap.add_argument(
        "--capture-layer",
        type=int,
        default=None,
        help="with --activations, the capture slot the shards hold, which the checkpoint records",
    )
    ap.add_argument("--d-model", type=int, default=None, help="required without --manifest")
    ap.add_argument("--expansion-factor", type=int, required=True)
    ap.add_argument("--k", type=int, required=True, help="target active latents per token")
    ap.add_argument(
        "--recall-target",
        type=float,
        default=1.0,
        help="1.0 selects the true top-k; below 1.0 uses the TPU approx_max_k kernel",
    )
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    ap.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    ap.add_argument("--passes", type=int, default=1, help="walks over the shard list")
    ap.add_argument(
        "--shuffle-bytes",
        type=int,
        default=4 << 30,
        help="shuffle pool size in bytes; the token count follows from --d-model",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="destination .npz")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.manifest:
        if args.capture_layer is not None:
            raise SystemExit(
                "--capture-layer goes with --activations; with --manifest, --layer is the slot"
            )
        paths, d_model, layer = manifest_inputs(args.manifest, args.layer)
        if args.d_model is not None and args.d_model != d_model:
            raise SystemExit(f"--d-model {args.d_model} against {d_model} in {args.manifest}")
        capture_layer = manifest_slot(args.manifest, args.layer)
        axis = f", shard axis {layer}" if layer is not None else ""
        print(
            f"{len(paths)} shard(s) from {args.manifest}, d_model={d_model}, "
            f"capture slot {capture_layer}{axis}"
        )
    else:
        paths = sorted(glob.glob(args.activations))
        if not paths:
            raise SystemExit(f"no shards matched {args.activations!r}")
        if args.d_model is None:
            raise SystemExit("--d-model is required without --manifest")
        d_model, layer = args.d_model, args.layer
        try:
            capture_layer = glob_slot(paths[0], args.layer, args.capture_layer)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(f"{len(paths)} shard(s), capture slot {capture_layer}")
        if capture_layer is None:
            print(
                "no --capture-layer, so the checkpoint records no capture slot and from_sae.py "
                "can't name the --steering-layer"
            )

    cfg = SAEConfig(
        d_model=d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        recall_target=args.recall_target,
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    # A batch too wide for one BatchTopK selection fails at the first step's trace. Say so before
    # reading a single shard.
    try:
        sae_lib.check_selection(train_cfg.batch_size, cfg.d_sae, cfg.recall_target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    # Every step and every calibration batch takes a fresh batch, so a short capture would train
    # for minutes and then die at calibration. Count the rows from the .npy headers first.
    held = sum(np.load(p, mmap_mode="r").shape[0] for p in paths)
    batches = held * args.passes // train_cfg.batch_size
    need = batches_needed(train_cfg)
    if batches < need:
        raise SystemExit(
            f"{held:,} tokens over {args.passes} pass(es) make {batches:,} batches of "
            f"{train_cfg.batch_size}, and training reads {need:,}: "
            f"{max(train_cfg.steps, SCALE_BATCHES):,} for {train_cfg.steps:,} steps, which start "
            f"on the {SCALE_BATCHES} batches the scale fit reads, and "
            f"{train_cfg.calibration_batches} for calibration. "
            f"Lower --steps, raise --passes, or capture more."
        )
    print(f"{held:,} tokens, {batches:,} batches for {need:,}")
    stream = activation_stream(
        paths,
        batch_size=train_cfg.batch_size,
        layer=layer,
        shuffle_bytes=args.shuffle_bytes,
        seed=args.seed,
        passes=args.passes,
    )
    result = train(cfg, train_cfg, stream)
    written = save(args.out, cfg, result, capture_layer=capture_layer)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
