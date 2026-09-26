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

"""Turn a trained sparse autoencoder into steering vectors and probes.

`sae/train.py` writes a dictionary. `steering/steering.py` takes a direction `v`, a probe and a
threshold. This module is the join between them.

A latent is a pair of directions, not one:

    w_dec[f]      what the feature writes into the residual stream. The steering vector.
    w_enc[:, f]   what the feature reads out of it. The probe.

Steering with `w_dec[f]` adds the thing the feature itself adds. Reading with `w_enc[:, f]` asks
the question the feature asks. Steering along the encoder row steers along something else. The two
rows start tied in `init_params`, move apart over training, and nothing re-ties them.

Three conversions happen here.

1. **The vector.** `w_dec[f]`, normalized to unit length in float32. Rows are unit norm during
   training, and `train.unscale_params` then multiplies them by the input scale so inference reads
   raw activations. A loaded row therefore carries that scale, and normalizing here is what makes
   `alpha` mean the same thing across SAEs. With a unit vector, `alpha` is the length of the shift
   in activation units, so `alpha_for_fraction` can size it against the stream.

2. **The probe.** `w_enc[:, f]` unchanged. `steering.projection` normalizes it, so the scale it
   arrives at doesn't matter, only the direction.

3. **The threshold.** This one has arithmetic in it. The JumpReLU inference path fires latent `f`
   when

       w_enc[:, f] . x + b_enc[f] >= theta[f]

   and `steering.threshold_fire` fires a token when

       x . (p / ||p||) > t

   Those are the same predicate at

       t = (theta[f] - b_enc[f]) / ||w_enc[:, f]||

   so `conditional_threshold` divides out the probe norm and moves the encoder bias to the other
   side. The SAE compares inclusively and the probe compares strictly, which splits a token whose
   projection lands on the threshold, so the threshold comes down one float32 ulp. Feed
   `fit_jumprelu_thresholds` output straight in. A latent that never fired during calibration
   carries `inf`, and `inf` survives the conversion, so conditional steering never fires it.
   Static steering reads no threshold, so it would add a dead latent's decoder row to every
   token it steers, a direction the SAE never writes at inference. `main` refuses to write a
   bank that holds one, and `save_bank` warns.

Precision. Everything here runs in float32, which is what `steering/steering.py` documents for the
read side. `steering.projection` widens whatever arrives, so a bf16 probe still contracts wide. A
contraction that returns bf16 instead collapses the fired set: 512 tokens land on 422 distinct
projection values, so tokens tie against the threshold and break arbitrarily, and a token that
flips takes the whole `alpha * v` with it. The contraction asks for `lax.Precision.HIGHEST` for
the same reason.

Shapes follow `sae/sae.py`: `w_enc` is `[d_model, d_sae]`, `w_dec` is `[d_sae, d_model]`, `theta`
is `[d_sae]`. A `SteeringSpec` is one feature. A `SteeringBank` is several, stacked, which is what
a server loads.

    python3 steering/from_sae.py --sae sae_l20.npz --feature 40977 --feature 12 --out steer.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "sae"))
import sae as sae_lib  # noqa: E402
from sae import SAEParams  # noqa: E402
from steering import PRECISION, STEER_DTYPE  # noqa: E402

__all__ = [
    "SteeringSpec",
    "SteeringBank",
    "decoder_direction",
    "encoder_probe",
    "conditional_threshold",
    "from_sae",
    "build_bank",
    "alpha_for_fraction",
    "projection_matrix",
    "load_sae",
    "npz_path",
    "save_bank",
    "load_bank",
    "steering_layer_for",
]


class SteeringSpec(NamedTuple):
    """One feature, ready for `static_steer` and `conditional_steer`.

    Args:
      feature: the latent index this came from.
      vector: `[d_model]` float32, unit norm. Pass as `v`.
      probe: `[d_model]` float32, the encoder row. Pass as `probe`.
      threshold: float32 scalar. Pass as `threshold`. `inf` for a dead latent.
      scale: the length of the decoder row before normalization. The feature writes
        `scale * activation` into the stream, so it says what `alpha` compares against.
    """

    feature: int
    vector: jnp.ndarray
    probe: jnp.ndarray
    threshold: jnp.ndarray
    scale: float


class SteeringBank(NamedTuple):
    """Several features, stacked on a leading axis. What a server loads and holds on device.

    A serving hook gathers a row per token, so the bank is an operand and one executable covers
    every feature in it.
    """

    features: np.ndarray  # [n] int32
    vectors: np.ndarray  # [n, d_model] float32
    probes: np.ndarray  # [n, d_model] float32
    thresholds: np.ndarray  # [n] float32
    scales: np.ndarray  # [n] float32

    @property
    def d_model(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.features.shape[0])


def _check_feature(feature: int, d_sae: int) -> int:
    if isinstance(feature, bool) or not isinstance(feature, (int, np.integer)):
        raise TypeError(f"feature must be an int, got {type(feature).__name__}")
    feature = int(feature)
    if not 0 <= feature < d_sae:
        raise ValueError(f"feature {feature} is outside 0..{d_sae - 1}")
    return feature


def decoder_direction(params: SAEParams, feature: int) -> tuple[jnp.ndarray, float]:
    """The steering vector for one latent: its decoder row, unit norm, float32.

    Returns:
      `(vector [d_model] float32, scale)`. `scale` is the row's length before normalization.

    Raises:
      ValueError: the row is all zero, so it points nowhere.
    """
    feature = _check_feature(feature, params.w_dec.shape[0])
    row = jnp.asarray(params.w_dec[feature], STEER_DTYPE)
    scale = float(jnp.linalg.norm(row))
    if not scale > 0.0:
        raise ValueError(f"w_dec row {feature} has norm {scale}, so it has no direction")
    return row / jnp.asarray(scale, STEER_DTYPE), scale


def encoder_probe(params: SAEParams, feature: int) -> jnp.ndarray:
    """The read direction for one latent: its encoder column, float32.

    Left unnormalized. `steering.projection` divides by the norm, so a probe and the same probe
    times ten select the same tokens.
    """
    feature = _check_feature(feature, params.w_enc.shape[1])
    return jnp.asarray(params.w_enc[:, feature], STEER_DTYPE)


def conditional_threshold(
    params: SAEParams,
    theta: jnp.ndarray,
    feature: int,
    inclusive: bool = True,
) -> jnp.ndarray:
    """The `threshold_fire` threshold that reproduces this latent's JumpReLU gate.

    Args:
      params: trained parameters with the pre-encoder bias folded in, which is what
        `sae/train.py` writes. Call `sae.fold_pre_encoder_bias` first if yours aren't folded, or
        the bias lands on the wrong side and every token clears the bar.
      theta: `[d_sae]` from `fit_jumprelu_thresholds`.
      feature: which latent.
      inclusive: match the SAE's `>=` with the probe's `>` by dropping the threshold one float32
        ulp. Turn it off to keep the arithmetic value.

    Returns:
      float32 scalar. `inf` when the latent is dead, which holds conditional steering off.
      Static steering never reads it.
    """
    feature = _check_feature(feature, params.w_enc.shape[1])
    theta = jnp.asarray(theta, STEER_DTYPE)
    if theta.shape != (params.w_enc.shape[1],):
        raise ValueError(
            f"theta is {theta.shape}, expected {(params.w_enc.shape[1],)} from"
            " fit_jumprelu_thresholds"
        )
    probe = encoder_probe(params, feature)
    norm = jnp.linalg.norm(probe)
    if not float(norm) > 0.0:
        raise ValueError(f"w_enc column {feature} has norm 0, so it reads nothing")
    bias = jnp.asarray(params.b_enc[feature], STEER_DTYPE)
    value = (theta[feature] - bias) / norm
    if inclusive and jnp.isfinite(value):
        value = jnp.asarray(np.nextafter(np.float32(value), np.float32(-np.inf)))
    return jnp.asarray(value, STEER_DTYPE)


def from_sae(
    params: SAEParams,
    theta: jnp.ndarray,
    feature: int,
    subtract_pre_bias: bool = False,
    inclusive: bool = True,
) -> SteeringSpec:
    """Everything `conditional_steer` needs for one latent.

    Args:
      params: trained parameters.
      theta: `[d_sae]` thresholds from `fit_jumprelu_thresholds`.
      feature: which latent.
      subtract_pre_bias: the parameters still expect `x - b_dec` at the encoder. Folds the bias
        first so the threshold comes out right. `sae/train.py` folds before it fits, so its output
        wants `False`.
      inclusive: match the SAE's inclusive compare.
    """
    if subtract_pre_bias:
        params = sae_lib.fold_pre_encoder_bias(params)
    vector, scale = decoder_direction(params, feature)
    return SteeringSpec(
        feature=_check_feature(feature, params.w_dec.shape[0]),
        vector=vector,
        probe=encoder_probe(params, feature),
        threshold=conditional_threshold(params, theta, feature, inclusive=inclusive),
        scale=scale,
    )


def build_bank(
    params: SAEParams,
    theta: jnp.ndarray,
    features: Sequence[int],
    subtract_pre_bias: bool = False,
    inclusive: bool = True,
) -> SteeringBank:
    """Stack several features into one bank.

    Duplicate indices are an error. A server addresses a bank by row, so two rows for one feature
    means two ways to say the same thing and a request that picks the wrong one.
    """
    features = [int(f) for f in features]
    if not features:
        raise ValueError("a bank needs at least one feature")
    if len(set(features)) != len(features):
        seen = {f for f in features if features.count(f) > 1}
        raise ValueError(f"feature(s) {sorted(seen)} appear more than once")
    specs = [
        from_sae(params, theta, f, subtract_pre_bias=subtract_pre_bias, inclusive=inclusive)
        for f in features
    ]
    return SteeringBank(
        features=np.asarray([s.feature for s in specs], np.int32),
        vectors=np.stack([np.asarray(s.vector, np.float32) for s in specs]),
        probes=np.stack([np.asarray(s.probe, np.float32) for s in specs]),
        thresholds=np.asarray([float(s.threshold) for s in specs], np.float32),
        scales=np.asarray([s.scale for s in specs], np.float32),
    )


def alpha_for_fraction(h: jnp.ndarray, fraction: float) -> jnp.ndarray:
    """An `alpha` that shifts each token by `fraction` of the residual stream's own length.

    The steering vector is unit norm, so `alpha` is the length of the shift. Absolute lengths mean
    nothing across layers, because the stream grows by an order of magnitude from the embedding to
    the last layer. This reads the batch and returns a length in its units, using the median token
    norm so one outlier token doesn't set the scale.

    It also says when a bf16 stream will eat the shift. At a per-element RMS of 300 and 5,376 dim,
    a fraction of 0.001 is under half an ulp of the stream, so the add returns the stream
    unchanged. Steer at a fraction you can see in `applied_cosine`.

    Args:
      h: `[T, D]` residual stream at the hook site.
      fraction: shift length as a share of the median token norm.

    Returns:
      float32 scalar.
    """
    if not fraction > 0.0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    norms = jnp.linalg.norm(jnp.asarray(h, STEER_DTYPE), axis=-1)
    return jnp.asarray(fraction, STEER_DTYPE) * jnp.median(norms)


def projection_matrix(bank: SteeringBank, h: jnp.ndarray) -> jnp.ndarray:
    """Every probe in the bank read against every token. `[T, n]` float32.

    Normalizes each probe and contracts in float32, the same way `steering.projection` does for
    one probe. Use it to pick which feature to steer with: the column with the largest margin over
    its threshold is the feature that's already active.
    """
    h = jnp.asarray(h, STEER_DTYPE)
    probes = jnp.asarray(bank.probes, STEER_DTYPE)
    norms = jnp.maximum(
        jnp.linalg.norm(probes, axis=-1, keepdims=True), jnp.asarray(1e-12, STEER_DTYPE)
    )
    return jnp.dot(h, (probes / norms).T, precision=PRECISION)


# --- Files --------------------------------------------------------------------------------------


def load_sae(path: str) -> tuple[SAEParams, jnp.ndarray, dict]:
    """Read what `sae/train.py::save` wrote.

    Returns:
      `(params, theta, config)`. Parameters and thresholds come back float32, whatever the file
      holds, because a bf16 threshold snaps onto the same coarse grid as the pre-activations it
      gates.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in ("w_enc", "b_enc", "w_dec", "b_dec", "threshold") if k not in data]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        params = SAEParams(
            w_enc=jnp.asarray(data["w_enc"], STEER_DTYPE),
            b_enc=jnp.asarray(data["b_enc"], STEER_DTYPE),
            w_dec=jnp.asarray(data["w_dec"], STEER_DTYPE),
            b_dec=jnp.asarray(data["b_dec"], STEER_DTYPE),
        )
        theta = jnp.asarray(data["threshold"], STEER_DTYPE)
        config = json.loads(str(data["config"])) if "config" in data else {}
    return params, theta, config


def npz_path(path: str) -> str:
    """The file `np.savez` writes for `path`. It adds `.npz` to a name that lacks it."""
    return path if path.endswith(".npz") else path + ".npz"


def save_bank(path: str, bank: SteeringBank, meta: dict | None = None) -> str:
    """Write a bank to one `.npz`, ready for a serving hook to load. Returns the path written.

    A row with an `inf` threshold still gets written, with a warning. Conditional steering never
    fires it, but static steering reads no threshold and adds that row to every token it steers.
    """
    path = npz_path(path)
    dead = [int(f) for f, t in zip(bank.features, bank.thresholds) if not np.isfinite(t)]
    if dead:
        warnings.warn(
            f"{path}: feature(s) {dead} have inf thresholds, so static steering adds their "
            f"decoder rows to every token it steers",
            stacklevel=2,
        )
    np.savez(
        path,
        features=bank.features,
        vectors=bank.vectors,
        probes=bank.probes,
        thresholds=bank.thresholds,
        scales=bank.scales,
        meta=json.dumps(meta or {}),
    )
    return path


def steering_layer_for(config: dict) -> int | None:
    """The `--steering-layer` a bank built from this SAE checkpoint takes, or None.

    Capture slot `k` is the residual stream entering block `k`. The steering hook fires after
    the block it names returns, so it writes that same tensor at `k - 1`. `sae/train.py` records
    the slot as `capture_layer`; a checkpoint written without one gives None, and the operator
    picks the flag by hand.
    """
    capture_layer = config.get("capture_layer")
    if capture_layer is None:
        return None
    capture_layer = int(capture_layer)
    if capture_layer < 1:
        raise ValueError(
            f"capture slot {capture_layer} is the embedding output, which no block writes, "
            f"so no --steering-layer reaches it"
        )
    return capture_layer - 1


def load_bank(path: str) -> tuple[SteeringBank, dict]:
    """Read a bank back. Returns `(bank, meta)`."""
    with np.load(path, allow_pickle=False) as data:
        bank = SteeringBank(
            features=np.asarray(data["features"], np.int32),
            vectors=np.asarray(data["vectors"], np.float32),
            probes=np.asarray(data["probes"], np.float32),
            thresholds=np.asarray(data["thresholds"], np.float32),
            scales=np.asarray(data["scales"], np.float32),
        )
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return bank, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sae", required=True, help=".npz written by sae/train.py")
    ap.add_argument(
        "--feature",
        type=int,
        action="append",
        required=True,
        help="a live latent index; repeat for a bank of several",
    )
    ap.add_argument(
        "--subtract-pre-bias",
        action="store_true",
        help="the parameters still expect x - b_dec at the encoder",
    )
    ap.add_argument("--out", required=True, help="destination .npz")
    args = ap.parse_args(argv)

    params, theta, config = load_sae(args.sae)
    bank = build_bank(params, theta, args.feature, subtract_pre_bias=args.subtract_pre_bias)
    # A server steers statically unless a request asks otherwise, and static steering reads no
    # threshold, so a dead latent's `inf` wouldn't hold it off.
    dead = [int(f) for f, t in zip(bank.features, bank.thresholds) if not np.isfinite(t)]
    if dead:
        raise SystemExit(
            f"{args.sae}: feature(s) {dead} never fired during calibration, so their thresholds "
            f"are inf. Conditional steering would never fire them, and static steering would add "
            f"their decoder rows to every token it steers. No bank written; pick live latents "
            f"with steering/pick_feature.py."
        )
    capture_layer = config.get("capture_layer")
    steering_layer = steering_layer_for(config)
    written = save_bank(
        args.out,
        bank,
        meta={
            "sae": os.path.basename(args.sae),
            "sae_config": config,
            "capture_layer": capture_layer,
            "steering_layer": steering_layer,
        },
    )

    for i, feature in enumerate(bank.features):
        print(
            json.dumps(
                {
                    "feature": int(feature),
                    "threshold": float(bank.thresholds[i]),
                    "decoder_norm": float(bank.scales[i]),
                    "probe_norm": float(np.linalg.norm(bank.probes[i])),
                    "enc_dec_cosine": float(
                        np.dot(bank.vectors[i], bank.probes[i])
                        / np.linalg.norm(bank.probes[i])
                    ),
                }
            )
        )
    print(f"wrote {written}: {len(bank)} feature(s), d_model={bank.d_model}")
    if steering_layer is None:
        print(f"{args.sae} records no capture layer, so pick --steering-layer by hand")
    else:
        print(f"capture slot {capture_layer} serves as --steering-layer {steering_layer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
