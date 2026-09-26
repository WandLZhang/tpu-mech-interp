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

"""Correctness gate for turning a trained SAE into a steering vector and a probe.

Runs on CPU with a forced 8-device mesh. No TPU needed. Trains a small SAE on the way, which
takes about ten seconds.

    python3 steering/test_from_sae.py

The data has a planted answer. Tokens are sparse positive mixtures of 64 known atoms, and atom 0
is planted into a known 15% of them. So "did the extraction recover the direction" and "does the
probe find the tokens that carry it" both have a number attached.

Seven checks, each with a control that has to fail:

1. The three conversions are the arithmetic the module documents, against a float64 NumPy
   reference built straight from `w_dec`, `w_enc`, `b_enc` and `theta`, to within four float32
   ulps. The inclusive nudge is pinned to one ulp below the conversion's own un-nudged value.
   Float32 arithmetic lands an ulp or two either side of the float64 value, and where it lands
   follows the host's BLAS kernels and XLA's code, so the nudge can't be pinned to the
   reference itself. Controls: the same three read from a different latent, and a threshold one
   ulp lower, which mustn't fire.
2. The probe and the threshold reproduce the SAE's own JumpReLU gate, token for token, against a
   float64 `z >= theta` reference. `subtract_pre_bias` gets its own float64 reference, since a
   fold with the wrong sign also differs from no fold. The fold is held to that reference within
   four ulps, and its nudge is pinned the way check 1 pins it. Controls: thresholds permuted across
   latents, the encoder bias dropped, the probe norm not divided out, the pre-encoder bias folded
   twice, that fold with the wrong sign, the operands rounded to bfloat16, and a contraction
   that returns bfloat16.
3. The steering vector is the planted atom. Controls: the next-best latent, a live latent drawn
   at random, and the encoder row of the right latent.
4. The probe finds the tokens carrying the planted atom. Control: the probe of the random latent.
5. Steering with the extracted spec inside `steering.run_forward` lands `alpha * v` on the tokens
   the probe fires on and nothing elsewhere. Controls: `alpha = 0`, and a dead latent, whose
   threshold is `inf`.
6. A bank survives a round trip through `.npz`, vectors, probes, thresholds and scales alike;
   `save_bank` warns when it writes a dead latent; the capture slot converts to the
   `--steering-layer` that writes the same tensor; and every malformed input raises. Controls: a
   bank of live latents, which mustn't warn, the embedding slot, which no block writes, and each
   rejection re-run with a valid input, which mustn't raise.
7. The command line refuses to write a bank that holds a dead latent, because a server steers
   statically unless asked otherwise and static steering never reads the `inf`. It runs on the
   checkpoint `sae/train.py` writes for this SAE. Control: the same command on a live latent,
   which writes the bank at the path it prints.

The reference is NumPy at float64, written from the definitions in `sae/sae.py`. It calls nothing
in `from_sae.py`.

If a control passes, this file fails itself.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import warnings

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "sae"))
import train as train_lib  # noqa: E402
from sae import LATENT_AXIS, SAEConfig  # noqa: E402
from train import TrainConfig, train  # noqa: E402

import from_sae as convert  # noqa: E402
from steering import (  # noqa: E402
    applied_cosine,
    conditional_steer,
    projection,
    run_forward,
    threshold_fire,
)

AXIS = "tok"
D_MODEL = 32
EXPANSION = 4
ATOMS = 64
K_TRUE = 3
TOKENS = 16_384
TRAIN_TOKENS = 12_288

# Atom 0 is the planted one. It rides on this share of the tokens, on top of the K_TRUE atoms
# every token already carries.
PLANTED = 0
PLANT_RATE = 0.15

# Toy stack for check 5.
LAYERS = 4
HOOK_LAYER = 1
STEER_TOKENS = 512


def make_activations(seed):
    """Tokens from a known dictionary, with one atom planted on a known subset.

    Returns `(x, dictionary, present)`. `present` is `[tokens]` bool, true where atom `PLANTED`
    contributed. Coefficients are positive, since the SAE only represents positive magnitudes.
    """
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(ATOMS, D_MODEL)).astype(np.float32)
    dictionary /= np.linalg.norm(dictionary, axis=-1, keepdims=True)
    dictionary *= (0.5 + rng.random((ATOMS, 1))).astype(np.float32)

    codes = np.zeros((TOKENS, ATOMS), np.float32)
    rows = np.repeat(np.arange(TOKENS), K_TRUE)
    cols = np.concatenate(
        [rng.choice(np.arange(1, ATOMS), size=K_TRUE, replace=False) for _ in range(TOKENS)]
    )
    codes[rows, cols] = 1.0 + rng.exponential(size=TOKENS * K_TRUE).astype(np.float32)

    present = rng.random(TOKENS) < PLANT_RATE
    codes[present, PLANTED] = 1.0 + rng.exponential(size=int(present.sum())).astype(np.float32)

    x = codes @ dictionary
    x += 0.01 * rng.normal(size=x.shape).astype(np.float32)
    return x.astype(np.float32), dictionary, present


def batch_stream(pool, batch_size, seed):
    """Endless random batches drawn from a fixed pool."""
    rng = np.random.default_rng(seed)
    while True:
        yield pool[rng.integers(0, pool.shape[0], size=batch_size)]


def toy_stack(seed, dim, layers=LAYERS):
    """A residual stack small enough to read, for check 5.

    Weights are scaled down so the blocks stay in the linear part of `tanh`.
    """
    rng = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(rng.normal(size=(dim, dim)) / np.sqrt(dim), jnp.float32),
            jnp.asarray(rng.normal(size=(dim, dim)) / np.sqrt(dim), jnp.float32),
        )
        for _ in range(layers)
    ]


# --- The independent reference --------------------------------------------------------------


def reference_extraction(w_enc, b_enc, w_dec, theta, feature):
    """What the three conversions have to produce, in float64 NumPy.

    Written from the definitions in `sae/sae.py`: the decoder row is the latent vector, the
    encoder column is the read direction, and JumpReLU fires when
    `w_enc[:, f] . x + b_enc[f] >= theta[f]`. Solving that for the normalized read gives the
    threshold. Calls nothing in `from_sae.py`.
    """
    w_enc = np.asarray(w_enc, np.float64)
    b_enc = np.asarray(b_enc, np.float64)
    w_dec = np.asarray(w_dec, np.float64)
    theta = np.asarray(theta, np.float64)

    row = w_dec[feature]
    scale = np.linalg.norm(row)
    probe = w_enc[:, feature]
    threshold = (theta[feature] - b_enc[feature]) / np.linalg.norm(probe)
    return row / scale, probe, threshold, scale


def reference_fired(w_enc, b_enc, theta, feature, x):
    """Which tokens the SAE's own JumpReLU gate opens for one latent. float64 NumPy."""
    z = np.asarray(x, np.float64) @ np.asarray(w_enc, np.float64) + np.asarray(b_enc, np.float64)
    return z[:, feature] >= np.float64(np.asarray(theta, np.float64)[feature])


def reference_pre_bias_threshold(params, theta, feature):
    """The threshold for parameters that still subtract `b_dec` before the encoder. float64.

    `w_enc . (x - b_dec) + b_enc >= theta` is `w_enc . x + (b_enc - w_enc . b_dec) >= theta`,
    so folding takes `w_enc . b_dec` out of the bias. Written from `sae/sae.py`, calls nothing
    in `from_sae.py`.
    """
    w_enc = np.asarray(params.w_enc, np.float64)
    b_enc = np.asarray(params.b_enc, np.float64)
    b_dec = np.asarray(params.b_dec, np.float64)
    bias = b_enc[feature] - b_dec @ w_enc[:, feature]
    return (np.asarray(theta, np.float64)[feature] - bias) / np.linalg.norm(w_enc[:, feature])


def reference_fired_pre_bias(params, theta, feature, x):
    """The JumpReLU gate of an SAE that subtracts `b_dec` at the encoder. float64 NumPy."""
    w_enc = np.asarray(params.w_enc, np.float64)
    b_enc = np.asarray(params.b_enc, np.float64)
    b_dec = np.asarray(params.b_dec, np.float64)
    z = (np.asarray(x, np.float64) - b_dec) @ w_enc + b_enc
    return z[:, feature] >= np.asarray(theta, np.float64)[feature]


def bf16_projection(h, probe):
    """The same read with the contraction and its result at bfloat16.

    `steering.projection` widens to float32 before it contracts. This does what it refuses to
    do: round both operands, accumulate narrow, and return narrow. It calls nothing in
    `steering/`, so it says what the widening buys.
    """
    h = jnp.asarray(h, jnp.bfloat16)
    probe = jnp.asarray(probe, jnp.bfloat16)
    norm = jnp.asarray(np.linalg.norm(np.asarray(probe, np.float64)), jnp.bfloat16)
    unit = (probe / norm).astype(jnp.bfloat16)
    return jnp.dot(h, unit, preferred_element_type=jnp.bfloat16)


def max_err(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    return float(np.abs(got - want).max())


def ulp(value):
    """One float32 ulp at this magnitude."""
    return float(np.spacing(np.float32(max(abs(float(value)), 1e-30))))


def in_ulps(err, want):
    """An absolute error in float32 ulps, at the scale of the largest value in `want`."""
    return err / ulp(np.abs(np.asarray(want, np.float64)).max())


def pinned_one_ulp(got, arithmetic):
    """Is `got` the float32 one ulp below `arithmetic`, and not two, and not zero?

    The inclusive nudge has to be one ulp. `got < arithmetic` says it moved, and the float32
    one ulp under `arithmetic` not being above `got` says it moved no further. A float32 in
    `[below, arithmetic)` is `below`.
    """
    edge = np.float32(arithmetic)
    got = np.float32(got)
    below = np.nextafter(edge, np.float32(-np.inf))
    return bool(edge > got) and not bool(below > got)


# --- Checks ---------------------------------------------------------------------------------


def check_arithmetic(params, theta, feature, other):
    """1. The conversions match the float64 reference, and read the feature they were asked for."""
    failures = 0
    spec = convert.from_sae(params, theta, feature)
    want_v, want_p, want_t, want_scale = reference_extraction(
        params.w_enc, params.b_enc, params.w_dec, theta, feature
    )
    control_v, control_p, control_t, _ = reference_extraction(
        params.w_enc, params.b_enc, params.w_dec, theta, other
    )

    # The tolerance is in float32 ulps at the scale of the value, not in absolute units. An
    # absolute 1e-6 is 33 ulps at a threshold of 0.27, which is room for a defect to hide in.
    # The threshold row compares against the arithmetic value, so it reads the un-nudged
    # conversion; the nudge itself is pinned separately below.
    tol_ulps = 4
    rows = [
        ("vector", spec.vector, want_v, control_v),
        ("probe", spec.probe, want_p, control_p),
        (
            "threshold",
            convert.conditional_threshold(params, theta, feature, inclusive=False),
            want_t,
            control_t,
        ),
    ]
    for name, got, want, control in rows:
        err = in_ulps(max_err(got, want), want)
        control_err = in_ulps(max_err(got, control), want)
        ok = err <= tol_ulps
        detected = control_err > tol_ulps
        print(
            f"  [{'PASS' if ok and detected else 'FAIL'}] {name:<22} err={err:.2f} ulp"
            f"  control (latent {other})={control_err:.3g} ulp"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        failures += not (ok and detected)

    # The recorded scale is the row's length before normalization. Training holds every row at
    # unit norm and `unscale_params` then multiplies them all by the one input scale, so every row
    # has the same length and the control here is the normalized vector rather than another row.
    unit = abs(float(jnp.linalg.norm(spec.vector)) - 1.0)
    raw = abs(float(jnp.linalg.norm(jnp.asarray(params.w_dec[feature]))) - 1.0)
    scale_err = abs(spec.scale - want_scale)
    ok = unit < 1e-6 and scale_err < 1e-6
    detected = raw > 1e-6
    print(
        f"  [{'PASS' if ok and detected else 'FAIL'}] {'vector is unit norm':<22}"
        f" |1-||v|||={unit:.2e}, scale={spec.scale:.4f} err={scale_err:.2e}"
        f"  control (raw w_dec row)={raw:.2e}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    failures += not (ok and detected)

    # The inclusive nudge, pinned to one ulp. A token whose projection lands on the conversion's
    # own value has to fire, because the SAE's compare is `>=` and `threshold_fire` is `>`. The
    # bar is `conditional_threshold` with the nudge off, which the threshold row above holds to
    # the float64 reference within four ulps. The reference rounded to float32 can't be the bar:
    # the float32 arithmetic lands an ulp or two either side of it, and where it lands follows
    # the host's BLAS kernels and XLA's code. The control is the float32 one ulp below the bar.
    # It mustn't fire, which caps the nudge at the one ulp the conversion is allowed.
    edge = np.float32(convert.conditional_threshold(params, theta, feature, inclusive=False))
    below = np.nextafter(edge, np.float32(-np.inf))
    got = np.float32(spec.threshold)
    fires = bool(edge > got)
    control_fires = bool(below > got)
    dropped = (float(edge) - float(got)) / ulp(edge)
    ok = fires and not control_fires
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'inclusive nudge':<22}"
        f" {dropped:.2f} ulp under the un-nudged value {float(edge):.9f}, which fires={fires}"
        f"  control (one ulp lower) fires={control_fires}"
    )
    failures += not ok
    return failures, spec


def check_gate(params, theta, spec, feature, x):
    """2. The probe and threshold reproduce the SAE's JumpReLU gate token for token."""
    failures = 0
    want = reference_fired(params.w_enc, params.b_enc, theta, feature, x)

    def fired(probe, threshold, data=x):
        return np.asarray(threshold_fire(jnp.asarray(data), probe, threshold))

    got = fired(spec.probe, spec.threshold)
    proj = np.asarray(projection(jnp.asarray(x), spec.probe), np.float64)
    agree = int((got == want).sum())
    ok = agree == len(want)
    # How hard the gate had to work: the closest token's margin, in float32 ulps. A batch whose
    # nearest token sits thousands of ulps from the bar agrees with any threshold nearby, so
    # the number says whether the boundary was tested or read off easy tokens.
    closest = float(np.abs(proj - float(spec.threshold)).min()) / ulp(spec.threshold)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'gate reproduced':<22}"
        f" {agree}/{len(want)} tokens, fired {int(got.sum())} against {int(want.sum())},"
        f" closest token {closest:,.0f} ulp from the threshold"
    )
    failures += not ok

    # `subtract_pre_bias` is the documented path for parameters that still expect `x - b_dec`
    # at the encoder. It gets a positive reference of its own, because a fold with the wrong
    # sign still differs from the unfolded value and would pass a negative control alone. The
    # un-nudged fold is held to that reference within four ulps, and the nudge is pinned against
    # the un-nudged fold, for the reason check 1 gives.
    folded = convert.from_sae(params, theta, feature, subtract_pre_bias=True)
    folded_bar = convert.from_sae(
        params, theta, feature, subtract_pre_bias=True, inclusive=False
    ).threshold
    want_folded = reference_pre_bias_threshold(params, theta, feature)
    fold_ulps = in_ulps(max_err(folded_bar, want_folded), want_folded)
    want_pre = reference_fired_pre_bias(params, theta, feature, x)
    got_pre = fired(folded.probe, folded.threshold)
    agree_pre = int((got_pre == want_pre).sum())
    pinned = pinned_one_ulp(folded.threshold, folded_bar)
    ok = fold_ulps <= 4 and pinned and agree_pre == len(want_pre)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'pre-encoder bias':<22}"
        f" threshold {float(folded_bar):.9f} against {float(want_folded):.9f}"
        f" ({fold_ulps:.2f} ulp), nudged one ulp under={pinned}, gate {agree_pre}/{len(want_pre)}"
        f" tokens"
    )
    failures += not ok

    # The read is float32 whatever arrives. Both operands at bfloat16 still come back on a
    # float32 grid, which is what makes the probe scale invariant and the fired set stable.
    narrow_in = projection(jnp.asarray(x, jnp.bfloat16), spec.probe.astype(jnp.bfloat16))
    wide_out = jnp.dtype(narrow_in.dtype).name == "float32"
    print(
        f"  [{'PASS' if wide_out else 'FAIL'}] {'the read widens':<22}"
        f" bfloat16 operands give a {jnp.dtype(narrow_in.dtype).name} projection on"
        f" {len(np.unique(np.asarray(narrow_in))):,} distinct values"
    )
    failures += not wide_out

    rng = np.random.default_rng(0)
    permuted = jnp.asarray(np.asarray(theta)[rng.permutation(theta.shape[0])])
    probe_norm = float(jnp.linalg.norm(spec.probe))
    w_enc_col = np.asarray(params.w_enc, np.float64)[:, feature]
    b_dec = np.asarray(params.b_dec, np.float64)
    wrong_fold = (
        float(theta[feature]) - (float(params.b_enc[feature]) + b_dec @ w_enc_col)
    ) / np.linalg.norm(w_enc_col)
    controls = {
        "thresholds permuted": (
            fired(spec.probe, convert.conditional_threshold(params, permuted, feature)),
            want,
        ),
        "encoder bias dropped": (
            fired(spec.probe, jnp.asarray(float(theta[feature]) / probe_norm, jnp.float32)),
            want,
        ),
        "probe norm kept": (
            fired(
                spec.probe,
                jnp.asarray(float(theta[feature]) - float(params.b_enc[feature]), jnp.float32),
            ),
            want,
        ),
        "pre-encoder bias folded twice": (fired(spec.probe, folded.threshold), want),
        "the pre-encoder fold with the wrong sign": (
            fired(folded.probe, jnp.asarray(np.float32(wrong_fold))),
            want_pre,
        ),
        "operands rounded to bfloat16": (
            fired(
                spec.probe.astype(jnp.bfloat16),
                spec.threshold.astype(jnp.bfloat16),
                jnp.asarray(x, jnp.bfloat16),
            ),
            want,
        ),
    }
    for name, (control, against) in controls.items():
        missed = int((control != against).sum())
        detected = missed > 0
        print(
            f"      control ({name}): {missed} token(s) on the wrong side"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        failures += not detected

    # Why the read is float32, on the same tokens and the same probe. A contraction that
    # returns bfloat16 lands the batch on a coarse grid, so tokens tie against the threshold
    # and break arbitrarily, and a token that flips takes the whole `alpha * v` with it.
    narrow = np.asarray(bf16_projection(x, spec.probe), np.float32)
    wide = np.asarray(projection(jnp.asarray(x), spec.probe), np.float32)
    narrow_fired = narrow > np.float32(spec.threshold)
    missed = int((narrow_fired != want).sum())
    detected = missed > 0 and len(np.unique(narrow)) < len(np.unique(wide))
    print(
        f"      control (the contraction at bfloat16): {len(want):,} tokens on"
        f" {len(np.unique(narrow)):,} distinct values against {len(np.unique(wide)):,} at"
        f" float32, {missed} on the wrong side"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    failures += not detected
    return failures


def check_direction(params, spec, truth, feature, second, other):
    """3. The steering vector is the planted atom, and the near misses aren't."""
    atom = np.asarray(truth[PLANTED], np.float64)
    atom /= np.linalg.norm(atom)

    def cosine(vec):
        vec = np.asarray(vec, np.float64)
        return float(vec @ atom / np.linalg.norm(vec))

    got = cosine(spec.vector)
    controls = {
        f"next-best latent {second}": cosine(params.w_dec[second]),
        f"random latent {other}": cosine(params.w_dec[other]),
        "encoder row of the same latent": cosine(spec.probe),
    }
    ok = got > 0.99
    detected = all(abs(v) < 0.95 for v in controls.values())
    print(
        f"  [{'PASS' if ok and detected else 'FAIL'}] {'planted direction':<22}"
        f" cos(v, atom {PLANTED})={got:.4f} from latent {feature}"
    )
    for name, value in controls.items():
        print(f"      control ({name}): cos={value:.4f}")
    if not detected:
        print("      FAIL: a control matched the atom too, so this check separates nothing.")
    return int(not (ok and detected))


def check_detection(spec, x, present, params, other):
    """4. The probe fires on the tokens that carry the planted atom."""
    got = np.asarray(threshold_fire(jnp.asarray(x), spec.probe, spec.threshold))
    precision = float(present[got].mean()) if got.any() else 0.0
    recall = float(got[present].mean())
    ok = precision > 0.7 and recall > 0.85

    control_spec_probe = jnp.asarray(params.w_enc[:, other])
    control = np.asarray(
        threshold_fire(jnp.asarray(x), control_spec_probe, spec.threshold)
    )
    c_precision = float(present[control].mean()) if control.any() else 0.0
    c_recall = float(control[present].mean())
    detected = not (c_precision > 0.7 and c_recall > 0.85)
    print(
        f"  [{'PASS' if ok and detected else 'FAIL'}] {'probe finds the atom':<22}"
        f" precision={precision:.3f} recall={recall:.3f} on {int(got.sum())} fired"
    )
    print(
        f"      control (probe of latent {other}): precision={c_precision:.3f}"
        f" recall={c_recall:.3f} -> {'detected' if detected else 'NOT DETECTED'}"
    )
    return int(not (ok and detected))


def check_steering(spec, dead_spec, x, mesh):
    """5. The spec drives `conditional_steer` inside a stack and lands `alpha * v`."""
    failures = 0
    stack = toy_stack(5, D_MODEL)

    # The parameters came off a one-device mesh, so the spec is committed to that device.
    # Replicate it across the token mesh, which is what a server does with a loaded bank.
    def replicate(spec_in):
        vector, probe, threshold = jax.device_put(
            (spec_in.vector, spec_in.probe, spec_in.threshold), NamedSharding(mesh, P())
        )
        return spec_in._replace(vector=vector, probe=probe, threshold=threshold)

    spec, dead_spec = replicate(spec), replicate(dead_spec)
    h_in = jax.device_put(
        jnp.asarray(x[:STEER_TOKENS], jnp.float32), NamedSharding(mesh, P(AXIS, None))
    )
    alpha = convert.alpha_for_fraction(h_in, 0.25)
    want_alpha = 0.25 * float(np.median(np.linalg.norm(np.asarray(h_in, np.float64), axis=-1)))
    ok = abs(float(alpha) - want_alpha) < 1e-4 * want_alpha
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {'alpha for fraction':<22}"
        f" alpha={float(alpha):.4f} against 0.25 * median token norm = {want_alpha:.4f}"
    )
    failures += not ok

    @jax.jit
    def program(h, v, alpha_in, probe, threshold):
        fired = {}

        def hook(stream):
            out, fire = conditional_steer(stream, v, alpha_in, probe, threshold)
            fired["mask"] = fire
            return out

        _, h_hook = run_forward(stack, h, hook=hook, hook_layer=HOOK_LAYER)
        return h_hook, fired["mask"]

    def steer(spec_in, alpha_in):
        return program(h_in, spec_in.vector, alpha_in, spec_in.probe, spec_in.threshold)

    # The stream right after the hook site, with no intervention. Comparing there reads the shift
    # the hook landed rather than what the rest of the stack did with it.
    _, base = jax.jit(lambda h: run_forward(stack, h, hook=None, hook_layer=HOOK_LAYER))(h_in)
    steered, fire = steer(spec, alpha)
    fire = np.asarray(fire)

    delta = np.asarray(steered, np.float64) - np.asarray(base, np.float64)
    moved = np.abs(delta).max(axis=1) > 0.0
    cos = np.asarray(applied_cosine(base, steered, spec.vector), np.float64)
    length = np.linalg.norm(delta[fire], axis=-1) / float(alpha)
    quiet = float(np.abs(delta[~fire]).max()) if (~fire).any() else 0.0
    landed = (
        bool(np.array_equal(moved, fire))
        and fire.any()
        and cos[fire].min() > 0.999
        and float(np.abs(length - 1.0).max()) < 1e-5
        and quiet == 0.0
    )
    print(
        f"  [{'PASS' if landed else 'FAIL'}] {'spec steers the stack':<22}"
        f" fired {int(fire.sum())}/{STEER_TOKENS}  min_cos={cos[fire].min():.6f}"
        f"  |delta|/alpha within {float(np.abs(length - 1.0).max()):.1e} of 1"
        f"  unfired rows moved by {quiet:.1e}"
    )
    failures += not landed

    zero, _ = steer(spec, jnp.float32(0.0))
    unchanged = float(np.abs(np.asarray(zero, np.float64) - np.asarray(base, np.float64)).max())
    print(
        f"      control (alpha = 0): stream moved by {unchanged:.1e}"
        f" -> {'unchanged, as a no-op must be' if unchanged == 0.0 else 'MOVED'}"
    )
    failures += unchanged != 0.0

    _, dead_fire = steer(dead_spec, alpha)
    dead_fired = int(np.asarray(dead_fire).sum())
    print(
        f"      control (dead latent {dead_spec.feature}, threshold inf):"
        f" fired {dead_fired} -> {'holds off' if dead_fired == 0 else 'FIRED'}"
    )
    failures += dead_fired != 0
    return failures


def check_bank(params, theta, feature, other, dead):
    """6. Round trip through `.npz`, and every malformed input raises."""
    failures = 0
    bank = convert.build_bank(params, theta, [feature, other, dead])
    live_bank = convert.build_bank(params, theta, [feature, other])
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "bank.npz")
        with warnings.catch_warnings(record=True) as dead_warnings:
            warnings.simplefilter("always")
            convert.save_bank(path, bank, meta={"layer": HOOK_LAYER})
        with warnings.catch_warnings(record=True) as live_warnings:
            warnings.simplefilter("always")
            convert.save_bank(os.path.join(workdir, "live.npz"), live_bank)
        loaded, meta = convert.load_bank(path)

    # The library writes a dead row, because a conditional-only caller can use it, but it has to
    # say so: a server steers statically unless a request asks otherwise.
    warned = [str(w.message) for w in dead_warnings]
    warn_ok = len(warned) == 1 and f"[{dead}]" in warned[0]
    print(
        f"  [{'PASS' if warn_ok else 'FAIL'}] {'dead row warns':<22}"
        f" save_bank with dead latent {dead}: {len(warned)} warning(s)"
    )
    failures += not warn_ok
    print(
        f"      control (a bank of live latents): {len(live_warnings)} warning(s)"
        f" -> {'quiet' if not live_warnings else 'WARNED'}"
    )
    if live_warnings:
        print("      FAIL: the control warned too, so this check separates nothing.")
        failures += 1

    # `scales` rides along because it's what `alpha` compares against: the feature writes
    # `scale * activation` into the stream, so a caller sizing alpha reads it. A round trip
    # that drops it loses the units.
    same = (
        np.array_equal(loaded.features, bank.features)
        and max_err(loaded.vectors, bank.vectors) == 0.0
        and max_err(loaded.probes, bank.probes) == 0.0
        and np.array_equal(loaded.thresholds, bank.thresholds)
        and np.array_equal(loaded.scales, bank.scales)
        and (loaded.scales > 0).all()
        and loaded.d_model == D_MODEL
        and len(loaded) == 3
        and meta == {"layer": HOOK_LAYER}
    )
    print(
        f"  [{'PASS' if same else 'FAIL'}] {'bank round trip':<22}"
        f" {len(loaded)} features, d_model={loaded.d_model},"
        f" scales={np.array2string(loaded.scales, precision=4)}, meta={meta}"
    )
    failures += not same

    # The capture slot and the steering flag are one apart, so the bank carries both rather
    # than leaving the arithmetic to whoever writes the launch line.
    layer_ok = (
        convert.steering_layer_for({"capture_layer": 20}) == 19
        and convert.steering_layer_for({"capture_layer": 1}) == 0
        and convert.steering_layer_for({}) is None
    )
    print(
        f"  [{'PASS' if layer_ok else 'FAIL'}] {'capture slot to flag':<22}"
        f" slot 20 -> --steering-layer {convert.steering_layer_for({'capture_layer': 20})}"
    )
    failures += not layer_ok
    refused = 0
    for bad in ({"capture_layer": 0}, {"capture_layer": -1}):
        try:
            convert.steering_layer_for(bad)
        except ValueError:
            refused += 1
    print(
        f"      control (the embedding slot, which no block writes): "
        f"{'refused' if refused == 2 else 'ACCEPTED'} {refused} of 2"
    )
    if refused != 2:
        print("      FAIL: the control didn't fail, so this check detects nothing.")
        failures += 1

    dead_inf = bool(np.isinf(loaded.thresholds[2])) and bool(
        np.isfinite(loaded.thresholds[:2]).all()
    )
    print(
        f"  [{'PASS' if dead_inf else 'FAIL'}] {'dead latent holds off':<22}"
        f" thresholds={np.array2string(loaded.thresholds, precision=4)}"
    )
    failures += not dead_inf

    # Every probe read against a few tokens, for picking which feature is already active.
    tokens = np.arange(4 * D_MODEL, dtype=np.float32).reshape(4, D_MODEL) / D_MODEL
    got = np.asarray(convert.projection_matrix(bank, jnp.asarray(tokens)), np.float64)
    probes = np.asarray(bank.probes, np.float64)
    want = np.stack(
        [
            [row @ (probe / np.linalg.norm(probe)) for probe in probes]
            for row in tokens.astype(np.float64)
        ]
    )
    err = max_err(got, want)
    control = max_err(got, tokens.astype(np.float64) @ probes.T)
    ok, detected = err < 1e-5, control > 1e-5
    print(
        f"  [{'PASS' if ok and detected else 'FAIL'}] {'every probe at once':<22}"
        f" {got.shape} err={err:.2e}  control (probes unnormalized)={control:.2e}"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    failures += not (ok and detected)

    rows = {
        "duplicate feature": (lambda: convert.build_bank(params, theta, [feature, feature]),
                              lambda: convert.build_bank(params, theta, [feature, other])),
        "feature out of range": (lambda: convert.from_sae(params, theta, theta.shape[0]),
                                 lambda: convert.from_sae(params, theta, theta.shape[0] - 1)),
        "feature is a float": (lambda: convert.from_sae(params, theta, 1.0),
                               lambda: convert.from_sae(params, theta, 1)),
        "theta the wrong length": (lambda: convert.from_sae(params, theta[:-1], feature),
                                   lambda: convert.from_sae(params, theta, feature)),
        "empty bank": (lambda: convert.build_bank(params, theta, []),
                       lambda: convert.build_bank(params, theta, [feature])),
        "fraction not positive": (lambda: convert.alpha_for_fraction(jnp.ones((4, 4)), 0.0),
                                  lambda: convert.alpha_for_fraction(jnp.ones((4, 4)), 0.5)),
    }
    for name, (bad, good) in rows.items():
        try:
            bad()
        except (TypeError, ValueError, IndexError) as exc:
            raised = type(exc).__name__
        else:
            raised = None
        try:
            good()
        except Exception as exc:  # noqa: BLE001 - the control mustn't raise at all
            control = f"RAISED {type(exc).__name__}: {exc}"
        else:
            control = "accepted"
        ok = raised is not None and control == "accepted"
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:<22} raises {raised}"
            f"  control (valid input): {control}"
        )
        failures += not ok
    return failures


def check_main(cfg, result, feature, dead):
    """7. The command line writes no bank for a dead latent, and prints the file it does write."""
    failures = 0

    def run(argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                code = convert.main(argv)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue()

    with tempfile.TemporaryDirectory() as workdir:
        sae_path = train_lib.save(
            os.path.join(workdir, "sae.npz"), cfg, result, capture_layer=HOOK_LAYER + 1
        )
        refused_out = os.path.join(workdir, "dead")
        code, _ = run(["--sae", sae_path, "--feature", str(dead), "--out", refused_out])
        written = any(os.path.exists(p) for p in (refused_out, refused_out + ".npz"))
        refused = code not in (0, None) and not written
        print(
            f"  [{'PASS' if refused else 'FAIL'}] {'dead latent refused':<22}"
            f" --feature {dead} (threshold inf): exit {code!r}, bank written: {written}"
        )
        failures += not refused

        # The name lacks `.npz`, which `np.savez` adds. The printed path has to be the file.
        live_out = os.path.join(workdir, "live")
        code, text = run(["--sae", sae_path, "--feature", str(feature), "--out", live_out])
        wrote = [line for line in text.splitlines() if line.startswith("wrote ")]
        printed = wrote[0].split()[1].rstrip(":") if wrote else None
        accepted = code == 0 and printed is not None and os.path.exists(printed)
        loaded = convert.load_bank(printed)[0] if accepted else None
        accepted = accepted and loaded is not None and list(loaded.features) == [feature]
        print(
            f"      control (live --feature {feature}, --out without .npz): exit {code!r},"
            f" printed {printed}, which exists: {printed is not None and os.path.exists(printed)}"
            f" -> {'written' if accepted else 'NOT WRITTEN'}"
        )
        failures += not accepted
    return failures


def main() -> int:
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAILED: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))

    cfg = SAEConfig(d_model=D_MODEL, expansion_factor=EXPANSION, k=K_TRUE + 1)
    train_cfg = TrainConfig(
        steps=4_000,
        batch_size=256,
        learning_rate=3e-3,
        warmup_steps=200,
        calibration_batches=32,
        log_every=2_000,
        seed=0,
    )
    print(f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}")
    print(
        f"d_model={cfg.d_model} d_sae={cfg.d_sae} k={cfg.k} atoms={ATOMS}"
        f" planted=atom {PLANTED} on {PLANT_RATE:.0%} of tokens\n"
    )

    x, truth, present = make_activations(seed=1)
    pool, held = x[:TRAIN_TOKENS], x[TRAIN_TOKENS:]
    held_present = present[TRAIN_TOKENS:]

    result = train(
        cfg,
        train_cfg,
        batch_stream(pool, train_cfg.batch_size, seed=3),
        mesh=Mesh(np.array(devices[:1]), (LATENT_AXIS,)),
        log=lambda line: print(f"    {line}"),
    )
    params, theta = result.params, result.threshold
    print()

    # Which latent learned the planted atom, picked by the generating dictionary and not by
    # anything in from_sae.py.
    rows = np.asarray(params.w_dec, np.float64)
    rows /= np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12)
    atom = np.asarray(truth[PLANTED], np.float64)
    cosines = rows @ (atom / np.linalg.norm(atom))
    order = np.argsort(cosines)[::-1]
    feature, second = int(order[0]), int(order[1])
    live = np.isfinite(np.asarray(theta))
    # A live latent drawn at random, not the lowest-numbered one. The controls in checks 1, 3
    # and 4 report separation against it, and separation from latent 1 every run says less than
    # separation from a latent the run picked.
    pool_of_controls = np.flatnonzero(live & (np.arange(cfg.d_sae) != feature))
    other = int(np.random.default_rng(7).choice(pool_of_controls))
    dead = int(np.argmax(~live))
    print(
        f"planted atom {PLANTED} -> latent {feature} (cos {cosines[feature]:.4f}),"
        f" next {second} ({cosines[second]:.4f}), live control {other} drawn from"
        f" {len(pool_of_controls)} live latents, dead latent {dead}\n"
    )

    print("1. the three conversions")
    failures, spec = check_arithmetic(params, theta, feature, other)

    print("\n2. the JumpReLU gate, reproduced")
    failures += check_gate(params, theta, spec, feature, held)

    print("\n3. the planted direction")
    failures += check_direction(params, spec, truth, feature, second, other)

    print("\n4. the probe as a detector")
    failures += check_detection(spec, held, held_present, params, other)

    print("\n5. steering a stack with the spec")
    failures += check_steering(spec, convert.from_sae(params, theta, dead), held, mesh)

    print("\n6. the bank")
    failures += check_bank(params, theta, feature, other, dead)

    print("\n7. the command line")
    failures += check_main(cfg, result, feature, dead)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
