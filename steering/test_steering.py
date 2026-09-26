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

"""Correctness and compile-cost gate for causal activation steering.

Runs on CPU with a forced 8-device mesh, tokens sharded over it. No TPU needed.

    python3 steering/test_steering.py

Four groups of checks:

1. Static steering shifts the residual stream by `alpha * v` on the masked
   tokens and by nothing anywhere else, against a float64 NumPy forward pass.
   The masked form, the Python form and the reference agree on a repeated
   position and on the -1 padding.
2. Conditional steering fires on the tokens the predicate selects and no
   others, and the `lax.cond` form matches the masked-select form.
3. One executable covers every set of steered positions and every alpha.
4. The same properties hold when the residual stream arrives as BF16.

Every check carries a control. A no-op control (alpha = 0) must leave the
stream bit for bit unchanged. A detection control breaks the thing under test
and must be caught; if a detection control passes, this file fails itself.
"""

from __future__ import annotations

import os
import sys
import time

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steering import (  # noqa: E402
    applied_cosine,
    conditional_steer,
    conditional_steer_cond,
    mask_from_positions,
    projection,
    python_branch_steer,
    run_forward,
    sequential_reference,
    static_steer,
    threshold_fire,
)

AXIS = "tok"
TOKENS = 64
DIM = 128
LAYERS = 4
HOOK_LAYER = 1

# A served residual stream is large next to a unit steering vector. This is the
# per-element RMS the BF16 section scales to.
STREAM_RMS = 300.0

# A prefill batch's worth of tokens. Enough of them that neighboring
# projections sit closer together than a BF16 read error.
PRED_TOKENS = 512


def make_problem(seed=0, tokens=TOKENS, dim=DIM, layers=LAYERS, dtype=jnp.float32):
    """A small stack and one batch of embedded tokens.

    Weights are scaled down so four residual blocks stay in the linear part of
    `tanh` and the stream doesn't saturate.
    """
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(layers):
        w_in = rng.normal(size=(dim, dim)).astype(np.float32) / np.sqrt(dim)
        w_out = rng.normal(size=(dim, dim)).astype(np.float32) / np.sqrt(dim)
        params.append((jnp.asarray(w_in, dtype), jnp.asarray(w_out, dtype)))
    h_in = jnp.asarray(rng.normal(size=(tokens, dim)), dtype)
    v = jnp.asarray(rng.normal(size=(dim,)), dtype)
    probe = jnp.asarray(rng.normal(size=(dim,)), dtype)
    return params, h_in, v, probe


def shard_tokens(x, mesh):
    return jax.device_put(x, NamedSharding(mesh, P(AXIS, None)))


def errors(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    max_abs = np.abs(got - want).max()
    denom = max(np.abs(want).max(), 1e-12)
    return max_abs, max_abs / denom


def unit(x):
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x)


def report(failures, ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return failures + (not ok)


def control(failures, detected, text):
    print(f"      control ({text}) -> {'detected' if detected else 'NOT DETECTED'}")
    if not detected:
        print("      FAIL: the control didn't fail, so this check detects nothing.")
        return failures + 1
    return failures


def static_program(params):
    """Jitted forward with a static-steering hook. Mask and alpha are operands."""

    def fn(h_in, v, alpha, mask):
        hook = lambda h: static_steer(h, v, alpha, mask)  # noqa: E731
        return run_forward(params, h_in, hook=hook, hook_layer=HOOK_LAYER)

    return jax.jit(fn)


def conditional_program(params, variant):
    """Jitted forward with a conditional hook. `variant` picks select or cond."""
    steer = conditional_steer if variant == "select" else conditional_steer_cond

    def fn(h_in, v, alpha, probe, threshold, gate):
        fired = {}

        def hook(h):
            out, fire = steer(h, v, alpha, probe, threshold, gate)
            fired["mask"] = fire
            return out

        h_out, h_hook = run_forward(params, h_in, hook=hook, hook_layer=HOOK_LAYER)
        return h_out, h_hook, fired["mask"]

    return jax.jit(fn)


def check_static(params, h_in, v, mesh):
    """Static steering moves the masked rows by alpha * v and nothing else."""
    print("1. static steering")
    failures = 0

    positions = np.array([0, 5, 17, 40, 63], dtype=np.int32)
    alpha = 2.5
    mask = mask_from_positions(jnp.asarray(positions), TOKENS)

    h_s = shard_tokens(h_in, mesh)
    run = static_program(params)

    got_out, got_hook = run(h_s, v, jnp.float32(alpha), mask)
    want_out, want_hook = sequential_reference(
        params, h_in, v, alpha, positions, hook_layer=HOOK_LAYER
    )

    max_abs, rel = errors(got_out, want_out)
    failures = report(
        failures,
        rel < 1e-5,
        f"forward matches NumPy float64   max_abs={max_abs:.3e}  rel={rel:.3e}",
    )

    # The shift itself, read at the hook site against an unsteered run.
    _, base_hook = run(h_s, v, jnp.float32(0.0), mask)
    delta = np.asarray(got_hook, np.float64) - np.asarray(base_hook, np.float64)
    want_delta = np.zeros_like(delta)
    want_delta[positions] = alpha * np.asarray(v, np.float64)
    shift_err = np.abs(delta - want_delta).max()
    off_mask = np.delete(delta, positions, axis=0)
    failures = report(
        failures,
        shift_err < 1e-4 and np.abs(off_mask).max() == 0.0,
        f"shift equals alpha*v on {len(positions)} masked rows"
        f"   err={shift_err:.3e}  off_mask_max={np.abs(off_mask).max():.3e}",
    )

    # A repeated position is one mask bit, so it steers once. The masked form,
    # the Python form and the NumPy reference all have to agree on that.
    dup = (1, 1, 3)
    dup_mask = mask_from_positions(jnp.asarray(dup, dtype=np.int32), TOKENS)
    dup_masked, _ = run(h_s, v, jnp.float32(2.0), dup_mask)
    dup_python, _ = jax.jit(
        lambda h, v_, a: run_forward(
            params,
            h,
            hook=lambda x: python_branch_steer(x, v_, a, dup),
            hook_layer=HOOK_LAYER,
        )
    )(h_s, v, jnp.float32(2.0))
    dup_ref, _ = sequential_reference(
        params, h_in, v, 2.0, dup, hook_layer=HOOK_LAYER
    )
    _, dup_mask_rel = errors(dup_masked, dup_ref)
    _, dup_py_rel = errors(dup_python, dup_ref)
    failures = report(
        failures,
        dup_mask_rel < 1e-5 and dup_py_rel < 1e-5,
        f"repeated position {dup} steers once in all three forms"
        f"   masked={dup_mask_rel:.3e}  python={dup_py_rel:.3e}",
    )

    # Control, detection: steering the repeat twice is a different answer, so
    # the check above would see a form that accumulated per occurrence.
    twice_ref, _ = sequential_reference(
        params, h_in, v, 2.0, (1, 3), hook_layer=HOOK_LAYER
    )
    twice_ref = twice_ref.copy()
    twice_ref[1] += 2.0 * np.asarray(v, np.float64)
    _, twice_rel = errors(dup_masked, twice_ref)
    failures = control(
        failures, twice_rel > 1e-3, f"repeat added twice: rel={twice_rel:.3e}"
    )

    # The -1 padding holds a positions array at a fixed length and steers
    # nothing. The masked form, the Python form and the NumPy reference all
    # have to skip it rather than wrap it to the last row.
    padded = (1, 3, -1)
    pad_mask = mask_from_positions(jnp.asarray(padded, dtype=np.int32), TOKENS)
    pad_masked, _ = run(h_s, v, jnp.float32(2.0), pad_mask)
    pad_python, _ = jax.jit(
        lambda h, v_, a: run_forward(
            params,
            h,
            hook=lambda x: python_branch_steer(x, v_, a, padded),
            hook_layer=HOOK_LAYER,
        )
    )(h_s, v, jnp.float32(2.0))
    pad_ref, _ = sequential_reference(
        params, h_in, v, 2.0, padded, hook_layer=HOOK_LAYER
    )
    _, pad_mask_rel = errors(pad_masked, pad_ref)
    _, pad_py_rel = errors(pad_python, pad_ref)
    failures = report(
        failures,
        pad_mask_rel < 1e-5 and pad_py_rel < 1e-5,
        f"padding -1 in {padded} steers nothing in all three forms"
        f"   masked={pad_mask_rel:.3e}  python={pad_py_rel:.3e}",
    )

    # Control, detection: -1 read as the last row is a different answer.
    last_ref, _ = sequential_reference(
        params, h_in, v, 2.0, (1, 3, TOKENS - 1), hook_layer=HOOK_LAYER
    )
    _, last_rel = errors(pad_masked, last_ref)
    failures = control(
        failures, last_rel > 1e-3, f"-1 read as row {TOKENS - 1}: rel={last_rel:.3e}"
    )

    # Control, no-op: alpha = 0 must return the unsteered stream bit for bit.
    no_hook_out, no_hook_hook = run(h_s, v, jnp.float32(0.0), mask)
    plain_out, plain_hook = jax.jit(
        lambda h: run_forward(params, h, hook=None, hook_layer=HOOK_LAYER)
    )(h_s)
    zero_gap = max(
        np.abs(np.asarray(no_hook_out) - np.asarray(plain_out)).max(),
        np.abs(np.asarray(no_hook_hook) - np.asarray(plain_hook)).max(),
    )
    failures = report(
        failures, zero_gap == 0.0, f"control, alpha=0 is a no-op   gap={zero_gap:.3e}"
    )

    # The bit-level version of the same claim, which a sign flip on zero can't
    # slip past. Unmasked rows come back byte for byte. Masked rows take the
    # add, so -0.0 becomes +0.0, and the docstring says so.
    signed = jnp.asarray([[-0.0, 1.0, -2.0], [-0.0, 3.0, -4.0]], jnp.float32)
    part = jnp.asarray([True, False])
    out_bits = np.asarray(
        static_steer(signed, jnp.zeros(3, jnp.float32), jnp.float32(0.0), part)
    ).view(np.int32)
    in_bits = np.asarray(signed).view(np.int32)
    unmasked_same = bool(np.array_equal(out_bits[1], in_bits[1]))
    masked_zero_normalized = int(out_bits[0][0]) == 0 and int(in_bits[0][0]) != 0
    masked_rest_same = bool(np.array_equal(out_bits[0][1:], in_bits[0][1:]))
    failures = report(
        failures,
        unmasked_same and masked_zero_normalized and masked_rest_same,
        "alpha=0 keeps unmasked rows byte for byte and normalizes -0.0 on"
        " masked rows",
    )

    # An out-of-range hook site is an error, not a silently unsteered forward.
    raised = []
    for bad in (len(params), -1, LAYERS + 10):
        try:
            run_forward(params, h_in, hook=lambda h: h, hook_layer=bad)
            raised.append(False)
        except ValueError:
            raised.append(True)
        try:
            sequential_reference(params, h_in, v, 1.0, positions, hook_layer=bad)
            raised.append(False)
        except ValueError:
            raised.append(True)
    failures = report(
        failures,
        all(raised),
        f"hook_layer outside 0..{len(params) - 1} raises in both the forward"
        f" and the reference   ({sum(raised)}/{len(raised)})",
    )

    # Control, detection: steer one position over and the comparison must fail.
    wrong = (positions + 1) % TOKENS
    ctrl_out, _ = sequential_reference(
        params, h_in, v, alpha, wrong, hook_layer=HOOK_LAYER
    )
    _, ctrl_rel = errors(got_out, ctrl_out)
    failures = control(
        failures, ctrl_rel > 1e-3, f"mask off by one: rel={ctrl_rel:.3e}"
    )

    return failures


def check_conditional(params, h_in, v, probe, mesh):
    """Conditional steering fires on the predicate's tokens and no others."""
    print("\n2. conditional steering")
    failures = 0

    h_s = shard_tokens(h_in, mesh)
    alpha = 1.75

    # Read the unsteered stream at the hook site, then build the expected fired
    # set in NumPy float64 from the raw activations. This reference calls
    # nothing in the module, so a broken `projection` shows up here.
    _, base_hook = jax.jit(
        lambda h: run_forward(params, h, hook=None, hook_layer=HOOK_LAYER)
    )(h_s)
    base64 = np.asarray(base_hook, np.float64)
    proj_ref = base64 @ unit(probe)
    threshold = float(np.median(proj_ref))
    ungated_fire = proj_ref > threshold

    # Without a gate the predicate stands on its own.
    got_ungated = np.asarray(
        jax.jit(lambda h, p, t: threshold_fire(h, p, t))(
            base_hook, probe, jnp.float32(threshold)
        )
    )
    failures = report(
        failures,
        bool(np.array_equal(got_ungated, ungated_fire)),
        f"ungated predicate fires on the float64 reference's"
        f" {int(ungated_fire.sum())}/{TOKENS} tokens",
    )

    # Control, detection: a predicate that ignored the probe would read some
    # other statistic of the same rows. Row sums split the batch differently,
    # so the check above catches a `projection` that dropped the direction.
    sum_fire = base64.sum(axis=-1) > np.median(base64.sum(axis=-1))
    failures = control(
        failures,
        not np.array_equal(sum_fire, ungated_fire),
        f"probe-free row sums fire on a different"
        f" {int((sum_fire != ungated_fire).sum())} tokens",
    )

    # `projection` normalizes the probe, so the threshold means one thing
    # whatever scale the probe arrives at. Rescale it and the fired set holds.
    fire_at_scale = jax.jit(lambda h, p, t: threshold_fire(h, p, t))
    same_at_every_scale = True
    for scale in (0.1, 10.0, 1000.0):
        scaled = np.asarray(
            fire_at_scale(base_hook, probe * scale, jnp.float32(threshold))
        )
        same_at_every_scale &= bool(np.array_equal(scaled, ungated_fire))
    failures = report(
        failures,
        same_at_every_scale,
        "probe scale doesn't move the fired set   scales=0.1, 10, 1000",
    )

    # A gate holds the intervention off the first eight tokens, the way a
    # serving hook holds it off a system prompt.
    gate_np = np.ones(TOKENS, dtype=bool)
    gate_np[:8] = False
    gate = jnp.asarray(gate_np)
    want_fire = np.logical_and(ungated_fire, gate_np)

    results = {}
    for variant in ("select", "cond"):
        run = conditional_program(params, variant)
        out, hook_h, fire = run(
            h_s, v, jnp.float32(alpha), probe, jnp.float32(threshold), gate
        )
        fire = np.asarray(fire)
        results[variant] = (np.asarray(out, np.float64), fire)

        same_set = bool(np.array_equal(fire, want_fire))
        delta = np.asarray(hook_h, np.float64) - base64
        moved = np.abs(delta).max(axis=1) > 0.0
        want_delta = np.where(
            want_fire[:, None], alpha * np.asarray(v, np.float64), 0.0
        )
        shift_err = np.abs(delta - want_delta).max()
        failures = report(
            failures,
            same_set and bool(np.array_equal(moved, want_fire)) and shift_err < 1e-4,
            f"{variant:<6} fires on {int(want_fire.sum())}/{TOKENS} tokens,"
            f" gated off 8   err={shift_err:.3e}",
        )

    max_abs, rel = errors(results["select"][0], results["cond"][0])
    failures = report(
        failures,
        rel < 1e-6,
        f"masked select and lax.cond agree   max_abs={max_abs:.3e}  rel={rel:.3e}",
    )

    # The two forms agreeing is only interesting if they're different programs.
    # `lax.cond` lowers to a `stablehlo.case` with a second branch. The masked
    # select has no branch, so replacing the cond with a bare call to `apply`
    # would drop the case and fail this.
    def lowered(variant):
        steer = conditional_steer if variant == "select" else conditional_steer_cond

        def fn(h, v_, a, p, t, g):
            hook = lambda x: steer(x, v_, a, p, t, g)[0]  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        args = (h_s, v, jnp.float32(alpha), probe, jnp.float32(threshold), gate)
        return jax.jit(fn).lower(*args).as_text()

    cases = {name: lowered(name).count("stablehlo.case") for name in ("select", "cond")}
    failures = report(
        failures,
        cases["cond"] >= 1 and cases["select"] == 0,
        f"lax.cond lowers a real branch the select doesn't"
        f"   case ops: cond={cases['cond']} select={cases['select']}",
    )

    # Control, no-op: a threshold above every projection fires nothing, and the
    # lax.cond form takes its skip branch.
    high = float(proj_ref.max() + 1.0)
    for variant in ("select", "cond"):
        run = conditional_program(params, variant)
        out, _, fire = run(h_s, v, jnp.float32(alpha), probe, jnp.float32(high), gate)
        plain_out, _ = jax.jit(
            lambda h: run_forward(params, h, hook=None, hook_layer=HOOK_LAYER)
        )(h_s)
        gap = np.abs(np.asarray(out) - np.asarray(plain_out)).max()
        failures = report(
            failures,
            gap == 0.0 and not bool(np.asarray(fire).any()),
            f"control, {variant:<6} predicate never true is a no-op"
            f"   gap={gap:.3e}",
        )

    # Control, no-op: alpha = 0 with the predicate firing must still be a no-op.
    run = conditional_program(params, "select")
    out, _, fire = run(h_s, v, jnp.float32(0.0), probe, jnp.float32(threshold), gate)
    plain_out, _ = jax.jit(
        lambda h: run_forward(params, h, hook=None, hook_layer=HOOK_LAYER)
    )(h_s)
    gap = np.abs(np.asarray(out) - np.asarray(plain_out)).max()
    failures = report(
        failures,
        gap == 0.0 and bool(np.asarray(fire).any()),
        f"control, alpha=0 with a live predicate   gap={gap:.3e}",
    )

    # Control, detection: build the float64 forward that a fired set rolled by
    # one would produce and confirm the module's output is nowhere near it.
    # This runs the reference against the real output, so it fails if the
    # module stops steering the tokens it reports.
    rolled = np.roll(want_fire, 1)
    rolled_out, _ = sequential_reference(
        params, h_in, v, alpha, np.flatnonzero(rolled), hook_layer=HOOK_LAYER
    )
    _, rolled_rel = errors(results["select"][0], rolled_out)
    failures = control(
        failures, rolled_rel > 1e-3, f"fired set rolled by one: rel={rolled_rel:.3e}"
    )

    return failures


def compile_cost(build, args, repeats=3):
    """Mean lower-plus-compile seconds and HLO program size for one mode.

    `build` returns a fresh jitted callable each call, so every repeat pays a
    real compile. Wall clock moves run to run; the HLO line count doesn't, and
    that's the figure the checks below read.
    """
    times = []
    text = ""
    for _ in range(repeats):
        fn = build()
        start = time.perf_counter()
        compiled = fn.lower(*args).compile()
        times.append(time.perf_counter() - start)
        text = compiled.as_text()
    return float(np.mean(times)), len(text.splitlines())


def distinct_programs(lower_request, requests):
    """How many distinct programs a list of requests needs.

    Counts unique lowered HLO text. Lowering keys on shape and dtype, not on
    value, so a decision carried in an operand collapses every request onto one
    module. A decision read on the host lands in the text as a literal, so each
    request gets its own module.

    This measures the program, not the jit cache. A cache count keyed on a
    static argument would report one entry per request whatever the function
    body did, which would prove nothing. Two requests that lower to the same
    text share an executable here even when they arrive by different routes.
    """
    return len({lower_request(request) for request in requests})


PAD_TO = 8


def pad_positions(positions, pad_to=PAD_TO):
    """Positions padded to a fixed length with -1, which steers nothing."""
    positions = list(positions)
    if len(positions) > pad_to:
        raise ValueError(f"{len(positions)} positions won't fit in {pad_to}")
    return np.asarray(positions + [-1] * (pad_to - len(positions)), np.int32)


def check_compile(params, h_in, v, probe, mesh):
    """One program covers every position set and every alpha. The host forms don't."""
    print("\n3. compile cost")
    failures = 0

    h_s = shard_tokens(h_in, mesh)
    alpha = jnp.float32(1.0)
    threshold = jnp.float32(0.0)
    gate = jnp.ones(TOKENS, dtype=bool)

    # Every mode builds the mask where a serving hook would: inside the jit,
    # from a positions operand. Padding to a fixed length is what holds the
    # aval steady across position sets of different length.
    def build_static():
        def fn(h, v_, a, positions):
            mask = mask_from_positions(positions, TOKENS)
            hook = lambda x: static_steer(x, v_, a, mask)  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        return jax.jit(fn)

    def build_cond(variant):
        steer = conditional_steer if variant == "select" else conditional_steer_cond

        def fn(h, v_, a, p, t, g):
            hook = lambda x: steer(x, v_, a, p, t, g)[0]  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        return jax.jit(fn)

    def build_python(positions):
        def fn(h, v_, a):
            hook = lambda x: python_branch_steer(x, v_, a, positions)  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        return jax.jit(fn)

    def build_host_threshold(t):
        def fn(h, v_, a, p, g):
            hook = lambda x: conditional_steer(x, v_, a, p, float(t), g)[0]  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        return jax.jit(fn)

    # Six requests: three sets of steered positions, two strengths. Every mode
    # serves the same six.
    sets = [(1, 2, 3), (7, 8), (0, 31, 62, 63)]
    alphas = [jnp.float32(1.0), jnp.float32(2.0)]
    requests = [(pos, a) for pos in sets for a in alphas]

    # The conditional modes take no positions, so their six requests vary the
    # strength and the threshold instead.
    thresholds = [-1.0, 0.0, 1.0]
    cond_requests = [(t, a) for t in thresholds for a in alphas]

    def lower_static(request):
        pos, a = request
        p = jnp.asarray(pad_positions(pos))
        return build_static().lower(h_s, v, a, p).as_text()

    def lower_static_unpadded(request):
        pos, a = request
        p = jnp.asarray(np.asarray(pos, np.int32))
        return build_static().lower(h_s, v, a, p).as_text()

    def lower_cond(variant):
        def inner(request):
            t, a = request
            tt = jnp.float32(t)
            return build_cond(variant).lower(h_s, v, a, probe, tt, gate).as_text()

        return inner

    def lower_host_threshold(request):
        t, a = request
        return build_host_threshold(t).lower(h_s, v, a, probe, gate).as_text()

    def lower_python(request):
        pos, a = request
        return build_python(pos).lower(h_s, v, a).as_text()

    static_args = (h_s, v, alpha, jnp.asarray(pad_positions(sets[0])))
    cond_args = (h_s, v, alpha, probe, threshold, gate)
    py_args = (h_s, v, alpha)

    rows = [
        ("static mask", *compile_cost(build_static, static_args)),
        ("conditional select", *compile_cost(lambda: build_cond("select"), cond_args)),
        ("conditional lax.cond", *compile_cost(lambda: build_cond("cond"), cond_args)),
        ("python branch", *compile_cost(lambda: build_python(sets[0]), py_args)),
    ]

    device_side = ["static mask", "conditional select", "conditional lax.cond"]
    counts = {
        "static mask": distinct_programs(lower_static, requests),
        "conditional select": distinct_programs(lower_cond("select"), cond_requests),
        "conditional lax.cond": distinct_programs(lower_cond("cond"), cond_requests),
        "python branch": distinct_programs(lower_python, requests),
    }
    host_counts = {
        "static mask": distinct_programs(lower_static_unpadded, requests),
        "conditional select": distinct_programs(lower_host_threshold, cond_requests),
        "conditional lax.cond": distinct_programs(lower_host_threshold, cond_requests),
        "python branch": counts["python branch"],
    }

    print(f"  {'mode':<22}{'compile s':>11}{'hlo lines':>11}{'programs':>10}{'on host':>9}")
    for name, seconds, lines in rows:
        print(
            f"  {name:<22}{seconds:>11.3f}{lines:>11}"
            f"{counts[name]:>10}{host_counts[name]:>9}"
        )
    print("  compile s is wall clock and moves run to run. The other columns don't.")
    print(
        "  programs: the decision carried on device. on host: the same six"
        " requests with\n  the decision read at trace time."
    )

    lines_by_mode = dict((r[0], r[2]) for r in rows)
    extra = lines_by_mode["conditional lax.cond"] - lines_by_mode["conditional select"]
    failures = report(
        failures,
        extra > 0,
        f"lax.cond carries {extra:+d} HLO lines over the masked select for its"
        f" second branch",
    )

    failures = report(
        failures,
        all(counts[name] == 1 for name in device_side),
        f"one program per mode over {len(requests)} requests"
        f"   static={counts['static mask']}"
        f" select={counts['conditional select']}"
        f" cond={counts['conditional lax.cond']}",
    )

    # The three modes each serve one program. They aren't the same program.
    sizes = {name: lines_by_mode[name] for name in device_side}
    failures = report(
        failures,
        len(set(sizes.values())) == len(sizes),
        "the three modes are three distinct programs   "
        + "  ".join(f"{k.split()[-1]}={n}" for k, n in sizes.items()),
    )

    # Control, detection: move each decision to the host and the count has to
    # climb. Unpadded positions change the aval per set, a Python threshold
    # lands in the HLO as a literal, and the Python branch bakes in the
    # positions. If any of these still lowers to one program, the counts above
    # aren't measuring where the decision lives.
    for name, want in (
        ("static mask", len(sets)),
        ("conditional select", len(thresholds)),
        ("python branch", len(sets)),
    ):
        got = host_counts[name]
        failures = control(
            failures,
            got == want,
            f"{name} decided on the host: {got} programs, expected {want}",
        )

    return failures


def threshold_grid(ref_proj, min_gap=1e-3):
    """Every threshold that splits a batch differently.

    The midpoint between each pair of adjacent projections. Pairs closer
    together than `min_gap` are dropped, because a float32 threshold can't
    name a point between them.
    """
    order = np.sort(ref_proj)
    gaps = order[1:] - order[:-1]
    return ((order[:-1] + order[1:]) / 2.0)[gaps > min_gap]


def disagreements(fired, ref_proj, mids):
    """How many of those thresholds produce the wrong fired set."""
    want = ref_proj[None, :] > mids[:, None]
    return int((np.asarray(fired) != want).any(axis=1).sum())


def sweep_disagreements(got_proj, ref_proj):
    """The same count for a read, without going through the predicate."""
    mids = threshold_grid(ref_proj)
    return disagreements(got_proj[None, :] > mids[:, None], ref_proj, mids), len(mids)


def check_bf16(mesh):
    """The BF16 stream the engine actually serves."""
    print("\n4. bf16 residual stream")
    failures = 0

    params, h_in, v, probe = make_problem(seed=3, dtype=jnp.bfloat16)
    # Scale to a served stream's magnitude. This is where a BF16 add loses the
    # steering vector: alpha * v sits below half an ulp of the stream.
    h_in = jnp.asarray(np.asarray(h_in, np.float32) * STREAM_RMS, jnp.bfloat16)
    v = jnp.asarray(unit(v), jnp.bfloat16)
    v_norm = float(np.linalg.norm(np.asarray(v, np.float64)))
    h_s = shard_tokens(h_in, mesh)

    positions = np.array([0, 5, 17, 40, 63], dtype=np.int32)
    alpha = 1.0
    mask = mask_from_positions(jnp.asarray(positions), TOKENS)
    run = static_program(params)

    # Read the base at the hook site from the same program, at alpha = 0. Two
    # BF16 programs over the same weights land an ulp apart at the hook, so a
    # shift read across programs measures the compiler, not the steering.
    _, base_hook = run(h_s, v, jnp.float32(0.0), mask)
    _, steered_hook = run(h_s, v, jnp.float32(alpha), mask)
    base64 = np.asarray(base_hook, np.float64)

    # A bf16 significand is 8 bits, so the gap between neighboring values at
    # the stream's magnitude swallows a unit steering vector whole.
    ulp = float(2.0 ** (np.floor(np.log2(np.abs(base64).max())) - 7))
    print(
        f"  stream rms={base64.std():.1f}  |alpha*v|={alpha * v_norm:.2f}"
        f"  bf16 ulp at the top of the stream={ulp:.1f}"
        f"  hook dtype={steered_hook.dtype}"
    )

    cos = np.asarray(applied_cosine(base_hook, steered_hook, v), np.float64)
    delta = np.asarray(steered_hook, np.float64) - base64
    ratio = np.linalg.norm(delta[positions], axis=-1) / (alpha * v_norm)
    failures = report(
        failures,
        cos[positions].min() > 0.999 and np.abs(ratio - 1.0).max() < 1e-3,
        f"the shift that lands is v   min_cos={cos[positions].min():.4f}"
        f"  norm ratio={ratio.min():.4f}..{ratio.max():.4f}",
    )

    off = np.delete(delta, positions, axis=0)
    failures = report(
        failures,
        np.abs(off).max() == 0.0,
        f"unmasked rows don't move at bf16   off_mask_max={np.abs(off).max():.3e}",
    )

    # Control, detection: round the hook's output back to BF16, the way a hook
    # that kept `h.dtype` would, and the same steering vector disappears.
    def narrow_program():
        def fn(h, v_, a, m):
            hook = lambda x: static_steer(x, v_, a, m).astype(jnp.bfloat16)  # noqa: E731
            return run_forward(params, h, hook=hook, hook_layer=HOOK_LAYER)

        return jax.jit(fn)

    narrow = narrow_program()
    _, narrow_base = narrow(h_s, v, jnp.float32(0.0), mask)
    _, narrow_steered = narrow(h_s, v, jnp.float32(alpha), mask)
    narrow_delta = np.asarray(narrow_steered, np.float64) - np.asarray(
        narrow_base, np.float64
    )
    narrow_cos = np.asarray(applied_cosine(narrow_base, narrow_steered, v), np.float64)
    narrow_ratio = np.linalg.norm(narrow_delta[positions], axis=-1) / (alpha * v_norm)
    dropped = 100.0 * float(np.mean(narrow_delta[positions] == 0.0))
    failures = control(
        failures,
        narrow_cos[positions].min() < 0.9 or narrow_ratio.min() < 0.9,
        f"hook rounded back to bf16: mean cos={narrow_cos[positions].mean():.3f},"
        f" norm ratio {narrow_ratio.min():.3f}..{narrow_ratio.max():.3f},"
        f" {dropped:.0f}% of the delta's components dropped",
    )

    # The predicate at bf16, on a prefill-sized batch. 512 tokens crowd the
    # projection axis, so a rounding-scale read error moves a token across the
    # threshold. Sweeping every threshold that splits the batch differently
    # makes that countable instead of luck.
    rng = np.random.default_rng(11)
    stream = jnp.asarray(
        rng.normal(size=(PRED_TOKENS, DIM)) * STREAM_RMS, jnp.bfloat16
    )
    probe_f32 = jnp.asarray(probe, jnp.float32)
    proj_ref = np.asarray(stream, np.float64) @ unit(probe)
    wide_proj = np.asarray(jax.jit(projection)(stream, probe), np.float64)
    mids = threshold_grid(proj_ref)
    sweep = jax.jit(
        jax.vmap(lambda t: threshold_fire(stream, probe, t))
    )(jnp.asarray(mids, jnp.float32))
    wide_bad = disagreements(sweep, proj_ref, mids)
    failures = report(
        failures,
        wide_bad == 0,
        f"the predicate matches float64 at all {len(mids)} thresholds that"
        f" split the batch   max proj error={np.abs(wide_proj - proj_ref).max():.2e}",
    )

    # One of those thresholds, at the median, with the gate off.
    threshold = float(np.median(proj_ref))
    want_fire = proj_ref > threshold
    fire_fn = jax.jit(lambda h, p, t: threshold_fire(h, p, t))
    got = np.asarray(fire_fn(stream, probe, jnp.float32(threshold)))
    flips = int((got != want_fire).sum())
    failures = report(
        failures,
        flips == 0,
        f"bf16 predicate matches the float64 fired set at the median"
        f"   flips={flips}/{PRED_TOKENS}",
    )

    # Control, detection: read in BF16, the way a predicate that stayed in
    # `h.dtype` would, and the fired set drifts off float64 on a large share of
    # those thresholds.
    def narrow_projection(h, p):
        p = p.astype(jnp.bfloat16)
        return h.astype(jnp.bfloat16) @ (p / jnp.linalg.norm(p))

    narrow_proj = np.asarray(jax.jit(narrow_projection)(stream, probe_f32), np.float64)
    narrow_bad = disagreements(narrow_proj[None, :] > mids[:, None], proj_ref, mids)
    def narrow_compare(t):
        return projection(stream, probe).astype(jnp.bfloat16) > jnp.asarray(
            t, jnp.bfloat16
        )

    down = jax.jit(jax.vmap(narrow_compare))(jnp.asarray(mids, jnp.float32))
    down_bad = disagreements(down, proj_ref, mids)
    failures = control(
        failures,
        narrow_bad > 0 and down_bad > 0,
        f"read in bf16: wrong fired set at {narrow_bad}/{len(mids)} thresholds;"
        f" compared in bf16: {down_bad}/{len(mids)}."
        f" Max proj error {np.abs(narrow_proj - proj_ref).max():.2e} against a"
        f" projection sd of {proj_ref.std():.1f}",
    )

    # Probe scale invariance. The probe arrives at whatever scale the trainer
    # left it at, and `projection` normalizes in float32, so the fired set holds.
    scale_ok = True
    scale_gap = 0.0
    for scale in (0.1, 10.0, 1000.0):
        scaled_probe = jnp.asarray(np.asarray(probe_f32) * scale)
        scaled = np.asarray(fire_fn(stream, scaled_probe, jnp.float32(threshold)))
        scale_ok &= bool(np.array_equal(scaled, want_fire))
        scale_gap = max(
            scale_gap,
            np.abs(
                np.asarray(jax.jit(projection)(stream, scaled_probe), np.float64)
                - wide_proj
            ).max(),
        )
    failures = report(
        failures,
        scale_ok,
        f"bf16 probe scale doesn't move the fired set   proj_gap={scale_gap:.2e}",
    )

    # Control, detection: normalize the probe in bf16 and the rescale itself
    # rounds, so the read moves and the fired set with it.
    narrow_bad_scale = 0
    narrow_gap = 0.0
    for scale in (0.1, 10.0):
        b = np.asarray(
            jax.jit(narrow_projection)(
                stream, jnp.asarray(np.asarray(probe_f32) * scale)
            ),
            np.float64,
        )
        narrow_gap = max(narrow_gap, np.abs(b - narrow_proj).max())
        moved_thresholds, _ = sweep_disagreements(b, narrow_proj)
        narrow_bad_scale += moved_thresholds
    failures = control(
        failures,
        narrow_gap > 0.0 and narrow_bad_scale > 0,
        f"probe normalized in bf16: rescaling moves the read by {narrow_gap:.2e}"
        f" and the fired set with it at {narrow_bad_scale} thresholds",
    )

    # The conditional path end to end at bf16. XLA fuses the widening cast into
    # the residual add before it, so the rows the predicate reads inside the
    # program are a half-ulp off the BF16 tensor the same program hands back.
    # The fired set is therefore checked against float64 only outside a band
    # that wide, and against the rows that actually moved everywhere.
    band = ulp
    gate_np = np.ones(TOKENS, dtype=bool)
    gate_np[:8] = False
    gate = jnp.asarray(gate_np)
    for variant in ("select", "cond"):
        prog = conditional_program(params, variant)
        _, cond_base, _ = prog(
            h_s, v, jnp.float32(0.0), probe, jnp.float32(threshold), gate
        )
        _, hook_h, fire = prog(
            h_s, v, jnp.float32(2.0), probe, jnp.float32(threshold), gate
        )
        fire = np.asarray(fire)
        base_proj = np.asarray(cond_base, np.float64) @ unit(probe)
        want_gated = np.logical_and(base_proj > threshold, gate_np)
        clear = np.abs(base_proj - threshold) > band

        d = np.asarray(hook_h, np.float64) - np.asarray(cond_base, np.float64)
        moved = np.abs(d).max(axis=1) > 0.0
        cos = np.asarray(applied_cosine(cond_base, hook_h, v), np.float64)
        ratio = np.linalg.norm(d[fire], axis=-1) / (2.0 * v_norm)
        quiet = np.abs(d[~fire]).max()
        failures = report(
            failures,
            bool(np.array_equal(moved, fire))
            and bool(np.array_equal(fire[clear], want_gated[clear]))
            and not bool(fire[:8].any())
            and cos[fire].min() > 0.999
            and np.abs(ratio - 1.0).max() < 1e-3
            and quiet == 0.0,
            f"{variant:<6} at bf16 fires on {int(fire.sum())}/{TOKENS}, gated"
            f" off 8, and lands v   min_cos={cos[fire].min():.4f}"
            f"  within {band:.0f} of the threshold: {int((~clear).sum())}",
        )

    return failures


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(f"FAIL: need 8 simulated devices, got {len(devices)}")
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))
    print(
        f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}"
        f"   tokens={TOKENS} dim={DIM} layers={LAYERS} hook_layer={HOOK_LAYER}\n"
    )

    params, h_in, v, probe = make_problem()

    failures = 0
    failures += check_static(params, h_in, v, mesh)
    failures += check_conditional(params, h_in, v, probe, mesh)
    failures += check_compile(params, h_in, v, probe, mesh)
    failures += check_bf16(mesh)

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
