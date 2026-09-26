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

"""Correctness gate for the sequence-sharded affine scan.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 scan/test_affine_scan.py

Each case compares the sharded path against the recurrence applied chunk by chunk, then runs a
negative control that breaks the cross-device chain. The control must fail; if it passes, this
file fails itself.

Every case runs under `shard_map`'s default `check_vma=True`, twice: once with the initial state
handed to each shard as its own copy, and once replicated, one state every shard shares. The
replicated run is the one where a scan carry starts out varying over no mesh axis at all.

Two more checks pin claims the docs make. `incoming_state` sends ceil(log2 D) + 1 pairs per device
at every mesh size up to 8, counted in its jaxpr. And the 8-device flag lands in `XLA_FLAGS` beside
any flag already there, where `setdefault` drops it.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testing import (  # noqa: E402
    DEVICE_COUNT_FLAG,
    force_host_devices,
    sharded_states,
    too_few_devices,
)

# Before jax starts a backend.
force_host_devices(8)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from affine_scan import incoming_state, sequential_reference  # noqa: E402

try:
    from jax import shard_map
except ImportError:  # jax < 0.6
    from jax.experimental.shard_map import shard_map

AXIS = "ctx"


def make_problem(seed, num_chunks, heads, k, v, decay=0.9):
    """A stable affine recurrence with the shape of a real GDN/Mamba state.

    A is scaled below one so the composition over many chunks stays finite;
    that matches the gated recurrences these models use, where A carries
    exp(g) with g <= 0.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(num_chunks, heads, k, k)).astype(np.float32)
    # Scale each A so its spectral radius sits below `decay`.
    for c in range(num_chunks):
        for h in range(heads):
            norm = np.linalg.norm(a[c, h], ord=2)
            a[c, h] *= decay / max(norm, 1e-6)
    b = rng.normal(size=(num_chunks, heads, k, v)).astype(np.float32) * 0.1
    h0 = rng.normal(size=(heads, k, v)).astype(np.float32) * 0.1
    return jnp.asarray(a), jnp.asarray(b), jnp.asarray(h0)


def sharded(a, b, h0, mesh, break_chain=False, replicated=False):
    """`_testing.sharded_states` on this file's mesh axis.

    `replicated` hands every shard the same initial state with `P()`. The
    default gives each shard its own copy along the mesh axis.
    `break_chain` is the negative control: every shard starts from `h0`.
    """
    return sharded_states(a, b, h0, mesh, AXIS, break_chain=break_chain, replicated=replicated)


def run_sharded(label, *args, **kwargs):
    """`sharded`, with a trace-time error reported instead of raised.

    A carry whose varying axes change across the scan body fails when
    `shard_map` traces it, before anything runs. Report that as one failure
    and keep going, so every case still prints.
    """
    try:
        return sharded(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{label} raised {type(exc).__name__}: {exc}"


def errors(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    max_abs = np.abs(got - want).max()
    denom = max(np.abs(want).max(), 1e-12)
    rel = max_abs / denom
    gf, wf = got.ravel(), want.ravel()
    cos = float(gf @ wf / max(np.linalg.norm(gf) * np.linalg.norm(wf), 1e-12))
    return max_abs, rel, cos


def ppermute_avals(jaxpr):
    """The operand of every `ppermute` in a jaxpr, nested bodies included."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "ppermute":
            found.extend(var.aval for var in eqn.invars)
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found.extend(ppermute_avals(inner))
    return found


def pairs_sent(exchange, devices, heads=2, k=8, v=8):
    """(A, B) pairs one device sends per layer, read off the jaxpr of `exchange`."""
    mesh = Mesh(np.array(jax.devices()[:devices]), (AXIS,))
    a = jnp.ones((devices, heads, k, k), jnp.float32)
    b = jnp.ones((devices, heads, k, v), jnp.float32)
    h = jnp.ones((devices, heads, k, v), jnp.float32)

    def body(a_loc, b_loc, h_loc):
        a_loc, b_loc, h_loc = (jnp.squeeze(t, 0) for t in (a_loc, b_loc, h_loc))
        return jnp.expand_dims(exchange(a_loc, b_loc, h_loc, AXIS), 0)

    fn = shard_map(body, mesh=mesh, in_specs=(P(AXIS),) * 3, out_specs=P(AXIS))
    avals = ppermute_avals(jax.make_jaxpr(fn)(a, b, h).jaxpr)
    sent = sum(math.prod(aval.shape) * aval.dtype.itemsize for aval in avals)
    return sent / ((a[0].size + b[0].size) * a.dtype.itemsize)


def ring_gather(a_local, b_local, h_init, axis_name):
    """The all-gather the docs compare against: D - 1 shifts, each passing one pair on.

    Only its traffic matters here, so it returns a value of the right shape and
    nothing more.
    """
    num_devices = lax.axis_size(axis_name)
    shift = [(src, (src + 1) % num_devices) for src in range(num_devices)]
    a_cur, b_cur = a_local, b_local
    for _ in range(num_devices - 1):
        a_cur = lax.ppermute(a_cur, axis_name, shift)
        b_cur = lax.ppermute(b_cur, axis_name, shift)
    return jnp.matmul(a_cur, h_init, precision=lax.Precision.HIGHEST) + b_cur


def check_traffic(max_devices):
    """`incoming_state` sends ceil(log2 D) + 1 pairs per device, the figure the docs give."""
    failures = 0
    counts = {d: pairs_sent(incoming_state, d) for d in range(1, max_devices + 1)}
    want = {d: math.ceil(math.log2(d)) + 1 for d in counts}
    ok = counts == want
    table = ", ".join(f"D={d}: {counts[d]:g}" for d in counts)
    print(f"  [{'PASS' if ok else 'FAIL'}] pairs each device sends per layer   {table}")
    if not ok:
        print(f"      FAIL: ceil(log2 D) + 1 gives {want}")
        failures += 1

    # NEGATIVE CONTROL: the ring all-gather has to count D - 1 at the widest
    # mesh, apart from the prefix scan, or the count can't tell O(D) from
    # O(log D).
    ring = pairs_sent(ring_gather, max_devices)
    detected = ring == max_devices - 1 and ring != counts[max_devices]
    print(
        f"      control (ring all-gather): {ring:g} pairs at D={max_devices},"
        f" against {counts[max_devices]:g}  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def setdefault_flags(count, environ):
    """What these tests did before `force_host_devices`, kept as the control."""
    environ.setdefault("XLA_FLAGS", f"{DEVICE_COUNT_FLAG}={count}")
    return environ["XLA_FLAGS"]


def check_device_flag():
    """The device count lands in `XLA_FLAGS` beside other flags, and a count already set wins."""
    fast_math = "--xla_cpu_enable_fast_math=false"
    cases = (
        ("XLA_FLAGS unset", {}, f"{DEVICE_COUNT_FLAG}=8"),
        ("another flag set", {"XLA_FLAGS": fast_math}, f"{fast_math} {DEVICE_COUNT_FLAG}=8"),
        ("a count already set", {"XLA_FLAGS": f"{DEVICE_COUNT_FLAG}=4"}, f"{DEVICE_COUNT_FLAG}=4"),
    )

    def misses(set_flags):
        wrong = []
        for label, start, want in cases:
            environ = dict(start)
            got = set_flags(8, environ)
            if got != want or environ.get("XLA_FLAGS") != want:
                wrong.append(f"{label}: {got!r}")
        return wrong

    failures = 0
    wrong = misses(force_host_devices)
    print(
        f"  [{'PASS' if not wrong else 'FAIL'}] device flag, {len(cases)} starting values of"
        f" XLA_FLAGS"
    )
    for line in wrong:
        print(f"      FAIL: {line}")
    failures += 1 if wrong else 0

    # NEGATIVE CONTROL: `setdefault` drops the count once another flag is set.
    wrong = misses(setdefault_flags)
    print(
        f"      control (setdefault): {'; '.join(wrong) or 'no case missed'}"
        f"  -> {'detected' if wrong else 'NOT DETECTED'}"
    )
    if not wrong:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def main():
    devices = jax.devices()
    if len(devices) < 8:
        print(too_few_devices(devices, 8))
        return 1
    mesh = Mesh(np.array(devices[:8]), (AXIS,))
    print(f"mesh: {AXIS}={mesh.shape[AXIS]} on {devices[0].platform}\n")

    cases = [
        # (name, seed, num_chunks, heads, k, v)
        ("small   ", 1, 16, 2, 8, 8),
        ("gdn-like", 2, 64, 4, 128, 128),
        ("deep    ", 3, 256, 2, 64, 64),
    ]

    REL_TOL = 1e-4
    CONTROL_MIN = 1e-3

    failures = 0
    for name, seed, c, h, k, v in cases:
        a, b, h0 = make_problem(seed, c, h, k, v)
        want = sequential_reference(a, b, h0)

        for replicated in (False, True):
            spec = "h0 replicated" if replicated else "h0 per shard "
            got, error = run_sharded(name, a, b, h0, mesh, replicated=replicated)
            if error:
                print(f"  [FAIL] {name}  {spec}  {error}")
                failures += 1
                continue
            max_abs, rel, cos = errors(got, want)
            ok = rel < REL_TOL
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {name}  {spec}  chunks={c:<4} heads={h}"
                f" k={k} v={v}   max_abs={max_abs:.3e}  rel={rel:.3e}  cos={cos:.8f}"
            )
            if not ok:
                failures += 1

            # Negative control on the same problem. With h0 replicated the
            # broken chain hands `replay_local` a carry that varies over no
            # mesh axis.
            ctrl, error = run_sharded(
                name, a, b, h0, mesh, break_chain=True, replicated=replicated
            )
            if error:
                print(f"      [FAIL] control (chain broken)  {error}")
                failures += 1
                continue
            c_abs, c_rel, c_cos = errors(ctrl, want)
            detected = c_rel > CONTROL_MIN
            print(
                f"      control (chain broken): rel={c_rel:.3e} cos={c_cos:.6f}"
                f"  -> {'detected' if detected else 'NOT DETECTED'}"
            )
            if not detected:
                print("      FAIL: the control did not fail, so this test detects nothing.")
                failures += 1

    print()
    failures += check_traffic(len(devices[:8]))
    failures += check_device_flag()

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
