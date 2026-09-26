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

"""Helpers the scan tests share. None of this is part of the scan API.

`force_host_devices` has to run before jax starts a backend, so this module
imports jax only inside the functions that need it.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import MutableMapping

DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"


def force_host_devices(count: int, environ: MutableMapping[str, str] = os.environ) -> str:
    """Add the CPU device-count flag to `XLA_FLAGS` and keep every flag already there.

    `os.environ.setdefault` drops the count whenever `XLA_FLAGS` holds any
    other flag, and the test then stops at one device. This appends the count
    instead. A count the caller already set wins, so the error names their
    number rather than overriding it. The flag moves only the CPU backend, so a
    TPU host still hands jax its chips.

    Returns:
      The `XLA_FLAGS` value jax reads.
    """
    flags = environ.get("XLA_FLAGS", "")
    if DEVICE_COUNT_FLAG not in flags:
        flags = f"{flags} {DEVICE_COUNT_FLAG}={count}".strip()
        environ["XLA_FLAGS"] = flags
    return flags


def too_few_devices(devices, count: int) -> str:
    """The failure line for a run that got fewer devices than the mesh needs."""
    platform = devices[0].platform if devices else "no"
    return (
        f"FAIL: need {count} devices, got {len(devices)} {platform} device(s)."
        f" XLA_FLAGS is {os.environ.get('XLA_FLAGS', '')!r}. On a host with fewer"
        f" chips, run with JAX_PLATFORMS=cpu to get {count} CPU devices."
    )


def missing_reference(module: str, needs_torch: bool = False) -> str | None:
    """Why `transformers.models.<module>` can't serve as a reference here.

    Returns None when it can. Otherwise the reason names what's missing: the
    package, the model in the version that's installed, or torch.
    """
    if importlib.util.find_spec("transformers") is None:
        return "transformers isn't installed"
    import transformers

    if importlib.util.find_spec(f"transformers.models.{module}") is None:
        return f"transformers {transformers.__version__} has no {module} model"
    if needs_torch and importlib.util.find_spec("torch") is None:
        return "torch isn't installed"
    return None


def skip_reason(module: str, error: Exception, needs_torch: bool = False) -> str:
    """The reason to print when importing a reference class raised `error`."""
    return missing_reference(module, needs_torch) or (
        f"importing it raised {type(error).__name__}: {error}"
    )


def sharded_states(a, b, h0, mesh, axis, break_chain=False, replicated=False):
    """Run the three-step sharded path and return the state after every chunk.

    `replicated` hands every shard one initial state with `P()`. The default
    gives each shard its own copy along `axis`. `break_chain` is the negative
    control the fold tests share: every shard starts from `h0`, so any shard
    after the first inherits nothing. `shard_map` runs at its default
    `check_vma=True`, the way a caller runs it.
    """
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from affine_scan import compose_local, incoming_state, replay_local

    try:
        from jax import shard_map
    except ImportError:  # jax < 0.6
        from jax.experimental.shard_map import shard_map

    devices = mesh.shape[axis]
    chunks = a.shape[0]
    assert chunks % devices == 0, f"{chunks} chunks don't divide over {devices} devices"
    per = chunks // devices
    a_s = a.reshape(devices, per, *a.shape[1:])
    b_s = b.reshape(devices, per, *b.shape[1:])
    h0 = h0.astype(a.dtype)

    def body(a_loc, b_loc, h_init):
        a_loc = jnp.squeeze(a_loc, 0)
        b_loc = jnp.squeeze(b_loc, 0)
        if not replicated:
            h_init = jnp.squeeze(h_init, 0)
        a_tot, b_tot = compose_local(a_loc, b_loc)
        if break_chain:
            h_in = h_init
        else:
            h_in = incoming_state(a_tot, b_tot, h_init, axis)
        return jnp.expand_dims(replay_local(a_loc, b_loc, h_in), 0)

    if replicated:
        h0_in, h0_spec = h0, P()
    else:
        h0_in, h0_spec = jnp.broadcast_to(h0, (devices, *h0.shape)), P(axis)
    fn = shard_map(body, mesh=mesh, in_specs=(P(axis), P(axis), h0_spec), out_specs=P(axis))
    out = fn(a_s, b_s, h0_in)
    return out.reshape(chunks, *out.shape[2:])
