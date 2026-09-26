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

"""Correctness gate for the Inkling layer plan.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 scan/test_inkling_layers.py

The plan checks read the published fields of both sizes: the scan groups have
to land on the boundaries the model card implies, `dense_mlp_idx` has to put
the dense MLPs where the config puts them, and `local_layer_ids` has to decide
the attention cycle. The same fields spelled the way a save through
`transformers` writes them have to plan the same stack, a config that leaves
fields out has to take the defaults `InklingTextConfig` takes, and every alias
in that class's `attribute_map` that names a planned field has to be read. The
convolution check sends one sconv state through `affine_scan.py` over 8 shards
and compares the result against a convolution written out token by token in
this file. A padded sequence has to give the real tokens the same outputs and
hand a decode step the last real rows as its window.

Every check carries a negative control that perturbs an input to the code under
test and requires the output to move. A control must fail; if one passes, this
file fails itself.

Two rules keep the checks from restating the code. Every reference value is a
literal in this file, transcribed from the published `config.json`, never a
field read back off a parsed `InklingConfig`. `check_published_fields` pins the
literals in `inkling_layers.py` against those transcriptions, so the two copies
have to agree. Every reference computation is written a second way here, never
by calling the function under test.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testing import force_host_devices, skip_reason, too_few_devices  # noqa: E402

# Before jax starts a backend.
force_host_devices(8)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from affine_scan import (  # noqa: E402
    compose_local,
    incoming_state,
    replay_local,
)
from inkling_layers import (  # noqa: E402
    DENSE,
    GLOBAL,
    INKLING,
    INKLING_SMALL,
    LOCAL,
    MOE,
    InklingConfig,
    RelativePosition,
    default_local_layer_ids,
    prefix_states,
    sconv_chunk_pairs,
    sconv_incoming_state,
    sconv_initial_state,
)
from mamba2 import causal_conv  # noqa: E402

try:
    from jax import shard_map
except ImportError:  # jax < 0.6
    from jax.experimental.shard_map import shard_map

AXIS = "ctx"
REL_TOL = 1e-6
CONTROL_MIN = 1e-3


# --- the published config, transcribed ----------------------------------
#
# The `text_config` block of each `config.json` on the Hub, field by field, as
# a literal. `check_published_fields` compares the dicts in `inkling_layers.py`
# against these, so a literal that drifts in either copy fails the run. Every
# other check reads its reference values from here.

PUBLISHED_INKLING_TEXT = {
    "model_max_length": 1048576,
    "hidden_size": 6144,
    "num_hidden_layers": 66,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "log_scaling_n_floor": 128000,
    "log_scaling_alpha": 0.1,
    "local_layer_ids": [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
        24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44,
        45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64,
    ],
    "dense_mlp_idx": 2,
    "use_sconv": True,
    "sconv_kernel_size": 4,
    "swa_head_dim": 128,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 16,
    "sliding_window_size": 512,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "dense_intermediate_size": 24576,
    "intermediate_size": 3072,
}

PUBLISHED_INKLING_SMALL_TEXT = {
    "model_max_length": 1048576,
    "hidden_size": 4096,
    "num_hidden_layers": 42,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "log_scaling_n_floor": 128000,
    "log_scaling_alpha": 0.1,
    "local_layer_ids": [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
        24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40,
    ],
    "dense_mlp_idx": 2,
    "use_sconv": True,
    "sconv_kernel_size": 4,
    "swa_head_dim": 128,
    "swa_num_attention_heads": 32,
    "swa_num_key_value_heads": 8,
    "sliding_window_size": 512,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "dense_intermediate_size": 16384,
    "intermediate_size": 2048,
}


# --- what the model card says -------------------------------------------
#
# Every number here is read off the published card, not computed from the
# parsed config. These are the claims `models/inkling.md` makes.


@dataclasses.dataclass(frozen=True)
class Expected:
    """The published claims one model size has to satisfy."""

    dense_layers: tuple[int, ...]
    num_groups: int
    num_kinds: int
    first_groups: tuple[tuple[str, str, int, int], ...]
    num_local: int
    num_global: int
    halo_kib: int
    capture_kib: int
    kv_kib: int
    window_mib: int
    tau_at_256k: float
    tau_at_1m: float
    conv_channels_local: int
    conv_channels_global: int
    conv_state_mib: float
    conv_state_gib_at_1024: float
    # The page states these three for Inkling only.
    hidden_only_mib: float | None = None
    hidden_only_percent: int | None = None
    conv_vs_window_percent: float | None = None
    bias_gib: float | None = None


EXPECTED_INKLING = Expected(
    dense_layers=(0, 1),
    num_groups=23,
    num_kinds=3,
    first_groups=(
        (LOCAL, DENSE, 0, 2),
        (LOCAL, MOE, 2, 3),
        (GLOBAL, MOE, 5, 1),
        (LOCAL, MOE, 6, 5),
        (GLOBAL, MOE, 11, 1),
    ),
    num_local=55,
    num_global=11,
    halo_kib=6204,
    capture_kib=792,
    kv_kib=44,
    window_mib=220,
    tau_at_256k=1.07,
    tau_at_1m=1.21,
    conv_channels_local=16384,
    conv_channels_global=14336,
    conv_state_mib=12.12,
    conv_state_gib_at_1024=12.12,
    hidden_only_mib=4.64,
    hidden_only_percent=38,
    conv_vs_window_percent=5.5,
    bias_gib=1.0,
)

EXPECTED_INKLING_SMALL = Expected(
    dense_layers=(0, 1),
    num_groups=15,
    num_kinds=3,
    first_groups=(
        (LOCAL, DENSE, 0, 2),
        (LOCAL, MOE, 2, 3),
        (GLOBAL, MOE, 5, 1),
        (LOCAL, MOE, 6, 5),
        (GLOBAL, MOE, 11, 1),
    ),
    num_local=35,
    num_global=7,
    halo_kib=2520,
    capture_kib=336,
    kv_kib=28,
    window_mib=70,
    tau_at_256k=1.07,
    tau_at_1m=1.21,
    conv_channels_local=10240,
    conv_channels_global=10240,
    conv_state_mib=4.92,
    conv_state_gib_at_1024=4.92,
)


# --- the plan -----------------------------------------------------------


def covered_layers(groups) -> list[int]:
    """Every layer id the groups hold, in the order they hold it."""
    out: list[int] = []
    for group in groups:
        out.extend(group.indices)
    return out


def partitions_stack(groups, num_layers: int) -> bool:
    """True when the groups hold every layer once, in order."""
    return covered_layers(groups) == list(range(num_layers))


def group_tuples(groups) -> tuple[tuple[str, str, int, int], ...]:
    """Groups as plain tuples, so a literal can stand next to them."""
    return tuple((g.attention, g.mlp, g.start, g.length) for g in groups)


def check_plan(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """Groups land on the boundaries the card implies and cover the stack."""
    groups = cfg.scan_groups()
    specs = cfg.layer_specs()
    failures = 0

    ok = partitions_stack(groups, int(raw["num_hidden_layers"]))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] plan covers {raw['num_hidden_layers']} layers"
        f" once   groups={len(groups)} kinds={len(cfg.group_kinds())}"
    )
    failures += 0 if ok else 1

    counts_ok = (
        len(groups) == want.num_groups and len(cfg.group_kinds()) == want.num_kinds
    )
    head = group_tuples(groups)[: len(want.first_groups)]
    boundaries_ok = head == want.first_groups
    print(
        f"  [{'PASS' if counts_ok and boundaries_ok else 'FAIL'}] {want.num_groups} runs"
        f" over {want.num_kinds} kinds, opening {want.first_groups[0]}"
        f" then {want.first_groups[1]}"
    )
    if not boundaries_ok:
        print(f"      FAIL: groups open {head}, expected {want.first_groups}")
    failures += 0 if counts_ok and boundaries_ok else 1

    kinds_match = all(
        specs[i].kind == group.kind for group in groups for i in group.indices
    )
    maximal = all(
        groups[i].kind != groups[i + 1].kind for i in range(len(groups) - 1)
    )
    print(
        f"  [{'PASS' if kinds_match and maximal else 'FAIL'}] every group holds one kind,"
        f" and neighboring groups differ"
    )
    failures += 0 if kinds_match and maximal else 1

    # NEGATIVE CONTROL: make layer 6 global in the raw config. That splits the
    # fourth run, so the boundaries the card gives no longer hold. This
    # perturbs an input to `scan_groups`, not the list it handed back.
    moved = dict(raw)
    moved["local_layer_ids"] = [i for i in raw["local_layer_ids"] if i != 6]
    shifted = InklingConfig.from_config_dict(moved, cfg.repo)
    detected = group_tuples(shifted.scan_groups())[: len(want.first_groups)] != (
        want.first_groups
    )
    print(
        f"      control (layer 6 made global): runs {len(groups)} ->"
        f" {len(shifted.scan_groups())}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def check_dense_mlp_idx(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """The dense MLPs sit below the raw `dense_mlp_idx` and nowhere else."""
    raw_idx = int(raw["dense_mlp_idx"])
    failures = 0

    parsed_ok = cfg.dense_mlp_idx == raw_idx
    print(
        f"  [{'PASS' if parsed_ok else 'FAIL'}] parser reads dense_mlp_idx={raw_idx}"
        f" as {cfg.dense_mlp_idx}"
    )
    failures += 0 if parsed_ok else 1

    got = cfg.dense_layers
    ok = got == want.dense_layers
    print(
        f"  [{'PASS' if ok else 'FAIL'}] dense MLPs at {got}, card says"
        f" {want.dense_layers}   routed={int(raw['num_hidden_layers']) - len(got)}"
    )
    failures += 0 if ok else 1

    # Widths come from the raw dict, so a parser that swapped the two fails.
    dense_width = int(raw["dense_intermediate_size"])
    expert_width = int(raw["intermediate_size"])
    sizes_split = all(
        cfg.layer_spec(i).intermediate_size
        == (dense_width if i in want.dense_layers else expert_width)
        for i in range(int(raw["num_hidden_layers"]))
    )
    experts_split = all(
        (cfg.layer_spec(i).num_experts == 0)
        == (i in want.dense_layers)
        for i in range(int(raw["num_hidden_layers"]))
    )
    ok = sizes_split and experts_split
    print(
        f"  [{'PASS' if ok else 'FAIL'}] dense width {dense_width} and expert width"
        f" {expert_width} follow that split, and only routed layers carry experts"
    )
    failures += 0 if ok else 1

    # NEGATIVE CONTROL: move the index. A parser that ignores the field hands
    # back the same plan.
    moved = InklingConfig.from_config_dict(
        dict(raw, dense_mlp_idx=raw_idx + 3), cfg.repo
    )
    detected = moved.dense_layers != want.dense_layers
    print(
        f"      control (dense_mlp_idx moved to {raw_idx + 3}):"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def check_local_layer_ids(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """`local_layer_ids` decides the attention cycle, and nothing else does."""
    wanted = set(raw["local_layer_ids"])
    got = set(cfg.local_layers)
    ok = (
        got == wanted
        and len(got) == want.num_local
        and len(cfg.global_layers) == want.num_global
    )
    period = sorted({b - a for a, b in zip(cfg.global_layers, cfg.global_layers[1:])})
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {len(got)} local and {len(cfg.global_layers)}"
        f" global layers, global stride {period}. Card says"
        f" {want.num_local} and {want.num_global}"
    )
    failures = 0 if ok else 1

    swa_kv = int(raw["swa_num_key_value_heads"])
    full_kv = int(raw["num_key_value_heads"])
    window = int(raw["sliding_window_size"])
    widths_split = all(
        cfg.layer_spec(i).num_key_value_heads == swa_kv
        and cfg.layer_spec(i).window == window
        for i in cfg.local_layers
    ) and all(
        cfg.layer_spec(i).num_key_value_heads == full_kv
        and cfg.layer_spec(i).window is None
        for i in cfg.global_layers
    )
    print(
        f"  [{'PASS' if widths_split else 'FAIL'}] local layers carry {swa_kv} KV heads"
        f" and a {window} window, global layers carry {full_kv} and none"
    )
    failures += 0 if widths_split else 1

    # A config that omits the key means the reference cycle
    # `{i for i in range(layers) if (i + 1) % 6}`, which is what the published
    # list writes out. Dropping the key has to reproduce the published plan.
    layers = int(raw["num_hidden_layers"])
    cycle = tuple(i for i in range(layers) if (i + 1) % 6 != 0)
    dropped = dict(raw)
    dropped.pop("local_layer_ids")
    fallback = InklingConfig.from_config_dict(dropped, cfg.repo)
    fallback_ok = (
        cycle == tuple(sorted(wanted))
        and fallback.local_layers == cfg.local_layers
        and default_local_layer_ids(layers) == cycle
    )
    print(
        f"  [{'PASS' if fallback_ok else 'FAIL'}] a config without the key falls back to"
        f" the 5:1 cycle, which is the published list"
    )
    failures += 0 if fallback_ok else 1

    # NEGATIVE CONTROL: an explicit empty list, which means every layer is
    # global. That's a different input from a missing key, and the plan has to
    # follow it.
    flat = InklingConfig.from_config_dict(dict(raw, local_layer_ids=[]), cfg.repo)
    detected = (
        flat.local_layers == ()
        and len(flat.global_layers) == layers
        and len(flat.scan_groups()) < len(cfg.scan_groups())
    )
    print(
        f"      control (local_layer_ids set to []): groups"
        f" {len(cfg.scan_groups())} -> {len(flat.scan_groups())}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def reference_tau(position: int, floor: int, alpha: float) -> float:
    """`tau` written a second way, in plain Python with the natural log."""
    return 1.0 + alpha * alpha_log(position, floor)


def alpha_log(position: int, floor: int) -> float:
    """`log(max(1, (pos + 1) / floor))`, natural log."""
    return math.log(max((position + 1) / floor, 1.0))


def check_positions(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """No rotary table, a bounded relative extent, and the card's tau values."""
    pos = cfg.positional()
    floor = int(raw["log_scaling_n_floor"])
    alpha = float(raw["log_scaling_alpha"])
    extent = int(raw["rel_extent"])
    window = int(raw["sliding_window_size"])
    d_rel = int(raw["d_rel"])
    heads = int(raw["num_attention_heads"])
    failures = 0

    shapes_ok = (
        pos.proj_shape(GLOBAL) == (d_rel, extent)
        and pos.proj_shape(LOCAL) == (d_rel, window)
        and pos.bias_shape(GLOBAL, heads) == (heads, extent)
        and pos.bias_shape(LOCAL, heads) == (heads, window)
    )
    print(
        f"  [{'PASS' if shapes_ok else 'FAIL'}] relative bias in place of RoPE. rel projection"
        f" {pos.proj_shape(GLOBAL)} global, {pos.proj_shape(LOCAL)} local."
        f" Bias row {pos.bias_shape(GLOBAL, heads)}"
    )
    failures += 0 if shapes_ok else 1

    # Compare against the closed form, position by position, not against the
    # shape of the curve.
    probe = [0, 1, floor - 1, floor, floor * 2, cfg.model_max_length - 1]
    tau = np.asarray(pos.tau(jnp.asarray(probe), GLOBAL), dtype=np.float64)
    reference = np.asarray(
        [reference_tau(p, floor, alpha) for p in probe], dtype=np.float64
    )
    max_err = float(np.abs(tau - reference).max())
    matches = max_err < 1e-6
    local_flat = bool(
        np.all(np.asarray(pos.tau(jnp.asarray(probe), LOCAL)) == 1.0)
    )
    # The card quotes two of these to two decimals.
    card_ok = (
        round(float(tau[4]), 2) == want.tau_at_256k
        and round(float(tau[5]), 2) == want.tau_at_1m
    )
    ok = matches and local_flat and card_ok
    print(
        f"  [{'PASS' if ok else 'FAIL'}] tau matches 1 + {alpha} ln(max(1,(pos+1)/{floor}))"
        f" to {max_err:.2e}. {tau[4]:.4f} at {floor * 2}, {tau[5]:.4f} at"
        f" {cfg.model_max_length}, card says {want.tau_at_256k} and {want.tau_at_1m}."
        f" Local layers stay at 1.0"
    )
    failures += 0 if ok else 1

    scaled_ok = pos.scaled_positions() == min(floor, int(raw["model_max_length"]))
    print(
        f"  [{'PASS' if scaled_ok else 'FAIL'}] a sequence runs"
        f" {pos.scaled_positions():,} tokens before the scaling turns on"
    )
    failures += 0 if scaled_ok else 1

    # NEGATIVE CONTROL: pull the floor out. Nothing scales then.
    off = dataclasses.replace(pos, log_scaling_n_floor=None)
    detected = bool(np.all(np.asarray(off.tau(jnp.asarray(probe), GLOBAL)) == 1.0))
    print(
        f"      control (floor removed): -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1

    # NEGATIVE CONTROL: the same curve on a base-10 log, measured against what
    # `tau` returns. A `tau` that took the log base 10 would match this and the
    # control would report nothing. The curve keeps its shape either way, so a
    # check on the shape alone would miss it.
    base_ten = 1.0 + alpha * np.asarray(
        [alpha_log(p, floor) / math.log(10.0) for p in probe]
    )
    detected = float(np.abs(base_ten - tau).max()) > 1e-6
    print(
        f"      control (log base 10): {base_ten[5]:.4f} against tau {tau[5]:.4f}"
        f" at 1M  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


# --- parsing ------------------------------------------------------------


def check_parser_rejects(raw: dict) -> int:
    """Every validation branch raises on the input it's there to reject."""
    layers = int(raw["num_hidden_layers"])
    cases = {
        "num_hidden_layers=0": dict(raw, num_hidden_layers=0),
        "local_layer_ids repeats": dict(raw, local_layer_ids=[0, 1, 1]),
        "local_layer_ids out of range": dict(raw, local_layer_ids=[layers]),
        "dense_mlp_idx out of range": dict(raw, dense_mlp_idx=layers + 1),
        "dense_mlp_idx negative": dict(raw, dense_mlp_idx=-1),
        "log_scaling_n_floor=0": dict(raw, log_scaling_n_floor=0),
        "log_scaling_n_floor negative": dict(raw, log_scaling_n_floor=-5),
    }
    failures = 0
    rejected = []
    for name, bad in cases.items():
        try:
            InklingConfig.from_config_dict(bad, "reject")
        except ValueError:
            rejected.append(name)
        else:
            print(f"      FAIL: the parser accepted {name}")
            failures += 1
    print(
        f"  [{'PASS' if not failures else 'FAIL'}] parser rejects all"
        f" {len(cases)} bad configs"
    )

    # A floor of 0 reaches `tau` through a direct construction too.
    try:
        RelativePosition(16, 1024, 512, 0, 0.1, 1024)
    except ValueError:
        direct_ok = True
    else:
        direct_ok = False
        failures += 1
    print(
        f"  [{'PASS' if direct_ok else 'FAIL'}] RelativePosition rejects a zero floor,"
        f" which would make tau infinite at every position"
    )

    # The published config has to survive the same path. This one breaks
    # nothing, so it's a check rather than a control.
    try:
        InklingConfig.from_config_dict(dict(raw), "accept")
        accepted = True
    except ValueError:
        accepted = False
    print(
        f"  [{'PASS' if accepted else 'FAIL'}] the parser accepts the published config"
        f" those {len(cases)} mutations come from"
    )
    if not accepted:
        failures += 1
    return failures


def check_json_round_trip(cfg: InklingConfig, raw: dict) -> int:
    """`from_json_file` reads a wrapped `text_config` off disk."""
    failures = 0
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"architectures": ["Inkling"], "text_config": dict(raw)}, handle)
        loaded = InklingConfig.from_json_file(path, cfg.repo)
    ok = loaded == cfg
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a wrapped config.json off disk parses to the"
        f" same {len(dataclasses.fields(cfg))} fields"
    )
    failures += 0 if ok else 1

    # NEGATIVE CONTROL: the text block disagrees with clean copies of the same
    # fields in the outer dict. The parser reads the text block, so it has to
    # land on 8 layers. A parser that let the outer keys win would read the
    # published count and miss.
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                dict(
                    raw,
                    architectures=["Inkling"],
                    text_config=dict(raw, num_hidden_layers=8, local_layer_ids=[0, 1, 2]),
                ),
                handle,
            )
        shrunk = InklingConfig.from_json_file(path, cfg.repo)
    detected = shrunk.num_hidden_layers == 8 != cfg.num_hidden_layers
    print(
        f"      control (text block says 8 layers, the outer dict says"
        f" {raw['num_hidden_layers']}): read {shrunk.num_hidden_layers}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def transformers_spelling(raw: dict) -> dict:
    """The text block the way a save through `InklingTextConfig` writes it.

    Written out from `transformers` 5.17: `__post_init__` pops `dense_mlp_idx`
    into `mlp_layer_types` and `dense_intermediate_size` onto
    `intermediate_size`, derives `layer_types` from `local_layer_ids`, and the
    attribute map files `model_max_length` and `sconv_kernel_size` under
    `max_position_embeddings` and `conv_kernel_size`. The expert width goes
    under `moe_intermediate_size`.
    """
    renamed = ("dense_mlp_idx", "dense_intermediate_size", "model_max_length", "sconv_kernel_size")
    out = {k: v for k, v in raw.items() if k not in renamed}
    layers = int(raw["num_hidden_layers"])
    local = set(raw["local_layer_ids"])
    out["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(layers)]
    out["mlp_layer_types"] = [
        "dense" if i < int(raw["dense_mlp_idx"]) else "sparse" for i in range(layers)
    ]
    out["intermediate_size"] = raw["dense_intermediate_size"]
    out["moe_intermediate_size"] = raw["intermediate_size"]
    out["max_position_embeddings"] = raw["model_max_length"]
    out["conv_kernel_size"] = raw["sconv_kernel_size"]
    return out


def saved_by_transformers(raw: dict) -> tuple[dict | None, str | None]:
    """`raw` after a save through the stock `transformers` config class.

    Returns the `text_config` block of the file it writes and None, or None
    and the reason there's no save: `transformers` missing, or a version
    without the Inkling config.
    """
    try:
        from transformers.models.inkling import InklingConfig as ReferenceConfig
    except ImportError as exc:
        return None, skip_reason("inkling", exc)
    with tempfile.TemporaryDirectory() as folder:
        ReferenceConfig(text_config=dict(raw)).save_pretrained(folder)
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
            return json.load(handle)["text_config"], None


def check_transformers_spelling(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """A config saved through `transformers` plans the same stack.

    The save writes six fields the published file doesn't: `layer_types`,
    `mlp_layer_types`, `moe_intermediate_size`, `max_position_embeddings`,
    `conv_kernel_size`, and `intermediate_size` as the dense width. Read with
    the published names alone, the dense MLPs vanish, the expert width turns
    into the dense width and the context comes out as 0.
    """
    failures = 0
    spelled = transformers_spelling(raw)
    try:
        got = InklingConfig.from_config_dict({"text_config": spelled}, cfg.repo)
    except (KeyError, ValueError) as exc:
        print(f"  [FAIL] the transformers spelling raised {type(exc).__name__}: {exc}")
        return 1
    moe = [i for i in range(got.num_hidden_layers) if got.mlp_kind(i) == MOE]
    literal = (
        want.dense_layers,
        want.num_groups,
        want.num_kinds,
        int(raw["dense_intermediate_size"]),
        int(raw["intermediate_size"]),
        min(int(raw["log_scaling_n_floor"]), int(raw["model_max_length"])),
    )
    seen = (
        got.dense_layers,
        len(got.scan_groups()),
        len(got.group_kinds()),
        got.layer_spec(0).intermediate_size,
        got.layer_spec(moe[0]).intermediate_size if moe else None,
        got.positional().scaled_positions(),
    )
    ok = seen == literal and got == cfg
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the transformers spelling plans dense layers"
        f" {seen[0]}, {seen[1]} runs over {seen[2]} kinds, widths {seen[3]} and {seen[4]},"
        f" scaling from {seen[5]:,}, and every field matches the published parse"
    )
    if seen != literal:
        print(f"      FAIL: read {seen}, the published fields say {literal}")
    failures += 0 if ok else 1

    # NEGATIVE CONTROLS: the two spellings of one field disagree, or a list
    # says something the plan can't hold. Each has to be refused.
    layers = int(raw["num_hidden_layers"])
    late_dense = ["sparse"] * layers
    late_dense[0] = late_dense[layers - 1] = "dense"
    bad = {
        "layer_types disagrees with local_layer_ids": dict(
            spelled, layer_types=["hybrid"] * layers
        ),
        "a dense MLP after a routed one": dict(spelled, mlp_layer_types=late_dense),
        "dense_mlp_idx disagrees with mlp_layer_types": dict(spelled, dense_mlp_idx=3),
        "model_max_length disagrees with max_position_embeddings": dict(
            spelled, model_max_length=4096
        ),
        "intermediate_size with no width beside it": {
            k: v
            for k, v in spelled.items()
            if k not in ("moe_intermediate_size", "dense_intermediate_size")
        },
    }
    for label, corrupt in bad.items():
        try:
            InklingConfig.from_config_dict(corrupt, cfg.repo)
        except ValueError as exc:
            print(f"      control ({label}): rejected -> {exc}")
            continue
        print(f"      control ({label}): ACCEPTED")
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1

    saved, missing = saved_by_transformers(raw)
    if missing:
        print(f"  skipped: a save through transformers, because {missing}")
        return failures
    # The stock class keeps its own default for the expert width, 3072, and
    # never reads the published `intermediate_size` into it. That's right for
    # Inkling and wrong for Inkling-Small, so compare it to what the file says
    # rather than to the published width.
    through = InklingConfig.from_config_dict({"text_config": saved}, cfg.repo)
    same = dataclasses.replace(through, intermediate_size=cfg.intermediate_size) == cfg
    expert_ok = through.intermediate_size == int(saved["moe_intermediate_size"])
    ok = same and expert_ok
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a save through transformers plans the same stack,"
        f" and reads the expert width it wrote, {through.intermediate_size}"
    )
    failures += 0 if ok else 1
    return failures


# Defaults of `InklingTextConfig` in `transformers` 5.17, transcribed from the
# class. `REFERENCE_DEFAULTS` in `inkling_layers.py` is the other copy, and this
# is the one the check grades against. Keys are `InklingConfig` field names.
TRANSFORMERS_TEXT_DEFAULTS = {
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 16,
    "swa_head_dim": 128,
    "sliding_window_size": 512,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "d_rel": 16,
    "rel_extent": 1024,
    "log_scaling_alpha": 0.1,
    "log_scaling_n_floor": None,
    "sconv_kernel_size": 4,
    "model_max_length": 131072,
    "dense_intermediate_size": 24576,
    "intermediate_size": 3072,
}

# Where an `InklingConfig` field goes by another name in `InklingTextConfig`.
# The plan keeps the published names, so its expert width is
# `intermediate_size` where the class calls that `moe_intermediate_size`, and
# the class's `intermediate_size` is the dense width.
TRANSFORMERS_NAMES = {
    "sconv_kernel_size": "conv_kernel_size",
    "model_max_length": "max_position_embeddings",
    "dense_intermediate_size": "intermediate_size",
    "intermediate_size": "moe_intermediate_size",
}


def check_reference_defaults() -> int:
    """A config that leaves fields out takes the reference class's defaults.

    The engine port reads every config through `InklingTextConfig`, so a
    missing field takes that class's default in the served model. The plan
    has to size the same model.
    """
    print("defaults")
    failures = 0
    # Required keys only, shaped like Inkling-Small, plus the published MLP
    # split. No width field, so both widths come from the defaults.
    minimal = {
        "hidden_size": 4096,
        "num_hidden_layers": 42,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "dense_mlp_idx": 2,
    }
    cfg = InklingConfig.from_config_dict(minimal, "minimal")
    got = {name: getattr(cfg, name) for name in TRANSFORMERS_TEXT_DEFAULTS}
    wrong = {
        name: (got[name], want)
        for name, want in TRANSFORMERS_TEXT_DEFAULTS.items()
        if got[name] != want
    }
    ok = not wrong
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {len(TRANSFORMERS_TEXT_DEFAULTS)} fields left out"
        f" take the defaults transformers takes"
    )
    for name, (seen, want) in wrong.items():
        print(f"      FAIL: {name} reads {seen}, transformers says {want}")
    failures += 0 if ok else 1

    # Written out from those defaults: a local layer carries 16 KV heads, a
    # 512-token window and k and v convolutions 16 * 128 wide each.
    local = cfg.local_layers[0]
    spec = cfg.layer_spec(local)
    shape = (spec.num_key_value_heads, spec.window, cfg.sconv_channels(local))
    expected = (16, 512, 2 * 4096 + 2 * 16 * 128)
    ok = shape == expected
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a local layer carries {shape[0]} KV heads, a"
        f" {shape[1]} window and {shape[2]:,} conv channels, as the engine builds it"
    )
    failures += 0 if ok else 1

    # The MLPs, from the same defaults: layer 0 runs a dense MLP 24,576 wide,
    # and layer 2 routes to 256 experts of 3,072, 6 of them per token, beside
    # 2 shared ones.
    dense, routed = cfg.layer_spec(0), cfg.layer_spec(2)
    mlp = (
        dense.intermediate_size,
        routed.intermediate_size,
        routed.num_experts,
        routed.experts_per_token,
        routed.shared_experts,
    )
    expected = (24576, 3072, 256, 6, 2)
    ok = mlp == expected and dense.mlp == DENSE and routed.mlp == MOE
    print(
        f"  [{'PASS' if ok else 'FAIL'}] a dense layer runs {mlp[0]:,} channels and a routed"
        f" layer {mlp[2]} experts of {mlp[1]:,}, {mlp[3]} per token and {mlp[4]} shared"
    )
    if mlp != expected:
        print(f"      FAIL: read {mlp}, the defaults say {expected}")
    failures += 0 if ok else 1

    # NEGATIVE CONTROL: write one of those fields out. The plan has to follow
    # the file, which is what makes the defaults above a reading and not a
    # constant.
    written = InklingConfig.from_config_dict(dict(minimal, sliding_window_size=256), "written")
    detected = written.sliding_window_size == 256 != cfg.sliding_window_size
    print(
        f"      control (sliding_window_size written as 256): read"
        f" {written.sliding_window_size}  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1

    try:
        from transformers.models.inkling import InklingTextConfig
    except ImportError as exc:
        print(
            f"  skipped: the defaults against InklingTextConfig, because"
            f" {skip_reason('inkling', exc)}"
        )
        print()
        return failures
    live = InklingTextConfig(**minimal)
    drift = {
        name: (getattr(live, TRANSFORMERS_NAMES.get(name, name)), want)
        for name, want in TRANSFORMERS_TEXT_DEFAULTS.items()
        if getattr(live, TRANSFORMERS_NAMES.get(name, name)) != want
    }
    ok = not drift
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the transcription matches InklingTextConfig in the"
        f" installed transformers"
    )
    for name, (seen, want) in drift.items():
        print(f"      FAIL: {name} is {seen} there, transcribed as {want}")
    failures += 0 if ok else 1
    print()
    return failures


# `attribute_map` of `InklingTextConfig` in `transformers` 5.17, transcribed.
# The class reads the key on the left as the field on the right, and the key
# on the left wins when a file holds both.
TRANSFORMERS_ATTRIBUTE_MAP = {
    "embedding_multiplier": "logits_mup_width_multiplier",
    "sliding_window": "sliding_window_size",
    "num_local_experts": "n_routed_experts",
    "sconv_kernel_size": "conv_kernel_size",
    "model_max_length": "max_position_embeddings",
}

# The `InklingConfig` field each mapped field lands in, and a value to write
# under the alias that's neither the default nor a published value. The plan
# doesn't read `logits_mup_width_multiplier`, so it has no row.
ALIAS_PROBES = {
    "sliding_window_size": ("sliding_window_size", 256),
    "n_routed_experts": ("n_routed_experts", 64),
    "conv_kernel_size": ("sconv_kernel_size", 3),
    "max_position_embeddings": ("model_max_length", 65536),
}


def check_attribute_map(raw: dict) -> int:
    """The plan reads every alias `InklingTextConfig` reads a planned field under.

    The engine port reads a config through that class, so a file that writes
    `sliding_window` or `num_local_experts` sizes the served model by it. Each
    alias goes into a copy of `raw` alone, with a value that's neither the
    default nor the published one, and the plan has to read that value.
    """
    print("attribute map")
    failures = 0
    for alias, target in TRANSFORMERS_ATTRIBUTE_MAP.items():
        if target not in ALIAS_PROBES:
            continue
        field, value = ALIAS_PROBES[target]
        alone = {k: v for k, v in raw.items() if k not in (alias, target)}
        alone[alias] = value
        try:
            got = getattr(InklingConfig.from_config_dict(alone, "alias"), field)
        except (KeyError, ValueError) as exc:
            got = f"{type(exc).__name__}: {exc}"
        ok = got == value
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {alias}={value} with no {target} reads as"
            f" {field}={got}"
        )
        failures += 0 if ok else 1

        # NEGATIVE CONTROL: both names in one file, disagreeing.
        both = dict(alone, **{target: value + 1})
        try:
            InklingConfig.from_config_dict(both, "alias")
        except ValueError as exc:
            print(f"      control ({alias} disagrees with {target}): rejected -> {exc}")
        else:
            print(f"      control ({alias} disagrees with {target}): ACCEPTED")
            print("      FAIL: the control did not fail, so this check detects nothing.")
            failures += 1

    try:
        from transformers.models.inkling import InklingConfig as ReferenceConfig
        from transformers.models.inkling import InklingTextConfig
    except ImportError as exc:
        print(
            f"  skipped: the aliases against InklingTextConfig, because"
            f" {skip_reason('inkling', exc)}"
        )
        print()
        return failures
    live = dict(InklingTextConfig.attribute_map)
    ok = live == TRANSFORMERS_ATTRIBUTE_MAP
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the transcription matches the attribute_map of"
        f" InklingTextConfig in the installed transformers"
    )
    if not ok:
        print(f"      FAIL: the installed class maps {live}")
    failures += 0 if ok else 1

    # Every alias at once, read by the class the engine reads through.
    aliased = {
        k: v for k, v in raw.items() if k not in TRANSFORMERS_ATTRIBUTE_MAP.values()
    }
    for alias, target in TRANSFORMERS_ATTRIBUTE_MAP.items():
        if target in ALIAS_PROBES:
            aliased[alias] = ALIAS_PROBES[target][1]
    served = ReferenceConfig(text_config=dict(aliased)).text_config
    plan = InklingConfig.from_config_dict({"text_config": aliased}, "alias")
    pairs = {
        target: (getattr(served, target), getattr(plan, field))
        for target, (field, _) in ALIAS_PROBES.items()
    }
    ok = all(seen == read for seen, read in pairs.values())
    print(
        f"  [{'PASS' if ok else 'FAIL'}] with every alias written, the plan reads what"
        f" InklingConfig in transformers reads"
    )
    for target, (seen, read) in pairs.items():
        if seen != read:
            print(f"      FAIL: transformers reads {target}={seen}, the plan reads {read}")
    failures += 0 if ok else 1
    print()
    return failures


# --- the short convolution state ----------------------------------------


def errors(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    max_abs = np.abs(got - want).max()
    denom = max(np.abs(want).max(), 1e-12)
    rel = max_abs / denom
    gf, wf = got.ravel(), want.ravel()
    cos = float(gf @ wf / max(np.linalg.norm(gf) * np.linalg.norm(wf), 1e-12))
    return max_abs, rel, cos


def reference_conv(x, weight, bias, prefix):
    """One depthwise causal convolution, token by token, in float64 numpy.

    `mamba2.causal_conv` stacks shifted views and contracts them in one
    einsum. This walks the taps instead, so the two share no code and a bug in
    either one shows up as a difference.
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    prefix = np.asarray(prefix, dtype=np.float64)
    kernel = weight.shape[0]
    padded = np.concatenate([prefix, x], axis=0)
    out = np.empty_like(x)
    for token in range(x.shape[0]):
        total = bias.copy()
        for tap in range(kernel):
            total = total + padded[token + tap] * weight[tap]
        out[token] = total
    return out


def sharded_conv(
    x,
    weight,
    bias,
    kernel,
    chunk_size,
    mesh,
    h0,
    break_chain=False,
    window_from_head=False,
    exchange="collapsed",
    real_tokens=None,
    with_states=False,
):
    """Convolve a sharded sequence, carrying the window across shards.

    `exchange` picks how the window crosses a shard boundary. `collapsed` uses
    `sconv_incoming_state`, one `ppermute`. `affine` uses the general
    `affine_scan.incoming_state` prefix scan. Both have to agree. `shard_map`
    runs at its default `check_vma=True`, the way a caller runs it.
    `real_tokens` goes to `sconv_chunk_pairs` as `tokens`, and `with_states`
    also returns the window after every chunk.
    """
    d = mesh.shape[AXIS]
    tokens, channels = x.shape
    a_chunks, b_chunks = sconv_chunk_pairs(x, kernel, chunk_size, tokens=real_tokens)
    chunks = a_chunks.shape[0]
    assert chunks % d == 0, f"{chunks} chunks does not divide over {d} devices"
    per = chunks // d
    state = kernel - 1

    if window_from_head:
        # NEGATIVE CONTROL: carry the head of each chunk instead of its tail,
        # which is the fold done backwards. This goes through the whole
        # pipeline, so the conv output is what gets compared.
        b_chunks = x.reshape(chunks, chunk_size, channels)[:, :state, :]

    a_s = a_chunks.reshape(d, per, *a_chunks.shape[1:])
    b_s = b_chunks.reshape(d, per, *b_chunks.shape[1:])
    x_s = x.reshape(d, per, chunk_size, channels)
    h0_s = jnp.broadcast_to(h0, (d, *h0.shape))

    def body(a_loc, b_loc, x_loc, h_init):
        a_loc = jnp.squeeze(a_loc, 0)
        b_loc = jnp.squeeze(b_loc, 0)
        x_loc = jnp.squeeze(x_loc, 0)
        h_init = jnp.squeeze(h_init, 0)

        a_tot, b_tot = compose_local(a_loc, b_loc)
        if break_chain:
            # NEGATIVE CONTROL: every shard opens with the sequence's own
            # window, so any shard after the first loses the tokens on its
            # left.
            h_in = h_init
        elif exchange == "affine":
            h_in = incoming_state(a_tot, b_tot, h_init, AXIS)
        else:
            h_in = sconv_incoming_state(b_tot, h_init, AXIS)
        states = replay_local(a_loc, b_loc, h_in)
        prefixes = prefix_states(h_in, states)
        out = jax.vmap(lambda xc, pc: causal_conv(xc, weight, bias, pc))(x_loc, prefixes)
        return jnp.expand_dims(out, 0), jnp.expand_dims(states, 0)

    fn = shard_map(
        body,
        mesh=mesh,
        in_specs=(P(AXIS), P(AXIS), P(AXIS), P(AXIS)),
        out_specs=(P(AXIS), P(AXIS)),
    )
    out, states = fn(a_s, b_s, x_s, h0_s)
    out = out.reshape(tokens, channels)
    if with_states:
        return out, states.reshape(chunks, state, channels)
    return out


def check_sconv(cfg: InklingConfig, raw: dict, mesh, chunk_size=16) -> int:
    """The window crosses shards and matches an independent convolution.

    Runs at the real per-layer sconv width of both attention kinds, and at one
    chunk per shard as well as three. One chunk per shard is the case that
    hands `prefix_states` an empty slice.
    """
    kernel = int(raw["sconv_kernel_size"])
    state = kernel - 1
    d = mesh.shape[AXIS]
    failures = 0

    state_ok = cfg.sconv_state_tokens == state and all(
        cfg.layer_spec(i).sconv_state_tokens == state
        for i in range(cfg.num_hidden_layers)
    )
    print(
        f"  [{'PASS' if state_ok else 'FAIL'}] kernel {kernel} holds {state} tokens of"
        f" state, per layer and for the stack"
    )
    failures += 0 if state_ok else 1

    local_layer = cfg.local_layers[0]
    global_layer = cfg.global_layers[0]
    widths = {
        f"local layer {local_layer}": cfg.sconv_channels(local_layer),
        f"global layer {global_layer}": cfg.sconv_channels(global_layer),
    }

    controls_run = False
    for label, channels in widths.items():
        for per in (1, 3):
            chunks = d * per
            tokens = chunks * chunk_size
            rng = np.random.default_rng(7)
            x = jnp.asarray(rng.normal(size=(tokens, channels)).astype(np.float32))
            weight = jnp.asarray(rng.normal(size=(kernel, channels)).astype(np.float32))
            bias = jnp.asarray(rng.normal(size=(channels,)).astype(np.float32))
            # A nonzero opening window. With zeros here, a shard that inherits
            # the identity map and one that inherits the zero map agree.
            h0 = jnp.asarray(rng.normal(size=(state, channels)).astype(np.float32))

            want = reference_conv(x, weight, bias, h0)
            pairs = sconv_chunk_pairs(x, kernel, chunk_size)
            tail_ok = pairs[1].shape == (chunks, state, channels)

            got = sharded_conv(x, weight, bias, kernel, chunk_size, mesh, h0)
            max_abs, rel, cos = errors(got, want)
            affine = sharded_conv(
                x, weight, bias, kernel, chunk_size, mesh, h0, exchange="affine"
            )
            _, affine_rel, _ = errors(affine, want)
            ok = rel < REL_TOL and affine_rel < REL_TOL and tail_ok
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {label}: {channels:,} channels,"
                f" {tokens} tokens, {per} chunk(s) on each of {d} shards"
                f"   rel={rel:.3e} affine_rel={affine_rel:.3e} cos={cos:.8f}"
            )
            failures += 0 if ok else 1

            if not controls_run:
                controls_run = True
                broken = sharded_conv(
                    x, weight, bias, kernel, chunk_size, mesh, h0, break_chain=True
                )
                _, c_rel, c_cos = errors(broken, want)
                detected = c_rel > CONTROL_MIN
                print(
                    f"      control (chain broken): rel={c_rel:.3e} cos={c_cos:.6f}"
                    f"  -> {'detected' if detected else 'NOT DETECTED'}"
                )
                if not detected:
                    print(
                        "      FAIL: the control did not fail,"
                        " so this check detects nothing."
                    )
                    failures += 1

                heads = sharded_conv(
                    x, weight, bias, kernel, chunk_size, mesh, h0, window_from_head=True
                )
                _, h_rel, h_cos = errors(heads, want)
                detected = h_rel > CONTROL_MIN
                print(
                    f"      control (window from the head of the chunk):"
                    f" rel={h_rel:.3e} cos={h_cos:.6f}"
                    f"  -> {'detected' if detected else 'NOT DETECTED'}"
                )
                if not detected:
                    print(
                        "      FAIL: the control did not fail,"
                        " so this check detects nothing."
                    )
                    failures += 1

    # The dtype of the opening window decides the dtype of the whole path, and
    # a float32 window in a BF16 model doubles the halo.
    small = jnp.zeros((d * chunk_size, 8), jnp.bfloat16)
    a_bf, b_bf = sconv_chunk_pairs(small, kernel, chunk_size)
    h_bf = sconv_initial_state(8, kernel, small.dtype)
    dtype_ok = (
        h_bf.dtype == jnp.bfloat16
        and prefix_states(h_bf, replay_local(a_bf, b_bf, h_bf)).dtype == jnp.bfloat16
    )
    print(
        f"  [{'PASS' if dtype_ok else 'FAIL'}] a BF16 activation keeps the window in"
        f" BF16 through the scan"
    )
    failures += 0 if dtype_ok else 1
    return failures


def check_sconv_padding(raw: dict, mesh, chunk_size=16, channels=8) -> int:
    """A sequence that doesn't fill its chunks, right padded to one chunk per shard.

    The real tokens' outputs have to match the unpadded convolution, and the
    window leaving the last shard has to be the last `kernel - 1` real rows,
    the window a decode step reads. 37 tokens leave 5 real rows in their last
    chunk. 33 leave 1, fewer than the window, so the window reaches back into
    the chunk before.
    """
    print("sconv padding")
    kernel = int(raw["sconv_kernel_size"])
    state = kernel - 1
    stride = chunk_size * mesh.shape[AXIS]
    failures = 0
    for tokens in (37, 33):
        rng = np.random.default_rng(tokens)
        real = rng.normal(size=(tokens, channels)).astype(np.float32)
        weight = jnp.asarray(rng.normal(size=(kernel, channels)).astype(np.float32))
        bias = jnp.asarray(rng.normal(size=(channels,)).astype(np.float32))
        h0 = jnp.asarray(rng.normal(size=(state, channels)).astype(np.float32))
        padded = -(-tokens // stride) * stride
        x = jnp.asarray(np.concatenate([real, np.zeros((padded - tokens, channels), np.float32)]))
        want = reference_conv(real, weight, bias, h0)
        last_rows = real[tokens - state :]

        got, states = sharded_conv(
            x, weight, bias, kernel, chunk_size, mesh, h0, real_tokens=tokens, with_states=True
        )
        _, rel, _ = errors(got[:tokens], want)
        window_ok = bool(np.array_equal(np.asarray(states[-1]), last_rows))
        ok = rel < REL_TOL and window_ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {tokens} tokens padded to {padded}: real outputs"
            f" rel={rel:.3e}, and the window leaving the last shard"
            f" {'is' if window_ok else 'is NOT'} rows {tokens - state}..{tokens - 1}"
        )
        failures += 0 if ok else 1

        # NEGATIVE CONTROL: the same run with the real count left out. The
        # real outputs can't move, and the window leaving the last shard
        # holds padding instead of the last real rows.
        loose, loose_states = sharded_conv(
            x, weight, bias, kernel, chunk_size, mesh, h0, with_states=True
        )
        same = bool(np.array_equal(np.asarray(loose[:tokens]), np.asarray(got[:tokens])))
        detected = not np.array_equal(np.asarray(loose_states[-1]), last_rows)
        print(
            f"      control (real count left out): real outputs"
            f" {'unchanged' if same else 'CHANGED'}, final window"
            f" {'holds padding' if detected else 'still the last real rows'}"
            f"  -> {'detected' if detected else 'NOT DETECTED'}"
        )
        if not same:
            print("      FAIL: the real count moved a real token's output.")
            failures += 1
        if not detected:
            print("      FAIL: the control did not fail, so this check detects nothing.")
            failures += 1

    # A real count shorter than the window would need rows from before the
    # sequence, and one past the padded length names rows that aren't there.
    x = jnp.zeros((stride, channels), jnp.float32)
    bad_counts = (("shorter than the window", state - 1), ("past the padded rows", stride + 1))
    for label, bad in bad_counts:
        try:
            sconv_chunk_pairs(x, kernel, chunk_size, tokens=bad)
        except ValueError as exc:
            print(f"      control (a real count {label}): rejected -> {exc}")
        else:
            print(f"      control (a real count {label}): ACCEPTED")
            print("      FAIL: the control did not fail, so this check detects nothing.")
            failures += 1
    print()
    return failures


# --- sizes --------------------------------------------------------------


def check_sizes(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """The four numbers the layer plan sizes a slice with."""
    window_bytes = sum(
        2 * cfg.layer_spec(i).kv_channels * cfg.sliding_window_size * 2
        for i in cfg.local_layers
    )
    rows = (
        ("sconv halo per shard boundary", cfg.sconv_halo_bytes(), want.halo_kib * 1024, "KiB", 1024),
        ("capture per token", cfg.capture_bytes_per_token(), want.capture_kib * 1024, "KiB", 1024),
        ("KV per token", cfg.kv_bytes_per_token(), want.kv_kib * 1024, "KiB", 1024),
        ("local window per sequence", window_bytes, want.window_mib * 1024**2, "MiB", 1024**2),
    )
    failures = 0
    for label, got, expect, unit, divisor in rows:
        ok = got == expect
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {label}: {got / divisor:,.0f} {unit},"
            f" doc says {expect / divisor:,.0f} {unit}"
        )
        failures += 0 if ok else 1

    # NEGATIVE CONTROL: make every layer global in the raw config. Every layer
    # then grows its cache, so `kv_bytes_per_token` has to jump to the
    # all-layers figure written out here. A function that ignored the attention
    # split, or returned a constant, would miss.
    layers = int(raw["num_hidden_layers"])
    flat = InklingConfig.from_config_dict(dict(raw, local_layer_ids=[]), cfg.repo)
    all_layers = (
        2 * layers * int(raw["num_key_value_heads"]) * int(raw["head_dim"]) * 2
    )
    detected = (
        flat.kv_bytes_per_token() == all_layers
        and all_layers != cfg.kv_bytes_per_token()
    )
    print(
        f"      control (every layer made global): {flat.kv_bytes_per_token() / 1024:,.0f}"
        f" KiB against {cfg.kv_bytes_per_token() / 1024:,.0f}, written out"
        f" {all_layers / 1024:,.0f}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1

    # NEGATIVE CONTROL: shorten the kernel to 3. Each convolution then holds two
    # tokens instead of three, so the halo has to fall to two thirds.
    short = InklingConfig.from_config_dict(dict(raw, sconv_kernel_size=3), cfg.repo)
    two_thirds = cfg.sconv_halo_bytes() // 3 * 2
    detected = (
        short.sconv_halo_bytes() == two_thirds
        and two_thirds != cfg.sconv_halo_bytes()
    )
    print(
        f"      control (kernel 3, two tokens of state):"
        f" {short.sconv_halo_bytes() / 1024:,.0f} KiB against"
        f" {cfg.sconv_halo_bytes() / 1024:,.0f}, written out {two_thirds / 1024:,.0f}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1

    # NEGATIVE CONTROL: turn the convolutions off. The halo has to go to zero.
    off = InklingConfig.from_config_dict(dict(raw, use_sconv=False), cfg.repo)
    detected = off.sconv_halo_bytes() == 0 != cfg.sconv_halo_bytes()
    print(
        f"      control (use_sconv false): {off.sconv_halo_bytes()} bytes against"
        f" {cfg.sconv_halo_bytes() / 1024:,.0f} KiB"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def field_diff(module_raw: dict, published: dict) -> tuple[list, list, list]:
    """Keys the two copies disagree on: missing, extra, and differing values."""
    missing = sorted(set(published) - set(module_raw))
    extra = sorted(set(module_raw) - set(published))
    differing = []
    for key in sorted(set(published) & set(module_raw)):
        want, got = published[key], module_raw[key]
        if isinstance(want, list) or isinstance(got, list):
            same = list(want) == list(got)
        else:
            same = want == got
        if not same:
            differing.append(key)
    return missing, extra, differing


def check_published_fields(module_raw: dict, published: dict, repo: str) -> int:
    """The literals in `inkling_layers.py` match the published `config.json`.

    Every other check reads `published`, which is transcribed in this file. This
    is the one place the two copies meet, so a field that drifts in either one
    fails here rather than comparing itself to itself somewhere else.
    """
    failures = 0
    missing, extra, differing = field_diff(module_raw, published)
    ok = not (missing or extra or differing)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {len(published)} published fields of {repo}"
        f" match the literals the module carries"
    )
    if missing:
        print(f"      FAIL: the module is missing {missing}")
    if extra:
        print(f"      FAIL: the module carries fields the config doesn't: {extra}")
    for key in differing:
        print(f"      FAIL: {key} is {module_raw[key]!r}, config says {published[key]!r}")
    failures += 0 if ok else 1

    # NEGATIVE CONTROL: change one transcribed value. The comparison has to see
    # it, which is what makes the other checks' reference values independent.
    bent = dict(published, rel_extent=int(published["rel_extent"]) + 1)
    detected = field_diff(module_raw, bent)[2] == ["rel_extent"]
    print(
        f"      control (rel_extent moved to {bent['rel_extent']}):"
        f" -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        print("      FAIL: the control did not fail, so this check detects nothing.")
        failures += 1
    return failures


def check_doc_numbers(cfg: InklingConfig, raw: dict, want: Expected) -> int:
    """The conv-state table and the three shares the page quotes beside it."""
    failures = 0
    tokens = int(raw["sconv_kernel_size"]) - 1
    hidden = int(raw["hidden_size"])
    head_dim = int(raw["head_dim"])
    layers = int(raw["num_hidden_layers"])
    local_ids = set(raw["local_layer_ids"])
    local_channels = 2 * hidden + 2 * int(raw["swa_num_key_value_heads"]) * head_dim
    global_channels = 2 * hidden + 2 * int(raw["num_key_value_heads"]) * head_dim

    widths_ok = (
        local_channels == want.conv_channels_local
        and global_channels == want.conv_channels_global
        and cfg.sconv_channels(min(local_ids)) == want.conv_channels_local
        and cfg.sconv_channels(min(set(range(layers)) - local_ids))
        == want.conv_channels_global
    )
    print(
        f"  [{'PASS' if widths_ok else 'FAIL'}] conv channels {want.conv_channels_local:,}"
        f" per local layer and {want.conv_channels_global:,} per global layer"
    )
    failures += 0 if widths_ok else 1

    # Written out a second way: count the two kinds of layer separately.
    num_local = len(local_ids)
    per_sequence = (
        tokens * (num_local * local_channels + (layers - num_local) * global_channels) * 4
    )
    got = cfg.conv_state_bytes_per_sequence()
    state_ok = (
        got == per_sequence
        and round(got / 1024**2, 2) == want.conv_state_mib
        and round(got * 1024 / 1024**3, 2) == want.conv_state_gib_at_1024
    )
    print(
        f"  [{'PASS' if state_ok else 'FAIL'}] conv state {got / 1024**2:,.2f} MiB per"
        f" sequence in float32, {got * 1024 / 1024**3:,.2f} GiB at 1,024 sequences."
        f" Card says {want.conv_state_mib} and {want.conv_state_gib_at_1024}"
    )
    failures += 0 if state_ok else 1

    if want.hidden_only_mib is not None:
        naive = tokens * layers * hidden * 4
        share = round(100 * naive / got)
        window_bytes = sum(
            2 * cfg.layer_spec(i).kv_channels * cfg.sliding_window_size * 2
            for i in cfg.local_layers
        )
        conv_share = round(100 * got / window_bytes, 1)
        bias_bytes = 8192 * int(raw["num_attention_heads"]) * int(raw["rel_extent"]) * 2
        shares_ok = (
            round(naive / 1024**2, 2) == want.hidden_only_mib
            and share == want.hidden_only_percent
            and conv_share == want.conv_vs_window_percent
            and bias_bytes / 1024**3 == want.bias_gib
        )
        print(
            f"  [{'PASS' if shares_ok else 'FAIL'}] one conv per layer at hidden width"
            f" budgets {naive / 1024**2:,.2f} MiB, {share}% of the real figure. The state"
            f" adds {conv_share}% to the {window_bytes / 1024**2:,.0f} MiB window, and one"
            f" layer's bias at 8,192 tokens is {bias_bytes / 1024**3:,.0f} GiB"
        )
        failures += 0 if shares_ok else 1

    # NEGATIVE CONTROL: halve the local KV heads in the raw config. The local
    # layers narrow, so the per-sequence state has to fall by the amount written
    # out here.
    halved = int(raw["swa_num_key_value_heads"]) // 2
    thinner = InklingConfig.from_config_dict(
        dict(raw, swa_num_key_value_heads=halved), cfg.repo
    )
    expect = (
        tokens
        * (
            num_local * (2 * hidden + 2 * halved * head_dim)
            + (layers - num_local) * global_channels
        )
        * 4
    )
    detected = (
        thinner.conv_state_bytes_per_sequence() == expect
        and expect != got
    )
    print(
        f"      control (local KV heads halved to {halved}):"
        f" {thinner.conv_state_bytes_per_sequence() / 1024**2:,.2f} MiB against"
        f" {got / 1024**2:,.2f}, written out {expect / 1024**2:,.2f}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
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

    from inkling_layers import _INKLING_SMALL_TEXT, _INKLING_TEXT

    # `raw` is the transcription in this file, never the module's own dict. The
    # two meet once, in `check_published_fields`.
    cases = (
        (INKLING, PUBLISHED_INKLING_TEXT, _INKLING_TEXT, EXPECTED_INKLING),
        (
            INKLING_SMALL,
            PUBLISHED_INKLING_SMALL_TEXT,
            _INKLING_SMALL_TEXT,
            EXPECTED_INKLING_SMALL,
        ),
    )

    failures = 0
    for cfg, raw, module_raw, want in cases:
        print(f"{cfg.repo}  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")
        failures += check_published_fields(module_raw, raw, cfg.repo)
        failures += check_plan(cfg, raw, want)
        failures += check_dense_mlp_idx(cfg, raw, want)
        failures += check_local_layer_ids(cfg, raw, want)
        failures += check_positions(cfg, raw, want)
        failures += check_parser_rejects(raw)
        failures += check_json_round_trip(cfg, raw)
        failures += check_transformers_spelling(cfg, raw, want)
        failures += check_sconv(cfg, raw, mesh)
        failures += check_sizes(cfg, raw, want)
        failures += check_doc_numbers(cfg, raw, want)
        print()

    failures += check_reference_defaults()
    failures += check_attribute_map(PUBLISHED_INKLING_SMALL_TEXT)
    failures += check_sconv_padding(PUBLISHED_INKLING_TEXT, mesh)

    groups = INKLING.scan_groups()
    shown = min(5, len(groups))
    print(f"scan groups, {INKLING.repo}:")
    for group in groups[:shown]:
        print(
            f"  layers {group.start:>2}..{group.stop - 1:<2}"
            f"  {group.attention:<6} {group.mlp}"
        )
    print(f"  the remaining {len(groups) - shown} repeat the 5 local, 1 global cycle")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
