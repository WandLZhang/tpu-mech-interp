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

"""Layer plan for the Nemotron 3 hybrid stack.

Nemotron 3 doesn't repeat one layer shape, so an engine can't wrap the whole
stack in a single `lax.scan` over layers. Each layer is a Mamba-2 mixer, an
attention mixer, or a feed-forward, and the types interleave on a cycle that
changes length part way down the stack. The MoE sizes spell the feed-forward
`moe`; Nano spells it `mlp`.

Two spellings of the same field exist. Super and Nano ship
`hybrid_override_pattern`, a string over `M`, `E`, `*` and `-`. Ultra ships
`layers_block_type`, a list over `"mamba"`, `"moe"`, `"attention"` and
`"mlp"`. The reference config class treats them as one field and derives the
layer count from it, so `num_hidden_layers` is a comment rather than a source
of truth. `block_types` reads either spelling and returns the list form.

`transformers` reads both spellings into one list, and a `save_pretrained`
writes that list and no pattern. Version 5.17 renames two entries on the way:
`mamba` becomes `"linear_attention"` and `attention` becomes
`"full_attention"`, so a checkpoint fine-tuned there carries the new names.
Version 5.12.1 writes the list under the reference names. `block_types` reads
both and hands back the four names above. A field set to `null` counts as
absent, the way the reference class treats it.

Two groupings come out of that list:

`scan_groups` returns contiguous runs of one block type, with their indices.
On the two MoE sizes every run is one layer long, because no two adjacent
layers share a type. That's the result a stack builder needs to see: grouping
by type alone never fills a scan. Nano does hold runs of two and three Mamba-2
layers.

`unit_runs` returns the grouping that does fill one. Every feed-forward layer
closes a unit, so the stack cuts into runs of mixers each ending on one
feed-forward, with nothing left over. Identical units sit next to each other
in runs, and a run of R identical units is one `lax.scan` of R steps over
stacked weights.

`cache_plan` splits the layer indices the two cache managers own. Mamba-2
layers need an SSM state slot and a conv state slot, neither of which depends
on sequence length. Attention layers need KV pages that grow with it. Hand it
the resolved list of the stack the engine builds. Hand it a config dict and it
plans the main stack plus the multi-token prediction head, which is what the
checkpoint declares rather than what a runner allocates.

The resolved plans for Super, Ultra and Nano are in `NEMOTRON_3_SUPER_CONFIG`,
`NEMOTRON_3_ULTRA_CONFIG` and `NEMOTRON_3_NANO_CONFIG`, each in the spelling
its own config ships.

Mamba-2 shapes and the chunked scan itself live in `mamba2.py`. This file
decides which layers run it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

__all__ = [
    "BLOCK_TYPES",
    "FEED_FORWARD_BLOCK_TYPES",
    "NEMOTRON_3_NANO_CONFIG",
    "NEMOTRON_3_SUPER_CONFIG",
    "NEMOTRON_3_ULTRA_CONFIG",
    "RECURRENT_BLOCK_TYPES",
    "TRANSFORMERS_BLOCK_TYPES",
    "ScanGroup",
    "Unit",
    "UnitRun",
    "block_types",
    "block_types_with_mtp",
    "cache_plan",
    "check_cover",
    "mtp_block_types",
    "pattern",
    "scan_groups",
    "unit_runs",
    "units",
]

# The four types the reference config accepts. `mlp` is the dense feed-forward
# the Nano size uses where the MoE sizes use `moe`.
BLOCK_TYPES = ("mamba", "moe", "attention", "mlp")

# Types that close a unit. Every mixer before one, `mamba` or `attention`,
# feeds it.
FEED_FORWARD_BLOCK_TYPES = ("moe", "mlp")

# Types that carry a recurrent state across chunks, so `affine_scan.py` moves
# the state and `mamba2.py` builds the (A, B) pairs.
RECURRENT_BLOCK_TYPES = ("mamba",)

_PATTERN_TO_TYPE = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
_TYPE_TO_PATTERN = {v: k for k, v in _PATTERN_TO_TYPE.items()}

# The names `transformers` 5.17 writes, mapped back. Its `_LEGACY_LAYER_TYPE_REMAP`
# renames `mamba` and `attention` when it loads a config, and in `nemotron_h`
# the only linear-attention mixer is Mamba-2.
TRANSFORMERS_BLOCK_TYPES = {"linear_attention": "mamba", "full_attention": "attention"}


@dataclasses.dataclass(frozen=True)
class ScanGroup:
    """A run of adjacent layers that share one block type."""

    block_type: str
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclasses.dataclass(frozen=True)
class Unit:
    """One MoE layer and the mixers that feed it, in order."""

    types: tuple[str, ...]
    start: int

    @property
    def length(self) -> int:
        return len(self.types)

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclasses.dataclass(frozen=True)
class UnitRun(Unit):
    """Adjacent units with the same shape. One `lax.scan` of `repeats` steps.

    It carries the fields of the unit it repeats, and `length` counts every
    repeat.
    """

    repeats: int = 1

    @property
    def unit_length(self) -> int:
        return len(self.types)

    @property
    def length(self) -> int:
        return self.unit_length * self.repeats


def _from_pattern(text: str) -> list[str]:
    """Expand the `M`/`E`/`*` string into the list form."""
    unknown = sorted(set(text) - set(_PATTERN_TO_TYPE))
    if unknown:
        raise ValueError(
            f"hybrid_override_pattern holds unknown symbols {unknown}; "
            f"allowed: {sorted(_PATTERN_TO_TYPE)}"
        )
    return [_PATTERN_TO_TYPE[char] for char in text]


def _validate(types: Sequence[str], field: str) -> list[str]:
    """Return the list form, or raise naming the bad entries.

    The names `transformers` writes map back to `BLOCK_TYPES` first.
    """
    if isinstance(types, str):
        raise ValueError(f"{field} must be a list of strings, not a string")
    out = [TRANSFORMERS_BLOCK_TYPES.get(t, t) for t in types]
    if not out:
        raise ValueError(f"{field} is empty, so the stack has no layers")
    unknown = sorted(set(out) - set(BLOCK_TYPES), key=str)
    if unknown:
        allowed = list(BLOCK_TYPES) + list(TRANSFORMERS_BLOCK_TYPES)
        raise ValueError(f"{field} holds unknown block types {unknown}; allowed: {allowed}")
    return out


def block_types(config: dict) -> list[str]:
    """Per-layer block types, read from either spelling of the field.

    Args:
      config: a parsed `config.json`, or any dict holding
        `layers_block_type` or `hybrid_override_pattern`. The list may use
        the names `transformers` writes. A field set to `None` counts as
        absent.

    Returns:
      One entry per layer, each one of `BLOCK_TYPES`.

    Raises:
      ValueError: neither field is set, an entry is unknown, or
        `num_hidden_layers` disagrees with the length. The reference config
        class only warns on that last one and trusts the list. An engine that
        sizes its stack from the count builds the wrong model, so reject it.
    """
    if config.get("layers_block_type") is not None:
        found = _validate(config["layers_block_type"], "layers_block_type")
    elif config.get("hybrid_override_pattern") is not None:
        found = _validate(
            _from_pattern(config["hybrid_override_pattern"]), "hybrid_override_pattern"
        )
    else:
        raise ValueError(
            "config holds neither layers_block_type nor hybrid_override_pattern, "
            "so the layer stack is undefined"
        )

    declared = config.get("num_hidden_layers")
    if declared is not None and declared != len(found):
        raise ValueError(
            f"num_hidden_layers is {declared} but the block list holds {len(found)} layers"
        )
    return found


def mtp_block_types(config: dict) -> list[str]:
    """Block types of the multi-token prediction head.

    Returns an empty list when `num_nextn_predict_layers` is 0 or absent. The
    head repeats for each predicted token, so the returned list covers one
    repeat. Reads the same names and `None` rule as `block_types`.
    """
    if not config.get("num_nextn_predict_layers"):
        return []
    if config.get("mtp_layers_block_type") is not None:
        return _validate(config["mtp_layers_block_type"], "mtp_layers_block_type")
    if config.get("mtp_hybrid_override_pattern") is not None:
        return _validate(
            _from_pattern(config["mtp_hybrid_override_pattern"]),
            "mtp_hybrid_override_pattern",
        )
    raise ValueError(
        "num_nextn_predict_layers is set but no MTP block list is present"
    )


def block_types_with_mtp(config: dict) -> list[str]:
    """The main stack, then one copy of the MTP head per predicted token.

    This is what the checkpoint holds, not what an engine builds. The head is
    a draft model for speculative decoding and it belongs in its own module,
    so a runner that skips it sizes its KV from `block_types` alone. Use this
    when the question is what the config declares.

    Args:
      config: a parsed `config.json`.

    Returns:
      One entry per block, main stack first.
    """
    repeats = config.get("num_nextn_predict_layers") or 0
    return block_types(config) + mtp_block_types(config) * repeats


def pattern(types: Iterable[str]) -> str:
    """Pack the list form back into the `M`/`E`/`*`/`-` string.

    Takes the names `transformers` writes as well as `BLOCK_TYPES`.

    Raises:
      ValueError: an entry is neither.
    """
    out = []
    for t in types:
        name = TRANSFORMERS_BLOCK_TYPES.get(t, t)
        if name not in _TYPE_TO_PATTERN:
            raise ValueError(
                f"{t!r} is not a block type; allowed: {list(BLOCK_TYPES)}"
            )
        out.append(_TYPE_TO_PATTERN[name])
    return "".join(out)


def _as_types(source) -> list[str]:
    """Take a config dict or an already-resolved list."""
    if isinstance(source, dict):
        return block_types(source)
    return _validate(source, "block types")


def scan_groups(source) -> list[ScanGroup]:
    """Contiguous runs of one block type.

    Args:
      source: a config dict or a resolved block type list.

    Returns:
      Groups in layer order. Concatenating their indices rebuilds
      `range(num_layers)`.
    """
    types = _as_types(source)
    groups: list[ScanGroup] = []
    start = 0
    for i in range(1, len(types) + 1):
        if i == len(types) or types[i] != types[start]:
            groups.append(ScanGroup(types[start], start, i - start))
            start = i
    return groups


def units(source) -> list[Unit]:
    """Cut the stack at every feed-forward layer.

    A feed-forward layer consumes the mixers before it, so it closes a unit.
    The MoE sizes cut into `(mamba, moe)` pairs and `(mamba, attention, moe)`
    triples. Nano runs longer mixer runs, so it also cuts into
    `(mamba, mamba, mlp)` and `(mamba, mamba, mamba, mlp)`. No size leaves a
    layer over.

    Raises:
      ValueError: the stack ends on mixers that no feed-forward layer closes.
    """
    types = _as_types(source)
    out: list[Unit] = []
    start = 0
    for i, block in enumerate(types):
        if block in FEED_FORWARD_BLOCK_TYPES:
            out.append(Unit(tuple(types[start : i + 1]), start))
            start = i + 1
    if start != len(types):
        raise ValueError(
            f"layers {start}..{len(types) - 1} are {types[start:]} and no "
            f"{list(FEED_FORWARD_BLOCK_TYPES)} layer closes them"
        )
    return out


def unit_runs(source) -> list[UnitRun]:
    """Adjacent units of the same shape, folded into runs.

    Each run is one `lax.scan` over stacked weights. A stack builder walks
    this list and emits one scan per entry.
    """
    runs: list[UnitRun] = []
    for unit in units(source):
        if runs and runs[-1].types == unit.types:
            last = runs[-1]
            runs[-1] = UnitRun(last.types, last.start, last.repeats + 1)
        else:
            runs.append(UnitRun(unit.types, unit.start, 1))
    return runs


def cache_plan(source) -> dict[str, list[int]]:
    """Which layers each cache manager owns.

    Args:
      source: a config dict, which plans the main stack plus the MTP head
        through `block_types_with_mtp`, or a resolved block type list, which
        plans that list alone. Pass the list of the stack the engine builds.
        Pass the config only when the question is what the checkpoint
        declares, because the head's indices then continue past the main
        stack.

    Returns:
      `recurrent`: layers holding an SSM state and a conv state. Both sizes
        are fixed per sequence, so neither pages.
      `paged`: layers holding KV, which grows with the sequence.
      `stateless`: the rest, which hold nothing between tokens.
    """
    types = block_types_with_mtp(source) if isinstance(source, dict) else _as_types(source)
    plan: dict[str, list[int]] = {"recurrent": [], "paged": [], "stateless": []}
    for i, block in enumerate(types):
        if block in RECURRENT_BLOCK_TYPES:
            plan["recurrent"].append(i)
        elif block == "attention":
            plan["paged"].append(i)
        else:
            plan["stateless"].append(i)
    return plan


def check_cover(groups: Sequence, num_layers: int) -> None:
    """Confirm the groups tile `range(num_layers)` once each, in order.

    Works on anything carrying `start` and `length`, so it checks both
    `scan_groups` and `unit_runs` output.

    Raises:
      ValueError: a gap, an overlap, an out-of-order group, an empty group, or
        a layer count that doesn't match.
    """
    cursor = 0
    for pos, group in enumerate(groups):
        if group.length <= 0:
            raise ValueError(f"group {pos} covers {group.length} layers")
        if group.start != cursor:
            gap = group.start - cursor
            kind = "gap" if gap > 0 else "overlap"
            raise ValueError(
                f"group {pos} starts at {group.start} but layer {cursor} is next: "
                f"{kind} of {abs(gap)}"
            )
        cursor += group.length
    if cursor != num_layers:
        raise ValueError(f"groups cover {cursor} layers, the stack holds {num_layers}")


# --- Published plans, each in the spelling its own config.json ships. --------

NEMOTRON_3_SUPER_CONFIG = {
    "model_type": "nemotron_h",
    "hidden_size": 4096,
    "num_hidden_layers": 88,
    "hybrid_override_pattern": (
        "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
        "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
    ),
    "num_nextn_predict_layers": 1,
    "mtp_hybrid_override_pattern": "*E",
}

NEMOTRON_3_ULTRA_CONFIG = {
    "model_type": "nemotron_h",
    "hidden_size": 8192,
    "layers_block_type": [
        "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
        "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe",
        "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
        "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
        "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
        "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba",
        "moe", "mamba", "moe", "mamba", "moe",
    ],
    "num_nextn_predict_layers": 1,
    "mtp_layers_block_type": ["attention", "moe"],
}

NEMOTRON_3_NANO_CONFIG = {
    "model_type": "nemotron_h",
    "hidden_size": 3136,
    "num_hidden_layers": 42,
    "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
}

PUBLISHED_CONFIGS = {
    "Nemotron 3 Super 120B-A12B": NEMOTRON_3_SUPER_CONFIG,
    "Nemotron 3 Ultra 550B-A55B": NEMOTRON_3_ULTRA_CONFIG,
    "Nemotron 3 Nano 4B": NEMOTRON_3_NANO_CONFIG,
}
