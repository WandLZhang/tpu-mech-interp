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

"""Correctness gate for the Nemotron 3 layer plan.

Runs on CPU. No TPU needed. Only the state budget runs jax, on one device, and
the first line names the backend it got.

    python3 scan/test_nemotron3_layers.py

The published Super, Ultra and Nano configs are the fixtures, each in the
spelling its own `config.json` ships. Every case checks the parsed plan against
counts, indices and pattern strings transcribed by hand from that config, then
confirms the groups tile the stack once each. Each case also spells its stack
the way `transformers` 5.17 saves it, and saves it through the `transformers`
that's installed, when it has `nemotron_h`, and both have to parse to the same
stack. 5.17 writes the renamed entries and 5.12.1 the reference names, so the
check takes either spelling and prints which one it read. After that a
negative control corrupts the input and the parser has to reject it. If a
control passes, this file fails itself.

Two rules keep the checks from grading themselves. Nothing compares one parser
output against another parser output, and every byte figure the model docs
publish appears here as a literal.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testing import skip_reason  # noqa: E402
from affine_scan import compose_local  # noqa: E402
from mamba2 import NEMOTRON_3_SUPER, chunk_pairs, discretize  # noqa: E402
from nemotron3_layers import (  # noqa: E402
    NEMOTRON_3_NANO_CONFIG,
    NEMOTRON_3_SUPER_CONFIG,
    NEMOTRON_3_ULTRA_CONFIG,
    ScanGroup,
    block_types,
    cache_plan,
    check_cover,
    mtp_block_types,
    pattern,
    scan_groups,
    block_types_with_mtp,
    unit_runs,
    units,
)

# What each pattern symbol means, written out here so no expected list comes
# from the parser. `TRANSFORMERS_NAMES` is what `transformers` 5 writes for the
# same symbols: its `_pattern_to_list` and `_LEGACY_LAYER_TYPE_REMAP` rename
# `mamba` and `attention`.
SYMBOL_NAMES = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
TRANSFORMERS_NAMES = {"M": "linear_attention", "E": "moe", "*": "full_attention", "-": "mlp"}

MAMBA_MOE = ("mamba", "moe")
MAMBA_ATTN_MOE = ("mamba", "attention", "moe")
MAMBA_MLP = ("mamba", "mlp")
MAMBA2_MLP = ("mamba", "mamba", "mlp")
MAMBA3_MLP = ("mamba", "mamba", "mamba", "mlp")
MAMBA_ATTN_MLP = ("mamba", "attention", "mlp")
MAMBA2_ATTN_MLP = ("mamba", "mamba", "attention", "mlp")

# --- Figures the model docs publish, written here as literals. ---------------
#
# Super Mamba-2 shapes, from the published config: 128 heads, state 128, head
# dim 64, 8 groups, conv kernel 4, chunk 128. `mamba_ssm_cache_dtype` is
# float32, so every state byte count below uses 4 bytes.

SUPER_MAMBA_SHAPE = {
    "ssm_state_size": 128,
    "mamba_num_heads": 128,
    "mamba_head_dim": 64,
    "n_groups": 8,
    "conv_kernel": 4,
    "chunk_size": 128,
}

# 128 heads * 64 head dim * 128 state * 4 bytes.
SUPER_SSM_BYTES_PER_LAYER = 4 * 2**20
# conv_dim = 128 * 64 + 2 * 8 * 128 = 10,240 channels. The slot holds the
# conv_kernel - 1 tokens before the current one, at 4 bytes each.
SUPER_CONV_DIM = 10_240
SUPER_CONV_BYTES_PER_LAYER = 120 * 2**10
SUPER_RECURRENT_LAYERS = 40
SUPER_SSM_BYTES = 160 * 2**20
SUPER_CONV_BYTES = 4800 * 2**10  # 4.6875 MiB

# 2 KV heads * 128 head dim * 2 tensors * 2 bytes of BF16.
SUPER_KV_BYTES_PER_TOKEN_PER_LAYER = 1024
# The config declares 9 attention blocks, the 8 of the main stack plus the one
# in the MTP head. `models/nemotron_h.py` builds the main stack only, so the
# engine's KV budget is the second pair.
SUPER_DECLARED_ATTENTION_LAYERS = 9
SUPER_BUILT_ATTENTION_LAYERS = 8
SUPER_KV_BYTES_PER_TOKEN = 8192
SUPER_CONTEXT = 262_144
SUPER_KV_BYTES_AT_FULL_CONTEXT = 2 * 2**30  # 2.0 GiB

# Transcribed by hand from each published config, not from the parser.
EXPECTED = {
    "Super 120B-A12B": {
        "config": NEMOTRON_3_SUPER_CONFIG,
        "layers": 88,
        "counts": {"mamba": 40, "moe": 40, "attention": 8},
        "first_eight": [
            "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        ],
        "attention_at": [7, 16, 25, 36, 47, 58, 69, 78],
        "served_attention_at": [7, 16, 25, 36, 47, 58, 69, 78, 88],
        "groups": 88,
        "widest_group": 1,
        "adjacent_repeats": 0,
        "runs": [
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4),
        ],
        "unit_shapes": {MAMBA_MOE, MAMBA_ATTN_MOE},
        "units": 40,
        "mtp": ["attention", "moe"],
        "pattern": (
            "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
            "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
        ),
    },
    "Ultra 550B-A55B": {
        "config": NEMOTRON_3_ULTRA_CONFIG,
        "layers": 108,
        "counts": {"mamba": 48, "moe": 48, "attention": 12},
        "first_eight": [
            "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
        ],
        "attention_at": [7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98],
        "served_attention_at": [7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98, 108],
        "groups": 108,
        "widest_group": 1,
        "adjacent_repeats": 0,
        "runs": [
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 2), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 2), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 2), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 2), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 3), (MAMBA_ATTN_MOE, 1),
            (MAMBA_MOE, 4),
        ],
        "unit_shapes": {MAMBA_MOE, MAMBA_ATTN_MOE},
        "units": 48,
        "mtp": ["attention", "moe"],
        "pattern": (
            "MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*"
            "EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        ),
    },
    "Nano 4B": {
        "config": NEMOTRON_3_NANO_CONFIG,
        "layers": 42,
        "counts": {"mamba": 21, "mlp": 17, "attention": 4},
        "first_eight": [
            "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "mamba",
        ],
        "attention_at": [12, 17, 24, 32],
        "served_attention_at": [12, 17, 24, 32],
        "groups": 38,
        "widest_group": 3,
        "adjacent_repeats": 4,
        "runs": [
            (MAMBA_MLP, 3), (MAMBA2_MLP, 1), (MAMBA_MLP, 1),
            (MAMBA_ATTN_MLP, 1), (MAMBA_MLP, 1), (MAMBA_ATTN_MLP, 1),
            (MAMBA_MLP, 2), (MAMBA_ATTN_MLP, 1), (MAMBA_MLP, 2),
            (MAMBA2_ATTN_MLP, 1), (MAMBA3_MLP, 1), (MAMBA_MLP, 2),
        ],
        "unit_shapes": {
            MAMBA_MLP, MAMBA2_MLP, MAMBA3_MLP, MAMBA_ATTN_MLP, MAMBA2_ATTN_MLP
        },
        "units": 17,
        "mtp": [],
        "pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
    },
}


class Report:
    """Collects PASS and FAIL lines and counts the failures."""

    def __init__(self):
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        tail = f"  {detail}" if detail else ""
        print(f"  [{tag}] {label}{tail}")
        if not ok:
            self.failures += 1
        return ok

    def expect_ok(self, label: str, fn) -> None:
        """A call that has to return. A raise counts as one failure, not a crash.

        `check_cover` reports a tiling bug by raising. Letting that escape
        aborts the run at the first failure and hides every check after it, so
        catch it here and carry on.
        """
        try:
            fn()
        except ValueError as exc:
            self.check(False, label, str(exc))
            return
        self.check(True, label)

    def control(self, label: str, fn) -> None:
        """A control must raise ValueError. Passing is itself a failure."""
        try:
            fn()
        except ValueError as exc:
            print(f"      control ({label}): rejected -> {exc}")
            return
        print(f"      control ({label}): ACCEPTED")
        print("      FAIL: the control did not fail, so this test detects nothing.")
        self.failures += 1

    def equals(self, label: str, fn, want, detail: str = "") -> None:
        """`fn()` has to return `want`. A raise counts as one failure."""
        try:
            got = fn()
        except Exception as exc:  # noqa: BLE001
            self.check(False, label, f"raised {type(exc).__name__}: {exc}")
            return
        self.check(got == want, label, detail if got == want else f"got {got}")


def transformers_resave(config: dict) -> tuple[dict | None, str | None]:
    """The config after `transformers` loads it and writes it back.

    That's the `config.json` a fine-tune saves. Returns the saved dict and
    None, or None and the reason there's no save: `transformers` missing, or a
    version without `nemotron_h`.
    """
    try:
        from transformers.models.nemotron_h import NemotronHConfig
    except ImportError as exc:
        return None, skip_reason("nemotron_h", exc)
    with tempfile.TemporaryDirectory() as folder:
        NemotronHConfig(**config).save_pretrained(folder)
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
            return json.load(handle), None


def check_model(name: str, spec: dict, report: Report) -> None:
    print(f"{name}")
    config = spec["config"]
    types = block_types(config)

    report.check(
        len(types) == spec["layers"],
        "layer count",
        f"{len(types)} layers",
    )

    counts = {t: types.count(t) for t in sorted(set(types))}
    report.check(
        counts == spec["counts"],
        "block type counts",
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )

    # Pins the symbol-to-type map itself. A map that swapped two symbols keeps
    # every count above and fails here.
    report.check(
        types[:8] == spec["first_eight"],
        "the first eight layers",
        f"{types[:8]}",
    )

    plan = cache_plan(types)
    report.check(
        plan["paged"] == spec["attention_at"],
        "attention layer indices",
        f"{plan['paged']}",
    )
    report.check(
        len(plan["recurrent"]) == spec["counts"]["mamba"],
        "SSM state slots",
        f"{len(plan['recurrent'])} layers hold a recurrent state",
    )
    report.check(
        sorted(plan["recurrent"] + plan["paged"] + plan["stateless"])
        == list(range(len(types))),
        "cache plan partitions the stack",
    )

    # The config declares the MTP head on top of the main stack. A runner that
    # builds the head allocates for these indices; one that skips it, which is
    # what `models/nemotron_h.py` does, allocates for the main stack alone.
    declared = block_types_with_mtp(config)
    declared_plan = cache_plan(config)
    report.check(
        len(declared) == spec["layers"] + len(spec["mtp"]),
        "the declared stack covers the MTP head",
        f"{len(declared)} blocks",
    )
    report.check(
        declared_plan["paged"] == spec["served_attention_at"],
        "declared attention indices",
        f"{declared_plan['paged']}",
    )

    groups = scan_groups(types)
    report.expect_ok("same-type runs tile the stack", lambda: check_cover(groups, len(types)))
    widest = max(g.length for g in groups)
    report.check(
        len(groups) == spec["groups"] and widest == spec["widest_group"],
        "same-type run count and width",
        f"{len(groups)} runs, widest {widest}",
    )
    report.check(
        [g.block_type for g in groups] == [types[g.start] for g in groups],
        "each run carries the type of the layers it covers",
    )
    # Derived from `types` alone, so it checks the stack rather than the parser.
    adjacent = sum(1 for a, b in zip(types, types[1:]) if a == b)
    report.check(
        adjacent == spec["adjacent_repeats"],
        "adjacent layers of one type",
        f"{adjacent} pairs",
    )

    runs = unit_runs(types)
    report.expect_ok("unit runs tile the stack", lambda: check_cover(runs, len(types)))
    got = [(r.types, r.repeats) for r in runs]
    report.check(got == spec["runs"], "unit runs", f"{len(runs)} scans")

    every_unit = units(types)
    shapes = {u.types for u in every_unit}
    report.check(
        shapes == spec["unit_shapes"] and len(every_unit) == spec["units"],
        "unit shapes",
        f"{len(every_unit)} units in {len(shapes)} shapes",
    )

    # Against the literal string, not against another parser call.
    report.check(
        pattern(types) == spec["pattern"],
        "types pack back to the published pattern",
        pattern(types)[:24] + "...",
    )
    # The stack the pattern's symbols name, written out from `SYMBOL_NAMES`
    # rather than read back from the parser. Super and Nano ship this same
    # pattern, so comparing the two parser calls graded the parser against
    # itself.
    literal = [SYMBOL_NAMES[c] for c in spec["pattern"]]
    report.equals(
        "the published pattern parses to the types its symbols name",
        lambda: block_types({"hybrid_override_pattern": spec["pattern"]}),
        literal,
    )
    report.check(mtp_block_types(config) == spec["mtp"], "MTP head", f"{spec['mtp']}")

    # A fine-tune saves the config back through `transformers` 5.17, which
    # writes the list under two new names and drops the pattern. Built here
    # from the literal pattern, so the expected stack doesn't come from the
    # parser.
    renamed = {
        k: v
        for k, v in config.items()
        if k not in ("hybrid_override_pattern", "mtp_hybrid_override_pattern")
    }
    renamed["layers_block_type"] = [TRANSFORMERS_NAMES[c] for c in spec["pattern"]]
    renamed["mtp_layers_block_type"] = ["full_attention", "moe"]
    report.equals(
        "the names transformers writes parse to the same stack",
        lambda: block_types(renamed),
        literal,
        f"{len(literal)} layers",
    )
    report.equals(
        "and to the same declared attention indices",
        lambda: cache_plan(renamed)["paged"],
        spec["served_attention_at"],
    )

    resaved, missing = transformers_resave(config)
    if missing:
        print(f"  skipped: a save through transformers, because {missing}")
    else:
        # 5.17 writes `linear_attention` and `full_attention`, 5.12.1 writes
        # `mamba` and `attention`. Either spelling parses. A list that mixes the
        # two, or a save that keeps the pattern beside it, fails here.
        import transformers

        names = set(resaved.get("layers_block_type") or [])
        renamed_names = {"linear_attention", "full_attention"}
        reference_names = {"mamba", "attention"}
        spelling = "renamed" if names & renamed_names else "reference"
        report.check(
            "hybrid_override_pattern" not in resaved
            and bool(names & (renamed_names | reference_names))
            and not (names & renamed_names and names & reference_names),
            f"transformers {transformers.__version__} writes one spelling, the"
            f" {spelling} names, and no pattern",
            f"{sorted(names)}",
        )
        report.equals(
            "a config saved through transformers parses to the same stack",
            lambda: block_types(resaved),
            literal,
        )
        report.equals(
            "and to the same declared attention indices and MTP head",
            lambda: (cache_plan(resaved)["paged"], mtp_block_types(resaved)),
            (spec["served_attention_at"], spec["mtp"]),
        )

    # Negative controls on this model's own stack.
    broken = copy.deepcopy(config)
    broken.pop("hybrid_override_pattern", None)
    broken["layers_block_type"] = list(types)
    broken["layers_block_type"][3] = "swiglu"
    report.control("unknown block type", lambda: block_types(broken))

    short = copy.deepcopy(config)
    short.pop("hybrid_override_pattern", None)
    short["layers_block_type"] = list(types)[:-1]
    short["num_hidden_layers"] = len(types)
    report.control("layer count disagrees", lambda: block_types(short))

    trailing = list(types) + ["mamba"]
    report.control("stack ends on a mixer", lambda: units(trailing))

    dropped = [ScanGroup(g.block_type, g.start, g.length) for g in groups]
    dropped.pop(1)
    report.control("a group is missing", lambda: check_cover(dropped, len(types)))

    doubled = list(groups) + [ScanGroup(types[-1], len(types), 1)]
    report.control("a group runs past the end", lambda: check_cover(doubled, len(types)))

    # The three below keep the total layer count right, so only the ordering
    # and empty-group branches of `check_cover` can catch them. Without these
    # a `check_cover` that summed lengths and checked nothing else would pass.
    swapped = list(groups)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    report.control("groups are out of order", lambda: check_cover(swapped, len(types)))

    overlapped = list(groups)
    overlapped[2] = ScanGroup(
        overlapped[2].block_type, overlapped[1].start, overlapped[2].length
    )
    report.control(
        "two groups cover one layer", lambda: check_cover(overlapped, len(types))
    )

    empty = list(groups)
    empty.insert(2, ScanGroup(types[groups[2].start], groups[2].start, 0))
    report.control("a group covers no layers", lambda: check_cover(empty, len(types)))
    print()


def check_parser(report: Report) -> None:
    """The parser branches no published config reaches.

    Every rejection a stack builder relies on gets a control here, and every
    quiet return gets a positive check. These run once, because they don't
    depend on a model.
    """
    print("Parser")

    # `scan_groups` has to merge adjacent layers of one type. Both published
    # MoE sizes hold no adjacent pair, so without this a version that split
    # after every layer would pass.
    merged = scan_groups(["mamba", "mamba", "moe", "moe", "moe", "attention"])
    report.check(
        [(g.block_type, g.start, g.length) for g in merged]
        == [("mamba", 0, 2), ("moe", 2, 3), ("attention", 5, 1)],
        "scan_groups merges adjacent layers of one type",
        f"{[(g.block_type, g.length) for g in merged]}",
    )

    report.check(mtp_block_types({}) == [], "no MTP field means no head")
    report.check(
        mtp_block_types({"num_nextn_predict_layers": 0, "mtp_hybrid_override_pattern": "*E"})
        == [],
        "a disabled MTP head reads as empty",
    )
    report.check(
        block_types_with_mtp({"hybrid_override_pattern": "ME"}) == ["mamba", "moe"],
        "a config with no MTP head declares the main stack alone",
    )
    report.check(
        block_types_with_mtp(
            {"hybrid_override_pattern": "ME", "num_nextn_predict_layers": 2,
             "mtp_hybrid_override_pattern": "*E"}
        )
        == ["mamba", "moe", "attention", "moe", "attention", "moe"],
        "the MTP head repeats once per predicted token",
    )

    # A field set to null counts as absent, the way the reference class reads
    # it, so the pattern next to it decides the stack.
    report.equals(
        "a null block list falls back to the pattern",
        lambda: block_types({"layers_block_type": None, "hybrid_override_pattern": "M-M*-"}),
        ["mamba", "mlp", "mamba", "attention", "mlp"],
    )
    report.equals(
        "a null MTP list falls back to the MTP pattern",
        lambda: mtp_block_types(
            {
                "num_nextn_predict_layers": 1,
                "mtp_layers_block_type": None,
                "mtp_hybrid_override_pattern": "*E",
            }
        ),
        ["attention", "moe"],
    )
    report.control(
        "a name outside both spellings",
        lambda: block_types({"layers_block_type": ["linear_attention", "sliding_attention"]}),
    )
    report.control(
        "every block field null",
        lambda: block_types({"layers_block_type": None, "hybrid_override_pattern": None}),
    )
    report.control(
        "neither block field present", lambda: block_types({"num_hidden_layers": 4})
    )
    report.control(
        "block list is empty", lambda: block_types({"layers_block_type": []})
    )
    report.control(
        "block list is a bare string",
        lambda: block_types({"layers_block_type": "MEME"}),
    )
    report.control(
        "pattern holds an unknown symbol",
        lambda: block_types({"hybrid_override_pattern": "MEMXE"}),
    )
    report.control(
        "MTP is on with no head list",
        lambda: mtp_block_types({"num_nextn_predict_layers": 1}),
    )
    report.control(
        "MTP head holds an unknown symbol",
        lambda: mtp_block_types(
            {"num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*X"}
        ),
    )
    report.control(
        "packing a type the pattern has no symbol for",
        lambda: pattern(["mamba", "swiglu"]),
    )
    print()


def check_state_budget(report: Report) -> None:
    """Tie the plan to the allocation it drives.

    Every figure the model doc publishes appears here as a literal. Nothing
    reads a shape out of the config object and then asserts against that same
    object, so a wrong config field fails rather than moving both sides.
    """
    print("State budget for Super 120B-A12B")
    cfg = NEMOTRON_3_SUPER

    wrong = {
        field: getattr(cfg, field)
        for field, want in SUPER_MAMBA_SHAPE.items()
        if getattr(cfg, field) != want
    }
    report.check(
        not wrong,
        "Mamba-2 config matches the published shapes",
        f"{wrong or 'all six fields'}",
    )
    report.check(
        cfg.conv_dim == SUPER_CONV_DIM,
        "conv_dim",
        f"{cfg.conv_dim} channels",
    )

    heads = SUPER_MAMBA_SHAPE["mamba_num_heads"]
    state = SUPER_MAMBA_SHAPE["ssm_state_size"]
    head_dim = SUPER_MAMBA_SHAPE["mamba_head_dim"]
    tokens = SUPER_MAMBA_SHAPE["chunk_size"]

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(tokens, heads, head_dim)).astype(np.float32))
    b = jnp.asarray(
        rng.normal(size=(tokens, SUPER_MAMBA_SHAPE["n_groups"], state)).astype(np.float32)
    )
    dt_raw = jnp.asarray(rng.normal(size=(tokens, heads)).astype(np.float32))
    dt_bias = jnp.zeros((heads,), jnp.float32)
    a_log = jnp.zeros((heads,), jnp.float32)
    dt, log_decay = discretize(dt_raw, dt_bias, a_log, cfg.time_step_limit)

    a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, tokens)
    a_total, b_total = compose_local(a_chunks, b_chunks)

    report.check(
        a_chunks.shape == (1, 128, 128, 128) and b_chunks.shape == (1, 128, 128, 64),
        "chunk pairs match the Super Mamba-2 shapes",
        f"A{tuple(a_chunks.shape)} B{tuple(b_chunks.shape)}",
    )
    report.check(
        a_total.shape == (128, 128, 128) and b_total.shape == (128, 128, 64),
        "one folded pair per layer",
        f"A{tuple(a_total.shape)} B{tuple(b_total.shape)}",
    )
    report.check(
        float(np.abs(np.asarray(b_total)).max()) > 0.0,
        "the folded pair holds a state, not zeros",
        f"max|B|={float(np.abs(np.asarray(b_total)).max()):.3e}",
    )

    ssm_per_layer = cfg.mamba_num_heads * cfg.state_bytes_per_head
    report.check(
        ssm_per_layer == SUPER_SSM_BYTES_PER_LAYER,
        "SSM state bytes per layer",
        f"{ssm_per_layer / 2**20:.2f} MiB",
    )
    report.check(
        cfg.conv_state_bytes == SUPER_CONV_BYTES_PER_LAYER,
        "conv state bytes per layer",
        f"{cfg.conv_state_bytes / 2**10:.0f} KiB",
    )
    report.check(
        cfg.state_bytes_per_layer == SUPER_SSM_BYTES_PER_LAYER + SUPER_CONV_BYTES_PER_LAYER,
        "both slots add up",
        f"{cfg.state_bytes_per_layer / 2**20:.2f} MiB per layer",
    )

    recurrent = cache_plan(NEMOTRON_3_SUPER_CONFIG)["recurrent"]
    report.check(
        len(recurrent) == SUPER_RECURRENT_LAYERS,
        "Mamba-2 layers holding a recurrent state",
        f"{len(recurrent)}",
    )
    report.check(
        len(recurrent) * ssm_per_layer == SUPER_SSM_BYTES,
        "SSM state per sequence",
        f"{len(recurrent) * ssm_per_layer / 2**20:.0f} MiB, fixed at any length",
    )
    report.check(
        len(recurrent) * cfg.conv_state_bytes == SUPER_CONV_BYTES,
        "conv state per sequence",
        f"{len(recurrent) * cfg.conv_state_bytes / 2**20:.2f} MiB, fixed at any length",
    )

    declared = cache_plan(NEMOTRON_3_SUPER_CONFIG)["paged"]
    report.check(
        len(declared) == SUPER_DECLARED_ATTENTION_LAYERS,
        "attention layers the config declares",
        f"{len(declared)}, MTP head included",
    )
    paged = cache_plan(block_types(NEMOTRON_3_SUPER_CONFIG))["paged"]
    report.check(
        len(paged) == SUPER_BUILT_ATTENTION_LAYERS,
        "attention layers the engine builds",
        f"{len(paged)}, MTP head left out",
    )
    kv_per_token = len(paged) * SUPER_KV_BYTES_PER_TOKEN_PER_LAYER
    report.check(
        kv_per_token == SUPER_KV_BYTES_PER_TOKEN,
        "KV bytes per token",
        f"{kv_per_token / 2**10:.0f} KiB",
    )
    report.check(
        kv_per_token * SUPER_CONTEXT == SUPER_KV_BYTES_AT_FULL_CONTEXT,
        "KV per sequence at the full context",
        f"{kv_per_token * SUPER_CONTEXT / 2**30:.2f} GiB over {SUPER_CONTEXT} tokens",
    )
    print()


def main() -> int:
    report = Report()
    print(f"jax backend: {jax.default_backend()}, for the state budget\n")
    # Each section runs behind a catch, so one broken model still leaves the
    # other two, the parser checks and the state budget on screen. The run
    # ends on a full list of failures rather than the first traceback.
    sections = [
        (name, lambda spec=spec, name=name: check_model(name, spec, report))
        for name, spec in EXPECTED.items()
    ]
    sections.append(("Parser", lambda: check_parser(report)))
    sections.append(("State budget", lambda: check_state_budget(report)))

    for label, section in sections:
        try:
            section()
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {label} raised {type(exc).__name__}: {exc}\n")
            report.failures += 1

    if report.failures:
        print(f"FAILED: {report.failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
