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

"""Correctness gate for the DeepSeek V4.1-Flash layer plan.

Runs on CPU. The plan is config arithmetic, so this needs no accelerator and no
checkpoint download.

    python3 scan/test_deepseek_v4_layers.py

Four independent references grade the parser. None of them reads the parser.

1. **The shipped checkpoint keys.** `CHECKPOINT_TENSORS` lists, for all 43
   blocks, the tensors that block carries beyond the set every block carries,
   transcribed from the safetensors headers of
   `deepseek-ai/DeepSeek-V4.1-Flash`. It's keyed by the prefix the index gives
   each block, so the plan has to land on the namespace as well as the names.
   `.scale` siblings are listed, because only four of the mode-specific
   tensors have one and no suffix rule recovers which.

2. **The shipped checkpoint bytes.** `ENGRAM_SHARD_BYTES` is the file size of
   the two Engram shards. A safetensors file is an 8-byte length, a JSON
   header and the tensor payload back to back, so the plan's row count times
   its row width, plus the four small tensors beside it, plus the header, has
   to reproduce the file size to the byte.

3. **The published hashing rule.** Every (n-gram size, head) pair takes the
   next unused prime at or above `engram_vocab_size`, and a table's row count
   is the sum of its pairs' primes. `derive_engram_rows` runs that rule with
   its own prime sieve and reproduces `engram_num_embeddings` to the row. It
   calls nothing in `deepseek_v4_layers`.

4. **The model card.** 890 bytes of global KV per token, 196B Engram
   parameters, 552B backbone, 20 encoder layers and 20 decoder layers. Those
   sit below as literals and the plan has to land on them.

After the checks, negative controls corrupt the config and the parser has to
reject each one with the message that names the fault. A control that passes,
or that raises for an unrelated reason, fails this file.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v4_layers import (  # noqa: E402
    DECODER,
    DEEPSEEK_V41_FLASH_CONFIG,
    DRAFT,
    ENCODER,
    FULL,
    POOL_BUILDS,
    POOL_READS,
    REINDEX,
    REUSE,
    ROOT_TENSORS,
    WINDOW,
    KVQuantization,
    Unit,
    cache_plan,
    check_causal_encoder_decoder,
    check_cover,
    compress_ratios,
    engram_plan,
    gib,
    global_kv_bytes_per_token,
    layer_plans,
    padded_bf16_kv_bytes_per_token,
    residual_stream,
    rotary_tables,
    stage_split,
    text_config,
    unit_runs,
    units,
)

# --- Reference 4: figures the model card publishes. --------------------------

BACKBONE_LAYERS = 40
DRAFT_LAYERS = 3
BLOCKS = 43
DECODER_START = 20
ENCODER_LAYERS = 20
DECODER_LAYERS = 20

# "reduce the global KV cache footprint to 890 bytes per token"
GLOBAL_KV_BYTES_PER_TOKEN = 890.0
CONTEXT = 1_048_576

# "Engram conditional memory (196B parameters)", to the nearest billion.
ENGRAM_PARAMS_BILLIONS = 196

# 384,006,168 + 384,016,682 rows, each 256 fp8 values plus 8 e8m0 scales.
ENGRAM_ROW_BYTES = 264
# Row counts times the row width and the row bytes. These are the plan's own
# arithmetic written out, and reference 2 below is what grades them.
ENGRAM_PARAMS = 196_613_849_600
ENGRAM_BYTES = 202_758_032_400
ENGRAM_TABLE_BYTES = (101_377_628_352, 101_380_404_048)

# 2-gram through 4-gram, 8 heads each, at two layers.
ENGRAM_HASH_COLUMNS = 24
ENGRAM_ROWS_PER_TOKEN = 48
ENGRAM_GATHER_BYTES_PER_TOKEN = 12_672
ENGRAM_PAYLOAD_BYTES_PER_TOKEN = 24_576

# Window KV: 528 bytes a slot, one KV head, 128 slots, 43 blocks.
WINDOW_KV_BYTES_PER_SEQUENCE = 2_906_112

# --- The slice this repo targets, from the top-level README. -----------------

CHIPS = 32
HBM_BYTES_PER_CHIP = 95 * 2**30
ICI_BYTES_PER_S = 1.2e12
HOST_LINK_BYTES_PER_S = 32e9
SLICE_BF16_FLOPS = 14.7e15

# One 8,192-token prefill chunk at 8B active parameters, two flops a parameter,
# at the slice's peak BF16 rate.
PREFILL_TOKENS = 8192
PREFILL_SECONDS = PREFILL_TOKENS * 8e9 * 2 / SLICE_BF16_FLOPS
# One decode step at 16B active parameters in BF16, read from HBM once.
DECODE_TOKENS = 256
DECODE_SECONDS = 16e9 * 2 / (CHIPS * 2765e9)

# --- Reference 1: tensors the checkpoint carries, by block. ------------------
#
# Read out of the safetensors headers. `PLAIN_BLOCK` is the intersection over
# all 43 blocks, with the expert index collapsed. `CHECKPOINT_TENSORS` is what
# each block holds on top of it, keyed by the prefix the index gives that
# block. The `layers.` namespace stops at 39 and the three draft blocks sit
# under `mtp.`.

PLAIN_BLOCK = (
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn.experts.N.w1.scale",
    "ffn.experts.N.w1.weight",
    "ffn.experts.N.w2.scale",
    "ffn.experts.N.w2.weight",
    "ffn.experts.N.w3.scale",
    "ffn.experts.N.w3.weight",
    "ffn.gate.bias",
    "ffn.gate.bias_vl",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

# A ratio-2 Full layer: compressor with a pooling gate, and the whole indexer.
# `wq_b` is the one indexer tensor stored fp8, so it alone brings a scale.
FULL_POOLED = (
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
    "attn.indexer.k_norm.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wk.weight",
    "attn.indexer.wq_b.scale",
    "attn.indexer.wq_b.weight",
)
# A ratio-1 Full layer. One token a latent needs no pooling gate.
FULL_FLAT = tuple(n for n in FULL_POOLED if n != "attn.compressor.wgate.weight")
# A Reindex layer scores against somebody else's keys, so it carries the query
# side of the indexer and nothing else.
REINDEX_ONLY = (
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.scale",
    "attn.indexer.wq_b.weight",
)
# An Engram layer. `embed` and `wkv` are fp8 and bring scales; the two gate
# vectors are bf16 and don't.
ENGRAM_SET = (
    "engram.embed.scale",
    "engram.embed.weight",
    "engram.k_weight",
    "engram.q_weight",
    "engram.wkv.scale",
    "engram.wkv.weight",
)
# The first draft stage projects the concatenated target hidden states down.
DRAFT_INPUT = (
    "main_norm.weight",
    "main_proj.scale",
    "main_proj.weight",
)
# The last draft stage carries the heads that turn the stream into tokens.
DRAFT_HEAD = (
    "confidence_head.proj.weight",
    "markov_head.embed.weight",
    "markov_head.head.weight",
    "norm.weight",
)

CHECKPOINT_TENSORS = {f"layers.{i}.": () for i in range(BACKBONE_LAYERS)}
CHECKPOINT_TENSORS.update(
    {
        "layers.1.": ENGRAM_SET,
        "layers.2.": FULL_POOLED,
        "layers.8.": FULL_POOLED,
        "layers.14.": FULL_POOLED + ENGRAM_SET,
        "layers.20.": FULL_FLAT,
        "layers.24.": REINDEX_ONLY,
        "layers.28.": REINDEX_ONLY,
        "layers.32.": REINDEX_ONLY,
        "layers.36.": REINDEX_ONLY,
        "mtp.0.": DRAFT_INPUT,
        "mtp.1.": (),
        "mtp.2.": DRAFT_HEAD,
    }
)

# Shapes the shards declare, for the tensors the config has to reproduce.
CHECKPOINT_SHAPES = {
    "attn.wq_a.weight": (1280, 5120),
    "attn.wq_b.weight": (32768, 1280),
    "attn.wkv.weight": (512, 5120),
    "attn.wo_a.weight": (8192, 4096),
    "attn.wo_b.weight": (5120, 8192),
    "attn.attn_sink": (64,),
    "hc_attn_fn": (24, 20480),
    "ffn.gate.bias": (384,),
    "attn.compressor.wkv.weight": (512, 5120),
    "attn.indexer.wq_b.weight": (4096, 1280),
    "attn.indexer.wk.weight": (128, 512),
    "attn.indexer.weights_proj.weight": (32, 5120),
    "engram.wkv.weight": (25600, 6144),
    "main_proj.weight": (5120, 15360),
    "markov_head.embed.weight": (129280, 256),
    "confidence_head.proj.weight": (1, 5376),
}

# Tensor shapes the Engram shards declare, for the table and its projection.
ENGRAM_EMBED_SHAPES = {1: (384_006_168, 256), 14: (384_016_682, 256)}
ENGRAM_SCALE_SHAPES = {1: (384_006_168, 8), 14: (384_016_682, 8)}

# --- Reference 2: the size of the two Engram shards on disk. -----------------
#
# A safetensors file is an 8-byte little-endian header length, that many bytes
# of JSON, then the payload. Each shard holds one table plus the four small
# tensors beside it, so the file size pins the row count.

ENGRAM_SHARD_BYTES = {1: 101_535_150_936, 14: 101_537_926_640}
ENGRAM_SHARD_HEADER_BYTES = {1: 656, 14: 664}
# q_weight and k_weight are bf16 [hc_mult, hidden_size]; wkv is fp8
# [hidden_size * (hc_mult + 1), hash_columns * engram_head_dim] with an e8m0
# scale per 32 by 32 block.
ENGRAM_SHARD_SIDECAR_BYTES = 2 * (4 * 5120 * 2) + 25_600 * 6_144 + 800 * 192

# --- The parsed plan, transcribed by hand from config.json. ------------------

EXPECTED_MODES = (
    [WINDOW] * 2
    + ([FULL] + [REUSE] * 5) * 3
    + [FULL]
    + [REUSE] * 3
    + ([REINDEX] + [REUSE] * 3) * 4
    + [WINDOW] * 3
)

# Nine runs, not seven. The three draft blocks all run window attention, but
# they carry three different parameter sets, so their weights can't be stacked
# and each needs its own scan body.
EXPECTED_RUNS = [
    (ENCODER, (WINDOW,), (False,), 0, 1),
    (ENCODER, (WINDOW,), (True,), 1, 1),
    (ENCODER, (FULL,) + (REUSE,) * 5, (False,) * 6, 2, 2),
    (ENCODER, (FULL,) + (REUSE,) * 5, (True,) + (False,) * 5, 14, 1),
    (DECODER, (FULL,) + (REUSE,) * 3, (False,) * 4, 20, 1),
    (DECODER, (REINDEX,) + (REUSE,) * 3, (False,) * 4, 24, 4),
    (DRAFT, (WINDOW,), (False,), 40, 1),
    (DRAFT, (WINDOW,), (False,), 41, 1),
    (DRAFT, (WINDOW,), (False,), 42, 1),
]
SCAN_BODIES = 9
# Layer 20 builds the candidate pool and the Reindex layers from 24 index
# inside it. Every other run opens on a layer the pool doesn't touch.
POOL_ROLES = {20: POOL_BUILDS, 24: POOL_READS}

KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
POOLING_LAYERS = [2, 8, 14]
CANDIDATE_LAYER = 20
ENGRAM_LAYERS = [1, 14]

# Rotary. One table per layer, picked by `compress_ratios[i]` alone, and it
# turns the last 64 channels of a 512-wide head.
ROTATED_CHANNELS = 64
YARN_LAYERS = tuple(range(2, 40))
PLAIN_ROPE_LAYERS = (0, 1, 40, 41, 42)

# Hyper-Connections. Two collapse points a block, 40 backbone blocks.
COLLAPSE_POINTS = 80
COLLAPSED_KIB_PER_TOKEN = 800
EXPANDED_KIB_PER_TOKEN = 3200

# What a paged latent-KV pool costs at bf16 with 128-aligned halves.
PADDED_LATENT_SLOT = 1280
PADDED_INDEX_SLOT = 256
PADDED_ALL_LAYERS = 61_440
PADDED_FOUR_OWNERS = 6_144


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
        """A call that has to return. A raise counts as one failure, not a crash."""
        try:
            fn()
        except ValueError as exc:
            self.check(False, label, str(exc))
            return
        self.check(True, label)

    def equals(self, label: str, fn, want) -> None:
        """`fn()` has to return `want`. A raise counts as one failure."""
        try:
            got = fn()
        except Exception as exc:  # noqa: BLE001
            self.check(False, label, f"raised {type(exc).__name__}: {exc}")
            return
        self.check(got == want, label, f"{got}")

    def control(self, label: str, expect: str, fn) -> None:
        """A control must raise ValueError, and name the fault it was built for.

        `expect` is a fragment of the message the check responsible for this
        corruption emits. Without it two controls can trip the same check and
        read as two detectors, or a control can fire for a reason that has
        nothing to do with what it claims to test.
        """
        try:
            fn()
        except ValueError as exc:
            if expect in str(exc):
                print(f"      control ({label}): rejected -> {exc}")
                return
            print(f"      control ({label}): WRONG CHECK -> {exc}")
            print(f"      FAIL: expected a message carrying {expect!r}.")
            self.failures += 1
            return
        print(f"      control ({label}): ACCEPTED")
        print("      FAIL: the control did not fail, so this test detects nothing.")
        self.failures += 1


# --- Reference 3: the bucket rule, run independently. ------------------------


def is_prime(n: int) -> bool:
    """Trial division. The primes here sit near 1.6e7, so this is fast enough."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def bucket_primes(count: int, hash_vocab_size: int) -> list[int]:
    """The first `count` primes at or above `hash_vocab_size`."""
    primes: list[int] = []
    candidate = hash_vocab_size - 1
    while len(primes) < count:
        candidate += 1
        if is_prime(candidate):
            primes.append(candidate)
    return primes


def derive_engram_rows(
    num_tables: int, max_ngram_size: int, n_heads: int, hash_vocab_size: int
) -> list[int]:
    """Row count of each table, from the published bucket rule alone.

    A position is hashed as the 2-gram through the `max_ngram_size`-gram ending
    there, each split over `n_heads` heads. Every (n-gram size, head) pair takes
    the next prime at or above `hash_vocab_size` that no earlier pair took, and
    owns that many rows. A table holds one row per bucket of its own pairs.

    This reads nothing from `deepseek_v4_layers`.
    """
    pairs = (max_ngram_size - 1) * n_heads
    primes = bucket_primes(pairs * num_tables, hash_vocab_size)
    return [sum(primes[t * pairs : (t + 1) * pairs]) for t in range(num_tables)]


# --- Checks ------------------------------------------------------------------


def check_split(report: Report) -> None:
    print("Causal encoder-decoder split")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    split = stage_split(cfg)

    report.check(
        "text_config" in cfg and "num_hidden_layers" not in cfg,
        "the config under test is the nested file the checkpoint ships",
        f"{sorted(cfg)}",
    )
    report.check(
        split.num_backbone_layers == BACKBONE_LAYERS,
        "backbone layers",
        f"{split.num_backbone_layers}",
    )
    report.check(
        split.num_draft_layers == DRAFT_LAYERS,
        "draft blocks past the backbone",
        f"{split.num_draft_layers}",
    )
    report.check(
        len(compress_ratios(cfg)) == BLOCKS,
        "blocks the engine builds",
        f"{len(compress_ratios(cfg))}, draft blocks included",
    )
    report.check(
        split.decoder_start == DECODER_START,
        "the decoder starts at",
        f"layer {split.decoder_start}",
    )
    report.check(
        len(split.encoder) == ENCODER_LAYERS and len(split.decoder) == DECODER_LAYERS,
        "the split is even",
        f"{len(split.encoder)} encoder, {len(split.decoder)} decoder",
    )
    report.expect_ok(
        "one decoder KV source, at the decoder's first layer",
        lambda: check_causal_encoder_decoder(cfg),
    )

    plans = layer_plans(cfg)
    stages = [p.stage for p in plans]
    report.check(
        stages.count(ENCODER) == ENCODER_LAYERS
        and stages.count(DECODER) == DECODER_LAYERS
        and stages.count(DRAFT) == DRAFT_LAYERS,
        "every block lands in one stage",
        f"{stages.count(ENCODER)}/{stages.count(DECODER)}/{stages.count(DRAFT)}",
    )
    print()


def check_modes(report: Report) -> None:
    print("CSA2 modes and their sources")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    plans = layer_plans(cfg)

    modes = [p.mode for p in plans]
    report.check(modes == EXPECTED_MODES, "the mode of every block", _mode_summary(modes))

    counts = {m: modes.count(m) for m in sorted(set(modes))}
    report.check(
        counts == {FULL: 4, REINDEX: 4, REUSE: 30, WINDOW: 5},
        "mode counts",
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )

    owners = [p.index for p in plans if p.owns_compressed_kv]
    report.check(owners == KV_SOURCES, "layers that own compressed KV", f"{owners}")
    report.check(
        all(plans[s].mode == FULL for s in KV_SOURCES),
        "every kv_source_layer_id resolves to a Full-mode layer",
        f"{[plans[s].mode for s in KV_SOURCES]}",
    )
    indexers = [p.index for p in plans if p.owns_topk]
    report.check(indexers == INDEX_SOURCES, "layers that run an indexer", f"{indexers}")

    # A source has to bind to itself, or the wiring has an off-by-one that the
    # mode labels alone would hide.
    report.check(
        all(plans[s].kv_source == s for s in KV_SOURCES)
        and all(plans[s].index_source == s for s in INDEX_SOURCES),
        "every source binds to itself",
    )

    # Every consumer binds backward, never forward, and never across the ratio
    # change at the encoder-decoder boundary.
    bad = [
        p.index
        for p in plans
        if p.kv_source is not None
        and not (p.kv_source <= p.index and p.compress_ratio == plans[p.kv_source].compress_ratio)
    ]
    report.check(not bad, "every binding runs backward at one ratio", f"{bad or 'all 38'}")

    # The 30 Reuse layers are 70 percent of the stack and a port resolves three
    # fields on each of them. Check all three, not just the owners.
    reuse = [p for p in plans if p.mode == REUSE]
    expected_reuse = {
        p.index: (
            max(s for s in KV_SOURCES if s <= p.index),
            max(s for s in INDEX_SOURCES if s <= p.index),
            None,
        )
        for p in reuse
    }
    got_reuse = {p.index: (p.kv_source, p.index_source, p.candidate_source) for p in reuse}
    report.check(
        got_reuse == expected_reuse,
        "every Reuse layer's KV source, index source and candidate source",
        f"{len(reuse)} layers"
        if got_reuse == expected_reuse
        else f"{ {k: v for k, v in got_reuse.items() if v != expected_reuse[k]} }",
    )
    # A Reuse layer reads a selection somebody else already narrowed, so it
    # binds to no candidate pool of its own. Drop that term from the parser and
    # all 30 report layer 20 here.
    report.check(
        all(p.candidate_source is None for p in reuse),
        "no Reuse layer claims a candidate pool",
        f"{sorted({p.candidate_source for p in reuse})}",
    )

    ratios = {p.index: p.compress_ratio for p in plans}
    report.check(
        [ratios[i] for i in range(0, 2)] == [0, 0],
        "the first two layers pool nothing and run a sliding window only",
    )
    report.check(
        [ratios[i] for i in range(2, 20)] == [2] * 18,
        "layers 2 through 19 pool two tokens into one latent",
    )
    report.check(
        [ratios[i] for i in range(20, 40)] == [1] * 20,
        "the decoder keeps one latent a token",
    )

    pooling = [p.index for p in plans if p.pools_kv]
    report.check(
        pooling == POOLING_LAYERS,
        "layers holding a partial compression group between decode steps",
        f"{pooling}",
    )

    pool_builders = [p.index for p in plans if p.is_candidate_source]
    report.check(
        pool_builders == [CANDIDATE_LAYER],
        "the layer that builds the candidate pool",
        f"{pool_builders}",
    )
    narrowed = [
        p.index for p in plans if p.owns_topk and p.candidate_source is not None and not p.is_candidate_source
    ]
    report.check(
        narrowed == [24, 28, 32, 36],
        "indexers held inside that pool, all in the decoder",
        f"{narrowed}",
    )
    free = [p.index for p in plans if p.owns_topk and p.candidate_source is None]
    report.check(
        free == [2, 8, 14],
        "encoder indexers score every compressed position",
        f"{free}",
    )

    engram = [p.index for p in plans if p.engram_table is not None]
    report.check(engram == ENGRAM_LAYERS, "layers carrying an Engram table", f"{engram}")
    # `engram_table` is the slot a port indexes the table list with. The two
    # tables differ in row count, so a slot off by one reads the wrong table.
    slots = {p.index: p.engram_table for p in plans if p.engram_table is not None}
    report.check(
        all(ENGRAM_LAYERS[slot] == layer for layer, slot in slots.items()),
        "each table slot indexes back to its own layer",
        f"{slots}",
    )

    plan = cache_plan(cfg)
    report.check(
        len(plan["window"]) == BLOCKS,
        "every block holds a sliding-window ring",
        f"{len(plan['window'])}",
    )
    report.check(
        plan["compressed_kv"] == KV_SOURCES,
        "growing KV caches behind 40 layers",
        f"{len(plan['compressed_kv'])}",
    )
    print()


def _mode_summary(modes) -> str:
    letters = {WINDOW: "w", FULL: "F", REINDEX: "X", REUSE: "r"}
    return "".join(letters[m] for m in modes)


def check_tiling(report: Report) -> None:
    print("Scan grouping")
    cfg = DEEPSEEK_V41_FLASH_CONFIG

    got_units = units(cfg)
    report.expect_ok(
        "units tile the 43 blocks once each",
        lambda: check_cover(got_units, BLOCKS),
    )
    covered = [i for u in got_units for i in u.indices]
    report.check(
        covered == list(range(BLOCKS)),
        "concatenating unit indices rebuilds the stack",
        f"{len(covered)} indices",
    )

    # A WINDOW block shares nothing, so it can never sit inside another block's
    # unit. Fold one in and its weights get stacked with layers that carry a
    # compressor.
    window_units = [u for u in got_units if u.modes[0] == WINDOW]
    report.check(
        [u.start for u in window_units] == [0, 1, 40, 41, 42]
        and all(u.length == 1 for u in window_units),
        "every window block is a unit of one",
        f"{[u.start for u in window_units]}",
    )

    runs = unit_runs(cfg)
    report.expect_ok(
        "runs tile the 43 blocks once each", lambda: check_cover(runs, BLOCKS)
    )
    got = [(r.stage, r.modes, r.engram, r.start, r.repeats) for r in runs]
    report.check(got == EXPECTED_RUNS, "the runs themselves", f"{len(got)} runs")
    report.check(
        len({r.key for r in runs}) == SCAN_BODIES,
        "distinct scan bodies XLA compiles",
        f"{len({r.key for r in runs})} bodies over {BLOCKS} blocks",
    )
    report.check(
        sum(r.length for r in runs) == BLOCKS,
        "run lengths add up",
        f"{sum(r.length for r in runs)}",
    )

    # The three draft blocks share a stage, a mode and a ratio. Only the weight
    # sets tell them apart, so a run key blind to weights folds them into one
    # scan over stacked parameters that don't exist.
    draft = [r for r in runs if r.stage == DRAFT]
    report.check(
        len(draft) == DRAFT_LAYERS and len({r.key for r in draft}) == DRAFT_LAYERS,
        "each draft block gets its own scan body",
        f"{[r.start for r in draft]}",
    )
    report.check(
        len({r.modes for r in draft}) == 1 and len({r.weights for r in draft}) == 3,
        "and it's the weight sets that separate them, not the modes",
        f"{[len(w[0]) for w in (r.weights for r in draft)]} tensors",
    )

    # The decoder's repeated unit is the one that pays for itself: four
    # identical (Reindex, Reuse, Reuse, Reuse) units, one scan of four steps.
    widest = max(runs, key=lambda r: r.repeats)
    report.check(
        (widest.start, widest.repeats, widest.unit_length) == (24, 4, 4),
        "the widest run",
        f"{widest.repeats} units of {widest.unit_length} from layer {widest.start}",
    )

    # Layer 20 builds the candidate pool and every opener from 24 reads it.
    # The roles ride in the run key, so a builder never shares a body with
    # the layers that index inside its pool.
    roles = {r.start: r.pool_roles[0] for r in runs}
    report.check(
        roles == {start: POOL_ROLES.get(start) for start in roles},
        "the pool role each run opens with",
        f"{ {s: r for s, r in roles.items() if r} }",
    )
    moved = _broken(candidate_source_layer_id=24)
    report.equals(
        "with the pool built at 24, the builder leaves the readers' run",
        lambda: [(r.start, r.repeats, r.pool_roles[0]) for r in unit_runs(moved) if r.start >= 20 and r.stage == DECODER],
        [(20, 1, None), (24, 1, POOL_BUILDS), (28, 3, POOL_READS)],
    )

    # A Reuse layer after a window block still binds to its owner, the way
    # layer_plans resolves it, and opens a unit of its own.
    ratios = list(text_config(cfg)["compress_ratios"])
    ratios[5] = 0
    gap = _broken(compress_ratios=ratios)
    report.equals(
        "a Reuse layer after a window block opens its own unit, bound to its owner",
        lambda: (
            [(u.start, u.modes) for u in units(gap) if 2 <= u.start <= 6],
            (layer_plans(gap)[6].kv_source, layer_plans(gap)[6].index_source),
        ),
        ([(2, (FULL, REUSE, REUSE)), (5, (WINDOW,)), (6, (REUSE, REUSE))], (2, 2)),
    )
    report.expect_ok(
        "and its units still tile the 43 blocks",
        lambda: check_cover(units(gap), BLOCKS),
    )
    print()


def check_checkpoint_tensors(report: Report) -> None:
    print("Tensors the plan predicts, against the shipped checkpoint")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    text = text_config(cfg)
    plans = layer_plans(cfg)

    got = {p.checkpoint_prefix: p.extra_weights for p in plans}
    want = {k: tuple(sorted(v)) for k, v in CHECKPOINT_TENSORS.items()}
    wrong = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    unexpected = sorted(set(got) - set(want))
    report.check(
        not wrong and not unexpected,
        f"per-block tensor sets for all {len(want)} blocks",
        f"{wrong or unexpected or 'every block matches, prefix included'}",
    )

    # The namespace itself. A loader that builds keys from the layer index
    # resolves nothing for three of the 43 blocks.
    draft_prefixes = [p.checkpoint_prefix for p in plans if p.stage == DRAFT]
    report.check(
        draft_prefixes == ["mtp.0.", "mtp.1.", "mtp.2."],
        "the draft blocks live outside the layers namespace",
        f"{draft_prefixes}",
    )

    # Four of the mode-specific tensors ship fp8 with an e8m0 scale beside
    # them and the rest are bf16. The suffix doesn't say which, so a loader
    # that strips `.scale` and re-adds it by rule reads nine layers wrong.
    scaled = sorted(
        {n for p in plans for n in p.extra_weights if n.endswith(".scale")}
    )
    report.check(
        scaled
        == [
            "attn.indexer.wq_b.scale",
            "engram.embed.scale",
            "engram.wkv.scale",
            "main_proj.scale",
        ],
        "the scale siblings the plan names",
        f"{len(scaled)} of them",
    )
    missing_scale = [
        p.index
        for p in plans
        for n in p.extra_weights
        if n.endswith(".weight")
        and n.removesuffix(".weight") + ".scale" in want[p.checkpoint_prefix]
        and n.removesuffix(".weight") + ".scale" not in p.extra_weights
    ]
    report.check(
        not missing_scale,
        "and no quantized weight is named without one",
        f"{missing_scale or 'none missing'}",
    )

    # The mode-specific set never restates a tensor every block already
    # carries, or `extra_weights` would double-count the baseline.
    collision = sorted(
        {n for p in plans for n in p.extra_weights} & set(PLAIN_BLOCK)
    )
    report.check(
        not collision,
        "the mode-specific tensors are disjoint from the plain block",
        f"{collision or f'{len(PLAIN_BLOCK)} baseline tensors untouched'}",
    )

    # The pooling gate is the sharpest single signal. It exists at ratio 2 and
    # not at ratio 1, so it separates the encoder's KV owners from the
    # decoder's even though both are Full mode.
    gated = [
        p.index for p in plans if "attn.compressor.wgate.weight" in p.extra_weights
    ]
    report.check(
        gated == POOLING_LAYERS,
        "the pooling gate appears only at ratio 2",
        f"{gated}",
    )

    # No `model.` or `language_model.` prefix anywhere at the root, so a key
    # map written against that convention resolves nothing.
    report.check(
        not [k for k in ROOT_TENSORS if k.startswith(("model.", "language_model."))],
        "the root tensors carry no model prefix",
        f"{len(ROOT_TENSORS)} keys, first {ROOT_TENSORS[0]!r}",
    )

    # Shapes the config has to reproduce. These are the tensors no upstream
    # key map recognizes: LoRA-factored q, a grouped output projection, a
    # per-head sink and the dual router bias.
    heads = text["num_attention_heads"]
    hidden = text["hidden_size"]
    head_dim = text["head_dim"]
    hc = text["hc_mult"]
    derived = {
        "attn.wq_a.weight": (text["q_lora_rank"], hidden),
        "attn.wq_b.weight": (heads * head_dim, text["q_lora_rank"]),
        "attn.wkv.weight": (head_dim, hidden),
        "attn.wo_a.weight": (
            text["o_groups"] * text["o_lora_rank"],
            heads * head_dim // text["o_groups"],
        ),
        "attn.wo_b.weight": (hidden, text["o_groups"] * text["o_lora_rank"]),
        "attn.attn_sink": (heads,),
        "hc_attn_fn": ((2 + hc) * hc, hc * hidden),
        "ffn.gate.bias": (text["n_routed_experts"],),
        "attn.compressor.wkv.weight": (head_dim, hidden),
        "attn.indexer.wq_b.weight": (
            text["index_n_heads"] * text["index_head_dim"],
            text["q_lora_rank"],
        ),
        "attn.indexer.wk.weight": (text["index_head_dim"], head_dim),
        "attn.indexer.weights_proj.weight": (text["index_n_heads"], hidden),
        "engram.wkv.weight": (
            hidden * (hc + 1),
            (text["engram_max_ngram_size"] - 1)
            * text["engram_n_heads"]
            * text["engram_head_dim"],
        ),
        "main_proj.weight": (hidden, hidden * len(text["dspark_target_layer_ids"])),
        "markov_head.embed.weight": (
            text["vocab_size"],
            text["dspark_markov_rank"],
        ),
        "confidence_head.proj.weight": (1, hidden + text["dspark_markov_rank"]),
    }
    bad_shapes = {k: (v, CHECKPOINT_SHAPES[k]) for k, v in derived.items() if v != CHECKPOINT_SHAPES[k]}
    report.check(
        not bad_shapes,
        f"{len(derived)} tensor shapes rebuilt from the config",
        f"{bad_shapes or 'every shape matches the shard'}",
    )
    print()


def check_rotary(report: Report) -> None:
    print("Rotary positions")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    tables = rotary_tables(cfg)

    report.check(len(tables) == 2, "tables the stack reads", f"{len(tables)}")
    yarn, plain = tables
    report.check(
        (yarn.theta, yarn.yarn, yarn.original_max_position_embeddings)
        == (160000, True, 65536)
        and yarn.yarn_factor == 16,
        "the compressed table",
        f"theta {yarn.theta}, YaRN factor {yarn.yarn_factor} over "
        f"{yarn.original_max_position_embeddings}",
    )
    report.check(
        (plain.theta, plain.yarn) == (10000, False),
        "the sliding-window table, with YaRN off",
        f"theta {plain.theta}",
    )
    report.check(
        yarn.layers == YARN_LAYERS and plain.layers == PLAIN_ROPE_LAYERS,
        "which blocks read which",
        f"{len(yarn.layers)} compressed, {len(plain.layers)} window-only",
    )
    # The split is per layer, not per tensor. One table rotates that layer's
    # queries, its sliding-window K and its compressed latent alike, so the
    # window KV of layers 2 through 39 is rotated at 160,000 with YaRN and the
    # window KV of the other five at 10,000 with none.
    report.check(
        set(yarn.layers) | set(plain.layers) == set(range(BLOCKS))
        and not set(yarn.layers) & set(plain.layers),
        "every block reads one table and no block reads two",
        f"{len(yarn.layers) + len(plain.layers)} blocks",
    )
    report.check(
        yarn.rotated_channels == ROTATED_CHANNELS
        and yarn.unrotated_channels == 448
        and plain.rotated_channels == ROTATED_CHANNELS,
        "channels the rotation touches",
        f"{yarn.rotated_channels} of {yarn.head_dim}, "
        f"{yarn.unrotated_channels} left alone",
    )
    print()


def check_residual_stream(report: Report) -> None:
    print("Hyper-Connections residual stream")
    stream = residual_stream(DEEPSEEK_V41_FLASH_CONFIG)

    report.check(stream.copies == 4, "parallel copies", f"{stream.copies}")
    report.check(
        stream.collapses_per_block == 2 and stream.collapse_points == COLLAPSE_POINTS,
        "collapse points a capture hook can hang on",
        f"{stream.collapses_per_block} a block, {stream.collapse_points} over "
        f"{stream.blocks}",
    )
    report.check(
        stream.collapsed_bytes_per_token() // 1024 == COLLAPSED_KIB_PER_TOKEN,
        "capturing every collapsed sublayer input",
        f"{stream.collapsed_bytes_per_token() // 1024} KiB a token",
    )
    report.check(
        stream.expanded_bytes_per_token() // 1024 == EXPANDED_KIB_PER_TOKEN,
        "capturing all four copies at both points",
        f"{stream.expanded_bytes_per_token() // 1024} KiB a token",
    )
    print()


def check_engram(report: Report) -> None:
    print("Engram")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    plan = engram_plan(cfg)

    report.check(
        [t.layer_id for t in plan.tables] == ENGRAM_LAYERS,
        "tables and the layers that read them",
        f"{[t.layer_id for t in plan.tables]}",
    )

    # Reference 3. The rule, run here with its own sieve, has to land on the
    # row counts the config declares.
    derived = derive_engram_rows(
        len(plan.tables), plan.max_ngram_size, plan.n_heads, plan.hash_vocab_size
    )
    report.check(
        derived == [t.rows for t in plan.tables],
        "row counts derived from the bucket rule",
        f"{derived}",
    )

    report.check(
        all(t.row_dim == 256 for t in plan.tables),
        "row width",
        f"{plan.tables[0].row_dim} values",
    )
    report.check(
        all(t.row_bytes == ENGRAM_ROW_BYTES for t in plan.tables),
        "bytes a row",
        f"{plan.tables[0].row_bytes} = 256 fp8 plus 8 e8m0 scales",
    )
    report.check(
        tuple(t.bytes for t in plan.tables) == ENGRAM_TABLE_BYTES,
        "bytes a table",
        f"{[round(gib(t.bytes), 2) for t in plan.tables]} GiB",
    )

    # Reference 2. The two Engram shards hold one table plus four small
    # tensors, so the plan's byte total plus that sidecar plus the safetensors
    # header has to reproduce the file size.
    rebuilt = {
        t.layer_id: t.bytes
        + ENGRAM_SHARD_SIDECAR_BYTES
        + 8
        + ENGRAM_SHARD_HEADER_BYTES[t.layer_id]
        for t in plan.tables
    }
    report.check(
        rebuilt == ENGRAM_SHARD_BYTES,
        "and the shard sizes those bytes rebuild",
        f"{rebuilt}" if rebuilt != ENGRAM_SHARD_BYTES else "both to the byte",
    )

    report.check(
        plan.params == ENGRAM_PARAMS
        and plan.params // 10**9 == ENGRAM_PARAMS_BILLIONS,
        "parameters, and the model card's figure they truncate to",
        f"{plan.params} = {plan.params / 1e9:.1f}B",
    )
    report.check(
        plan.bytes == ENGRAM_BYTES,
        "packed table bytes",
        f"{gib(plan.bytes):.2f} GiB",
    )
    # A loader that widens the table on the way in pays this instead. The
    # reference runtime keeps it packed and dequantizes a row on lookup.
    report.check(
        plan.dequantized_bytes() == plan.params * 2,
        "and what they cost widened to BF16",
        f"{gib(plan.dequantized_bytes()):.1f} GiB against "
        f"{gib(plan.bytes):.1f} GiB packed",
    )

    report.check(
        plan.hash_columns == ENGRAM_HASH_COLUMNS,
        "rows a token reads from one table",
        f"{plan.hash_columns} = 3 n-gram sizes by 8 heads",
    )
    report.check(
        plan.rows_per_token == ENGRAM_ROWS_PER_TOKEN,
        "rows a token reads in total",
        f"{plan.rows_per_token}",
    )
    report.check(
        plan.gather_bytes_per_token == ENGRAM_GATHER_BYTES_PER_TOKEN,
        "packed bytes a token gathers",
        f"{plan.gather_bytes_per_token}",
    )
    report.check(
        plan.payload_bytes_per_token == ENGRAM_PAYLOAD_BYTES_PER_TOKEN,
        "dequantized bytes a token puts on the wire",
        f"{plan.payload_bytes_per_token} at BF16",
    )

    embed = {t.layer_id: (t.rows, t.row_dim) for t in plan.tables}
    report.check(
        embed == ENGRAM_EMBED_SHAPES,
        "the embedding table shapes the checkpoint declares",
        f"{embed}",
    )
    scales = {
        t.layer_id: (t.rows, t.row_dim // t.scale_group) for t in plan.tables
    }
    report.check(
        scales == ENGRAM_SCALE_SHAPES,
        "the scale table shapes the checkpoint declares",
        f"{scales}",
    )
    print()


def one_link_stall(host_layers, tokens, link_bytes_per_s, step_seconds, blocks=BACKBONE_LAYERS):
    """Stall a host fetch adds to one step, walked block by block.

    Written a second way, from the literals above: one per-chip link, the
    fetches issued back to back in layer order when the step starts, and each
    block computing for `step_seconds / blocks`. A block that reads a hosted
    table waits first until that table's rows have landed.
    """
    fetch = tokens * ENGRAM_ROW_BYTES * ENGRAM_HASH_COLUMNS / link_bytes_per_s
    landed, elapsed = {}, 0.0
    for layer in sorted(host_layers):
        elapsed += fetch
        landed[layer] = elapsed
    clock = stall = 0.0
    for block in range(blocks):
        if block in landed and landed[block] > clock:
            stall += landed[block] - clock
            clock = landed[block]
        clock += step_seconds / blocks
    return stall


def check_placement(report: Report) -> None:
    print("Engram placement, on a 32-chip v5p slice")
    plan = engram_plan(DEEPSEEK_V41_FLASH_CONFIG)

    def cost(host_layers, tokens, seconds):
        return plan.placement(
            host_layers,
            tokens=tokens,
            devices=CHIPS,
            ici_bytes_per_s=ICI_BYTES_PER_S,
            link_bytes_per_s=HOST_LINK_BYTES_PER_S,
            step_seconds=seconds,
            num_blocks=BACKBONE_LAYERS,
        )

    sharded = cost((), PREFILL_TOKENS, PREFILL_SECONDS)
    report.check(
        abs(gib(sharded.bytes_per_device) - 5.90) < 0.01,
        "sharded across 32 chips costs per chip",
        f"{gib(sharded.bytes_per_device):.2f} GiB, "
        f"{100 * sharded.bytes_per_device / HBM_BYTES_PER_CHIP:.1f}% of HBM",
    )
    report.check(
        abs(sharded.ici_seconds * 1e6 - 325.1) < 1.0,
        "and on ICI for one 8,192-token chunk",
        f"{sharded.ici_seconds * 1e6:.0f} us against a {PREFILL_SECONDS * 1e3:.1f} ms chunk",
    )
    report.check(
        sharded.exposed_host_seconds == 0.0,
        "sharded placement touches the host link",
        "never",
    )

    hosted = cost(None, PREFILL_TOKENS, PREFILL_SECONDS)
    report.check(
        hosted.bytes_per_device == 0.0,
        "host residency costs HBM",
        "nothing",
    )
    report.check(
        abs(hosted.exposed_host_seconds * 1e3 - 1.40) < 0.02,
        "and leaves on the prefill critical path",
        f"{hosted.exposed_host_seconds * 1e3:.2f} ms, "
        f"{100 * hosted.exposed_host_seconds / PREFILL_SECONDS:.0f}% of the chunk",
    )

    hosted_decode = cost(None, DECODE_TOKENS, DECODE_SECONDS)
    report.check(
        abs(hosted_decode.exposed_host_seconds * 1e6 - 41.6) < 1.0,
        "and on a 256-token decode step",
        f"{hosted_decode.exposed_host_seconds * 1e6:.0f} us against a "
        f"{DECODE_SECONDS * 1e6:.0f} us step",
    )

    # Both tables share one host link. With 800 tokens in a decode step the
    # layer-1 fetch still holds the link when the layer-14 fetch needs it, and
    # the stall comes to 190.2 us. A link for each table would put it at
    # 181.2 us, below what any schedule on one link reaches.
    for label, tokens, link, seconds, want in (
        ("800 tokens at the decode step", 800, HOST_LINK_BYTES_PER_S, DECODE_SECONDS, 190.2e-6),
        ("an 8,192-token chunk at 16 GB/s", PREFILL_TOKENS, 16e9, PREFILL_SECONDS, 3.367e-3),
    ):
        got = plan.exposed_host_seconds(tokens, link, seconds, BACKBONE_LAYERS)
        walked = one_link_stall((1, 14), tokens, link, seconds)
        report.check(
            abs(got - walked) < 1e-12 and abs(got - want) < 0.001 * want,
            f"both tables hosted, {label}, stall on one shared link",
            f"{got * 1e6:.1f} us, walked block by block {walked * 1e6:.1f} us",
        )

    # Layer 1 runs one block in. Layer 14 runs fourteen blocks in, so the same
    # fetch has fourteen times the cover. Splitting the tables on that is the
    # placement the plan recommends.
    split_prefill = cost((14,), PREFILL_TOKENS, PREFILL_SECONDS)
    split_decode = cost((14,), DECODE_TOKENS, DECODE_SECONDS)
    report.check(
        abs(gib(split_prefill.bytes_per_device) - 2.95) < 0.01,
        "layer 1 on chip and layer 14 on the host costs per chip",
        f"{gib(split_prefill.bytes_per_device):.2f} GiB",
    )
    report.check(
        split_prefill.exposed_host_seconds == 0.0
        and split_decode.exposed_host_seconds == 0.0,
        "and leaves on the critical path, at prefill and at decode",
        "nothing",
    )
    report.check(
        split_prefill.sharded_layers == (1,) and split_prefill.host_layers == (14,),
        "the split it describes",
        f"chip {split_prefill.sharded_layers}, host {split_prefill.host_layers}",
    )
    # One table on chip instead of two, so the all-reduce carries half the
    # payload. A placement that costs ICI for the hosted table too would land
    # on 325 us here.
    report.check(
        abs(split_prefill.ici_seconds * 1e6 - 162.5) < 1.0,
        "and puts half the all-sharded payload on ICI",
        f"{split_prefill.ici_seconds * 1e6:.0f} us against "
        f"{sharded.ici_seconds * 1e6:.0f} us for both tables",
    )

    # `placement` costs a subset of the tables. The three methods below cost
    # every table, and they're the same arithmetic, so the all-sharded
    # placement has to land on them. Without this the two could drift apart.
    report.check(
        plan.bytes_per_device(CHIPS) == sharded.bytes_per_device,
        "bytes_per_device agrees with the all-sharded placement",
        f"{gib(plan.bytes_per_device(CHIPS)):.2f} GiB",
    )
    direct = plan.allreduce_bytes_per_device(PREFILL_TOKENS, CHIPS) / ICI_BYTES_PER_S
    report.check(
        direct == sharded.ici_seconds,
        "allreduce_bytes_per_device agrees with it too",
        f"{direct * 1e6:.0f} us",
    )
    report.check(
        plan.sharded_seconds(PREFILL_TOKENS, CHIPS, ICI_BYTES_PER_S) == direct,
        "sharded_seconds is that figure over the ICI rate",
    )
    # One chip has nothing to combine, so the all-reduce is free.
    report.check(
        plan.allreduce_bytes_per_device(PREFILL_TOKENS, 1) == 0.0
        and plan.placement((), PREFILL_TOKENS, 1, ICI_BYTES_PER_S,
                           HOST_LINK_BYTES_PER_S, PREFILL_SECONDS,
                           BACKBONE_LAYERS).ici_seconds == 0.0,
        "one chip pays no interconnect",
    )

    # The whole checkpoint, engram included, against one chip's HBM.
    backbone_bf16 = 552e9 * 2
    per_chip = (backbone_bf16 + plan.bytes) / CHIPS
    report.check(
        per_chip < HBM_BYTES_PER_CHIP,
        "backbone at BF16 plus the packed tables, per chip",
        f"{gib(per_chip):.1f} GiB of {gib(HBM_BYTES_PER_CHIP):.0f} GiB",
    )
    print()


def check_kv_budget(report: Report) -> None:
    print("KV budget")
    cfg = DEEPSEEK_V41_FLASH_CONFIG
    text = text_config(cfg)
    quant = KVQuantization()

    report.check(
        text["num_key_value_heads"] == 1,
        "KV heads a layer writes",
        "one 512-wide latent, shared by all 64 query heads",
    )
    report.check(
        quant.latent_bytes(text["head_dim"]) == 288,
        "a compressed KV latent",
        f"{quant.latent_bytes(text['head_dim'])} bytes, "
        "512 fp4 values plus 32 e4m3 scales",
    )
    report.check(
        quant.index_key_bytes(text["index_head_dim"]) == 68,
        "an index key",
        f"{quant.index_key_bytes(text['index_head_dim'])} bytes",
    )
    report.check(
        quant.window_slot_bytes(text["head_dim"]) == 528,
        "a sliding-window slot",
        f"{quant.window_slot_bytes(text['head_dim'])} bytes",
    )

    per_token = global_kv_bytes_per_token(cfg)
    report.check(
        per_token["total"] == GLOBAL_KV_BYTES_PER_TOKEN,
        "global KV a token adds, against the model card's 890",
        f"{per_token['total']:.0f} bytes",
    )
    report.check(
        [per_token[str(i)] for i in KV_SOURCES] == [178.0, 178.0, 178.0, 356.0],
        "split across the four owners",
        "three encoder owners at 178, one decoder owner at 356",
    )

    from deepseek_v4_layers import window_kv_bytes_per_sequence

    window = window_kv_bytes_per_sequence(cfg)
    report.check(
        window == WINDOW_KV_BYTES_PER_SEQUENCE,
        "sliding-window KV a sequence, at any length",
        f"{window / 2**20:.2f} MiB",
    )

    # What the same four caches cost in a paged pool with no FP4 path: one
    # bf16 slot a layer, each half padded to 128. The default layer count is
    # the whole stack, because nothing in the config tells a pool that four
    # layers own the caches.
    padded = padded_bf16_kv_bytes_per_token(cfg)
    owners_only = padded_bf16_kv_bytes_per_token(cfg, len(KV_SOURCES), len(KV_SOURCES))
    report.check(
        (padded["latent_slot"], padded["index_slot"])
        == (PADDED_LATENT_SLOT, PADDED_INDEX_SLOT),
        "a bf16 128-aligned latent slot and index-key slot",
        f"{padded['latent_slot']} and {padded['index_slot']} bytes",
    )
    report.check(
        padded["total"] == PADDED_ALL_LAYERS
        and owners_only["total"] == PADDED_FOUR_OWNERS,
        "one slot a layer against one slot an owner",
        f"{padded['total']} bytes a token over 40 layers, "
        f"{owners_only['total']} over 4",
    )
    report.check(
        owners_only["total"] / per_token["total"] > 6,
        "and how far either sits above the shipped figure",
        f"{owners_only['total'] / per_token['total']:.1f}x at four caches, "
        f"{padded['total'] / per_token['total']:.0f}x at forty",
    )

    # 128 sequences at the full context, against one slice.
    concurrent = 128
    slice_bytes = CHIPS * HBM_BYTES_PER_CHIP
    kv = concurrent * (per_token["total"] * CONTEXT + window)
    report.check(
        kv < slice_bytes,
        f"{concurrent} sequences at 1M tokens",
        f"{gib(kv):.0f} GiB of KV, {100 * kv / slice_bytes:.0f}% of the slice",
    )
    print()


# --- Controls ----------------------------------------------------------------


def _broken(**edits) -> dict:
    """The shipped config with `edits` applied to its `text_config` block.

    The edits have to land inside the nested block, because that's where the
    parser reads. Applied to the outer dict they'd be silently ignored and
    every control below would pass for the wrong reason.
    """
    cfg = copy.deepcopy(DEEPSEEK_V41_FLASH_CONFIG)
    cfg["text_config"].update(edits)
    return cfg


def _without(field: str) -> dict:
    cfg = copy.deepcopy(DEEPSEEK_V41_FLASH_CONFIG)
    cfg["text_config"].pop(field)
    return cfg


def check_controls(report: Report) -> None:
    print("Negative controls")
    base = text_config(DEEPSEEK_V41_FLASH_CONFIG)

    # The plan says every kv_source_layer_id is a Full-mode layer. Drop one
    # from the index list and that layer owns a compressor with no indexer, so
    # nothing publishes index keys for the six layers bound to it.
    report.control(
        "a KV source that owns no indexer",
        "own no indexer",
        lambda: layer_plans(
            _broken(index_source_layer_ids=[2, 14, 20, 24, 28, 32, 36])
        ),
    )
    # A layer at ratio 1 bound to a ratio-2 source would read a cache sized and
    # indexed for half as many positions.
    ratios = list(base["compress_ratios"])
    ratios[7] = 1
    report.control(
        "a layer whose ratio disagrees with its KV source",
        "binds to KV source",
        lambda: layer_plans(_broken(compress_ratios=ratios)),
    )
    # An index source at ratio 0 sitting between a consumer and its KV source.
    # The KV binding still agrees, and a ratio-0 layer runs no indexer, so it
    # can't publish the top-k the consumer would read.
    index_hole = list(base["compress_ratios"])
    index_hole[3] = 0
    report.control(
        "an index source at ratio 0",
        "run compress ratio 0, so they run no indexer",
        lambda: layer_plans(
            _broken(
                compress_ratios=index_hole,
                index_source_layer_ids=[2, 3, 8, 14, 20, 24, 28, 32, 36],
            )
        ),
    )
    # The pool built at the ratio-2 encoder layer 2, read by the ratio-1
    # indexers in the decoder. Every KV and index binding still agrees.
    report.control(
        "a candidate pool built at another ratio",
        "indexes inside the candidate pool of layer 2 at ratio 2",
        lambda: layer_plans(_broken(candidate_source_layer_id=2)),
    )
    # This one is internally consistent. Layer 30 owns a compressor and an
    # indexer, every binding runs backward at one ratio, and the ratio boundary
    # is still at 20. It just isn't a causal encoder-decoder any more, because
    # the decoder now projects KV twice.
    report.control(
        "a second KV source inside the decoder",
        "the decoder's KV owners are",
        lambda: check_causal_encoder_decoder(
            _broken(
                kv_source_layer_ids=[2, 8, 14, 20, 30],
                index_source_layer_ids=[2, 8, 14, 20, 24, 28, 30, 32, 36],
            )
        ),
    )
    # One decoder layer that takes no part in compressed attention. The
    # boundary and the single owner both survive, and only the binding check
    # sees that layer 39 reads nothing layer 20 wrote.
    tail = list(base["compress_ratios"])
    tail[39] = 0
    report.control(
        "a decoder layer that binds its KV nowhere",
        "bind their KV outside layer 20",
        lambda: check_causal_encoder_decoder(_broken(compress_ratios=tail)),
    )
    # The first compressed layer has to have a source at or before it.
    report.control(
        "a compressed layer ahead of every source",
        "no source sits at or",
        lambda: layer_plans(
            _broken(
                kv_source_layer_ids=[4, 8, 14, 20],
                index_source_layer_ids=[4, 8, 14, 20, 24, 28, 32, 36],
            )
        ),
    )
    # compress_ratios covers the draft blocks too. A list sized from
    # num_hidden_layers alone is three entries short.
    report.control(
        "compress_ratios sized from num_hidden_layers alone",
        "compress_ratios holds 40 entries",
        lambda: compress_ratios(_broken(compress_ratios=base["compress_ratios"][:40])),
    )
    # A draft block is never a source, so it can't run compressed attention.
    report.control(
        "a draft block asking for compressed attention",
        "asks for compress ratio",
        lambda: stage_split(
            _broken(compress_ratios=base["compress_ratios"][:40] + [1, 0, 0])
        ),
    )
    negative = list(base["compress_ratios"])
    negative[5] = -2
    report.control(
        "a negative compression ratio",
        "negative or non-integer",
        lambda: compress_ratios(_broken(compress_ratios=negative)),
    )
    report.control(
        "source ids out of order",
        "sorted and free of duplicates",
        lambda: layer_plans(_broken(kv_source_layer_ids=[2, 14, 8, 20])),
    )
    report.control(
        "a source id that isn't an integer",
        "non-integer entry",
        lambda: layer_plans(_broken(kv_source_layer_ids=[2, 8.0, 14, 20])),
    )
    report.control(
        "no KV source at all",
        "is empty, so no layer owns",
        lambda: layer_plans(_broken(kv_source_layer_ids=[])),
    )
    report.control(
        "a source past the end of the backbone",
        "outside the 40 backbone layers",
        lambda: layer_plans(_broken(kv_source_layer_ids=[2, 8, 14, 20, 44])),
    )
    # Layer 0 has to sit in both lists, or the orphan check fires first. The
    # index check then refuses it too, and its message also says "run compress
    # ratio 0", so this pins the words only the KV check uses.
    report.control(
        "a KV source at ratio 0",
        "so they compress nothing",
        lambda: layer_plans(
            _broken(
                kv_source_layer_ids=[0, 2, 8, 14, 20],
                index_source_layer_ids=[0, 2, 8, 14, 20, 24, 28, 32, 36],
            )
        ),
    )
    report.control(
        "a candidate source that owns no indexer",
        "can build no candidate pool",
        lambda: layer_plans(_broken(candidate_source_layer_id=21)),
    )
    # The KV budget rests on one shared latent. Nothing downstream contradicts
    # a missing head count, so the field can't be defaulted.
    report.control(
        "a config with no KV head count",
        "config has no num_key_value_heads",
        lambda: global_kv_bytes_per_token(_without("num_key_value_heads")),
    )
    report.control(
        "a rotated tail wider than the head",
        "doesn't fit in a",
        lambda: rotary_tables(_broken(qk_rope_head_dim=1024)),
    )
    report.control(
        "an Engram table with no layer to read it",
        "engram_num_embeddings",
        lambda: engram_plan(_broken(engram_num_embeddings=[384_006_168])),
    )
    report.control(
        "Engram layer ids out of order",
        "sorted and unique",
        lambda: engram_plan(_broken(engram_layer_ids=[14, 1])),
    )
    report.control(
        "an Engram table too small for its bucket ranges",
        "fewer than the",
        lambda: engram_plan(_broken(engram_num_embeddings=[16_000_057, 384_016_682])),
    )
    # The other side of the same rule. 24 ranges of a prime just over 16,000,000
    # leave a few thousand rows over the floor, never a whole extra bucket, so a
    # count that far out means the pairs aren't what the rule says they are.
    report.control(
        "an Engram table a whole bucket larger than its ranges",
        "rows past the",
        lambda: engram_plan(_broken(engram_num_embeddings=[400_000_000, 384_016_682])),
    )
    report.control(
        "an Engram layer past the end of the backbone",
        "outside the 40 backbone layers",
        lambda: engram_plan(_broken(engram_layer_ids=[1, 41])),
    )
    report.control(
        "an n-gram size no n-gram fits in",
        "no n-gram spans two tokens",
        lambda: engram_plan(_broken(engram_max_ngram_size=1)),
    )
    report.control(
        "a row width the scale group doesn't divide",
        "32-element scale group",
        lambda: engram_plan(_broken(engram_head_dim=200)),
    )
    report.control(
        "an empty compressed vocabulary",
        "every hash",
        lambda: engram_plan(_broken(engram_compressed_vocab_size=0)),
    )
    report.control(
        "a host placement naming a layer with no table",
        "carry no Engram table",
        lambda: engram_plan(DEEPSEEK_V41_FLASH_CONFIG).placement(
            (13,), 8192, 32, ICI_BYTES_PER_S, HOST_LINK_BYTES_PER_S, 1e-3, 40
        ),
    )
    # Sharding over no chip has no answer. Returning zero bytes a chip would
    # read as a placement that costs nothing.
    report.control(
        "a sharded placement over zero chips",
        "no chip holds the rows",
        lambda: engram_plan(DEEPSEEK_V41_FLASH_CONFIG).placement(
            (), 8192, 0, ICI_BYTES_PER_S, HOST_LINK_BYTES_PER_S, 1e-3, 40
        ),
    )
    report.control(
        "bytes_per_device over zero chips",
        "no chip holds the rows",
        lambda: engram_plan(DEEPSEEK_V41_FLASH_CONFIG).bytes_per_device(0),
    )

    runs = unit_runs(DEEPSEEK_V41_FLASH_CONFIG)
    report.control(
        "a cover missing its last run",
        "groups cover",
        lambda: check_cover(runs[:-1], BLOCKS),
    )
    report.control(
        "a cover with a gap in the middle",
        "gap of",
        lambda: check_cover([r for r in runs if r.start != 20], BLOCKS),
    )
    report.control(
        "a cover whose runs overlap",
        "overlap of",
        lambda: check_cover(
            [runs[0], dataclasses.replace(runs[1], start=0)] + list(runs[2:]), BLOCKS
        ),
    )
    report.control(
        "a cover holding an empty group",
        "covers 0 blocks",
        lambda: check_cover(
            [Unit(modes=(), engram=(), weights=(), stage=ENCODER, compress_ratio=0, start=0)]
            + list(runs),
            BLOCKS,
        ),
    )

    # The parser reads `text_config`, so a corrupt inner block has to be
    # rejected even when the outer dict carries a clean copy of the same field.
    # A parser that reads the outer dict accepts this.
    shadowed = copy.deepcopy(DEEPSEEK_V41_FLASH_CONFIG)
    shadowed.update(
        {
            "num_hidden_layers": base["num_hidden_layers"],
            "num_nextn_predict_layers": base["num_nextn_predict_layers"],
            "compress_ratios": list(base["compress_ratios"]),
            "kv_source_layer_ids": list(base["kv_source_layer_ids"]),
            "index_source_layer_ids": list(base["index_source_layer_ids"]),
        }
    )
    shadowed["text_config"]["kv_source_layer_ids"] = [2, 14, 8, 20]
    report.control(
        "a clean outer dict shadowing a corrupt text_config",
        "sorted and free of duplicates",
        lambda: layer_plans(shadowed),
    )
    print()


def check_control_sensitivity(report: Report) -> None:
    """Confirm the checks on a well-formed plan react to one thing going wrong.

    The controls above all make the parser raise. These make it return a plan
    that's wrong but well formed, or cost it the wrong way, and confirm the
    comparisons catch it. Without them a comparison that never fires would look
    clean.
    """
    print("Controls on the checks that grade a well-formed plan")
    # Layer 24 was a Reindex layer. Make it Reuse and nothing raises: it just
    # stops carrying the three indexer tensors the shard declares.
    demoted = _broken(index_source_layer_ids=[2, 8, 14, 20, 28, 32, 36])
    plans = layer_plans(demoted)
    report.check(
        plans[24].mode == REUSE,
        "dropping layer 24 from the index sources leaves a valid plan",
        f"layer 24 is now {plans[24].mode}",
    )
    report.check(
        plans[24].extra_weights != tuple(sorted(CHECKPOINT_TENSORS["layers.24."])),
        "and the checkpoint comparison rejects it",
        f"predicts {plans[24].extra_weights}, the shard holds "
        f"{tuple(sorted(CHECKPOINT_TENSORS['layers.24.']))}",
    )

    # Send the draft blocks back into the layers namespace and three of the 43
    # prefixes stop resolving.
    real = layer_plans(DEEPSEEK_V41_FLASH_CONFIG)
    renamed = {f"layers.{p.index}.": p.extra_weights for p in real}
    report.check(
        set(renamed) != set(CHECKPOINT_TENSORS),
        "keying the draft blocks by layer index rejects three prefixes",
        f"{sorted(set(renamed) - set(CHECKPOINT_TENSORS))}",
    )

    # Fold the three draft blocks into one run, the way a key blind to weights
    # would, and the run count drops below the number of distinct bodies.
    blind = {(r.stage, r.compress_ratio, r.modes) for r in unit_runs(DEEPSEEK_V41_FLASH_CONFIG)}
    report.check(
        len(blind) < SCAN_BODIES,
        "and a run key that ignores weights undercounts the bodies",
        f"{len(blind)} against {SCAN_BODIES}",
    )

    # Build the pool at 24 and a key blind to the pool role folds the builder
    # into the run of the layers that index inside its pool.
    moved = unit_runs(_broken(candidate_source_layer_id=24))
    decoder = [r for r in moved if r.stage == DECODER]
    folded = []
    for r in decoder:
        blind_key = (r.stage, r.compress_ratio, r.modes, r.weights)
        if folded and folded[-1] == blind_key:
            continue
        folded.append(blind_key)
    report.check(
        len(folded) < len(decoder),
        "and a run key that ignores the pool role folds the builder into its readers",
        f"{len(folded)} decoder runs against {len(decoder)}",
    )

    # Give each hosted table a link of its own, and the stall at 800 tokens
    # comes out below the one-link figure.
    plan = engram_plan(DEEPSEEK_V41_FLASH_CONFIG)
    per_table = sum(
        max(0.0, 800 * t.row_bytes * plan.hash_columns / HOST_LINK_BYTES_PER_S
            - DECODE_SECONDS * t.layer_id / BACKBONE_LAYERS)
        for t in plan.tables
    )
    shared = plan.exposed_host_seconds(800, HOST_LINK_BYTES_PER_S, DECODE_SECONDS, BACKBONE_LAYERS)
    report.check(
        shared - per_table > 1e-6,
        "and a link for each hosted table undercounts the stall",
        f"{per_table * 1e6:.1f} us against {shared * 1e6:.1f} us on one link",
    )

    # Same idea on the Engram side: move a table one layer and the placement
    # arithmetic moves with it, because cover scales with depth.
    moved = _broken(engram_layer_ids=[1, 2])
    exposed_real = plan.exposed_host_seconds(
        DECODE_TOKENS, HOST_LINK_BYTES_PER_S, DECODE_SECONDS, BACKBONE_LAYERS, [14]
    )
    exposed_moved = engram_plan(moved).exposed_host_seconds(
        DECODE_TOKENS, HOST_LINK_BYTES_PER_S, DECODE_SECONDS, BACKBONE_LAYERS, [2]
    )
    report.check(
        exposed_real == 0.0 and exposed_moved > 0.0,
        "the same table at layer 2 no longer hides behind a prefetch",
        f"{exposed_moved * 1e6:.0f} us exposed at layer 2, {exposed_real * 1e6:.0f} at layer 14",
    )
    print()


# --- Bucket layout -----------------------------------------------------------


def bucket_ranges(sizes: np.ndarray) -> np.ndarray:
    """`[start, stop)` of each bucket range, laid out back to back."""
    offsets = np.concatenate([[0], np.cumsum(sizes[:-1])])
    return np.stack([offsets, offsets + sizes], axis=1)


def range_faults(ranges: np.ndarray, rows: int, hash_vocab: int) -> dict[str, int]:
    """Count the ways a bucket layout can send a hash to the wrong row.

    Returns the number of overlapping range pairs, the number of rows no range
    covers, and the number of probe hashes that leave their own range. A hash
    lands at `start + h % width`, so a range narrower than the hash space folds
    two hashes onto one row and a range that overlaps another reads an n-gram
    from a different (size, head) pair.
    """
    overlaps = 0
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            lo = max(ranges[i, 0], ranges[j, 0])
            hi = min(ranges[i, 1], ranges[j, 1])
            if hi > lo:
                overlaps += 1

    covered = 0
    for lo, hi in sorted(map(tuple, ranges.tolist())):
        lo, hi = max(lo, 0), min(hi, rows)
        if hi > lo:
            covered += hi - lo
    uncovered = rows - covered

    # The top of the hash space is where a narrow range folds first.
    probe = np.arange(hash_vocab - 1024, hash_vocab, dtype=np.int64)
    escaped = 0
    for lo, hi in ranges.tolist():
        landed = lo + probe % (hi - lo)
        escaped += int(np.count_nonzero((landed < lo) | (landed >= hi)))
        escaped += int(np.count_nonzero(landed != lo + probe))
    return {"overlaps": overlaps, "uncovered": uncovered, "escaped": escaped}


def check_bucket_layout(report: Report) -> None:
    """One numeric pass, so the byte figures come from arithmetic on arrays.

    The bucket ranges have to tile the row space with no gap and no overlap,
    because a hash landing in the wrong range reads another n-gram's row and
    the model gets a silently wrong memory.
    """
    print("Bucket layout")
    plan = engram_plan(DEEPSEEK_V41_FLASH_CONFIG)
    pairs = plan.hash_columns
    primes = bucket_primes(pairs * len(plan.tables), plan.hash_vocab_size)

    for table_idx, table in enumerate(plan.tables):
        sizes = np.array(primes[table_idx * pairs : (table_idx + 1) * pairs], dtype=np.int64)
        ranges = bucket_ranges(sizes)
        faults = range_faults(ranges, table.rows, plan.hash_vocab_size)
        report.check(
            int(ranges[-1, 1]) == table.rows,
            f"layer {table.layer_id} bucket ranges close on the table",
            f"{int(ranges[-1, 1])} rows",
        )
        report.check(
            faults == {"overlaps": 0, "uncovered": 0, "escaped": 0},
            f"layer {table.layer_id} ranges tile the rows and hold every hash",
            f"{pairs} ranges, {faults}",
        )

        # The same counter, on a layout that's wrong two ways: one range
        # shifted past its neighbor and one narrower than the hash space.
        broken_ranges = ranges.copy()
        broken_ranges[5, 0] -= 7
        broken_ranges[9, 1] = broken_ranges[9, 0] + plan.hash_vocab_size - 1
        bad = range_faults(broken_ranges, table.rows, plan.hash_vocab_size)
        report.check(
            bad["overlaps"] > 0 and bad["uncovered"] > 0 and bad["escaped"] > 0,
            f"layer {table.layer_id} control: a shifted and a narrowed range",
            f"{bad}",
        )
    print()


def main() -> int:
    report = Report()
    sections = [
        ("Split", lambda: check_split(report)),
        ("Modes", lambda: check_modes(report)),
        ("Tiling", lambda: check_tiling(report)),
        ("Checkpoint tensors", lambda: check_checkpoint_tensors(report)),
        ("Rotary", lambda: check_rotary(report)),
        ("Residual stream", lambda: check_residual_stream(report)),
        ("Engram", lambda: check_engram(report)),
        ("Placement", lambda: check_placement(report)),
        ("KV budget", lambda: check_kv_budget(report)),
        ("Bucket layout", lambda: check_bucket_layout(report)),
        ("Controls", lambda: check_controls(report)),
        ("Control sensitivity", lambda: check_control_sensitivity(report)),
    ]
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
