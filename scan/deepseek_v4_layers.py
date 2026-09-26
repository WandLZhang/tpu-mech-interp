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

"""Layer plan for DeepSeek V4.1-Flash.

`deepseek-ai/DeepSeek-V4.1-Flash` breaks four assumptions a serving runtime
makes about a decoder stack. This module parses `config.json` into the plan that
survives all four, so a port can size its caches and emit its scan bodies
before any model code exists.

1. **It's a causal encoder-decoder.** The 40 backbone layers split 20/20. Every
   decoder layer reads one compressed KV cache that layer 20 projects from the
   hidden state leaving the encoder. `stage_split` finds the boundary and
   `check_causal_encoder_decoder` confirms one decoder KV source covers the
   whole decoder, which is what makes the split a split.

2. **CSA2 ties layers together.** Each attention layer runs one of four modes.
   A `FULL` layer compresses its own KV and builds its own sparse index. A
   `REINDEX` layer reads the KV and index keys of an earlier layer and runs its
   own top-k over them. A `REUSE` layer takes the KV, the keys and the top-k.
   A `WINDOW` layer carries no compressed attention at all.
   `kv_source_layer_ids` and `index_source_layer_ids` list the owners, and a
   consumer binds to the nearest owner at or before it, because the reference
   runtime hands state down the stack in layer order. `layer_plans` resolves
   every binding.

3. **Engram is a table larger than one chip.** Two n-gram hash tables of about
   384 million rows each sit at layers 1 and 14, 196.6B parameters of
   conditional memory over a 16M-entry hash space. `engram_plan` returns the
   sharding descriptor and the bandwidth either placement costs.

4. **The three DSpark blocks aren't three more backbone layers.** They live
   under the `mtp.*` checkpoint namespace, not `layers.*`, and each carries a
   different parameter set. `LayerPlan.checkpoint_prefix` names the namespace
   and `LayerPlan.extra_weights` names the tensors, so a weight loader built
   from the plan resolves all 43 blocks.

The shape a stack builder wants is `unit_runs`. A unit is one source layer plus
the `REUSE` layers bound to it, and adjacent units of the same shape fold into
one `lax.scan` over stacked weights. Two units fold only when their weight sets
match and their layers play the same part in the candidate pool, so the run
key carries the tensor names and the pool roles.

    from deepseek_v4_layers import DEEPSEEK_V41_FLASH_CONFIG, unit_runs

    for run in unit_runs(DEEPSEEK_V41_FLASH_CONFIG):
        print(run.modes, run.start, run.repeats)

`DEEPSEEK_V41_FLASH_CONFIG` is the shipped `config.json`, transcribed whole.
The text fields nest under `text_config` there, so every function here reads
through `text_config` and a caller can hand over `json.load(f)` directly.

This module plans. It builds no layers. A full implementation needs the
Hyper-Connections residual stream, the two-level sparse indexer, the FP4 KV
path and the Engram gather, none of which live here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

__all__ = [
    "DECODER",
    "DEEPSEEK_V41_FLASH_CONFIG",
    "DRAFT",
    "ENCODER",
    "FULL",
    "POOL_BUILDS",
    "POOL_READS",
    "REINDEX",
    "REUSE",
    "ROOT_TENSORS",
    "SCALED_WEIGHTS",
    "WINDOW",
    "EngramPlacement",
    "EngramPlan",
    "EngramTable",
    "KVQuantization",
    "LayerPlan",
    "ResidualStream",
    "RotaryTable",
    "StageSplit",
    "Unit",
    "UnitRun",
    "cache_plan",
    "check_causal_encoder_decoder",
    "check_cover",
    "compress_ratios",
    "engram_plan",
    "gib",
    "global_kv_bytes_per_token",
    "layer_plans",
    "padded_bf16_kv_bytes_per_token",
    "residual_stream",
    "rotary_tables",
    "stage_split",
    "text_config",
    "unit_runs",
    "units",
    "window_kv_bytes_per_sequence",
]

# --- Modes -------------------------------------------------------------------
#
# The three CSA2 modes, plus the mode a layer runs when it takes no part in
# CSA2 at all. A layer's mode follows from `compress_ratios[i]` and from
# whether `i` appears in the two source lists.

# No compressed KV, no indexer. Sliding window only.
WINDOW = "window"
# Compresses its own KV, builds its own index keys, runs its own top-k.
FULL = "full"
# Reads another layer's compressed KV and index keys, runs its own top-k.
REINDEX = "reindex"
# Reads another layer's compressed KV, index keys and top-k.
REUSE = "reuse"

# What a layer does with the candidate pool. The builder indexes over every
# compressed position and keeps the best blocks. A reader indexes inside that
# pool. Every other layer leaves it alone, so its role is None.
POOL_BUILDS = "builds"
POOL_READS = "reads"

# --- Stages ------------------------------------------------------------------

ENCODER = "encoder"
DECODER = "decoder"
# The DSpark speculative-decoding blocks. `compress_ratios` covers them too, so
# they carry layer indices past the backbone, but the checkpoint stores them
# under `mtp.<stage>.` rather than `layers.<index>.`.
DRAFT = "draft"

# --- Checkpoint layout -------------------------------------------------------

# Tensors the checkpoint stores at the root, with no `model.` or
# `language_model.` prefix in front of them. A loader written against the
# `model.embed_tokens.weight` convention resolves none of these.
ROOT_TENSORS = (
    "aligner.w1.bias",
    "aligner.w1.weight",
    "aligner.w2.bias",
    "aligner.w2.weight",
    "embed.weight",
    "head.weight",
    "image_end",
    "image_newline",
    "image_start",
    "norm.weight",
)

# Mode-specific weights the checkpoint stores quantized. Each ships an FP8
# E4M3 payload plus an E8M0 `.scale` sibling, and the suffix alone doesn't say
# which: `attn.indexer.wk.weight` sits beside `attn.indexer.wq_b.weight` and
# stays BF16. A loader that names only the payload drops the scale and reads
# the tensor as if it were dense.
SCALED_WEIGHTS = frozenset(
    {
        "attn.indexer.wq_b.weight",
        "engram.embed.weight",
        "engram.wkv.weight",
        "main_proj.weight",
    }
)


def _with_scales(names: Sequence[str]) -> list[str]:
    """Each weight, plus the `.scale` sibling the checkpoint stores beside it."""
    out: list[str] = []
    for name in names:
        out.append(name)
        if name in SCALED_WEIGHTS:
            out.append(name.removesuffix(".weight") + ".scale")
    return out


@dataclasses.dataclass(frozen=True)
class LayerPlan:
    """One block, with every binding a stack builder has to resolve."""

    index: int
    stage: str
    mode: str
    compress_ratio: int
    # Layer whose compressed KV this one attends over, and whose index keys it
    # scores against, because the keys project from that layer's latent. None
    # for WINDOW. A FULL layer names itself.
    kv_source: int | None
    # Layer whose top-k selection covers this one. A FULL or REINDEX layer runs
    # its own top-k and names itself. A REUSE layer names the owner it reads.
    # None for WINDOW.
    index_source: int | None
    # Layer that narrows this one's index to a candidate pool. None when this
    # layer indexes over every compressed position, or doesn't index at all.
    candidate_source: int | None
    # Position of this layer's Engram table in `engram_layer_ids`, or None.
    engram_table: int | None
    # Position of this block in the DSpark draft chain, and the chain's length.
    # Both None on a backbone layer. The first stage carries the projection
    # that reads the target hidden states, the last carries the draft heads,
    # and the stages in between carry neither.
    draft_stage: int | None = None
    draft_stages: int | None = None

    @property
    def checkpoint_prefix(self) -> str:
        """Key prefix this block's tensors sit under in the safetensors index.

        The `layers.` namespace stops at the last backbone layer. A draft block
        is `mtp.<stage>.`, so a loader that builds keys from `index` alone
        resolves nothing for three of the 43 blocks.
        """
        if self.draft_stage is None:
            return f"layers.{self.index}."
        return f"mtp.{self.draft_stage}."

    @property
    def owns_compressed_kv(self) -> bool:
        """Runs a Compressor and writes the shared compressed KV cache."""
        return self.mode == FULL

    @property
    def owns_index_keys(self) -> bool:
        """Projects index keys from its own latent. Only a FULL layer can."""
        return self.mode == FULL

    @property
    def owns_topk(self) -> bool:
        """Runs an indexer and publishes a top-k selection."""
        return self.mode in (FULL, REINDEX)

    @property
    def is_candidate_source(self) -> bool:
        """Builds the candidate pool that later indexers are held inside."""
        return self.candidate_source == self.index

    @property
    def pool_role(self) -> str | None:
        """`POOL_BUILDS`, `POOL_READS`, or None for a layer the pool doesn't touch.

        The builder and a reader run different indexer code, one over every
        compressed position and one inside the pool, so the role is part of
        what a scan body has to agree on.
        """
        if self.candidate_source is None:
            return None
        return POOL_BUILDS if self.is_candidate_source else POOL_READS

    @property
    def pools_kv(self) -> bool:
        """Pools several tokens into one latent, so it holds a partial group.

        A ratio of 1 projects one latent per token and carries no partial
        state. A ratio above 1 holds up to `compress_ratio - 1` tokens between
        decode steps, which is a second per-sequence state a cache manager owns.
        """
        return self.owns_compressed_kv and self.compress_ratio > 1

    @property
    def extra_weights(self) -> tuple[str, ...]:
        """Weight-key suffixes this block adds on top of a plain block.

        A plain block is the attention, feed-forward and Hyper-Connections set
        every one of the 43 blocks carries. The suffixes below are what the
        mode and the draft stage add to it, `.scale` siblings included, so the
        plan predicts the checkpoint key for key.
        """
        out: list[str] = []
        if self.owns_compressed_kv:
            out += ["attn.compressor.norm.weight", "attn.compressor.wkv.weight"]
            if self.compress_ratio > 1:
                out.append("attn.compressor.wgate.weight")
        if self.owns_index_keys:
            out += ["attn.indexer.wk.weight", "attn.indexer.k_norm.weight"]
        if self.owns_topk:
            out += ["attn.indexer.weights_proj.weight", "attn.indexer.wq_b.weight"]
        if self.engram_table is not None:
            out += [
                "engram.embed.weight",
                "engram.wkv.weight",
                "engram.q_weight",
                "engram.k_weight",
            ]
        if self.draft_stage is not None:
            # The first stage projects the concatenated target hidden states
            # down to one hidden width. The last stage carries the heads that
            # turn the draft stream into tokens and a confidence score.
            if self.draft_stage == 0:
                out += ["main_proj.weight", "main_norm.weight"]
            if self.draft_stage == self.draft_stages - 1:
                out += [
                    "norm.weight",
                    "markov_head.embed.weight",
                    "markov_head.head.weight",
                    "confidence_head.proj.weight",
                ]
        return tuple(sorted(_with_scales(out)))


@dataclasses.dataclass(frozen=True)
class StageSplit:
    """Where the encoder ends and the decoder begins."""

    num_backbone_layers: int
    num_draft_layers: int
    decoder_start: int

    @property
    def encoder(self) -> range:
        return range(0, self.decoder_start)

    @property
    def decoder(self) -> range:
        return range(self.decoder_start, self.num_backbone_layers)

    @property
    def draft(self) -> range:
        return range(
            self.num_backbone_layers, self.num_backbone_layers + self.num_draft_layers
        )

    @property
    def num_blocks(self) -> int:
        return self.num_backbone_layers + self.num_draft_layers

    def stage_of(self, index: int) -> str:
        if index in self.encoder:
            return ENCODER
        if index in self.decoder:
            return DECODER
        if index in self.draft:
            return DRAFT
        raise ValueError(f"layer {index} lies outside the {self.num_blocks} blocks")


@dataclasses.dataclass(frozen=True)
class Unit:
    """One source layer and the REUSE layers bound to it, in order.

    A `REUSE` layer right after a `WINDOW` layer opens a unit of its own,
    because the window layer closed its owner's unit. It still reads that
    owner's shared state.
    """

    modes: tuple[str, ...]
    engram: tuple[bool, ...]
    weights: tuple[tuple[str, ...], ...]
    stage: str
    compress_ratio: int
    start: int
    pool_roles: tuple[str | None, ...] = ()

    @property
    def length(self) -> int:
        return len(self.modes)

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)

    @property
    def key(self) -> tuple:
        """What two units have to agree on to share one scan body.

        `weights` is in here because stacking is per tensor: two bodies that
        read different parameter sets can't run the same traced code, whatever
        their modes say. The three DSpark blocks share a mode and carry three
        parameter sets.
        `pool_roles` is in here because the candidate pool's builder indexes
        over every compressed position and a reader indexes inside the pool,
        so a builder that shares a mode with its readers still runs other code.
        """
        return (self.stage, self.compress_ratio, self.modes, self.weights, self.pool_roles)


@dataclasses.dataclass(frozen=True)
class UnitRun(Unit):
    """Adjacent units of one shape. One `lax.scan` of `repeats` steps.

    It carries the fields of the unit it repeats, so its `key` is that unit's
    key, and `length` counts every repeat.
    """

    repeats: int = 1

    @property
    def unit_length(self) -> int:
        return len(self.modes)

    @property
    def length(self) -> int:
        return self.unit_length * self.repeats


# --- Config access -----------------------------------------------------------


def text_config(config: dict) -> dict:
    """The language-model half of the config.

    The shipped `config.json` nests the text fields under `text_config` and the
    vision tower under `vision_config`. Accepts either the whole file or the
    inner block, so a caller can hand over `json.load(f)` directly.
    """
    if "text_config" in config:
        return config["text_config"]
    return config


def _require(cfg: dict, field: str):
    if field not in cfg:
        raise ValueError(f"config has no {field}, so the layer plan is undefined")
    return cfg[field]


def _sources(cfg: dict, field: str, num_backbone: int) -> tuple[int, ...]:
    """A source-layer list, checked for order, duplicates and range."""
    raw = list(_require(cfg, field))
    if not raw:
        raise ValueError(f"{field} is empty, so no layer owns any shared state")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in raw):
        raise ValueError(f"{field} holds a non-integer entry: {raw}")
    if sorted(set(raw)) != raw:
        raise ValueError(f"{field} must be sorted and free of duplicates, got {raw}")
    out_of_range = [v for v in raw if not 0 <= v < num_backbone]
    if out_of_range:
        raise ValueError(
            f"{field} names layers {out_of_range} outside the "
            f"{num_backbone} backbone layers"
        )
    return tuple(raw)


def compress_ratios(config: dict) -> list[int]:
    """One compression ratio per block, draft blocks included.

    0 means the block runs a sliding window and nothing else. `r` means the
    block attends over a compressed KV cache that pools `r` tokens into one
    latent.

    Raises:
      ValueError: the list length disagrees with
        `num_hidden_layers + num_nextn_predict_layers`, or an entry is
        negative. The list is the source of truth for how many blocks the
        engine builds, so a runtime that sizes its stack from
        `num_hidden_layers` alone comes up short by the draft blocks.
    """
    cfg = text_config(config)
    ratios = list(_require(cfg, "compress_ratios"))
    backbone = int(_require(cfg, "num_hidden_layers"))
    draft = int(cfg.get("num_nextn_predict_layers") or 0)
    if len(ratios) != backbone + draft:
        raise ValueError(
            f"compress_ratios holds {len(ratios)} entries but the engine builds "
            f"{backbone} backbone plus {draft} draft blocks"
        )
    bad = [(i, r) for i, r in enumerate(ratios) if not isinstance(r, int) or r < 0]
    if bad:
        raise ValueError(f"compress_ratios holds negative or non-integer entries {bad}")
    return ratios


def _bind(sources: Sequence[int], index: int) -> int | None:
    """The nearest source at or before `index`.

    Layers run in order and a source writes its shared state before any
    consumer reads it, so the binding is the greatest source id that doesn't
    exceed the consumer.
    """
    found = None
    for s in sources:
        if s <= index:
            found = s
        else:
            break
    return found


def stage_split(config: dict) -> StageSplit:
    """Encoder, decoder and draft ranges.

    The boundary is a structural reading of `compress_ratios` alone. The
    encoder pools tokens into latents and the decoder keeps one latent a token,
    so the decoder begins where the final run of one compression ratio begins.
    `check_causal_encoder_decoder` then confirms the KV wiring agrees with that
    boundary.

    Raises:
      ValueError: no backbone layer runs compressed attention, or a draft block
        asks for compressed attention that no source provides.
    """
    cfg = text_config(config)
    backbone = int(_require(cfg, "num_hidden_layers"))
    draft = int(cfg.get("num_nextn_predict_layers") or 0)
    ratios = compress_ratios(config)

    boundary = _last_ratio_run_start(ratios[:backbone])

    for i in range(backbone, backbone + draft):
        if ratios[i] != 0:
            raise ValueError(
                f"draft block {i} asks for compress ratio {ratios[i]}, but a draft "
                "block is never a source and no backbone source outlives the stack"
            )

    return StageSplit(
        num_backbone_layers=backbone, num_draft_layers=draft, decoder_start=boundary
    )


def _last_ratio_run_start(ratios: Sequence[int]) -> int:
    """Start of the final run of one nonzero compression ratio."""
    nonzero = [i for i, r in enumerate(ratios) if r != 0]
    if not nonzero:
        raise ValueError("no backbone layer runs compressed attention")
    last = nonzero[-1]
    start = last
    while start - 1 in nonzero and ratios[start - 1] == ratios[last]:
        start -= 1
    return start


def layer_plans(config: dict) -> list[LayerPlan]:
    """One `LayerPlan` per block, backbone first, then the draft blocks.

    Every consumer binds to the nearest source at or before it, and the
    binding has to be consistent: a consumer's compression ratio equals its
    source's, because the shared cache is sized `max_seq_len // ratio` and both
    ends index it the same way.

    The candidate pool binds the same way. Its blocks are positions in the
    builder's compressed cache, so an indexer that reads the pool runs the
    builder's ratio.

    Raises:
      ValueError: a KV source owns no indexer, a source and its consumer
        disagree on the compression ratio, a compressed layer has no source
        before it, a KV or index source runs ratio 0, the candidate source
        isn't an index source, an indexer reads a pool built at another
        ratio, or an Engram field is inconsistent.
    """
    cfg = text_config(config)
    split = stage_split(config)
    ratios = compress_ratios(config)
    backbone = split.num_backbone_layers
    num_draft = split.num_draft_layers

    kv_sources = _sources(cfg, "kv_source_layer_ids", backbone)
    index_sources = _sources(cfg, "index_source_layer_ids", backbone)

    orphan = [s for s in kv_sources if s not in index_sources]
    if orphan:
        raise ValueError(
            f"KV sources {orphan} own no indexer, so nothing publishes index keys "
            "for the layers bound to them; every kv_source_layer_id has to appear "
            "in index_source_layer_ids"
        )
    flat = [s for s in kv_sources if ratios[s] == 0]
    if flat:
        raise ValueError(
            f"KV sources {flat} run compress ratio 0, so they compress nothing"
        )
    # A ratio-0 layer runs a sliding window and no indexer, so it can't
    # publish a top-k or build a pool. With this in place an index source and
    # its consumer can't disagree on the ratio either. A consumer's index
    # source sits at or after its KV source, and the KV check covers both.
    flat = [s for s in index_sources if ratios[s] == 0]
    if flat:
        raise ValueError(
            f"index sources {flat} run compress ratio 0, so they run no indexer"
        )

    candidate = cfg.get("candidate_source_layer_id")
    if candidate is not None and candidate >= 0:
        if candidate not in index_sources:
            raise ValueError(
                f"candidate_source_layer_id is {candidate}, which owns no indexer, "
                "so it can build no candidate pool"
            )
    else:
        candidate = None

    engram_at = _engram_layer_index(cfg, backbone)

    plans: list[LayerPlan] = []
    for i, ratio in enumerate(ratios):
        stage = split.stage_of(i)
        draft_stage = i - backbone if stage == DRAFT else None
        draft_stages = num_draft if stage == DRAFT else None
        if stage == DRAFT or ratio == 0:
            plans.append(
                LayerPlan(
                    index=i,
                    stage=stage,
                    mode=WINDOW,
                    compress_ratio=0,
                    kv_source=None,
                    index_source=None,
                    candidate_source=None,
                    engram_table=engram_at.get(i),
                    draft_stage=draft_stage,
                    draft_stages=draft_stages,
                )
            )
            continue

        kv = _bind(kv_sources, i)
        idx = _bind(index_sources, i)
        if kv is None or idx is None:
            raise ValueError(
                f"layer {i} runs compress ratio {ratio} but no source sits at or "
                "before it, so it has no compressed KV to read"
            )
        if ratios[kv] != ratio:
            raise ValueError(
                f"layer {i} runs compress ratio {ratio} and binds to KV source {kv} "
                f"at ratio {ratios[kv]}; the shared cache is sized by the ratio, so "
                "the two have to match"
            )

        if i in kv_sources:
            mode = FULL
        elif i in index_sources:
            mode = REINDEX
        else:
            mode = REUSE

        # A layer indexes inside a candidate pool when it owns a top-k and the
        # pool was built strictly earlier. The builder itself indexes freely,
        # and a REUSE layer reads a selection somebody else already narrowed,
        # so it binds to no pool of its own.
        uses_pool = candidate is not None and mode in (FULL, REINDEX) and candidate < i
        if uses_pool and ratios[candidate] != ratio:
            raise ValueError(
                f"layer {i} runs compress ratio {ratio} and indexes inside the "
                f"candidate pool of layer {candidate} at ratio {ratios[candidate]}; "
                "the pool's blocks are positions in that layer's compressed cache, "
                "so the two have to match"
            )
        plans.append(
            LayerPlan(
                index=i,
                stage=stage,
                mode=mode,
                compress_ratio=ratio,
                kv_source=kv,
                index_source=idx,
                candidate_source=candidate if (uses_pool or i == candidate) else None,
                engram_table=engram_at.get(i),
            )
        )
    return plans


def _engram_layer_index(cfg: dict, backbone: int) -> dict[int, int]:
    """Map each Engram layer id onto its slot in `engram_num_embeddings`."""
    ids = list(cfg.get("engram_layer_ids") or [])
    rows = list(cfg.get("engram_num_embeddings") or [])
    if len(ids) != len(rows):
        raise ValueError(
            f"engram_layer_ids holds {len(ids)} layers and engram_num_embeddings "
            f"holds {len(rows)} tables"
        )
    if sorted(set(ids)) != ids:
        raise ValueError(f"engram_layer_ids must be sorted and unique, got {ids}")
    bad = [i for i in ids if not 0 <= i < backbone]
    if bad:
        raise ValueError(
            f"engram_layer_ids names layers {bad} outside the {backbone} backbone layers"
        )
    return {layer: slot for slot, layer in enumerate(ids)}


def check_causal_encoder_decoder(config: dict) -> None:
    """Confirm the decoder reads one KV cache, projected at its first layer.

    That's the property the architecture is named for, and it's a claim about
    the KV wiring that `stage_split` doesn't make: the boundary comes from the
    compression ratios, and this checks that the sources land on it. A config
    can pass every other check here and still be an ordinary stack that happens
    to change ratio part way down.

    A runtime that misses this lays out one KV cache per layer, forty where the
    model holds four, and projects each decoder layer's KV from its own hidden
    state instead of from the encoder's output.

    A third property needs no check. An encoder layer can never read the
    decoder's KV, because `_bind` returns a source at or before the consumer
    and an encoder layer sits below `decoder_start` by definition.

    Raises:
      ValueError: the decoder holds more than one KV source, its source isn't
        its first layer, or a decoder layer binds its KV somewhere else.
    """
    split = stage_split(config)
    plans = layer_plans(config)
    decoder = [p for p in plans if p.stage == DECODER]

    owners = [p.index for p in decoder if p.owns_compressed_kv]
    if owners != [split.decoder_start]:
        raise ValueError(
            f"the decoder's KV owners are {owners}; a causal encoder-decoder has "
            f"one, at layer {split.decoder_start}"
        )
    stray = [p.index for p in decoder if p.kv_source != split.decoder_start]
    if stray:
        raise ValueError(
            f"decoder layers {stray} bind their KV outside layer {split.decoder_start}"
        )


# --- Scan grouping -----------------------------------------------------------


def units(config: dict) -> list[Unit]:
    """Cut the stack at every layer that owns shared state.

    A `FULL` or `REINDEX` layer opens a unit and the `REUSE` layers bound to it
    close it. A `WINDOW` layer is a unit of one, because it shares nothing. A
    `REUSE` layer right after a `WINDOW` layer still binds to its owner, the
    way `layer_plans` resolves it, but the window layer closed that owner's
    unit, so the `REUSE` layer opens one of its own.

    A unit never spans the encoder-decoder boundary. The layers inside one all
    bind to the same owner, so they all run its compression ratio, and a
    contiguous run of one ratio lies wholly on one side of the boundary.
    """
    plans = layer_plans(config)
    out: list[Unit] = []
    current: list[LayerPlan] = []

    def flush():
        if current:
            out.append(
                Unit(
                    modes=tuple(p.mode for p in current),
                    engram=tuple(p.engram_table is not None for p in current),
                    weights=tuple(p.extra_weights for p in current),
                    stage=current[0].stage,
                    compress_ratio=current[0].compress_ratio,
                    start=current[0].index,
                    pool_roles=tuple(p.pool_role for p in current),
                )
            )
            current.clear()

    for plan in plans:
        if plan.mode in (FULL, REINDEX, WINDOW):
            flush()
            current.append(plan)
            if plan.mode == WINDOW:
                flush()
            continue
        # A REUSE layer. `layer_plans` has bound it to an owner at or before
        # it, so it either extends that owner's unit or, after a window layer,
        # opens its own.
        current.append(plan)
    flush()
    return out


def unit_runs(config: dict) -> list[UnitRun]:
    """Adjacent units of one shape, folded into runs.

    Each run is one `lax.scan` over stacked weights, and the number of distinct
    keys is the number of bodies XLA compiles.
    """
    runs: list[UnitRun] = []
    for unit in units(config):
        if runs and runs[-1].key == unit.key:
            last = runs[-1]
            runs[-1] = dataclasses.replace(last, repeats=last.repeats + 1)
        else:
            runs.append(
                UnitRun(
                    modes=unit.modes,
                    engram=unit.engram,
                    weights=unit.weights,
                    stage=unit.stage,
                    compress_ratio=unit.compress_ratio,
                    start=unit.start,
                    repeats=1,
                    pool_roles=unit.pool_roles,
                )
            )
    return runs


def check_cover(groups: Sequence, num_blocks: int) -> None:
    """Confirm the groups tile `range(num_blocks)` once each, in order.

    Works on anything carrying `start` and `length`, so it checks `units` and
    `unit_runs` output alike.

    Raises:
      ValueError: a gap, an overlap, an out-of-order group, an empty group, or
      a block count that doesn't match.
    """
    cursor = 0
    for pos, group in enumerate(groups):
        if group.length <= 0:
            raise ValueError(f"group {pos} covers {group.length} blocks")
        if group.start != cursor:
            delta = group.start - cursor
            kind = "gap" if delta > 0 else "overlap"
            raise ValueError(
                f"group {pos} starts at {group.start} but block {cursor} is next: "
                f"{kind} of {abs(delta)}"
            )
        cursor += group.length
    if cursor != num_blocks:
        raise ValueError(f"groups cover {cursor} blocks, the stack holds {num_blocks}")


def cache_plan(config: dict) -> dict[str, list[int]]:
    """Which blocks each cache manager owns.

    Returns:
      `window`: every block. Each holds a ring of `sliding_window` slots, sized
        once per sequence and never grown.
      `compressed_kv`: the blocks that own a compressed KV cache. This is the
        cache that grows with the sequence, and there are four of them behind
        forty layers.
      `index_keys`: the blocks that own indexer keys, which is the same set.
      `pooling`: the blocks that hold a partial compression group between
        decode steps.
      `candidates`: the block that publishes the candidate pool.
    """
    plans = layer_plans(config)
    return {
        "window": [p.index for p in plans],
        "compressed_kv": [p.index for p in plans if p.owns_compressed_kv],
        "index_keys": [p.index for p in plans if p.owns_index_keys],
        "pooling": [p.index for p in plans if p.pools_kv],
        "candidates": [p.index for p in plans if p.is_candidate_source],
    }


# --- Rotary positions --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RotaryTable:
    """One rotary table and the blocks that read it.

    The reference builds one table per layer and picks its parameters from
    `compress_ratios[i]` alone. A compressed layer takes `compress_rope_theta`
    with YaRN on; a sliding-window layer takes `rope_theta` with YaRN off. One
    table then rotates that layer's queries, its sliding-window K and its
    compressed latent alike, so the split is per layer and never per tensor.

    Attributes:
      theta: rotary base.
      yarn_factor: the YaRN extension factor, or None when YaRN is off.
      original_max_position_embeddings: context YaRN interpolates from. 0 with
        YaRN off.
      beta_fast, beta_slow: ends of the YaRN ramp. None with YaRN off.
      rotated_channels: channels at the tail of each head the table turns.
      head_dim: full width of a head, most of which stays unrotated.
      layers: blocks that read this table.
    """

    theta: int
    yarn_factor: float | None
    original_max_position_embeddings: int
    beta_fast: float | None
    beta_slow: float | None
    rotated_channels: int
    head_dim: int
    layers: tuple[int, ...]

    @property
    def unrotated_channels(self) -> int:
        return self.head_dim - self.rotated_channels

    @property
    def yarn(self) -> bool:
        return self.yarn_factor is not None


def rotary_tables(config: dict) -> tuple[RotaryTable, ...]:
    """The rotary tables the stack reads, compressed table first.

    Raises:
      ValueError: `rope_theta`, `compress_rope_theta`, `head_dim` or
        `qk_rope_head_dim` is missing, or the rotated tail doesn't fit inside a
        head.
    """
    cfg = text_config(config)
    head_dim = int(_require(cfg, "head_dim"))
    rope_dim = int(_require(cfg, "qk_rope_head_dim"))
    if not 0 < rope_dim <= head_dim:
        raise ValueError(
            f"qk_rope_head_dim is {rope_dim}, which doesn't fit in a {head_dim} head"
        )
    if rope_dim % 2:
        raise ValueError(
            f"qk_rope_head_dim is {rope_dim}; the rotation pairs channels, so it "
            "has to be even"
        )
    base_theta = int(_require(cfg, "rope_theta"))
    compress_theta = int(_require(cfg, "compress_rope_theta"))
    scaling = _require(cfg, "rope_scaling")
    ratios = compress_ratios(config)

    compressed = tuple(i for i, r in enumerate(ratios) if r != 0)
    windowed = tuple(i for i, r in enumerate(ratios) if r == 0)

    tables = [
        RotaryTable(
            theta=compress_theta,
            yarn_factor=float(scaling["factor"]),
            original_max_position_embeddings=int(
                scaling["original_max_position_embeddings"]
            ),
            beta_fast=float(scaling["beta_fast"]),
            beta_slow=float(scaling["beta_slow"]),
            rotated_channels=rope_dim,
            head_dim=head_dim,
            layers=compressed,
        ),
        RotaryTable(
            theta=base_theta,
            yarn_factor=None,
            original_max_position_embeddings=0,
            beta_fast=None,
            beta_slow=None,
            rotated_channels=rope_dim,
            head_dim=head_dim,
            layers=windowed,
        ),
    ]
    return tuple(t for t in tables if t.layers)


# --- Residual stream ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ResidualStream:
    """The Hyper-Connections stream, and what capturing it costs.

    The stream is `copies` parallel vectors of `hidden_size`. Each block
    collapses them twice, once into the attention input and once into the
    feed-forward input, and expands twice on the way out. So a block offers two
    collapsed tensors, not one.

    A block also hands the next block the mix coefficients its feed-forward
    produced. That carry crosses the block boundary, so a `lax.scan` body has
    to thread it alongside the stream itself.
    """

    copies: int
    hidden_size: int
    blocks: int
    collapses_per_block: int = 2

    @property
    def collapse_points(self) -> int:
        return self.blocks * self.collapses_per_block

    def collapsed_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes one token costs when every collapsed sublayer input is kept."""
        return self.collapse_points * self.hidden_size * dtype_bytes

    def expanded_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes one token costs when all `copies` are kept at every point."""
        return self.collapsed_bytes_per_token(dtype_bytes) * self.copies


def residual_stream(config: dict, blocks: int | None = None) -> ResidualStream:
    """The residual-stream descriptor. `blocks` defaults to the backbone."""
    cfg = text_config(config)
    return ResidualStream(
        copies=int(_require(cfg, "hc_mult")),
        hidden_size=int(_require(cfg, "hidden_size")),
        blocks=int(_require(cfg, "num_hidden_layers")) if blocks is None else blocks,
    )


# --- KV arithmetic -----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KVQuantization:
    """The packed layout of each cached tensor, the one the model card prices.

    None of this is in `config.json`. The grids and scale formats come from the
    runtime the checkpoint ships with, and the card's 890 bytes of global KV a
    token assume this packing. That runtime rounds each tensor to its grid and
    writes the result back in BF16, so its own buffers hold 1,024 bytes a
    latent, 256 an index key and 1,024 a window slot, where this layout holds
    288, 68 and 528.

    Attributes:
      latent_bits: bits per element of a compressed KV latent. FP4 E2M1.
      latent_scale_group: elements sharing one latent scale.
      latent_scale_bytes: bytes of that scale. E4M3.
      index_bits: bits per element of an index key. FP4.
      index_scale_group: elements sharing one index-key scale.
      index_scale_bytes: bytes of that scale. E8M0.
      window_bits: bits per element of a sliding-window KV entry. FP8 E4M3.
      window_scale_group: elements sharing one window scale.
      window_scale_bytes: bytes of that scale. E8M0.
    """

    latent_bits: int = 4
    latent_scale_group: int = 16
    latent_scale_bytes: int = 1
    index_bits: int = 4
    index_scale_group: int = 32
    index_scale_bytes: int = 1
    window_bits: int = 8
    window_scale_group: int = 32
    window_scale_bytes: int = 1

    def latent_bytes(self, head_dim: int) -> int:
        return head_dim * self.latent_bits // 8 + (
            head_dim // self.latent_scale_group
        ) * self.latent_scale_bytes

    def index_key_bytes(self, index_head_dim: int) -> int:
        return index_head_dim * self.index_bits // 8 + (
            index_head_dim // self.index_scale_group
        ) * self.index_scale_bytes

    def window_slot_bytes(self, head_dim: int) -> int:
        return head_dim * self.window_bits // 8 + (
            head_dim // self.window_scale_group
        ) * self.window_scale_bytes


def _kv_heads(cfg: dict) -> int:
    """KV heads a layer writes. One latent shared by every query head, here.

    The whole KV budget rests on this being 1. A port that reads
    `num_attention_heads` into its KV head count builds a 64 times larger
    cache and nothing downstream contradicts it, so the field is required
    rather than defaulted.
    """
    heads = int(_require(cfg, "num_key_value_heads"))
    if heads < 1:
        raise ValueError(f"num_key_value_heads is {heads}, so no layer writes KV")
    return heads


def global_kv_bytes_per_token(
    config: dict, quant: KVQuantization | None = None
) -> dict[str, float]:
    """Bytes of persistent KV each token adds, per owning layer and in total.

    A `FULL` layer at ratio `r` writes `num_key_value_heads` latents and one
    shared index key every `r` tokens, so it costs
    `(heads * latent + index_key) / r` bytes per token. The index key is one
    per compressed position however many KV heads there are, because it
    projects from the pooled latent. Nothing else in the stack grows with the
    sequence.

    Returns:
      A map from the owning layer id, as a string, to its bytes per token, plus
      a `total` key.
    """
    cfg = text_config(config)
    quant = quant or KVQuantization()
    head_dim = int(_require(cfg, "head_dim"))
    index_head_dim = int(_require(cfg, "index_head_dim"))
    per_position = _kv_heads(cfg) * quant.latent_bytes(head_dim) + quant.index_key_bytes(
        index_head_dim
    )

    out: dict[str, float] = {}
    total = 0.0
    for plan in layer_plans(config):
        if not plan.owns_compressed_kv:
            continue
        share = per_position / plan.compress_ratio
        out[str(plan.index)] = share
        total += share
    out["total"] = total
    return out


def window_kv_bytes_per_sequence(
    config: dict, quant: KVQuantization | None = None
) -> int:
    """Bytes of sliding-window KV one sequence holds, at any context length.

    Every block keeps a ring of `sliding_window` slots. The ring never grows,
    so this is the whole window cost of a sequence, and it's what the reference
    runtime rebuilds by replaying the last tokens instead of persisting.
    """
    cfg = text_config(config)
    quant = quant or KVQuantization()
    head_dim = int(_require(cfg, "head_dim"))
    window = int(_require(cfg, "sliding_window"))
    blocks = len(compress_ratios(config))
    return quant.window_slot_bytes(head_dim) * _kv_heads(cfg) * window * blocks


def padded_bf16_kv_bytes_per_token(
    config: dict,
    latent_layers: int | None = None,
    indexer_layers: int | None = None,
    align: int = 128,
    dtype_bytes: int = 2,
) -> dict[str, int]:
    """What a paged latent-KV pool costs without the FP4 path or the sharing.

    A paged MLA pool holds one slot per layer per token in one float dtype,
    with the nope and rope halves each padded up to `align`. It has nowhere to
    put a block scale, so the FP4 latent and the FP4 index key both land at
    BF16, and it sizes its layer count from the stack rather than from the four
    layers that own a cache. This prices that default against
    `global_kv_bytes_per_token`.

    Args:
      latent_layers: latent buffers allocated. Defaults to the backbone count.
      indexer_layers: index-key buffers allocated. Defaults to the same.
      align: page-layout alignment for each half of a slot.
      dtype_bytes: bytes per element of the pool dtype.

    Returns:
      `latent_slot`, `index_slot` and `total` bytes a token.
    """
    cfg = text_config(config)
    head_dim = int(_require(cfg, "head_dim"))
    rope_dim = int(_require(cfg, "qk_rope_head_dim"))
    index_head_dim = int(_require(cfg, "index_head_dim"))
    backbone = int(_require(cfg, "num_hidden_layers"))
    if align <= 0:
        raise ValueError(f"align is {align}, so a slot has no width")
    latent_layers = backbone if latent_layers is None else latent_layers
    indexer_layers = backbone if indexer_layers is None else indexer_layers

    def up(value: int) -> int:
        return -(-value // align) * align

    latent_slot = (up(head_dim - rope_dim) + up(rope_dim)) * dtype_bytes
    index_slot = up(index_head_dim) * dtype_bytes
    return {
        "latent_slot": latent_slot,
        "index_slot": index_slot,
        "total": latent_slot * latent_layers + index_slot * indexer_layers,
    }


# --- Engram ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EngramTable:
    """One n-gram hash table.

    Attributes:
      layer_id: backbone layer that reads it.
      rows: hash buckets. Every (n-gram size, head) pair owns a disjoint
        prime-sized range inside this row space.
      row_dim: elements per row, which is `engram_head_dim`.
      value_bytes: bytes per element as stored. FP8 E4M3 in the checkpoint.
      scale_group: elements sharing one scale.
      scale_bytes: bytes of that scale. E8M0 in the checkpoint.
    """

    layer_id: int
    rows: int
    row_dim: int
    value_bytes: int = 1
    scale_group: int = 32
    scale_bytes: int = 1

    @property
    def row_bytes(self) -> int:
        return (
            self.row_dim * self.value_bytes
            + (self.row_dim // self.scale_group) * self.scale_bytes
        )

    @property
    def bytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def params(self) -> int:
        return self.rows * self.row_dim

    def dequantized_bytes(self, dtype_bytes: int = 2) -> int:
        """Bytes the table takes when a loader widens it on the way in."""
        return self.rows * self.row_dim * dtype_bytes


@dataclasses.dataclass(frozen=True)
class EngramPlan:
    """Where the conditional memory lives and what it costs to read.

    The tables hold more parameters than the backbone that reads them, and
    every token touches a fixed handful of rows in each. Two placements are
    open: shard the rows across chips, or hold them in host memory and prefetch.
    The methods below give the bandwidth arithmetic for both.

    One fact decides the argument. A row index is a hash of token ids alone, so
    every index for a whole chunk is known before the first layer runs. The
    gather has no dependence on any hidden state.
    """

    tables: tuple[EngramTable, ...]
    n_heads: int
    head_dim: int
    max_ngram_size: int
    hash_vocab_size: int
    compressed_vocab_size: int
    pad_token_id: int

    @property
    def hash_columns(self) -> int:
        """Rows one token reads from one table.

        A position is hashed as the 2-gram through the `max_ngram_size`-gram
        ending there, each split over `n_heads` heads.
        """
        return (self.max_ngram_size - 1) * self.n_heads

    @property
    def rows_per_token(self) -> int:
        """Rows one token reads across every table."""
        return self.hash_columns * len(self.tables)

    @property
    def bytes(self) -> int:
        return sum(t.bytes for t in self.tables)

    @property
    def params(self) -> int:
        return sum(t.params for t in self.tables)

    def dequantized_bytes(self, dtype_bytes: int = 2) -> int:
        """Bytes both tables take when a loader widens them on the way in."""
        return sum(t.dequantized_bytes(dtype_bytes) for t in self.tables)

    @property
    def gather_bytes_per_token(self) -> int:
        """Table bytes one token pulls, summed over tables."""
        return sum(t.row_bytes * self.hash_columns for t in self.tables)

    @property
    def payload_bytes_per_token(self) -> int:
        """Bytes of dequantized rows one token produces, at BF16.

        This is what crosses ICI, not the packed table bytes, because each
        device dequantizes its own slice before the sum.
        """
        return self.rows_per_token * self.head_dim * 2

    @staticmethod
    def _split(total_bytes: float, devices: int) -> float:
        """Split a byte total evenly over the chips holding it."""
        if devices <= 0:
            raise ValueError(f"devices is {devices}, so no chip holds the rows")
        return total_bytes / devices

    @staticmethod
    def _ring_allreduce_bytes(payload_bytes: float, devices: int) -> float:
        """Bytes one chip moves to sum a payload with a ring all-reduce.

        A ring all-reduce of `S` bytes over `N` devices moves `2 S (N-1) / N`
        per device. At one device the term goes to zero on its own. One chip
        holds every row, so it has nothing to combine.
        """
        if devices <= 0:
            raise ValueError(f"devices is {devices}, so no chip holds the rows")
        return 2.0 * payload_bytes * (devices - 1) / devices

    def bytes_per_device(self, devices: int) -> float:
        """HBM the sharded placement takes on each chip.

        Rows shard across devices. Each device gathers against its own slice,
        zeros the rows it doesn't hold, and one all-reduce sums the result.
        """
        return self._split(self.bytes, devices)

    def allreduce_bytes_per_device(self, tokens: int, devices: int) -> float:
        """Bytes one chip moves over ICI to combine a sharded gather.

        Costs every table as sharded. `placement` costs a subset.
        """
        return self._ring_allreduce_bytes(
            tokens * self.payload_bytes_per_token, devices
        )

    def sharded_seconds(self, tokens: int, devices: int, ici_bytes_per_s: float) -> float:
        """Time the sharded placement spends on ICI for `tokens`."""
        return self.allreduce_bytes_per_device(tokens, devices) / ici_bytes_per_s

    def hidden_fraction(self, layer_id: int, num_blocks: int) -> float:
        """Share of a forward pass that runs before this table is read.

        A host prefetch hides inside that share. Layer 1 offers one block of
        cover, layer 14 offers fourteen, so the two tables face different
        deadlines even though they're the same size.
        """
        if not 0 <= layer_id < num_blocks:
            raise ValueError(f"layer {layer_id} lies outside {num_blocks} blocks")
        return layer_id / num_blocks

    def exposed_host_seconds(
        self,
        tokens: int,
        link_bytes_per_s: float,
        step_seconds: float,
        num_blocks: int,
        host_layers: Sequence[int] | None = None,
    ) -> float:
        """Host-gather time left on the critical path.

        Every hosted table comes over one per-chip link at `link_bytes_per_s`,
        and a table's fetch moves its `row_bytes` for each hash column of each
        token. The fetches start with the step and run back to back in the
        order the layers need them, which is the order that stalls least on
        one link. Compute waits at a table's layer until its rows have landed,
        and the wait pushes back every layer after it. The waits add up to
        what the step grows by. `host_layers` defaults to every table.
        """
        chosen = self._host_set(host_layers)
        exposed = 0.0
        landed = 0.0
        for table in sorted(self.tables, key=lambda t: t.layer_id):
            if table.layer_id not in chosen:
                continue
            landed += tokens * table.row_bytes * self.hash_columns / link_bytes_per_s
            reached = step_seconds * self.hidden_fraction(table.layer_id, num_blocks)
            exposed += max(0.0, landed - reached - exposed)
        return exposed

    def _host_set(self, host_layers: Sequence[int] | None) -> frozenset[int]:
        known = {t.layer_id for t in self.tables}
        if host_layers is None:
            return frozenset(known)
        chosen = frozenset(host_layers)
        stray = sorted(chosen - known)
        if stray:
            raise ValueError(f"layers {stray} carry no Engram table")
        return chosen

    def placement(
        self,
        host_layers: Sequence[int] | None,
        tokens: int,
        devices: int,
        ici_bytes_per_s: float,
        link_bytes_per_s: float,
        step_seconds: float,
        num_blocks: int,
    ) -> "EngramPlacement":
        """Cost one split of the tables between host memory and chip HBM.

        Args:
          host_layers: tables to hold in host memory. The rest shard across
            chips. `None` sends every table to the host, `()` shards them all.
          tokens: tokens in the forward pass being costed. A prefill chunk, or
            a decode batch.
          devices: chips the sharded tables spread over.
          ici_bytes_per_s: per-chip interconnect rate.
          link_bytes_per_s: per-chip host link rate.
          step_seconds: compute time of that forward pass. The host fetch hides
            inside the share of it that runs before each table's layer.
          num_blocks: blocks the forward pass runs, which sets that share.
        """
        chosen = self._host_set(host_layers)
        sharded = [t for t in self.tables if t.layer_id not in chosen]
        sharded_bytes = sum(t.bytes for t in sharded)
        payload = tokens * self.hash_columns * self.head_dim * 2 * len(sharded)
        return EngramPlacement(
            host_layers=tuple(sorted(chosen)),
            sharded_layers=tuple(t.layer_id for t in sharded),
            bytes_per_device=self._split(sharded_bytes, devices),
            ici_seconds=self._ring_allreduce_bytes(payload, devices) / ici_bytes_per_s,
            exposed_host_seconds=self.exposed_host_seconds(
                tokens, link_bytes_per_s, step_seconds, num_blocks, sorted(chosen)
            ),
        )


@dataclasses.dataclass(frozen=True)
class EngramPlacement:
    """What one split of the Engram tables costs.

    Attributes:
      host_layers: tables held in host memory.
      sharded_layers: tables held in chip HBM, split by row.
      bytes_per_device: HBM one chip gives up for the sharded tables.
      ici_seconds: interconnect time the sharded gather adds.
      exposed_host_seconds: host-gather time the prefetch fails to hide.
    """

    host_layers: tuple[int, ...]
    sharded_layers: tuple[int, ...]
    bytes_per_device: float
    ici_seconds: float
    exposed_host_seconds: float


def engram_plan(config: dict) -> EngramPlan | None:
    """The Engram sharding descriptor, or None when the model carries no table.

    Raises:
      ValueError: a table's row width doesn't divide the scale group, or the
        hash-space fields are inconsistent.
    """
    cfg = text_config(config)
    ids = list(cfg.get("engram_layer_ids") or [])
    if not ids:
        return None
    backbone = int(_require(cfg, "num_hidden_layers"))
    _engram_layer_index(cfg, backbone)

    rows = list(cfg["engram_num_embeddings"])
    head_dim = int(_require(cfg, "engram_head_dim"))
    n_heads = int(_require(cfg, "engram_n_heads"))
    max_ngram = int(_require(cfg, "engram_max_ngram_size"))
    hash_vocab = int(_require(cfg, "engram_vocab_size"))
    compressed_vocab = int(_require(cfg, "engram_compressed_vocab_size"))

    if max_ngram < 2:
        raise ValueError(
            f"engram_max_ngram_size is {max_ngram}, so no n-gram spans two tokens"
        )
    if head_dim % 32:
        raise ValueError(
            f"engram_head_dim is {head_dim}, which the 32-element scale group "
            "doesn't divide"
        )
    # Each (n-gram size, head) pair owns a disjoint bucket range, sized by the
    # next unused prime at or above `engram_vocab_size`. So a table holds at
    # least one `engram_vocab_size` per pair, and the excess over that floor is
    # the prime gaps, which stay far below one bucket.
    floor = (max_ngram - 1) * n_heads * hash_vocab
    for layer, count in zip(ids, rows):
        if count < floor:
            raise ValueError(
                f"the table at layer {layer} holds {count} rows, fewer than the "
                f"{floor} its {(max_ngram - 1) * n_heads} bucket ranges need"
            )
        if count - floor >= hash_vocab:
            raise ValueError(
                f"the table at layer {layer} holds {count - floor} rows past the "
                f"{floor} its bucket ranges need, which is a whole extra bucket"
            )
    if compressed_vocab <= 0:
        raise ValueError(
            f"engram_compressed_vocab_size is {compressed_vocab}; every hash "
            "multiplier derives from it"
        )

    return EngramPlan(
        tables=tuple(
            EngramTable(layer_id=layer, rows=count, row_dim=head_dim)
            for layer, count in zip(ids, rows)
        ),
        n_heads=n_heads,
        head_dim=head_dim,
        max_ngram_size=max_ngram,
        hash_vocab_size=hash_vocab,
        compressed_vocab_size=compressed_vocab,
        pad_token_id=int(cfg.get("engram_pad_token_id", 0)),
    )


def gib(value: float) -> float:
    """Bytes as GiB, for printing a budget."""
    return value / 2**30


# --- The published config, transcribed from the checkpoint. ------------------
#
# The whole file, nesting included. The language fields sit under
# `text_config`, the tower under `vision_config`, and the quantization block
# and the image token id sit at the root, which is where a loader learns the
# checkpoint is multimodal.

DEEPSEEK_V41_FLASH_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "transformers_version": "5.6.0",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "text_config": {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [
            0, 0,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0,
        ],
        "compress_rope_theta": 160000,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    },
    "vision_config": {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 32,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2816,
        "patch_size": 14,
        "rope_theta": 10000,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
    },
}

PUBLISHED_CONFIGS = {"DeepSeek V4.1-Flash": DEEPSEEK_V41_FLASH_CONFIG}
