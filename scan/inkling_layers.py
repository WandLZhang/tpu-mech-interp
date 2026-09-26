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

"""Layer plan for Inkling and Inkling-Small.

`thinkingmachines/Inkling` and `thinkingmachines/Inkling-Small` run the same
stack shape at two widths. The stack isn't uniform, so a JAX port can't fold it
into one `lax.scan` over stacked weights. Three things break the uniformity.

1. There's no RoPE. Every attention layer adds a learned relative bias to the
   logits instead. The qkvr projection emits a fourth stream `r` of width
   `d_rel` per head, and a `[d_rel, rel_extent]` matrix turns it into one bias
   per backward distance. Distances at or beyond the extent get zero. Local
   layers run the extent at `sliding_window_size`, global layers at
   `rel_extent`. Above `log_scaling_n_floor` tokens the global layers scale
   their logits by `1 + alpha * log(pos / n_floor)`. `RelativePosition` carries
   all of that, and `tau` computes the scale.

2. `use_sconv` puts four depthwise causal convolutions in every layer: one on
   `k`, one on `v`, one after attention and one after the MLP. Each holds
   `sconv_kernel_size - 1` tokens of state, so a sequence shard reads that many
   tokens from the shard on its left. `sconv_chunk_pairs` writes that state as
   affine `(A, B)` pairs for `affine_scan.py`. `A` comes out zero, so the
   cross-device composition collapses to a shift and `sconv_incoming_state`
   takes it in one exchange.

3. The stack breaks twice, and the two breaks don't line up. `dense_mlp_idx`
   splits the MLPs: layers below the index run one dense MLP, the rest route to
   experts. `local_layer_ids` gives the local-to-global attention cycle, and the
   two attention kinds carry different KV head counts and different windows.

`scan_groups` cuts the stack into maximal runs of layers that share a kind. One
run is one `lax.scan` over stacked weights, and the number of distinct kinds is
the number of bodies XLA compiles.

    cfg = INKLING
    for group in cfg.scan_groups():
        print(group.attention, group.mlp, group.start, group.length)

Both published configs are in `INKLING` and `INKLING_SMALL`, parsed from the
fields their `config.json` carries.
"""

from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
from jax import lax

from affine_scan import entering_states

__all__ = [
    "DENSE",
    "GLOBAL",
    "INKLING",
    "INKLING_SMALL",
    "LOCAL",
    "MOE",
    "InklingConfig",
    "LayerSpec",
    "RelativePosition",
    "ScanGroup",
    "default_local_layer_ids",
    "prefix_states",
    "sconv_chunk_pairs",
    "sconv_incoming_state",
    "sconv_initial_state",
]

LOCAL = "local"
GLOBAL = "global"
DENSE = "dense"
MOE = "moe"


@dataclasses.dataclass(frozen=True)
class RelativePosition:
    """The positional scheme, in place of a rotary table.

    Attributes:
      d_rel: per-head width of the `r` stream the qkvr projection emits.
      rel_extent: backward distance a global layer biases. Beyond it the bias
        is zero.
      local_extent: the same distance for a local layer. It's the sliding
        window, so a local layer biases every position it can attend to.
      log_scaling_n_floor: position where the long-context logit scaling
        starts. `None` turns the scaling off, and a config that means "off"
        has to say `None` rather than 0. It's positive whenever it's set.
      log_scaling_alpha: strength of that scaling.
      max_position: context length the checkpoint advertises.
    """

    d_rel: int
    rel_extent: int
    local_extent: int
    log_scaling_n_floor: int | None
    log_scaling_alpha: float
    max_position: int

    def __post_init__(self):
        floor = self.log_scaling_n_floor
        if floor is not None and floor <= 0:
            raise ValueError(
                f"log_scaling_n_floor is {floor}. Turn the scaling off with None."
            )

    def extent(self, attention: str) -> int:
        """Backward distance this attention kind biases."""
        if attention == LOCAL:
            return self.local_extent
        if attention == GLOBAL:
            return self.rel_extent
        raise ValueError(f"unknown attention kind {attention!r}")

    def bias_shape(self, attention: str, heads: int) -> tuple[int, int]:
        """Shape of one token's relative bias row, `[heads, extent]`."""
        return (heads, self.extent(attention))

    def proj_shape(self, attention: str) -> tuple[int, int]:
        """Shape of the projection that turns `r` into that row."""
        return (self.d_rel, self.extent(attention))

    def tau(self, positions: jnp.ndarray, attention: str = GLOBAL) -> jnp.ndarray:
        """Long-context logit scale, one value per position.

        `1 + alpha * log(max(1, (pos + 1) / n_floor))`. It's one below the
        floor, so a short sequence is untouched. Local layers never scale, and
        a config without a floor never scales.

        Args:
          positions: [T] zero-based token positions.
          attention: `LOCAL` or `GLOBAL`.

        Returns:
          [T] float32.
        """
        ones = jnp.ones_like(positions, dtype=jnp.float32)
        if self.log_scaling_n_floor is None or attention == LOCAL:
            return ones
        effective = (positions + 1).astype(jnp.float32) / float(self.log_scaling_n_floor)
        return 1.0 + self.log_scaling_alpha * jnp.log(jnp.maximum(effective, 1.0))

    def scaled_positions(self) -> int:
        """Tokens a sequence can hold before the scaling turns on."""
        if self.log_scaling_n_floor is None:
            return self.max_position
        return min(self.log_scaling_n_floor, self.max_position)


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """One layer, with everything that decides which scan body it joins."""

    index: int
    attention: str
    mlp: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    window: int | None
    rel_extent: int
    intermediate_size: int
    num_experts: int
    experts_per_token: int
    shared_experts: int
    sconv_kernel: int

    @property
    def kind(self) -> tuple[str, str]:
        """The pair that has to match for two layers to share a scan body."""
        return (self.attention, self.mlp)

    @property
    def kv_channels(self) -> int:
        """Width of one of `k` or `v`, which is also one sconv's width."""
        return self.num_key_value_heads * self.head_dim

    @property
    def sconv_state_tokens(self) -> int:
        """Tokens of convolution state, and the halo a sequence shard takes."""
        return max(self.sconv_kernel - 1, 0)


@dataclasses.dataclass(frozen=True)
class ScanGroup:
    """A run of neighboring layers that share one scan body."""

    attention: str
    mlp: str
    start: int
    length: int

    @property
    def kind(self) -> tuple[str, str]:
        return (self.attention, self.mlp)

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


def _as_text_config(raw: dict) -> dict:
    """The text block of a published `config.json`, or the block itself."""
    if "text_config" in raw:
        return raw["text_config"]
    return raw


def default_local_layer_ids(num_hidden_layers: int) -> tuple[int, ...]:
    """The cycle a config without `local_layer_ids` means.

    The reference builds `{i for i in range(num_hidden_layers) if (i + 1) % 6}`,
    which is five sliding-window layers then one full-attention layer. Both
    published configs write that same cycle out as an explicit list.
    """
    return tuple(i for i in range(num_hidden_layers) if (i + 1) % 6)


# Defaults of `InklingTextConfig` in `transformers` 5.17, the reference class.
# The engine port reads a config through that class, so a field a file leaves
# out takes the value the served model takes. `intermediate_size` is the dense
# width in that class and `moe_intermediate_size` the expert width.
REFERENCE_DEFAULTS = {
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 16,
    "swa_head_dim": 128,
    "sliding_window_size": 512,
    "d_rel": 16,
    "rel_extent": 1024,
    "log_scaling_alpha": 0.1,
    "max_position_embeddings": 131072,
    "conv_kernel_size": 4,
    "intermediate_size": 24576,
    "moe_intermediate_size": 3072,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
}

# `attribute_map` of `InklingTextConfig` in `transformers` 5.17, for the fields
# the plan reads. The class reads the key on the left as the field on the
# right, so the engine port sizes the model from either name. The published
# file writes `model_max_length` and `sconv_kernel_size`, and a save writes the
# names on the right. The class also maps `embedding_multiplier`, which names a
# field the plan doesn't read.
_ALIASES = {
    "sliding_window": "sliding_window_size",
    "num_local_experts": "n_routed_experts",
    "sconv_kernel_size": "conv_kernel_size",
    "model_max_length": "max_position_embeddings",
}

# `layer_types` and `mlp_layer_types` entries, as `transformers` writes them.
_LAYER_TYPES = {"hybrid_sliding": LOCAL, "hybrid": GLOBAL}
_MLP_LAYER_TYPES = {"dense": DENSE, "sparse": MOE}


def _per_layer(text: dict, field: str, allowed: dict, layers: int) -> list[str] | None:
    """A per-layer list `transformers` writes, mapped to this module's kinds."""
    kinds = text.get(field)
    if kinds is None:
        return None
    if isinstance(kinds, str) or len(kinds) != layers:
        raise ValueError(f"{field} has to hold one entry for each of {layers} layers")
    unknown = sorted({kind for kind in kinds if kind not in allowed}, key=str)
    if unknown:
        raise ValueError(f"{field} holds {unknown}; allowed: {sorted(allowed)}")
    return [allowed[kind] for kind in kinds]


def _local_layer_ids(text: dict, layers: int) -> tuple[int, ...]:
    """The local layers: `local_layer_ids`, `layer_types`, or the 5:1 cycle.

    A save through `transformers` writes `layer_types` beside
    `local_layer_ids`, `hybrid_sliding` for a local layer and `hybrid` for a
    global one. When both are there they have to agree. An explicit empty list
    is a config that means every layer is global, so it isn't the same as a
    missing key.
    """
    raw_ids = text.get("local_layer_ids")
    ids = None if raw_ids is None else tuple(int(i) for i in raw_ids)
    if ids is not None:
        if len(set(ids)) != len(ids):
            raise ValueError(f"local_layer_ids repeats a layer: {ids}")
        for i in ids:
            if not 0 <= i < layers:
                raise ValueError(f"local_layer_ids holds {i}, outside [0, {layers})")
    kinds = _per_layer(text, "layer_types", _LAYER_TYPES, layers)
    if kinds is None:
        return default_local_layer_ids(layers) if ids is None else ids
    marked = tuple(i for i, kind in enumerate(kinds) if kind == LOCAL)
    if ids is not None and tuple(sorted(ids)) != marked:
        raise ValueError(
            f"local_layer_ids names {sorted(ids)} and layer_types marks {list(marked)} "
            "as local; they describe one field"
        )
    return marked if ids is None else ids


def _dense_mlp_idx(text: dict, layers: int) -> int:
    """The index below which layers run a dense MLP, from either spelling.

    A save through `transformers` drops `dense_mlp_idx` and writes
    `mlp_layer_types`, `dense` or `sparse` per layer. The plan keeps the dense
    MLPs below one index, so a list that puts a dense MLP after a routed one is
    refused rather than bent. When both fields are there they have to agree.
    """
    raw_idx = text.get("dense_mlp_idx")
    idx = None if raw_idx is None else int(raw_idx)
    kinds = _per_layer(text, "mlp_layer_types", _MLP_LAYER_TYPES, layers)
    if kinds is None:
        return 0 if idx is None else idx
    leading = next((i for i, kind in enumerate(kinds) if kind != DENSE), layers)
    late = [i for i in range(leading, layers) if kinds[i] == DENSE]
    if late:
        raise ValueError(
            f"mlp_layer_types puts a dense MLP at layers {late}, after the routed "
            f"layer {leading}; the plan keeps every dense MLP below dense_mlp_idx"
        )
    if idx is not None and idx != leading:
        raise ValueError(
            f"dense_mlp_idx is {idx} and mlp_layer_types starts routing at {leading}; "
            "they describe one field"
        )
    return leading


def _mlp_widths(text: dict) -> tuple[int, int]:
    """The dense width and the expert width, from either spelling.

    The published file writes the dense width as `dense_intermediate_size` and
    the expert width as `intermediate_size`. A save through `transformers`
    writes the dense width as `intermediate_size` and the expert width as
    `moe_intermediate_size`. So `intermediate_size` means whatever its
    neighbor doesn't, and a file with neither neighbor can't say which.
    """
    dense = text.get("dense_intermediate_size")
    expert = text.get("moe_intermediate_size")
    plain = text.get("intermediate_size")
    if dense is None and expert is None:
        if plain is not None:
            raise ValueError(
                "intermediate_size is the expert width in the published config.json "
                "and the dense width in a transformers save; add "
                "dense_intermediate_size or moe_intermediate_size to say which"
            )
        return (
            REFERENCE_DEFAULTS["intermediate_size"],
            REFERENCE_DEFAULTS["moe_intermediate_size"],
        )
    if dense is None:
        dense = REFERENCE_DEFAULTS["intermediate_size"] if plain is None else plain
    if expert is None:
        expert = REFERENCE_DEFAULTS["moe_intermediate_size"] if plain is None else plain
    return int(dense), int(expert)


def _aliased(text: dict, name: str) -> int:
    """A field `InklingTextConfig` reads under two names, from either one.

    `name` is the field's own name in that class, and `_ALIASES` gives the
    other. The class lets the alias win when a file holds both. The plan
    refuses a file whose two names disagree, since it can't tell which one
    the author meant.
    """
    alias = next(key for key, field in _ALIASES.items() if field == name)
    first, second = text.get(alias), text.get(name)
    if first is not None and second is not None and int(first) != int(second):
        raise ValueError(f"{alias} is {first} and {name} is {second}; they name one field")
    value = first if first is not None else second
    return REFERENCE_DEFAULTS[name] if value is None else int(value)


@dataclasses.dataclass(frozen=True)
class InklingConfig:
    """The fields of a published config that the layer plan reads."""

    repo: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: tuple[int, ...]
    dense_mlp_idx: int
    dense_intermediate_size: int
    intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    use_sconv: bool
    sconv_kernel_size: int
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int | None
    log_scaling_alpha: float
    model_max_length: int

    @classmethod
    def from_config_dict(cls, raw: dict, repo: str) -> "InklingConfig":
        """Parse a `config.json`, whole or just its text block.

        Reads the published spelling and the one a save through `transformers`
        writes. The published file writes `local_layer_ids`, `dense_mlp_idx`,
        `dense_intermediate_size` for the dense width, `intermediate_size` for
        the expert width, `model_max_length` and `sconv_kernel_size`. A save
        writes `layer_types` beside `local_layer_ids`, `mlp_layer_types` in
        place of `dense_mlp_idx`, `intermediate_size` for the dense width,
        `moe_intermediate_size` for the expert width, `max_position_embeddings`
        and `conv_kernel_size`. `InklingTextConfig` also reads `sliding_window`
        as `sliding_window_size` and `num_local_experts` as `n_routed_experts`,
        and so does this. Where a file holds both spellings of one field they
        have to agree. `intermediate_size` with neither width field beside it
        is refused, because the two spellings give it opposite meanings.

        Unknown keys are dropped. `hidden_size`, `num_hidden_layers`,
        `num_attention_heads`, `num_key_value_heads` and `head_dim` are
        required. Any other key a file leaves out takes the default of
        `InklingTextConfig` in `transformers`, which is what the engine port
        reads a config through, so the plan sizes the model the engine builds.
        `REFERENCE_DEFAULTS` holds them. A missing `local_layer_ids` and
        `layer_types` mean the 5:1 cycle
        `{i for i in range(num_hidden_layers) if (i + 1) % 6}`, an explicit
        empty list means every layer is global, a missing `use_sconv` means the
        four short convolutions are on, and a missing `dense_mlp_idx` and
        `mlp_layer_types` mean every MLP routes to experts.
        """
        text = _as_text_config(raw)
        layers = int(text["num_hidden_layers"])
        if layers <= 0:
            raise ValueError(f"num_hidden_layers is {layers}")
        floor = text.get("log_scaling_n_floor")
        if floor is not None and int(floor) <= 0:
            raise ValueError(
                f"log_scaling_n_floor is {int(floor)}."
                " Turn the scaling off with null, not with 0."
            )
        local = _local_layer_ids(text, layers)
        dense_mlp_idx = _dense_mlp_idx(text, layers)
        if not 0 <= dense_mlp_idx <= layers:
            raise ValueError(f"dense_mlp_idx is {dense_mlp_idx}, outside [0, {layers}]")
        dense_width, expert_width = _mlp_widths(text)

        def field(name, cast=int):
            value = text.get(name)
            return cast(REFERENCE_DEFAULTS[name] if value is None else value)

        return cls(
            repo=repo,
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=layers,
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            swa_num_attention_heads=field("swa_num_attention_heads"),
            swa_num_key_value_heads=field("swa_num_key_value_heads"),
            swa_head_dim=field("swa_head_dim"),
            sliding_window_size=_aliased(text, "sliding_window_size"),
            local_layer_ids=local,
            dense_mlp_idx=dense_mlp_idx,
            dense_intermediate_size=dense_width,
            intermediate_size=expert_width,
            n_routed_experts=_aliased(text, "n_routed_experts"),
            num_experts_per_tok=field("num_experts_per_tok"),
            n_shared_experts=field("n_shared_experts"),
            use_sconv=bool(text.get("use_sconv", True)),
            sconv_kernel_size=_aliased(text, "conv_kernel_size"),
            d_rel=field("d_rel"),
            rel_extent=field("rel_extent"),
            log_scaling_n_floor=None if floor is None else int(floor),
            log_scaling_alpha=field("log_scaling_alpha", float),
            model_max_length=_aliased(text, "max_position_embeddings"),
        )

    @classmethod
    def from_json_file(cls, path: str, repo: str) -> "InklingConfig":
        """Parse a `config.json` from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_config_dict(json.load(handle), repo)

    # --- the plan -------------------------------------------------------

    def attention_kind(self, index: int) -> str:
        """`LOCAL` for a sliding-window layer, `GLOBAL` for a full one."""
        return LOCAL if index in set(self.local_layer_ids) else GLOBAL

    def mlp_kind(self, index: int) -> str:
        """`DENSE` below `dense_mlp_idx`, `MOE` from it on."""
        return DENSE if index < self.dense_mlp_idx else MOE

    def positional(self) -> RelativePosition:
        """The positional descriptor, which replaces a rotary table."""
        return RelativePosition(
            d_rel=self.d_rel,
            rel_extent=self.rel_extent,
            local_extent=self.sliding_window_size,
            log_scaling_n_floor=self.log_scaling_n_floor,
            log_scaling_alpha=self.log_scaling_alpha,
            max_position=self.model_max_length,
        )

    def layer_spec(self, index: int) -> LayerSpec:
        """One layer of the plan."""
        if not 0 <= index < self.num_hidden_layers:
            raise IndexError(f"layer {index} outside [0, {self.num_hidden_layers})")
        attention = self.attention_kind(index)
        mlp = self.mlp_kind(index)
        local = attention == LOCAL
        return LayerSpec(
            index=index,
            attention=attention,
            mlp=mlp,
            num_attention_heads=(
                self.swa_num_attention_heads if local else self.num_attention_heads
            ),
            num_key_value_heads=(
                self.swa_num_key_value_heads if local else self.num_key_value_heads
            ),
            head_dim=self.swa_head_dim if local else self.head_dim,
            window=self.sliding_window_size if local else None,
            rel_extent=self.positional().extent(attention),
            intermediate_size=(
                self.dense_intermediate_size if mlp == DENSE else self.intermediate_size
            ),
            num_experts=0 if mlp == DENSE else self.n_routed_experts,
            experts_per_token=0 if mlp == DENSE else self.num_experts_per_tok,
            shared_experts=0 if mlp == DENSE else self.n_shared_experts,
            sconv_kernel=self.sconv_kernel_size if self.use_sconv else 0,
        )

    def layer_specs(self) -> tuple[LayerSpec, ...]:
        """Every layer, bottom to top."""
        return tuple(self.layer_spec(i) for i in range(self.num_hidden_layers))

    def scan_groups(self) -> tuple[ScanGroup, ...]:
        """Maximal runs of neighboring layers that share a kind.

        One run is one `lax.scan` over stacked weights. The runs partition the
        stack: they're ordered, they don't overlap, and together they hold
        every layer once.
        """
        groups: list[ScanGroup] = []
        for spec in self.layer_specs():
            if groups and groups[-1].kind == spec.kind and groups[-1].stop == spec.index:
                last = groups[-1]
                groups[-1] = dataclasses.replace(last, length=last.length + 1)
            else:
                groups.append(
                    ScanGroup(
                        attention=spec.attention,
                        mlp=spec.mlp,
                        start=spec.index,
                        length=1,
                    )
                )
        return tuple(groups)

    def group_kinds(self) -> tuple[tuple[str, str], ...]:
        """The distinct kinds, which is how many bodies XLA compiles."""
        seen: list[tuple[str, str]] = []
        for group in self.scan_groups():
            if group.kind not in seen:
                seen.append(group.kind)
        return tuple(seen)

    # --- sizes ----------------------------------------------------------

    @property
    def local_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.attention_kind(i) == LOCAL)

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.attention_kind(i) == GLOBAL)

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.mlp_kind(i) == DENSE)

    @property
    def sconv_state_tokens(self) -> int:
        """Tokens of convolution state each sconv holds."""
        return max(self.sconv_kernel_size - 1, 0) if self.use_sconv else 0

    def sconv_channels(self, index: int) -> int:
        """Channels the four convolutions of one layer cover together.

        Two of them run on the residual stream, after attention and after the
        MLP. The other two run on `k` and on `v`, so they're KV-width.
        """
        if not self.use_sconv:
            return 0
        spec = self.layer_spec(index)
        return 2 * self.hidden_size + 2 * spec.kv_channels

    def sconv_halo_bytes(self, bytes_per_element: int = 2) -> int:
        """Bytes one sequence shard reads from the shard on its left.

        The whole stack, one sequence, BF16 by default.
        """
        tokens = self.sconv_state_tokens
        channels = sum(self.sconv_channels(i) for i in range(self.num_hidden_layers))
        return tokens * channels * bytes_per_element

    def conv_state_bytes_per_sequence(self, bytes_per_element: int = 4) -> int:
        """Convolution state the cache manager holds for one sequence.

        Four convolutions per layer, `sconv_state_tokens` each, at the widths
        `sconv_channels` reports. Those are the rows `sconv_halo_bytes`
        counts, held at the cache dtype. The reference runs this path in
        float32, so that's the default element size. The state is per
        sequence, so it doesn't grow with context.
        """
        return self.sconv_halo_bytes(bytes_per_element)

    def capture_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """Host traffic for one residual stream per layer per token."""
        return self.num_hidden_layers * self.hidden_size * bytes_per_element

    def kv_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """KV cache one token adds, counting global layers only.

        A local layer holds at most `sliding_window_size` tokens, so its cache
        stops growing. The global layers are what a long context pays for.
        """
        total = 0
        for index in self.global_layers:
            total += 2 * self.layer_spec(index).kv_channels * bytes_per_element
        return total


# Published fields of `thinkingmachines/Inkling/config.json`.
_INKLING_TEXT = {
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
    "local_layer_ids": [i for i in range(66) if i % 6 != 5],
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

# Published fields of `thinkingmachines/Inkling-Small/config.json`.
_INKLING_SMALL_TEXT = dict(
    _INKLING_TEXT,
    hidden_size=4096,
    num_hidden_layers=42,
    num_attention_heads=32,
    swa_num_attention_heads=32,
    swa_num_key_value_heads=8,
    local_layer_ids=[i for i in range(42) if i % 6 != 5],
    dense_intermediate_size=16384,
    intermediate_size=2048,
)

INKLING = InklingConfig.from_config_dict(_INKLING_TEXT, "thinkingmachines/Inkling")
INKLING_SMALL = InklingConfig.from_config_dict(
    _INKLING_SMALL_TEXT, "thinkingmachines/Inkling-Small"
)


# --- short convolution state, as affine pairs ---------------------------


def sconv_initial_state(channels: int, kernel: int, dtype) -> jnp.ndarray:
    """The window entering the first token: `[kernel - 1, channels]` of zeros.

    `dtype` is required, and it's the activation dtype. Default it to float32
    and a BF16 model promotes the whole sconv path to float32, which doubles
    the halo a sequence shard reads.
    """
    return jnp.zeros((max(kernel - 1, 0), channels), dtype)


def sconv_chunk_pairs(
    x: jnp.ndarray,
    kernel: int,
    chunk_size: int,
    tokens: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Affine `(A, B)` pairs for one sconv's state, one pair per chunk.

    The state is the window the convolution still needs: the last `kernel - 1`
    input tokens, oldest row first. A chunk of at least that many tokens
    replaces the whole window, so `A` is zero and `B` is the chunk's tail. The
    recurrence is still affine, so `affine_scan.py` carries it across devices
    the same way it carries a Mamba-2 state.

    `A` is zero rather than merely small, so the composition across shards
    collapses to a shift. `sconv_incoming_state` takes that shift in one
    exchange.

    Args:
      x: [T, D] this layer's convolution input, right padded to whole chunks.
      kernel: `sconv_kernel_size`.
      chunk_size: tokens per chunk. It divides T and is at least `kernel - 1`.
      tokens: real tokens before the padding, or `None` when `x` holds none.
        With it, each window stops at the last real token, so every chunk from
        the one that holds that token on carries the last `kernel - 1` real
        rows, the window a decode step reads. `A` stays zero. Without it the
        chunks past the sequence end carry padding. A real token's output
        comes out the same either way.

    Returns:
      a_chunks: [C, K, K] with K = kernel - 1.
      b_chunks: [C, K, D]

    Raises:
      ValueError: the chunk size doesn't divide T or is shorter than the
        window, or `tokens` is shorter than the window or longer than T.

    Feed both to `compose_local`, `sconv_incoming_state` and `replay_local`.
    `prefix_states` turns what comes out into one `[K, D]` window per chunk,
    which is the `prefix` argument of `mamba2.causal_conv`. That call runs per
    chunk, so vmap it over the chunk axis of `x.reshape(C, chunk_size, D)`.
    """
    total, channels = x.shape
    state = max(kernel - 1, 0)
    if total % chunk_size:
        raise ValueError(f"{total} tokens don't divide into chunks of {chunk_size}")
    if chunk_size < state:
        raise ValueError(f"chunk of {chunk_size} is shorter than the {state}-token window")
    chunks = total // chunk_size

    if tokens is None:
        tails = x.reshape(chunks, chunk_size, channels)[:, chunk_size - state :, :]
    else:
        if not state <= tokens <= total:
            raise ValueError(
                f"tokens is {tokens}, and it has to cover the {state}-token window"
                f" and fit in the {total} rows x holds"
            )
        # Each window ends at its chunk's last row or at the last real token,
        # whichever comes first. The indices are static, so this is one gather.
        ends = np.minimum(np.arange(1, chunks + 1) * chunk_size, tokens)
        tails = x[ends[:, None] - state + np.arange(state)[None, :]]
    a_chunks = jnp.zeros((chunks, state, state), x.dtype)
    return a_chunks, tails


def sconv_incoming_state(
    b_total: jnp.ndarray,
    h_init: jnp.ndarray,
    axis_name: str,
) -> jnp.ndarray:
    """Window entering this shard, taken in one exchange.

    `sconv_chunk_pairs` sets `A` to zero, so composing two shards keeps the
    right-hand `B` and drops everything to its left. The whole prefix scan in
    `affine_scan.incoming_state` therefore lands on the value its final shift
    alone would produce, and the ceil(log2 D) rounds before that shift are
    dead work. This takes the shift and skips the rounds.

    `shard_map` gives every shard the same number of chunks, so right pad the
    tokens until the chunk count divides over the mesh. Padding never reaches
    a real token's output: the padded rows sit after every real token, and
    each padded chunk still folds to `A = 0`. It does reach the windows past
    the sequence end. Pass the real token count to `sconv_chunk_pairs` as
    `tokens`, and those windows hold the last real rows, so the state leaving
    the last shard is the window a decode step reads. A shard can't hold zero
    chunks while another holds some.

    Args:
      b_total: [K, D] this shard's folded `B`, which is the tail of its last
        chunk.
      h_init: [K, D] window entering the whole sequence.
      axis_name: mesh axis the sequence is sharded over.

    Returns:
      [K, D] window entering this shard's first chunk. Shard 0 gets `h_init`.
    """
    num_devices = lax.axis_size(axis_name)
    index = lax.axis_index(axis_name)
    shift = [(src, (src + 1) % num_devices) for src in range(num_devices)]
    received = lax.ppermute(b_total, axis_name, shift)
    return jnp.where(index == 0, h_init, received)


def prefix_states(h_in: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
    """Turn end-of-chunk states into start-of-chunk states.

    `replay_local` returns the state after each chunk. A convolution wants the
    state before each chunk. That's `affine_scan.entering_states`, which takes
    the same two arrays in the other order.

    Args:
      h_in: [K, D] state entering the first local chunk.
      states: [C, K, D] state after each local chunk.

    Returns:
      [C, K, D] state entering each local chunk.
    """
    return entering_states(states, h_in)
