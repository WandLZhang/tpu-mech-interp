#!/usr/bin/env python3
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

"""Check a model's captured residual stream against a HuggingFace float32 forward, every layer.

    python3 scripts/check_capture.py --model-path SNAPSHOT_DIR --tp-size 8

Both sides read the same token ids. The engine captures on the TPU in BF16; `transformers` runs
the same checkpoint on the host CPU in float32 with `output_hidden_states=True`. Slot `i` of the
capture is the stream entering block `i`, which is HuggingFace's `hidden_states[i]`, so the two
line up index for index. The last HuggingFace entry is the final norm's output and has no slot.
The engine starts without `--return-hidden-states-layers`, so each reply holds every slot, and
`--engine-arg return_hidden_states_layers` is refused.

Each layer line prints Pearson correlation beside the readings the gate takes, and the RESULT
line carries the worst max absolute error. A residual stream carries large values, so BF16 against
float32 differs by tens in absolute terms while the vectors stay collinear to many decimal places.

BF16 drifts from float32 with depth, and how far depends on the model. So the gate is relative:
the check also runs `transformers` in BF16, and each layer has to pass two tests against that
forward. A token's error is its row's L2 distance from the reference row, as a fraction of the
reference row's norm. The capture's median per-token error can be at most `--floor-factor` times
the BF16 forward's. The capture's diverged tokens can number at most `--floor-factor` times the
BF16 forward's, plus a slack of 2% of the prompt's tokens. The slack is 3 tokens on a prompt short
enough that 2% is fewer. On Gemma 4 26B-A4B the BF16 forward alone reads Pearson 0.9909 at layer
27, and a fixed 0.999 would fail a sound capture.

A diverged token sits past relative error 0.25. In an MoE a routing near-tie sends a handful of
tokens to other experts, on the BF16 side and on the capture side, a different handful each time:
12 to 14 of 441 tokens on Gemma 4 26B-A4B, one of them at relative error 2.0 by the last layer. One
such token decides a Pearson over the whole prompt, so Pearson gates nothing. The median holds
until close to half a prompt's rows go wrong. A prefill chunk filed under another request's rows,
or zeroed, puts each of its rows past 0.25 at every layer from the embedding on, and the count
catches it where the median doesn't.

`--no-floor` skips the BF16 forward. Each layer then needs a median per-token Pearson of at least
`--min-pearson`, and no more diverged tokens than the slack. Both read one value per token, so a
large-norm row such as Gemma's BOS doesn't decide the layer.

The control compares each slot against the next layer's reference, one token at a time: each row's
error as a fraction of its own norm, then the median over tokens, so a large-norm token such as
Gemma's BOS doesn't decide it. At every layer the capture has to sit `--control-factor` times
closer to its own layer than to the next, or the check can't tell layers apart and proves nothing.
On 441-token Gemma 4 prompts a BF16 forward sits at least 5.3 times closer, layer by layer.

The ids get the tokenizer's BOS token when the tokenizer doesn't add one itself, as Gemma's
doesn't. A Gemma forward without BOS runs off its trained distribution. `--no-bos` sends the ids
as they are.

`--prompts-file prompts.jsonl` sends the first `--num-prompts` prompts to the engine in one call,
plus one prompt joined from the next three, and compares each against its own forward. On
`build_corpus.py`'s 440-token prompts the joined prompt runs 1,323 tokens with BOS, past the
engine's 1,024-token prefill pass, so it splits across passes however the scheduler groups the
requests. Whether the others share a pass depends on when they reach the scheduler, so they may or
may not split. The joined prompt is the one that tests the chunked-prefill path, so the check stops
before any forward when the file holds fewer than `--num-prompts` + 3 prompts, or when the joined
prompt fits one pass of `--chunked-prefill-size` tokens. That size reaches the engine through its
own flag, and `--engine-arg chunked_prefill_size` is refused.

The float32 reference needs host RAM of about 4 bytes per parameter: 103 GB for Gemma 4 26B-A4B.
The BF16 forward loads after the float32 model is freed.

`--trust-remote-code` builds the reference from the checkpoint's own modeling code, which Kimi K3
needs, through `remote_code_reference.py`. When fla isn't installed, `fla_torch.py` stands in for
its Triton kernels. MXFP4 experts dequantize to BF16 on load. With `--offload-folder` the decoder
layers stream one at a time straight from the checkpoint's safetensors and nothing is written.
`--reference-layers N` builds only the first N layers, for a dry run with `--reference-only`. It
works on the `transformers` path too: the text config's layer count and every per-layer list cut
to N before the model builds, and the checkpoint's later layers stay unread.

`--engine-layers N` serves only the model's first N decoder layers. The engine gets
`json_model_override_args` with `num_hidden_layers` set to N, merged into any override passed with
`--engine-arg`, and the check compares slots 0 to N-1 against reference entries 0 to N-1. Entry k
is the stream entering layer k, which depends only on the layers before it, so a whole-model
reference stays valid for a cut engine, and the engine builds only N layers. N has to be
below the reference's entry count, since its last entry is the final norm's output. A config whose
`num_hidden_layers` write doesn't cut its layer list, as `configs/nemotron_h.py`'s does, builds
the whole model, and the check then stops on the slot count.

A model whose residual stream is several copies wide, such as GLM-5.3-Flash's four mHC streams,
hands back `[seq, copies, d]` per layer. Each entry flattens to `[seq, copies * d]`, which is how
the engine's capture slot lays the same values out. The last entry, the final norm's output, is
then narrower than the rest, and `save_reference` writes it under its own key.

`--deepseek-inference` builds the reference from the checkpoint's own `inference/model.py`,
DeepSeek's runtime for V4.1-Flash, through `deepseek_reference.py`. `deepseek_cpu_kernels.py`
stands in for its tilelang kernels. FP8 dense weights and FP4 experts dequantize to BF16 on load,
the Engram tables read their rows from the memory-mapped safetensors, and each entry is the
collapsed attention input of a layer, the slot the engine captures. `--offload-folder` streams the
blocks one at a time from the safetensors, and nothing is written to the folder.

Both HuggingFace forwards run before the engine starts. Importing `sgl_jax` registers its own
config classes with `AutoConfig`, Gemma 4's among them, and `transformers` can't build the model
from those. Engine() re-imports this module in its subprocesses, so everything sits behind
`__main__`. `--engine-arg` gets parsed first, so a bad one stops the run before the forwards.

A multimodal checkpoint runs through `AutoModelForImageTextToText`. `transformers` 5.17's Inkling
class norms the embeddings twice there: `InklingModel.forward` applies `embed_norm`, then the text
model applies it again. The reference lets each `embed_norm` run once per forward, the way
upstream's fix (#48786), SGLang and the engine port run it. Each prompt's forward prints how many
times `embed_norm` ran.

`--save-capture NPZ` writes the engine capture and the token ids. `--capture-npz NPZ` reads them
back instead of starting the engine, so a capture gates again against a new `--reference-npz`
without loading the model. The ids have to match, and `--save-npz` files carry them too.

`--save-npz caps` writes `caps.npz`, since `np.savez` adds the suffix, and prints that name.
`--log-payloads`, or `LOG_PAYLOADS=1`, logs the request the engine gets, token ids and all, and a
summary of each reply with every array reduced to its shape and dtype.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROMPT = "The capital of France is Paris, and the capital of Japan is"

# Keys this script sets itself, and the flag that sets each. `--engine-arg` refuses them.
RESERVED_ENGINE_ARGS = {
    "model_path": "use --model-path",
    "enable_return_hidden_states": "the check always turns it on",
    "return_hidden_states_layers": "the check compares every slot, so the engine returns them all",
    "tp_size": "use --tp-size",
    "batch_size": "use --batch-size",
    "token_padding": "use --token-padding",
    "chunked_prefill_size": "use --chunked-prefill-size, which the joined prompt is checked against",
}

# The engine_settings default, which splits the joined prompt of 440-token corpus prompts.
CHUNKED_PREFILL_SIZE = 1024

# A per-token error below this counts as a match whatever the floor says. It keeps a layer that
# both sides reproduce to float rounding, such as the embedding, from failing on noise.
ERROR_EPSILON = 1e-6

# A token whose error passes this has left its reference. In an MoE that's a routing near-tie that
# resolved to other experts, which BF16 does on both sides at different tokens.
DIVERGED = 0.25

# The diverged tokens a capture may hold beyond `floor_factor` times the BF16 forward's count: this
# share of the prompt's tokens, and never fewer than DIVERGED_MIN_SLACK tokens. The measured
# Gemma 4 26B-A4B runs held at most 7 against the BF16 forward's 2 on a 441-token prompt. A chunk
# filed under the wrong rows diverges on every row it holds.
DIVERGED_SLACK = 0.02
DIVERGED_MIN_SLACK = 3


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def token_pearsons(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Each row's Pearson correlation with its reference row, one value per token.

    A row with no spread reads 1.0 when it equals its reference and 0.0 otherwise, so a zeroed
    row counts as a miss.
    """
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    g = got - got.mean(axis=-1, keepdims=True)
    w = want - want.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1)
    same = np.all(got == want, axis=-1)
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, np.sum(g * w, axis=-1) / safe, np.where(same, 1.0, 0.0))


def token_errors(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Each row's L2 error as a fraction of the reference row's norm, one value per token."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    norms = np.linalg.norm(want, axis=-1)
    norms = np.where(norms > 0, norms, 1.0)
    return np.linalg.norm(got - want, axis=-1) / norms


def token_error(got: np.ndarray, want: np.ndarray) -> float:
    """The median over tokens of `token_errors`: every token counts the same."""
    return float(np.median(token_errors(got, want)))


def diverged(errors: np.ndarray) -> int:
    """Tokens past DIVERGED. A NaN or inf error counts as diverged too."""
    return int(np.sum(~(np.asarray(errors) <= DIVERGED)))


def diverged_slack(tokens: int, share: float = DIVERGED_SLACK) -> float:
    """The diverged tokens a prompt of `tokens` may hold beyond what the floor accounts for."""
    return max(float(DIVERGED_MIN_SLACK), share * tokens)


def compare_layers(
    captured: np.ndarray,
    reference: list,
    min_pearson: float,
    control_factor: float,
    floor: list | None = None,
    floor_factor: float = 2.0,
    slack_share: float = DIVERGED_SLACK,
):
    """Aligned and shifted comparisons of `[seq, layers, d]` against per-layer `[seq, d]` arrays.

    `floor` holds a BF16 forward's per-layer arrays. With it, a layer passes two tests: the
    capture's median per-token error is at most `floor_factor` times the floor's, and its diverged
    tokens number at most `floor_factor` times the floor's plus `diverged_slack`. Without it, a
    layer passes when its median per-token Pearson reaches `min_pearson` and its diverged tokens
    stay within `diverged_slack`. `slack_share` sets that slack's share of the prompt.

    Returns `(rows, summary)`. `rows` holds one dict per layer, with `failed_tests` naming the
    tests it failed. `summary` holds the worst readings, whether the aligned check passed, which
    layers failed which test, and whether the control was detected.
    """
    layers = min(captured.shape[1], len(reference))
    if floor is not None:
        layers = min(layers, len(floor))
    if layers == 0:
        raise ValueError("nothing to compare: the capture or the reference has no layers")
    slack = diverged_slack(captured.shape[0], slack_share)
    rows = []
    for i in range(layers):
        got = captured[:, i, :]
        want = np.asarray(reference[i], np.float64)
        if got.shape != want.shape:
            raise ValueError(f"layer {i}: capture is {got.shape}, reference is {want.shape}")
        errors = token_errors(got, want)
        row = {
            "layer": i,
            "pearson": pearson(got, want),
            "max_abs": float(np.max(np.abs(got - want))),
            "token_error": float(np.median(errors)),
            "diverged": diverged(errors),
        }
        if floor is None:
            row["token_pearson"] = float(np.median(token_pearsons(got, want)))
            row["diverged_allowed"] = slack
            first = ("pearson", row["token_pearson"] >= min_pearson)
        else:
            floor_errors = token_errors(floor[i], want)
            row["floor_pearson"] = pearson(floor[i], want)
            row["floor_token_error"] = float(np.median(floor_errors))
            row["floor_diverged"] = diverged(floor_errors)
            error, floor_error = row["token_error"], row["floor_token_error"]
            row["ratio"] = error / floor_error if floor_error > 0 else (0.0 if error <= 0 else float("inf"))
            row["diverged_allowed"] = floor_factor * row["floor_diverged"] + slack
            first = ("median", error <= floor_factor * floor_error + ERROR_EPSILON)
        tests = (first, ("diverged", row["diverged"] <= row["diverged_allowed"]))
        row["failed_tests"] = [name for name, ok in tests if not ok]
        row["passed"] = not row["failed_tests"]
        rows.append(row)
    ratios = []
    for i in range(layers - 1):
        if i + 1 >= len(reference):
            break
        aligned = rows[i]["token_error"]
        shifted = token_error(captured[:, i, :], reference[i + 1])
        if aligned > 0:
            rows[i]["shift_ratio"] = shifted / aligned
        else:  # an exact match: the layers are apart only if the next one differs at all
            rows[i]["shift_ratio"] = float("inf") if shifted > 0 else 1.0
        ratios.append((rows[i]["shift_ratio"], i))
    worst = min(rows, key=lambda r: r["pearson"])
    worst_abs = max(r["max_abs"] for r in rows)
    control_ratio, control_layer = min(ratios) if ratios else (0.0, None)
    failed = {}
    for r in rows:
        for name in r["failed_tests"]:
            failed.setdefault(name, []).append(r["layer"])
    summary = {
        "layers": layers,
        "gate": "floor" if floor is not None else "min_pearson",
        "worst_pearson": worst["pearson"],
        "worst_pearson_layer": worst["layer"],
        "worst_max_abs": worst_abs,
        "diverged_tokens": max(r["diverged"] for r in rows),
        "diverged_slack": slack,
        "control_ratio": control_ratio,
        "control_ratio_layer": control_layer,
        "passed": all(r["passed"] for r in rows),
        "failed_layers": failed,
        "control_detected": bool(ratios) and control_ratio >= control_factor,
        "control_factor": control_factor,
    }
    if floor is None:
        worst_token = min(rows, key=lambda r: r["token_pearson"])
        summary["min_pearson"] = min_pearson
        summary["worst_token_pearson"] = worst_token["token_pearson"]
        summary["worst_token_pearson_layer"] = worst_token["layer"]
    else:
        worst_ratio = max(rows, key=lambda r: r["ratio"])
        summary["floor_factor"] = floor_factor
        summary["worst_ratio"] = worst_ratio["ratio"]
        summary["worst_ratio_layer"] = worst_ratio["layer"]
        summary["floor_diverged_tokens"] = max(r["floor_diverged"] for r in rows)
        summary["ratios"] = [round(r["ratio"], 4) for r in rows]
    return rows, summary


def engine_layer_override(current, num_layers: int) -> str:
    """`json_model_override_args` with `num_hidden_layers` set to `num_layers`, as a JSON string.

    `current` is what `--engine-arg json_model_override_args=...` gave: None, a dict, or a JSON
    string of one. A different `num_hidden_layers` in it raises.
    """
    if current is None:
        merged = {}
    elif isinstance(current, dict):
        merged = dict(current)
    else:
        merged = json.loads(current)
        if not isinstance(merged, dict):
            raise ValueError(f"json_model_override_args is {current!r}, not a JSON object")
    if merged.get("num_hidden_layers", num_layers) != num_layers:
        raise ValueError(
            f"--engine-layers {num_layers} and json_model_override_args num_hidden_layers "
            f"{merged['num_hidden_layers']} disagree")
    merged["num_hidden_layers"] = num_layers
    return json.dumps(merged, sort_keys=True)


def cut_entries(entries: list, num_layers: int) -> list:
    """The first `num_layers` reference entries, the streams entering layers 0 to N-1.

    The last entry of a reference is the final norm's output, which no cut engine's slot holds,
    so `num_layers` has to be below the entry count.
    """
    if not 0 < num_layers < len(entries):
        raise ValueError(
            f"--engine-layers {num_layers}: the reference holds {len(entries)} entries, the "
            f"streams entering {len(entries) - 1} layers plus the final norm's output")
    return entries[:num_layers]


def engine_capture(args, batch: list, extra: dict) -> list:
    """The engine's prefill capture for each request in one call, `[seq, layers, d]` apiece.

    `extra` holds the parsed `--engine-arg` keywords.
    """
    from capture_activations import hidden_states_from_output, log_replies, log_request, open_engine

    engine = open_engine(
        args.model_path,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        token_padding=args.token_padding,
        chunked_prefill_size=args.chunked_prefill_size,
        **extra,
    )
    try:
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        log_request(call="generate", input_ids=batch, sampling_params=sampling,
                    return_hidden_states=True)
        out = engine.generate(input_ids=batch, sampling_params=sampling, return_hidden_states=True)
        log_replies(out)
        captured = []
        for ids, reply in zip(batch, out):
            array, prompt_rows = hidden_states_from_output(reply)
            if prompt_rows != len(ids):
                raise ValueError(f"a {len(ids)}-token prompt came back with {prompt_rows} prefill rows")
            captured.append(array[:prompt_rows])
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return captured


def ceil_block_dequantize(quantized, scales, block, output_dtype):
    """Block-FP8 weights to `output_dtype` when a dimension isn't a multiple of the block.

    DeepSeek-style checkpoints scale each `block` tile of a weight, and the last tile along a
    dimension that isn't a multiple of the block is partial: GLM-5.3's `kv_a_proj_with_mqa` is
    576 x 6,144 with a 5 x 48 scale grid at block 128. transformers 5.17 derives the block by
    dividing the weight by the grid, so it refuses these. Each scale repeats over its tile and the
    padding comes off, which is how sglang and vLLM read the same checkpoint.
    """
    import torch

    q = quantized.to(torch.float32)
    rows, cols = q.shape[-2:]
    s = scales.to(torch.float32)
    if scales.dtype == torch.uint8:
        s = (s - 127.0).exp2()
    bm, bn = block
    s = s.repeat_interleave(bm, dim=-2)[..., :rows, :].repeat_interleave(bn, dim=-1)[..., :cols]
    return (q * s).to(output_dtype)


def patch_fp8_ceil_blocks():
    """Let transformers' FP8 dequantizer take partial tiles, through `ceil_block_dequantize`.

    It changes nothing for a weight transformers already handles: the fallback runs only on the
    "not divisible by scale grid" error, with the block from the checkpoint's
    `weight_block_size`, default 128 x 128.
    """
    try:
        from transformers.integrations import finegrained_fp8
    except ImportError:
        return
    cls = getattr(finegrained_fp8, "Fp8Dequantize", None)
    if cls is None or getattr(cls, "_ceil_blocks", False):
        return
    original = cls._dequantize_one

    def dequantize_one(self, quantized, scales, output_dtype=None):
        try:
            return original(self, quantized, scales, output_dtype=output_dtype)
        except ValueError as exc:
            if "not divisible by scale grid" not in str(exc):
                raise
            import torch

            cfg = getattr(getattr(self, "hf_quantizer", None), "quantization_config", None)
            block = tuple(getattr(cfg, "weight_block_size", None) or (128, 128))
            return ceil_block_dequantize(quantized, scales, block, output_dtype or torch.bfloat16)

    cls._dequantize_one = dequantize_one
    cls._ceil_blocks = True


def model_class(config):
    """The transformers auto class that builds this checkpoint's text model.

    A causal LM goes through `AutoModelForCausalLM`. A multimodal checkpoint such as Inkling's
    registers only an image-text class, and its forward on text ids alone runs the language model,
    so the reference takes that class and reads the language model's hidden states.
    """
    import transformers

    for cls in (transformers.AutoModelForCausalLM, transformers.AutoModelForImageTextToText):
        if type(config) in cls._model_mapping:
            return cls
    return transformers.AutoModelForCausalLM


def decoder_layers(model):
    """`(prefix, layers)`: the model's decoder block list and its dotted name.

    It's the shallowest `nn.ModuleList` named `layers`, `h` or `blocks`, with the config's layer
    count when the config gives one. Nemotron 3 Ultra's config has no `num_hidden_layers`, and an
    expert list can be longer than the decoder list, so length alone can't pick it.
    """
    import torch

    want = getattr(model.config, "num_hidden_layers", None)
    text = getattr(model.config, "text_config", None)
    if want is None and text is not None:
        want = getattr(text, "num_hidden_layers", None)
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if name.rsplit(".", 1)[-1] not in ("layers", "h", "blocks"):
            continue
        if want is not None and len(module) != want:
            continue
        found.append((name.count("."), name, module))
    if not found:
        raise ValueError("no decoder layer list named layers, h or blocks in the model")
    _, name, module = min(found, key=lambda f: f[0])
    return name, module


def release_freed_memory() -> None:
    """Hand freed heap memory back to the OS after a streamed layer drops its weights.

    glibc keeps freed blocks under its mmap threshold, which grows to 32 MiB, in its heap arenas.
    Kimi K3's BF16 expert weights sit under it, so each streamed BF16 layer left about 55 GiB of
    freed heap resident: 7 layers reached 327 GiB anonymous RSS with 4 GiB of live tensors, and
    the full BF16 pass was killed on a 354 GiB and then a 732 GiB machine (2026-09-26). With
    `malloc_trim(0)` after each layer the same 5 layers held 0 to 1 GiB. No-op off glibc.
    """
    import ctypes

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def cut_config(config, num_layers: int | None):
    """Cut the text config to its first `num_layers` layers, in place, and return it.

    The layer count sits on `text_config` for a multimodal checkpoint. Every list attribute as
    long as the old count is a per-layer plan (`layer_types`, `mlp_layer_types`,
    `indexer_types`) and gets cut too, since transformers validates the two against each other.
    """
    if num_layers is None:
        return config
    text = getattr(config, "text_config", None) or config
    old = int(text.num_hidden_layers)
    if not 0 < num_layers <= old:
        raise ValueError(f"--reference-layers {num_layers}: the model has {old} layers")
    for key, value in list(vars(text).items()):
        if isinstance(value, list) and len(value) == old:
            setattr(text, key, value[:num_layers])
    text.num_hidden_layers = num_layers
    return config


def flat_entry(h) -> np.ndarray:
    """One hidden-states entry of a batch-1 forward as `[seq, width]` float64: a multi-copy stream
    `[seq, copies, d]` flattens to `[seq, copies * d]`, the engine's slot layout."""
    import torch

    a = h[0]
    return a.reshape(a.shape[0], -1).to(torch.float64).numpy()


def checkpoint_value(loader, name: str, as_stored: bool):
    """`loader[name]`, or with `as_stored` the tensor at the dtype its safetensors file holds.

    accelerate's `OffloadedWeightsLoader` indexes a checkpoint tensor that needs no conversion in
    place, and its index entry carries the dtype `from_pretrained` loaded it at. `streamed_model`
    loads at BF16, so `loader[name]` rounds a float32 checkpoint tensor to BF16 before the float32
    pass can upcast it. Nemotron 3 Ultra's router correction bias sits at 56.98 with experts
    about 0.005 apart, and BF16's step there is 0.25: every expert reads 57.0, and the reference
    routes as if the bias were one constant. The float32 pass and the modules transformers keeps
    in float32 read the file's own dtype here. A tensor that from_pretrained converted and wrote to
    the offload folder comes back from the loader as before.
    """
    info = getattr(loader, "index", {}).get(name) or {}
    if not as_stored or name in getattr(loader, "state_dict", {}) or not info.get("safetensors_file"):
        return loader[name]
    from safetensors import safe_open

    with safe_open(info["safetensors_file"], framework="pt", device="cpu") as f:
        return f.get_tensor(info.get("weight_name", name))


def streamed_model(model_path: str, dtype_name: str, offload_folder: str,
                   num_layers: int | None = None):
    """A HuggingFace model whose decoder layers load one at a time, for a model bigger than RAM.

    `from_pretrained` runs once with every decoder layer on "disk", so transformers converts the
    checkpoint's names and layouts itself and writes the converted weights to `offload_folder`
    (or indexes the safetensors in place). Everything else stays in RAM. accelerate's hooks come
    off, and a hook on each layer loads that layer's weights just before it runs and drops them
    after, so RAM holds one layer at a time.

    It builds transformers' own class for the model, as the plain path does, and never a repo's
    remote code, so both paths run the same modeling file.

    For float32 every floating weight upcasts from what the checkpoint holds, which is what
    `from_pretrained(dtype=float32)` gives for a BF16 checkpoint. For bfloat16 each weight keeps
    the dtype `from_pretrained` loaded it in, so modules transformers keeps in float32 stay there.
    """
    import torch
    import transformers
    from accelerate import init_empty_weights
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import set_module_tensor_to_device

    upcast = dtype_name == "float32"
    config = cut_config(transformers.AutoConfig.from_pretrained(model_path), num_layers)
    with init_empty_weights():
        skeleton = model_class(config).from_config(config)
    prefix, layers = decoder_layers(skeleton)
    # accelerate's dispatch refuses a root "" entry beside "disk" entries (transformers 5.17,
    # accelerate 1.15), so name every module beside the path down to the layer list.
    device_map = {}
    parts = prefix.split(".")
    for depth in range(len(parts)):
        parent = skeleton.get_submodule(".".join(parts[:depth])) if depth else skeleton
        if any(True for _ in parent.named_parameters(recurse=False)):
            raise ValueError(f"{'.'.join(parts[:depth]) or 'the model'} holds parameters of its own")
        for child, _ in parent.named_children():
            if child != parts[depth]:
                device_map[".".join(parts[:depth] + [child])] = "cpu"
    device_map.update({f"{prefix}.{i}": "disk" for i in range(len(layers))})
    del skeleton
    os.makedirs(offload_folder, exist_ok=True)
    model = model_class(config).from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, device_map=device_map,
        offload_folder=offload_folder,
    )
    loader = None
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is not None:
            loader = getattr(weights_map, "dataset", weights_map)
            break
    if loader is None:
        raise RuntimeError("from_pretrained offloaded nothing, so there's no weight map to stream from")
    remove_hook_from_module(model, recurse=True)
    if upcast:
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.device.type != "meta" and tensor.is_floating_point():
                tensor.data = tensor.data.to(torch.float32)
    _, layers = decoder_layers(model)
    # from_pretrained keeps these modules in float32 even at BF16 (Inkling's short convolutions),
    # and accelerate's meta tensors forget it, so the BF16 pass applies the rule itself.
    keep_fp32 = set(getattr(model, "_keep_in_fp32_modules_strict", None) or [])

    # accelerate materializes some small tensors at dispatch, through the same BF16 cast, so a
    # tensor the offload index knows reloads each time even when it already sits on the CPU.
    # On a tiny Nemotron-H the router correction bias arrives that way, as 57.0 for every expert.
    indexed = set(getattr(loader, "index", {}) or {})

    def names(i, layer):
        return [(n, f"{prefix}.{i}.{n}") for n, t in
                list(layer.named_parameters()) + list(layer.named_buffers())
                if t.device.type == "meta" or f"{prefix}.{i}.{n}" in indexed]

    def load(i):
        def hook(layer, args, kwargs=None):
            for local, full in names(i, layer):
                keep = bool(keep_fp32 & set(full.split(".")))
                value = checkpoint_value(loader, full, upcast or keep)
                if upcast and value.is_floating_point():
                    value = value.to(torch.float32)
                # Without dtype=, accelerate casts the value to the meta tensor's dtype, the one
                # from_pretrained chose: BF16, or float32 for a module transformers keeps there.
                # The BF16 pass wants that. The float32 pass passes float32, or the upcast is lost.
                keep = keep and value.is_floating_point()
                set_module_tensor_to_device(
                    layer, local, "cpu", value=value,
                    dtype=torch.float32 if (upcast or keep) and value.is_floating_point() else None)
        return hook

    def drop(layer, args, output):
        for local, t in list(layer.named_parameters()) + list(layer.named_buffers()):
            if t.device.type == "cpu":
                set_module_tensor_to_device(layer, local, "meta")
        gc.collect()
        release_freed_memory()

    for i, layer in enumerate(layers):
        loaded = names(i, layer)
        if not loaded:
            continue
        layer.register_forward_pre_hook(load(i))
        layer.register_forward_hook(drop)
        layer._streamed_names = loaded
    print(f"streamed reference: {len(layers)} layers under {prefix!r} load one at a time from "
          f"{offload_folder}, {dtype_name}", flush=True)
    return model


def norm_embeddings_once(model) -> dict:
    """Let every module named `embed_norm` run once per forward of `model`, and pass later calls through.

    `transformers` 5.17 norms Inkling's embeddings twice. `InklingModel.forward` applies
    `language_model.embed_norm`, then `InklingTextModel.forward` applies the same module again, so
    the stream entering layer 0 is `embed_norm(embed_norm(embed))`. The published weight has mean
    0.17 and max 7.6, so the second norm moves that stream by 0.94 of its norm, median over
    tokens. Upstream commit 3384908511 ("Fix inkling embedding norm", #48786, 2026-09-14) moves the
    norm into the embedding, where it runs once, and SGLang's PyTorch model and this repo's engine
    port run it once too. On a `transformers` with the fix the module runs once and the hook never
    changes an output.

    A pre-hook on `model` resets the count at each forward. Returns the counts: `skipped` is how
    many second calls passed their input through.
    """
    counts = {"modules": 0, "calls": 0, "skipped": 0}
    norms = [m for name, m in model.named_modules() if name.rsplit(".", 1)[-1] == "embed_norm"]
    if not norms:
        return counts
    counts["modules"] = len(norms)

    def reset(module, args):
        counts["calls"] = 0

    def once(module, args, output):
        counts["calls"] += 1
        if counts["calls"] > 1:
            counts["skipped"] += 1
            return args[0]
        return output

    model.register_forward_pre_hook(reset)
    for norm in norms:
        norm.register_forward_hook(once)
    return counts


def reference_forward(model_path: str, batch: list, dtype_name: str,
                      offload_folder: str | None = None, trust_remote_code: bool = False,
                      num_layers: int | None = None, deepseek_inference: bool = False) -> list:
    """HuggingFace hidden states on the host CPU: per prompt, one `[seq, d]` float64 array per entry.

    Each prompt runs alone, so no padding or attention mask enters the reference. With
    `offload_folder` the decoder layers stream from disk one at a time (`streamed_model`).

    `trust_remote_code` runs the checkpoint's own modeling file through `remote_code_reference`,
    which Kimi K3 needs. There `offload_folder` only turns streaming on: the layers stream from the
    checkpoint's safetensors in place and nothing goes to the folder. `num_layers` builds only
    the first layers, for a dry run.

    `deepseek_inference` runs the checkpoint's own `inference/model.py`, DeepSeek V4.1-Flash's
    runtime, through `deepseek_reference` on CPU stand-ins for its CUDA kernels. There too
    `offload_folder` only turns streaming on.
    """
    import torch
    import transformers

    if deepseek_inference:
        import deepseek_reference

        return deepseek_reference.reference_forward(
            model_path, batch, dtype_name, streamed=bool(offload_folder), num_layers=num_layers)
    if trust_remote_code:
        import remote_code_reference

        return remote_code_reference.reference_forward(
            model_path, batch, dtype_name, streamed=bool(offload_folder), num_layers=num_layers)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    patch_fp8_ceil_blocks()
    if offload_folder:
        model = streamed_model(model_path, dtype_name, offload_folder, num_layers)
    else:
        config = cut_config(transformers.AutoConfig.from_pretrained(model_path), num_layers)
        model = model_class(config).from_pretrained(model_path, config=config, dtype=dtype)
    model.eval()
    norm_counts = norm_embeddings_once(model)
    states = []
    with torch.no_grad():
        for n, ids in enumerate(batch):
            # No cache: the reference runs each prompt once, and a cut GLM-5.3-Flash of KDA layers
            # alone has no attention layer for transformers' cache to read a length from.
            out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
            states.append([flat_entry(h) for h in out.hidden_states])
            del out
            if norm_counts["modules"]:
                print(f"{dtype_name} reference, prompt {n}: embed_norm ran {norm_counts['calls']} "
                      f"time(s); {'the second call passed its input through' if norm_counts['calls'] > 1 else 'nothing skipped'}",
                      flush=True)
    del model
    gc.collect()
    return states


def _stack_entries(entries: list, key: str, arrays: dict) -> None:
    """`key` holds the entries stacked. When the last entry is narrower than the rest, as the
    final norm is beside a model's multi-copy streams, it goes under `key_last` instead."""
    entries = [np.asarray(e, np.float32) for e in entries]
    if len(entries) > 1 and entries[-1].shape != entries[0].shape:
        arrays[key] = np.stack(entries[:-1])
        arrays[key + "_last"] = entries[-1]
    else:
        arrays[key] = np.stack(entries)


def _unstack_entries(data, key: str) -> list:
    out = [a.astype(np.float64) for a in data[key]]
    if key + "_last" in data.files:
        out.append(data[key + "_last"].astype(np.float64))
    return out


def save_reference(path: str, batch: list, reference: list, floor: list | None) -> str:
    """Write the token ids and the reference forwards, float32, so a TPU host can read them."""
    if not path.endswith(".npz"):
        path += ".npz"
    arrays = {}
    for n, ids in enumerate(batch):
        arrays[f"ids{n}"] = np.asarray(ids, np.int64)
        _stack_entries(reference[n], f"reference{n}", arrays)
        if floor:
            _stack_entries(floor[n], f"floor{n}", arrays)
    np.savez(path, **arrays)
    return path


def load_reference(path: str, batch: list, no_floor: bool):
    """The reference and floor `save_reference` wrote, per prompt as lists of `[seq, d]` arrays.

    Every prompt's token ids have to equal the ids this run built, or the arrays belong to other
    prompts. Values come back float64, as `reference_forward` returns them. They went through a
    float32 file, which holds both the BF16 floor and the float32 reference without rounding.
    """
    data = np.load(path)
    stored = sorted(int(k[3:]) for k in data.files if k.startswith("ids"))
    if stored != list(range(len(batch))):
        raise SystemExit(f"{path} holds {len(stored)} prompt(s); this run built {len(batch)}")
    reference, floor = [], []
    for n, ids in enumerate(batch):
        if list(data[f"ids{n}"]) != list(ids):
            raise SystemExit(f"prompt {n}: the token ids in {path} differ from this run's")
        reference.append(_unstack_entries(data, f"reference{n}"))
        if not no_floor:
            if f"floor{n}" not in data.files:
                raise SystemExit(f"{path} has no bf16 floor; pass --no-floor or rerun --reference-only")
            floor.append(_unstack_entries(data, f"floor{n}"))
    return reference, (floor if not no_floor else None)


def save_capture(path: str, batch: list, captured: list) -> str:
    """Write each prompt's token ids and its engine capture, `[seq, slots, d]` float32."""
    if not path.endswith(".npz"):
        path += ".npz"
    arrays = {}
    for n, (ids, got) in enumerate(zip(batch, captured)):
        arrays[f"ids{n}"] = np.asarray(ids, np.int64)
        arrays[f"capture{n}"] = np.asarray(got, np.float32)
    np.savez(path, **arrays)
    return path


def load_capture(path: str, batch: list) -> list:
    """The captures `save_capture` or `--save-npz` wrote, one `[seq, slots, d]` array per prompt.

    Every prompt's token ids have to equal the ids this run built, or the capture belongs to
    other prompts. A `--save-npz` file from before the ids went into it can't show them, so it's
    refused.
    """
    data = np.load(path)
    stored = sorted(int(k[7:]) for k in data.files if k.startswith("capture"))
    if stored != list(range(len(batch))):
        raise SystemExit(f"{path} holds {len(stored)} capture(s); this run built {len(batch)} prompts")
    captured = []
    for n, ids in enumerate(batch):
        if f"ids{n}" not in data.files:
            raise SystemExit(f"{path} holds no token ids for prompt {n}, so nothing ties the "
                             "capture to this run's prompts")
        if not np.array_equal(data[f"ids{n}"], np.asarray(ids, np.int64)):
            raise SystemExit(f"prompt {n}: {path} holds other token ids than this run built")
        got = data[f"capture{n}"].astype(np.float32)
        if got.shape[0] != len(ids):
            raise SystemExit(f"prompt {n}: {path} holds {got.shape[0]} rows for {len(ids)} tokens")
        captured.append(got)
    return captured


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", required=True, help="the local snapshot directory")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompts-file", help="a prompt file; its first --num-prompts go in one engine call")
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--no-bos", action="store_true", help="send the tokenizer's ids with no BOS added")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="the engine's precompile bucket")
    ap.add_argument("--token-padding", type=int, default=1024)
    ap.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=CHUNKED_PREFILL_SIZE,
        help="tokens per prefill pass; the joined --prompts-file prompt has to hold more",
    )
    ap.add_argument("--floor-factor", type=float, default=2.0)
    ap.add_argument("--no-floor", action="store_true", help="skip the BF16 forward; gate on --min-pearson")
    ap.add_argument(
        "--min-pearson",
        type=float,
        default=0.999,
        help="the gate with --no-floor, on each layer's median per-token Pearson",
    )
    ap.add_argument("--control-factor", type=float, default=3.0)
    ap.add_argument("--save-npz", help="write every capture and reference array here, for a closer look")
    ap.add_argument(
        "--save-capture",
        metavar="NPZ",
        help="write the engine capture and the token ids to NPZ, so --capture-npz can gate it again "
        "against another reference without the engine",
    )
    ap.add_argument(
        "--capture-npz",
        metavar="NPZ",
        help="read the capture from an NPZ that --save-capture or --save-npz wrote, instead of "
        "starting the engine; the token ids have to match",
    )
    ap.add_argument(
        "--offload-folder",
        help="stream the reference one decoder layer at a time through this folder, for a model "
        "bigger than host RAM",
    )
    ap.add_argument(
        "--reference-only",
        metavar="NPZ",
        help="run the reference forwards, write them and the token ids to NPZ, and stop before "
        "the engine; for a CPU VM that holds the checkpoint",
    )
    ap.add_argument(
        "--reference-npz",
        help="read the reference forwards from an NPZ that --reference-only wrote, instead of "
        "running them; the token ids have to match",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="run the checkpoint's own modeling code for the reference, with a pure-torch fla when "
        "fla isn't installed; Kimi K3 needs it",
    )
    ap.add_argument(
        "--deepseek-inference",
        action="store_true",
        help="run the checkpoint's own inference/model.py for the reference, on CPU stand-ins for "
        "its CUDA kernels; DeepSeek V4.1-Flash needs it",
    )
    ap.add_argument(
        "--reference-layers",
        type=int,
        metavar="N",
        help="build only the first N decoder layers, for a dry run of the reference; needs "
        "--reference-only",
    )
    ap.add_argument(
        "--engine-layers",
        type=int,
        metavar="N",
        help="serve only the first N decoder layers, through json_model_override_args, and "
        "compare slots 0 to N-1 against reference entries 0 to N-1",
    )
    ap.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Engine keyword, repeatable; VALUE reads as JSON, True, False, None or a string",
    )
    ap.add_argument(
        "--log-payloads",
        action="store_true",
        help="log the engine request and a summary of each reply to stderr; LOG_PAYLOADS=1 too",
    )
    args = ap.parse_args(argv)
    if args.reference_layers is not None and not args.reference_only:
        ap.error("--reference-layers needs --reference-only: a cut model can't gate a capture")
    if args.trust_remote_code and args.deepseek_inference:
        ap.error("--trust-remote-code and --deepseek-inference pick two different references")
    if args.capture_npz and args.reference_only:
        ap.error("--capture-npz gates a saved capture, and --reference-only stops before any gate")
    if args.capture_npz and args.save_capture:
        ap.error("--capture-npz reads a capture, so there's no new one for --save-capture to write")
    if args.capture_npz and args.engine_layers is not None:
        ap.error("--engine-layers cuts the engine, and --capture-npz never starts it")

    from capture_activations import (
        enable_payload_log,
        parse_engine_args,
        payload_log_requested,
        read_prompts,
    )

    try:
        extra = parse_engine_args(args.engine_arg, reserved=RESERVED_ENGINE_ARGS)
    except ValueError as exc:
        ap.error(str(exc))
    if args.engine_layers is not None:
        if args.engine_layers < 1:
            ap.error("--engine-layers takes a count of 1 or more")
        if args.reference_only:
            ap.error("--engine-layers cuts the engine, and --reference-only never starts it")
        try:
            extra["json_model_override_args"] = engine_layer_override(
                extra.get("json_model_override_args"), args.engine_layers)
        except ValueError as exc:
            ap.error(str(exc))
        print(f"engine cut to its first {args.engine_layers} decoder layers: "
              f"json_model_override_args={extra['json_model_override_args']}")
    if payload_log_requested(args.log_payloads):
        enable_payload_log()
    save_npz = args.save_npz
    if save_npz and not save_npz.endswith(".npz"):
        save_npz += ".npz"

    if int(os.environ.get("SGL_NODE_RANK", "0") or 0) > 0 and not args.reference_only:
        # A non-zero rank of a multi-host slice starts its engine and blocks in the scheduler.
        # Only rank 0 sends prompts and compares, so only rank 0 reads the prompt file and the
        # reference; both live on host 0 alone (multihost_exec.sh copies nothing by default).
        # With --capture-npz no rank starts an engine.
        if not args.capture_npz:
            engine_capture(args, [], extra)
        os._exit(0)

    from transformers import AutoTokenizer

    if args.prompts_file:
        corpus = read_prompts(args.prompts_file)
        need = args.num_prompts + 3
        if len(corpus) < need:
            raise SystemExit(
                f"{args.prompts_file} holds {len(corpus)} prompt(s). --prompts-file needs "
                f"--num-prompts + 3 = {need}: the last three join into the prompt that has to split "
                f"across prefill passes."
            )
        prompts = corpus[: args.num_prompts]
        prompts.append("\n".join(corpus[args.num_prompts : need]))
    else:
        prompts = [args.prompt]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    bos = tokenizer.bos_token_id
    batch = []
    for text in prompts:
        ids = tokenizer(text)["input_ids"]
        added = not args.no_bos and bos is not None and (not ids or ids[0] != bos)
        batch.append([bos] + ids if added else ids)
        print(f"prompt {len(batch) - 1}: {len(batch[-1])} tokens{', BOS added' if added else ''}")
    if args.prompts_file and len(batch[-1]) <= args.chunked_prefill_size:
        raise SystemExit(
            f"prompt {len(batch) - 1}, joined from three corpus prompts, holds {len(batch[-1])} "
            f"tokens and fits one {args.chunked_prefill_size}-token prefill pass, so nothing "
            f"splits and the check can't reach the chunked-prefill path. Use longer prompts, such "
            f"as build_corpus.py's 440-token ones, or a lower --chunked-prefill-size."
        )
    if len(batch) > args.batch_size:
        raise SystemExit(f"{len(batch)} prompts need --batch-size {len(batch)} or more")

    if args.reference_npz:
        reference, floor = load_reference(args.reference_npz, batch, args.no_floor)
        print(f"reference read from {args.reference_npz}: {len(reference[0])} entries per prompt"
              f"{', with the bf16 floor' if floor else ''}")
    else:
        remote = dict(trust_remote_code=args.trust_remote_code, num_layers=args.reference_layers,
                      deepseek_inference=args.deepseek_inference)
        reference = reference_forward(args.model_path, batch, "float32", args.offload_folder, **remote)
        print(f"float32 reference: {len(reference[0])} entries per prompt")
        floor = None
        if not args.no_floor:
            floor = reference_forward(args.model_path, batch, "bfloat16", args.offload_folder, **remote)
            print("bf16 floor: done")
    if args.reference_only:
        path = save_reference(args.reference_only, batch, reference, floor)
        print(f"wrote {path}: {len(batch)} prompt(s), float32 reference"
              f"{' and bf16 floor' if floor else ''}. The engine didn't run.")
        os._exit(0)
    if args.engine_layers is not None:
        try:
            reference = [cut_entries(r, args.engine_layers) for r in reference]
            floor = [cut_entries(f, args.engine_layers) for f in floor] if floor else floor
        except ValueError as exc:
            raise SystemExit(str(exc))
        print(f"comparing reference entries 0 to {args.engine_layers - 1}")
    if args.capture_npz:
        captured = load_capture(args.capture_npz, batch)
        print(f"capture read from {args.capture_npz}: " + ", ".join(str(c.shape) for c in captured))
    else:
        captured = engine_capture(args, batch, extra)
        print("engine capture: " + ", ".join(str(c.shape) for c in captured))
    if args.engine_layers is not None:
        for n, got in enumerate(captured):
            if got.shape[1] != args.engine_layers:
                raise SystemExit(
                    f"prompt {n}: --engine-layers {args.engine_layers}, and the capture holds "
                    f"{got.shape[1]} slots, so the model didn't take the cut")
    if args.save_capture:
        print(f"capture written to {save_capture(args.save_capture, batch, captured)}: "
              f"{len(batch)} prompt(s) with their token ids")
    if save_npz:
        arrays = {}
        for n, got in enumerate(captured):
            arrays[f"ids{n}"] = np.asarray(batch[n], np.int64)
            arrays[f"capture{n}"] = got.astype(np.float32)
            _stack_entries(reference[n], f"reference{n}", arrays)
            if floor:
                _stack_entries(floor[n], f"floor{n}", arrays)
        np.savez(save_npz, **arrays)
        print(f"wrote {save_npz}")

    ok = True
    for n, got in enumerate(captured):
        rows, summary = compare_layers(
            got, reference[n], args.min_pearson, args.control_factor,
            floor[n] if floor else None, args.floor_factor,
        )
        summary["prompt"] = n
        summary["tokens"] = got.shape[0]
        if len(captured) > 1:
            print(f"prompt {n}, {got.shape[0]} tokens")
        for r in rows:
            mark = "PASS" if r["passed"] else "FAIL"
            why = f"  <- fails on {' and '.join(r['failed_tests'])}" if r["failed_tests"] else ""
            if floor:
                print(
                    f"  [{mark}] layer {r['layer']:3d}  token err {r['token_error']:.5f}  "
                    f"bf16 {r['floor_token_error']:.5f}  ratio {r['ratio']:5.2f}  "
                    f"pearson {r['pearson']:.6f} (bf16 {r['floor_pearson']:.6f})  "
                    f"diverged {r['diverged']} (bf16 {r['floor_diverged']}, "
                    f"allowed {r['diverged_allowed']:g}){why}"
                )
            else:
                print(
                    f"  [{mark}] layer {r['layer']:3d}  token pearson {r['token_pearson']:.8f}  "
                    f"pearson {r['pearson']:.8f}  max_abs {r['max_abs']:.4e}  "
                    f"diverged {r['diverged']} (allowed {r['diverged_allowed']:g}){why}"
                )
        print(
            f"      control (each slot against the next layer, per token): closest layer "
            f"{summary['control_ratio_layer']} sits {summary['control_ratio']:.1f}x closer to its own -> "
            f"{'detected' if summary['control_detected'] else 'NOT DETECTED'}"
        )
        for name, failed_at in summary["failed_layers"].items():
            print(f"      {len(failed_at)} layer(s) fail on {name}: {failed_at}")
        print("RESULT " + json.dumps(summary))
        ok = ok and summary["passed"] and summary["control_detected"]
    print("All checks passed." if ok else "FAILED")
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
