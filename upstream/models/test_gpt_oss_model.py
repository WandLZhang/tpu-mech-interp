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

"""Correctness gate for the gpt-oss model patch.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 upstream/models/test_gpt_oss_model.py

The test builds the tree the TPU VM serves from, `sglang-jax` at `eb061d8` with
`sglang-jax-877.patch` and both steering patches, through `scripts/cpu_engine.build_tree`,
and applies this patch on top. Then the patched `sgl_jax.srt.models.gpt_oss` is imported and
run against HuggingFace `transformers.GptOssForCausalLM` at float32. Both models get the same
random weights, from a real MXFP4 checkpoint written to disk with a config.json that carries
the published `mxfp4` quantization dict. The JAX side loads it the way the runner does,
through `ModelConfig` and `JAXModelLoader`, and runs on the engine's `ForwardBatch` and
`MemoryPools`. The HuggingFace side gets the same tensors decoded by
`transformers.integrations.mxfp4.convert_moe_packed_tensors`, which is the decoder
`transformers` itself runs on a real gpt-oss checkpoint.

Ten checks.

1. The patch applies to that tree and every file it touches compiles. Control: the same
   patch with one context line rewritten must be refused.
2. The MXFP4 decoder in the patch matches a decoder written from the OCP microscaling spec
   over every code point, every exponent, and a random packed tensor, and matches
   `transformers`' own decoder including the axis layout. Control: reversing the nibble
   order must disagree.
3. Both sides run YaRN and agree on every inverse frequency.
4. Every per-layer residual stream matches HuggingFace, plus the final norm output and the
   logits. Reported per layer as max absolute error and Pearson correlation.
5. Six mutants of the loaded model must fail check 4's tolerance.
6. The LM head: the shipped `LogitsProcessor` drives both the untied and the tied branch.
7. The capture hook: `layers_to_capture` fills `aux_hidden_states` in order, the flag on
   the entry class gates it, the logits processor receives it, and the two production
   setters place the layers they claim.
8. EPLB redundant experts: the loader fills every physical slot, both dispatch maps route
   ids sharded over the batch, and a forward that carries the static map in its
   `ForwardBatch` sends logical experts 0 and 1 to slots 8 and 9. With slots 0 and 1
   poisoned, that forward still matches HuggingFace. Control: the same forward without the
   map reads the poisoned slots and misses.
9. Quantization configs: the checkpoint's `mxfp4` dict stays out of `EPMoE`, the
   built-in `int8.yaml` quantizes the routed experts, and int8 experts 1,024 wide match
   numpy. Under `int8_w8a8.yaml`, where `gmm` quantizes the activations too, each expert
   bias adds whole. Controls: handing `EPMoE` the dict breaks the load, and the int8
   output lands more than 1e-3 off the float weights.
10. A dummy-weight load at `--tp-size 8` never holds a whole expert stack on one device.
    Control: with the model's own expert fill turned off, the loader's pass leaves the
    stacks whole on device 0.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download. `SGL_COMMIT`
picks the commit, as it does for every gate.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import tempfile

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "gpt-oss-model.patch")

# `scripts/cpu_engine.py` builds the tree bootstrap_tpu_vm.sh builds, and the capture-hook
# suite carries the git helpers every patch suite shares.
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "scripts"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "capture-hooks"))
import cpu_engine  # noqa: E402
from test_capture_hooks import corrupt, git  # noqa: E402


def touched_files() -> list[str]:
    """Every file the patch touches, read off its `diff --git` headers."""
    with open(PATCH) as handle:
        return [
            line.rstrip("\n").split(" b/", 1)[1]
            for line in handle
            if line.startswith("diff --git a/")
        ]

# float32 end to end. The gate sits well above the accumulation-order floor the two
# frameworks produce and well below what any mutant produces. Every mutant has to fail
# this same gate, so the clean run and the controls are separated by one number: the
# clean worst is 1.3e-05 and the weakest control is 6.5e-03.
TOL = 2e-4
MIN_PEARSON = 1.0 - 1e-8

# A tiny gpt-oss. Same architecture flags as the 20b and 120b configs: alternating
# sliding/full attention, attention bias on, GQA, MXFP4 experts, YaRN with truncate off.
# hidden_size and intermediate_size are multiples of 32 so the MXFP4 blocks divide evenly.
TINY = {
    "hidden_size": 128,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_local_experts": 8,
    "num_experts_per_tok": 4,
    "vocab_size": 256,
    "sliding_window": 4,
    "rms_norm_eps": 1e-5,
    "rope_theta": 150000.0,
    "max_position_embeddings": 32768,
    "swiglu_limit": 7.0,
    "attention_bias": True,
    "tie_word_embeddings": False,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
}
ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 8.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 4096,
    "truncate": False,
}
SEQ_LEN = 64
MXFP4_BLOCK = 32


# ---------------------------------------------------------------------------
# repo bootstrap
# ---------------------------------------------------------------------------


def get_checkout(workdir):
    """The tree the TPU VM serves from, which the patch goes on top of.

    `cpu_engine.build_tree` clones `SGLANG_JAX_REPO` with `--shared`, or GitHub without it,
    checks out `SGL_COMMIT`, takes `sglang-jax-877.patch` through `git am` and both steering
    patches after it, the steps `bootstrap_tpu_vm.sh` runs.
    """
    return cpu_engine.build_tree(
        os.path.join(workdir, "sglang-jax"), os.environ.get("SGLANG_JAX_REPO") or None
    )


def check_patch_applies(repo, workdir):
    """Apply the patch, compile what it touches, and refuse a corrupted copy of it."""
    failures = 0
    with open(PATCH) as handle:
        text = handle.read()

    dry = git(repo, "apply", "--check", PATCH)
    ok = dry.returncode == 0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] gpt-oss-model.patch applies to {cpu_engine.SGL_COMMIT}"
        " with the capture and steering patches"
    )
    if not ok:
        print(f"      {dry.stderr.strip()}")
        return 1

    bad = os.path.join(workdir, "corrupt.patch")
    with open(bad, "w") as handle:
        handle.write(corrupt(text))
    refused = git(repo, "apply", "--check", bad).returncode != 0
    print(f"  [{'PASS' if refused else 'FAIL'}] control: corrupted patch refused")
    failures += 0 if refused else 1

    applied = git(repo, "apply", PATCH)
    if applied.returncode != 0:
        print(f"      FAIL: apply failed after --check passed: {applied.stderr.strip()}")
        return failures + 1

    for path in touched_files():
        full = os.path.join(repo, path)
        exists = os.path.isfile(full)
        compiled = False
        if exists:
            with open(full) as handle:
                source = handle.read()
            try:
                compile(source, full, "exec")
                compiled = True
            except SyntaxError as exc:
                print(f"      {exc}")
        print(f"  [{'PASS' if compiled else 'FAIL'}] {path} compiles")
        failures += 0 if compiled else 1

    return failures


# ---------------------------------------------------------------------------
# independent MXFP4 decoder
#
# Written from the OCP microscaling spec, not from the patch. An FP4 E2M1 code is
# sign | 2-bit exponent | 1-bit mantissa. Exponent 0 is subnormal with value mantissa/2;
# otherwise the value is 2 ** (exponent - 1) * (1 + mantissa / 2). The shared scale is an
# E8M0 biased exponent, so its multiplier is 2 ** (scale - 127).
# ---------------------------------------------------------------------------


def fp4_e2m1_code_to_value(code: int) -> float:
    sign = -1.0 if code & 0b1000 else 1.0
    exponent = (code >> 1) & 0b11
    mantissa = code & 0b1
    if exponent == 0:
        magnitude = mantissa / 2.0
    else:
        magnitude = 2.0 ** (exponent - 1) * (1.0 + mantissa / 2.0)
    return sign * magnitude


def reference_dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode [..., out, k // 32, 16] uint8 blocks plus [..., out, k // 32] uint8 scales."""
    out = np.zeros((*blocks.shape[:-1], MXFP4_BLOCK), dtype=np.float32)
    flat_blocks = blocks.reshape(-1, blocks.shape[-1])
    flat_out = out.reshape(-1, MXFP4_BLOCK)
    for row in range(flat_blocks.shape[0]):
        for byte_index in range(flat_blocks.shape[1]):
            byte = int(flat_blocks[row, byte_index])
            flat_out[row, 2 * byte_index] = fp4_e2m1_code_to_value(byte % 16)
            flat_out[row, 2 * byte_index + 1] = fp4_e2m1_code_to_value(byte // 16)
    multiplier = np.ldexp(np.ones_like(scales, dtype=np.float32), scales.astype(np.int32) - 127)
    out = out * multiplier[..., None]
    return out.reshape(*blocks.shape[:-2], -1)


def random_mxfp4(rng, out_features: int, in_features: int, num_experts: int):
    """Random packed MXFP4 for one expert GEMM."""
    num_blocks = in_features // MXFP4_BLOCK
    blocks = rng.integers(
        0, 256, size=(num_experts, out_features, num_blocks, MXFP4_BLOCK // 2), dtype=np.uint8
    )
    # Hold the exponents below 2**0 so the tiny model's activations stay in a sane range.
    scales = rng.integers(120, 126, size=(num_experts, out_features, num_blocks)).astype(np.uint8)
    return blocks, scales


def transformers_dense(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode one packed expert GEMM the way `transformers` does.

    `convert_moe_packed_tensors` is what `transformers` runs on a real gpt-oss checkpoint,
    so it owns both the nibble order and the `[experts, out, in]` to `[experts, in, out]`
    layout. The spec reference above fixes the code table and the E8M0 bias, but the spec
    says nothing about which element of a 32-wide block a byte's low nibble holds. That's
    a serialization choice, and this is the party that made it.

    Returns float32 `[experts, in, out]`, which is the layout HuggingFace holds.
    """
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    decoded = convert_moe_packed_tensors(
        torch.from_numpy(np.ascontiguousarray(blocks)),
        torch.from_numpy(np.ascontiguousarray(scales)),
        dtype=torch.float32,
    )
    return decoded.numpy()


def check_dequantizer(gpt_oss) -> int:
    """The decoder in the patch against the spec and against `transformers`."""
    failures = 0

    blocks = np.arange(256, dtype=np.uint8).reshape(1, 16, 1, 16)
    flat_scale = np.full((1, 16, 1), 127, dtype=np.uint8)
    mine = gpt_oss.dequantize_mxfp4(blocks, flat_scale)
    theirs = reference_dequantize_mxfp4(blocks, flat_scale)
    ok = np.array_equal(mine, theirs)
    print(f"  [{'PASS' if ok else 'FAIL'}] all 256 byte values decode identically")
    failures += 0 if ok else 1

    exponents = np.arange(100, 155, dtype=np.uint8).reshape(1, 55, 1)
    blocks = np.full((1, 55, 1, 16), 0x72, dtype=np.uint8)
    mine = gpt_oss.dequantize_mxfp4(blocks, exponents)
    theirs = reference_dequantize_mxfp4(blocks, exponents)
    ok = np.array_equal(mine, theirs)
    print(f"  [{'PASS' if ok else 'FAIL'}] E8M0 exponents 100..154 decode identically")
    failures += 0 if ok else 1

    rng = np.random.default_rng(7)
    blocks, scales = random_mxfp4(rng, out_features=20, in_features=96, num_experts=3)
    theirs = reference_dequantize_mxfp4(blocks, scales)
    mine = gpt_oss.dequantize_mxfp4(blocks, scales)
    ok = np.array_equal(mine, theirs)
    print(f"  [{'PASS' if ok else 'FAIL'}] random [3, 20, 3, 16] tensor decodes identically")
    failures += 0 if ok else 1

    # The spec pins the code table and the E8M0 bias. It doesn't pin which element of a
    # block a byte's low nibble holds, and it doesn't pin the axis order. `transformers`
    # pins both, and the patch has to agree with it over the full exponent range.
    wide_scales = rng.integers(100, 150, size=scales.shape).astype(np.uint8)
    hf_layout = transformers_dense(blocks, wide_scales)
    ok = np.array_equal(gpt_oss.dequantize_mxfp4(blocks, wide_scales).transpose(0, 2, 1), hf_layout)
    print(f"  [{'PASS' if ok else 'FAIL'}] matches transformers convert_moe_packed_tensors")
    failures += 0 if ok else 1

    # Control: swapping the two halves of every byte has to move it off `transformers`.
    swapped_bytes = ((blocks << 4) | (blocks >> 4)).astype(np.uint8)
    swapped = gpt_oss.dequantize_mxfp4(swapped_bytes, wide_scales).transpose(0, 2, 1)
    ok = not np.array_equal(swapped, hf_layout)
    print(f"  [{'PASS' if ok else 'FAIL'}] control: swapped nibbles miss transformers")
    failures += 0 if ok else 1

    # Control: a decoder that reads the nibbles in the wrong order must disagree.
    table = np.array([fp4_e2m1_code_to_value(code) for code in range(16)], dtype=np.float32)
    codes = np.stack([blocks >> 4, blocks & 0x0F], axis=-1)
    values = table[codes.reshape(*blocks.shape[:-1], MXFP4_BLOCK)]
    multiplier = np.ldexp(np.ones_like(scales, np.float32), scales.astype(np.int32) - 127)
    swapped = (values * multiplier[..., None]).reshape(*blocks.shape[:-2], -1)
    ok = not np.array_equal(mine, swapped)
    print(f"  [{'PASS' if ok else 'FAIL'}] control: reversed nibble order disagrees")
    failures += 0 if ok else 1

    return failures


# ---------------------------------------------------------------------------
# the tiny checkpoint
# ---------------------------------------------------------------------------


def build_checkpoint(path: str, rng):
    """Write a tiny gpt-oss checkpoint, MXFP4 experts and all.

    Returns the decoded float32 tensors so the HuggingFace model can be given the same
    weights without going through the patch's decoder.
    """
    hidden = TINY["hidden_size"]
    inter = TINY["intermediate_size"]
    heads = TINY["num_attention_heads"]
    kv_heads = TINY["num_key_value_heads"]
    head_dim = TINY["head_dim"]
    experts = TINY["num_local_experts"]
    vocab = TINY["vocab_size"]

    def normal(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    tensors: dict[str, np.ndarray] = {}
    dense: dict[str, np.ndarray] = {}

    tensors["model.embed_tokens.weight"] = normal(vocab, hidden, scale=0.2)
    tensors["model.norm.weight"] = np.ones((hidden,), np.float32) + normal(hidden, scale=0.02)
    tensors["lm_head.weight"] = normal(vocab, hidden, scale=0.2)

    for layer in range(TINY["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones((hidden,), np.float32) + normal(
            hidden, scale=0.02
        )
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(
            (hidden,), np.float32
        ) + normal(hidden, scale=0.02)

        tensors[f"{prefix}.self_attn.q_proj.weight"] = normal(heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.q_proj.bias"] = normal(heads * head_dim)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = normal(kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.k_proj.bias"] = normal(kv_heads * head_dim)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = normal(kv_heads * head_dim, hidden)
        tensors[f"{prefix}.self_attn.v_proj.bias"] = normal(kv_heads * head_dim)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = normal(hidden, heads * head_dim)
        tensors[f"{prefix}.self_attn.o_proj.bias"] = normal(hidden)
        tensors[f"{prefix}.self_attn.sinks"] = normal(heads, scale=1.0)

        tensors[f"{prefix}.mlp.router.weight"] = normal(experts, hidden, scale=0.3)
        tensors[f"{prefix}.mlp.router.bias"] = normal(experts, scale=0.3)

        gate_up_blocks, gate_up_scales = random_mxfp4(rng, 2 * inter, hidden, experts)
        down_blocks, down_scales = random_mxfp4(rng, hidden, inter, experts)
        tensors[f"{prefix}.mlp.experts.gate_up_proj_blocks"] = gate_up_blocks
        tensors[f"{prefix}.mlp.experts.gate_up_proj_scales"] = gate_up_scales
        tensors[f"{prefix}.mlp.experts.down_proj_blocks"] = down_blocks
        tensors[f"{prefix}.mlp.experts.down_proj_scales"] = down_scales
        tensors[f"{prefix}.mlp.experts.gate_up_proj_bias"] = normal(experts, 2 * inter)
        tensors[f"{prefix}.mlp.experts.down_proj_bias"] = normal(experts, hidden)

        # `transformers` decodes the experts for its own side, so the reference never
        # sees the patch's byte-packing choice or its axis order.
        dense[f"{prefix}.mlp.experts.gate_up_proj"] = transformers_dense(
            gate_up_blocks, gate_up_scales
        )
        dense[f"{prefix}.mlp.experts.down_proj"] = transformers_dense(down_blocks, down_scales)

    save_file(tensors, os.path.join(path, "model.safetensors"))
    return tensors, dense


# ---------------------------------------------------------------------------
# HuggingFace reference
# ---------------------------------------------------------------------------


def build_hf_model(tensors, dense):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    kwargs = dict(TINY)
    kwargs["rope_scaling"] = dict(ROPE_SCALING)
    kwargs["rope_theta"] = TINY["rope_theta"]
    config = GptOssConfig(**kwargs)
    config._attn_implementation = "eager"
    model = GptOssForCausalLM(config)
    model = model.to(torch.float32).eval()

    state = dict(model.state_dict())
    for name in list(state):
        if name in tensors:
            state[name] = torch.from_numpy(np.array(tensors[name], dtype=np.float32))
        elif name in dense:
            state[name] = torch.from_numpy(np.array(dense[name], dtype=np.float32))
        else:
            raise KeyError(f"no source tensor for HuggingFace parameter {name}")
    model.load_state_dict(state, strict=True)
    return model


# The `quantization_config` openai/gpt-oss-120b ships in its config.json. It describes the
# packed experts. `ModelConfig` doesn't convert it and leaves the dict on the config.
PUBLISHED_QUANTIZATION = {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ],
    "quant_method": "mxfp4",
}


def write_config(path, hf_config):
    """The checkpoint's config.json: the reference config plus the published quantization dict."""
    config = hf_config.to_dict()
    config["architectures"] = ["GptOssForCausalLM"]
    config["quantization_config"] = PUBLISHED_QUANTIZATION
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(config, handle)


def run_hf(model, input_ids):
    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(input_ids[None, :].astype(np.int64)),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = [h[0].float().numpy() for h in out.hidden_states]
    return hidden, out.logits[0].float().numpy()


# ---------------------------------------------------------------------------
# JAX driver
# ---------------------------------------------------------------------------


class DenseAttentionBackend:
    """Stand-in for the ragged paged attention kernel, on one unpacked sequence.

    Implements the contract the kernel documents in
    `ragged_paged_attention_v3.ref_ragged_paged_attention`: repeat the KV heads up to the
    query heads, scale, mask acausally and outside the sliding window, prepend the per-head
    sink logit, softmax, drop the sink column, then weight the values. Everything the patch
    owns on either side of this call stays under test.
    """

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool, attention_sink=None):
        q = np.asarray(jax.device_get(q), np.float64)
        k = np.asarray(jax.device_get(k), np.float64)
        v = np.asarray(jax.device_get(v), np.float64)
        tokens, num_q_heads, head_dim = q.shape
        repeats = num_q_heads // k.shape[1]
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)

        logits = np.einsum("qhd,khd->hqk", q, k) * layer.scaling
        q_span = np.arange(tokens)[None, :, None]
        kv_span = np.arange(tokens)[None, None, :]
        masked = q_span < kv_span
        if layer.sliding_window_size:
            masked = np.logical_or(masked, q_span - layer.sliding_window_size >= kv_span)
        logits = np.where(masked, -np.inf, logits)

        if attention_sink is not None:
            sink = np.asarray(jax.device_get(attention_sink), np.float64)
            sink = np.broadcast_to(sink.reshape(num_q_heads, 1, 1), (num_q_heads, tokens, 1))
            logits = np.concatenate([sink, logits], axis=-1)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        if attention_sink is not None:
            probs = probs[..., 1:]

        out = np.einsum("hqk,khd->qhd", probs, v)
        return jnp.asarray(out.reshape(tokens, num_q_heads * head_dim), jnp.float32), None


def build_jax_model(mesh, model_path, quantization=None):
    """The model the way `ModelRunner.load_model` gets it.

    `ModelConfig` reads the checkpoint's `config.json`, `mxfp4` quantization dict and all,
    `JAXModelLoader` builds the model under `nnx.eval_shape` and loads it, and a config
    that asks for online quantization goes through `apply_quantization` after that. The
    runner makes all three calls outside a mesh context; the loader sets its own.
    """
    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.model_loader.loader import get_model_loader
    from sgl_jax.srt.utils.quantization.quantization_utils import apply_quantization

    model_config = ModelConfig(
        model_path=model_path,
        trust_remote_code=False,
        dtype="float32",
        quantization_config_path=quantization,
    )
    model = get_model_loader(LoadConfig(), mesh).load_model(model_config=model_config)
    if model_config.quantization_config is not None:
        model = apply_quantization(model_config, model)
    return model


def memory_pools(mesh):
    """`MemoryPools` around one `MHATokenToKVPool` with the head width padded to 128.

    The runner builds a `SWAKVPool` for gpt-oss, because the config sets a sliding window.
    That pool holds one `MHATokenToKVPool` for the sliding layers and one for the full
    layers, each padded to 128 the same way. The attention stand-in reads q, k and v
    straight from the model, so it never touches the pool.
    """
    from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool

    pool = MHATokenToKVPool(
        size=2 * SEQ_LEN,
        page_size=1,
        dtype=jnp.float32,
        head_num=TINY["num_key_value_heads"],
        head_dim=128,
        layer_num=TINY["num_hidden_layers"],
        mesh=mesh,
        dp_size=mesh.shape["data"],
    )
    return MemoryPools(token_to_kv_pool=pool)


def decode_metadata():
    """The metadata a decode step hands the logits processor.

    `DECODE` with no logprob request keeps every token row, so `next_token_logits` comes
    back as `[tokens, vocab]` and lines up with what HuggingFace returns.
    """
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

    return LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def make_batch(mesh, input_ids, backend, expert_location_metadata=None):
    """A `ForwardBatch` for one prefill of `input_ids`.

    The per-token arrays sit on the `data` axis, the placement `ForwardBatch.init_new`
    gives them. The per-request arrays hold one entry, so they sit replicated. Token slot
    0 is the pool's padding slot, so the tokens take slots 1 onward.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sgl_jax.srt.utils.jax_utils import device_array

    count = len(input_ids)
    per_token = lambda value: device_array(
        (np.asarray(value, np.int32),), NamedSharding(mesh, P("data"))
    )[0]
    per_request = lambda value: jax.device_put(
        np.asarray(value, np.int32), NamedSharding(mesh, P())
    )
    slots = np.arange(1, count + 1)
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=per_token(input_ids),
        req_pool_indices=per_request([0]),
        seq_lens=per_request([count]),
        out_cache_loc=per_token(slots),
        positions=per_token(np.arange(count)),
        attn_backend=backend,
        cache_loc=per_token(slots),
        extend_prefix_lens=per_request([0]),
        extend_seq_lens=per_request([count]),
        expert_location_metadata=expert_location_metadata,
    )


def run_jax(model, mesh, input_ids, expert_location_metadata=None):
    """Every captured residual stream, the final norm output, and the logits.

    The logits come out of the entry class, so the shipped `LogitsProcessor` and the
    `lm_head` selection are on the path rather than a matmul this file writes.
    """
    backend = DenseAttentionBackend()
    pools = memory_pools(mesh)
    with jax.sharding.set_mesh(mesh):
        batch = make_batch(mesh, input_ids, backend, expert_location_metadata)
        model.model.layers_to_capture = list(range(TINY["num_hidden_layers"]))
        hidden, aux, _, _, _ = model.model(batch, pools.token_to_kv_pool)

        capture = model.capture_aux_hidden_states
        model.capture_aux_hidden_states = False
        output, _, _, _ = model(batch, pools, decode_metadata())
        model.capture_aux_hidden_states = capture
        logits = output.next_token_logits
    return (
        [np.asarray(a, np.float32) for a in aux],
        np.asarray(hidden, np.float32),
        np.asarray(logits, np.float32),
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def pearson(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return 1.0 if denom == 0.0 else float(a @ b) / denom


def compare(jax_hidden, jax_final, jax_logits, hf_hidden, hf_logits, tol, verbose=True):
    """Max absolute error and Pearson, per layer.

    Returns the worst relative error: a row's max error over its reference's max.
    """
    rows = []
    for layer, captured in enumerate(jax_hidden):
        rows.append((f"layer {layer} in", captured, hf_hidden[layer]))
    rows.append(("final norm", jax_final, hf_hidden[-1]))
    rows.append(("logits", jax_logits, hf_logits))

    worst = 0.0
    for name, mine, theirs in rows:
        error = float(np.abs(mine - theirs).max())
        scale = float(np.abs(theirs).max()) or 1.0
        worst = max(worst, error / scale)
        if verbose:
            flag = "PASS" if error / scale <= tol else "FAIL"
            print(
                f"  [{flag}] {name:<14} max abs err {error:.3e}"
                f"   rel {error / scale:.3e}   pearson {pearson(mine, theirs):.9f}"
            )
    return worst


def check_rope(model, hf_model):
    """Both sides must actually be running YaRN, with the correction range untruncated."""
    failures = 0
    rope = model.model.layers[0].self_attn.rotary_emb
    ok = type(rope).__name__ == "YarnRotaryEmbedding"
    print(f"  [{'PASS' if ok else 'FAIL'}] JAX rope is {type(rope).__name__}")
    failures += 0 if ok else 1

    ok = getattr(rope, "truncate", True) is False
    print(f"  [{'PASS' if ok else 'FAIL'}] JAX rope honors truncate=false")
    failures += 0 if ok else 1

    ok = hf_model.model.rotary_emb.rope_type == "yarn"
    print(f"  [{'PASS' if ok else 'FAIL'}] HuggingFace rope type is yarn")
    failures += 0 if ok else 1

    mine = np.asarray(rope._inv_freq_np, np.float64)
    theirs = hf_model.model.rotary_emb.inv_freq.double().numpy()
    error = float(np.abs(mine - theirs).max())
    # HuggingFace holds inv_freq in float32, and the two sides compute it in different
    # libraries, so the last bits can follow the host: an AMD EPYC 7B12 matched to the bit, and
    # an AMD EPYC 9B14 with the same package versions landed two float32 ulps apart. Four float32
    # ulps of each value is the bar, the margin test_from_sae.py allows. The truncate=true
    # control below misses by about 1e5 of them.
    spacing = np.spacing(np.abs(theirs).astype(np.float32)).astype(np.float64)
    ulps = float((np.abs(mine - theirs) / spacing).max())
    ok = ulps <= 4.0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] inverse frequencies match, max abs err {error:.3e}"
        f" ({ulps:.2f} float32 ulp)"
    )
    failures += 0 if ok else 1

    # Control: rounding the correction range to whole dimensions, which is what every
    # other YaRN checkpoint wants, has to move the inverse frequencies off HuggingFace.
    from sgl_jax.srt.layers.embeddings import get_rope

    scaling = dict(ROPE_SCALING)
    scaling["truncate"] = True
    truncated = get_rope(
        head_size=TINY["head_dim"],
        rotary_dim=TINY["head_dim"],
        max_position=TINY["max_position_embeddings"],
        base=TINY["rope_theta"],
        is_neox_style=True,
        rope_scaling=scaling,
        dtype=jnp.float32,
    )
    gap = float(np.abs(np.asarray(truncated._inv_freq_np, np.float64) - theirs).max())
    phase = gap * TINY["max_position_embeddings"]
    ok = gap > 1e-6
    print(
        f"  [{'PASS' if ok else 'FAIL'}] control: truncate=true disagrees by {gap:.3e}"
        f" per position, {phase:.1f} rad at {TINY['max_position_embeddings']} tokens"
    )
    failures += 0 if ok else 1

    return failures


def check_forward(model, mesh, input_ids, hf_hidden, hf_logits):
    aux, final, logits = run_jax(model, mesh, input_ids)
    if len(aux) + 1 != len(hf_hidden):
        print(f"  [FAIL] captured {len(aux)} layers, HuggingFace reported {len(hf_hidden) - 1}")
        return 1
    worst = compare(aux, final, logits, hf_hidden, hf_logits, TOL)
    for name, mine, theirs in (
        ("logits", logits, hf_logits),
        ("final norm", final, hf_hidden[-1]),
    ):
        if pearson(mine, theirs) < MIN_PEARSON:
            print(f"  [FAIL] {name} pearson below {MIN_PEARSON}")
            return 1
    return 0 if worst <= TOL else 1


# ---------------------------------------------------------------------------
# mutants
# ---------------------------------------------------------------------------


def assign(param, value):
    """Write a numpy array back into a parameter without losing its sharding."""
    current = param.get_value()
    param.set_value(jax.device_put(jnp.asarray(value, current.dtype), current.sharding))


def mutate(model, name):
    """Break one connection. Every one of these has to fail the clean gate."""
    layer = model.model.layers[1]
    experts = layer.mlp.experts
    if name == "expert gate weight nudged":
        value = np.array(experts.wi_0.get_value())
        value[0, 0, 0] += 0.5
        assign(experts.wi_0, value)
    elif name == "gate and up halves swapped":
        gate, up = np.array(experts.wi_0.get_value()), np.array(experts.wi_1.get_value())
        gate_bias = np.array(experts.wi_0_bias.get_value())
        up_bias = np.array(experts.wi_1_bias.get_value())
        assign(experts.wi_0, up)
        assign(experts.wi_1, gate)
        assign(experts.wi_0_bias, up_bias)
        assign(experts.wi_1_bias, gate_bias)
    elif name == "down-projection bias dropped":
        assign(experts.wo_bias, np.zeros(experts.wo_bias.get_value().shape, np.float32))
    elif name == "attention sinks zeroed":
        sinks = layer.self_attn.sinks
        assign(sinks, np.zeros(sinks.get_value().shape, np.float32))
    elif name == "expert reduction axis reversed":
        assign(experts.wo, np.ascontiguousarray(np.array(experts.wo.get_value())[:, ::-1, :]))
    elif name == "router bias dropped":
        router = layer.mlp.router
        assign(router.proj.bias, np.zeros(router.proj.bias.get_value().shape, np.float32))
    else:
        raise AssertionError(f"unknown mutant {name}")


def check_mutants(mesh, model_path, input_ids, hf_hidden, hf_logits):
    names = [
        "expert gate weight nudged",
        "gate and up halves swapped",
        "down-projection bias dropped",
        "attention sinks zeroed",
        "expert reduction axis reversed",
        "router bias dropped",
    ]
    failures = 0
    for name in names:
        model = build_jax_model(mesh, model_path)
        with jax.sharding.set_mesh(mesh):
            mutate(model, name)
        aux, final, logits = run_jax(model, mesh, input_ids)
        worst = compare(aux, final, logits, hf_hidden, hf_logits, TOL, verbose=False)
        caught = worst > TOL
        print(f"  [{'PASS' if caught else 'FAIL'}] caught: {name:<30} worst rel err {worst:.3e}")
        failures += 0 if caught else 1
    return failures


# ---------------------------------------------------------------------------
# capture hook
# ---------------------------------------------------------------------------


class RecordingLogitsProcessor:
    def __init__(self):
        self.aux_hidden_states = "unset"

    def __call__(self, hidden_states, head, logits_metadata, aux_hidden_states=None):
        self.aux_hidden_states = aux_hidden_states
        return hidden_states


def check_capture_hook(model, mesh, input_ids):
    failures = 0
    backend = DenseAttentionBackend()
    pools = memory_pools(mesh)
    num_layers = TINY["num_hidden_layers"]

    with jax.sharding.set_mesh(mesh):
        batch = make_batch(mesh, input_ids, backend)

        for gate in ([], [1, 3], list(range(num_layers))):
            model.model.layers_to_capture = gate
            _, aux, _, _, _ = model.model(batch, pools.token_to_kv_pool)
            ok = len(aux) == len(gate)
            print(f"  [{'PASS' if ok else 'FAIL'}] gate {gate} captures {len(aux)} layers")
            failures += 0 if ok else 1

        # The captured stream at layer 0 is the embedding output, before any layer runs.
        model.model.layers_to_capture = [0]
        _, aux, _, _, _ = model.model(batch, pools.token_to_kv_pool)
        embedded = model.model.embed_tokens(batch.input_ids)
        ok = bool(np.allclose(np.asarray(aux[0]), np.asarray(embedded), atol=0, rtol=0))
        print(f"  [{'PASS' if ok else 'FAIL'}] layer 0 capture is the embedding output")
        failures += 0 if ok else 1

        # Control: the None-residual form must not be reading a stale residual. Layer 1's
        # capture has to differ from layer 0's.
        model.model.layers_to_capture = [0, 1]
        _, aux, _, _, _ = model.model(batch, pools.token_to_kv_pool)
        ok = not np.allclose(np.asarray(aux[0]), np.asarray(aux[1]))
        print(f"  [{'PASS' if ok else 'FAIL'}] control: layer 1 capture differs from layer 0")
        failures += 0 if ok else 1

        recorder = RecordingLogitsProcessor()
        real_processor = model.logits_processor
        model.logits_processor = recorder
        model.model.layers_to_capture = list(range(num_layers))

        model.capture_aux_hidden_states = False
        model(batch, pools, None)
        ok = recorder.aux_hidden_states is None
        print(f"  [{'PASS' if ok else 'FAIL'}] flag off: logits processor gets None")
        failures += 0 if ok else 1

        model.capture_aux_hidden_states = True
        output = model(batch, pools, None)
        got = recorder.aux_hidden_states
        ok = isinstance(got, list) and len(got) == num_layers
        print(f"  [{'PASS' if ok else 'FAIL'}] flag on: logits processor gets {num_layers} layers")
        failures += 0 if ok else 1

        ok = isinstance(output, tuple) and len(output) == 4
        print(f"  [{'PASS' if ok else 'FAIL'}] entry class returns a 4-tuple")
        failures += 0 if ok else 1

        model.logits_processor = real_processor

        # The two production entry points. Both shift the caller's ids by one, matching
        # llama and qwen3 upstream: the aux state for layer i is the input of layer i + 1.
        model.model.layers_to_capture = list(range(num_layers))
        _, every_layer, _, _, _ = model.model(batch, pools.token_to_kv_pool)

        for setter, ids, expected in (
            ("set_eagle3_layers_to_capture", [0, 2], [1, 3]),
            ("set_dflash_layers_to_capture", [2], [3]),
        ):
            model.capture_aux_hidden_states = False
            model.model.layers_to_capture = []
            getattr(model, setter)(ids)
            placed = list(model.model.layers_to_capture)
            ok = placed == expected and model.capture_aux_hidden_states
            print(f"  [{'PASS' if ok else 'FAIL'}] {setter}({ids}) places {placed}")
            failures += 0 if ok else 1

            _, aux, _, _, _ = model.model(batch, pools.token_to_kv_pool)
            ok = len(aux) == len(expected) and all(
                np.array_equal(np.asarray(got), np.asarray(every_layer[layer]))
                for got, layer in zip(aux, expected)
            )
            print(f"  [{'PASS' if ok else 'FAIL'}] {setter} captures the input of {expected}")
            failures += 0 if ok else 1

        # Control: the offset is one, not zero. Capturing the caller's own ids has to
        # give a different set of tensors.
        ok = not np.array_equal(np.asarray(every_layer[2]), np.asarray(every_layer[3]))
        print(f"  [{'PASS' if ok else 'FAIL'}] control: layer 2 and layer 3 inputs differ")
        failures += 0 if ok else 1

        # The no-argument EAGLE3 default follows upstream's formula. At four layers the
        # formula repeats itself, which is why the gate checks above use explicit ids.
        model.set_eagle3_layers_to_capture()
        expected = [2, num_layers // 2, num_layers - 3]
        ok = list(model.model.layers_to_capture) == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] eagle3 default follows [2, n//2, n-3] = {expected}")
        failures += 0 if ok else 1

        try:
            model.set_dflash_layers_to_capture(None)
            raised = False
        except ValueError:
            raised = True
        print(f"  [{'PASS' if raised else 'FAIL'}] dflash refuses a missing layer list")
        failures += 0 if raised else 1

        model.model.layers_to_capture = []
        model.capture_aux_hidden_states = False

    return failures


def check_lm_head(model, mesh, input_ids, hf_logits):
    """The shipped `LogitsProcessor`, on both the untied and the tied branch."""
    failures = 0
    backend = DenseAttentionBackend()
    pools = memory_pools(mesh)

    with jax.sharding.set_mesh(mesh):
        batch = make_batch(mesh, input_ids, backend)
        hidden, _, _, _, _ = model.model(batch, pools.token_to_kv_pool)

        output, _, _, _ = model(batch, pools, decode_metadata())
        untied = np.asarray(output.next_token_logits, np.float32)
        error = float(np.abs(untied - hf_logits).max()) / (float(np.abs(hf_logits).max()) or 1.0)
        ok = error <= TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] untied head matches HuggingFace, rel {error:.3e}")
        failures += 0 if ok else 1

        # Tied embeddings take the other branch of the entry class, which hands the
        # embedding table to the processor in place of `lm_head`.
        model.config.tie_word_embeddings = True
        output, _, _, _ = model(batch, pools, decode_metadata())
        tied = np.asarray(output.next_token_logits, np.float32)
        model.config.tie_word_embeddings = False

        embedding = np.asarray(model.model.embed_tokens.embedding.get_value(), np.float32)
        expected = np.asarray(hidden, np.float32) @ embedding.T
        error = float(np.abs(tied - expected).max()) / (float(np.abs(expected).max()) or 1.0)
        ok = error <= TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] tied head reads the embedding, rel {error:.3e}")
        failures += 0 if ok else 1

        # Control: the two branches read different tables, so they can't agree.
        ok = not np.allclose(untied, tied)
        print(f"  [{'PASS' if ok else 'FAIL'}] control: tied and untied logits differ")
        failures += 0 if ok else 1

    return failures


# ---------------------------------------------------------------------------
# EPLB redundant experts
# ---------------------------------------------------------------------------


EXPERT_PARAMS = ("wi_0", "wi_1", "wo", "wi_0_bias", "wi_1_bias", "wo_bias")


def redundant_metadata(mesh, num_redundant: int, algorithm: str = "static"):
    """Expert-location metadata that parks two logical experts on redundant slots.

    `physical_to_logical_map` wraps, so physical slot `num_logical + i` holds a copy of
    logical expert `i`. The static dispatch map sends logical 0 and 1 to those copies
    instead of to slots 0 and 1. The dynamic one lists both copies of each and picks one
    per token.

    The maps land on device here, under the serving mesh, the way server startup places
    them.
    """
    from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata

    num_logical = TINY["num_local_experts"]
    num_layers = TINY["num_hidden_layers"]
    num_physical = num_logical + num_redundant

    physical_to_logical = np.tile(np.arange(num_physical) % num_logical, (num_layers, 1))
    dispatch = np.tile(np.arange(num_logical), (num_layers, 1))
    every_copy = np.full((num_layers, num_logical, 2), -1, dtype=np.int32)
    every_copy[:, :, 0] = np.arange(num_logical)
    copies = np.ones((num_layers, num_logical), dtype=np.int32)
    for i in range(num_redundant):
        dispatch[:, i] = num_logical + i
        every_copy[:, i, 1] = num_logical + i
        copies[:, i] = 2

    with jax.sharding.set_mesh(mesh):
        return ExpertLocationMetadata(
            ep_dispatch_algorithm=algorithm,
            logical_to_rank_dispatch_physical_map=dispatch,
            logical_to_all_physical_map=every_copy,
            logical_to_all_physical_map_num_valid=copies,
            physical_to_logical_map=physical_to_logical,
            num_physical_experts=num_physical,
        )


def check_redundant_experts(mesh, model_path, input_ids, hf_hidden, hf_logits):
    """EPLB redundant experts, from the loader through the dispatch map to the forward.

    `EPMoE` sizes the expert parameters at the physical count, and the loader has to fill
    them. The forward carries the metadata in its `ForwardBatch`, as the runner's does, so
    `TopK` sends logical experts 0 and 1 to slots 8 and 9. Poisoning slots 0 and 1 then
    leaves that forward on HuggingFace, and moves a forward that routes logically.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.eplb.expert_location import (
        set_global_expert_location_metadata,
        topk_ids_logical_to_physical,
    )

    failures = 0
    num_logical = TINY["num_local_experts"]
    num_redundant = 2
    num_physical = num_logical + num_redundant
    metadata = redundant_metadata(mesh, num_redundant)

    set_global_expert_location_metadata(metadata)
    try:
        model = build_jax_model(mesh, model_path)
        experts = model.model.layers[1].mlp.experts

        ok = experts.num_experts == num_physical
        print(f"  [{'PASS' if ok else 'FAIL'}] EPMoE allocates {experts.num_experts} slots")
        failures += 0 if ok else 1

        loaded = {name: np.asarray(getattr(experts, name).get_value()) for name in EXPERT_PARAMS}
        shaped = all(value.shape[0] == num_physical for value in loaded.values())
        print(f"  [{'PASS' if shaped else 'FAIL'}] every expert parameter carries every slot")
        failures += 0 if shaped else 1

        if shaped:
            copied = all(
                np.array_equal(value[num_logical + i], value[i])
                for value in loaded.values()
                for i in range(num_redundant)
            )
            print(f"  [{'PASS' if copied else 'FAIL'}] redundant slots copy their logical expert")
            failures += 0 if copied else 1

            # Control: the copy is indexed, not accidental. Slot 8 holds logical 0, which
            # is a different expert from logical 2.
            distinct = all(
                not np.array_equal(value[num_logical + i], value[i + 2])
                for value in loaded.values()
                for i in range(num_redundant)
            )
            print(f"  [{'PASS' if distinct else 'FAIL'}] control: the copy follows the map")
            failures += 0 if distinct else 1

        # The routing side of the same map. `TopK` hands `topk_ids` through this call,
        # sharded over the batch on the serving mesh, so logical 0 and 1 land on the slots
        # the loader just filled.
        logical = np.tile(np.arange(num_logical, dtype=np.int32), (16, 1))
        with jax.sharding.set_mesh(mesh):
            batch_ids = jax.device_put(logical, NamedSharding(mesh, P("data", None)))
            physical = np.asarray(topk_ids_logical_to_physical(batch_ids, metadata, 1))
            dynamic = redundant_metadata(mesh, num_redundant, algorithm="dynamic")
            picked = np.asarray(topk_ids_logical_to_physical(batch_ids, dynamic, 1))
        expected = np.arange(num_logical)
        expected[:num_redundant] = num_logical + np.arange(num_redundant)
        ok = np.array_equal(physical, np.tile(expected, (16, 1)))
        print(
            f"  [{'PASS' if ok else 'FAIL'}] static map sends logical 0..7 to slots "
            f"{physical[0].tolist()}, on ids sharded over the batch"
        )
        failures += 0 if ok else 1

        holder = np.arange(num_physical) % num_logical
        both = all(
            set(picked[logical == i].tolist()) == {i, num_logical + i} for i in range(num_redundant)
        )
        ok = bool(np.all(holder[picked] == logical)) and both
        print(
            f"  [{'PASS' if ok else 'FAIL'}] dynamic map sends every id to a copy of its expert, "
            f"and logical 0 and 1 to both of theirs: {sorted(set(picked.ravel().tolist()))}"
        )
        failures += 0 if ok else 1

        # Poison slots 0 and 1 in every layer. The static map never routes there, so the
        # forward that carries it has to stay on HuggingFace.
        with jax.sharding.set_mesh(mesh):
            for layer in model.model.layers:
                for name in EXPERT_PARAMS:
                    param = getattr(layer.mlp.experts, name)
                    value = np.array(param.get_value())
                    value[:num_redundant] = 3.0
                    assign(param, value)

        aux, final, logits = run_jax(model, mesh, input_ids, metadata)
        worst = compare(aux, final, logits, hf_hidden, hf_logits, TOL, verbose=False)
        finite = all(np.isfinite(array).all() for array in (*aux, final, logits))
        ok = finite and worst <= TOL
        print(
            f"  [{'PASS' if ok else 'FAIL'}] forward through the map, slots 0 and 1 poisoned, "
            f"matches HuggingFace, rel {worst:.3e}, finite {finite}"
        )
        failures += 0 if ok else 1

        # Control: the same forward without the map routes logically and reads the poison,
        # so it has to fail the gate the forward above passed.
        aux, final, logits = run_jax(model, mesh, input_ids)
        missed = compare(aux, final, logits, hf_hidden, hf_logits, TOL, verbose=False)
        finite = all(np.isfinite(array).all() for array in (*aux, final, logits))
        caught = not (finite and missed <= TOL)
        print(
            f"  [{'PASS' if caught else 'FAIL'}] control: without the map the forward reads "
            f"slots 0 and 1, rel {missed:.3e}, finite {finite}"
        )
        failures += 0 if caught else 1
    finally:
        set_global_expert_location_metadata(None)

    return failures


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------


def check_quantization(mesh, model_path):
    """What reaches `EPMoE` as a quantization config.

    Every model above loaded from a config.json that carries the published `mxfp4` dict,
    which `ModelConfig` leaves on `hf_config`. `EPMoE` calls methods on the config it gets,
    so the model hands it a `QuantizationConfig` or nothing. Control: handing it whatever
    sits on the config, the way the upstream MoE models do, breaks the default load.

    Then the built-in `int8.yaml`, applied the way the runner applies it. The routed experts
    come out int8 with a scale per output channel and hold the decoded MXFP4 weights to
    within half a quantization step, and the three expert biases stay what the checkpoint
    holds.

    On CPU the `gmm` kernel's interpret path returns NaN for int8 weights with a scale when
    an expert matrix is under 128 on either side, counted per shard. `EPMoE` splits the
    experts 8 ways on this mesh, so the loaded model's 64-wide experts stop at the weights.
    A block built the same way 1,024 wide, 128 a shard, runs its int8 experts, and they
    have to land within 1e-5 of numpy on the dequantized weights and more than 1e-3 off
    the float ones.
    """
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.models import gpt_oss

    failures = 0
    model_config = ModelConfig(model_path=model_path, trust_remote_code=False, dtype="float32")
    carried = getattr(model_config.hf_config, "quantization_config", None)
    ok = carried == PUBLISHED_QUANTIZATION and model_config.quantization_config is None
    print(
        f"  [{'PASS' if ok else 'FAIL'}] config.json's mxfp4 dict stays on hf_config and "
        f"resolves to {model_config.quantization_config}"
    )
    failures += 0 if ok else 1

    filtered = gpt_oss._quantization_config
    gpt_oss._quantization_config = lambda config: getattr(config, "quantization_config", None)
    try:
        build_jax_model(mesh, model_path)
        refused = None
    except AttributeError as exc:
        refused = str(exc)
    finally:
        gpt_oss._quantization_config = filtered
    print(
        f"  [{'PASS' if refused else 'FAIL'}] control: the dict handed to EPMoE breaks the "
        f"load: {refused}"
    )
    failures += 0 if refused else 1

    plain = build_jax_model(mesh, model_path)
    quantized = build_jax_model(mesh, model_path, quantization="int8.yaml")
    experts = quantized.model.layers[1].mlp.experts
    reference = plain.model.layers[1].mlp.experts
    weights = ("wi_0", "wi_1", "wo")
    dtypes = {name: str(getattr(experts, name)[...].dtype) for name in weights}
    scaled = all(getattr(experts, f"{name}_scale") is not None for name in weights)
    ok = set(dtypes.values()) == {"int8"} and scaled
    print(f"  [{'PASS' if ok else 'FAIL'}] int8.yaml quantizes the experts: {dtypes}")
    failures += 0 if ok else 1
    if not ok:
        return failures

    # Per output channel, `quantize_tensor` rounds w / scale to the nearest integer with
    # scale = max|w| / 127, so a dequantized weight sits within half a scale of the original.
    steps = 0.0
    for name in weights:
        value = np.asarray(getattr(experts, name)[...], np.float32)
        scale = np.asarray(getattr(experts, f"{name}_scale")[...])[:, 0]
        original = np.asarray(getattr(reference, name)[...])
        steps = max(steps, float((np.abs(value * scale - original) / scale).max()))
    ok = steps <= 0.5 + 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] int8 experts sit {steps:.3f} of a step off MXFP4")
    failures += 0 if ok else 1

    biases = ("wi_0_bias", "wi_1_bias", "wo_bias")
    ok = all(
        np.array_equal(np.asarray(getattr(experts, n)[...]), np.asarray(getattr(reference, n)[...]))
        for n in biases
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] the expert biases stay what the checkpoint holds")
    failures += 0 if ok else 1

    errors, per_shard = int8_experts_against_numpy(mesh, 1024)
    ok = errors["dequantized"] < 1e-5 and errors["float"] > 1e-3
    print(
        f"  [{'PASS' if ok else 'FAIL'}] int8 experts 1,024 wide, {per_shard} a shard, against "
        f"numpy: max relative {errors['dequantized']:.3e} on the dequantized weights; "
        f"control: {errors['float']:.3e} on the float ones"
    )
    failures += 0 if ok else 1

    # Under int8_w8a8.yaml gmm quantizes the activations too. Off TPU that's its v1 path,
    # which rescales each row by its activation scale after the kernel runs, so a bias the
    # kernel added would come out scaled.
    gaps = gmm_bias_gaps()
    ok = max(gaps.values()) < 1e-5
    print(
        f"  [{'PASS' if ok else 'FAIL'}] gmm under int8 activations adds each group's bias whole: "
        + ", ".join(f"{label} {gap:.3e}" for label, gap in gaps.items())
    )
    failures += 0 if ok else 1

    # The gap comes out of two float32 outputs near 1e2 each, so rounding alone leaves about
    # 1e-5 of the bias sum. A bias scaled by the activation scale misses by most of it.
    gap = w8a8_bias_against_numpy(mesh, 1024)
    ok = gap < 1e-3
    print(
        f"  [{'PASS' if ok else 'FAIL'}] int8_w8a8.yaml experts 1,024 wide: the down-projection "
        f"bias reaches the output whole, max relative {gap:.3e}"
    )
    failures += 0 if ok else 1
    return failures


def gmm_bias_gaps() -> dict[str, float]:
    """How far gmm's bias lands from whole, with int8 activations, on the path CPU runs.

    The bias contribution is the output with a bias minus the output without one, and it
    has to equal the bias row of each row's group. `group_offset` 2 computes groups 2 and 3
    of 4, the way an expert shard computes its own experts, and the rows of groups 0 and 1
    take no bias. Returns the largest gap per case.
    """
    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

    rng = np.random.default_rng(17)
    sizes = np.array([128, 128, 128, 128], np.int32)
    lhs = (rng.standard_normal((512, 256)) * rng.uniform(0.5, 8.0, size=(512, 1))).astype(
        np.float32
    )
    rhs = (rng.standard_normal((4, 256, 256)) * 0.05).astype(np.float32)
    bias = rng.uniform(-2.0, 2.0, size=(4, 1, 256)).astype(np.float32)
    rows = np.repeat(bias[:, 0, :], sizes, axis=0)
    local = rows.copy()
    local[:256] = 0.0

    def contribution(weights, biases, **kwargs):
        run = lambda **extra: np.asarray(
            gmm(
                jnp.asarray(lhs),
                jnp.asarray(weights),
                jnp.asarray(sizes),
                preferred_element_type=jnp.float32,
                activation_quantized_dtype=jnp.int8,
                **kwargs,
                **extra,
            )
        )
        return run(rhs_bias=jnp.asarray(biases)) - run()

    offset = jnp.asarray(2, jnp.int32)
    return {
        "all 4 groups": float(np.abs(contribution(rhs, bias) - rows).max()),
        "groups 2 and 3": float(
            np.abs(contribution(rhs[2:], bias[2:], group_offset=offset) - local).max()
        ),
    }


def w8a8_bias_against_numpy(mesh, width) -> float:
    """The down-projection bias under `int8_w8a8.yaml`, through a gpt-oss MoE block.

    Builds the block the way `int8_experts_against_numpy` does, with `int8_w8a8.yaml` on it,
    so both expert GEMMs quantize their activations. The bias adds after the second GEMM, so
    the block's output with it minus the output with it zeroed has to equal the routed sum
    of each token's bias rows. Returns the largest gap relative to the largest entry of that
    sum.
    """
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.models.gpt_oss import GptOssSparseMoeBlock

    config = GptOssConfig(**dict(TINY, intermediate_size=width, rope_scaling=dict(ROPE_SCALING)))
    config.quantization_config = QuantizationConfig.from_path("int8_w8a8.yaml")
    with jax.sharding.set_mesh(mesh):
        block = GptOssSparseMoeBlock(config, mesh=mesh, layer_id=1, dtype=jnp.float32)
    experts = block.experts
    rng = np.random.default_rng(19)
    count, top_k, hidden = config.num_local_experts, config.num_experts_per_tok, config.hidden_size
    floats = {
        "wi_0": rng.standard_normal((count, hidden, width)) * 0.2,
        "wi_1": rng.standard_normal((count, hidden, width)) * 0.2,
        "wo": rng.standard_normal((count, width, hidden)) * 0.2,
        "wi_0_bias": rng.standard_normal((count, 1, width)) * 0.5,
        "wi_1_bias": rng.standard_normal((count, 1, width)) * 0.5,
        "wo_bias": rng.standard_normal((count, 1, hidden)) * 0.5,
    }
    with jax.sharding.set_mesh(mesh):
        for name, value in floats.items():
            assign(getattr(experts, name), value.astype(np.float32))
    experts.quantize_weights()

    tokens = jnp.asarray(rng.standard_normal((32, hidden)).astype(np.float32))
    ids = np.argsort(rng.random((32, count)), axis=1)[:, :top_k].astype(np.int32)
    weights = rng.random((32, top_k)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    with jax.sharding.set_mesh(mesh):
        with_bias = np.asarray(experts(tokens, jnp.asarray(weights), jnp.asarray(ids)), np.float64)
        assign(experts.wo_bias, np.zeros_like(floats["wo_bias"], dtype=np.float32))
        without = np.asarray(experts(tokens, jnp.asarray(weights), jnp.asarray(ids)), np.float64)

    routed = np.einsum("tk,tkh->th", weights, floats["wo_bias"][ids][:, :, 0])
    return float(np.abs(with_bias - without - routed).max() / np.abs(routed).max())


def int8_experts_against_numpy(mesh, width):
    """Run the int8 experts of a gpt-oss MoE block `width` wide against numpy.

    Builds `GptOssSparseMoeBlock` the way the model does, from the tiny config with the
    expert width set to `width` and `int8.yaml` on it. Random experts and biases go
    through `quantize_weights`, the call `apply_quantization` makes, and run on 32 random
    tokens routed to distinct experts. Returns the max error relative to the largest
    reference output, against numpy on the dequantized weights and on the float weights,
    and the expert width each shard holds.
    """
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.models.gpt_oss import GptOssSparseMoeBlock

    config = GptOssConfig(**dict(TINY, intermediate_size=width, rope_scaling=dict(ROPE_SCALING)))
    config.quantization_config = QuantizationConfig.from_path("int8.yaml")
    with jax.sharding.set_mesh(mesh):
        block = GptOssSparseMoeBlock(config, mesh=mesh, layer_id=1, dtype=jnp.float32)
    experts = block.experts
    rng = np.random.default_rng(11)
    count, top_k, hidden = config.num_local_experts, config.num_experts_per_tok, config.hidden_size
    floats = {
        "wi_0": rng.standard_normal((count, hidden, width)) * 0.2,
        "wi_1": rng.standard_normal((count, hidden, width)) * 0.2,
        "wo": rng.standard_normal((count, width, hidden)) * 0.2,
        "wi_0_bias": rng.standard_normal((count, 1, width)) * 0.5,
        "wi_1_bias": rng.standard_normal((count, 1, width)) * 0.5,
        "wo_bias": rng.standard_normal((count, 1, hidden)) * 0.5,
    }
    with jax.sharding.set_mesh(mesh):
        for name, value in floats.items():
            assign(getattr(experts, name), value.astype(np.float32))
    experts.quantize_weights()
    dequantized = dict(floats)
    for name in ("wi_0", "wi_1", "wo"):
        value = np.asarray(getattr(experts, name)[...], np.float64)
        scale = np.asarray(getattr(experts, f"{name}_scale")[...], np.float64)[:, 0]
        dequantized[name] = value * scale

    tokens = rng.standard_normal((32, hidden)).astype(np.float32)
    ids = np.argsort(rng.random((32, count)), axis=1)[:, :top_k].astype(np.int32)
    weights = rng.random((32, top_k)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    with jax.sharding.set_mesh(mesh):
        out = experts(jnp.asarray(tokens), jnp.asarray(weights), jnp.asarray(ids))
    out = np.asarray(out, np.float64)

    def routed(matrices):
        # The clamped SwiGLU transformers runs, alpha 1.702: the gate capped at the limit,
        # the up half clipped to it both ways.
        limit, alpha = config.swiglu_limit, 1.702
        gate = np.einsum("th,tkhi->tki", tokens, matrices["wi_0"][ids])
        gate = np.minimum(gate + matrices["wi_0_bias"][ids][:, :, 0], limit)
        up = np.einsum("th,tkhi->tki", tokens, matrices["wi_1"][ids])
        up = np.clip(up + matrices["wi_1_bias"][ids][:, :, 0], -limit, limit)
        act = (up + 1.0) * gate / (1.0 + np.exp(-alpha * gate))
        down = np.einsum("tki,tkih->tkh", act, matrices["wo"][ids])
        return np.einsum("tk,tkh->th", weights, down + matrices["wo_bias"][ids][:, :, 0])

    errors = {}
    for label, matrices in (("dequantized", dequantized), ("float", floats)):
        reference = routed(matrices)
        errors[label] = float(np.abs(out - reference).max() / np.abs(reference).max())
    return errors, width // experts.tp_size


# ---------------------------------------------------------------------------
# dummy weights
# ---------------------------------------------------------------------------


# The layouts `EPMoE.__init__` declares for its expert parameters, on its (expert, tensor)
# mesh, when `--moe-dp-size` doesn't replicate the experts.
EXPERT_LAYOUT = {
    "wi_0": ("expert", None, "tensor"),
    "wi_1": ("expert", None, "tensor"),
    "wo": ("expert", "tensor", None),
    "wi_0_bias": ("expert", None, "tensor"),
    "wi_1_bias": ("expert", None, "tensor"),
    "wo_bias": ("expert", None, None),
}


def device_bytes(array, device) -> int:
    """Bytes `device` holds of `array`, read off its sharding, so no buffer gets touched."""
    sharding = array.sharding
    if device not in sharding.device_set:
        return 0
    return math.prod(sharding.shard_shape(array.shape)) * array.dtype.itemsize


def expert_shards(model) -> list[tuple[str, tuple, tuple]]:
    """Each expert parameter's per-device shard shape beside the one `EPMoE`'s layout gives it."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    rows = []
    for index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        for name, spec in EXPERT_LAYOUT.items():
            value = getattr(experts, name).get_value()
            want = NamedSharding(experts.moe_mesh, P(*spec)).shard_shape(value.shape)
            got = value.sharding.shard_shape(value.shape)
            rows.append((f"layer {index} {name}", tuple(got), tuple(want)))
    return rows


def dummy_load(mesh, model_path, fill_experts: bool = True):
    """`--load-format dummy` through `JAXDummyModelLoader`, watched from device 0.

    Records the expert shards right after the loader's dummy pass, the most device 0 holds
    of the arrays the load allocates while it runs, and what the loaded model holds on
    device 0. `fill_experts=False` turns the model's own expert fill off, for the control.
    """
    from flax import nnx

    from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.model_loader.loader import get_model_loader
    from sgl_jax.srt.models import gpt_oss
    from sgl_jax.srt.utils.weight_utils import WeightLoader

    device = mesh.devices.flat[0]
    made = {}  # id of each array the load allocates -> the bytes device 0 holds of it
    record = {"peak": 0}
    dummy_pass = WeightLoader.load_weights_from_safetensors
    allocate = WeightLoader._dummy_array
    fill = gpt_oss.GptOssForCausalLM._init_dummy_experts

    def watched_pass(self, *args, **kwargs):
        result = dummy_pass(self, *args, **kwargs)
        record["after pass"] = expert_shards(self.model)
        return result

    def watched_allocate(self, *args, **kwargs):
        array = allocate(self, *args, **kwargs)
        made[id(array)] = device_bytes(array, device)
        live = {id(held) for held in jax.live_arrays()}
        record["peak"] = max(record["peak"], sum(v for k, v in made.items() if k in live))
        return array

    WeightLoader.load_weights_from_safetensors = watched_pass
    WeightLoader._dummy_array = watched_allocate
    if not fill_experts:
        gpt_oss.GptOssForCausalLM._init_dummy_experts = lambda *args, **kwargs: None
    try:
        gc.collect()
        model_config = ModelConfig(model_path=model_path, trust_remote_code=False, dtype="float32")
        loader = get_model_loader(LoadConfig(load_format=LoadFormat.DUMMY), mesh)
        model = loader.load_model(model_config=model_config)
        record["held"] = sum(
            device_bytes(leaf, device)
            for leaf in jax.tree.leaves(nnx.state(model))
            if isinstance(leaf, jax.Array)
        )
    finally:
        WeightLoader.load_weights_from_safetensors = dummy_pass
        WeightLoader._dummy_array = allocate
        gpt_oss.GptOssForCausalLM._init_dummy_experts = fill
    del model
    return record


def check_dummy_load(model_path):
    """`--load-format dummy` at `--tp-size 8` never holds a whole expert stack on one device.

    The loader's dummy pass fills every parameter the mapping table leaves out, on the model
    mesh, which has no "expert" axis, so an expert stack it filled would sit whole on every
    device, 6.37 GB a layer at gpt-oss-120b's shapes. The model fills its experts on
    `EPMoE`'s mesh before that pass. Right after the pass every expert shard has the shape
    `EPMoE`'s layout gives it, and the most device 0 holds of what the load allocates stays
    at what the loaded model holds there. Control: with the model's fill turned off, the
    pass leaves the stacks whole on every device.
    """
    failures = 0
    mesh = jax.make_mesh(
        (1, 8), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )

    record = dummy_load(mesh, model_path)
    wrong = [row for row in record["after pass"] if row[1] != row[2]]
    ok = not wrong
    print(
        f"  [{'PASS' if ok else 'FAIL'}] after the loader's pass, all {len(record['after pass'])} "
        f"expert arrays sit in EPMoE's layout"
        + ("" if ok else f"; {len(wrong)} don't, the first {wrong[0][0]} at {wrong[0][1]}")
    )
    failures += 0 if ok else 1

    ok = record["peak"] <= 1.25 * record["held"]
    print(
        f"  [{'PASS' if ok else 'FAIL'}] device 0 holds at most {record['peak']:,} bytes of what "
        f"the load allocates, and the loaded model holds {record['held']:,} there"
    )
    failures += 0 if ok else 1

    control = dummy_load(mesh, model_path, fill_experts=False)
    whole = [row for row in control["after pass"] if row[1] != row[2]]
    caught = bool(whole)
    print(
        f"  [{'PASS' if caught else 'FAIL'}] control: with the model's fill off, {len(whole)} "
        f"expert arrays sit whole on every device after the pass, and device 0 peaks at "
        f"{control['peak']:,} bytes"
    )
    failures += 0 if caught else 1
    return failures


# ---------------------------------------------------------------------------


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as workdir:
        repo = get_checkout(workdir)

        print("1. patch applies and compiles")
        failures += check_patch_applies(repo, workdir)
        if failures:
            print("\nFAIL: the patch does not apply, nothing else can run")
            return 1

        sys.path.insert(0, os.path.join(repo, "python"))
        from sgl_jax.srt.models import gpt_oss

        print("\n2. MXFP4 decoder")
        failures += check_dequantizer(gpt_oss)

        checkpoint = os.path.join(workdir, "checkpoint")
        os.makedirs(checkpoint)
        rng = np.random.default_rng(20240917)
        tensors, dense = build_checkpoint(checkpoint, rng)

        hf_model = build_hf_model(tensors, dense)
        write_config(checkpoint, hf_model.config)
        input_ids = rng.integers(0, TINY["vocab_size"], size=SEQ_LEN).astype(np.int32)
        hf_hidden, hf_logits = run_hf(hf_model, input_ids)

        mesh = jax.make_mesh(
            (4, 2),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        model = build_jax_model(mesh, checkpoint)

        print("\n3. rope")
        failures += check_rope(model, hf_model)

        print("\n4. every layer against HuggingFace, float32")
        failures += check_forward(model, mesh, input_ids, hf_hidden, hf_logits)

        print("\n5. mutants (each must be caught)")
        failures += check_mutants(mesh, checkpoint, input_ids, hf_hidden, hf_logits)

        print("\n6. lm head")
        failures += check_lm_head(model, mesh, input_ids, hf_logits)

        print("\n7. capture hook")
        failures += check_capture_hook(model, mesh, input_ids)

        print("\n8. EPLB redundant experts")
        failures += check_redundant_experts(mesh, checkpoint, input_ids, hf_hidden, hf_logits)

        print("\n9. quantization configs")
        failures += check_quantization(mesh, checkpoint)

        print("\n10. dummy weights at --tp-size 8")
        failures += check_dummy_load(checkpoint)

    print()
    if failures:
        print(f"FAIL: {failures} check(s) failed")
        return 1
    print("PASS: all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
