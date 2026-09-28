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

"""Correctness gate for the DeepSeek V4.1-Flash model patch.

Runs on CPU, with 8 and 32 simulated devices in child processes. No TPU needed.

    python3 upstream/models/test_deepseek_v41_model.py

The reference is DeepSeek's own runtime from the checkpoint repo, `inference/model.py` and
`inference/engram.py` at one pinned revision, each checked against its pinned SHA-256 before the
test imports it. Its `kernel.py` runs on CUDA only, so this file carries CPU torch equivalents of
the six kernels it calls, written from the tilelang source, and the reference runs on them in
float32. The JAX side is the patched `sgl_jax/srt/models/deepseek_v41.py`, imported from a tree
that `scripts/cpu_engine.build_tree` builds and this file patches.

Checks:

1. The patch applies to `sglang-jax` at `SGL_COMMIT` (default `eb061d8`) with
   `sglang-jax-877.patch` and both steering patches, and every file it touches compiles. A
   corrupted copy has to be refused.
2. The CPU kernels against the tilelang source's arithmetic: FP8 and FP4 round trips against
   independent JAX casts, the Sinkhorn split against the engine's own `kernels/mhc.mhc_gates`
   in interpret mode.
3. The checkpoint layout. A tiny checkpoint in the published format: FP8 E4M3 dense weights with
   E8M0 scales per 32x32 tile, routed experts as packed E2M1 with E8M0 scales per 32 inputs, the
   Engram table in FP8. Its tensor names per layer mode have to match the published index's, and
   the patched loader has to read it back to the values the reference gets.
4. Prefill. Two requests of different lengths in one batch. Every layer's four copies, every
   capture slot and the logits against the reference, which runs each request alone.
5. Split prefill. The same two requests in two passes, the cut inside an open ratio-2 group.
6. Decode. Three steps of both requests in one batch against the reference's decode path.
7. Engram. The on-device hash against `NgramHashState`, across the chunk cut and in decode,
   and the FP8 row lookup against `ParallelEngramEmbedding`.
8. Mutants. Each rewrites one line of the patched model. Each has to fail check 4, 5 or 6.
9. The real Engine on CPU with capture on, against the reference.
10. The tiny model on 8 simulated devices at --tp-size 4/8 and --ep-size 4/8/2 against itself on
    one device, with two controls that break the row-sharded Engram lookup.
11. The published config under nnx.eval_shape on 32 simulated devices: one chip's weights, the
    state pool, and the capture serve against --mem-fraction-static 0.8 of a v5p chip.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download. The published
files are cached under `DSV41_CACHE`, or `~/.cache/deepseek-v41-published/<revision>`.
`LOG_PAYLOADS=1` prints every forward batch the JAX model gets.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback
import types
import urllib.request

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PALLAS_INTERPRET", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "capture-hooks"))
import cpu_engine  # noqa: E402
from test_capture_hooks import corrupt  # noqa: E402

PATCH = os.path.join(HERE, "deepseek-v41-model.patch")
SRT = "python/sgl_jax/srt"
MODEL_FILE = f"{SRT}/models/deepseek_v41.py"
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS") == "1"

HUB = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
# SHA-256 of each published file the test reads, at REVISION.
PUBLISHED = {
    "inference/model.py": None,
    "inference/engram.py": None,
    "inference/kernel.py": None,
    "inference/vision.py": None,
    "inference/image_processor.py": None,
    "inference/config.json": None,
    "config.json": None,
    "model.safetensors.index.json": None,
}
PINS_FILE = os.path.join(HERE, "deepseek-v41-published.sha256")

# The tiny model, in the reference runtime's names. It keeps every attention mode the published
# config uses: window-only layers 0-1, ratio-2 owners 2 and 4 with reuse layers after each, the
# ratio-1 owner 6 at the encoder-decoder boundary with the candidate pool, a reindex layer 8, and
# reuse layers 7 and 9. Engram sits on a window layer and on a ratio-2 owner, as at 1 and 14.
# candidate_topk_blocks * candidate_block_size >= index_topk, as in the published config.
TINY = dict(
    vocab_size=400,
    dim=64,
    moe_inter_dim=32,
    n_layers=10,
    n_heads=4,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=2.0,
    q_lora_rank=32,
    head_dim=64,
    rope_head_dim=16,
    norm_eps=1e-20,
    o_groups=2,
    o_lora_rank=16,
    window_size=8,
    compress_ratios=[0, 0, 2, 2, 2, 2, 1, 1, 1, 1],
    kv_source_layers=[2, 4, 6],
    index_source_layers=[2, 4, 6, 8],
    compress_rope_theta=160000.0,
    original_seq_len=16,
    rope_theta=10000.0,
    rope_factor=4.0,
    beta_fast=32,
    beta_slow=1,
    # A score is 0 when every head's rectified dot product is 0, and torch.topk breaks
    # ties in no set order. With 4 heads that happens to 1 position in 16; with 16 heads to 1
    # in 65,536. The published config has 32.
    index_n_heads=16,
    index_head_dim=32,
    index_topk=4,
    candidate_source_layer=6,
    candidate_topk_blocks=2,
    candidate_block_size=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=[1, 4],
    engram_max_ngram_size=4,
    engram_vocab_size=97,
    engram_n_heads=2,
    engram_head_dim=64,
    # The published pad id. Its compressed id isn't 0, so a hash that forgot the start padding
    # reads a fresh slot's zeroed history and lands elsewhere.
    engram_pad_id=2,
)
MAX_SEQ = 64
LENGTHS = (37, 29)
CUTS = (13, 20)  # split prefill: request A cut at 13 (inside a ratio-2 group), B at 20
DECODE_STEPS = 3
TOKEN_BUCKET = 64
PAD_TOKEN = 5  # a real id, so a padding row that leaked into the state would change the result
# Per-token relative error, max over tokens and elements, against the tensor's RMS. See
# `compare` for why FP8 and FP4 rounding sets the floor.
TOL = 1e-4
FLIP_TOL = 3e-2
FLIP_FRAC = 0.25
FLIP_MIN_TOKENS = 20
ENGINE_FLIP_FRAC = 0.05
MUTANT_WORKERS = int(os.environ.get("DSV41_MUTANT_WORKERS", "8"))

RESULTS = []


def record(ok, label):
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)
    return ok


# --------------------------------------------------------------------------------------------
# Published files
# --------------------------------------------------------------------------------------------


def cache_dir():
    root = os.environ.get("DSV41_CACHE") or os.path.expanduser(
        f"~/.cache/deepseek-v41-published/{REVISION}"
    )
    os.makedirs(root, exist_ok=True)
    return root


def load_pins():
    pins = {}
    with open(PINS_FILE) as fp:
        for line in fp:
            digest, name = line.split()
            pins[name] = digest
    return pins


def fetch_published():
    """Download each file once, and check its SHA-256 against the pin before anything reads it."""
    root = cache_dir()
    pins = load_pins()
    for name in PUBLISHED:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"{HUB}/{REVISION}/{name}"
            print(f"  fetching {url}", flush=True)
            with urllib.request.urlopen(url) as resp, open(path + ".tmp", "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(path + ".tmp", path)
        with open(path, "rb") as fp:
            digest = hashlib.sha256(fp.read()).hexdigest()
        if digest != pins[name]:
            raise SystemExit(f"{name} has SHA-256 {digest}, pinned {pins[name]}")
    return root


# --------------------------------------------------------------------------------------------
# CPU equivalents of inference/kernel.py
# --------------------------------------------------------------------------------------------


def make_cpu_kernel_module():
    """The CPU stand-ins for `inference/kernel.py`, from `scripts/deepseek_cpu_kernels.py`."""
    import deepseek_cpu_kernels

    return deepseek_cpu_kernels.make_module()


def import_reference(root):
    """`inference/model.py` and `engram.py` with the CPU kernel module in place of kernel.py."""
    sys.modules["kernel"] = make_cpu_kernel_module()
    inference = os.path.join(root, "inference")
    if inference not in sys.path:
        sys.path.insert(0, inference)
    for name in ("model", "engram", "vision", "image_processor"):
        sys.modules.pop(name, None)
    model = importlib.import_module("model")
    fix_index_handoff(model)
    return model, importlib.import_module("engram")


# The reference's Indexer publishes `shared_attn.index_k = self.k_cache` only when its layer
# closed a group this step (`if self.owns_k and latent is not None`). On a decode step that
# closes no ratio-2 group, layers 2 and 4 then score against whatever cache the last publisher
# left there: layer 6's ratio-1 keys from the step before. The port reads each owner's own
# keys. HANDOFF["fixed"] makes an owner publish its cache before it scores, which is the
# reference with that one line moved; HANDOFF["reads"] records which cache each indexer read.
HANDOFF = {"fixed": True, "reads": []}


def fix_index_handoff(model):
    original = model.Indexer.forward

    def forward(self, x, qr, latent, start_pos, offset):
        if self.owns_k and HANDOFF["fixed"]:
            model.shared_attn.index_k = self.k_cache
        out = original(self, x, qr, latent, start_pos, offset)
        HANDOFF["reads"].append((start_pos, self, model.shared_attn.index_k))
        return out

    model.Indexer.forward = forward


# --------------------------------------------------------------------------------------------
# The tiny checkpoint
# --------------------------------------------------------------------------------------------


def published_config(tiny, compressed_vocab):
    """`config.json` in the published layout: text fields nested under `text_config` with the
    published names, the FP8 quantization block at the top."""
    rename = {
        "dim": "hidden_size",
        "moe_inter_dim": "moe_intermediate_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "rope_head_dim": "qk_rope_head_dim",
        "norm_eps": "rms_norm_eps",
        "n_activated_experts": "num_experts_per_tok",
        "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor",
        "window_size": "sliding_window",
        "kv_source_layers": "kv_source_layer_ids",
        "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id",
        "engram_pad_id": "engram_pad_token_id",
    }
    text = {"model_type": "deepseek_v41_text", "num_key_value_heads": 1,
            "norm_topk_prob": True, "max_position_embeddings": MAX_SEQ,
            "engram_compressed_vocab_size": compressed_vocab}
    for key, value in tiny.items():
        if key in ("original_seq_len", "rope_factor", "beta_fast", "beta_slow"):
            continue
        text[rename.get(key, key)] = value
    text["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": tiny["rope_factor"],
        "beta_fast": tiny["beta_fast"],
        "beta_slow": tiny["beta_slow"],
        "original_max_position_embeddings": tiny["original_seq_len"],
    }
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": text,
    }


# The published dtype of each tensor family, read from the shard headers of the pinned revision.
def published_dtype(name):
    if ".experts." in name and not name.endswith(".scale"):
        return "I8"
    if name.endswith(".scale"):
        return "F8_E8M0"
    fp8 = ("attn.wq_a.", "attn.wq_b.", "attn.wkv.", "attn.wo_a.", "attn.wo_b.",
           "shared_experts.", "indexer.wq_b.", "engram.wkv.", "engram.embed.")
    if any(f in name for f in fp8):
        return "F8_E4M3"
    if name.split(".")[-1].startswith("hc_") or name.endswith(("attn_sink", "gate.bias")):
        return "F32"
    return "BF16"


def e8m0(x):
    """The E8M0 code of a positive power of two."""
    return (np.round(np.log2(x)).astype(np.int32) + 127).astype(np.uint8)


def quantize_fp8_block(w, block=32):
    """Master weight -> (E4M3 bytes as float8, E8M0 codes per tile), power-of-2 scales."""
    import ml_dtypes

    out, inn = w.shape
    bo, bi = -(-out // block), -(-inn // block)
    pad = np.zeros((bo * block, bi * block), np.float32)
    pad[:out, :inn] = w
    tiles = pad.reshape(bo, block, bi, block)
    amax = np.maximum(np.abs(tiles).max(axis=(1, 3)), 1e-4)
    s = np.exp2(np.ceil(np.log2(amax / 448.0))).astype(np.float32)
    q = (tiles / s[:, None, :, None]).reshape(bo * block, bi * block)[:out, :inn]
    q = np.clip(q, -448, 448).astype(ml_dtypes.float8_e4m3fn)
    return q, e8m0(s)


def quantize_fp4(w, block=32):
    """Master weight -> (packed E2M1 as int8, even element in the low nibble; E8M0 codes)."""
    out, inn = w.shape
    wb = w.reshape(out, inn // block, block)
    amax = np.maximum(np.abs(wb).max(-1), 6 * 2.0**-126)
    s = np.exp2(np.ceil(np.log2(amax / 6.0))).astype(np.float32)
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    v = np.clip(wb / s[..., None], -6, 6).reshape(out, inn)
    code = np.abs(np.abs(v)[..., None] - grid).argmin(-1).astype(np.uint8)
    code = code | (np.signbit(v).astype(np.uint8) << 3)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed.view(np.int8), e8m0(s)


def dequant_fp8_ref(q, codes, shape, block=32):
    s = np.exp2(codes.astype(np.float32) - 127)
    s = np.repeat(np.repeat(s, block, 0), block, 1)[: shape[0], : shape[1]]
    return q.astype(np.float32) * s


def dequant_fp4_ref(packed, codes, block=32):
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
    p = packed.view(np.uint8)
    lo, hi = p & 0xF, p >> 4
    vals = np.empty((p.shape[0], p.shape[1] * 2), np.float32)
    vals[:, 0::2], vals[:, 1::2] = table[lo], table[hi]
    s = np.exp2(codes.astype(np.float32) - 127)
    return vals * np.repeat(s, block, 1)


def write_safetensors(path, tensors):
    """Raw writer: {name: (dtype tag, numpy array)}."""
    header, blobs, offset = {}, [], 0
    for name, (tag, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", len(head)))
        fp.write(head)
        for raw in blobs:
            fp.write(raw)


def to_bf16_bits(x):
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded


def from_bf16_bits(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def build_checkpoint(path, ref_names_shapes, rng, engram_rows):
    """Random masters in the published format. Returns (the values the reference loads,
    the tensors written)."""
    tensors, values = {}, {}
    for name, shape in ref_names_shapes.items():
        tag = published_dtype(name)
        if name.endswith(".scale"):
            continue
        fan_in = shape[-1] if len(shape) > 1 else 1
        if name.endswith(("norm.weight", "q_weight", "k_weight")):
            master = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith("hc_attn_fn") or name.endswith("hc_ffn_fn"):
            master = 0.05 * rng.standard_normal(shape)
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            master = 0.5 + 0.2 * rng.standard_normal(shape)
        elif name.endswith(("hc_attn_base", "hc_ffn_base", "attn_sink", "gate.bias")):
            master = 0.3 * rng.standard_normal(shape)
        elif name == "embed.weight":
            master = rng.standard_normal(shape)
        else:
            master = rng.standard_normal(shape) / math.sqrt(fan_in)
        master = master.astype(np.float32)
        if name.endswith("engram.embed.weight"):
            q, codes = quantize_fp8_rows(master)
            tensors[name] = ("F8_E4M3", q.view(np.uint8))
            tensors[name[: -len("weight")] + "scale"] = ("F8_E8M0", codes)
            values[name] = q.astype(np.float32)
            values[name[: -len("weight")] + "scale"] = np.exp2(codes.astype(np.float32) - 127)
        elif tag == "F8_E4M3":
            q, codes = quantize_fp8_block(master)
            tensors[name] = (tag, q.view(np.uint8))
            tensors[name[: -len("weight")] + "scale"] = ("F8_E8M0", codes)
            values[name] = dequant_fp8_ref(q, codes, master.shape)
        elif tag == "I8":
            packed, codes = quantize_fp4(master)
            tensors[name] = (tag, packed)
            tensors[name[: -len("weight")] + "scale"] = ("F8_E8M0", codes)
            values[name] = dequant_fp4_ref(packed, codes)
        elif tag == "BF16":
            bits = to_bf16_bits(master)
            tensors[name] = (tag, bits)
            values[name] = from_bf16_bits(bits)
        else:
            tensors[name] = (tag, master)
            values[name] = master
        if name.endswith("ffn.gate.bias"):
            # the image-token routing bias; a text-only reference and port both leave it unread
            tensors[name + "_vl"] = ("F32", (0.3 * rng.standard_normal(shape)).astype(np.float32))
    write_safetensors(os.path.join(path, "model-00001-of-00001.safetensors"), tensors)
    return values, tensors


def quantize_fp8_rows(t, block=32):
    """Engram rows: E4M3 with one E8M0 scale per 32 channels of a row."""
    import ml_dtypes

    rows, dim = t.shape
    tb = t.reshape(rows, dim // block, block)
    amax = np.maximum(np.abs(tb).max(-1), 1e-4)
    s = np.exp2(np.ceil(np.log2(amax / 448.0))).astype(np.float32)
    q = np.clip(tb / s[..., None], -448, 448).reshape(rows, dim).astype(ml_dtypes.float8_e4m3fn)
    return q, e8m0(s)


# --------------------------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------------------------


def reference_args(ref_model, tiny, extra):
    fields = {f for f in ref_model.ModelArgs.__dataclass_fields__}
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in tiny.items() if k in fields}
    kwargs.update(extra)
    return ref_model.ModelArgs(**kwargs)


class Reference:
    """The reference Transformer in float32, with hooks that record every layer."""

    def __init__(self, ref_model, args, tokenizer, values, bf16=False):
        import torch

        torch.set_default_dtype(torch.bfloat16 if bf16 else torch.float32)
        torch.manual_seed(0)
        self.torch = torch
        self.model = ref_model.Transformer(args, tokenizer)
        torch.set_default_dtype(torch.float32)
        if not bf16:
            self.model = self.model.float()
        state = self.model.state_dict()
        missing = []
        load = {}
        for name, tensor in state.items():
            if name not in values:
                missing.append(name)
                continue
            load[name] = torch.from_numpy(
                np.asarray(values[name], np.float32).reshape(tensor.shape)).to(tensor.dtype)
        if missing:
            raise AssertionError(f"reference parameters with no checkpoint value: {missing[:8]}")
        self.model.load_state_dict(load, strict=True)
        self.args, self.values = args, values
        self.records = {}
        for i, layer in enumerate(self.model.layers):
            layer.attn_norm.register_forward_pre_hook(self._pre("collapsed", i))
            layer.register_forward_hook(self._post("block", i))
            if layer.engram is not None:
                layer.engram.register_forward_hook(self._out("engram", i))
                # ParallelEngramEmbedding returns BF16 for the FP8 GEMM behind it. A dequantized
                # row is an E4M3 value times a power of two, which BF16 holds, so widening it
                # back to float32 for the float `wkv` loses nothing.
                wkv = layer.engram.wkv.weight
                layer.engram.embed.register_forward_hook(lambda m, a, out, w=wkv: out.to(w.dtype))
        self.model.norm.register_forward_hook(self._out("final", -1))

    def _pre(self, kind, i):
        def hook(module, args):
            self.records[(kind, i)] = args[0].detach().clone()
        return hook

    def _post(self, kind, i):
        def hook(module, args, output):
            self.records[(kind, i)] = output[0].detach().clone()
        return hook

    def _out(self, kind, i):
        def hook(module, args, output):
            self.records[(kind, i)] = output.detach().clone()
        return hook

    def run(self, ids, start_pos):
        torch = self.torch
        self.records = {}
        self.model(torch.tensor([ids]), start_pos)
        out = {k: v[0].double().numpy() for k, v in self.records.items()}  # noqa: E501
        out[("logits", -1)] = (self.records[("final", -1)][0].double()
                                @ self.model.head.weight.detach().double().T).numpy()
        return out


def reference_state_names(ref_model, args, tokenizer):
    import torch

    torch.set_default_dtype(torch.float32)
    model = ref_model.Transformer(args, tokenizer)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------------------------
# JAX side
# --------------------------------------------------------------------------------------------


def import_patched(tree):
    sys.path.insert(0, os.path.join(tree, "python"))
    for name in list(sys.modules):
        if name.startswith("sgl_jax.srt.models.deepseek_v41") or name.startswith("sgl_jax.srt.configs.deepseek_v41"):
            del sys.modules[name]
    mod = importlib.import_module("sgl_jax.srt.models.deepseek_v41")
    cfgmod = importlib.import_module("sgl_jax.srt.configs.deepseek_v41")
    return mod, cfgmod


def make_mesh(tensor=1):
    from jax.sharding import AxisType, Mesh

    devices = np.array(jax.devices()[:tensor]).reshape(1, tensor)
    return Mesh(devices, ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit))


class JaxRunner:
    def __init__(self, mod, cfgmod, ckpt, tokenizer, mesh, dtype=jnp.float32, ep_size=1):
        self.mod = mod
        self.mesh = mesh
        self.dtype = dtype
        with open(os.path.join(ckpt, "config.json")) as fp:
            raw = json.load(fp)
        self.cfg = cfgmod.DeepseekV41Config(**{k: v for k, v in raw.items() if k != "model_type"})
        self.cfg.ep_size = ep_size  # the runner writes --ep-size onto hf_config the same way
        with jax.set_mesh(mesh):
            self.model = mod.DeepseekV41ForCausalLM(self.cfg, mesh, dtype=dtype)
            self.used = mod.load_checkpoint(self.model, ckpt)
            if self.model.model.engram_hash is not None:
                self.model.set_token_map(tokenizer)
        self.model.model.return_streams = True
        self.model.set_layers_to_capture(range(int(self.cfg.n_layers)))
        self.state = mod.init_state(self.cfg, 3, MAX_SEQ, dtype)

    def forward(self, chunks):
        """chunks: [(req_slot, prefix_len, ids)]. Returns per-request dicts like the reference's."""
        ids = np.concatenate([np.asarray(c[2], np.int32) for c in chunks])
        lens = np.array([len(c[2]) for c in chunks], np.int32)
        prefix = np.array([c[1] for c in chunks], np.int32)
        slots = np.array([c[0] for c in chunks], np.int32)
        if LOG_PAYLOADS:
            print(f"    payload: ids={ids.tolist()} lens={lens.tolist()} prefix={prefix.tolist()} "
                  f"req_pool_indices={slots.tolist()}", flush=True)
        # The engine pads every batch to a token bucket. The megablox GMM needs the routed row
        # count, tokens times top-k, to divide by its 128 tile, which a 64-token bucket gives.
        # Padding tokens have to leave the state and the real rows alone.
        padded = -(-len(ids) // TOKEN_BUCKET) * TOKEN_BUCKET
        ids = np.concatenate([ids, np.full(padded - len(ids), PAD_TOKEN, np.int32)])
        with jax.set_mesh(self.mesh):
            meta = self.mod.token_meta(len(ids), slots, prefix, lens)
            hidden, aux, self.state, streams = self.model.model(jnp.asarray(ids), meta, self.state)
            logits = hidden.astype(jnp.float32) @ self.model.lm_head.embedding.value.T
        out = []
        start = 0
        for n in lens:
            rows = slice(start, start + n)
            rec = {}
            for kind, i, arr in streams:
                rec[(kind, i)] = np.asarray(arr[rows].astype(jnp.float32), np.float64)
            rec[("final", -1)] = np.asarray(hidden[rows].astype(jnp.float32), np.float64)
            rec[("logits", -1)] = np.asarray(logits[rows], np.float64)
            rec[("aux", -1)] = [np.asarray(a[rows].astype(jnp.float32), np.float64) for a in aux]
            out.append(rec)
            start += n
        return out


COMPARED = ("block", "collapsed", "engram", "final", "logits")


def compare(label, jax_rec, ref_rec, rows=None):
    """Per-token error over every recorded tensor: for each tensor, the max abs difference over a
    token's elements divided by the reference tensor's RMS; a token's error is its worst tensor.
    Returns the per-token errors and the tensor that set the worst one."""
    tok_err, where, worst = None, None, -1.0
    for key, ref in ref_rec.items():
        if key not in jax_rec or key[0] not in COMPARED:
            continue
        got = jax_rec[key]
        ref = ref if rows is None else ref[rows]
        if got.shape != ref.shape:
            raise AssertionError(f"{label} {key}: shape {got.shape} vs reference {ref.shape}")
        scale = np.sqrt(np.mean(ref**2)) + 1e-30
        err = np.max(np.abs(got - ref).reshape(ref.shape[0], -1), axis=1) / scale
        err = np.where(np.isfinite(got.reshape(ref.shape[0], -1)).all(1), err, np.inf)
        tok_err = err if tok_err is None else np.maximum(tok_err, err)
        if err.max() > worst:
            worst, where = float(err.max()), key
    for i, cap in enumerate(jax_rec.get(("aux", -1), [])):
        ref = ref_rec[("collapsed", i)] if rows is None else ref_rec[("collapsed", i)][rows]
        err = np.max(np.abs(cap - ref), axis=1) / (np.sqrt(np.mean(ref**2)) + 1e-30)
        tok_err = np.maximum(tok_err, err)
        if err.max() > worst:
            worst, where = float(err.max()), ("capture slot", i)
    return tok_err, where


def verdict(tok_err):
    """Float32 agreement on most tokens, a bounded rounding step on the rest.

    Both sides round cached latents to FP8 and FP4 grids. A float32 difference of one ulp in front
    of a round can move one element one grid step, and every query that reads that latent then
    carries an error of a few 1e-3 (seen: token 26 of layer 6 at a 128-token bucket, and the 7
    queries whose window holds it). So a check passes when at most FLIP_FRAC of its tokens exceed
    TOL and none exceeds FLIP_TOL. One flipped latent reaches every later decode step that reads
    it, so a sample of a few decode tokens can be half flips; the fraction rule applies from
    FLIP_MIN_TOKENS tokens up, and the FLIP_TOL bound applies always."""
    frac = float(np.mean(tok_err > TOL))
    worst = float(np.max(tok_err))
    frac_ok = frac <= FLIP_FRAC or tok_err.size < FLIP_MIN_TOKENS
    return (frac_ok and worst <= FLIP_TOL), frac, worst


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def git(repo, *args, stdin=None):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, input=stdin)


def check_patch(workdir):
    print("check 1: the patch applies and compiles", flush=True)
    tree = cpu_engine.build_tree(os.path.join(workdir, "sglang-jax"),
                                 os.environ.get("SGLANG_JAX_REPO") or None)
    record(git(tree, "apply", "--check", PATCH).returncode == 0,
           f"deepseek-v41-model.patch applies to {cpu_engine.SGL_COMMIT} with the capture and steering patches")
    with open(PATCH) as fp:
        bad = os.path.join(workdir, "corrupt.patch")
        with open(bad, "w") as out:
            out.write(corrupt(fp.read()))
    record(git(tree, "apply", "--check", bad).returncode != 0, "control: a corrupted patch is refused")
    applied = git(tree, "apply", PATCH)
    if applied.returncode != 0:
        record(False, f"apply failed: {applied.stderr.strip()}")
        return None
    touched = [l[6:].strip() for l in open(PATCH) if l.startswith("+++ b/")]
    record(len(touched) == 5, f"the patch touches {len(touched)} files: {', '.join(touched)}")
    for path in touched:
        if path.endswith(".py"):
            done = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(tree, path)],
                                  capture_output=True, text=True)
            record(done.returncode == 0, f"{path} compiles")
    return tree


def check_kernels(mod, tree):
    print("check 2: CPU kernels against independent arithmetic", flush=True)
    import torch

    k = sys.modules["kernel"]
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((6, 64)) * np.exp(rng.uniform(-6, 6, (6, 1)))).astype(np.float32)
    t = torch.from_numpy(x.copy())
    k.act_quant(t, 32, "ue8m0", torch.float8_e8m0fnu, True)
    j = np.asarray(mod.fp8_roundtrip(jnp.asarray(x), 32))
    record(np.array_equal(t.numpy(), j), "FP8 round trip: torch bit-level scale == JAX frexp scale, every element")
    for block, e4 in ((32, False), (16, True)):
        t = torch.from_numpy(x.copy())
        k.fp4_act_quant(t, block, True, torch.float8_e4m3fn if e4 else torch.float8_e8m0fnu)
        j = np.asarray(mod.fp4_roundtrip(jnp.asarray(x), block, e4m3_scale=e4))
        record(np.array_equal(t.numpy(), j),
               f"FP4 round trip, block {block}, {'E4M3' if e4 else 'E8M0'} scale: torch grid search == ml_dtypes cast")
    # tie cases of E2M1: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 round to the even code
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0])
    record(torch.equal(k.round_e2m1(ties), want), "E2M1 ties go to the even code")
    wrong = torch.tensor([0.5, 0.5, 1.5, 1.5, 3.0, 3.0, 6.0, -0.5])
    record(not torch.equal(k.round_e2m1(ties), wrong), "control: ties-away rounding differs")

    # Sinkhorn split against the engine's own mhc_gates kernel in interpret mode
    from sgl_jax.srt.kernels.mhc import mhc as mhc_kernels
    from sgl_jax.srt.kernels.mhc import mhc_gates

    # kernels/mhc/tune.py schedules only "TPU v6e" and raises on any other device kind, v5p and
    # CPU included. The math doesn't depend on the tile, so interpret mode runs under that one.
    mhc_kernels._device_kind = lambda: "TPU v6e"
    hc = 4
    mixes = rng.standard_normal((5, (2 + hc) * hc)).astype(np.float32)
    scale = rng.uniform(0.2, 1.0, 3).astype(np.float32)
    base = rng.standard_normal((2 + hc) * hc).astype(np.float32)
    pre, post, comb = mod.sinkhorn_split(jnp.asarray(mixes), jnp.asarray(scale), jnp.asarray(base), hc, 20, 1e-6)
    epost, ecomb = mhc_gates(jnp.asarray(mixes), jnp.asarray(scale), jnp.asarray(base),
                             hc_mult=hc, sinkhorn_iters=20, eps=1e-6, interpret=True)
    err = max(float(jnp.max(jnp.abs(post - epost))), float(jnp.max(jnp.abs(comb - ecomb))))
    record(err < 1e-6, f"Sinkhorn split == engine kernels/mhc mhc_gates, max err {err:.2e}")
    tp, tpost, tcomb = k.hc_split_sinkhorn(torch.from_numpy(mixes), torch.from_numpy(scale),
                                           torch.from_numpy(base), hc, 20, 1e-6)
    err = max(float(np.max(np.abs(np.asarray(pre) - tp.numpy()))),
              float(np.max(np.abs(np.asarray(comb) - tcomb.numpy()))))
    record(err < 1e-6, f"Sinkhorn split == CPU reference kernel, max err {err:.2e}")
    record(float(jnp.max(jnp.abs(jnp.sum(comb, -2) - 1))) < 1e-4, "comb columns sum to one")


def check_layout(root, names_shapes, tensors, mod, runner):
    print("check 3: checkpoint layout and loader", flush=True)
    with open(os.path.join(root, "model.safetensors.index.json")) as fp:
        published = json.load(fp)["weight_map"]
    # layer modes: tiny layer -> published layer of the same mode
    same_mode = {0: 0, 1: 1, 2: 2, 3: 3, 4: 14, 5: 15, 6: 20, 7: 21, 8: 24, 9: 25}

    def family(names, layer):
        pre = f"layers.{layer}."
        out = set()
        for n in names:
            if n.startswith(pre):
                rest = n[len(pre):]
                parts = rest.split(".")
                if parts[:2] == ["ffn", "experts"]:
                    rest = "ffn.experts.N." + ".".join(parts[3:])
                out.add(rest)
        return out

    ok = True
    for tiny_layer, pub_layer in same_mode.items():
        mine = family(tensors, tiny_layer)
        theirs = family(published, pub_layer)
        if mine != theirs:
            ok = False
            print(f"      layer {tiny_layer} vs published {pub_layer}: only tiny {sorted(mine - theirs)}, "
                  f"only published {sorted(theirs - mine)}")
    record(ok, "tiny tensor names per layer match the published layer of the same mode (0,1,2,3,14,15,20,21,24,25)")
    tops = {n for n in tensors if not n.startswith("layers.")}
    record(tops == {"embed.weight", "head.weight", "norm.weight"}, f"root tensors {sorted(tops)}")
    text_only = {n for n in tensors if not n.endswith(".ffn.gate.bias_vl")}
    record(set(runner.used) == text_only,
           f"the loader read all {len(text_only)} text tensors and skipped the "
           f"{len(tensors) - len(text_only)} image-token biases")

    # partial last tile, which no tiny shape reaches
    w = np.random.default_rng(3).standard_normal((40, 70)).astype(np.float32)
    q, codes = quantize_fp8_block(w)
    got = mod.dequant_fp8_block(q, codes)
    record(np.array_equal(got, dequant_fp8_ref(q, codes, w.shape)), "FP8 dequant with a partial last tile in both axes")
    packed, codes = quantize_fp4(w[:, :64])
    got = mod.dequant_fp4(packed, codes)
    record(np.array_equal(got, dequant_fp4_ref(packed, codes)), "FP4 dequant, low nibble is the even element")
    swapped = ((packed.view(np.uint8) >> 4) | (packed.view(np.uint8) << 4)).astype(np.uint8).view(np.int8)
    record(not np.array_equal(mod.dequant_fp4(swapped, codes), got), "control: swapped nibbles decode differently")


def run_all(tree, root, quiet=False):
    ref_model, ref_engram = import_reference(root)
    mod, cfgmod = import_patched(tree)
    if not quiet:
        check_kernels(mod, tree)

    tokdir = tempfile.mkdtemp(prefix="dsv41-tok-")
    atexit.register(shutil.rmtree, tokdir, True)
    cpu_engine.write_tokenizer(tokdir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokdir)
    _, compressed_vocab = ref_engram.build_compressed_token_map(tokenizer)
    tiny = dict(TINY, vocab_size=len(tokenizer))
    args = reference_args(ref_model, tiny, dict(
        max_batch_size=1, max_seq_len=MAX_SEQ, dtype="bf16", expert_dtype=None,
        engram_compressed_vocab_size=compressed_vocab, n_mtp_layers=0, temperature=0.0,
    ))
    layout = ref_engram.EngramLayout.from_args(
        types.SimpleNamespace(**{**{k: getattr(args, k) for k in args.__dataclass_fields__},
                                 "engram_num_embeddings": (1,) * len(args.engram_layer_ids)}))
    rows = tuple(sum(sum(p) for p in layer) for layer in layout.primes)
    args.engram_num_embeddings = rows
    tiny["engram_num_embeddings"] = list(rows)

    ckpt = tempfile.mkdtemp(prefix="dsv41-ckpt-")
    atexit.register(shutil.rmtree, ckpt, True)
    names_shapes = reference_state_names(ref_model, args, tokenizer)
    values, tensors = build_checkpoint(ckpt, names_shapes, np.random.default_rng(0), rows)
    with open(os.path.join(ckpt, "config.json"), "w") as fp:
        json.dump(published_config(tiny, compressed_vocab), fp, indent=1)
    tokenizer.save_pretrained(ckpt)

    reference = Reference(ref_model, args, tokenizer, values)
    runner = JaxRunner(mod, cfgmod, ckpt, tokenizer, make_mesh())
    if not quiet:
        check_layout(root, names_shapes, tensors, mod, runner)

    rng = np.random.default_rng(7)
    seqs = [rng.integers(3, len(tokenizer), n + DECODE_STEPS).tolist() for n in LENGTHS]
    return mod, cfgmod, ref_model, ref_engram, reference, runner, seqs, ckpt, tokenizer


def check_reference_handoff(reference, seqs):
    """The reference bug `fix_index_handoff` works around, shown on the tiny model."""
    print("reference index-key handoff", flush=True)
    layers = list(reference.model.layers)

    def owner_of(cache):
        for i, l in enumerate(layers):
            ix = l.attn.indexer
            if ix is not None and ix.owns_k and ix.k_cache.data_ptr() == cache.data_ptr():
                return i
        return None

    def reads(fixed):
        HANDOFF["fixed"], HANDOFF["reads"] = fixed, []
        n = LENGTHS[0]
        reference.run(seqs[0][:n], 0)
        reference.run([seqs[0][n]], n)
        reference.run([seqs[0][n + 1]], n + 1)
        out = {(sp, [i for i, l in enumerate(layers) if l.attn.indexer is ix][0]): owner_of(c)
               for sp, ix, c in HANDOFF["reads"]}
        HANDOFF["fixed"], HANDOFF["reads"] = True, []
        return out

    n = LENGTHS[0]
    raw, fixed = reads(False), reads(True)
    record(raw[(n + 1, 2)] == 6 and raw[(n + 1, 4)] == 6,
           f"unpatched reference: at start_pos {n + 1}, which closes no ratio-2 group, the layer 2 "
           f"and layer 4 indexers read layer {raw[(n + 1, 2)]} and {raw[(n + 1, 4)]}'s ratio-1 keys")
    record(fixed[(n + 1, 2)] == 2 and fixed[(n + 1, 4)] == 4 and fixed[(n + 1, 8)] == 6,
           "patched reference: layers 2, 4 and 8 read the keys of their owners 2, 4 and 6")


def reference_runs(reference, seqs):
    """Full prefill of each request alone, then its decode steps."""
    runs = []
    for seq, n in zip(seqs, LENGTHS):
        steps = [reference.run(seq[:n], 0)]
        for s in range(DECODE_STEPS):
            steps.append(reference.run([seq[n + s]], n + s))
        runs.append(steps)
    return runs


def check_forward(runner, ref_runs, seqs, label_prefix="", quiet=False):
    """Checks 4-6 on one runner. Returns {check: (passed, fraction over TOL, worst)}."""
    out = {}

    def report(name, errs, label):
        tok = np.concatenate([e[0] for e in errs])
        ok, frac, worst = verdict(tok)
        out[name] = (ok, frac, worst)
        if not quiet:
            where = max(errs, key=lambda e: e[0].max())[1]
            record(ok, f"{label_prefix}{label}: {tok.size} tokens, {frac:.0%} above {TOL:.0e}, "
                       f"worst {worst:.2e} at {where}")

    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    got = runner.forward([(0, 0, seqs[0][: LENGTHS[0]]), (1, 0, seqs[1][: LENGTHS[1]])])
    report("prefill", [compare("prefill", g, r[0]) for g, r in zip(got, ref_runs)],
           f"prefill of {LENGTHS[0]} and {LENGTHS[1]} tokens in one batch, every layer's four "
           "copies, every capture slot, the logits")

    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    a, b = seqs[0][: LENGTHS[0]], seqs[1][: LENGTHS[1]]
    first = runner.forward([(0, 0, a[: CUTS[0]]), (1, 0, b[: CUTS[1]])])
    second = runner.forward([(0, CUTS[0], a[CUTS[0]:]), (1, CUTS[1], b[CUTS[1]:])])
    errs = []
    for i, (f, s_, r) in enumerate(zip(first, second, ref_runs)):
        errs.append(compare("split1", f, r[0], rows=slice(0, CUTS[i])))
        errs.append(compare("split2", s_, r[0], rows=slice(CUTS[i], LENGTHS[i])))
    report("split", errs, f"prefill split at {CUTS[0]} and {CUTS[1]} against the unsplit reference")

    errs = []
    for s in range(DECODE_STEPS):
        step = runner.forward([(0, LENGTHS[0] + s, [seqs[0][LENGTHS[0] + s]]),
                               (1, LENGTHS[1] + s, [seqs[1][LENGTHS[1] + s]])])
        errs += [compare("decode", g, ref_runs[i][1 + s]) for i, g in enumerate(step)]
    report("decode", errs, f"{DECODE_STEPS} decode steps x 2 requests after the split prefill")
    return out


def check_bf16(mod, cfgmod, ref_model, args, tokenizer, values, ckpt, seqs, ref_runs):
    """The serving dtype. The port in BF16 against the float32 reference has to sit within twice
    the reference's own BF16 error, per token at the median, the floor `scripts/check_capture.py`
    holds a capture to."""
    print("BF16: the port against the reference's own BF16 floor", flush=True)
    ref16 = Reference(ref_model, args, tokenizer, values, bf16=True)
    runner16 = JaxRunner(mod, cfgmod, ckpt, tokenizer, make_mesh(), dtype=jnp.bfloat16)
    got = runner16.forward([(0, 0, seqs[0][: LENGTHS[0]]), (1, 0, seqs[1][: LENGTHS[1]])])
    for i in range(2):
        floor_rec = ref16.run(seqs[i][: LENGTHS[i]], 0)
        port_err, _ = compare("bf16", got[i], ref_runs[i][0])
        floor_err, _ = compare("bf16-floor", floor_rec, ref_runs[i][0])
        port_med, floor_med = float(np.median(port_err)), float(np.median(floor_err))
        agree = float(np.mean(np.argmax(got[i][("logits", -1)], -1) ==
                              np.argmax(ref_runs[i][0][("logits", -1)], -1)))
        record(np.isfinite(port_err).all() and port_med <= 2 * floor_med,
               f"request {i}: BF16 port median per-token error {port_med:.2e}, reference BF16 floor "
               f"{floor_med:.2e}; argmax agrees with float32 on {agree:.0%} of positions")


def check_capture_path(mod, runner, seqs):
    """The capture hook with a subset of slots, and the engine's LogitsProcessor on the port's
    final hidden state."""
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

    print("capture hook and logits processor", flush=True)
    n = LENGTHS[0]
    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    full = runner.forward([(0, 0, seqs[0][:n])])[0]
    runner.model.set_layers_to_capture([2, 5, 9])
    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    part = runner.forward([(0, 0, seqs[0][:n])])[0]
    runner.model.set_layers_to_capture(range(int(runner.cfg.n_layers)))
    aux = part[("aux", -1)]
    ok = len(aux) == 3 and all(np.array_equal(aux[i], full[("collapsed", l)]) for i, l in enumerate((2, 5, 9)))
    record(ok, "layers_to_capture [2, 5, 9] returns those three collapsed attention inputs, in order")
    record(not np.array_equal(full[("collapsed", 2)], full[("collapsed", 5)]),
           "control: slots 2 and 5 differ")
    hidden = jnp.asarray(full[("final", -1)], jnp.float32)
    slots = [jnp.asarray(a, jnp.float32) for a in aux]
    with jax.set_mesh(runner.mesh):
        out = runner.model.logits_processor(
            hidden, runner.model.lm_head,
            LogitsMetadata(forward_mode=ForwardMode.DECODE, capture_hidden_mode=CaptureHiddenMode.FULL),
            aux_hidden_states=slots)
    got = np.asarray(out.next_token_logits, np.float64)
    err = np.max(np.abs(got - full[("logits", -1)])) / np.sqrt(np.mean(full[("logits", -1)] ** 2))
    record(err < 1e-5, f"LogitsProcessor on the FP32 head gives the logits the gate compares, err {err:.1e}")
    stored = np.asarray(out.hidden_states, np.float64)
    want = np.concatenate([full[("collapsed", l)] for l in (2, 5, 9)], axis=-1)
    record(stored.shape == want.shape and np.allclose(stored, want, rtol=0, atol=1e-6),
           f"LogitsProcessor stores the three slots side by side, {stored.shape}")


def check_engram_published(mod, ref_engram, root):
    """The hash at the published sizes: primes near 16,000,000, a compressed vocab of 99,092,
    8 heads, n-grams up to 4, layers 1 and 14. The tiny model's primes sit near 97, where the
    limb arithmetic never carries far, so this runs the real moduli."""
    import torch

    print("check 7: Engram hash at the published sizes against NgramHashState", flush=True)
    with open(os.path.join(root, "inference", "config.json")) as fp:
        pub = json.load(fp)
    ns = types.SimpleNamespace(**pub)
    ref_layout = ref_engram.EngramLayout.from_args(ns)
    rows = [sum(sum(per) for per in layer) for layer in ref_layout.primes]
    record(rows == list(pub["engram_num_embeddings"]),
           f"bucket primes sum to the published row counts {pub['engram_num_embeddings']}")
    cfg = types.SimpleNamespace(**pub)
    cfg.engram_pad_id = pub["engram_pad_id"]
    layout = mod.EngramLayout.from_config(cfg)
    record(layout.primes == ref_layout.primes, "the port draws the same 48 primes as find_next_prime")

    vocab = pub["engram_compressed_vocab_size"]
    state_ref = ref_engram.NgramHashState.__new__(ref_engram.NgramHashState)
    torch.nn.Module.__init__(state_ref)
    state_ref.layout = ref_layout
    state_ref.pad_id = pub["engram_pad_id"]
    flat = [[p for per in layer for p in per] for layer in ref_layout.primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    state_ref.register_buffer("primes", torch.tensor(ref_layout.primes), persistent=False)
    state_ref.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
    state_ref.register_buffer("multipliers", ref_engram.compute_hash_multipliers(
        ref_layout.layer_ids, ref_layout.max_ngram_size, vocab), persistent=False)
    state_ref.register_buffer("token_map", torch.arange(vocab), persistent=False)
    state_ref.register_buffer("cache", torch.empty(1, 64, dtype=torch.int64), persistent=False)

    hashing = mod.DeepseekV41EngramHash(cfg, layout, vocab)
    hashing.token_map.value = jnp.arange(vocab, dtype=jnp.int32)
    hashing.pad_id.value = jnp.asarray(pub["engram_pad_id"], jnp.int32)
    rng = np.random.default_rng(11)
    ids = rng.integers(0, vocab, 40)
    ids[5], ids[6] = vocab - 1, 0
    state = {"engram_hist": [jnp.zeros((2, 3), jnp.int32)]}
    ok = True
    for prefix, n in ((0, 13), (13, 1), (14, 1), (15, 25)):
        chunk = ids[prefix : prefix + n]
        want = state_ref(torch.tensor([chunk]), prefix)[0].numpy()
        meta = mod.token_meta(n, np.array([0]), np.array([prefix]), np.array([n]))
        got, state["engram_hist"][0] = hashing(jnp.asarray(chunk, jnp.int32), meta, state)
        same = np.array_equal(np.asarray(got), want)
        ok &= same
        if not same:
            print(f"      chunk at {prefix}: {int(np.sum(np.asarray(got) != want))} ids differ")
    record(ok, "hash ids equal NgramHashState's over chunks of 13, 1, 1 and 25 tokens, every id")


def reference_greedy(reference, prompt, steps):
    """Greedy decode on the reference: rows are the collapsed attention input of every layer for
    each position a forward pass read, plus the generated ids and each step's top-2 logit gap."""
    ids, rows, gaps = list(prompt), [], []
    out = reference.run(ids, 0)
    n_layers = len(reference.model.layers)
    for step in range(steps):
        rows.append(np.stack([out[("collapsed", l)] for l in range(n_layers)], 1))
        logits = out[("logits", -1)][-1]
        top = np.sort(logits)[-2:]
        gaps.append(float(top[1] - top[0]))
        ids.append(int(np.argmax(logits)))
        if step + 1 < steps:
            out = reference.run([ids[-1]], len(ids) - 1)
    return ids[len(prompt):], np.concatenate(rows, 0), gaps


def check_engine(ckpt, reference, seqs):
    """The real Engine on CPU: loader, runner, the state pool hook, chunked prefill, the
    scheduler's batches, and capture."""
    print("check 9: the real Engine on CPU with capture on", flush=True)
    steps = 3
    prompts = [seqs[0][: LENGTHS[0]], seqs[1][: LENGTHS[1]]]
    want = [reference_greedy(reference, p, steps) for p in prompts]
    engine = cpu_engine.open_engine(ckpt, capture=True, batch_size=4, token_padding=64,
                                    disable_radix_cache=True)
    try:
        outs = engine.generate(input_ids=prompts,
                               sampling_params={"temperature": 0.0, "max_new_tokens": steps},
                               return_hidden_states=True)
        if LOG_PAYLOADS:
            for o in outs:
                print("    reply:", {k: v for k, v in o["meta_info"].items() if k != "hidden_states"})
    finally:
        engine.shutdown()
    import capture_activations

    for i, (o, (ids, rows, gaps)) in enumerate(zip(outs, want)):
        got_ids = list(o["output_ids"])
        record(got_ids == ids, f"request {i}: greedy ids {got_ids} == reference {ids} "
                               f"(smallest top-2 logit gap {min(gaps):.2e})")
        hidden, prompt_rows = capture_activations.hidden_states_from_output(o)
        n = min(len(hidden), len(rows))
        err = np.max(np.abs(hidden[:n] - rows[:n]), axis=(1, 2)) / np.sqrt(np.mean(rows**2))
        # The engine jits the whole forward, so its float32 rounding differs from the eager
        # runner's by an ulp here and there. In front of an FP4 round that can move one index
        # query element a grid step, and a query whose candidate-block margin is that thin keeps
        # another block; the reindex layers after the candidate source then read another
        # latent for that one token. A selection flip isn't bounded in size, so this check
        # bounds how many rows it touches instead: at most ENGINE_FLIP_FRAC.
        frac = float(np.mean(err > TOL))
        worst = float(err.max())
        ok = frac <= ENGINE_FLIP_FRAC
        record(ok and prompt_rows == len(prompts[i]) and len(hidden) == len(rows),
               f"request {i}: {len(hidden)} captured rows x {hidden.shape[1]} slots against the "
               f"reference's collapsed attention inputs, {frac:.0%} above {TOL:.0e}, worst {worst:.2e}"
               + (f", rows over {FLIP_TOL:.0e}: {np.nonzero(err > FLIP_TOL)[0].tolist()} "
                  f"(prefill rows 0-{prompt_rows - 1})" if worst > FLIP_TOL else ""))
        if worst > FLIP_TOL:
            bad = int(np.argmax(err))
            per_slot = np.max(np.abs(hidden[bad] - rows[bad]), axis=-1) / np.sqrt(np.mean(rows**2))
            print(f"      row {bad} error per slot: {np.array2string(per_slot, precision=2)}")


# Each mutant rewrites one line of the patched model. (name, old, new).
MUTANTS = [
    ("attention output not rotated back", "o = apply_rope(o, cos, sin, rd, inverse=True)",
     "o = apply_rope(o, cos, sin, rd, inverse=False)"),
    ("attention collapses with its own pre mix", "collapsed = hc_pre(x, pre_mix)",
     "collapsed = hc_pre(x, a_pre)"),
    ("comb transposed in hc_post", '"tij,tid->tjd"', '"tji,tid->tjd"'),
    ("attention sink dropped", "denom = jnp.sum(p, axis=-1) + jnp.exp(sink - m)",
     "denom = jnp.sum(p, axis=-1)"),
    ("compressor softmax over channels", "jax.nn.softmax(mem_sc, axis=1)",
     "jax.nn.softmax(mem_sc, axis=-1)"),
    ("candidate pool ignored", 'score = jnp.where(shared["candidates"], score, -jnp.inf)',
     "score = score"),
    ("latent rotated at its token, not its group start", "gpos = (meta.pos // r) * r",
     "gpos = meta.pos"),
    ("window drops the query's own latent", "band_ok = (o <= meta.idx[:, None])",
     "band_ok = (o < meta.idx[:, None])"),
    ("ring window one short", "ring_ok = (ring_pos >= 0) & (ring_pos >= lo)",
     "ring_ok = (ring_pos >= 0) & (ring_pos > lo)"),
    ("routed gate branch unclamped", "gate = jnp.minimum(gate, self.swiglu_limit)\n        act =",
     "gate = gate\n        act ="),
    ("routing weights carry the selection bias", "weights = jnp.take_along_axis(scores, idx, axis=-1)",
     "weights = jnp.take_along_axis(scores + self.gate_bias.value[None, :], idx, axis=-1)"),
    ("window latent not rounded to FP8", "kv = fp8_roundtrip(kv, 32)", "kv = kv"),
    ("compressed latent with an E8M0 scale", "lat = fp4_roundtrip(lat, 16, e4m3_scale=True)",
     "lat = fp4_roundtrip(lat, 16, e4m3_scale=False)"),
    ("compressor tail state ignored", "in_chunk = q >= meta.prefix[:, None]", "in_chunk = q >= 0"),
    ("YaRN off on compressed layers", "                int(cfg.original_seq_len),",
     "                0,"),
    ("reuse layer binds the first index source", 'idxs, ok = shared["topk"][self.role.index_source]',
     'idxs, ok = shared["topk"][min(shared["topk"])]'),
    ("Engram gate without the signed square root",
     "jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)), dot)", "dot"),
    ("Engram hash ignores the history", "tok = jnp.where(shift <= meta.idx, from_chunk, from_hist)",
     "tok = from_chunk"),
    ("Engram hash without start padding", "tokens.append(jnp.where(meta.pos - shift < 0, pad, tok))",
     "tokens.append(tok)"),
    ("index keys not rotated", "k = apply_rope(k, cos_c, sin_c, rd)", "k = k"),
    ("mHC mixes without the stream norm",
     "rs = jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + self.eps)", "rs = 1.0"),
    ("FP4 nibbles read high first", "codes = np.stack([p & 0x0F, p >> 4], -1)",
     "codes = np.stack([p >> 4, p & 0x0F], -1)"),
]


def run_one_mutant(index):
    """Child process: apply mutant `index` to this process's own tree copy and print the verdict
    of checks 4-6 as one JSON line."""
    name, old, new = MUTANTS[index]
    tree = os.environ["DSV41_TREE"]
    path = os.path.join(tree, MODEL_FILE)
    with open(path) as fp:
        src = fp.read()
    if src.count(old) != 1:
        print("MUTANT " + json.dumps({"error": f"target found {src.count(old)} times"}), flush=True)
        return 0
    with open(path, "w") as fp:
        fp.write(src.replace(old, new))
    root = fetch_published()
    try:
        mod, cfgmod, ref_model, ref_engram, reference, runner, seqs, ckpt, tok = run_all(tree, root, quiet=True)
        res = check_forward(runner, reference_runs(reference, seqs), seqs, quiet=True)
        out = {k: [bool(v[0]), v[1], v[2]] for k, v in res.items()}
    except Exception as exc:  # a mutant that crashes counts as caught, and says so
        out = {"raised": f"{type(exc).__name__}: {exc}"[:200]}
    print("MUTANT " + json.dumps(out), flush=True)
    return 0


def check_mutants(tree):
    """Each mutant runs in its own process on its own copy of the tree, MUTANT_WORKERS at once."""
    print("check 8: mutants of the patched model", flush=True)
    workdir = tempfile.mkdtemp(prefix="dsv41-mutants-")
    procs = []
    try:
        for i in range(len(MUTANTS)):
            copy = os.path.join(workdir, f"t{i}")
            shutil.copytree(os.path.join(tree, "python"), os.path.join(copy, "python"))
            env = dict(os.environ, DSV41_TREE=copy, DSV41_MUTANT=str(i))
            procs.append(None)
            procs[i] = (env, copy)
        running, results = {}, {}
        pending = list(range(len(MUTANTS)))
        while pending or running:
            while pending and len(running) < MUTANT_WORKERS:
                i = pending.pop(0)
                env, _ = procs[i]
                running[i] = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for i, p in list(running.items()):
                if p.poll() is not None:
                    out = p.stdout.read()
                    line = [l for l in out.splitlines() if l.startswith("MUTANT ")]
                    results[i] = json.loads(line[-1][7:]) if line else {"raised": out[-300:]}
                    del running[i]
            import time
            time.sleep(2)
        for i, (name, _, _) in enumerate(MUTANTS):
            r = results[i]
            if "error" in r:
                record(False, f"mutant '{name}': {r['error']}")
            elif "raised" in r:
                record(True, f"caught: {name} (raised {r['raised']})")
            else:
                caught = not all(v[0] for v in r.values())
                detail = ", ".join(f"{k} {v[1]:.0%} over, worst {v[2]:.1e}" for k, v in r.items())
                record(caught, f"caught: {name} ({detail})")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)



# --------------------------------------------------------------------------------------------
# Sharding: the tiny model on 8 simulated devices, and the full model on 32
# --------------------------------------------------------------------------------------------

SHARD_DEVICES = 8
# (tensor ways, expert-parallel ways). The mesh is (data 1, tensor N), the layout --tp-size N
# --dp-size 1 builds. The Engram rows split over all N devices, the experts over `ep`.
SHARD_LAYOUTS = ((4, 4), (8, 8), (8, 2))
# Each control rewrites one line of the Engram lookup. Both only matter once the rows split.
SHARD_CONTROLS = (
    ("Engram lookup keeps rows it doesn't hold", "rows = jnp.where(mask[..., None], rows, 0.0)",
     "rows = rows"),
    ("Engram lookup without the psum", "return jax.lax.psum(rows, axes)", "return rows"),
)


def run_sequence(runner, seqs, phases=("prefill", "split", "decode")):
    """The batch prefill, the split prefill and the decode steps, per phase a list of records."""
    out = {}
    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    out["prefill"] = runner.forward([(0, 0, seqs[0][: LENGTHS[0]]), (1, 0, seqs[1][: LENGTHS[1]])])
    if phases == ("prefill",):
        return out
    runner.state = runner.mod.init_state(runner.cfg, 3, MAX_SEQ, runner.dtype)
    a, b = seqs[0][: LENGTHS[0]], seqs[1][: LENGTHS[1]]
    out["split"] = runner.forward([(0, 0, a[: CUTS[0]]), (1, 0, b[: CUTS[1]])])
    out["split"] += runner.forward([(0, CUTS[0], a[CUTS[0]:]), (1, CUTS[1], b[CUTS[1]:])])
    out["decode"] = []
    for step in range(DECODE_STEPS):
        out["decode"] += runner.forward([(0, LENGTHS[0] + step, [seqs[0][LENGTHS[0] + step]]),
                                         (1, LENGTHS[1] + step, [seqs[1][LENGTHS[1] + step]])])
    return out


def compare_sequences(got, want):
    res = {}
    for phase in want:
        errs = [compare(phase, g, w) for g, w in zip(got[phase], want[phase])]
        tok = np.concatenate([e[0] for e in errs])
        res[phase] = verdict(tok)
    return res


def sharded_child() -> int:
    """Child process: the tiny model at one device against each sharded layout."""
    if jax.device_count() < SHARD_DEVICES:
        print(f"SHARDED {json.dumps({'error': f'{jax.device_count()} devices'})}", flush=True)
        return 1
    tree = os.environ["DSV41_TREE"]
    root = fetch_published()
    mod, cfgmod, ref_model, ref_engram, reference, base, seqs, ckpt, tok = run_all(tree, root, quiet=True)
    phases = tuple(os.environ.get("DSV41_SHARD_PHASES", "prefill,split,decode").split(","))
    want = run_sequence(base, seqs, phases)
    for tensor, ep in SHARD_LAYOUTS:
        runner = JaxRunner(mod, cfgmod, ckpt, tok, make_mesh(tensor), ep_size=ep)
        experts = runner.model.model.layers[0].ffn.experts
        table = runner.model.model.layers[1].engram.table.value
        shards = len({tuple((s.start, s.stop) for s in idx)
                      for idx in table.sharding.devices_indices_map(table.shape).values()})
        res = compare_sequences(run_sequence(runner, seqs, phases), want)
        out = {"tensor": tensor, "ep": ep, "epmoe": [experts.ep_size, experts.tp_size],
               "engram_row_shards": shards,
               "result": {k: [bool(v[0]), v[1], v[2]] for k, v in res.items()}}
        print(f"SHARDED {json.dumps(out)}", flush=True)
    return 0


def check_sharding(tree):
    """The tiny model sharded three ways against itself on one device, and two controls that
    break the Engram lookup's sharded path. Each runs in a child with 8 simulated devices."""
    print(f"check 10: sharded against unsharded on {SHARD_DEVICES} simulated devices", flush=True)
    workdir = tempfile.mkdtemp(prefix="dsv41-shard-")

    def launch(label, mutant=None):
        copy = os.path.join(workdir, label.replace(" ", "_"))
        shutil.copytree(os.path.join(tree, "python"), os.path.join(copy, "python"))
        if mutant:
            path = os.path.join(copy, MODEL_FILE)
            src = open(path).read()
            if src.count(mutant[0]) != 1:
                return f"target found {src.count(mutant[0])} times"
            open(path, "w").write(src.replace(mutant[0], mutant[1]))
        env = dict(os.environ, DSV41_TREE=copy, DSV41_CHILD="sharded")
        if mutant:  # the Engram lookup runs before the first block: prefill shows it
            env["DSV41_SHARD_PHASES"] = "prefill"
        flags = " ".join(f for f in env.get("XLA_FLAGS", "").split()
                         if not f.startswith("--xla_force_host_platform_device_count"))
        env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={SHARD_DEVICES}".strip()
        return subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def collect(proc):
        if isinstance(proc, str):
            return None, proc
        out, _ = proc.communicate()
        rows = [json.loads(l[8:]) for l in out.splitlines() if l.startswith("SHARDED ")]
        if proc.returncode or len(rows) != len(SHARD_LAYOUTS):
            return None, out[-2000:]
        return rows, None

    try:
        procs = [launch("plain")] + [launch(name, (old, new)) for name, old, new in SHARD_CONTROLS]
        rows, err = collect(procs[0])
        if rows is None:
            record(False, f"sharded child failed: {err}")
        else:
            for r in rows:
                ok = all(v[0] for v in r["result"].values())
                detail = ", ".join(f"{k} {v[1]:.0%} over, worst {v[2]:.1e}" for k, v in r["result"].items())
                record(ok and r["epmoe"] == [r["ep"], r["tensor"] // r["ep"]] and r["engram_row_shards"] == r["tensor"],
                       f"--tp-size {r['tensor']} --ep-size {r['ep']} (EPMoE {r['epmoe'][0]} x {r['epmoe'][1]}, "
                       f"Engram rows in {r['engram_row_shards']} shards) equals one device: {detail}")
        for (name, _, _), proc in zip(SHARD_CONTROLS, procs[1:]):
            rows, err = collect(proc)
            if rows is None:
                record(False, f"control '{name}': {err}")
                continue
            caught = all(not all(v[0] for v in r["result"].values()) for r in rows)
            worst = min(max(v[2] for v in r["result"].values()) for r in rows)
            record(caught, f"control: {name} differs from one device at every layout, smallest worst {worst:.1e}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


V5P64_CHIPS = 32
V5P_USABLE_GIB = 95.73
MEM_FRACTION_STATIC = 0.8
GIB = 2**30
# what the capture run serves: 8 requests (build_corpus prompts, 1,323 tokens joined), and the
# context the multi-host row passes
CAPTURE_REQS = 8
CAPTURE_CONTEXT = 4096


def shard_bytes(leaf):
    shape = list(leaf.shape)
    sharding = getattr(leaf, "sharding", None)
    spec = getattr(sharding, "spec", None) or ()
    sizes = dict(sharding.mesh.shape) if sharding is not None else {}
    for axis, names in enumerate(spec):
        if names is None:
            continue
        names = names if isinstance(names, tuple) else (names,)
        ways = int(np.prod([sizes[n] for n in names]))
        shape[axis] = -(-shape[axis] // ways)
    return int(np.prod(shape, dtype=np.int64)) * jnp.dtype(leaf.dtype).itemsize


def fullsize_child() -> int:
    """Child process: the published config under nnx.eval_shape on 32 devices, one chip's bytes."""
    from flax import nnx

    tree = os.environ["DSV41_TREE"]
    root = fetch_published()
    mod, cfgmod = import_patched(tree)
    if jax.device_count() != V5P64_CHIPS:
        print(f"      need {V5P64_CHIPS} devices, got {jax.device_count()}")
        return 1
    with open(os.path.join(root, "config.json")) as fp:
        raw = json.load(fp)
    cfg = cfgmod.DeepseekV41Config(**{k: v for k, v in raw.items() if k != "model_type"})
    cfg.ep_size = V5P64_CHIPS
    mesh = make_mesh(V5P64_CHIPS)
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: mod.DeepseekV41ForCausalLM(cfg, mesh, jnp.bfloat16))
    groups = {"routed experts": 0, "Engram tables": 0, "replicated": 0, "sharded other": 0}
    whole = {"Engram tables": 0}
    for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
        keys = [str(getattr(k, "key", getattr(k, "name", k))) for k in path]
        b = shard_bytes(leaf)
        full = int(np.prod(leaf.shape, dtype=np.int64)) * jnp.dtype(leaf.dtype).itemsize
        if "experts" in keys:
            groups["routed experts"] += b
        elif "engram" in keys and ("table" in keys or "table_scale" in keys):
            groups["Engram tables"] += b
            whole["Engram tables"] += full
        elif b == full:
            groups["replicated"] += b
        else:
            groups["sharded other"] += b
    experts = model.model.layers[0].ffn.experts
    weights = sum(groups.values())
    budget = V5P_USABLE_GIB * MEM_FRACTION_STATIC
    print(f"      one v5p-64: {V5P64_CHIPS} chips at {V5P_USABLE_GIB} GiB usable; "
          f"--mem-fraction-static {MEM_FRACTION_STATIC} reserves {budget:.2f} GiB a chip")
    print(f"      --tp-size 32 --dp-size 1 --ep-size 32 (EPMoE {experts.ep_size} x {experts.tp_size}, "
          f"{experts.experts_per_device} experts a chip), BF16 weights, FP8 Engram: "
          + ", ".join(f"{k} {v / GIB:.2f} GiB" for k, v in groups.items())
          + f" = {weights / GIB:.2f} GiB a chip")
    for ctx in (CAPTURE_CONTEXT, 65536, 1048576):
        shapes = mod.state_shapes(cfg, 2, ctx, jnp.bfloat16)
        per = sum(int(np.prod(sh)) * jnp.dtype(dt).itemsize for v in shapes.values() for sh, dt in v) / 2
        print(f"      state pool, one request slot at context {ctx:,}: {per / 2**20:.1f} MiB, replicated")
    slots = CAPTURE_REQS + 1
    shapes = mod.state_shapes(cfg, slots, CAPTURE_CONTEXT, jnp.bfloat16)
    pool = sum(int(np.prod(sh)) * jnp.dtype(dt).itemsize for v in shapes.values() for sh, dt in v)
    class _MC:  # the two fields paged_kv_token_cap reads
        context_len = CAPTURE_CONTEXT
    kv_tokens = mod.DeepseekV41ForCausalLM.paged_kv_token_cap(None, _MC, CAPTURE_REQS)
    kv = kv_tokens * mod.DeepseekV41ForCausalLM.paged_kv_layers * 2 * 512 * 2
    total = weights + pool + kv
    print(f"      capture serve, {CAPTURE_REQS} requests at --context-length {CAPTURE_CONTEXT}: "
          f"state pool {pool / GIB:.2f} GiB, paged KV pool {kv_tokens:,} tokens x "
          f"{mod.DeepseekV41ForCausalLM.paged_kv_layers} layer = {kv / GIB:.3f} GiB; total "
          f"{total / GIB:.2f} GiB of {budget:.2f} GiB, {budget - total / GIB:.2f} GiB left")
    replicated_engram = total - groups["Engram tables"] + whole["Engram tables"]
    print(f"      control: the same with the Engram tables replicated: "
          f"{replicated_engram / GIB:.2f} GiB, {'refused' if replicated_engram / GIB > budget else 'FITS'}")
    ok = total / GIB < budget and replicated_engram / GIB > budget and groups["Engram tables"] * V5P64_CHIPS >= whole["Engram tables"]
    print(f"FULLSIZE {json.dumps({'ok': ok, 'total_gib': total / GIB, 'budget_gib': budget})}", flush=True)
    return 0 if ok else 1


def check_full_size(tree):
    print("check 11: the full model on one v5p-64, from shapes alone", flush=True)
    env = dict(os.environ, DSV41_TREE=tree, DSV41_CHILD="fullsize")
    flags = " ".join(f for f in env.get("XLA_FLAGS", "").split()
                     if not f.startswith("--xla_force_host_platform_device_count"))
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={V5P64_CHIPS}".strip()
    done = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                          capture_output=True, text=True)
    for line in done.stdout.splitlines():
        if line.startswith("      "):
            print(line)
    res = [json.loads(l[9:]) for l in done.stdout.splitlines() if l.startswith("FULLSIZE ")]
    if not res:
        print((done.stdout + done.stderr)[-3000:])
    record(bool(res) and res[0]["ok"],
           f"V4.1-Flash fits one v5p-64 chip under --mem-fraction-static 0.8: "
           + (f"{res[0]['total_gib']:.2f} of {res[0]['budget_gib']:.2f} GiB" if res else "no report"))


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="dsv41-gate-")
    try:
        root = fetch_published()
        tree = os.environ.get("DSV41_TREE") or check_patch(workdir)
        if tree is None:
            return 1
        mod, cfgmod, ref_model, ref_engram, reference, runner, seqs, ckpt, tokenizer = run_all(tree, root)
        check_reference_handoff(reference, seqs)
        ref_runs = reference_runs(reference, seqs)
        print("checks 4-6: prefill, split prefill, decode against the reference", flush=True)
        check_forward(runner, ref_runs, seqs)
        check_capture_path(mod, runner, seqs)
        check_bf16(mod, cfgmod, ref_model, reference.args, tokenizer, reference.values, ckpt, seqs, ref_runs)
        check_engram_published(mod, ref_engram, root)
        # the engine imports from the patched tree this gate already put on sys.path
        cpu_engine._tree = tree
        if os.environ.get("DSV41_SKIP_ENGINE") != "1":
            check_engine(ckpt, reference, seqs)
        if os.environ.get("DSV41_SKIP_SHARDING") != "1":
            check_sharding(tree)
            check_full_size(tree)
        if os.environ.get("DSV41_SKIP_MUTANTS") != "1":
            check_mutants(tree)
    except Exception:
        traceback.print_exc()
        record(False, "the gate ran to the end")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    failed = [label for ok, label in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if os.environ.get("DSV41_MUTANT"):
        raise SystemExit(run_one_mutant(int(os.environ["DSV41_MUTANT"])))
    if os.environ.get("DSV41_CHILD") == "sharded":
        raise SystemExit(sharded_child())
    if os.environ.get("DSV41_CHILD") == "fullsize":
        raise SystemExit(fullsize_child())
    raise SystemExit(main())
