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

"""Correctness gate for the GLM-5.3-Flash model patch.

Runs on CPU, with 8 and 32 simulated devices in child processes. No TPU needed.

    python3 upstream/models/test_glm5_next_model.py

The reference is `transformers` 5.17's own `glm5_next` in float32. `transformers` also writes the
tiny checkpoint: a random `Glm5NextForConditionalGeneration` saved with `save_pretrained`, so the
tensor names and layouts are the ones its conversion mapping writes. The test then rewrites the
families the published checkpoint ships in FP8 as E4M3 with a `weight_scale_inv` per 128 x 128
tile, adds a multi-token-prediction layer, and keeps the vision tower, the way the published
checkpoint lays them out. The JAX side is the patched `sgl_jax/srt/models/glm5_next.py`, imported
from a tree that `scripts/cpu_engine.build_tree` builds and this file patches.

Checks:

1. The patch applies to `sglang-jax` at `SGL_COMMIT` (default `eb061d8`) with
   `sglang-jax-877.patch` and both steering patches, and every file it touches compiles. A
   corrupted copy has to be refused.
2. The checkpoint layout: the tiny tensor names per layer kind match the published index's
   layer of the same kind, FP8 wherever the published checkpoint has a `weight_scale_inv` and
   nowhere else, and the loader reads every text tensor and skips the vision tower and the MTP
   layer.
   `transformers` reading the FP8 copy gives the float copy's logits.
3. Prefill of two requests in one batch: the four copies entering every layer (every capture
   slot), the final hidden state and the logits, against `transformers` running each alone.
4. Split prefill: the same requests in two passes, cut inside a k-pool and across a KDA chunk.
5. Decode: three steps of both requests after the split, against `transformers` on the whole
   sequence.
6. KDA state: the chunked scan on a packed batch against a float64 token-by-token recurrence,
   and the pool's recurrent state and convolution window after the split prefill against
   `transformers`' cache after the whole prompt.
7. The capture hook with a subset of slots, and the engine's `LogitsProcessor`.
8. BF16: the port in BF16 against the float32 reference, within twice `transformers`' own BF16
   error, per token at the median.
9. The real `Engine` on CPU with capture on: greedy ids and every captured row.
10. The tiny model on 8 simulated devices at --tp-size 2/4/8 --ep-size 2/4/8/2 against itself on
    one device, and a control that drops the indexer's cross-shard sum.
11. The published config under `nnx.eval_shape` on 32 simulated devices: one chip's weights,
    the state pool and the capture serve against --mem-fraction-static 0.8 of a v5p chip, with
    a control that replicates the routed experts.
12. Mutants: each rewrites one line of the patched model or loader and has to fail check 3, 4
    or 5.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download. The published
`config.json` and index are cached under `GLM5NEXT_CACHE`, or
`~/.cache/glm5-next-published/<revision>`, and each has to match its pinned SHA-256.
`GLM5NEXT_SKIP_ENGINE=1`, `GLM5NEXT_SKIP_SHARDING=1` and `GLM5NEXT_SKIP_MUTANTS=1` skip checks 9,
10-11 and 12. `LOG_PAYLOADS=1` prints every batch the JAX model gets.
"""

from __future__ import annotations

import atexit
import glob
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
import time
import traceback
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

PATCH = os.path.join(HERE, "glm5-next-model.patch")
SRT = "python/sgl_jax/srt"
MODEL_FILE = f"{SRT}/models/glm5_next.py"
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS") == "1"

HUB = "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve"
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
PUBLISHED = {
    "config.json": "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    "model.safetensors.index.json": "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",
}

# The tiny model. Layers 3 and 7 are MLA, the rest KDA, the published 1-in-4 plan; layers 0-1
# dense, the rest routed. 16 indexer heads keep an all-zero pool score (every head's rectified
# dot product 0, a tie torch.topk breaks in no set order) at 1 in 65,536.
TINY = dict(
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    routed_scaling_factor=2.5,
    kv_lora_rank=32,
    q_lora_rank=48,
    qk_nope_head_dim=16,
    v_head_dim=16,
    qk_rope_head_dim=0,
    index_topk=8,
    index_head_dim=16,
    index_n_heads=16,
    index_kpool=4,
    linear_attn_config={"num_heads": 8, "head_dim": 16, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    mlp_layer_types=["dense", "dense"] + ["sparse"] * 6,
    swiglu_limit=2.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    rms_norm_eps=1e-5,
    max_position_embeddings=256,
    pad_token_id=0,
)
VISION = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=32, out_hidden_size=64,
              projection_intermediate_size=32)
MAX_SEQ = 160
LENGTHS = (90, 45)
CUTS = (70, 21)  # request A cut past the first 64-token KDA chunk, B inside a 4-token pool
DECODE_STEPS = 3
TOKEN_BUCKET = 64
PAD_TOKEN = 5  # a real id, so a padding row that leaked into the state would change the result
BLOCK = (128, 128)
# Per-token error: the max abs difference over a token's elements, over the reference tensor's
# RMS, worst over every compared tensor. Float32 on both sides. transformers' own float32 forward
# sits up to 2.9e-05 from its float64 forward on this model (the logits, 2026-09-26), and the port
# sits as far from float64 as it does; check 3 prints both.
TOL = 1e-4
# A query whose pool scores sit within float32 rounding of each other at the top-k boundary can
# keep another pool on one side; such a token and the queries after it that read its layer move
# by more. The check passes when at most FLIP_FRAC of the tokens exceed TOL.
FLIP_FRAC = 0.05
MUTANT_WORKERS = int(os.environ.get("GLM5NEXT_MUTANT_WORKERS", "8"))

RESULTS = []


def record(ok, label):
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)
    return ok


# --------------------------------------------------------------------------------------------
# Published files
# --------------------------------------------------------------------------------------------


def fetch_published():
    root = os.environ.get("GLM5NEXT_CACHE") or os.path.expanduser(
        f"~/.cache/glm5-next-published/{REVISION}")
    os.makedirs(root, exist_ok=True)
    for name, pin in PUBLISHED.items():
        path = os.path.join(root, name)
        if not os.path.exists(path):
            url = f"{HUB}/{REVISION}/{name}"
            print(f"  fetching {url}", flush=True)
            with urllib.request.urlopen(url) as resp, open(path + ".tmp", "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(path + ".tmp", path)
        with open(path, "rb") as fp:
            digest = hashlib.sha256(fp.read()).hexdigest()
        if digest != pin:
            raise SystemExit(f"{name} has SHA-256 {digest}, pinned {pin}")
    return root


# --------------------------------------------------------------------------------------------
# The tiny checkpoint
# --------------------------------------------------------------------------------------------


def hf_config(vocab):
    from transformers import Glm5NextConfig

    return Glm5NextConfig(text_config=dict(TINY, vocab_size=vocab), vision_config=VISION)


def randomize(model, seed=0):
    """Random weights at scales that make every branch matter: norms near 1, projections at
    1/sqrt(fan_in), the decay and mix parameters spread wide enough that the gates and clamps
    move."""
    import torch

    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        shape = p.shape
        n = torch.randn(shape, generator=g)
        if "visual" in name:
            v = 0.02 * n
        elif name.endswith(("norm.weight", "layernorm.weight")) or name.endswith("k_norm.weight"):
            v = 1.0 + 0.1 * n
        elif name.endswith("k_norm.bias"):
            v = 0.1 * n
        elif name.endswith("hc.fn"):
            v = 0.05 * n
        elif name.endswith("hc.scale"):
            v = 0.5 + 0.2 * n
        elif name.endswith("hc.base"):
            v = 0.3 * n
        elif name.endswith("A_log"):
            v = torch.log(torch.rand(shape, generator=g) * 3 + 0.5)
        elif name.endswith("dt_bias"):
            v = 0.5 * n
        elif name.endswith("compress_ape") or name.endswith("compress_gate"):
            v = 0.5 * n if name.endswith("ape") else n / math.sqrt(shape[-1])
        elif name.endswith("embed_tokens.weight"):
            v = n
        elif p.ndim >= 2:
            v = n / math.sqrt(shape[-1])
        else:
            v = 0.3 * n
        with torch.no_grad():
            p.copy_(v)
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            with torch.no_grad():
                b.copy_(0.3 * torch.randn(b.shape, generator=g))


def fp8_families(index_path):
    """Per layer kind, the relative names the published checkpoint ships in FP8: those with a
    `weight_scale_inv`."""
    with open(index_path) as fp:
        keys = set(json.load(fp)["weight_map"])
    out = {}
    for kind, layer in (("kda_dense", 0), ("mla_moe", 3), ("kda_moe", 4)):
        pre = f"model.language_model.layers.{layer}."
        out[kind] = sorted(k[len(pre):-len(".weight_scale_inv")] for k in keys
                           if k.startswith(pre) and k.endswith(".weight_scale_inv")
                           and ".experts." not in k)
    out["experts"] = any(".mlp.experts.0.gate_proj.weight_scale_inv" in k for k in keys)
    return out


def layer_kind(cfg_text, i):
    attn = "mla" if cfg_text["layer_types"][i] == "deepseek_sparse_attention" else "kda"
    mlp = "moe" if cfg_text["mlp_layer_types"][i] == "sparse" else "dense"
    return f"{attn}_{mlp}"


def quantize_e4m3(w, block=BLOCK):
    """float32 [out, in] -> (E4M3 as float8, FP32 power-of-two scale_inv per tile). A power-of-two
    scale makes the dequantized weight exact in float32 and BF16 alike."""
    import ml_dtypes

    out, inn = w.shape
    bo, bi = -(-out // block[0]), -(-inn // block[1])
    pad = np.zeros((bo * block[0], bi * block[1]), np.float32)
    pad[:out, :inn] = w
    tiles = pad.reshape(bo, block[0], bi, block[1])
    amax = np.maximum(np.abs(tiles).max(axis=(1, 3)), 1e-4)
    s = np.exp2(np.ceil(np.log2(amax / 448.0))).astype(np.float32)
    q = (tiles / s[:, None, :, None]).reshape(bo * block[0], bi * block[1])[:out, :inn]
    return np.clip(q, -448, 448).astype(ml_dtypes.float8_e4m3fn), s


def dequant_e4m3(q, s, block=BLOCK):
    out, inn = q.shape
    full = np.repeat(np.repeat(s, block[0], 0), block[1], 1)[:out, :inn]
    return q.astype(np.float32) * full


def read_all(path):
    from safetensors.numpy import load_file

    tensors = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        tensors.update(load_file(f))
    return tensors


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
    bits = np.asarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def build_checkpoints(workdir, tokenizer_dir, index_path, seed=0):
    """Three directories: `hf` (what `save_pretrained` wrote, FP8 families rounded to their
    E4M3 values), `pub` (the published layout: FP8 families as E4M3 with `weight_scale_inv`, an
    MTP layer, the vision tower, the published quantization block) and `pub16` (the same with
    every float tensor in BF16)."""
    import torch
    from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    cfg = hf_config(len(tok))
    torch.manual_seed(seed)
    model = Glm5NextForConditionalGeneration(cfg).float().eval()
    randomize(model, seed)
    raw = os.path.join(workdir, "raw")
    model.save_pretrained(raw)
    tensors = {k: np.asarray(v, np.float32) for k, v in read_all(raw).items()}
    with open(os.path.join(raw, "config.json")) as fp:
        config = json.load(fp)
    text = config["text_config"]
    fams = fp8_families(index_path)
    n_layers = text["num_hidden_layers"]
    # The MTP layer in the published names, at layer n_layers, like the published layer 45: an
    # MLA and MoE layer without mHC parameters, plus its own norms and projection.
    src = max(i for i in range(n_layers) if layer_kind(text, i) == "mla_moe")
    mtp_src = f"model.language_model.layers.{src}."
    mtp = f"model.language_model.layers.{n_layers}."
    rng = np.random.default_rng(seed + 1)
    extra = {}
    for name, arr in tensors.items():
        if name.startswith(mtp_src) and ".hc_" not in name:
            extra[mtp + name[len(mtp_src):]] = arr
    d = text["hidden_size"]
    for rel, shape in (("enorm.weight", (d,)), ("hnorm.weight", (d,)), ("eh_proj.weight", (d, 2 * d)),
                       ("shared_head.norm.weight", (d,))):
        extra[mtp + rel] = rng.standard_normal(shape).astype(np.float32)
    everything = dict(tensors, **extra)
    fp8_names = set()
    for i in range(n_layers + 1):
        pre = f"model.language_model.layers.{i}."
        kind = layer_kind(text, i) if i < n_layers else "mla_moe"
        for rel in fams.get(kind, []):
            fp8_names.add(pre + rel + ".weight")
        if kind.endswith("moe") and fams["experts"]:
            for name in everything:
                if name.startswith(pre + "mlp.experts."):
                    fp8_names.add(name)
    missing = sorted(n for n in fp8_names if n not in everything)
    if missing:
        raise AssertionError(f"FP8 families with no tiny tensor: {missing[:5]}")
    hf_tensors, pub, pub16 = {}, {}, {}
    for name, arr in everything.items():
        if name in fp8_names:
            q, s = quantize_e4m3(arr)
            deq = dequant_e4m3(q, s)
            if name not in extra:
                hf_tensors[name] = deq
            pub[name] = ("F8_E4M3", q.view(np.uint8))
            pub[name[: -len("weight")] + "weight_scale_inv"] = ("F32", s)
            pub16[name] = pub[name]
            pub16[name[: -len("weight")] + "weight_scale_inv"] = ("F32", s)
        else:
            if name not in extra:
                hf_tensors[name] = arr
            pub[name] = ("F32", arr)
            pub16[name] = ("BF16", to_bf16_bits(arr))
    dirs = {}
    for label, contents in (("hf", None), ("pub", pub), ("pub16", pub16)):
        out = os.path.join(workdir, label)
        os.makedirs(out, exist_ok=True)
        if contents is None:
            from safetensors.numpy import save_file

            save_file({k: np.ascontiguousarray(v) for k, v in hf_tensors.items()},
                      os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
            conf = dict(config)
        else:
            shard = "model-00001-of-00001.safetensors"
            write_safetensors(os.path.join(out, shard), contents)
            with open(os.path.join(out, "model.safetensors.index.json"), "w") as fp:
                json.dump({"metadata": {}, "weight_map": {k: shard for k in contents}}, fp)
            conf = dict(config)
            conf["quantization_config"] = {
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": list(BLOCK),
                "modules_to_not_convert": ["lm_head", "model.embed_tokens", "model.visual"],
            }
        with open(os.path.join(out, "config.json"), "w") as fp:
            json.dump(conf, fp, indent=1)
        for f in os.listdir(tokenizer_dir):
            shutil.copy(os.path.join(tokenizer_dir, f), out)
        dirs[label] = out
    return dirs, tok, pub, fp8_names


# --------------------------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------------------------


class Reference:
    """`transformers`' Glm5NextForConditionalGeneration on the float copy of the checkpoint."""

    def __init__(self, path, dtype="float32"):
        import torch
        from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

        self.torch = torch
        tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        cfg = Glm5NextConfig.from_pretrained(path)
        self.model = Glm5NextForConditionalGeneration.from_pretrained(path, config=cfg, dtype=tdt).eval()

    def run(self, ids):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        hs = out.hidden_states
        rec = {("stream", i): h[0].double().numpy() for i, h in enumerate(hs[:-1])}
        rec[("final", -1)] = hs[-1][0].double().numpy()
        rec[("logits", -1)] = out.logits[0].double().numpy()
        return rec

    def run64(self, ids):
        """The same weights in float64, experts on the eager loop, which takes float64."""
        import copy

        torch = self.torch
        model = copy.deepcopy(self.model).double()
        for m in model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None:
                cfg._experts_implementation = "eager"
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        hs = out.hidden_states
        rec = {("stream", i): h[0].numpy() for i, h in enumerate(hs[:-1])}
        rec[("final", -1)] = hs[-1][0].numpy()
        rec[("logits", -1)] = out.logits[0].numpy()
        return rec

    def cache_after(self, ids):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]), use_cache=True)
        return out.past_key_values


# --------------------------------------------------------------------------------------------
# JAX side
# --------------------------------------------------------------------------------------------


def import_patched(tree):
    sys.path.insert(0, os.path.join(tree, "python"))
    for name in list(sys.modules):
        if name.startswith(("sgl_jax.srt.models.glm5_next", "sgl_jax.srt.configs.glm5_next")):
            del sys.modules[name]
    mod = importlib.import_module("sgl_jax.srt.models.glm5_next")
    cfgmod = importlib.import_module("sgl_jax.srt.configs.glm5_next")
    return mod, cfgmod


def make_mesh(tensor=1):
    from jax.sharding import AxisType, Mesh

    devices = np.array(jax.devices()[:tensor]).reshape(1, tensor)
    return Mesh(devices, ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def serving_config(cfgmod, path):
    with open(os.path.join(path, "config.json")) as fp:
        raw = json.load(fp)
    return cfgmod.Glm5NextServingConfig(**{k: v for k, v in raw.items() if k != "model_type"})


class JaxRunner:
    def __init__(self, mod, cfgmod, ckpt, mesh, dtype=jnp.float32, ep_size=1):
        self.mod, self.mesh, self.dtype = mod, mesh, dtype
        self.cfg = serving_config(cfgmod, ckpt)
        self.cfg.ep_size = ep_size  # the runner writes --ep-size onto hf_config the same way
        with jax.set_mesh(mesh):
            self.model = mod.Glm5NextForConditionalGeneration(self.cfg, mesh, dtype=dtype)
            self.used = mod.load_checkpoint(self.model, ckpt)
        self.model.model.return_streams = True
        self.model.set_layers_to_capture(range(int(self.cfg.num_hidden_layers)))
        self.reset()

    def reset(self):
        self.state = self.mod.init_state(self.cfg, 3, MAX_SEQ, self.dtype)

    def forward(self, chunks):
        """chunks: [(req_slot, prefix_len, ids)]. Returns per-request dicts like the reference's."""
        ids = np.concatenate([np.asarray(c[2], np.int32) for c in chunks])
        lens = np.array([len(c[2]) for c in chunks], np.int32)
        prefix = np.array([c[1] for c in chunks], np.int32)
        slots = np.array([c[0] for c in chunks], np.int32)
        if LOG_PAYLOADS:
            print(f"    payload: ids={ids.tolist()} lens={lens.tolist()} prefix={prefix.tolist()} "
                  f"req_pool_indices={slots.tolist()}", flush=True)
        # The engine pads every batch to a token bucket; the megablox GMM wants the routed row
        # count to divide by its 128 tile. Padding tokens have to leave the state alone.
        padded = -(-len(ids) // TOKEN_BUCKET) * TOKEN_BUCKET
        ids = np.concatenate([ids, np.full(padded - len(ids), PAD_TOKEN, np.int32)])
        with jax.set_mesh(self.mesh):
            meta = self.mod.token_meta(len(ids), slots, prefix, lens)
            hidden, aux, self.state, streams = self.model.model(jnp.asarray(ids), meta, self.state)
            logits = hidden.astype(jnp.float32) @ self.model.lm_head.embedding.value.astype(jnp.float32).T
        out, start = [], 0
        for n in lens:
            rows = slice(start, start + n)
            rec = {("stream", i): np.asarray(s[rows].astype(jnp.float32), np.float64)
                   for i, s in enumerate(streams)}
            rec[("final", -1)] = np.asarray(hidden[rows].astype(jnp.float32), np.float64)
            rec[("logits", -1)] = np.asarray(logits[rows], np.float64)
            rec[("aux", -1)] = [np.asarray(a[rows].astype(jnp.float32), np.float64) for a in aux]
            out.append(rec)
            start += n
        return out


def compare(jax_rec, ref_rec, rows=None):
    """Per-token error over every recorded tensor, and the tensor that set the worst one. Each
    capture slot is compared against the reference's stream entering the same layer, reshaped
    the way check_capture.py flattens it."""
    tok_err, where, worst = None, None, -1.0

    def fold(err, key):
        nonlocal tok_err, where, worst
        tok_err = err if tok_err is None else np.maximum(tok_err, err)
        if err.max() > worst:
            worst, where = float(err.max()), key

    for key, ref in ref_rec.items():
        if key not in jax_rec or key == ("aux", -1):
            continue
        got = jax_rec[key]
        ref = ref if rows is None else ref[rows]
        if got.shape != ref.shape:
            raise AssertionError(f"{key}: shape {got.shape} vs reference {ref.shape}")
        scale = np.sqrt(np.mean(ref**2)) + 1e-30
        err = np.max(np.abs(got - ref).reshape(ref.shape[0], -1), axis=1) / scale
        fold(np.where(np.isfinite(got.reshape(ref.shape[0], -1)).all(1), err, np.inf), key)
    for i, cap in enumerate(jax_rec.get(("aux", -1), [])):
        ref = ref_rec[("stream", i)] if rows is None else ref_rec[("stream", i)][rows]
        ref = ref.reshape(ref.shape[0], -1)
        err = np.max(np.abs(cap - ref), axis=1) / (np.sqrt(np.mean(ref**2)) + 1e-30)
        fold(np.where(np.isfinite(cap).all(1), err, np.inf), ("capture slot", i))
    return tok_err, where


def verdict(tok_err):
    frac = float(np.mean(tok_err > TOL))
    worst = float(np.max(tok_err))
    return bool(np.isfinite(tok_err).all() and frac <= FLIP_FRAC), frac, worst


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
           f"glm5-next-model.patch applies to {cpu_engine.SGL_COMMIT} with the capture and steering patches")
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
        done = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(tree, path)],
                              capture_output=True, text=True)
        record(done.returncode == 0, f"{path} compiles")
    return tree


def setup(tree, root, quiet=False):
    """The tiny checkpoints, the reference and the port on one device."""
    mod, cfgmod = import_patched(tree)
    work = tempfile.mkdtemp(prefix="glm5next-ckpt-")
    atexit.register(shutil.rmtree, work, True)
    tokdir = os.path.join(work, "tok")
    cpu_engine.write_tokenizer(tokdir)
    dirs, tok, pub, fp8_names = build_checkpoints(work, tokdir, os.path.join(root, "model.safetensors.index.json"))
    reference = Reference(dirs["hf"])
    runner = JaxRunner(mod, cfgmod, dirs["pub"], make_mesh())
    rng = np.random.default_rng(7)
    seqs = [rng.integers(3, len(tok), n + DECODE_STEPS).tolist() for n in LENGTHS]
    return dict(mod=mod, cfgmod=cfgmod, dirs=dirs, tok=tok, pub=pub, fp8=fp8_names,
                reference=reference, runner=runner, seqs=seqs)


def check_layout(env, root):
    print("check 2: checkpoint layout and loader", flush=True)
    with open(os.path.join(root, "model.safetensors.index.json")) as fp:
        published = json.load(fp)["weight_map"]

    def family(names, layer):
        pre = f"model.language_model.layers.{layer}."
        out = set()
        for n in names:
            if n.startswith(pre):
                parts = n[len(pre):].split(".")
                if parts[:2] == ["mlp", "experts"]:
                    parts[2] = "N"
                out.add(".".join(parts))
        return out

    pub = env["pub"]
    ok = True
    # tiny layer -> published layer of the same kind: KDA+dense, KDA+MoE, MLA+MoE, MTP
    for tiny_layer, pub_layer in ((0, 0), (2, 4), (3, 3), (7, 43), (8, 45)):
        mine, theirs = family(pub, tiny_layer), family(published, pub_layer)
        if mine != theirs:
            ok = False
            print(f"      layer {tiny_layer} vs published {pub_layer}: only tiny {sorted(mine - theirs)[:6]}, "
                  f"only published {sorted(theirs - mine)[:6]}")
    record(ok, "tiny tensor names per layer kind equal the published layer of the same kind (0, 4, 3, 43, "
               "and the MTP layer 45), FP8 scales included")
    tops = {n for n in pub if not n.startswith(("model.language_model.layers.", "model.visual."))}
    ptops = {n for n in published if not n.startswith(("model.language_model.layers.", "model.visual."))}
    record(tops == ptops, f"root tensors {sorted(tops)} equal the published {sorted(ptops)}")
    runner = env["runner"]
    n = int(runner.cfg.num_hidden_layers)
    text = {k for k in pub if not k.startswith("model.visual.") and not k.startswith(f"model.language_model.layers.{n}.")}
    record(set(runner.used) == text,
           f"the loader read all {len(text)} text tensors and skipped the vision tower and the MTP layer "
           f"({len(pub) - len(text)} tensors)")
    record(all(pub[k][0] == "F8_E4M3" for k in env["fp8"]) and len(env["fp8"]) > 0,
           f"{len(env['fp8'])} tensors ship E4M3 with a weight_scale_inv, the families the published index scales")
    w = np.random.default_rng(3).standard_normal((200, 300)).astype(np.float32)
    q, s = quantize_e4m3(w)
    got = env["mod"].dequant_fp8_block(q, s, BLOCK)
    record(np.array_equal(got, dequant_e4m3(q, s)), "FP8 dequant with a partial last tile in both axes")
    record(not np.array_equal(env["mod"].dequant_fp8_block(q, s.T.copy().T * 2, BLOCK), got),
           "control: a doubled scale dequantizes differently")
    # transformers' own FP8 read of the published copy, the path check_capture.py's reference takes
    try:
        import check_capture

        check_capture.patch_fp8_ceil_blocks()
        fp8_ref = Reference(env["dirs"]["pub"])
        ids = env["seqs"][1][:LENGTHS[1]]
        a = fp8_ref.run(ids)[("logits", -1)]
        b = env["reference"].run(ids)[("logits", -1)]
        err = float(np.max(np.abs(a - b)) / np.sqrt(np.mean(b**2)))
        record(err < 1e-5, f"transformers reads the FP8 published copy to the float copy's logits, err {err:.1e}")
    except Exception as exc:
        record(False, f"transformers reads the FP8 published copy: {type(exc).__name__}: {exc}"[:300])


def reference_runs(reference, seqs):
    """Each request alone: the prompt, then the prompt plus each decode token."""
    runs = []
    for seq, n in zip(seqs, LENGTHS):
        steps = [reference.run(seq[:n])]
        for s in range(DECODE_STEPS):
            full = reference.run(seq[: n + s + 1])
            steps.append({k: v[-1:] for k, v in full.items()})
        runs.append(steps)
    return runs


def run_sequence(runner, seqs):
    """The batch prefill, the split prefill and the decode steps, per phase a list of records."""
    out = {}
    runner.reset()
    out["prefill"] = runner.forward([(0, 0, seqs[0][: LENGTHS[0]]), (1, 0, seqs[1][: LENGTHS[1]])])
    runner.reset()
    a, b = seqs[0][: LENGTHS[0]], seqs[1][: LENGTHS[1]]
    first = runner.forward([(0, 0, a[: CUTS[0]]), (1, 0, b[: CUTS[1]])])
    second = runner.forward([(0, CUTS[0], a[CUTS[0]:]), (1, CUTS[1], b[CUTS[1]:])])
    out["split"] = first + second
    out["split_state"] = {k: [np.asarray(x) for x in v] for k, v in runner.state.items()}
    out["decode"] = []
    for step in range(DECODE_STEPS):
        out["decode"] += runner.forward([(0, LENGTHS[0] + step, [seqs[0][LENGTHS[0] + step]]),
                                         (1, LENGTHS[1] + step, [seqs[1][LENGTHS[1] + step]])])
    return out


def phase_errors(got, ref_runs):
    """Per phase, the per-token errors and where the worst sat."""
    res = {}
    errs = [compare(g, r[0]) for g, r in zip(got["prefill"], ref_runs)]
    res["prefill"] = errs
    errs = []
    for i in range(2):
        errs.append(compare(got["split"][i], ref_runs[i][0], rows=slice(0, CUTS[i])))
        errs.append(compare(got["split"][2 + i], ref_runs[i][0], rows=slice(CUTS[i], LENGTHS[i])))
    res["split"] = errs
    errs = []
    for s in range(DECODE_STEPS):
        for i in range(2):
            errs.append(compare(got["decode"][2 * s + i], ref_runs[i][1 + s]))
    res["decode"] = errs
    return res


def summarize(res):
    out = {}
    for phase, errs in res.items():
        tok = np.concatenate([e[0] for e in errs])
        out[phase] = verdict(tok) + (str(max(errs, key=lambda e: e[0].max())[1]),)
    return out


def check_forward(env, ref_runs):
    print("checks 3-5: prefill, split prefill, decode against transformers", flush=True)
    got = run_sequence(env["runner"], env["seqs"])
    summary = summarize(phase_errors(got, ref_runs))
    labels = {
        "prefill": f"prefill of {LENGTHS[0]} and {LENGTHS[1]} tokens in one batch: every layer's four "
                   "copies, every capture slot, the final norm, the logits",
        "split": f"prefill split at {CUTS[0]} (past the first KDA chunk) and {CUTS[1]} (inside a pool)",
        "decode": f"{DECODE_STEPS} decode steps x 2 requests after the split prefill",
    }
    for phase, (ok, frac, worst, where) in summary.items():
        record(ok, f"{labels[phase]}: {frac:.1%} of tokens above {TOL:.0e}, worst {worst:.2e} at {where}")
    # both float32 forwards against transformers in float64: the port has to sit no further from
    # it than transformers' own float32 forward does, within a factor of 2
    ids = env["seqs"][1][: LENGTHS[1]]
    r64 = env["reference"].run64(ids)
    port = got["prefill"][1]
    port_err = max(compare({k: v for k, v in port.items() if k != ("aux", -1)}, r64)[0])
    hf_err = max(compare(ref_runs[1][0], r64)[0])
    record(port_err <= 2 * hf_err + 1e-7,
           f"against transformers in float64: the port's worst per-token error {port_err:.2e}, "
           f"transformers float32's own {hf_err:.2e}")
    return got


def naive_kda(q, k, v, g, beta, s0):
    """Token-by-token KDA in float64: decay, predict, delta update, read."""
    s = s0.astype(np.float64).copy()
    out = np.zeros(v.shape, np.float64)
    for t in range(q.shape[0]):
        s = s * np.exp(g[t])[..., None]
        pred = np.einsum("hk,hkv->hv", k[t], s)
        s = s + np.einsum("hk,hv->hkv", k[t] * beta[t][:, None], v[t] - pred)
        out[t] = np.einsum("hk,hkv->hv", q[t], s)
    return out, s


def check_kda(env, got):
    print("check 6: KDA state", flush=True)
    mod = env["mod"]
    rng = np.random.default_rng(5)
    h, dk = 3, 8
    lens = [70, 5, 130]
    total = sum(lens)
    T = 256
    q = rng.standard_normal((T, h, dk))
    k = rng.standard_normal((T, h, dk))
    q /= np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k /= np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q *= dk**-0.5
    v = rng.standard_normal((T, h, dk))
    g = -5.0 / (1 + np.exp(-rng.standard_normal((T, h, dk))))
    beta = 1 / (1 + np.exp(-rng.standard_normal((T, h))))
    init = rng.standard_normal((3, h, dk, dk)) * 0.3
    meta = mod.token_meta(T, np.array([0, 1, 2]), np.array([4, 0, 9]), np.array(lens))
    out, finals = mod.kda_chunk_scan(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, g, beta, init)), meta)
    out, finals = np.asarray(out), np.asarray(finals)
    import torch
    from transformers.models.glm5_next import modeling_glm5_next as hf

    worst = floor = 0.0
    start = 0
    for r, n in enumerate(lens):
        sl = slice(start, start + n)
        o_ref, s_ref = naive_kda(q[sl], k[sl], v[sl], g[sl], beta[sl], init[r])
        worst = max(worst, float(np.max(np.abs(out[sl] - o_ref)) / np.sqrt(np.mean(o_ref**2))),
                    float(np.max(np.abs(finals[r] - s_ref)) / np.sqrt(np.mean(s_ref**2))))
        # transformers' own chunked kernel in float32 on the same request, its scale inside
        t = lambda x: torch.tensor(x[None], dtype=torch.float32)  # noqa: E731
        o_hf, s_hf = hf.chunk_kimi_delta_attention(
            t(q[sl] / dk**-0.5), t(k[sl]), t(v[sl]), t(g[sl]), t(beta[sl]),
            initial_state=torch.tensor(init[r][None], dtype=torch.float32), output_final_state=True)
        floor = max(floor, float(np.max(np.abs(o_hf[0].double().numpy() - o_ref)) / np.sqrt(np.mean(o_ref**2))),
                    float(np.max(np.abs(s_hf[0].double().numpy() - s_ref)) / np.sqrt(np.mean(s_ref**2))))
        start += n
    tail = float(np.max(np.abs(out[total:])))
    record(worst <= 2 * floor + 1e-7 and tail == 0.0,
           f"chunked scan on 70, 5 and 130 packed tokens (3 chunk grids, carried initial states) against "
           f"a float64 token recurrence: worst relative {worst:.1e}, transformers' float32 chunk kernel "
           f"{floor:.1e}; padding rows {tail}")
    wrong = mod.kda_chunk_scan(*(jnp.asarray(x, jnp.float32) for x in (q, k, v, g, beta, init * 0)), meta)[0]
    moved = float(np.max(np.abs(np.asarray(wrong)[:70] - out[:70])))
    record(moved > 1e-3, f"control: dropping the carried state moves the first request by {moved:.2e}")

    # the pool after the split prefill against transformers' cache after the whole prompt
    runner, reference = env["runner"], env["reference"]
    kda_ids = mod.kda_layer_ids(runner.cfg)
    worst_s, worst_c = 0.0, 0.0
    for r in range(2):
        cache = reference.cache_after(env["seqs"][r][: LENGTHS[r]])
        for slot, layer in enumerate(kda_ids):
            lc = cache.layers[layer]
            # transformers keys both by stream index; one stream here, batch 1
            ref_state = lc.recurrent_states[0][0].double().numpy()  # [H, K, V]
            conv = lc.conv_states[0][0].double().numpy()  # [channels, kernel], newest last
            mine = got["split_state"]["kda_state"][slot][r + 1]
            mine_c = got["split_state"]["kda_conv"][slot][r + 1]  # [kernel - 1, channels]
            worst_s = max(worst_s, float(np.max(np.abs(mine - ref_state)) / np.sqrt(np.mean(ref_state**2))))
            worst_c = max(worst_c, float(np.max(np.abs(mine_c.T - conv[:, -mine_c.shape[0]:]))
                                         / np.sqrt(np.mean(conv**2))))
    record(worst_s < 1e-4, f"recurrent state of all {len(kda_ids)} KDA layers after the split prefill against "
                           f"transformers' cache after the whole prompt: worst relative {worst_s:.1e}")
    record(worst_c < TOL, f"convolution windows against transformers' conv cache: worst relative {worst_c:.1e}")


def check_capture_path(env):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

    print("check 7: capture hook and logits processor", flush=True)
    runner, seqs = env["runner"], env["seqs"]
    n = LENGTHS[1]
    runner.reset()
    full = runner.forward([(0, 0, seqs[1][:n])])[0]
    runner.model.set_layers_to_capture([1, 3, 6])
    runner.reset()
    part = runner.forward([(0, 0, seqs[1][:n])])[0]
    runner.model.set_layers_to_capture(range(int(runner.cfg.num_hidden_layers)))
    aux = part[("aux", -1)]
    want = [full[("stream", l)].reshape(n, -1) for l in (1, 3, 6)]
    record(len(aux) == 3 and all(np.array_equal(a, w) for a, w in zip(aux, want)),
           f"layers_to_capture [1, 3, 6] returns those streams, in order, {aux[0].shape[1]} wide "
           f"(hc_mult x hidden)")
    record(not np.array_equal(want[0], want[1]), "control: slots 1 and 3 differ")
    hidden = jnp.asarray(full[("final", -1)], jnp.float32)
    slots = [jnp.asarray(a, jnp.float32) for a in aux]
    with jax.set_mesh(runner.mesh):
        out = runner.model.logits_processor(
            hidden, runner.model.lm_head,
            LogitsMetadata(forward_mode=ForwardMode.DECODE, capture_hidden_mode=CaptureHiddenMode.FULL),
            aux_hidden_states=slots)
    got = np.asarray(out.next_token_logits, np.float64)
    err = np.max(np.abs(got - full[("logits", -1)])) / np.sqrt(np.mean(full[("logits", -1)] ** 2))
    record(err < 1e-5, f"LogitsProcessor gives the logits the gate compares, err {err:.1e}")
    stored = np.asarray(out.hidden_states, np.float64)
    cat = np.concatenate(want, axis=-1)
    record(stored.shape == cat.shape and np.allclose(stored, cat, rtol=0, atol=1e-6),
           f"LogitsProcessor stores the three slots side by side, {stored.shape}")


def check_bf16(env, ref_runs):
    print("check 8: BF16 against transformers' own BF16 floor", flush=True)
    ref16 = Reference(env["dirs"]["hf"], "bfloat16")
    runner16 = JaxRunner(env["mod"], env["cfgmod"], env["dirs"]["pub16"], make_mesh(), dtype=jnp.bfloat16)
    got = runner16.forward([(0, 0, env["seqs"][0][: LENGTHS[0]]), (1, 0, env["seqs"][1][: LENGTHS[1]])])
    for i in range(2):
        floor = ref16.run(env["seqs"][i][: LENGTHS[i]])
        port_err, _ = compare(got[i], ref_runs[i][0])
        floor_err, _ = compare(floor, ref_runs[i][0])
        pm, fm = float(np.median(port_err)), float(np.median(floor_err))
        agree = float(np.mean(np.argmax(got[i][("logits", -1)], -1) == np.argmax(ref_runs[i][0][("logits", -1)], -1)))
        record(np.isfinite(port_err).all() and pm <= 2 * fm,
               f"request {i}: BF16 port median per-token error {pm:.2e}, transformers BF16 floor {fm:.2e}; "
               f"argmax agrees with float32 on {agree:.0%} of positions")


def check_engine(env):
    print("check 9: the real Engine on CPU with capture on", flush=True)
    steps = 3
    reference = env["reference"]
    prompts = [env["seqs"][0][: LENGTHS[0]], env["seqs"][1][: LENGTHS[1]]]
    want = []
    for p in prompts:
        ids, rows, gaps = list(p), [], []
        for _ in range(steps):
            rec = reference.run(ids)
            if not rows:
                rows.append(np.stack([rec[("stream", l)] for l in range(8)], 1))
            else:
                rows.append(np.stack([rec[("stream", l)][-1:] for l in range(8)], 1))
            top = np.sort(rec[("logits", -1)][-1])[-2:]
            gaps.append(float(top[1] - top[0]))
            ids.append(int(np.argmax(rec[("logits", -1)][-1])))
        want.append((ids[len(p):], np.concatenate(rows, 0), gaps))
    engine = cpu_engine.open_engine(env["dirs"]["pub"], capture=True, batch_size=4, token_padding=64,
                                    disable_radix_cache=True, context_length=MAX_SEQ)
    try:
        outs = engine.generate(input_ids=prompts, sampling_params={"temperature": 0.0, "max_new_tokens": steps},
                               return_hidden_states=True)
        if LOG_PAYLOADS:
            for o in outs:
                print("    reply:", {k: v for k, v in o["meta_info"].items() if k != "hidden_states"})
    finally:
        engine.shutdown()
    import capture_activations

    for i, (o, (ids, rows, gaps)) in enumerate(zip(outs, want)):
        got_ids = list(o["output_ids"])
        record(got_ids == ids, f"request {i}: greedy ids {got_ids} == transformers {ids} "
                               f"(smallest top-2 logit gap {min(gaps):.2e})")
        hidden, prompt_rows = capture_activations.hidden_states_from_output(o)
        rows = rows.reshape(rows.shape[0], rows.shape[1], -1)
        n = min(len(hidden), len(rows))
        err = np.max(np.abs(hidden[:n] - rows[:n]), axis=(1, 2)) / np.sqrt(np.mean(rows**2))
        frac = float(np.mean(err > TOL))
        record(frac <= FLIP_FRAC and prompt_rows == len(prompts[i]) and len(hidden) == len(rows),
               f"request {i}: {len(hidden)} captured rows x {hidden.shape[1]} slots x {hidden.shape[2]} "
               f"against transformers' hidden_states, {frac:.0%} above {TOL:.0e}, worst {float(err.max()):.2e}")


# Each mutant rewrites one line of the patched model. (name, old, new).
MUTANTS = [
    ("comb transposed in the stream update", '"tij,tid->tjd"', '"tji,tid->tjd"'),
    ("mHC mixes without the stream norm",
     "flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + norm_eps)", "flat = flat"),
    # Starting Sinkhorn without its first column pass isn't a mutant: 19 more rounds converge to
    # the same doubly stochastic matrix within float32 (measured 7.1e-05, the clean error).
    ("Sinkhorn stops after the first column pass", "for _ in range(iters - 1):", "for _ in range(0):"),
    ("final head takes the first copy", "h = jnp.mean(x, axis=1)", "h = x[:, 0]"),
    ("KDA decay without the lower bound", "return self.lower_bound * jax.nn.sigmoid(a * g)",
     "return -a * jax.nn.softplus(g)"),
    ("KDA state not carried between passes",
     "init = jnp.where((meta.req_prefix > 0)[:, None, None, None], init, 0.0)", "init = init * 0.0"),
    ("KDA convolution window ignored",
     "tok = jnp.where((shift <= meta.idx)[:, None], from_chunk, from_win)",
     "tok = jnp.where((shift <= meta.idx)[:, None], from_chunk, 0.0)"),
    ("KDA query not L2-normalized", "q = l2norm(q.reshape(-1, h, dk).astype(jnp.float32)) * (dk**-0.5)",
     "q = q.reshape(-1, h, dk).astype(jnp.float32) * (dk**-0.5)"),
    ("KDA intra-chunk read drops the diagonal", "intra = jnp.where(tri_incl[None], intra, 0.0)",
     "intra = jnp.where(tri_strict[None], intra, 0.0)"),
    ("KDA output norm without its gate", " * jax.nn.sigmoid(gate.astype(jnp.float32))", ""),
    ("indexer tail dropped", "tok = tok | tail", "tok = tok"),
    ("pool softmax over channels", "prob = jax.nn.softmax(logits, axis=2)", "prob = jax.nn.softmax(logits, axis=-1)"),
    ("pool position embedding dropped", " + self.compress_ape.value.astype(jnp.float32)[None, None]", ""),
    ("indexer keeps one pool too many", "k_sel = min(self.topk // kp, n_req * n_pools)",
     "k_sel = min(self.topk // kp + 1, n_req * n_pools)"),
    ("index scores without the ReLU", "jax.nn.relu(s * scale)", "(s * scale)"),
    ("pools of other requests are candidates",
     "cand = (pool_req[None, :] == meta.req[:, None]) & (pool_last[None, :] <= meta.pos[:, None])",
     "cand = (pool_last[None, :] <= meta.pos[:, None])"),
    ("attention scaled by the latent width", "self.scaling = (self.nope + int(cfg.qk_rope_head_dim)) ** -0.5",
     "self.scaling = self.lora ** -0.5"),
    ("routing weights carry the selection bias", "weights = jnp.take_along_axis(scores, idx, axis=-1)",
     "weights = jnp.take_along_axis(scores + self.gate_bias.value[None, :], idx, axis=-1)"),
    ("routed scaling factor dropped", "return weights * self.route_scale, idx", "return weights, idx"),
    ("SwiGLU gate unclamped", "gate = jnp.minimum(gate, limit)", "gate = gate"),
    ("FP8 scale ignored", "return weight.astype(np.float32) * s", "return weight.astype(np.float32)"),
    ("MLA latent written at the chunk index", "buf = _scatter_rows(buf, meta.slot, meta.pos, latent, meta.valid)",
     "buf = _scatter_rows(buf, meta.slot, meta.idx, latent, meta.valid)"),
]


def run_one_mutant(index):
    name, old, new = MUTANTS[index]
    tree = os.environ["GLM5NEXT_TREE"]
    path = os.path.join(tree, MODEL_FILE)
    src = open(path).read()
    if src.count(old) != 1:
        print("MUTANT " + json.dumps({"error": f"target found {src.count(old)} times"}), flush=True)
        return 0
    open(path, "w").write(src.replace(old, new))
    try:
        env = setup(tree, fetch_published(), quiet=True)
        res = summarize(phase_errors(run_sequence(env["runner"], env["seqs"]),
                                     reference_runs(env["reference"], env["seqs"])))
        out = {k: [bool(v[0]), v[1], v[2]] for k, v in res.items()}
    except Exception as exc:  # a mutant that crashes counts as caught, and says so
        out = {"raised": f"{type(exc).__name__}: {exc}"[:200]}
    print("MUTANT " + json.dumps(out), flush=True)
    return 0


def check_mutants(tree):
    print("check 12: mutants of the patched model", flush=True)
    workdir = tempfile.mkdtemp(prefix="glm5next-mutants-")
    try:
        running, results, pending = {}, {}, list(range(len(MUTANTS)))
        while pending or running:
            while pending and len(running) < MUTANT_WORKERS:
                i = pending.pop(0)
                copy = os.path.join(workdir, f"t{i}")
                shutil.copytree(os.path.join(tree, "python"), os.path.join(copy, "python"))
                env = dict(os.environ, GLM5NEXT_TREE=copy, GLM5NEXT_MUTANT=str(i))
                running[i] = subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for i, p in list(running.items()):
                if p.poll() is not None:
                    out = p.stdout.read()
                    line = [l for l in out.splitlines() if l.startswith("MUTANT ")]
                    results[i] = json.loads(line[-1][7:]) if line else {"raised": out[-300:]}
                    del running[i]
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
SHARD_LAYOUTS = ((2, 2), (4, 4), (8, 8), (8, 2))
SHARD_CONTROLS = (
    ("indexer scores without the cross-shard sum", 'return jax.lax.psum(score(q_, w_, pk), "tensor")',
     "return score(q_, w_, pk)"),
)


def sharded_child() -> int:
    if jax.device_count() < SHARD_DEVICES:
        print(f"SHARDED {json.dumps({'error': f'{jax.device_count()} devices'})}", flush=True)
        return 1
    tree = os.environ["GLM5NEXT_TREE"]
    env = setup(tree, fetch_published(), quiet=True)
    base = run_sequence(env["runner"], env["seqs"])
    for tensor, ep in SHARD_LAYOUTS:
        runner = JaxRunner(env["mod"], env["cfgmod"], env["dirs"]["pub"], make_mesh(tensor), ep_size=ep)
        experts = runner.model.model.layers[2].mlp.experts
        got = run_sequence(runner, env["seqs"])
        res = {}
        for phase in ("prefill", "split", "decode"):
            errs = [compare(g, w) for g, w in zip(got[phase], base[phase])]
            res[phase] = list(verdict(np.concatenate([e[0] for e in errs])))
        out = {"tensor": tensor, "ep": ep, "epmoe": [experts.ep_size, experts.tp_size], "result": res}
        print(f"SHARDED {json.dumps(out)}", flush=True)
    return 0


def check_sharding(tree):
    print(f"check 10: sharded against one device on {SHARD_DEVICES} simulated devices", flush=True)
    workdir = tempfile.mkdtemp(prefix="glm5next-shard-")

    def launch(label, mutant=None):
        copy = os.path.join(workdir, label.replace(" ", "_"))
        shutil.copytree(os.path.join(tree, "python"), os.path.join(copy, "python"))
        if mutant:
            path = os.path.join(copy, MODEL_FILE)
            src = open(path).read()
            if src.count(mutant[0]) != 1:
                return f"target found {src.count(mutant[0])} times"
            open(path, "w").write(src.replace(mutant[0], mutant[1]))
        env = dict(os.environ, GLM5NEXT_TREE=copy, GLM5NEXT_CHILD="sharded")
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
            return None, out[-3000:]
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
                record(ok and r["epmoe"] == [r["ep"], r["tensor"] // r["ep"]],
                       f"--tp-size {r['tensor']} --ep-size {r['ep']} (EPMoE {r['epmoe'][0]} x {r['epmoe'][1]}) "
                       f"equals one device: {detail}")
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
        shape[axis] = -(-shape[axis] // int(np.prod([sizes[n] for n in names])))
    return int(np.prod(shape, dtype=np.int64)) * jnp.dtype(leaf.dtype).itemsize


def fullsize_child() -> int:
    from flax import nnx

    tree = os.environ["GLM5NEXT_TREE"]
    root = fetch_published()
    mod, cfgmod = import_patched(tree)
    if jax.device_count() != V5P64_CHIPS:
        print(f"      need {V5P64_CHIPS} devices, got {jax.device_count()}")
        return 1
    cfg = serving_config(cfgmod, root)
    cfg.ep_size = V5P64_CHIPS
    mesh = make_mesh(V5P64_CHIPS)
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: mod.Glm5NextForConditionalGeneration(cfg, mesh, jnp.bfloat16))
    groups = {"routed experts": 0, "replicated": 0, "sharded other": 0}
    whole_experts = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]:
        keys = [str(getattr(k, "key", getattr(k, "name", k))) for k in path]
        b = shard_bytes(leaf)
        full = int(np.prod(leaf.shape, dtype=np.int64)) * jnp.dtype(leaf.dtype).itemsize
        if "experts" in keys:
            groups["routed experts"] += b
            whole_experts += full
        elif b == full:
            groups["replicated"] += b
        else:
            groups["sharded other"] += b
    experts = model.model.layers[3].mlp.experts
    weights = sum(groups.values())
    budget = V5P_USABLE_GIB * MEM_FRACTION_STATIC
    print(f"      one v5p-64: {V5P64_CHIPS} chips at {V5P_USABLE_GIB} GiB usable; "
          f"--mem-fraction-static {MEM_FRACTION_STATIC} reserves {budget:.2f} GiB a chip")
    print(f"      --tp-size 32 --dp-size 1 --ep-size 32 (EPMoE {experts.ep_size} x {experts.tp_size}, "
          f"{experts.experts_per_device} experts a chip), BF16 weights: "
          + ", ".join(f"{k} {v / GIB:.2f} GiB" for k, v in groups.items())
          + f" = {weights / GIB:.2f} GiB a chip")
    for ctx in (CAPTURE_CONTEXT, 65536):
        shapes = mod.state_shapes(cfg, 2, ctx, jnp.bfloat16)
        per = sum(int(np.prod(sh)) * jnp.dtype(dt).itemsize for v in shapes.values() for sh, dt in v) / 2
        print(f"      state pool, one request slot at context {ctx:,}: {per / 2**20:.1f} MiB, replicated")
    shapes = mod.state_shapes(cfg, CAPTURE_REQS + 1, CAPTURE_CONTEXT, jnp.bfloat16)
    pool = sum(int(np.prod(sh)) * jnp.dtype(dt).itemsize for v in shapes.values() for sh, dt in v)

    class _MC:
        context_len = CAPTURE_CONTEXT

    kv_tokens = mod.Glm5NextForConditionalGeneration.paged_kv_token_cap(None, _MC, CAPTURE_REQS)
    kv = kv_tokens * mod.Glm5NextForConditionalGeneration.paged_kv_layers * 2 * int(cfg.head_dim) * 2
    total = weights + pool + kv
    print(f"      capture serve, {CAPTURE_REQS} requests at --context-length {CAPTURE_CONTEXT}: "
          f"state pool {pool / GIB:.2f} GiB, paged KV pool {kv_tokens:,} tokens x 1 layer = {kv / GIB:.3f} GiB; "
          f"total {total / GIB:.2f} GiB of {budget:.2f} GiB, {budget - total / GIB:.2f} GiB left")
    replicated = total - groups["routed experts"] + whole_experts
    print(f"      control: the same with the routed experts replicated: {replicated / GIB:.2f} GiB, "
          f"{'refused' if replicated / GIB > budget else 'FITS'}")
    ok = (total / GIB < budget and replicated / GIB > budget
          and groups["routed experts"] * V5P64_CHIPS >= whole_experts)
    print(f"FULLSIZE {json.dumps({'ok': ok, 'weights_gib': weights / GIB, 'total_gib': total / GIB, 'budget_gib': budget})}",
          flush=True)
    return 0 if ok else 1


def check_full_size(tree):
    print("check 11: the full model on one v5p-64, from shapes alone", flush=True)
    env = dict(os.environ, GLM5NEXT_TREE=tree, GLM5NEXT_CHILD="fullsize")
    flags = " ".join(f for f in env.get("XLA_FLAGS", "").split()
                     if not f.startswith("--xla_force_host_platform_device_count"))
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={V5P64_CHIPS}".strip()
    done = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    for line in done.stdout.splitlines():
        if line.startswith("      "):
            print(line)
    res = [json.loads(l[9:]) for l in done.stdout.splitlines() if l.startswith("FULLSIZE ")]
    if not res:
        print((done.stdout + done.stderr)[-3000:])
    record(bool(res) and res[0]["ok"],
           "GLM-5.3-Flash fits one v5p-64 chip under --mem-fraction-static 0.8: "
           + (f"{res[0]['weights_gib']:.2f} GiB of weights, {res[0]['total_gib']:.2f} of "
              f"{res[0]['budget_gib']:.2f} GiB with the capture serve" if res else "no report"))


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="glm5next-gate-")
    try:
        root = fetch_published()
        tree = os.environ.get("GLM5NEXT_TREE") or check_patch(workdir)
        if tree is None:
            return 1
        env = setup(tree, root)
        check_layout(env, root)
        ref_runs = reference_runs(env["reference"], env["seqs"])
        got = check_forward(env, ref_runs)
        check_kda(env, got)
        check_capture_path(env)
        check_bf16(env, ref_runs)
        cpu_engine._tree = tree
        if os.environ.get("GLM5NEXT_SKIP_ENGINE") != "1":
            check_engine(env)
        if os.environ.get("GLM5NEXT_SKIP_SHARDING") != "1":
            check_sharding(tree)
            check_full_size(tree)
        if os.environ.get("GLM5NEXT_SKIP_MUTANTS") != "1":
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
    if os.environ.get("GLM5NEXT_MUTANT"):
        raise SystemExit(run_one_mutant(int(os.environ["GLM5NEXT_MUTANT"])))
    if os.environ.get("GLM5NEXT_CHILD") == "sharded":
        raise SystemExit(sharded_child())
    if os.environ.get("GLM5NEXT_CHILD") == "fullsize":
        raise SystemExit(fullsize_child())
    raise SystemExit(main())
