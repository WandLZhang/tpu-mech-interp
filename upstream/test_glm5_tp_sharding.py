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

"""Gate for `glm5-tp-sharding.patch`: GLM-5.3 in its FP8 layout serves at --tp-size 4 and 8.

Runs on CPU, one real `Engine` per child process on 8 simulated devices. No TPU needed.

    python3 upstream/test_glm5_tp_sharding.py

On a v5p-64 (2026-09-27) GLM-5.3 at --tp-size 32 died in the first extend precompile:

    ValueError: in_specs passed to shard_map: P('data', None) does not match the specs of the
    input: P('data', 'tensor') for arg: bfloat16[1024@data,2048@tensor].

The layer is a shared expert's `down_proj`. `QuantizedLinear.from_linear` drops a row-parallel
FP8 layer to a replicated reduce axis when its 128-wide input blocks don't tile across TP, and
GLM-5.3's shared expert has 2048 / 128 = 16 blocks against TP 32. Its producer, `a2 * silu(a1)`
from the col-parallel gate and up projections, stays sharded on "tensor", and the mesh runs
Explicit axes, so `shard_map` refuses it. The loader places the layer's `weight_q` row-parallel
too. The patch reshards both to the layer's specs, places each FP8 `weight_q` where its
`kernel_axes` say at load, and copies mesh-wide hidden states to the host before the capture
slices them on one host.

The engine runs on CPU as it stands except for three things. The block-wise FP8 matmul, the
MLA paged attention and the routed experts' grouped matmul are Pallas TPU kernels, and the
children swap each for an XLA stand-in with the kernel's contract and float32 sums; the
`shard_map`s around them, where the bug lives, stay the engine's. The model runner's refusal of
block-wise FP8 off a TPU opens for CPU in both test trees. The MLA backend reads a VMEM size from
`get_tpu_info()`, and the children register a v5p's figures for the CPU device kind.

The tiny checkpoint keeps the ratio. Its shared expert is 256 wide, 2 blocks, so --tp-size 4
leaves each shard half a block, as TP 32 does on the published model. Its tensor names per layer
kind are the published index's, FP8 E4M3 with a 128 x 128 `weight_scale_inv` wherever the
published checkpoint has one, and its config keeps GLM-5.3's `indexer_types`, dense layers and
MTP layer. The engine's MLA dims are GLM-5's own (q_lora_rank 2048, kv_lora_rank 512, 256-wide
heads), so the tiny model keeps them and shrinks the width, the heads, the experts and the depth.

Checks:

1. The patch applies to `sglang-jax` at `SGL_COMMIT` on the tree the GLM-5.3 row of
   `scripts/multihost_run.sh` builds (877, both steering patches,
   `multihost-hidden-states.patch`, `glm5-capture-hook.patch`), and a corrupted copy doesn't.
2. The tiny checkpoint's layout against the published index, and the block arithmetic that
   sends the shared expert's `down_proj` to the fallback at tp 4 and 8, as at tp 32 on GLM-5.3.
3. Without the patch, the engine at tp 4 and at tp 8 dies with the chip's ValueError. The same
   tree serves at tp 1.
4. With the patch, tp 1 serves and equals the unpatched tree's tp 1: the patch changes nothing
   where it has nothing to reshard.
5. With the patch, tp 8 equals tp 1, and tp 4 equals tp 1 on a copy of the checkpoint with the
   routed experts' scales zeroed: greedy ids, the log-softmax of the logits over the whole
   vocabulary at every prompt and decode position, and every captured layer, per token within
   float32 rounding. The served model's shared-expert `down_proj` holds its `weight_q`
   replicated from load on. tp 4 with the routed experts on is printed as a note. On CPU the
   routed experts at --ep-size 4 move the second prompt's tokens by up to 9%, which zeroing
   them removes and zeroing the shared expert doesn't.
6. Control: tp 4 on the routed-off copy with one shared-expert `down_proj` block scale at 1.5x
   has to differ from tp 1 on the routed-off copy. It shows check 5 reads the shared expert.

Every run serves at --ep-size equal to --tp-size, the way the GLM-5.3 row serves at 32 and 32, so
each routed expert sits whole on one device.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download. The published
`config.json` and index are cached under `GLM5TP_CACHE`, or `~/.cache/glm5.3-published/<revision>`,
and each has to match its pinned SHA-256. `LOG_PAYLOADS=1` prints every engine reply, and
`GLM5TP_KEEP=1` keeps the trees, checkpoints, child logs and replies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(HERE, "capture-hooks"))

PATCH = os.path.join(HERE, "glm5-tp-sharding.patch")
# The GLM-5.3 row of scripts/multihost_run.sh, in its order, before this patch.
BELOW = ("multihost-hidden-states.patch", "glm5-capture-hook.patch")
LOG_PAYLOADS = os.environ.get("LOG_PAYLOADS") == "1"
CPU_GUARD = ("python/sgl_jax/srt/model_executor/model_runner.py",
             'if wbs is not None and jax.default_backend() != "tpu":',
             'if wbs is not None and jax.default_backend() not in ("tpu", "cpu"):')

HUB = "https://huggingface.co/zai-org/GLM-5.3/resolve"
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
PUBLISHED = {
    "config.json": "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
    "model.safetensors.index.json": "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf",
}
# A published layer of each kind the tiny model holds, and the MTP layer.
PUBLISHED_KIND = {"dense_full": 0, "moe_full": 6, "moe_shared": 3, "mtp": 78}

DEVICES = 8
TP_SIZES = (1, 4, 8)
BLOCK = (128, 128)
VOCAB = 512
# The engine's own MLA dims (glm5_moe.py hardcodes GLM-5's), the rest shrunk. Layer 0 dense,
# layers 1-3 routed; layers 0-1 carry an indexer and layers 2-3 share it, the published pattern.
TINY = dict(
    hidden_size=256,
    intermediate_size=1024,
    moe_intermediate_size=256,
    num_hidden_layers=4,
    first_k_dense_replace=1,
    num_attention_heads=8,
    num_key_value_heads=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    indexer_types=["full", "full", "shared", "shared"],
    index_skip_topk_offset=2,
    index_topk=32,
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
)
PROMPT_LENGTHS = (90, 37)  # the first prefills over two 64-token chunks
DECODE_STEPS = 3
CONTEXT = 256  # --context-length
# Per-token error: the max abs difference over a token's elements, over the tp 1 tensor's RMS.
# Float32 on both sides; a sharded run reorders the sums, nothing more.
TOL = 1e-4
CONTROL_SCALE = 1.5

RESULTS = []


def record(ok, label):
    RESULTS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)
    return ok


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


# --------------------------------------------------------------------------------------------
# Published files and the tiny checkpoint
# --------------------------------------------------------------------------------------------


def fetch_published():
    root = os.environ.get("GLM5TP_CACHE") or os.path.expanduser(
        f"~/.cache/glm5.3-published/{REVISION}")
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


def published_names(root):
    """Per layer kind, the published tensor names relative to the layer, experts as expert 0."""
    with open(os.path.join(root, "model.safetensors.index.json")) as fp:
        keys = set(json.load(fp)["weight_map"])
    out = {}
    for kind, layer in PUBLISHED_KIND.items():
        pre = f"model.layers.{layer}."
        rels = set()
        for k in keys:
            if not k.startswith(pre):
                continue
            rel = k[len(pre):]
            if rel.startswith("mlp.experts."):
                parts = rel.split(".")
                if parts[2] != "0":
                    continue
            rels.add(rel)
        out[kind] = sorted(rels)
    out["top"] = sorted(k for k in keys if not k.startswith("model.layers."))
    return out


def tiny_kind(i):
    if i == TINY["num_hidden_layers"]:
        return "mtp"
    mlp = "dense" if TINY["mlp_layer_types"][i] == "dense" else "moe"
    return f"{mlp}_{TINY['indexer_types'][i]}"


def shape_of(rel):
    """The tiny shape of a published tensor name, HF layout [out, in]."""
    h, i, mi = TINY["hidden_size"], TINY["intermediate_size"], TINY["moe_intermediate_size"]
    nh, e = TINY["num_attention_heads"], TINY["n_routed_experts"]
    q_lora, kv_lora, nope, rope, v = 2048, 512, 192, 64, 256
    table = {
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
        "self_attn.q_a_proj.weight": (q_lora, h),
        "self_attn.q_a_layernorm.weight": (q_lora,),
        "self_attn.q_b_proj.weight": (nh * (nope + rope), q_lora),
        "self_attn.kv_a_proj_with_mqa.weight": (kv_lora + rope, h),
        "self_attn.kv_a_layernorm.weight": (kv_lora,),
        "self_attn.kv_b_proj.weight": (nh * (nope + v), kv_lora),
        "self_attn.o_proj.weight": (h, nh * v),
        "self_attn.indexer.wq_b.weight": (32 * 128, q_lora),
        "self_attn.indexer.wk.weight": (128, h),
        "self_attn.indexer.k_norm.weight": (128,),
        "self_attn.indexer.k_norm.bias": (128,),
        "self_attn.indexer.weights_proj.weight": (32, h),
        "mlp.gate_proj.weight": (i, h),
        "mlp.up_proj.weight": (i, h),
        "mlp.down_proj.weight": (h, i),
        "mlp.gate.weight": (e, h),
        "mlp.gate.e_score_correction_bias": (e,),
        "mlp.shared_experts.gate_proj.weight": (mi, h),
        "mlp.shared_experts.up_proj.weight": (mi, h),
        "mlp.shared_experts.down_proj.weight": (h, mi),
        "mlp.experts.0.gate_proj.weight": (mi, h),
        "mlp.experts.0.up_proj.weight": (mi, h),
        "mlp.experts.0.down_proj.weight": (h, mi),
        "eh_proj.weight": (h, 2 * h),
        "enorm.weight": (h,),
        "hnorm.weight": (h,),
        "shared_head.norm.weight": (h,),
    }
    return table[rel]


def random_tensor(rel, shape, rng):
    n = rng.standard_normal(shape).astype(np.float32)
    if rel.endswith(("norm.weight", "layernorm.weight")):
        return 1.0 + 0.1 * n
    if rel.endswith("k_norm.bias"):
        return 0.1 * n
    if rel.endswith("e_score_correction_bias"):
        return 0.3 * n
    if len(shape) >= 2:
        return n / math.sqrt(shape[-1])
    return 0.3 * n


def quantize_e4m3(w, block=BLOCK):
    """float32 [out, in] -> (E4M3 as float8, FP32 power-of-two scale_inv per tile). A
    power-of-two scale makes the dequantized weight exact in float32."""
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


def write_safetensors(path, tensors):
    """Raw writer: {name: (dtype tag, numpy array)}."""
    import struct

    header, blobs, offset = {}, [], 0
    for name, (tag, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": tag, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", len(head)))
        fp.write(head)
        for raw in blobs:
            fp.write(raw)


def tiny_config(root):
    """GLM-5.3's config.json with the tiny sizes on top."""
    with open(os.path.join(root, "config.json")) as fp:
        conf = json.load(fp)
    conf.update(TINY)
    conf.update(vocab_size=VOCAB, max_position_embeddings=512, pad_token_id=0, eos_token_id=[0])
    quant = dict(conf["quantization_config"])
    # The published list names every layer's unquantized modules; keep the entries that name no
    # layer, and the per-layer ones the tiny model has.
    keep = []
    for name in quant["modules_to_not_convert"]:
        parts = name.split(".")
        if len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit():
            if int(parts[2]) >= TINY["num_hidden_layers"]:
                continue
        keep.append(name)
    quant["modules_to_not_convert"] = keep
    conf["quantization_config"] = quant
    return conf


def build_checkpoint(out, root, tokdir, seed=0):
    """The tiny checkpoint in the published layout. Returns the tensors it wrote."""
    names = published_names(root)
    rng = np.random.default_rng(seed)
    tensors = {}
    h = TINY["hidden_size"]
    for name in names["top"]:
        shape = (VOCAB, h) if name.endswith(("embed_tokens.weight", "lm_head.weight")) else (h,)
        tensors[name] = ("F32", random_tensor(name, shape, rng))
    for i in range(TINY["num_hidden_layers"] + 1):
        rels = names[tiny_kind(i)]
        scaled = {r[: -len("_scale_inv")] for r in rels if r.endswith(".weight_scale_inv")}
        for rel in rels:
            if rel.endswith(".weight_scale_inv"):
                continue
            experts = range(TINY["n_routed_experts"]) if rel.startswith("mlp.experts.0.") else [None]
            for j in experts:
                full = f"model.layers.{i}." + (rel if j is None else rel.replace(".0.", f".{j}.", 1))
                arr = random_tensor(rel, shape_of(rel), rng)
                if rel in scaled:
                    q, s = quantize_e4m3(arr)
                    tensors[full] = ("F8_E4M3", q.view(np.uint8))
                    tensors[full + "_scale_inv"] = ("F32", s)
                else:
                    tensors[full] = ("F32", arr)
    os.makedirs(out, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(os.path.join(out, shard), tensors)
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as fp:
        json.dump({"metadata": {}, "weight_map": {k: shard for k in tensors}}, fp)
    with open(os.path.join(out, "config.json"), "w") as fp:
        json.dump(tiny_config(root), fp, indent=1)
    for f in os.listdir(tokdir):
        shutil.copy(os.path.join(tokdir, f), out)
    return tensors


CONTROL_TENSOR = "model.layers.2.mlp.shared_experts.down_proj.weight_scale_inv"


def variant_checkpoint(src, dst, tensors, *, routed_off=False, control=False):
    """A copy of `src`. `routed_off` zeroes every routed expert's scales, so the routed experts
    add nothing. `control` scales layer 2's shared-expert down_proj input block 1 by 1.5x."""
    shutil.copytree(src, dst)
    changed = dict(tensors)
    if routed_off:
        for name, (tag, arr) in tensors.items():
            if ".mlp.experts." in name and name.endswith("weight_scale_inv"):
                changed[name] = (tag, np.zeros_like(arr))
    if control:
        tag, arr = changed[CONTROL_TENSOR]
        arr = arr.copy()
        arr[:, 1] *= CONTROL_SCALE
        changed[CONTROL_TENSOR] = (tag, arr)
    write_safetensors(os.path.join(dst, "model-00001-of-00001.safetensors"), changed)
    return changed


# --------------------------------------------------------------------------------------------
# One engine per child process
# --------------------------------------------------------------------------------------------


def install_blockwise_standin():
    """Swap the block-wise FP8 matmul, a Pallas TPU kernel, for XLA on CPU.

    The swap sits inside `xla_quantized_matmul_local`, the function `QuantizedLinear` hands to
    `shard_map`, so the shard_map, its specs and its cross-shard sum stay the engine's. The
    stand-in does the kernel's math per 128-wide input block: the activation quantized per row
    with `util.quantize_block` when the kernel would, the block's dot product in float32, times
    the activation scale and the weight scale. It sums the blocks in float32 where the kernel
    sums them in BF16, so a sharded run differs from one device by float32 rounding alone.
    """
    import jax
    import jax.numpy as jnp
    from sgl_jax.srt.kernels.quantized_matmul import blockwise_utils
    from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import util

    def standin(x, w_q, w_scale, w_zp=None, block_size=None, x_q_dtype=None, *, tuned_value=None):
        x_q_dtype = x.dtype if x_q_dtype is None else x_q_dtype
        n_in = x.shape[1]
        out = jnp.zeros((x.shape[0], w_q.shape[0]), jnp.float32)
        for i in range(n_in // block_size):
            lo, hi = i * block_size, (i + 1) * block_size
            xs = x[:, lo:hi].astype(jnp.float32)
            if jnp.dtype(x_q_dtype) != x.dtype:
                xq, xscale = util.quantize_block(xs, 1, x_q_dtype)
                xs = xq.astype(jnp.float32)
            else:
                xscale = None
            res = jax.lax.dot_general(xs, w_q[:, lo:hi].astype(jnp.float32),
                                      (((1,), (1,)), ((), ())),
                                      preferred_element_type=jnp.float32)
            if xscale is not None:
                res = res * xscale
            out = out + res * w_scale[i, :, :].astype(jnp.float32)
        return out.astype(x.dtype)

    blockwise_utils._BLOCKWISE_KERNEL = standin
    blockwise_utils._TRIED_LOADING_BLOCKWISE_KERNEL = True


def install_mla_standin():
    """Swap the MLA ragged paged attention, a Pallas TPU kernel, for XLA on CPU.

    dsa_sparse runs it for dense prefill inside its own `shard_map` (`_run_dense`), and its
    sparse decode runs it over each query's top-k pages (`sparse_mla_page_level`); the
    shard_maps stay the engine's. The stand-in keeps the kernel's contract: it writes each new
    token's latent and rope key into its page of the cache, then attends each query over the
    positions of its own sequence up to its own, read through the page table, in float32, and
    returns the latent output and the updated cache.
    """
    import jax
    import jax.numpy as jnp
    from sgl_jax.srt.kernels.dsa import sparse_mla
    from sgl_jax.srt.layers.attention import dsa_sparse_backend

    def standin(ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices, cu_q_lens,
                cu_kv_lens, distribution, *, sm_scale=1.0, sliding_window=None, soft_cap=None,
                **_):
        if sliding_window is not None or soft_cap is not None:
            raise NotImplementedError("the MLA stand-in has no sliding window or soft cap")
        n_tok, _, lkv = ql_nope.shape
        r = q_pe.shape[-1]
        n_pages, ps_pack, pack, width = cache_kv.shape
        ps = ps_pack * pack
        lkv_al = width - (-(-r // 128) * 128)
        n_seq = distribution[-1]
        # Each token's sequence and position.
        t = jnp.arange(n_tok)
        s_t = jnp.clip(jnp.searchsorted(cu_q_lens, t, side="right") - 1, 0, kv_lens.shape[0] - 1)
        valid_t = t < cu_q_lens[n_seq]
        q_len = cu_q_lens[s_t + 1] - cu_q_lens[s_t]
        pos_t = kv_lens[s_t] - q_len + (t - cu_q_lens[s_t])
        page_t = page_indices[cu_kv_lens[s_t] // ps + pos_t // ps]
        slot_t = jnp.where(valid_t, page_t * ps + pos_t % ps, n_pages * ps)
        # Write the new rows into the flat cache.
        flat = cache_kv.reshape(n_pages * ps, width)
        row = jnp.zeros((n_tok, width), flat.dtype)
        row = row.at[:, :lkv].set(new_kv_c.astype(flat.dtype))
        row = row.at[:, lkv_al:lkv_al + r].set(new_k_pe.astype(flat.dtype))
        flat = flat.at[slot_t].set(row, mode="drop")
        # Each query's key positions 0..pos, through its sequence's pages. A sequence holds at
        # most the context plus a page, the sparse path's new-token page included.
        j = jnp.arange(CONTEXT + 2 * ps)
        start = cu_kv_lens[s_t] // ps
        entry = jnp.clip(start[:, None] + j[None, :] // ps, 0, page_indices.shape[0] - 1)
        slot = page_indices[entry] * ps + j[None, :] % ps
        mask = (j[None, :] <= pos_t[:, None]) & (j[None, :] < kv_lens[s_t][:, None])
        mask = mask & valid_t[:, None]
        keys = flat[jnp.where(mask, slot, 0)].astype(jnp.float32)
        kc, kr = keys[..., :lkv], keys[..., lkv_al:lkv_al + r]
        scores = (jnp.einsum("thd,tjd->thj", ql_nope.astype(jnp.float32), kc)
                  + jnp.einsum("thd,tjd->thj", q_pe.astype(jnp.float32), kr)) * sm_scale
        scores = jnp.where(mask[:, None, :], scores, -jnp.inf)
        probs = jnp.where(mask[:, None, :], jax.nn.softmax(scores, axis=-1), 0.0)
        out = jnp.einsum("thj,tjd->thd", probs, kc).astype(ql_nope.dtype)
        return out, flat.reshape(cache_kv.shape)

    dsa_sparse_backend.mla_ragged_paged_attention = standin
    sparse_mla.mla_ragged_paged_attention = standin


def install_gmm_standin():
    """Swap the routed experts' grouped matmul, a Pallas TPU kernel, for XLA on CPU.

    On CPU the engine runs the kernel's v1 through the Pallas interpreter, where a TPU runs v2.
    The stand-in keeps the contract `EPMoE._gmm_compute` relies on: rows sorted by expert,
    `group_sizes` over every expert, `group_offset` naming this shard's first local expert, rows
    of other shards' experts zero, the activation quantized per row when asked and the weight
    scaled per 128-row block, then summed in float32.
    """
    import jax.numpy as jnp
    from sgl_jax.srt.layers import moe
    from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple

    def standin(lhs, rhs, group_sizes, preferred_element_type=jnp.float32, rhs_scale=None,
                rhs_bias=None, tiling=None, group_offset=None, existing_out=None, interpret=None,
                maybe_quantize_lhs=True, zero_initialize=True, acc_dtype=None,
                activation_quantized_dtype=None, v2_tile_info=None):
        m, k = lhs.shape
        n_local = rhs.shape[0]
        x, x_scale = lhs.astype(jnp.float32), None
        if activation_quantized_dtype is not None:
            xq, x_scale = quantize_tensor_simple(lhs, activation_quantized_dtype, dim=-1)
            x = xq.astype(jnp.float32)
        w = rhs.astype(jnp.float32)
        if rhs_scale is not None:
            w = w * jnp.repeat(rhs_scale[:, :, 0, :].astype(jnp.float32),
                               k // rhs_scale.shape[1], axis=1)
        ends = jnp.cumsum(group_sizes)
        group = jnp.searchsorted(ends, jnp.arange(m), side="right")
        local = group - (0 if group_offset is None else group_offset)
        out = jnp.zeros((m, rhs.shape[-1]), jnp.float32)
        for g in range(n_local):
            out = out + jnp.where((local == g)[:, None], x @ w[g], 0.0)
        if x_scale is not None:
            out = out * x_scale.astype(jnp.float32)
        if rhs_bias is not None:
            keep = (local >= 0) & (local < n_local)
            out = out + jnp.where(keep[:, None],
                                  rhs_bias[jnp.clip(local, 0, n_local - 1), 0].astype(jnp.float32),
                                  0.0)
        return out.astype(preferred_element_type)

    moe.gmm = standin


def weight_placement():
    """The partition spec of each FP8 weight in layer 2's shared expert, read off the model the
    engine serves. The single-process engine keeps it in this process."""
    import gc

    for obj in gc.get_objects():
        if type(obj).__name__ == "GlmMoeDsaForCausalLM" and hasattr(obj, "model"):
            mlp = obj.model.layers[2].shared_experts
            out = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                spec = list(getattr(mlp, name).weight_q.value.sharding.spec)
                spec += [None] * (2 - len(spec))
                out[name] = [a if a is None else str(a) for a in spec]
            return out
    return None


def child(tree, ckpt, tp, out_path, ep) -> int:
    """Start the real Engine on `tree` at --tp-size `tp` and --ep-size `ep`, serve both prompts,
    save what it returned. Prints one CHILD line with the outcome."""
    os.environ["SGLANG_JAX_TREE"] = tree
    import cpu_engine

    cpu_engine.stacked_tree()
    import capture_activations
    from jax._src import tpu_info

    # The MLA backend under dsa_sparse sizes its VMEM limit from get_tpu_info() when it's built,
    # and on CPU that raises. jax's registry maps a device kind to a TpuInfo; give "cpu" the
    # figures of a v5p chip, the chip GLM-5.3 serves on.
    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V5P, 2)
    install_blockwise_standin()
    install_mla_standin()
    install_gmm_standin()

    prompts = [np.random.default_rng(7 + k).integers(3, 390, n).tolist()
               for k, n in enumerate(PROMPT_LENGTHS)]
    try:
        engine = cpu_engine.open_engine(
            ckpt, capture=True, batch_size=4, token_padding=64, tp_size=tp, ep_size=ep,
            device_indexes=list(range(tp)), attention_backend="dsa_sparse",
            disable_radix_cache=True, context_length=CONTEXT)
        try:
            outs = engine.generate(
                input_ids=prompts,
                sampling_params={"temperature": 0.0, "max_new_tokens": DECODE_STEPS},
                return_hidden_states=True, return_logprob=True, logprob_start_len=0,
                top_logprobs_num=VOCAB)
            placement = weight_placement()
        finally:
            engine.shutdown()
    except Exception as exc:  # the unpatched tree has to die here; the parent reads why
        traceback.print_exc()
        print("CHILD " + json.dumps({"tp": tp, "raised": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0
    arrays, summary = {}, {"tp": tp, "requests": [], "placement": placement}
    for k, o in enumerate(outs):
        meta = o["meta_info"]
        if LOG_PAYLOADS:
            print("    reply:", json.dumps(capture_activations.reply_summary(o), default=str), flush=True)
        hidden, prompt_rows = capture_activations.hidden_states_from_output(o)
        rows = []
        for entry in list(meta["input_top_logprobs"]) + list(meta["output_top_logprobs"]):
            if entry is None:
                continue
            row = np.full(VOCAB, np.nan, np.float64)
            for logprob, tok, _ in entry:
                row[int(tok)] = logprob
            rows.append(row)
        arrays[f"hidden{k}"] = hidden
        arrays[f"logprobs{k}"] = np.stack(rows)
        arrays[f"ids{k}"] = np.asarray(o["output_ids"], np.int64)
        summary["requests"].append({"ids": list(map(int, o["output_ids"])),
                                    "hidden": list(hidden.shape), "prompt_rows": prompt_rows,
                                    "logprob_rows": len(rows)})
    np.savez(out_path, **arrays)
    print("CHILD " + json.dumps(summary), flush=True)
    return 0


def launch(tree, ckpt, tp, ep, out_path, log_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GLM5TP_CHILD="1")
    flags = " ".join(f for f in env.get("XLA_FLAGS", "").split()
                     if not f.startswith("--xla_force_host_platform_device_count"))
    env["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={DEVICES}".strip()
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), tree, ckpt, str(tp), out_path, str(ep)],
        env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log


def collect(job):
    proc, log = job
    proc.wait()
    log.close()
    with open(log.name) as fp:
        out = fp.read()
    lines = [l for l in out.splitlines() if l.startswith("CHILD ")]
    if lines:
        result = json.loads(lines[-1][6:])
    else:
        # A scheduler that fails signals the engine, which kills the process before the child's
        # own except clause runs. The exception's last line is in the log.
        errors = [l for l in out.splitlines()
                  if l.split(":", 1)[0].endswith(("Error", "Exception")) and ": " in l]
        result = {"raised": errors[-1] if errors else "no exception line",
                  "exit": proc.returncode}
    print(f"    child payload: {json.dumps(result)}", flush=True)
    return result, out


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def build_trees(workdir):
    """Check 1. The unpatched tree and the patched one."""
    import cpu_engine
    from test_capture_hooks import corrupt

    print("check 1: the patch applies to the GLM-5.3 tree", flush=True)
    base = cpu_engine.build_tree(os.path.join(workdir, "base"),
                                 os.environ.get("SGLANG_JAX_REPO") or None)
    for name in BELOW:
        done = git(base, "apply", os.path.join(HERE, name))
        if not record(done.returncode == 0, f"{name} applies to the stack"):
            print(done.stderr)
            return None, None
    # The model runner refuses block-wise FP8 off a TPU, since the matmul is a Pallas TPU kernel.
    # The children run every Pallas TPU kernel in jax's TPU interpret mode, so both test trees
    # let the CPU through that one check.
    path = os.path.join(base, CPU_GUARD[0])
    with open(path) as fp:
        src = fp.read()
    if not record(src.count(CPU_GUARD[1]) == 1,
                  f"the block-wise FP8 TPU check in {CPU_GUARD[0]} is there to open for CPU"):
        return None, None
    with open(path, "w") as fp:
        fp.write(src.replace(CPU_GUARD[1], CPU_GUARD[2]))
    git(base, "add", "-A")
    git(base, "-c", "user.email=t@local", "-c", "user.name=t", "commit", "-qm", "glm5.3 row")
    fixed = os.path.join(workdir, "fixed")
    done = subprocess.run(["git", "clone", "-q", base, fixed], capture_output=True, text=True)
    if done.returncode:
        record(False, f"clone the base tree: {done.stderr}")
        return None, None
    shutil.copy(os.path.join(base, ".git", "cpu-engine-patches"),
                os.path.join(fixed, ".git", "cpu-engine-patches"))
    with open(PATCH) as fp:
        bad = os.path.join(workdir, "corrupt.patch")
        with open(bad, "w") as out:
            out.write(corrupt(fp.read()))
    record(git(fixed, "apply", "--check", bad).returncode != 0, "control: a corrupted copy is refused")
    done = git(fixed, "apply", PATCH)
    if not record(done.returncode == 0,
                  f"glm5-tp-sharding.patch applies on {' + '.join(BELOW)}"):
        print(done.stderr)
        return None, None
    touched = [l[6:].strip() for l in open(PATCH) if l.startswith("+++ b/")]
    for path in touched:
        done = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(fixed, path)],
                              capture_output=True, text=True)
        record(done.returncode == 0, f"{path} compiles")
    return base, fixed


def check_layout(root, tensors):
    print("check 2: the tiny checkpoint against the published layout", flush=True)
    names = published_names(root)
    for i in range(TINY["num_hidden_layers"] + 1):
        pre = f"model.layers.{i}."
        mine = set()
        for k in tensors:
            if k.startswith(pre):
                parts = k[len(pre):].split(".")
                if parts[:2] == ["mlp", "experts"]:
                    parts[2] = "0"
                mine.add(".".join(parts))
        mine = sorted(mine)
        want = names[tiny_kind(i)]
        record(mine == want, f"layer {i} ({tiny_kind(i)}) holds the {len(want)} names of published "
                             f"layer {PUBLISHED_KIND[tiny_kind(i)]}")
        fp8 = sorted(k[len(pre):] for k, (tag, _) in tensors.items()
                     if k.startswith(pre) and tag == "F8_E4M3" and ".experts." not in k)
        scaled = sorted(r[: -len("_scale_inv")] for r in want
                        if r.endswith(".weight_scale_inv") and ".experts." not in r)
        record(fp8 == scaled, f"layer {i}: FP8 where, and only where, the published layer has a "
                              f"weight_scale_inv ({len(fp8)} tensors)")
    with open(os.path.join(root, "config.json")) as fp:
        pub = json.load(fp)
    blk = pub["quantization_config"]["weight_block_size"][1]
    pub_blocks = pub["moe_intermediate_size"] * pub["n_shared_experts"] // blk
    tiny_blocks = TINY["moe_intermediate_size"] * TINY["n_shared_experts"] // BLOCK[1]
    record(pub_blocks % 32 != 0 and all(tiny_blocks % tp != 0 for tp in TP_SIZES if tp > 1),
           f"the shared expert's down_proj has {pub_blocks} input blocks on GLM-5.3 against tp 32, "
           f"and {tiny_blocks} here against tp 4 and 8: both drop to a replicated reduce axis")


def per_token(got, want):
    """Max abs difference per token over the reference's RMS."""
    got = np.asarray(got, np.float64).reshape(len(got), -1)
    want = np.asarray(want, np.float64).reshape(len(want), -1)
    rms = np.sqrt(np.mean(want[np.isfinite(want)] ** 2))
    diff = np.where(np.isfinite(want) & np.isfinite(got), np.abs(got - want), np.inf)
    diff = np.where(np.isnan(want) & np.isnan(got), 0.0, diff)
    return diff.max(axis=1) / rms


def compare(got_path, want_path):
    """(ids equal, worst logprob error, worst capture error, shapes equal) over both requests."""
    got, want = np.load(got_path), np.load(want_path)
    ids_ok, shapes_ok, lp, hid = True, True, 0.0, 0.0
    for k in range(len(PROMPT_LENGTHS)):
        ids_ok &= np.array_equal(got[f"ids{k}"], want[f"ids{k}"])
        for key in (f"logprobs{k}", f"hidden{k}"):
            if got[key].shape != want[key].shape:
                shapes_ok = False
            n = min(len(got[key]), len(want[key]))
            if n == 0 or got[key].shape[1:] != want[key].shape[1:]:
                continue
            err = float(per_token(got[key][:n], want[key][:n]).max())
            if key.startswith("logprobs"):
                lp = max(lp, err)
            else:
                hid = max(hid, err)
    return ids_ok, lp, hid, shapes_ok


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="glm5tp-")
    try:
        root = fetch_published()
        base, fixed = build_trees(workdir)
        if base is None:
            return 1
        import cpu_engine

        tokdir = os.path.join(workdir, "tok")
        cpu_engine.write_tokenizer(tokdir)
        ckpt = os.path.join(workdir, "ckpt")
        tensors = build_checkpoint(ckpt, root, tokdir)
        check_layout(root, tensors)
        # tp 4 at --ep-size 4 moves a few tokens by up to 9% in the routed experts on CPU (see
        # the note check 5 prints), so tp 4 also runs on a copy with the routed experts off.
        routed_off = os.path.join(workdir, "ckpt-routed-off")
        variant_checkpoint(ckpt, routed_off, tensors, routed_off=True)
        ctrl = os.path.join(workdir, "ckpt-control")
        variant_checkpoint(ckpt, ctrl, tensors, routed_off=True, control=True)

        # (label, tp): (tree, checkpoint, ep). Every run serves at --ep-size equal to --tp-size,
        # the way the GLM-5.3 row serves at 32 and 32.
        runs = {("base", 1): (base, ckpt, 1), ("base", 4): (base, ckpt, 4),
                ("base", 8): (base, ckpt, 8), ("fixed", 1): (fixed, ckpt, 1),
                ("fixed", 4): (fixed, ckpt, 4), ("fixed", 8): (fixed, ckpt, 8),
                ("off", 1): (fixed, routed_off, 1), ("off", 4): (fixed, routed_off, 4),
                ("control", 4): (fixed, ctrl, 4)}
        jobs = {}
        for key, (tree, path, ep) in runs.items():
            out = os.path.join(workdir, f"{key[0]}-tp{key[1]}.npz")
            jobs[key] = (launch(tree, path, key[1], ep, out, out[:-4] + ".log"), out)
        results = {}
        for key, (job, out) in jobs.items():
            print(f"  run: {key[0]} at --tp-size {key[1]} --ep-size {runs[key][2]}", flush=True)
            results[key] = collect(job) + (out,)

        print("check 3: without the patch", flush=True)
        for tp in (4, 8):
            res, log = results[("base", tp)][:2]
            raised = res.get("raised", "")
            ok = ("in_specs passed to shard_map: P('data', None) does not match the specs of the "
                  "input: P('data', 'tensor')") in raised
            if not record(ok, f"tp {tp} dies with the chip's ValueError: {raised[:240]}"):
                print(log)
        res, log = results[("base", 1)][:2]
        if not record("raised" not in res, "the unpatched tree serves at tp 1"):
            print(log)

        print("check 4: the patch at tp 1", flush=True)
        res, log = results[("fixed", 1)][:2]
        if not record("raised" not in res, "the patched tree serves at tp 1"):
            print(log)
            return 1
        want = results[("fixed", 1)][2]
        if "raised" not in results[("base", 1)][0]:
            ids_ok, lp, hid, shapes = compare(results[("base", 1)][2], want)
            record(ids_ok and shapes and lp == 0.0 and hid == 0.0,
                   f"tp 1 with the patch equals tp 1 without it: ids {ids_ok}, worst log-softmax "
                   f"{lp:.1e}, worst capture {hid:.1e}")

        print("check 5: the patch at tp 4 and 8 against tp 1", flush=True)

        def equal(key, base, label):
            res, log = results[key][:2]
            if "raised" in res:
                record(False, f"{label} serves: {res['raised'][:300]}")
                print(log)
                return None
            ids_ok, lp, hid, shapes = compare(results[key][2], results[base][2])
            reqs = res["requests"]
            return res, ids_ok and shapes and lp <= TOL and hid <= TOL, (
                f"{label}: greedy ids {[r['ids'] for r in reqs]} "
                f"{'match' if ids_ok else 'differ'}, row counts {'match' if shapes else 'differ'}, "
                f"log-softmax over {VOCAB} ids at {sum(r['logprob_rows'] for r in reqs)} "
                f"positions worst {lp:.1e}, {sum(r['hidden'][0] for r in reqs)} captured rows x "
                f"{reqs[0]['hidden'][1]} layers worst {hid:.1e} (tolerance {TOL:.0e})")

        want_place = {"gate_proj": ["tensor", None], "up_proj": ["tensor", None],
                      "down_proj": [None, None]}
        for key, base, label in (
                (("fixed", 8), ("fixed", 1), "tp 8 equals tp 1"),
                (("off", 4), ("off", 1), "tp 4 equals tp 1 with the routed experts off")):
            got = equal(key, base, label)
            if got is None:
                continue
            res, ok, text = got
            record(ok, text)
            place = res.get("placement")
            record(place == want_place,
                   f"tp {key[1]}: layer 2's shared-expert weight_q sits where its layer's "
                   f"kernel_axes say from load on, down_proj replicated: {place}")
        got = equal(("fixed", 4), ("fixed", 1), "tp 4 against tp 1 with the routed experts on")
        if got is not None:
            # Reported, not gated. The routed experts at --ep-size 4 move the second prompt's
            # tokens on CPU. Zeroing the routed experts removes
            # it, zeroing the shared expert doesn't, and giving every expert the same weights
            # doesn't either, so rows go missing or doubled in EPMoE's expert-parallel path.
            print(f"  note: {got[2]}", flush=True)

        print("check 6: control", flush=True)
        res, log = results[("control", 4)][:2]
        if "raised" in res:
            record(False, f"control run: {res['raised'][:300]}")
            print(log)
        else:
            ids_ok, lp, hid, shapes = compare(results[("control", 4)][2], results[("off", 1)][2])
            record(lp > TOL and hid > TOL,
                   f"tp 4 with the routed experts off and {CONTROL_TENSOR} block 1 at "
                   f"{CONTROL_SCALE}x differs from tp 1 with them off: worst log-softmax {lp:.1e}, "
                   f"worst capture {hid:.1e}")
    except Exception:
        traceback.print_exc()
        record(False, "the gate ran to the end")
    finally:
        if os.environ.get("GLM5TP_KEEP") == "1":
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    failed = [label for ok, label in RESULTS if not ok]
    print(f"{len(RESULTS)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if os.environ.get("GLM5TP_CHILD") == "1":
        raise SystemExit(child(*sys.argv[1:3], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])))
    raise SystemExit(main())
