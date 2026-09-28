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

"""CPU torch stand-ins for the kernels DeepSeek V4.1's reference runtime imports.

`inference/kernel.py` in `deepseek-ai/DeepSeek-V4.1-Flash` is tilelang and runs on CUDA only.
`install()` puts a module named `kernel` in `sys.modules` with the same six entry points, written
from the tilelang source, so `inference/model.py` imports and runs on a CPU host. The reference
loads its linears as float weights here, so only the in-place quant round trips, `sparse_attn` and
`hc_split_sinkhorn` run; the two quantized GEMMs raise.

`upstream/models/test_deepseek_v41_model.py` checks these against JAX's own casts and the engine's
`kernels/mhc`, and `deepseek_reference.py` runs the full-size capture reference on them.
"""

from __future__ import annotations

import sys
import types


def make_module():
    """`kernel.py`'s six entry points in torch, from the tilelang source. Only the forms the
    reference reaches with BF16-style weights are needed: the in-place quant round trips,
    `sparse_attn` and `hc_split_sinkhorn`. The GEMMs raise, since a float weight never reaches
    them."""
    import torch

    def pow2_ceil_bits(v):
        # fast_log2_ceil and fast_pow2 on the IEEE bits, as the tilelang kernel does
        bits = v.float().contiguous().view(torch.int32)
        exp = (bits >> 23) & 0xFF
        man = bits & ((1 << 23) - 1)
        k = exp - 127 + (man != 0).to(torch.int32)
        return ((k + 127) << 23).to(torch.int32).view(torch.float32)

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def round_e2m1(x):
        # nearest E2M1 magnitude, ties to the even code (mantissa bit 0), sign kept
        mag = x.abs().clamp(max=6.0)
        d = (mag.unsqueeze(-1) - grid).abs()
        best = d.min(dim=-1, keepdim=True).values
        tie = (d == best)
        codes = torch.arange(8)
        # among tied candidates prefer an even code
        score = torch.where(tie, (codes % 2) * 1 + 0, torch.full_like(codes, 9))
        idx = score.argmin(dim=-1)
        return torch.copysign(grid[idx], x)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not inplace:
            raise NotImplementedError("the float reference only reaches the in-place round trip")
        shape = x.shape
        xb = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
        amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        inv = torch.tensor(1.0 / 448.0, dtype=torch.float32)
        s = pow2_ceil_bits(amax * inv) if scale_fmt is not None else amax * inv
        y = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
        x.copy_(y.reshape(shape).to(x.dtype))
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if not inplace:
            raise NotImplementedError("the float reference only reaches the in-place round trip")
        shape = x.shape
        xb = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
        amax = xb.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            amax = amax.clamp_min(6 * 2.0**-9)
            s = (amax / 6.0).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp_min(6 * 2.0**-126)
            s = pow2_ceil_bits(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
        y = round_e2m1((xb / s).clamp(-6.0, 6.0)) * s
        x.copy_(y.reshape(shape).to(x.dtype))
        return x

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, m, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        gathered = torch.gather(
            kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
            idx.clamp_min(0).unsqueeze(-1).expand(b, m, idx.size(-1), d),
        )  # [b, m, k, d]
        s = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(2), -torch.inf)
        sink = attn_sink.float().view(1, 1, h)
        mx = torch.maximum(s.amax(-1), sink)
        p = torch.exp(s - mx.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sink - mx)
        o = torch.einsum("bmhk,bmkd->bmhd", p, gathered.float()) / denom.unsqueeze(-1)
        return o.to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        m = mixes.float()
        pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
        comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def no_gemm(*args, **kwargs):
        raise NotImplementedError("a float weight never reaches the quantized GEMMs")

    mod = types.ModuleType("kernel")
    mod.act_quant = act_quant
    mod.fp4_act_quant = fp4_act_quant
    mod.sparse_attn = sparse_attn
    mod.hc_split_sinkhorn = hc_split_sinkhorn
    mod.fp8_gemm = no_gemm
    mod.fp4_gemm = no_gemm
    mod.round_e2m1 = round_e2m1
    return mod



def install():
    """Put the stand-in `kernel` module in `sys.modules` and return it."""
    module = make_module()
    sys.modules["kernel"] = module
    return module
