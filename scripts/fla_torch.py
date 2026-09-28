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

"""A pure-torch stand-in for the fla symbols Kimi K3's modeling code imports, for a CPU forward.

    import fla_torch
    fla_torch.install()     # before transformers imports the checkpoint's remote code

`modeling_kimi_linear.py` in the Kimi K3 repo imports these from fla, whose kernels are Triton and
need a GPU, and raises ImportError without them:

- `fla.modules`: `FusedRMSNormGated`, `ShortConvolution`
- `fla.ops.kda`: `chunk_kda`, `fused_recurrent_kda`
- `fla.ops.utils.index`: `prepare_cu_seqlens_from_mask`, `prepare_lens_from_mask`
- `fla.utils`: `tensor_cache`

`install()` puts modules with those names into `sys.modules` only when `import fla` fails, so a
host with fla keeps the real kernels. Each function here follows the kernel's math, read from
fla-org/flash-linear-attention at 2a39ae5 (2026-09-26): `fla/ops/kda/naive.py`, `gate.py`,
`chunk.py` and `fused_recurrent.py`, `fla/modules/fused_norm_gate.py`, `conv/short_conv.py`,
`l2norm.py`, `fla/ops/utils/index.py` and `fla/utils/_decorators.py`. Everything runs in float32,
as the kernels do, and casts back to the input dtype where the kernel stores it.

- The KDA gate is `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` when `lower_bound` is set,
  and `-exp(A_log) * softplus(g + dt_bias)` otherwise. The kernels read one `A_log` per head at
  `A_log + i_h`, so the gate reads the first `H` entries.
- q and k go through fla's `l2norm`: `x / sqrt(sum(x^2) + 1e-6)`.
- `fused_recurrent_kda` runs the gated delta rule one token at a time.
- `chunk_kda` runs the same rule in 64-token chunks, the form `naive_chunk_kda` writes out, with
  the in-chunk triangular system solved by `torch.linalg.solve_triangular`. The two agree to float
  rounding; `scripts/test_k3_reference.py` checks both against a float64 recurrence.
- `state_v_first` (and its old name `transpose_state_layout`) stores the state as `[V, K]`.

Notes: training, context parallel and CUDA graphs aren't here. A call that asks for them raises.
"""

from __future__ import annotations

import functools
import importlib
import sys
import types
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

# fla's l2norm default.
L2NORM_EPS = 1e-6
# fla's FLA_TENSOR_CACHE_SIZE default.
TENSOR_CACHE_SIZE = 4

STANDIN_ATTR = "__fla_torch_standin__"


# ----------------------------------------------------------------------------------------------
# fla.utils
# ----------------------------------------------------------------------------------------------


def tensor_cache(fn):
    """Memoize the last few calls by argument identity, as fla's decorator does."""
    cached: deque = deque(maxlen=TENSOR_CACHE_SIZE)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for cached_args, cached_kwargs, result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args)) and all(
                    k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return result
        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result

    return wrapper


# ----------------------------------------------------------------------------------------------
# fla.ops.utils.index
# ----------------------------------------------------------------------------------------------


@tensor_cache
def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(lens: torch.Tensor, dtype: torch.dtype | None = torch.int32):
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(mask: torch.Tensor, dtype: torch.dtype | None = torch.int32):
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


def _segments(total: int, cu_seqlens) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, total)]
    bounds = [int(b) for b in cu_seqlens.tolist()]
    return list(zip(bounds[:-1], bounds[1:]))


# ----------------------------------------------------------------------------------------------
# fla.modules
# ----------------------------------------------------------------------------------------------


class FusedRMSNormGated(nn.Module):
    """RMS norm times an output gate: `x / rms(x) * weight * act(g)`, in float32.

    `activation` is `sigmoid` (Kimi K3's `o_norm`), or `swish` / `silu` for `g * sigmoid(g)`. The
    result takes `x`'s dtype, as fla's kernel stores it.
    """

    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5,
                 activation: str = "swish", device=None, dtype=None):
        super().__init__()
        if activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        x32 = x.to(torch.float32)
        if residual is not None:
            x32 = x32 + residual.to(torch.float32)
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(torch.float32)
        g32 = g.to(torch.float32)
        y = y * torch.sigmoid(g32) if self.activation == "sigmoid" else y * g32 * torch.sigmoid(g32)
        out = y.to(x.dtype)
        if not prenorm:
            return out
        return out, (x32 if residual_in_fp32 else x32.to(x.dtype))


class ShortConvolution(nn.Conv1d):
    """fla's causal depthwise convolution, `[D, 1, W]` weight, then `silu`, in float32.

    `cache` is `[N, D, W]`, the last `W` inputs of each sequence, newest last; a forward reads its
    last `W - 1` columns as history, as `step` does. The output takes `x`'s dtype.
    """

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: str | None = "silu", backend: str | None = "triton",
                 device=None, dtype=None, **kwargs):
        super().__init__(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size,
                         groups=hidden_size, bias=bias, padding=kernel_size - 1, device=device,
                         dtype=dtype)
        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(f"Activation `{activation}` not supported yet.")
        self.hidden_size = hidden_size
        self.activation = activation
        self.backend = backend

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, chunk_indices=None, **kwargs):
        batch, total, dim = x.shape
        width = self.kernel_size[0]
        if cu_seqlens is not None and batch != 1:
            raise ValueError("cu_seqlens needs a batch of 1")
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))
        weight = self.weight[:, 0, :].to(torch.float32)          # [D, W]
        bias = None if self.bias is None else self.bias.to(torch.float32)
        rows = [(b, s, e) for b in range(batch) for s, e in _segments(total, cu_seqlens)]
        y = torch.empty(batch, total, dim, dtype=torch.float32)
        states = []
        for n, (b, start, end) in enumerate(rows):
            seq = x[b, start:end].to(torch.float32).transpose(0, 1)       # [D, T]
            if cache is not None:
                history = cache[n].to(torch.float32)
            else:
                history = seq.new_zeros(dim, width)
            full = torch.cat([history, seq], dim=-1)                       # [D, W + T]
            windows = full[:, 1:].unfold(-1, width, 1)                     # [D, T, W]
            out = (windows * weight[:, None, :]).sum(-1)
            if bias is not None:
                out = out + bias[:, None]
            if self.activation is not None:
                out = F.silu(out)
            y[b, start:end] = out.transpose(0, 1)
            states.append(full[:, -width:])
        y = y.to(x.dtype)
        if residual is not None:
            y = y + residual
        final = None
        if output_final_state or cache is not None:
            final = torch.stack(states).to(x.dtype if cache is None else cache.dtype)
            if cache is not None:
                cache.copy_(final)
                final = cache
        return y, (final if output_final_state else None)


# ----------------------------------------------------------------------------------------------
# fla.ops.kda
# ----------------------------------------------------------------------------------------------


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    x = x.to(torch.float32)
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def kda_gate(g, A_log, dt_bias, lower_bound):
    """The per-channel log decay the KDA kernels build from the raw gate, float32 `[..., H, K]`."""
    heads, dim = g.shape[-2:]
    g = g.to(torch.float32)
    if dt_bias is not None:
        g = g + dt_bias.to(torch.float32).reshape(heads, dim)
    a = None if A_log is None else A_log.to(torch.float32).reshape(-1)[:heads].exp().reshape(heads, 1)
    if lower_bound is None:
        if a is None:
            raise ValueError("A_log is required when lower_bound isn't set")
        return -a * F.softplus(g)
    return lower_bound * torch.sigmoid(g if a is None else a * g)


def _prepare(q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel,
             use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, lower_bound,
             state_v_first, cu_seqlens, kwargs):
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and `transpose_state_layout`.")
        state_v_first = kwargs.pop("transpose_state_layout")
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    kwargs.pop("chunk_size", None)
    for unsupported in ("cp_context", "use_graph", "return_intermediate_states"):
        if kwargs.pop(unsupported, None):
            raise NotImplementedError(f"fla_torch doesn't implement {unsupported}")
    for ignored in ("cu_seqlens_cpu", "disable_recompute", "max_num_seqs"):
        kwargs.pop(ignored, None)
    if kwargs:
        raise TypeError(f"unexpected arguments: {sorted(kwargs)}")
    batch, total, heads, dim = q.shape
    value_heads = v.shape[2]
    if value_heads % heads:
        raise ValueError(f"HV={value_heads} isn't a multiple of H={heads}")
    if cu_seqlens is not None and batch != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {batch} with cu_seqlens")
    if scale is None:
        scale = dim ** -0.5
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    q, k = q.to(torch.float32), k.to(torch.float32)
    groups = value_heads // heads
    if groups > 1:
        q = q.repeat_interleave(groups, dim=2)
        k = k.repeat_interleave(groups, dim=2)
    if use_gate_in_kernel:
        if A_log is None and lower_bound is None:
            raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True` and "
                             "`lower_bound` is not set.")
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    elif allow_neg_eigval:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    rows = [(b, s, e) for b in range(batch) for s, e in _segments(total, cu_seqlens)]
    if initial_state is not None:
        if initial_state.shape[0] != len(rows):
            raise ValueError(f"{initial_state.shape[0]} initial states for {len(rows)} sequences")
        initial_state = initial_state.to(torch.float32)
        if state_v_first:
            initial_state = initial_state.transpose(-1, -2)
    return q * scale, k, v, g, beta, initial_state, state_v_first, rows


def _finish(o, v, states, output_final_state, state_v_first):
    final = None
    if output_final_state:
        final = torch.stack(states)
        if state_v_first:
            final = final.transpose(-1, -2).contiguous()
    return o.to(v.dtype), final


def fused_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False,
                        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                        allow_neg_eigval=False, lower_bound=None, state_v_first=False,
                        cu_seqlens=None, **kwargs):
    """The gated delta rule, one token at a time: `S <- S * exp(g_t)`, then the delta write.

    q, k: `[B, T, H, K]`. v: `[B, T, HV, V]`. g: `[B, T, HV, K]`. beta: `[B, T, HV]`. The state is
    `[N, HV, K, V]` float32, or `[N, HV, V, K]` with `state_v_first`.
    """
    q, k, v32, g, beta, initial_state, state_v_first, rows = _prepare(
        q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel, use_gate_in_kernel,
        use_beta_sigmoid_in_kernel, allow_neg_eigval, lower_bound, state_v_first, cu_seqlens,
        dict(kwargs))
    v32 = v32.to(torch.float32)
    heads, dim, vdim = v32.shape[2], q.shape[-1], v32.shape[-1]
    o = torch.zeros(v32.shape, dtype=torch.float32)
    states = []
    for n, (b, start, end) in enumerate(rows):
        state = (initial_state[n].clone() if initial_state is not None
                 else torch.zeros(heads, dim, vdim, dtype=torch.float32))
        for t in range(start, end):
            state = state * g[b, t].exp()[..., None]
            delta = beta[b, t][:, None] * (v32[b, t] - torch.einsum("hk,hkv->hv", k[b, t], state))
            state = state + k[b, t][:, :, None] * delta[:, None, :]
            o[b, t] = torch.einsum("hk,hkv->hv", q[b, t], state)
        states.append(state)
    return _finish(o, v, states, output_final_state, state_v_first)


def _chunk_segment(q, k, v, g, beta, state, chunk):
    """One sequence in chunks. q, k, g: `[T, H, K]`, v: `[T, H, V]`, beta: `[T, H]`."""
    total = q.shape[0]
    pad = (-total) % chunk
    if pad:
        # A zero key, value and beta with a zero log decay leaves the state as it was.
        q, k, v, g = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    n = q.shape[0] // chunk
    # [H, N, C, ...]
    q, k, v, g = (x.reshape(n, chunk, *x.shape[1:]).permute(2, 0, 1, 3) for x in (q, k, v, g))
    beta = beta.reshape(n, chunk, -1).permute(2, 0, 1)
    g = g.cumsum(-2)
    strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=-1)
    causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=0)
    # Kept entries have c > i (or c >= i), where g[c] - g[i] <= 0, so no kept exp overflows.
    diff = g[..., :, None, :] - g[..., None, :, :]                        # [H, N, C, C, K]
    kk = torch.einsum("hncd,hnid,hncid->hnci", k, k, diff.exp().masked_fill(~strict[..., None], 0))
    qk = torch.einsum("hncd,hnid,hncid->hnci", q, k, diff.exp().masked_fill(~causal[..., None], 0))
    eye = torch.eye(chunk, dtype=torch.float32)
    system = eye + beta[..., :, None] * kk                                 # unit lower triangular
    t_mat = torch.linalg.solve_triangular(system, torch.diag_embed(beta), upper=False,
                                          unitriangular=True)
    w = t_mat @ (g.exp() * k)
    u = t_mat @ v
    out = torch.empty_like(u)
    for i in range(n):
        v_new = u[:, i] - w[:, i] @ state
        out[:, i] = (q[:, i] * g[:, i].exp()) @ state + qk[:, i] @ v_new
        last = g[:, i, -1]                                                  # [H, K]
        state = state * last.exp()[..., None]
        state = state + ((last[:, None, :] - g[:, i]).exp() * k[:, i]).transpose(-1, -2) @ v_new
    out = out.permute(1, 2, 0, 3).reshape(n * chunk, out.shape[0], -1)
    return out[:total], state


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=False,
              lower_bound=None, state_v_first=False, cu_seqlens=None, **kwargs):
    """The gated delta rule in 64-token chunks. Same arguments and results as the kernel."""
    chunk = kwargs.get("chunk_size", 64)
    if chunk not in (32, 64):
        raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk}.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and "
                             "`use_gate_in_kernel=True`.")
        if not -5 <= lower_bound < 0:
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be in float32.")
    q, k, v32, g, beta, initial_state, state_v_first, rows = _prepare(
        q, k, v, g, beta, scale, initial_state, use_qk_l2norm_in_kernel, use_gate_in_kernel,
        use_beta_sigmoid_in_kernel, allow_neg_eigval, lower_bound, state_v_first, cu_seqlens,
        dict(kwargs))
    v32 = v32.to(torch.float32)
    heads, dim, vdim = v32.shape[2], q.shape[-1], v32.shape[-1]
    o = torch.zeros(v32.shape, dtype=torch.float32)
    states = []
    for n, (b, start, end) in enumerate(rows):
        state = (initial_state[n].clone() if initial_state is not None
                 else torch.zeros(heads, dim, vdim, dtype=torch.float32))
        if end > start:
            o[b, start:end], state = _chunk_segment(
                q[b, start:end], k[b, start:end], v32[b, start:end], g[b, start:end],
                beta[b, start:end], state, chunk)
        states.append(state)
    return _finish(o, v, states, output_final_state, state_v_first)


# ----------------------------------------------------------------------------------------------
# install
# ----------------------------------------------------------------------------------------------

MODULES = {
    "fla": {},
    "fla.modules": {"FusedRMSNormGated": FusedRMSNormGated, "ShortConvolution": ShortConvolution},
    "fla.ops": {},
    "fla.ops.kda": {"chunk_kda": chunk_kda, "fused_recurrent_kda": fused_recurrent_kda},
    "fla.ops.utils": {},
    "fla.ops.utils.index": {
        "prepare_cu_seqlens_from_mask": prepare_cu_seqlens_from_mask,
        "prepare_lens_from_mask": prepare_lens_from_mask,
        "prepare_cu_seqlens_from_lens": prepare_cu_seqlens_from_lens,
    },
    "fla.utils": {"tensor_cache": tensor_cache},
}


def installed() -> bool:
    """True when `fla` in `sys.modules` is this stand-in."""
    return getattr(sys.modules.get("fla"), STANDIN_ATTR, False)


def install(force: bool = False) -> str:
    """Make the fla imports resolve. Returns "fla" when the real package imports, else "fla_torch".

    `force` installs the stand-in even when fla is importable, for a test that has to run it.
    """
    if installed():
        return "fla_torch"
    if not force:
        try:
            for name, symbols in MODULES.items():
                module = importlib.import_module(name)
                for symbol in symbols:
                    getattr(module, symbol)
            return "fla"
        except (ImportError, AttributeError):
            for name in list(sys.modules):
                if name == "fla" or name.startswith("fla."):
                    del sys.modules[name]
    for name, symbols in MODULES.items():
        module = types.ModuleType(name)
        module.__dict__.update(symbols)
        setattr(module, STANDIN_ATTR, True)
        if "." not in name:
            module.__path__ = []
        sys.modules[name] = module
    for name in MODULES:
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, sys.modules[name])
    return "fla_torch"


def uninstall() -> None:
    """Remove the stand-in from `sys.modules`."""
    for name in list(sys.modules):
        if (name == "fla" or name.startswith("fla.")) and getattr(sys.modules[name], STANDIN_ATTR, False):
            del sys.modules[name]
