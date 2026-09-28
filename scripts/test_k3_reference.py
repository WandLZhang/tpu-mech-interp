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

"""CPU gate for the Kimi K3 capture reference: `fla_torch.py` and `check_capture.py --trust-remote-code`.

    python3 scripts/test_k3_reference.py

The float64 references come from `upstream/models/test_kimi_k3_model.py`, which wrote them from
the published `modeling_kimi_linear.py` and checked them against the engine patch:
`ref_kda_recurrence`, `ref_short_conv`, `ref_rms`, `ref_dequant_mxfp4` and `ref_forward`. The
checkpoint's own `modeling_kimi_linear.py` and `configuration_kimi_k3.py` come from the pinned Hub
revision, and each has to match its SHA-256.

1. `chunk_kda` and `fused_recurrent_kda` against `ref_kda_recurrence` on three packed sequences
   of 150, 37 and 64 tokens, and against each other. A sequence split in two with the state
   carried over has to equal the whole, in both state layouts.
2. `ShortConvolution` against `ref_short_conv` per packed sequence, and a carried cache.
3. `FusedRMSNormGated` against `ref_rms` times a sigmoid gate.
4. `dequantize_mxfp4` against the bit-field decoder, every code and exponent.
5. The checkpoint's own modeling file, on the stand-in, against `ref_forward` on a tiny K3
   checkpoint in the published layout (dense KDA, routed KDA, routed MLA, routed KDA, a block
   boundary every two layers, MXFP4 experts): every hidden state and the logits, float32, on a
   70-token prompt that crosses a KDA chunk and an 11-token one.
6. The streamed reference equals the plain one bit for bit, in float32 and in bfloat16.

Controls, each of which has to fail its check: the softplus KDA gate in place of the lower-bound
one; `A_log` read one value per channel; the MXFP4 nibbles swapped; each hidden state compared
with the next layer's reference; a streamed float32 run without the upcast; a conv that doesn't
restart at a sequence boundary; an `A_log` with a nonzero tail, which the load refuses.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "upstream", "models"))

# Float32 against float64. The kernels on a 150-token recurrence measured 1.6e-06 chunked and
# 1.4e-07 token by token. The four-layer model on a 70-token prompt measured 1.7e-05 with
# chunk_kda and 5.5e-06 with fused_recurrent_kda in its place, spread over the tokens on both sides
# of the 64-token chunk edge: the chunked form takes exp of differences of float32 cumulative
# decays, as fla's kernel does. MODEL_TOL is the patch test's ENGINE_TOL. A control has to move a
# result by MUTANT_TOL, 1e-2, the tolerance the patch test uses; they move it by 0.5 or more.
F32_TOL = 1e-5
MODEL_TOL = 1e-4
MUTANT_TOL = 1e-2

# The checkpoint repo's files this gate runs, at the revision test_kimi_k3_model.py pins.
EXTRA_SHA256 = {
    "configuration_kimi_k3.py": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
}

failures = 0


def report(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    failures += not ok


def control(detected, text):
    global failures
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}", flush=True)
    if not detected:
        print("      FAIL: the control didn't fail, so this check proves nothing.")
        failures += 1


def as_array(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().double().numpy()
    return np.asarray(x, np.float64)


def rel(got, want) -> float:
    got, want = as_array(got), as_array(want)
    if not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def kda_inputs(rng, lengths, heads=4, dim=16):
    total = sum(lengths)
    q, k, v, g = (rng.normal(size=(total, heads, dim)) for _ in range(4))
    beta_raw = rng.normal(size=(total, heads))
    a_log = np.log(rng.uniform(1.0, 16.0, size=heads))
    dt_bias = rng.normal(size=(heads, dim)) * 0.5
    return q, k, v, g, beta_raw, a_log, dt_bias


def check_kda(tk, fla):
    import torch

    print("1. chunk_kda and fused_recurrent_kda against the float64 recurrence", flush=True)
    rng = np.random.default_rng(0)
    lengths = [150, 37, 64]
    heads, dim, lower = 4, 16, -5.0
    q, k, v, g, beta_raw, a_log, dt_bias = kda_inputs(rng, lengths, heads, dim)
    beta = 1.0 / (1.0 + np.exp(-beta_raw))
    want = tk.ref_kda_recurrence(q, k, v, g, beta, tk.per_head_table(a_log, heads, dim), dt_bias,
                                 lengths, lower, dim**-0.5)
    t = lambda x: torch.tensor(x, dtype=torch.float32)[None]  # noqa: E731
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.long)
    common = dict(A_log=torch.tensor(a_log, dtype=torch.float32), dt_bias=t(dt_bias).reshape(-1),
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, lower_bound=lower, cu_seqlens=cu,
                  output_final_state=True, transpose_state_layout=True)
    chunk, chunk_state = fla.chunk_kda(t(q), t(k), t(v), t(g), t(beta_raw), safe_gate=True, **common)
    rec, rec_state = fla.fused_recurrent_kda(t(q), t(k), t(v), t(g), t(beta_raw), **common)
    e_chunk, e_rec = rel(chunk[0], want), rel(rec[0], want)
    e_pair = rel(chunk[0], rec[0].double())
    e_state = rel(chunk_state, rec_state.double())
    report(e_chunk < F32_TOL, f"chunk_kda, sequences {lengths}: rel err {e_chunk:.2e}")
    report(e_rec < F32_TOL, f"fused_recurrent_kda: rel err {e_rec:.2e}")
    report(e_pair < F32_TOL and e_state < F32_TOL,
           f"chunk against recurrent: outputs {e_pair:.2e}, final states {e_state:.2e}")

    # A sequence split at token 100, the state carried, has to equal the whole.
    split = 100
    first = {**common, "cu_seqlens": None}
    for layout in (True, False):
        first["transpose_state_layout"] = layout
        o1, s1 = fla.chunk_kda(t(q[:split]), t(k[:split]), t(v[:split]), t(g[:split]),
                               t(beta_raw[:split]), **first)
        o2, _ = fla.fused_recurrent_kda(t(q[split:150]), t(k[split:150]), t(v[split:150]),
                                        t(g[split:150]), t(beta_raw[split:150]),
                                        initial_state=s1, **first)
        carried = np.concatenate([o1[0].numpy(), o2[0].numpy()])
        e_carry = rel(carried, want[:150])
        report(e_carry < F32_TOL, f"150 tokens as 100 through chunk_kda then 50 through "
                                  f"fused_recurrent_kda, state_v_first={layout}: rel err {e_carry:.2e}")

    # Control: the softplus gate, which the kernels use only when lower_bound isn't set.
    soft, _ = fla.chunk_kda(t(q), t(k), t(v), t(g), t(beta_raw),
                            **{**common, "lower_bound": None})
    control(rel(soft[0], want) > MUTANT_TOL,
            f"the softplus gate -exp(A_log) * softplus(g + dt_bias): rel err {rel(soft[0], want):.2e}")
    per_channel = tk.ref_kda_recurrence(q, k, v, g, beta, tk.per_channel_table(
        np.concatenate([a_log, rng.normal(size=dim - heads)]), heads, dim), dt_bias, lengths,
        lower, dim**-0.5)
    control(rel(chunk[0], per_channel) > MUTANT_TOL,
            f"A_log read one value per channel: rel err {rel(chunk[0], per_channel):.2e}")


def check_conv_and_norm(tk, fla):
    import torch

    print("2. ShortConvolution against ref_short_conv", flush=True)
    rng = np.random.default_rng(1)
    lengths, dim, width = [9, 2, 5], 12, 4
    x = rng.normal(size=(sum(lengths), dim))
    weight = rng.normal(size=(dim, 1, width)) * 0.5
    conv = fla.ShortConvolution(dim, width, activation="silu")
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(weight))
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.long)
    y, state = conv(torch.tensor(x, dtype=torch.float32)[None], cu_seqlens=cu, output_final_state=True)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    want = np.concatenate([tk.ref_short_conv(x[s:e], weight) for s, e in zip(bounds[:-1], bounds[1:])])
    report(rel(y[0], want) < F32_TOL, f"packed sequences {lengths}: rel err {rel(y[0], want):.2e}")
    head, tail = x[:6], x[6:9]
    _, cache = conv(torch.tensor(head, dtype=torch.float32)[None], output_final_state=True)
    y_tail, _ = conv(torch.tensor(tail, dtype=torch.float32)[None], cache=cache.clone(),
                     output_final_state=True)
    e_tail = rel(y_tail[0], want[6:9])
    report(e_tail < F32_TOL, f"9 tokens as 6 then 3 with the cache carried: rel err {e_tail:.2e}")
    joined, _ = conv(torch.tensor(x, dtype=torch.float32)[None])
    control(rel(joined[0], want) > MUTANT_TOL, "one conv across all three sequences, no restart")

    print("3. FusedRMSNormGated against ref_rms times a sigmoid gate", flush=True)
    o, gate = rng.normal(size=(7, 3, 16)), rng.normal(size=(7, 3, 16))
    scale = 1.0 + rng.normal(size=16) * 0.1
    norm = fla.FusedRMSNormGated(16, eps=1e-5, activation="sigmoid")
    with torch.no_grad():
        norm.weight.copy_(torch.tensor(scale))
    got = norm(torch.tensor(o, dtype=torch.float32), torch.tensor(gate, dtype=torch.float32))
    want = tk.ref_rms(o, scale, 1e-5) / (1.0 + np.exp(-gate))
    report(rel(got, want) < F32_TOL, f"rel err {rel(got, want):.2e}")


def check_mxfp4(tk):
    import torch

    from remote_code_reference import dequantize_mxfp4

    print("4. dequantize_mxfp4 against the bit-field decoder", flush=True)
    rng = np.random.default_rng(2)
    packed = rng.integers(0, 256, size=(8, 64)).astype(np.uint8)
    for r in range(4):  # rows 0 to 3 hold all 256 byte values between them
        packed[r] = np.arange(64, dtype=np.uint8) * 4 + r
    scale = rng.integers(100, 150, size=(8, 4)).astype(np.uint8)
    scale[0, 0] = 127
    want = tk.ref_dequant_mxfp4(packed, scale, 32)
    got = dequantize_mxfp4(torch.tensor(packed), torch.tensor(scale), 32).to(torch.float64).numpy()
    report(np.array_equal(got, want), f"8 x 128 at group 32, BF16 holds every value: "
                                      f"{int(np.sum(got != want))} differ")
    swapped = ((packed & 0x0F) << 4) | (packed >> 4)
    bad = dequantize_mxfp4(torch.tensor(swapped), torch.tensor(scale), 32).to(torch.float64).numpy()
    control(not np.array_equal(bad, want), "high nibble first gives other weights")


@contextlib.contextmanager
def patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def tiny_checkpoint(tk, path, perturb=None):
    config, tensors = tk.write_tiny_checkpoint(path, perturb=perturb)
    for name in ("modeling_kimi_linear.py", "configuration_kimi_k3.py"):
        with open(os.path.join(path, name), "w", encoding="utf-8") as fh:
            fh.write(tk.published_source(name))
    return config, tensors


def check_model(tk, fla, root):
    import remote_code_reference
    from check_capture import reference_forward

    print("5. the checkpoint's modeling file on the stand-in against ref_forward", flush=True)
    path = os.path.join(root, "tiny")
    config, tensors = tiny_checkpoint(tk, path)
    text = config["text_config"]
    rng = np.random.default_rng(3)
    batch = [rng.integers(4, text["vocab_size"], size=70).tolist(),
             rng.integers(4, text["vocab_size"], size=11).tolist()]
    head = np.asarray(tensors["language_model.lm_head.weight"]).astype(np.float64)
    wants = []
    for ids in batch:
        streams = []
        logits = tk.ref_forward(tensors, text, ids, streams)
        wants.append((streams, logits))
    got = reference_forward(path, batch, "float32", trust_remote_code=True)
    layers = text["num_hidden_layers"]
    for n, (states, (streams, logits)) in enumerate(zip(got, wants)):
        errors = [rel(states[i], streams[i]) for i in range(layers)]
        e_logits = rel(states[layers] @ head.T, logits)
        report(len(states) == layers + 1 and max(errors) < MODEL_TOL and e_logits < MODEL_TOL,
               f"prompt {n}, {len(batch[n])} tokens: {len(states)} hidden states, worst layer rel "
               f"err {max(errors):.2e}, logits from the last {e_logits:.2e}")
        shifted = min(rel(states[i], streams[i + 1]) for i in range(layers - 1))
        control(shifted > MUTANT_TOL, f"prompt {n}: each hidden state against the next layer's "
                                      f"reference, closest rel err {shifted:.2e}")

    def softplus_gate(g, A_log, dt_bias, lower_bound):
        return original_gate(g, A_log, dt_bias, None)

    original_gate = fla.kda_gate
    with patched(fla, "kda_gate", softplus_gate):
        bad = reference_forward(path, batch[:1], "float32", trust_remote_code=True)[0]
    worst = max(rel(bad[i], wants[0][0][i]) for i in range(layers))
    control(worst > MUTANT_TOL, f"the model on the softplus KDA gate: worst layer rel err {worst:.2e}")

    def nibbles_swapped(packed, scale, group_size=32, dtype=None):
        return original_dequant(((packed & 0x0F) << 4) | (packed >> 4), scale, group_size, dtype)

    original_dequant = remote_code_reference.dequantize_mxfp4
    with patched(remote_code_reference, "dequantize_mxfp4", nibbles_swapped):
        bad = reference_forward(path, batch[:1], "float32", trust_remote_code=True)[0]
    worst = max(rel(bad[i], wants[0][0][i]) for i in range(layers))
    control(worst > MUTANT_TOL, f"MXFP4 nibbles swapped on load: worst layer rel err {worst:.2e}")

    def nonzero_tail(tensors):
        key = "language_model.model.layers.0.self_attn.A_log"
        tensors[key] = tensors[key].copy()
        tensors[key][-1] = 0.5

    bad_path = os.path.join(root, "tiny-bad-alog")
    tiny_checkpoint(tk, bad_path, perturb=nonzero_tail)
    refused = False
    try:
        reference_forward(bad_path, batch[:1], "float32", trust_remote_code=True)
    except ValueError as exc:
        refused = "nonzero" in str(exc)
    control(refused, "an A_log with a nonzero value past num_heads is refused on load")
    return path, batch


def check_streamed(path, batch, root):
    from check_capture import reference_forward

    print("6. streamed against plain, bit for bit", flush=True)
    for dtype in ("float32", "bfloat16"):
        plain = reference_forward(path, batch, dtype, trust_remote_code=True)
        streamed = reference_forward(path, batch, dtype, os.path.join(root, "unused-offload"),
                                     trust_remote_code=True)
        same = all(len(p) == len(s) and all(np.array_equal(a, b) for a, b in zip(p, s))
                   for p, s in zip(plain, streamed))
        worst = max(float(np.max(np.abs(a - b))) for p, s in zip(plain, streamed) for a, b in zip(p, s))
        report(same, f"{dtype}: {len(plain[0])} hidden states per prompt, streamed equals plain "
                     f"(worst abs diff {worst:.3g})")
        if dtype == "float32":
            bf = reference_forward(path, batch, "bfloat16", os.path.join(root, "unused-offload"),
                                   trust_remote_code=True)
            differs = any(not np.array_equal(a, b) for p, s in zip(plain, bf) for a, b in zip(p, s))
            control(differs, "a streamed run without the float32 upcast differs from the float32 forward")
    written = os.path.exists(os.path.join(root, "unused-offload"))
    report(not written, "the streamed reference wrote nothing to the offload folder")

    import torch
    from accelerate.utils import set_module_tensor_to_device

    import remote_code_reference

    for dtype_name, want in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for streamed in (False, True):
            model = remote_code_reference.load(path, dtype_name, streamed)
            seen = {}

            def record(module, args, output):
                for n, p in module.named_parameters():
                    seen[p.dtype] = seen.get(p.dtype, 0) + 1

            from check_capture import decoder_layers
            _, layers = decoder_layers(model)
            for layer in layers:
                layer.register_forward_hook(record, prepend=True)
            with torch.no_grad():
                model(input_ids=torch.tensor([batch[1]]), use_cache=False)
            outside = {p.dtype for n, p in model.named_parameters() if not n.startswith("model.layers.")}
            report(set(seen) == {want} and outside == {want},
                   f"{dtype_name}, {'streamed' if streamed else 'plain'}: every layer weight in the "
                   f"forward is {want} ({sum(seen.values())} reads), and every other weight is too")
    probe = torch.nn.Linear(2, 2, device="meta")
    set_module_tensor_to_device(probe, "weight", "cpu", value=torch.ones(2, 2, dtype=torch.bfloat16))
    control(probe.weight.dtype == torch.float32,
            "accelerate without dtype= casts a BF16 value to the float32 meta tensor's dtype")


def main() -> int:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import fla_torch

    kernels = fla_torch.install(force=True)
    print(f"fla symbols from {kernels}", flush=True)
    import test_kimi_k3_model as tk

    tk.PINNED_SHA256.update(EXTRA_SHA256)
    root = tempfile.mkdtemp(prefix="k3-ref-test-")
    try:
        check_kda(tk, fla_torch)
        check_conv_and_norm(tk, fla_torch)
        check_mxfp4(tk)
        path, batch = check_model(tk, fla_torch, root)
        check_streamed(path, batch, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{failures} failure(s)" if failures else "All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
