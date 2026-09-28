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

"""CPU gate for the streamed reference in `check_capture.py`, and for the multi-host settings.

    python3 scripts/test_streamed_reference.py

A model bigger than host RAM gets its capture reference from `reference_forward(...,
offload_folder=...)`, which loads one decoder layer at a time. That reference is only worth
having if it equals the ordinary forward. For tiny random checkpoints of five families this
gate runs both and needs every hidden state to match bit for bit, in float32 and in bfloat16: a
dense Qwen3, a Nemotron-H hybrid of Mamba-2, MoE and attention layers, GLM-5's MLA with a sparse
indexer, Inkling, whose checkpoint is multimodal and loads through the image-text class, and
GLM-5.3-Flash (`glm5_next`), KDA and sparse MLA under four mHC streams, also image-text. A family
the installed `transformers` doesn't know prints SKIP and fails the gate, since each one stands in
for a model this repo serves.

GLM-5.3-Flash's hidden states are the four streams, `[seq, 4, d]` a layer, and the last entry is
the final norm, `[seq, d]`. `reference_forward` flattens each to the engine's slot layout,
`save_reference` writes the narrower last entry under its own key, and `load_reference` gives the
same list back. `--reference-layers 2` on the streamed path has to reproduce the first two
entries of the whole model bit for bit.

The Nemotron-H checkpoint carries a float32 router correction bias near 57, the way Ultra's does,
where BF16's step is 0.25. A streamed run has to read it at float32 in both passes: the float32
pass upcasts, and transformers keeps the bias float32 at BF16 too. Its control streams through the
offload index's own BF16 cast and has to differ from the plain forward.

Controls: a streamed float32 run that skips the upcast has to differ from the float32 forward, and
a reference whose token ids don't match the prompt has to be refused on load. `save_reference`
and `load_reference` have to round-trip every array. `multinode_settings` has to read the three
SGL_ variables and refuse a rank outside the slice.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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


def tiny_configs():
    """(name, config) for each family, or (name, reason) when transformers lacks it."""
    import transformers

    out = []
    common = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    out.append(("qwen3", transformers.Qwen3Config(num_hidden_layers=3, **common)))
    try:
        out.append(("nemotron_h", transformers.NemotronHConfig(
            vocab_size=512, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, intermediate_size=128, mamba_num_heads=8, mamba_head_dim=16,
            ssm_state_size=16, n_groups=2, conv_kernel=4, chunk_size=16,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=32, moe_latent_size=32,
            layers_block_type=["mamba", "moe", "attention", "mamba", "moe"],
            max_position_embeddings=256)))
    except Exception as exc:  # noqa: BLE001 - any failure means the family can't be built
        out.append(("nemotron_h", f"{type(exc).__name__}: {exc}"))
    try:
        cfg_cls = transformers.AutoConfig.for_model("glm_moe_dsa").__class__
        out.append(("glm_moe_dsa", cfg_cls(
            vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
            num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
            num_key_value_heads=4, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
            q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
            v_head_dim=16, index_n_heads=2, index_head_dim=16, index_topk=8,
            max_position_embeddings=256)))
    except Exception as exc:  # noqa: BLE001
        out.append(("glm_moe_dsa", f"{type(exc).__name__}: {exc}"))
    try:
        sys.path.insert(0, os.path.join(HERE, os.pardir, "upstream", "models"))
        from test_inkling import CONFIG as INKLING_TEXT
        from transformers.models.inkling import InklingConfig
        out.append(("inkling", InklingConfig(
            text_config=dict(INKLING_TEXT),
            vision_config=dict(patch_size=2, temporal_patch_size=2, num_channels=3, num_hidden_layers=1),
            audio_config=dict(n_mel_bins=4, mel_vocab_size=8))))
    except Exception as exc:  # noqa: BLE001
        out.append(("inkling", f"{type(exc).__name__}: {exc}"))
    try:
        out.append(("glm5_next", glm5_next_config()))
    except Exception as exc:  # noqa: BLE001
        out.append(("glm5_next", f"{type(exc).__name__}: {exc}"))
    return out


def glm5_next_config():
    """A tiny GLM-5.3-Flash: KDA layers 0-2 and 4, sparse MLA at 3, dense then routed MLPs."""
    from transformers import Glm5NextConfig

    return Glm5NextConfig(
        text_config=dict(
            vocab_size=512, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
            num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=8,
            num_experts_per_tok=2, kv_lora_rank=32, q_lora_rank=48, qk_nope_head_dim=16,
            v_head_dim=16, qk_rope_head_dim=0, index_topk=8, index_head_dim=16, index_n_heads=4,
            index_kpool=4, mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse"],
            linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                                "gate_lower_bound": -5.0},
            max_position_embeddings=256, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=32,
                           out_hidden_size=64, projection_intermediate_size=32))


def main() -> int:
    import torch
    import transformers

    from capture_activations import multinode_settings
    from check_capture import load_reference, model_class, reference_forward, save_reference

    root = tempfile.mkdtemp(prefix="streamed-ref-test-")
    try:
        ids = [[1, 17, 99, 250, 3, 64, 401, 7, 7, 511, 42, 5], [2, 300, 12, 88, 9]]
        print("streamed reference against the plain forward, bit for bit", flush=True)
        for name, cfg in tiny_configs():
            if isinstance(cfg, str):
                report(False, f"{name}: SKIP, transformers {transformers.__version__} can't build it ({cfg})")
                continue
            torch.manual_seed(0)
            model = model_class(cfg).from_config(cfg, dtype=torch.bfloat16)
            if name == "nemotron_h":
                # Ultra's router correction bias sits at 56.98, experts about 0.005 apart, in
                # float32. BF16's step there is 0.25, so a path that rounds it routes as if the
                # bias were one constant. A zero bias rounds without error and hides that. This one
                # spreads 0.2 around 56.9, wide enough to move a top-2 choice.
                biases = [b for n, b in model.named_buffers() if n.endswith("e_score_correction_bias")]
                for b in biases:
                    b.data = (56.9 + 0.2 * torch.rand(b.shape)).to(torch.float32)
                report(bool(biases) and all(b.dtype == torch.float32 for b in biases),
                       f"nemotron_h: {len(biases)} router biases saved in float32, near 57")
            path = os.path.join(root, name)
            model.save_pretrained(path)
            del model
            text_cfg = getattr(cfg, "text_config", None) or cfg
            vocab = getattr(text_cfg, "unpadded_vocab_size", None) or text_cfg.vocab_size
            ids_m = [[i % vocab for i in seq] for seq in ids]
            for dtype in ("float32", "bfloat16"):
                plain = reference_forward(path, ids_m, dtype)
                streamed = reference_forward(path, ids_m, dtype, os.path.join(root, f"off-{name}-{dtype}"))
                same = all(len(p) == len(s) and all(np.array_equal(a, b) for a, b in zip(p, s))
                           for p, s in zip(plain, streamed))
                worst = max(float(np.max(np.abs(a - b))) for p, s in zip(plain, streamed)
                            for a, b in zip(p, s))
                report(same, f"{name} {dtype}: {len(plain[0])} hidden states per prompt, streamed "
                             f"equals plain (worst abs diff {worst:.3g})")
                if name == "nemotron_h":
                    import check_capture

                    kept = check_capture.checkpoint_value
                    check_capture.checkpoint_value = lambda loader, key, as_stored: loader[key]
                    try:
                        rounded = reference_forward(path, ids_m, dtype,
                                                    os.path.join(root, f"off-{name}-{dtype}-rounded"))
                    finally:
                        check_capture.checkpoint_value = kept
                    control(any(not np.array_equal(a, b) for p, s in zip(plain, rounded)
                                for a, b in zip(p, s)),
                            f"nemotron_h {dtype}: streaming through the offload index's BF16 cast "
                            f"rounds the router bias and differs from the plain forward")
                if name == "glm5_next" and dtype == "float32":
                    widths = [a.shape[1] for a in plain[0]]
                    report(widths[:-1] == [4 * 64] * 5 and widths[-1] == 64,
                           f"glm5_next: {len(widths) - 1} entries of 4 streams x 64 flattened to "
                           f"{widths[0]}, and the final norm at {widths[-1]}")
                    cut = reference_forward(path, ids_m, dtype, os.path.join(root, "off-glm5-cut"),
                                            num_layers=2)
                    same_cut = all(len(c) == 3 and all(np.array_equal(c[i], p[i]) for i in range(2))
                                   for c, p in zip(cut, plain))
                    report(same_cut, "glm5_next --reference-layers 2, streamed: 2 stream entries equal the "
                                     "whole model's first two, then the cut model's final norm")
                    control(any(not np.array_equal(c[1], p[2]) for c, p in zip(cut, plain)),
                            "the cut model's second entry differs from the whole model's third")
                    npz = save_reference(os.path.join(root, "ragged"), ids_m, plain, plain)
                    back, floor_back = load_reference(npz, ids_m, no_floor=False)
                    same_back = all(len(b) == len(p) and all(np.allclose(x, y, rtol=1e-6, atol=1e-6)
                                                               for x, y in zip(b, p))
                                    for b, p in zip(back + floor_back, plain + plain))
                    report(same_back, "glm5_next: save_reference and load_reference round-trip the 4-stream "
                                      "entries and the narrower final norm")
                    stored = np.load(npz)
                    control("reference0_last" in stored.files and stored["reference0"].shape[0] == 5,
                            "the final norm sits under its own key, beside 5 stacked stream entries")
                if dtype == "float32":
                    bf = reference_forward(path, ids_m, "bfloat16", os.path.join(root, f"off-{name}-ctl"))
                    differs = any(not np.array_equal(a, b) for p, s in zip(plain, bf) for a, b in zip(p, s))
                    control(differs, f"{name}: a streamed run without the float32 upcast differs from "
                                     f"the float32 forward")

        print("block FP8 with a partial last tile", flush=True)
        from check_capture import ceil_block_dequantize
        torch.manual_seed(0)
        w = torch.randn(576, 300)
        grid = (-(-576 // 128), -(-300 // 128))
        scales = torch.rand(grid) + 0.5
        full = scales.repeat_interleave(128, 0)[:576].repeat_interleave(128, 1)[:, :300]
        q = (w / full).to(torch.float8_e4m3fn)
        got = ceil_block_dequantize(q, scales, (128, 128), torch.float32)
        want = q.to(torch.float32) * full
        report(torch.equal(got, want), f"576 x 300 at block 128 with a {grid} grid: every tile takes its own scale")
        shifted = ceil_block_dequantize(q, torch.roll(scales, 1, 0), (128, 128), torch.float32)
        control(not torch.equal(shifted, want), "scales moved one tile down give other weights")

        print("save_reference and load_reference", flush=True)
        ref = [[np.random.default_rng(n).standard_normal((len(x), 8)) for n in range(4)] for x in ids]
        flo = [[a + 1e-3 for a in p] for p in ref]
        npz = save_reference(os.path.join(root, "ref"), ids, ref, flo)
        r2, f2 = load_reference(npz, ids, no_floor=False)
        close = all(np.allclose(a, b, rtol=1e-6, atol=1e-6) for p, q in zip(ref, r2) for a, b in zip(p, q))
        close = close and all(np.allclose(a, b, rtol=1e-6, atol=1e-6) for p, q in zip(flo, f2) for a, b in zip(p, q))
        report(close and npz.endswith(".npz"), f"{os.path.basename(npz)} round-trips reference and floor")
        # A reference in the layout the scripts wrote before multi-copy streams: every entry one
        # width, one stacked array a prompt. References built before that change still load.
        old = os.path.join(root, "old-layout.npz")
        np.savez(old, **{k: v for n, x in enumerate(ids) for k, v in (
            (f"ids{n}", np.asarray(x, np.int64)),
            (f"reference{n}", np.stack(ref[n]).astype(np.float32)),
            (f"floor{n}", np.stack(flo[n]).astype(np.float32)))})
        r3, f3 = load_reference(old, ids, no_floor=False)
        same_old = all(len(p) == 4 and all(np.array_equal(a.astype(np.float32), b) for a, b in
                                           zip(p, np.load(old)[f"reference{n}"]))
                       for n, p in enumerate(r3)) and all(len(p) == 4 for p in f3)
        report(same_old, "an npz in the pre-change layout loads unchanged: 4 entries a prompt, "
                         "reference and floor")
        rewritten = save_reference(os.path.join(root, "rewritten"), ids, r3, f3)
        report(not any(k.endswith("_last") for k in np.load(rewritten).files),
               "one-width entries keep the one-array layout on save, with no _last key")
        refused = False
        try:
            load_reference(npz, [ids[0], ids[1][:-1] + [0]], no_floor=False)
        except SystemExit:
            refused = True
        control(refused, "a reference for other token ids is refused")

        print("multinode_settings", flush=True)
        report(multinode_settings({}) == {}, "no SGL_ variables: single-host settings unchanged")
        got = multinode_settings({"SGL_NNODES": "8", "SGL_NODE_RANK": "3", "SGL_DIST_INIT_ADDR": "10.0.0.2:23456"})
        report(got == {"nnodes": 8, "node_rank": 3, "dist_init_addr": "10.0.0.2:23456"}, f"rank 3 of 8: {got}")
        bad = False
        try:
            multinode_settings({"SGL_NNODES": "8", "SGL_NODE_RANK": "8", "SGL_DIST_INIT_ADDR": "x:1"})
        except ValueError:
            bad = True
        control(bad, "rank 8 of an 8-host slice is refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{failures} failure(s)" if failures else "All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
