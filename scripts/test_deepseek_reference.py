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

"""CPU gate for the DeepSeek V4.1-Flash capture reference: `check_capture.py --deepseek-inference`.

    python3 scripts/test_deepseek_reference.py

The tiny checkpoint and the torch reference come from `upstream/models/test_deepseek_v41_model.py`,
which checks the engine patch against them. The checkpoint holds the published layout: FP8 E4M3
dense weights with E8M0 scales per 32x32 tile, packed E2M1 experts, FP8 Engram tables, and the
reference files under `inference/` as the real snapshot holds them. DeepSeek's `inference/model.py`
and `engram.py` come from the pinned Hub revision, each checked against its SHA-256.

1. `deepseek_reference.Reference` in float32, every weight resident, against the model gate's
   torch reference, which loads the same runtime by `load_state_dict`: every capture slot and
   the final norm's output, bit for bit, on a 37-token and a 29-token prompt.
2. The Engram tables read from the memory map against the resident tables, bit for bit.
3. The streamed reference against the resident one, bit for bit, in float32 and in bfloat16.
4. The BF16 pass keeps in float32 the parameters the runtime declares float32.
5. `check_capture.py --deepseek-inference --reference-only --reference-layers 4 --offload-folder`
   end to end: the file it writes holds, per prompt, the first four slots of the full reference
   plus the cut model's final norm, and the token ids.
6. The reference files are pinned: a changed `model.py` is refused before anything imports it.

Controls, each of which has to fail its check: the FP4 nibbles read high first; an Engram row
gathered without its E8M0 factor; a streamed float32 pass that keeps BF16 weights; each slot
compared with the next layer's reference.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "upstream", "models"))

RESULTS = []
INFERENCE_FILES = ("model.py", "engram.py", "kernel.py", "vision.py", "image_processor.py")


def report(ok, text):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    return ok


def worst(a_list, b_list):
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(a_list, b_list))


def tiny_snapshot(gate, root, workdir):
    """The model gate's tiny checkpoint, laid out as the published snapshot is, with the reference
    files under `inference/` and a tiny `inference/config.json` in the runtime's names."""
    import cpu_engine
    from transformers import AutoTokenizer

    ref_model, ref_engram = gate.import_reference(root)
    tokdir = os.path.join(workdir, "tok")
    cpu_engine.write_tokenizer(tokdir)
    tokenizer = AutoTokenizer.from_pretrained(tokdir)
    _, compressed = ref_engram.build_compressed_token_map(tokenizer)
    tiny = dict(gate.TINY, vocab_size=len(tokenizer))
    args = gate.reference_args(ref_model, tiny, dict(
        max_batch_size=1, max_seq_len=gate.MAX_SEQ, dtype="bf16", expert_dtype=None,
        engram_compressed_vocab_size=compressed, n_mtp_layers=0, temperature=0.0))
    import types

    layout = ref_engram.EngramLayout.from_args(types.SimpleNamespace(
        **{**{k: getattr(args, k) for k in args.__dataclass_fields__},
           "engram_num_embeddings": (1,) * len(args.engram_layer_ids)}))
    rows = tuple(sum(sum(p) for p in layer) for layer in layout.primes)
    args.engram_num_embeddings = rows
    tiny["engram_num_embeddings"] = list(rows)
    snap = os.path.join(workdir, "snapshot")
    os.makedirs(os.path.join(snap, "inference"))
    names = gate.reference_state_names(ref_model, args, tokenizer)
    values, tensors = gate.build_checkpoint(snap, names, np.random.default_rng(0), rows)
    with open(os.path.join(snap, "config.json"), "w") as fp:
        json.dump(gate.published_config(tiny, compressed), fp, indent=1)
    tokenizer.save_pretrained(snap)
    for name in INFERENCE_FILES:
        shutil.copy(os.path.join(root, "inference", name), os.path.join(snap, "inference", name))
    runtime_cfg = {k: v for k, v in tiny.items()}
    runtime_cfg.update(engram_compressed_vocab_size=compressed, n_mtp_layers=0,
                       max_seq_len=gate.MAX_SEQ, dtype="fp8", expert_dtype="fp4")
    with open(os.path.join(snap, "inference", "config.json"), "w") as fp:
        json.dump(runtime_cfg, fp, indent=1)
    return snap, tokenizer, args, values, ref_model


def prompts(tokenizer):
    rng = np.random.default_rng(7)
    return [rng.integers(3, len(tokenizer), n).tolist() for n in (37, 29)]


def main() -> int:
    import torch

    import deepseek_reference as dr
    import test_deepseek_v41_model as gate

    workdir = tempfile.mkdtemp(prefix="dsv41-ref-")
    try:
        root = gate.fetch_published()
        snap, tokenizer, args, values, ref_model = tiny_snapshot(gate, root, workdir)
        batch = prompts(tokenizer)
        pins = {}

        print("1. the reference path against the model gate's torch reference, float32", flush=True)
        torch_ref = gate.Reference(ref_model, args, tokenizer, values)
        want = []
        for ids in batch:
            rec = torch_ref.run(ids, 0)
            n = len(torch_ref.model.layers)
            want.append([rec[("collapsed", i)] for i in range(n)] + [rec[("final", -1)]])
        plain = dr.Reference(snap, "float32", streamed=False, max_seq_len=gate.MAX_SEQ,
                             tokenizer=tokenizer, mmap_engram=False, **pins)
        got = [plain.hidden_states(ids) for ids in batch]
        err = max(worst(g, w) for g, w in zip(got, want))
        report(err == 0.0, f"{len(want[0])} entries a prompt, 2 prompts: max abs difference {err:.1e}")
        shifted = max(worst(g[:-1], w[1:]) for g, w in zip(got, want))
        report(shifted > 1e-2, f"control: each slot against the next layer's reference differs by {shifted:.2e}")

        print("2. Engram rows from the memory map", flush=True)
        mapped = dr.Reference(snap, "float32", streamed=False, max_seq_len=gate.MAX_SEQ,
                              tokenizer=tokenizer, mmap_engram=True, **pins)
        got_m = [mapped.hidden_states(ids) for ids in batch]
        err = max(worst(g, w) for g, w in zip(got_m, got))
        report(err == 0.0, f"memory-mapped Engram rows equal the resident table, max abs {err:.1e}")
        original = dr.Reference._mmap_engram

        def no_scale(self, layer):
            import torch as t

            embed = layer.engram.embed
            key = f"layers.{layer.layer_id}.engram.embed."
            table, _ = self.checkpoint.rows(key + "weight")

            def forward(indices):
                flat = indices.reshape(-1).numpy()
                v = t.from_numpy(np.ascontiguousarray(table[flat])).view(t.float8_e4m3fn).float()
                return v.to(t.bfloat16).reshape(*indices.shape, -1)

            embed.forward = forward

        dr.Reference._mmap_engram = no_scale
        try:
            bad = dr.Reference(snap, "float32", streamed=False, max_seq_len=gate.MAX_SEQ,
                               tokenizer=tokenizer, mmap_engram=True, **pins)
            err = max(worst(bad.hidden_states(ids), w) for ids, w in zip(batch, got))
        finally:
            dr.Reference._mmap_engram = original
        report(err > 1e-2, f"control: Engram rows without their E8M0 factor differ by {err:.2e}")

        print("3. streamed against resident", flush=True)
        for dtype in ("float32", "bfloat16"):
            resident = dr.Reference(snap, dtype, streamed=False, max_seq_len=gate.MAX_SEQ,
                                    tokenizer=tokenizer, mmap_engram=True, **pins)
            a = [resident.hidden_states(ids) for ids in batch]
            streamed = dr.Reference(snap, dtype, streamed=True, max_seq_len=gate.MAX_SEQ,
                                    tokenizer=tokenizer, **pins)
            b = [streamed.hidden_states(ids) for ids in batch]
            err = max(worst(x, y) for x, y in zip(a, b))
            report(err == 0.0, f"{dtype}: streamed equals resident, max abs {err:.1e}")
            if dtype == "bfloat16":
                kept = {n: str(d).replace("torch.", "") for n, (s, d) in resident.shapes.items()
                        if n.startswith("layers.2.") and d == torch.float32}
                names = sorted({n.split(".", 2)[-1] for n in kept})
                expected = {"hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale",
                            "hc_ffn_scale", "attn.attn_sink", "ffn.gate.bias",
                            "attn.compressor.wkv.weight", "attn.compressor.wgate.weight"}
                report(set(names) == expected,
                       f"the BF16 pass keeps {len(names)} layer-2 tensors in float32: {', '.join(names)}")
        original_value = dr.Reference.value

        def keep_bf16(self, name):
            v = original_value(self, name)
            return v.to(torch.bfloat16).to(v.dtype) if v.dtype == torch.float32 and "hc_" not in name else v

        dr.Reference.value = keep_bf16
        try:
            lazy = dr.Reference(snap, "float32", streamed=True, max_seq_len=gate.MAX_SEQ,
                                tokenizer=tokenizer, **pins)
            err = max(worst(lazy.hidden_states(ids), w) for ids, w in zip(batch, got))
        finally:
            dr.Reference.value = original_value
        report(err > 1e-4, f"control: a streamed float32 pass on BF16-rounded weights differs by {err:.2e}")
        original_fp4 = dr.dequant_fp4

        def swapped(packed, scale, block=32):
            p = packed.view(torch.uint8)
            return original_fp4(((p >> 4) | (p << 4)).view(torch.int8), scale, block)

        dr.dequant_fp4 = swapped
        try:
            nib = dr.Reference(snap, "float32", streamed=False, max_seq_len=gate.MAX_SEQ,
                               tokenizer=tokenizer, mmap_engram=False, **pins)
            err = max(worst(nib.hidden_states(ids), w) for ids, w in zip(batch, want))
        finally:
            dr.dequant_fp4 = original_fp4
        report(err > 1e-2, f"control: FP4 nibbles read high first differ by {err:.2e}")

        print("5. check_capture.py --deepseek-inference end to end", flush=True)
        prompts_file = os.path.join(workdir, "prompt.txt")
        text = "the city of water and the river north of the stone house"
        out = os.path.join(workdir, "ref.npz")
        env = dict(os.environ)
        done = subprocess.run(
            [sys.executable, os.path.join(HERE, "check_capture.py"), "--model-path", snap,
             "--deepseek-inference", "--reference-only", out, "--reference-layers", "4",
             "--offload-folder", os.path.join(workdir, "off"), "--prompt", text],
            env=env, capture_output=True, text=True)
        ok = done.returncode == 0 and os.path.exists(out)
        if not ok:
            print(done.stdout[-3000:] + done.stderr[-3000:])
        else:
            data = np.load(out)
            ids = [int(i) for i in data["ids0"]]
            full = dr.Reference(snap, "float32", streamed=False, max_seq_len=gate.MAX_SEQ,
                                tokenizer=tokenizer, mmap_engram=False, **pins).hidden_states(ids)
            ref = data["reference0"]
            err = float(np.max(np.abs(ref[:4] - np.stack(full[:4]).astype(np.float32))))
            floor = data["floor0"]
            ok = ref.shape == (5, len(ids), 64) and floor.shape == ref.shape and err == 0.0
            print("\n".join(l for l in done.stdout.splitlines() if "deepseek reference" in l))
            report(ok, f"--reference-layers 4 writes {ref.shape} and a BF16 floor; its four slots "
                       f"equal the full reference's first four, max abs {err:.1e}")
        if not ok:
            report(False, "check_capture.py --deepseek-inference --reference-only")

        print("6. pinned reference files", flush=True)
        bad = os.path.join(workdir, "badsnap")
        shutil.copytree(snap, bad)
        with open(os.path.join(bad, "inference", "model.py"), "a") as fp:
            fp.write("\n# changed\n")
        try:
            dr.check_pins(bad)
            refused = False
        except SystemExit as exc:
            refused = "SHA-256" in str(exc)
        report(refused, "a changed inference/model.py is refused")
        try:
            dr.check_pins(snap)
            report(True, "the tiny snapshot's reference .py files match the pins")
        except SystemExit as exc:
            report(False, str(exc))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    failed = RESULTS.count(False)
    print(f"{len(RESULTS)} checks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
