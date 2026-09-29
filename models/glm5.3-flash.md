# GLM-5.3-Flash

`zai-org/GLM-5.3-Flash`

**Status: measured on TPU**, `v5p-64`, 2026-09-28, with the layer filter.
[Capture check on a `v5p-64`](#capture-check-on-a-v5p-64) has one capture check, which passes on
all 45 layers, and one measure run, on a Spot slice in us-east5-a.
[`upstream/models/glm5-next-model.patch`](../upstream/models/glm5-next-model.patch) adds the text
backbone to `sglang-jax` with per-layer capture: 45 layers, the four mHC streams, KDA, the sparse
MLA and its k-pool indexer, and the experts. It serves a tiny checkpoint in the published layout
through the real engine on CPU, and the published config fits one `v5p-64` chip at 35.20 GiB of
76.58 GiB by its shapes. The capture reference runs `transformers`' own `glm5_next`, and the full
45-layer reference on the real checkpoint sits in the bucket as `refs/ref-glm5.3-flash.npz`. The
vision tower and the MTP layer aren't ported.

| | |
|---|---|
| Parameters | 320B total / 18B active |
| Layers | 45: 34 KDA + 11 sparse MLA, every fourth layer MLA |
| Hidden dim | 4,096, carried as 4 mHC streams |
| KDA | 64 heads at 128, per-channel decay with a lower bound of -5, width-4 convolutions |
| Sparse MLA | 64 heads, no position encoding, q LoRA 1,536, KV latent 512, head dim 256; the indexer keeps 512 pools of 4 tokens plus the open tail |
| MLP | layers 0-2 dense at 12,288; the rest 288 routed experts top-8 at 2,048 plus 1 shared |
| Weights | FP8 e4m3, 305.8 GiB on disk in 62 files; 598.5 GiB at BF16 |

The checkpoint names `Glm5NextForConditionalGeneration`, with `model_type` `glm5_next`, at
revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. The MLA projections, the dense MLPs, the
shared expert and every routed expert ship E4M3 with a `weight_scale_inv` per 128 x 128 tile; KDA,
the indexer, `kv_b_proj`, the router and the mHC parameters ship unquantized. Layer 45 is the MTP
layer.

## Serve

Engine: `sglang-jax` at `eb061d8` with `sglang-jax-877.patch`, the steering patches and
[`glm5-next-model.patch`](../upstream/models/glm5-next-model.patch). Start it with
`--disable-radix-cache` and a `--context-length`, which sizes the model's own state pool.

```bash
bash scripts/bootstrap_tpu_vm.sh --model glm5-next
# engine args: --tp-size 32 --ep-size 32 --disable-radix-cache --context-length 4096
```

`scripts/multihost_run.sh` carries the row, `glm5.3-flash`, with capture slot 22, and
[Larger models](../README.md#larger-models) covers the setup. The runs below took this tree and these
engine arguments through that row, on a `v5p-64` in us-east5-a on 2026-09-28. The row also applies
`multihost-hidden-states.patch` and `glm5-next-probe.patch`.

## What the port holds

[`upstream/models/README.md`](../upstream/models/README.md#glm5-next-modelpatch) has the file
list, the design and the test.

- mHC runs in plain XLA. At `eb061d8` the mHC kernel in `kernels/mhc/` has a tile schedule for
  v6e alone and raises `ValueError` on v5p, so the port doesn't call it.
- KDA runs as the chunked form in plain XLA, 64 tokens a chunk, cut at every request start. Its
  FP32 state and convolution window sit in the model's per-request pool.
- The sparse MLA runs absorbed against the cached 512-wide latent, heads split over the tensor
  axis. The indexer is the k-pool one `transformers` writes: GLM-5's DSA path in
  `models/glm5_moe.py` fixes GLM-5's shapes and has no pooling, so the port doesn't use it.
- The routed experts run through `EPMoE` at `--ep-size 32`, 9 a chip, widened from FP8 to BF16 at
  load, each host dequantizing only its own devices' experts.

On one `v5p-64` at `--tp-size 32 --dp-size 1 --ep-size 32`, a chip holds 17.72 GiB of routed
experts, 15.60 GiB of replicated weights and 0.04 GiB of the head, 33.36 GiB, by the shapes
`nnx.eval_shape` builds (check 11 of the test, 32 simulated CPU devices, 2026-09-27). Eight capture
requests at `--context-length 4096` add 1.82 GiB of state and 0.03 GiB of paged KV: 35.20 GiB of
the 76.58 GiB `--mem-fraction-static 0.8` reserves on a 95.73 GiB chip. Replicated routed experts
would take 584 GiB a chip. A state slot costs 206.8 MiB at 4,096 and 1,196.8 MiB at 65,536.
Attention and KDA run replicated on every chip, which the fit allows and which spends compute a TP
split would save.

## Capture

Slot `k` holds the four streams entering layer `k`, flattened to 16,384 values a token: 32 KiB a
token in BF16, 1.41 MiB for all 45 slots. That's `transformers`' `hidden_states[k]` reshaped, so
`check_capture.py` compares it directly. `capture_activations.py` hands `--layers` to the engine
as `--return-hidden-states-layers`, so one kept slot moves 32 KiB a token.

The four streams aren't what a sublayer reads. Each sublayer reads its own `pre` collapse of them,
which the capture doesn't store.

### The capture reference

`scripts/check_capture.py` builds `transformers`' `Glm5NextForConditionalGeneration` on CPU and
streams the decoder layers with `--offload-folder`. `transformers` dequantizes the FP8 weights to
BF16 on CPU and writes the converted layers to the offload folder. `--reference-layers N` cuts the
text config to N layers for a dry run.

Dry runs on a `c4-highcpu-96` in `us-east5-a` with the checkpoint on a 700 GB Hyperdisk Balanced,
on 2026-09-27, with the 440-token and 1,322-token prompts from
`prompts-glm5.3.jsonl`, float32 and the BF16 floor, the checkpoint in the page cache:

| Cut | Wall clock | Peak RSS | Offload folder |
|---|---|---|---|
| `--reference-layers 3` (KDA, dense) | 27.6 s | 17.8 GiB | 1.7 GB |
| `--reference-layers 5` (adds MLA and routed layers 3 and 4) | 2 min 15 s | 51.6 GiB | 30 GB |

Slot 0 equals the embedding rows in all four copies, and the first three entries of both runs are
equal bit for bit. The BF16 floor's median per-token error sits at 0.6% at layers 1 and 2.

The full reference ran on the same VM on 2026-09-27, with the disk grown to 1.2 TB, from the copy
staged in a us-east5 GCS bucket (69 files, 328.4 GB, copied to the disk in 6 minutes). The prompts
file came from `build_corpus.py --prompts 400 --tokens 440`, and the check read its first prompt,
440 tokens, and the next three joined, 1,322 tokens:

```bash
python3 scripts/check_capture.py --model-path /mnt/data/glm5.3-flash \
  --offload-folder /mnt/data/off --prompts-file prompts-glm5.3-flash.jsonl --num-prompts 1 \
  --reference-only ref-glm5.3-flash.npz
```

It took 1 hour 41 minutes at a peak RSS of 109.9 GiB. About 30 minutes of that went to
`transformers` writing 624 GB of BF16 offload; the four streamed passes then read it back at 330
to 790 MB/s. The npz is 10.45 GB: 45 stream entries of 16,384 and the final norm, for each
prompt, float32 and the BF16 floor. The floor's median per-token error climbs from 0.6% at layer 1
to 7.8% and 11.6% at the final norm of the two prompts. It sits in the bucket as
`refs/ref-glm5.3-flash.npz` beside `refs/prompts-glm5.3-flash.jsonl`, which is what the `check`
step of `scripts/multihost_run.sh` reads for `glm5.3-flash`.

That first reference was wrong past the first routed layer. The streamed path rounded float32
checkpoint tensors to BF16 (fixed in `check_capture.py`, see
[nemotron3-ultra.md](nemotron3-ultra.md#capture)), and GLM-5.3-Flash's router correction bias is
float32 at 6.18 to 8.28. BF16 collapses a layer's 288 experts onto 7 to 11 distinct values, with
rounding error up to 0.031 against a spread of 0.38 to 0.46 (layers 10 to 12, read from the
published safetensors). The reference was rebuilt with the fix on a `c4-highcpu-96` in
`us-east5-a`, on 2026-09-28 in 1 hour 14 minutes at a peak RSS of 122 GiB. The new npz replaced
`refs/ref-glm5.3-flash.npz`; the old one is `refs/ref-glm5.3-flash.pre-f32fix.npz`.

## Capture check on a `v5p-64`

`scripts/multihost_run.sh NODE us-east5-a glm5.3-flash setup check` on a `v5p-64` Spot slice (32
chips, 8 hosts) in us-east5-a, tp 32, ep 32, BF16 experts widened from FP8, 2026-09-28 05:52 to
06:32Z, against the rebuilt reference, BF16 floor gate:

| Prompt | Layers | Result | Worst ratio to the floor | Worst Pearson | Control |
|---|---|---|---|---|---|
| 440 tokens | 45 | all pass | 0.92 (layer 1) | 0.9968 (layer 44) | 7.02x, detected |
| 1,322 tokens, split across prefill passes | 45 | all pass | 0.92 (layer 1) | 0.9898 (layer 24) | 4.93x, detected |

The 2026-09-27 check of the same engine against the old reference failed from layer 4 on, at 2.3 to
3.9 times the floor. The extend and decode graphs precompiled in 1.9 minutes each.

**Throughput and HBM**, `scripts/multihost_run.sh NODE us-east5-a glm5.3-flash measure`
(`scripts/measure_model.sh` with `PROMPTS=1000`, 440-token prompts), capture slot 22, tp 32, ep 32,
2026-09-28 07:27 to 08:44Z:

| | Tokens/s | Window | Peak HBM per chip (host 0's 4 chips) |
|---|---|---|---|
| Capture off | 1,791.0 | 30 batches, first 2 discarded, 105,600 tokens in 58.96 s | 44.86 of 95.73 GiB (1,066 samples) |
| Capture on, slot 22 | 1,293.5 | steady window 253.42 to 520.11 s, 344,960 tokens | 45.10 of 95.73 GiB (1,189 samples) |

Capture costs 1.38x (1,791.0 over 1,293.5), and the wire carries 32,768 bytes a token (the four
4,096-wide mHC streams in BF16) at 42.4 MB/s. The capture-on run stopped at 357,280 of 440,000
tokens with `OSError: [Errno 28] No space left on device` while writing capture shards to host 0's
`/dev/shm`, which also holds the gcsfuse file cache (the split between the two wasn't read), so
`measure_model.sh` exited 1. The steady window closed before that.

To reproduce, stage the checkpoint and its reference as [Larger models](../README.md#larger-models)
describes, then run from the repo root:

```bash
PROJECT=your-project BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE glm5.3-flash setup check measure
```

Notes: nobody has replayed that block as written from a clean start. The runs above came from the
same script and row. Since 2026-09-29 `setup` leaves host 0's `/dev/shm` 80 GB above the gcsfuse
cache for the capture shards; this run had 20 GB.

## Notes

`qk_rope_head_dim` is 0, so there's no RoPE on the attention path.

`mhc: true` with `hc_sinkhorn_iters: 20` puts a 20-iteration loop inside every sublayer, 90 in a
forward pass.

`upstream/models/test_glm5_next_model.py` checks the port on CPU against `transformers`.
