# GLM-5.3

`zai-org/GLM-5.3`

**Status: measured on TPU**, `v5p-64`, 2026-09-27 and 28, with the layer filter. Served and
captured on 32 chips, 8 hosts, tp 32, ep 32, in us-east5-a. Every captured layer passed the capture
check on 2026-09-27, and the measure ran on 2026-09-28. See
[Measured on a v5p-64](#measured-on-a-v5p-64).

| | |
|---|---|
| Parameters | 753B total / 40B active |
| Layers | 78: the first 3 dense, the rest MoE with 256 routed experts, 8 per token, and 1 shared |
| Hidden dim | 6,144 |
| Attention | MLA with 64 heads (q_lora 2,048, kv_lora 512, rope 64, nope 192, v 256), plus a sparse indexer that keeps the top 2,048 tokens (32 heads × 128) |
| Weights | FP8 e4m3, 703.7 GiB on disk in 141 files; 1,403 GiB at BF16 |
| Context | 1,048,576 positions |

Architecture from `config.json`. Size and file count from `model.safetensors.index.json`, which
reports 755,617,140,416 bytes. The tensor shapes that follow from `config.json` add up to both
parameter counts. The 753B total includes the MTP layer and matches the count on the Hugging Face
page. The 40B active figure leaves out the embedding table and the MTP layer. The BF16 size is the
753B total at 2 bytes each.

## Serve

`sglang-jax` ships the model. `models/glm5_moe.py` registers `GlmMoeDsaForCausalLM`, the
architecture this checkpoint names. At `--tp-size 32` it takes the two fixes [Capture](#capture)
lists.

The sparse indexer is off unless the server asks for it: `model_runner.py` sets
`use_dsa_sparse` only when `--attention-backend dsa_sparse`. Without the flag the model runs dense
MLA, which matches the trained model only while the context stays under 2,048 tokens.

```bash
python3 -m sgl_jax.launch_server \
  --model-path zai-org/GLM-5.3 \
  --dtype bfloat16 \
  --tp-size 32 \
  --attention-backend dsa_sparse \
  --enable-return-hidden-states
```

Untested as written. Across 8 hosts every host runs the engine with `--nnodes 8`, its own
`--node-rank` and host 0's `--dist-init-addr`. For this repo's scripts,
`scripts/multihost_exec.sh` sets the same three settings on every host, and the measured run
[below](#measured-on-a-v5p-64) started the engine that way through `scripts/multihost_run.sh`.

The indexer needs no particular TPU generation. By default it picks its top 2,048 at decode with
an approximate XLA top-k, `approx_max_k` at `recall_target=0.70` in `kernels/dsa/ref.py`. Prefill
runs dense MLA unless `DSA_PREFILL_SPARSE=1`, and so does a batch that mixes prefill with decode.
Set `DSA_INDEXER_KERNEL=1` in the environment to move the decode pick to
`kernels/dsa/streamindex_topk.py`. There, `topk_backend="auto"` uses a SparseCore kernel on
generation 6 and later when a score row holds at least 8,192 entries and fits SparseCore VMEM
(262,144 entries on v6e). Every other row takes an exact XLA top-k.

The engine keeps FP8 where its fused MoE takes it and dequantizes other layers to BF16
(`dequant_fp8_layers` in `utils/weight_utils.py`), so HBM holds between 703.7 and 1,403 GiB. A
`v5p-64`, 32 chips and 3,040 GiB, holds either end. The measured run below peaked at 77.46 GiB a
chip.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in `/dev/shm`, which
defaults to half the host's RAM. That host has 440 GB of RAM, less than this 703.7 GiB checkpoint,
so the weights can't live there. Each of the 8 hosts in a `v5p-64` reads it from a GCS bucket
through gcsfuse, as [Larger models](../README.md#larger-models) sets up.

## Capture

`glm5_moe.py` has no `layers_to_capture` hook. Apply both patches, in order:

1. [`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) for the engine flag and the
   output reshape.
2. [`upstream/glm5-capture-hook.patch`](../upstream/glm5-capture-hook.patch) for the hook.

`GlmMoeDsaForCausalLM` subclasses `Glm5ForCausalLM` and runs the same inner model, so the one hook
covers it.

At `--tp-size 32` two more patches go on top, in order:

3. [`upstream/glm5-tp-sharding.patch`](../upstream/glm5-tp-sharding.patch), since the shared
   expert's 16 FP8 input blocks don't tile 32 chips.
4. [`upstream/glm5-fp8-accumulate.patch`](../upstream/glm5-fp8-accumulate.patch), which keeps the
   FP8 matmul's accumulator, its block scales and the row-parallel reductions in float32.

With both, each FP8 linear takes BF16 activations and the FP8 weights times their float32 block
scales, sums in float32 and rounds to BF16 once. That's the float32 reference's product with BF16
inputs and a BF16 output. transformers' BF16 forward also rounds each dequantized weight to BF16.
The second capture check on the `v5p-64` (2026-09-27 08:48Z) ran without the fourth patch and
failed layers 10 to 13 at 2.16 to 2.75 times the BF16 floor.

The passing check and the measure [below](#measured-on-a-v5p-64) ran with all four patches. The
CPU evidence for the fourth is in
[`upstream/README.md`](../upstream/README.md#glm5-fp8-accumulatepatch).

[`scripts/capture_activations.py`](../scripts/capture_activations.py) pins the flash attention
backend by default. Pass `--engine-arg attention_backend=dsa_sparse` for the sparse indexer.
Prompt tokens still go through dense prefill unless `DSA_PREFILL_SPARSE=1`.

78 layers at 6,144 dim is 936 KiB per token of host traffic. That's every slot.
`capture_activations.py` hands `--layers` to the engine as `--return-hidden-states-layers`, and
one kept slot moves 12 KiB per token.

## Measured on a v5p-64

A `v5p-64` Spot slice (32 chips, 8 hosts) in us-east5-a, 2026-09-27. sglang-jax eb061d8 with
`sglang-jax-877.patch`, the steering patches, `multihost-hidden-states.patch`,
`glm5-capture-hook.patch`, `glm5-tp-sharding.patch` and `glm5-fp8-accumulate.patch`, built by the
`glm5.3` row of `scripts/multihost_run.sh` (engine args `attention_backend=dsa_sparse ep_size=32`,
`mem_fraction_static=0.8`, `--tp-size 32`). Every host read the weights from a GCS bucket in
us-east5 through gcsfuse.

**Capture check**, `scripts/check_capture.py` against the float32 CPU reference
`refs/ref-glm5.3.npz` with the BF16 floor gate (median per-token error at most 2x the transformers
BF16 forward's own), 11:11 to 11:29Z:

| Prompt | Layers | Result | Worst Pearson | Control (each slot against the next layer) |
|---|---|---|---|---|
| 440 tokens | 78 | all pass | 0.9910 (layer 74) | 3.79x, detected |
| 1,322 tokens, split across prefill passes | 78 | all pass | 0.9937 (layer 77) | 3.23x, detected |

Two earlier checks on the same slice failed and led to the two GLM patches: the first died in the
extend precompile on a shard_map spec mismatch in the shared expert's FP8 `down_proj` at tp 32
(`glm5-tp-sharding.patch`), and the second passed 74 of 78 layers with layers 10 to 13 at 2.16 to
2.75x the floor, from the block-wise FP8 kernel summing in BF16 (`glm5-fp8-accumulate.patch`).

**Throughput and HBM**, `scripts/measure_model.sh` with `PROMPTS=1000` (440-token prompts), capture
slot 39, with both GLM patches (the build that passed the check), 2026-09-28 02:55 to 03:18Z:

| | Tokens/s | Window | Peak HBM per chip (host 0's 4 chips) |
|---|---|---|---|
| Capture off | 2,685.8 | 30 batches, first 2 discarded, 105,600 tokens in 39.32 s | 77.29 of 95.73 GiB (423 samples) |
| Capture on, slot 39 | 2,417.1 | steady window 134.72 to 303.65 s, 408,320 tokens | 77.46 of 95.73 GiB (234 samples) |

Capture costs 1.11x (2,685.8 over 2,417.1), and the wire carries 12,288 bytes a token (one
6,144-wide BF16 slot) at 29.7 MB/s. The first measure on 2026-09-27, before
`glm5-fp8-accumulate.patch`, read 2,782.4 off and 2,487.9 on at the same peak HBM; this run reads
3.5% and 2.8% lower.

To reproduce, stage the checkpoint and its reference as [Larger models](../README.md#larger-models)
describes, then run from the repo root:

```bash
PROJECT=your-project BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE glm5.3 setup check measure
```

Notes: nobody has replayed that block as written from a clean start. The runs above came from the
same script and row. Each measure is one run, so the accumulator's share of the gap isn't settled.

