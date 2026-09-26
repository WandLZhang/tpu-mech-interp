# GLM-5.3

`zai-org/GLM-5.3`

**Status: untested.** `sglang-jax` serves it upstream, and nothing here has run it, on a chip or
on CPU. This page sizes it for a `v5p-64` across 8 hosts, and nothing here has run across hosts.

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

`sglang-jax` serves it as shipped. `models/glm5_moe.py` registers `GlmMoeDsaForCausalLM`, the
architecture this checkpoint names.

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

Untested: no run here has started this line.

The indexer needs no particular TPU generation. By default it picks its top 2,048 at decode with
an approximate XLA top-k, `approx_max_k` at `recall_target=0.70` in `kernels/dsa/ref.py`. Prefill
runs dense MLA unless `DSA_PREFILL_SPARSE=1`, and so does a batch that mixes prefill with decode.
Set `DSA_INDEXER_KERNEL=1` in the environment to move the decode pick to
`kernels/dsa/streamindex_topk.py`. There, `topk_backend="auto"` uses a SparseCore kernel on
generation 6 and later when a score row holds at least 8,192 entries and fits SparseCore VMEM
(262,144 entries on v6e). Every other row takes an exact XLA top-k.

The engine keeps FP8 where its fused MoE takes it and dequantizes other layers to BF16
(`dequant_fp8_layers` in `utils/weight_utils.py`), so HBM holds between 703.7 and 1,403 GiB. A
`v5p-64`, 32 chips and 3,040 GiB, holds either end. Measure it on the first run.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in `/dev/shm`, which
defaults to half the host's RAM. That host has 440 GB of RAM, less than this 703.7 GiB checkpoint,
so the weights can't live there. Each of the 8 hosts in a `v5p-64` reads from the checkpoint. No
run here has loaded a checkpoint this size.

## Capture

`glm5_moe.py` has no `layers_to_capture` hook. Apply both patches, in order:

1. [`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) for the engine flag and the
   output reshape.
2. [`upstream/glm5-capture-hook.patch`](../upstream/glm5-capture-hook.patch) for the hook.

`GlmMoeDsaForCausalLM` subclasses `Glm5ForCausalLM` and runs the same inner model, so the one hook
covers it.

[`scripts/capture_activations.py`](../scripts/capture_activations.py) pins the flash attention
backend by default. Pass `--engine-arg attention_backend=dsa_sparse` for the sparse indexer.
Prompt tokens still go through dense prefill unless `DSA_PREFILL_SPARSE=1`.

78 layers at 6,144 dim is 936 KiB per token of host traffic. That's every slot.
`capture_activations.py` hands `--layers` to the engine as `--return-hidden-states-layers`, and
one kept slot moves 12 KiB per token.
