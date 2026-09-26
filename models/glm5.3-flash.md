# GLM-5.3-Flash

`zai-org/GLM-5.3-Flash`

**Status: not runnable.** `sglang-jax` has no implementation of this architecture, and nothing
here serves it. This page scopes the port.

| | |
|---|---|
| Parameters | 320B total / 18B active |
| Layers | 45: 34 KDA + 11 sparse MLA |
| Hidden dim | 4,096 |
| Attention | KDA linear attention, NoPE sparse MLA, mHC |
| Weights | FP8 e4m3, 305.8 GiB on disk in 62 files; 598.5 GiB at BF16 |

The checkpoint names `Glm5NextForConditionalGeneration`, with `model_type` `glm5_next`, and
nothing in `sglang-jax` registers it. `models/glm5_moe.py` serves GLM-5 and [GLM-5.3](glm5.3.md),
which carry MLA with a sparse indexer and no KDA, so neither it nor
[`upstream/glm5-capture-hook.patch`](../upstream/glm5-capture-hook.patch) covers this model.

## What a port needs

Every part exists upstream; a port assembles them.

| Part | Where it lives |
|---|---|
| KDA linear attention | `models/kimi_linear.py`, which [`upstream/models/kimi-k3-model.patch`](../upstream/models/kimi-k3-model.patch) extends |
| Sparse MLA | the DSA path in `models/glm5_moe.py` |
| mHC residual | `layers/hyperconnection.py`, with a kernel in `kernels/mhc/` |
| Splitting a long sequence across chips | [`scan/kda.py`](../scan/kda.py) |

**Build it for v6e.** At `eb061d8` the mHC kernel in `kernels/mhc/` has a tile schedule for v6e
alone. On v5p and TPU7x it raises `ValueError`. The kernel needs a v5p entry in
`kernels/mhc/tune.py` to run on v5p. The KDA backend doesn't check the chip generation. The BF16
weights, 598.5 GiB, need a `v6e-32` (32 chips, 1,000 GiB at the
31.25 GiB a chip that `tpu-info` reports) before KV cache and capture buffers.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in `/dev/shm`, which
defaults to half the host's RAM. That host has 440 GB of RAM, so its default `/dev/shm` holds
neither the 305.8 GiB FP8 checkpoint nor the 598.5 GiB of `zai-org/GLM-5.3-Flash-BF16`. Every
host of the slice reads from the checkpoint. No run here has loaded a checkpoint of either size.

45 layers at 4,096 dim is 360 KiB per token of host traffic once it captures. That's every slot.
`capture_activations.py` hands `--layers` to the engine as `--return-hidden-states-layers`, and
one kept slot moves 8 KiB per token.

## Notes

`qk_rope_head_dim` is 0, so there's no RoPE on the attention path.

`mhc: true` with `hc_sinkhorn_iters: 20` puts a 20-iteration loop inside every layer. Cold compile
takes longer than the layer count suggests.

No KDA model has run on a chip here, and nothing here has run on v6e.
