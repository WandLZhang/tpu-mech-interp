# Nemotron 3 Ultra 550B-A55B

`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`

**Status: measured on TPU**, `v5p-64`, 2026-09-28, with the layer filter.
[Capture check on a `v5p-64`](#capture-check-on-a-v5p-64) has one capture check, which passes on
all 108 layers, and one measure run, on a Spot slice in us-east5-a across 8 hosts.
[Super](nemotron3-super.md) serves on a chip through the same model file, and
`upstream/models/test_nemotron_h_model.py` checks that file on CPU.

| | |
|---|---|
| Parameters | 560.5B total / 55B active, 549.3B in the served stack |
| Layers | 108: 48 Mamba-2 + 48 LatentMoE + 12 attention, plus a 2-block MTP head |
| Hidden dim | 8,192 |
| Attention | GQA, 64 query / 2 KV heads, head dim 128, no position encoding |
| Mamba-2 | 256 heads, head dim 64, state 128, 8 groups, conv kernel 4, chunk 128 |
| MoE | 512 routed experts top-22, 1 shared, latent dim 2,048 |
| Context | 262,144 |
| Weights | 1,023.2 GiB BF16 served, 1,044.1 GiB on disk |

The checkpoint holds 1,044.1 GiB, which `model.safetensors.index.json` reports as
1,121,049,257,984 bytes. The MTP head is 20.9 GiB of that and the engine doesn't build it, so
1,023.2 GiB lands in HBM.

Same architecture as [Super](nemotron3-super.md), twice the width. Every difference is a
number: 108 layers against 88, 8,192 hidden against 4,096, 256 Mamba-2 heads against 128, a
2,048 latent against 1,024. One model file serves both.

Ultra spells its layer field `layers_block_type`, a list. Super spells the same field
`hybrid_override_pattern`, a string. Ultra ships no `num_hidden_layers` at all, so the layer
count is the length of that list and nothing else.

## Layer plan

From `scan/`:

```python
from nemotron3_layers import NEMOTRON_3_ULTRA_CONFIG, block_types, cache_plan, scan_groups, unit_runs

scan_groups(NEMOTRON_3_ULTRA_CONFIG)               # 108 runs, every one a single layer
unit_runs(NEMOTRON_3_ULTRA_CONFIG)                 # 25 runs
cache_plan(block_types(NEMOTRON_3_ULTRA_CONFIG))   # 48 recurrent, 12 paged, 48 stateless
```

48 units in two shapes, `(mamba, moe)` and `(mamba, attention, moe)`, which fold into 25 runs.
Attention sits at layers 7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89 and 98, so the gaps run 7, 9, 9
and repeat. The period changes seven times down the stack, which is why one scan over all 108
layers builds the wrong model.

The 12 attention layers carry no rotary table and no learned position bias. Order reaches them
through the 48 Mamba-2 layers below. `config.json` ships `rope_theta` and
`partial_rotary_factor`, the reference config class declares neither and the reference model reads
neither, and the port does the same.

## State

The 48 Mamba-2 layers hold two slots each and neither grows with the sequence.

| Slot | Shape | Per layer | 48 layers |
|---|---|---|---|
| SSM state | `[256 heads, 64 head dim, 128 state]` | 8 MiB | 384 MiB |
| conv state | `[18,432 channels, 3 tokens]` | 216 KiB | 10.1 MiB |

`conv_dim` is `mamba_num_heads * mamba_head_dim + 2 * n_groups * ssm_state_size`, so the conv
carries `B` and `C` alongside the SSM input. The conv slot holds `conv_kernel - 1` tokens: the
current token arrives with the request, and the window needs the three before it.

Both slots are float32, because `mamba_ssm_cache_dtype` is float32 while the weights ship BF16.
That's 394 MiB of recurrent state per sequence, whatever the context length.

`RecurrentStatePool` allocated the SSM state square, `head_dim` by `head_dim`. KDA and GDN run
those two widths equal so it reads as one number there, but Mamba-2 runs 64 by 128 and a square
slot holds half the state. The model patch gives the second axis its own field, `head_k_dim`,
which leaves both existing callers alone.

The attention layers hold KV that grows with the sequence. 2 KV heads at head dim 128 is 1 KiB
per token per layer in BF16. The engine builds the 12 attention blocks of the main stack and no
more, so that's 12 KiB per token, or 3.0 GiB per sequence at the full 262,144 context.

## Serve

Engine: `sglang-jax` with
[`upstream/models/nemotron3-model.patch`](../upstream/models/nemotron3-model.patch), which adds
`python/sgl_jax/srt/models/nemotron_h.py`, the config that resolves the layer stack, and the
Mamba-2 backend that owns the recurrent state. Slice: `v5p-64`, 32 chips.

Ultra serves on a `v5p-64` ([below](#capture-check-on-a-v5p-64)), and nothing in this model needs
v6e. The Mamba-2 mixer is plain XLA, built from `cumsum`, `einsum`, `exp` and a `lax.scan`, with
no Pallas kernel and no generation check.

Budget the slice per chip, not in aggregate. At `--tp-size 32` each chip stores one of the 2 KV
heads, the same as [Super](nemotron3-super.md), so a sequence pays half its KV on every chip:
1.5 GiB at the full context.

| | `v5p-64` |
|---|---|
| Chips | 32 |
| HBM per chip | 95 GiB |
| Weights per chip | 32.0 GiB |
| Free per chip | 63.0 GiB |
| Full-context sequences | 41 |

The recurrent state shards with the Mamba-2 heads, so it adds 12.3 MiB per chip per sequence
against the 1.5 GiB of KV. Capture buffers come out of the same free space.

Notes: the table is computed from the weights and the KV and state sizes.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in `/dev/shm`, which
defaults to half the host's RAM. That host has 440 GB of RAM, less than this 1,044.1 GiB
checkpoint, so the weights can't live there. Each of the 8 hosts in a `v5p-64` reads it from a
GCS bucket through gcsfuse, as [Across hosts](../README.md#across-hosts) sets up.

```bash
python3 -m sgl_jax.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --dtype bfloat16 \
  --tp-size 32 \
  --disable-radix-cache \
  --enable-return-hidden-states
```

Untested as written. Across 8 hosts every host runs the engine with `--nnodes 8`, its own
`--node-rank` and host 0's `--dist-init-addr`. For this repo's scripts,
`scripts/multihost_exec.sh` sets the same three settings on every host, and the runs
[below](#capture-check-on-a-v5p-64) started the engine that way through `scripts/multihost_run.sh`.

A hybrid recurrent model needs `--disable-radix-cache` or `--enable-unified-radix-tree` to start.
[Super](nemotron3-super.md#serve) covers the prefix-caching flags.

Confirm the chip count first, on all 8 hosts at once, as
[gpt-oss-120b](gpt-oss-120b.md#scaling-out-to-v5p-64) describes. It reported 32 on every host on
2026-09-27; `scripts/multihost_setup.sh` runs this check.

```bash
python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count())"
```

Two counts don't divide over 32 chips. Each chip stores one of the 2 KV heads. The 8 Mamba-2
groups shard 8 ways and replicate four times, so `B` and `C` ride into the state path replicated
and each device slices the groups its heads read. The gated norm runs those same 8 groups over a
2,048 channel group, so a 32-way tensor axis splits one group over four shards and `GroupRMSNorm`
sums it with one all-reduce of a `[tokens, 8]` array. The 64 query heads, the 256 Mamba-2 heads,
the 18,432 conv channels, the 2,048 latent and the 512 routed experts all shard cleanly.

### What LatentMoE asks of the mesh

Top-22 of 512 is a far wider all-to-all than the top-8 the existing MoE layers assume. Two things
make it affordable.

The experts run in the latent. The router reads the full 8,192 wide token, then `fc1_latent_proj`
drops it to 2,048 and `fc2_latent_proj` lifts the result back, so the dispatch carries 22 copies of
a 2,048 wide token, 88 KiB in BF16, against the 352 KiB that top-22 at full width would cost. The
reference runs the same order: `NemotronHMoE.forward` calls the gate on `hidden_states`, then
`fc1_latent_proj`, then the experts. The port matches it, `GateLogit(input_size=hidden_size)` and
then the projection.

The expert axis wants its own mesh axis. `EPMoE` builds an `(expert, tensor)` mesh from `ep_size`
and shards the expert dimension across it, so at `--ep-size 32` each chip owns 16 whole experts.
At `ep_size=1` the same bytes land on each chip, sliced the other way: every chip holds a 160-wide
sliver of all 512 experts and reduces the output across the mesh. Both fit. The wide routing is
what makes the second layout expensive, because a 2,048 by 160 GEMM per expert leaves the MXU
mostly idle and the reduction runs on every token rather than on the 22 that were routed.

The model refuses the expert-placement flags `--ep-dispatch-algorithm`, `--init-expert-location`
and `--ep-num-redundant-experts` before it builds. It loads the routed experts in checkpoint order
and builds no expert-location metadata, so those flags would change nothing.

The shared expert runs at `hidden_size` on every token, so it shards like a dense MLP and needs
no dispatch.

The experts aren't gated: one `up_proj`, `mlp_hidden_act`, one `down_proj`. `EPMoE` grew a `gated`
flag and a `relu2` activation for that, which halves the expert GEMMs. `mlp_hidden_act` is `relu2`
in both published configs, and the port reads the field rather than assuming it, so a checkpoint
in this family that ships a different activation is refused at build time.

## Capture

[`upstream/models/nemotron3-capture-hook.patch`](../upstream/models/nemotron3-capture-hook.patch)
adds the `layers_to_capture` hook on top of the model patch, in the four parts
[`upstream/capture-hooks/README.md`](../upstream/capture-hooks/README.md) describes. Apply
[`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) for the flag and the output
reshape.

The hook takes the ordinary None-residual form, because the stack carries `hidden_states` and
`residual` as a pair and `residual` is None on layer 0:

```python
if layer_id in self.layers_to_capture:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )
```

108 layers at 8,192 dim is 1,728 KiB per token of host traffic. At 1,000 tokens/s that's 1.77 GB/s
against the host NIC. That's every slot. `capture_activations.py` hands `--layers` to the engine as
`--return-hidden-states-layers`, and one kept slot moves 16 KiB per token.

The first capture check on the `v5p-64` read 0.93, 1.29, 1.27, 2.77, 2.62, 12.08 and 10.51 times
the BF16 floor at slots 1 to 7. The fault sat in the reference. `check_capture.py --offload-folder`
loaded the checkpoint at BF16 and upcast afterwards, and accelerate's offload index cast each
tensor to BF16 on read. Ultra's router correction bias is float32 at 56.975 to 56.981, and BF16's
step there is 0.25, so the reference routed as if every expert carried a bias of 57.0. The float32
pass and the BF16 floor both lost it; transformers keeps that bias float32 at BF16. The streamed
path now reads float32 checkpoint tensors at float32. On CPU, the same engine on the first 8
layers of the published checkpoint, with `ep_size` equal to `tp_size` and `SGL_PROBE_MOE=dense`,
gives these ratios for prompt 0 on 2026-09-28. `ref8` is `transformers` 5.17 without streaming,
built with `--reference-only --reference-layers 8`.

| Reference | tp | Slots 1 to 7 |
| --- | --- | --- |
| published `ref-nemotron3-ultra.npz` | 1 | 0.81 1.25 1.23 2.76 2.62 12.10 10.54 |
| published `ref-nemotron3-ultra.npz` | 8 | 0.93 1.31 1.27 2.77 2.62 12.05 10.55 |
| published `ref-nemotron3-ultra.npz` | 32 | 0.93 1.31 1.28 2.77 2.62 12.03 10.54 |
| `ref8` | 1 | 0.81 0.88 0.85 0.88 0.86 0.89 0.80 |
| `ref8` | 8 | 0.93 0.95 0.92 0.93 0.92 0.92 0.87 |
| `ref8` | 32 | 0.93 0.95 0.92 0.93 0.92 0.92 0.87 |

The streamed path then got the same test on the same 8 layers. The tp 32 capture above, saved as
`cap-tp32-dense.npz`, was gated against two streamed 8-layer references built on 2026-09-28, one
before the fix and one after:

| Streamed reference | Worst ratio, prompt 0 / 1 | Control | Result |
| --- | --- | --- | --- |
| before the fix | 12.19 / 12.02 | 8.6x / 8.8x | FAILED at slots 4 to 7 |
| after the fix | 0.95 / 0.95 | 55.6x / 51.5x | PASS |

The published reference was rebuilt with the fix on a `c4-highcpu-96` in `us-east5-a`, on
2026-09-28 in 2 hours 20 minutes at a peak RSS of 147 GiB. The new npz replaced
`refs/ref-nemotron3-ultra.npz` in the bucket; the old one is
`refs/ref-nemotron3-ultra.pre-f32fix.npz`.

### Capture check on a `v5p-64`

`scripts/multihost_run.sh NODE us-east5-a nemotron3-ultra setup check` on a `v5p-64` Spot slice
(32 chips, 8 hosts) in us-east5-a, tp 32, ep 32, all 108 layers, 2026-09-28 06:34 to 06:55Z,
against the rebuilt reference, BF16 floor gate:

| Prompt | Layers | Result | Worst ratio to the floor | Worst Pearson | Control |
|---|---|---|---|---|---|
| 441 tokens | 108 | all pass | 1.04 (layer 72) | 0.9988 (layer 104) | 3.58x, detected |
| 1,322 tokens, split across prefill passes | 108 | all pass | 1.00 (layer 72) | 0.9990 (layer 59) | 4.47x, detected |

The extend graph precompiled in 2.4 minutes and the decode graph in 1.4.

**Throughput and HBM**, `scripts/multihost_run.sh NODE us-east5-a nemotron3-ultra measure`
(`scripts/measure_model.sh` with `PROMPTS=1000`, 440-token prompts), capture slot 54, tp 32, ep 32,
2026-09-28 06:57 to 07:25Z:

| | Tokens/s | Window | Peak HBM per chip (host 0's 4 chips) |
|---|---|---|---|
| Capture off | 3,095.3 | 30 batches, first 2 discarded, 105,600 tokens in 34.12 s | 78.01 of 95.73 GiB (448 samples) |
| Capture on, slot 54 | 2,218.2 | steady window 258.69 to 441.18 s, 404,800 tokens | 78.39 of 95.73 GiB (369 samples) |

Capture costs 1.40x (3,095.3 over 2,218.2), and the wire carries 16,384 bytes a token (one
8,192-wide BF16 slot) at 36.3 MB/s.

To reproduce, stage the checkpoint and its reference as [Across hosts](../README.md#across-hosts)
describes, then run from the repo root:

```bash
BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE nemotron3-ultra setup check measure
```

Notes: nobody has replayed that block as written from a clean start. The runs above came from the
same script and row.

## Notes

The MTP head isn't in the served stack. Both MoE sizes ship one, `(attention, moe)` with its own
`eh_proj`, `enorm` and `hnorm`, under the `mtp.` prefix in the checkpoint. It's a draft model for
speculative decoding and belongs in its own module, the way `mimo_mtp.py` is separate from
`mimo.py`. `NemotronHConfig.full_attention_layer_ids` reads the main stack, so the KV budget covers
the 12 attention blocks the engine builds. Hand `cache_plan` the config dict instead of the
resolved list and it counts the head's attention block too, which reserves an extra 1 KiB per token
per sequence that nothing allocates.

Tensor parallelism reaches 32 on both sizes. The gated norm was the ceiling: `GroupRMSNorm` needed
the tensor axis to divide the 8 RMS groups, and it now sums a group that straddles shards. The
recurrent pool was the other one: it asserted that `num_k_heads`, which carries the 8 Mamba-2
groups, divides an axis the recurrent buffer never shards, and it now checks the two segments of
the conv buffer that do shard.

`scan/test_nemotron3_layers.py` checks the layer plan against the published fields of Super, Ultra
and Nano on CPU. `upstream/models/test_nemotron_h_model.py` checks the model itself against the
HuggingFace implementation.
