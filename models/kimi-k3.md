# Kimi K3

`moonshotai/Kimi-K3`

**Status: captured on TPU**, `v5p-64`, 2026-09-27; the capture check passes, prefill runs at 0.6
tokens/s; not measured. The runs kept the routed experts MXFP4 on the chip at
`--tp-size 32 --dp-size 1 --ep-size 32`, on a Spot slice in us-east5-a. See
[Measured on a v5p-64](#measured-on-a-v5p-64). Throughput and HBM wait on the prefill work in
[the roadmap](../docs/roadmap.md).

| | |
|---|---|
| Parameters | 2.78T total / 104B active |
| Layers | 93: 69 KDA + 24 gated MLA, NoPE |
| Hidden dim | 7,168 |
| Attention | 96 KDA heads at 128. MLA: 96 heads, q-LoRA 1,536, kv-LoRA 512, 128 nope + 64 rope, sigmoid output gate |
| MLP | layer 0 dense at 33,792. 92 routed layers: 896 experts top-16 at 3,072, plus 2 shared |
| Expert latent | 3,584, half the residual width |
| Activation | `situ`, beta 4.0, linear beta 25.0 |
| Residual | learned softmax mixture, block size 12 |
| Per-layer state | 3 short convolutions, kernel 4, on the KDA layers |
| Context | 1,048,576 |
| Weights | 5,178 GiB BF16, 1,453 GiB with MXFP4 routed experts |

The checkpoint also carries a 27-layer vision tower and a projector under
`vision_tower.` and `mm_projector.`. The language model sits under
`language_model.` and is what this page covers.

`config.json` matches that split. Every language-model field sits under
`text_config`, and the top level holds `architectures`,
`["KimiK3ForConditionalGeneration"]`, `model_type`, `kimi_k3`, and the token
ids. Both names matter to the engine: the registry resolves the architecture
string, and `AutoConfig` resolves the model type.

## Serve

Engine: `sglang-jax` with
[`upstream/models/kimi-k3-model.patch`](../upstream/models/kimi-k3-model.patch).

Nothing on the engine's KDA path at `eb061d8` reads the chip generation.
Prefill picks one of two Pallas kernels from `SGLANG_JAX_KDA_PREFILL_KERNEL`,
`mega` by default or `chunked`, and from how many requests share each 64-token
tile. Decode runs `naive_recurrent_kda`, a plain JAX recurrence, on every TPU
generation. Every run on the `v5p-64` [below](#measured-on-a-v5p-64) took the
default, and nobody has timed the two prefill kernels apart yet.

```bash
python3 -m sgl_jax.launch_server \
  --model-path moonshotai/Kimi-K3 \
  --dtype bfloat16 \
  --tp-size 256 \
  --dp-size 8 \
  --ep-size 8 \
  --page-size 128 \
  --disable-radix-cache \
  --recurrent-state-memory-ratio 0.5
```

Untested: no run here has started this line or used a `v6e-256`.

Capture needs `--dp-size 1`: with more than one data-parallel rank the capture patch would hand a
request other requests' rows, so it refuses to start. The head axis below forces data parallel at
256 chips. One `v5p-64` with the routed experts resident in MXFP4 runs at `--dp-size 1`, so that's
the slice to capture from:

```bash
python3 -m sgl_jax.launch_server \
  --model-path moonshotai/Kimi-K3 \
  --dtype bfloat16 \
  --tp-size 32 \
  --dp-size 1 \
  --ep-size 32 \
  --page-size 128 \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --enable-return-hidden-states
```

Untested as written. The runs [below](#measured-on-a-v5p-64) started the engine with these
settings across 8 hosts through `scripts/multihost_run.sh`, which gives the engine 8 nodes, each
host's rank and host 0's address through `scripts/multihost_exec.sh`. The weights take 48.35 GiB
of each chip's 95.73 GiB (check 12 of [Test](#test), shapes only, no weights,
2026-09-26). `--ep-size 32` puts 28 whole experts on each chip. `EPMoE` lays the 32 devices out as
`ep_size x (32 / ep_size)`, and at `--ep-size 8` each chip holds 112 experts cut four ways plus
all of `wo`'s scales for them, which takes 50.82 GiB. `--ep-size 32` needs no reduction over the
tensor axis after the experts either.

Notes: the MXFP4 checkpoint turns the resident path on by default. The launch line keeps the
default `--recurrent-state-memory-ratio`; split the 28.24 GiB the reservation leaves between the
two pools from the workload, as [What the split costs](#two-layer-types-two-pools) describes.

`--disable-radix-cache` is mandatory. Any model with a
`linear_recurrent_config` asserts at startup for one of
`--disable-radix-cache` or `--enable-unified-radix-tree`, and both default off.
The unified radix tree caches prefixes across the recurrent pool as well, so
take it instead when the workload repeats prompts. At a page size above 1 it
also needs `--enable-recurrent-extra-buffer`.

The MLA layers need a `--page-size` above 1. `MLAAttentionBackend` asserts it
at startup, and the default is 1. 128 matches the two v6e MLA deploy lines in
the engine's `kernels/mla/v2/tuned_block_sizes.py`.

Confirm the chip count first, on every host at once, as
[gpt-oss-120b](gpt-oss-120b.md#scaling-out-to-v5p-64) describes. Untested: no run here has used a
`v6e-256`. It reports 256.

```bash
python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count())"
```

**The head axis stops at 32.** Both layer types run 96 heads, and attention
splits them over `tp_size / dp_size` ways, so that ratio has to divide 96. The
divisors that also suit a slice are 8, 16 and 32. `--tp-size 256 --dp-size 8`
puts it at 32, and the rest of the mesh goes to data parallel. `--ep-size 8`
splits the 896 experts 112 to a shard. On a `v5p-64`, `--tp-size 32
--dp-size 1` puts the head axis at 32 with no data parallel at all.

### Slice size

A v5p chip carries 95 GiB, so `v5p-64` holds 3,040 GiB. BF16 weights are
5,178 GiB. The engine shards a model within one slice, so BF16 takes a
`v5p-128`: 64 chips, 6,080 GiB.

One slice holds the model when the routed experts stay MXFP4: 1,453 GiB of
weights against 3,040 GiB, with 1,587 GiB left. `EPMoE` contracts through
`gmm`, which reads BF16 operands. So the patch keeps the packed codes and
scales in HBM and decodes each layer's experts to BF16 inside the MoE forward,
one stack at a time, just before its `gmm`. [Weights](#weights) describes both
expert paths.

Check 12 of [Test](#test) builds the full model under `nnx.eval_shape` on 32
simulated devices and sums one chip's bytes from the shapes, dtypes and
shardings (c3d-highcpu-90, us-central1-a, 2026-09-26; projected, no weights
loaded):

| `--tp-size 32 --dp-size 1` | Routed experts | Everything else | Weights a chip |
|---|---|---|---|
| `--ep-size 32`, MXFP4 resident | 42.10 GiB | 6.25 GiB | 48.35 GiB |
| `--ep-size 8`, MXFP4 resident | 44.57 GiB | 6.25 GiB | 50.82 GiB |
| `--ep-size 32`, decoded at load | 158.48 GiB | 6.25 GiB | 164.73 GiB |

At `--mem-fraction-static 0.8` the engine reserves 76.58 GiB a chip for
weights and the KV and state pools, so `--ep-size 32` leaves 28.24 GiB a chip
for the two pools. The decoded experts are a temporary outside that
reservation, in the 19.15 GiB left for activations. One MoE layer's decoded
stack takes 0.57 GiB a chip, and all three stacks of a layer would take
1.72 GiB. Decoded at load, the weights need 164.73 GiB a chip and don't fit.

The first extend precompile on a `v5p-64` (us-east5-a, 2026-09-27) stopped with
`RESOURCE_EXHAUSTED` and asked for 196.32 GB of HLO temporaries a chip. The
decode barriered the packed codes and left the E8M0 scales outside, so XLA ran
every layer's scale expansion at the start of the program and held it until
that layer's `gmm`. Each expansion is a per-element gather index and an f32
factor the size of the decoded stack, 1,176 MiB each a chip. The scales now go
through the barrier with the codes, and a stack whose rows start on a group
boundary reads one factor per 32-element group.

A TPU cross-compile of the full published config for a 32-chip `v5p:2x4x4`,
through libtpu on a c3d-highcpu-90 (us-central1-a, 2026-09-27), reproduces the
failure and the fix at the check's precompile shape: 6 requests, 1,024 tokens,
`--page-size 128`, `--ep-size 32`, `--attention-backend fa`, and a page table of
5,899,776 slots. The page table doesn't drive the temporaries: a four-layer
compile asks for 13.70 GiB at the default context and at `--context-length 4096`
alike, so the engine flags stay as they were.

| One chip | Before | After |
|---|---|---|
| Weights | 48.35 GiB | 48.35 GiB |
| MLA and KDA state pools | 28.25 GiB | 28.25 GiB |
| HLO temporaries | 201.83 GB, `RESOURCE_EXHAUSTED` | 5.63 GiB |
| Total against 95.73 GiB | | 82.23 GiB |

Notes: compiled, not run on a chip. The cross-compile builds the tree the chip
ran, capture hook included, and asks for 201.83 GB where the chip asked for
196.32 GB.

A v6e chip carries 32 GiB, so `v6e-256` holds 8,192 GiB. `--dp-size 8` cuts
the slice into 8 data groups of 32 chips. Each group holds its own copy of the
106 GiB of non-expert weights, except `lm_head`, which splits over all 256
chips. `EPMoE` splits the routed experts over all 256 chips. The router gates,
`q_a_proj`, `kv_a_proj` and `f_a_proj` sit whole on every chip. `GateLogit`
stores the router in float32. Weights take about 26 GiB a chip, 6,656 GiB
across the slice, and leave about 6 GiB a chip.

The MLA cache has no head axis, so every chip in a group holds the group's
whole cache at 30 KiB a token. A 1,048,576-token request needs 30 GiB a chip.
With every free byte spent on cache, one request tops out at about 210,000
tokens. At the default `--mem-fraction-static` of 0.88 and the launch line's
`--recurrent-state-memory-ratio 0.5`, the KV pool holds about 50,000 tokens a
group.

| | Weights | v5p slice | v6e slice |
|---|---|---|---|
| BF16 | 5,178 GiB | `v5p-128` | `v6e-256` |
| MXFP4 routed experts | 1,453 GiB | `v5p-64` | `v6e-256` |
| MXFP4 resident | 48.35 GiB a chip | `v5p-64` at tp 32, dp 1, capture works | |

Notes: the resident row comes from check 12 (shapes only, 2026-09-26), and the v6e figures are
computed. Kimi K3 served on a `v5p-64` that way on 2026-09-27 ([below](#measured-on-a-v5p-64)).

The checkpoint, vision tower included, holds 1,453.7 GiB on disk, which
`model.safetensors.index.json` reports as 1,560,860,324,864 bytes. A `v5p-8`
host has 440 GB of RAM. `/dev/shm` takes half of it by default. So the
checkpoint can't sit in `/dev/shm` the way the
[measured runs](nemotron3-super.md#measured) keep their weights. Every host of
the slice reads it from a GCS bucket through gcsfuse at load, as
[Across hosts](../README.md#across-hosts) sets up.

## Architecture

### Two layer types, two pools

`linear_attn_config` lists `kda_layers` and `full_attn_layers` 1-based, and
`is_kda_layer` reads them as `layer_idx + 1`. Layer 0 is KDA and layer 92
isn't. The cycle is three KDA layers then one full-attention layer, which gives
69 and 24.

The scheduler splits the memory the same way. `HybridLinearKVPool` sizes its
inner pool to `len(full_attention_layer_ids)`, so only the 24 MLA layers pay
KV. The 69 KDA layers hold their state in `RecurrentStatePool`.

**What the split costs.** The two pools have opposite shapes. The server
divides each chip's free HBM between them once, at startup, through
`--recurrent-state-memory-ratio`. The table gives one chip's share at the
launch line's 32-way head split.

| | Per request | Per token |
|---|---|---|
| KDA recurrent state, float32 | 12.9 MiB | 0 |
| KDA short-conv state, bfloat16 | 0.45 MiB | 0 |
| MLA KV cache, bfloat16 | 0 | 30 KiB |

Both KDA states split 32 ways, the recurrent state by head and the short-conv
state by channel. The MLA cache doesn't split. The pool pads the 64-wide rope
half of each layer to 128, which takes a token from 27 KiB to 30 KiB. One
request's slot in the recurrent pool holds the recurrent state and the
short-conv state. That slot costs what 457 tokens of KV cache cost. So:

* The recurrent pool is a fixed slot count, not a token budget. Each data group
  owns its own slots. 64 concurrent requests, 8 to a group, reserve 107 MiB a
  chip whatever the sequence lengths are, and a request that finds no free slot
  waits.
* The KV half grows with context and the radix cache reuses a shared prefix
  across requests. The recurrent half can't: the state is a running summary of
  the whole prefix, so two requests with the same prompt still hold two copies.
* Raising concurrency and raising context pull on different pools. At 128
  requests the recurrent pool takes 214 MiB a chip and the KV budget shrinks by
  that much. At 30 KiB a token, that's 7,314 tokens of cache in each data group.

Set `--recurrent-state-memory-ratio` from the workload. Long context with few
requests wants it low; short context with high concurrency wants it high.

### Residual stream

`attn_res_block_size` is 12. Kimi K3 doesn't carry one additive residual stream
the length of the stack. Layers 0, 12, 24, 36, 48, 60, 72 and 84 each stash the
running sum and restart it, and every layer reads a learned softmax mixture of
the current sum and every stash:

```
v     = concat([stashes, prefix_sum])        [tokens, blocks + 1, hidden]
score = sum_h (v / rms(v))[..., h] * norm[h] * proj[h]
read  = softmax(score) @ v
```

Each layer runs that twice, once before attention and once before the MLP, with
separate `self_attention_res_*` and `mlp_res_*` weights. The model runs it once
more after the last layer, over 9 vectors, before the final norm.

Two consequences for interpretation. A probe trained on the layer-30 stream
sees the sum since layer 24, not since layer 0. And the vector a layer
conditions on isn't the vector the hook returns; it's one softmax away.

### `situ`

`hidden_act` is `situ`, in every MLP: dense, shared and routed.

```
situ(g) = beta * tanh(g / beta) * sigmoid(g)
out     = situ(gate) * (linear_beta * tanh(up / linear_beta))
```

`beta` is 4.0 and `linear_beta` is 25.0. Both branches are bounded, which SiLU
isn't: the gate factor lands in (-0.270, 4.0) and the linear factor in
(-25, 25). Both run in float32. At beta 4.0 a bfloat16 `tanh(g / beta)` loses
three digits right where the curve bends.

### Latent MoE

`routed_expert_hidden_size` is 3,584, half the residual width. The router
scores the full 7,168 stream, `routed_expert_down_proj` enters the latent, the
experts run 3,584 to 3,072 to 3,584, `routed_expert_norm` normalizes the
combined result and `routed_expert_up_proj` returns to 7,168. The two shared
experts never enter the latent.

Halving the expert width is what pays for 896 experts. One expert matrix is 11M
parameters instead of 22M, and there are 2,688 of them per layer.

Both latent projections are row parallel. 7,168 x 3,584 in BF16 is 49.0 MiB,
98.0 MiB per layer across the pair, and layers 1 to 92 are all MoE. Replicating
them would put 8.80 GiB on every chip, 27.5% of a v6e chip and 2,254 GiB across
a `v6e-256` slice. Sharding the contraction axis costs one all-reduce per
projection instead.

### KDA

The recurrence is the gated delta rule, the same one
[`scan/kda.py`](../scan/kda.py) folds into affine chunk pairs, with
`KIMI_K3` holding the shapes. From `scan/`:

```python
from kda import KIMI_K3
KIMI_K3.state_bytes / 2**20      # 428.6 MiB per sequence: 414.0 recurrent, 14.6 conv windows
KIMI_K3.conv_halo                # 3 tokens from the left neighbor
```

`sglang-jax` already has the KDA layer for Kimi-Linear-48B in
`models/kimi_linear.py`. What transfers: the three projections, the three short
convolutions, `A_log`, `dt_bias`, the beta projection, the `gate_lower_bound` of
-5.0, `o_norm` and the recurrence `RadixLinearAttention` drives.

One thing doesn't. `use_full_rank_gate` is true, so one `g_proj` of
7,168 x 12,288 replaces the rank-128 `g_a_proj` / `g_b_proj` pair. Per KDA
layer that's 88.1M parameters against 2.5M. Across 69 layers, 6.08B against
172M.

`A_log` holds one log decay per head, as in Kimi-Linear-48B, but the checkpoint
pads it. Kimi K3 ships `language_model.model.layers.N.self_attn.A_log` as a
flat float32 `[128]` for 96 heads. In all 69 KDA layers the first 96 values
hold the decays and the last 32 are 0.0. The gate scales

```
exp(A_log[h]) * (g_raw[t, h, c] + dt_bias[h, c])
```

where `h` runs over the 96 heads and `c` over the 128 channels. The K3 repo's
`modeling_kimi_linear.py` declares `A_log` as `torch.empty(self.num_heads)`,
fla's KDA kernels read it at `A_log + i_h`, and vLLM and SGLang keep its first
96 entries. The patch's weight mapping does the same. It keeps the first
`num_heads` entries and holds them at `[1, 1, 96, 1]`, the layout
`kimi_linear.py` holds, so the KDA kernels run unchanged.

Nothing else about Kimi-Linear-48B transfers. Its stack is 27 layers with a
plain pre-norm residual, `silu`, no expert latent and no output gate on MLA.

### MLA

`mla_use_nope` is true, so there's no rotary table. The 64-wide
`qk_rope_head_dim` slice still exists and still carries content; it just never
rotates. `mla_use_output_gate` puts a sigmoid gate on the attention output,
read from the layer input rather than from the attention result, so it lands in
the `_pre_o_proj` hook `DeepseekV3Attention` already exposes and works on both
the absorbed and non-absorbed paths.

## Weights

The checkpoint is mixed precision. `quantization_config.ignore` exempts
attention, the shared experts, the dense layer and `lm_head`, so those load
BF16 through the usual `WeightLoader` path. Only the routed experts ship MXFP4.

| | Parameters | BF16 | MXFP4 |
|---|---|---|---|
| Routed experts | 2.72T | 5,072 GiB | 1,347 GiB |
| Everything else | 56.7B | 106 GiB | 106 GiB |

One MXFP4 tensor is two arrays. `weight_packed` is uint8 with two E2M1 codes
per byte along the input axis, low nibble first. `weight_scale` is uint8 with
one E8M0 exponent per 32 elements, and the scale is `2 ** (code - 127)`. E2M1
is one sign bit, two exponent bits and one mantissa bit, so the 16 codes cover
0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.

`utils/quantization/mxfp4.py` in the patch dequantizes.
`KimiK3ForCausalLM._load_routed_experts` reads the experts through
`jax.make_array_from_callback`, so a host reads only the experts its own device
shard owns. The experts take one of two paths, and the config field
`mxfp4_resident_experts` picks it.

**Resident, the default for an MXFP4 checkpoint.** `EPMoE` holds `wi_0`,
`wi_1` and `wo` as the checkpoint's packed E2M1 codes, uint8
`[experts, in / 2, out]`, and three more parameters hold the E8M0 scales,
uint8 `[experts, in / 32, out]`. Each takes the sharding of its BF16 twin over
the expert and tensor axes. `wo`'s scales stay whole on the tensor axis,
because a tensor shard of `wo`'s packed axis needn't end on a 32-element group.
The callback cuts each device's rectangle off the file and transposes it,
without decoding. Inside the MoE forward, each device decodes its own shard of
one stack to the serving dtype with `dequantize_mxfp4_jax` just before that
stack's `gmm`. An optimization barrier ties each decode, codes and scales both,
to the activation its `gmm` reads, so XLA can't decode every layer up front, and
only one decoded stack is live at a time. The E8M0 lookup makes one factor per
32-element group and broadcasts it over the group.

**Decoded at load.** `--json-model-override-args '{"mxfp4_resident_experts":
false}'` selects it. The callback decodes each device's rectangle on the host
with `dequantize_mxfp4`, and the chip holds BF16 stacks. One BF16 layer is
55 GiB across the three matrices and never exists on one host. A launch
quantization config or `--moe-dp-size` above 1 needs BF16 stacks, so either
one picks this path unless the config asks for residency, and then `EPMoE`
refuses the launch by name.

The two paths make the same float32 products and cast them the same way, so
the served logits and every captured layer agree bit for bit (check 10). Code 0's
scale, 2<sup>-127</sup>, is subnormal, and codes 0 and 1 give products below
2<sup>-126</sup>. XLA flushes subnormals to zero and numpy keeps them, so those
two codes decode to zero on the device. Shards 2 and 93 of the published
checkpoint, layers 1 and 92, hold 924,844,032 E8M0 codes each, from 110 to 123
and from 119 to 123, so no code there reaches that range (read on the VM from
both whole shards, 2026-09-26).

On the decode-at-load path the callback runs once per addressable device, so
it cuts the rectangle before it decodes. `read_mxfp4_block` slices the output
axis by whole checkpoint rows and the input axis whenever the cut lands on a
32-element scale boundary. At `ep=8` and `tp=32` those two edges are the
difference between reading a thirty-second of a matrix and reading all of it 32
times.

The expert index is logical. `EPMoE` sizes its parameter to
`num_physical_experts` when expert-location metadata exists, so
`_load_routed_experts` reads the physical-to-logical map and looks the
checkpoint key up by logical expert. Without it `--ep-num-redundant-experts 16`
asks for `experts.896`, which no shard holds.

With `--ep-dispatch-algorithm` set, `TopK` sends the router's logical ids
through the same map. The ids arrive sharded over the batch, and the stock
gathers in `eplb/expert_location.py` raise on them at the startup precompile,
so the patch gives each gather the ids' sharding.

The router gate, `routed_expert_down_proj` and `routed_expert_up_proj` ship
plain BF16 under `.weight`, so only the three expert matrices per layer take
the MXFP4 path. The router's `e_score_correction_bias` ships F32 and stays
F32, because top-16 picks on score plus bias.

`load_weights` turns on `validate_checkpoint_coverage`, so a language-model
tensor with no mapping stops the load. Three groups are accounted for rather
than mapped: the MXFP4 experts, which `_load_routed_experts` reads itself, the
168 `vision_tower.` and `mm_projector.` tensors in shards 95 and 96, which the
language tower serves nothing for, and every layer past a `--model-layer-nums`
cut.

`--load-format dummy` reads no file and fills the expert stacks with zeros,
split over the expert mesh the way a real load splits them. On the resident
path the scales fill with zeros too.

A `--quantization-config-path` doesn't work on this checkpoint. `ModelConfig`
marks it static because the checkpoint's own `quantization_config` says
compressed-tensors, and the model has no path for a statically quantized
checkpoint, so it refuses the config when it builds.

## What the engine has to resolve first

Two names decide whether the checkpoint loads, and both sit above
`text_config`.

`architectures` is `["KimiK3ForConditionalGeneration"]`.
`get_model_architecture` resolves that list against the registry, which keys on
the class name, so the patch's `EntryClass` carries that name beside
`KimiK3ForCausalLM`. The two run the same code, because the language tower is
the whole of what the model file serves. Neither joins
`InModelMultimodalContract`, which is what marks a model multimodal, so the
vision tower loads nowhere and a text request runs untouched.

`model_type` is `kimi_k3`. `AutoConfig` refuses a checkpoint whose model type
nothing claims, so `configs/kimi_linear.py` adds `KimiK3Config` and
`hf_transformers_utils.py` registers it. It subclasses `KimiLinearConfig` and
copies the text fields onto itself, so `hidden_size`, `linear_attn_config` and
`is_kda_layer` all read the language model. Top-level keys win the merge, which
keeps `architectures` at the outer name rather than the
`KimiLinearForCausalLM` the text tower carries. That inner name belongs to
Kimi-Linear-48B, and resolving through it would serve a different model.

## Capture

`sglang-jax` has no `kimi_k3` model, so per-layer capture needs three patches,
in order:

1. [`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch)
   for the engine flag and the output reshape.
2. [`upstream/models/kimi-k3-model.patch`](../upstream/models/kimi-k3-model.patch)
   for the model.
3. [`upstream/models/kimi-k3-capture-hook.patch`](../upstream/models/kimi-k3-capture-hook.patch)
   for the hook.

93 layers at 7,168 dim is 1,302 KiB per token of host traffic. At 1,000
tokens/s that's 1.33 GB/s against the host NIC. That's every slot.
`capture_activations.py` hands `--layers` to the engine as
`--return-hidden-states-layers`, and one kept slot moves 14 KiB per token.

Without the hook, `--enable-return-hidden-states` refuses to start. After the
weights load and before the model compiles, the model runner raises
`ValueError: KimiK3ForConditionalGeneration has no layers_to_capture hook, so
--enable-return-hidden-states can't return per-layer hidden states`.

The hook appends the prefix sum entering each layer. K3 carries no separate
`residual`, so nothing is None on layer 0 and the append reads the sum
directly. Every other model's hook has to test for it.

### CPU reference

`scripts/check_capture.py` gates a capture against a float32 forward of the checkpoint on a CPU
host. transformers has no Kimi K3 class, so `--trust-remote-code` runs the repo's own
`modeling_kimi_linear.py`. Its fla kernels are Triton, and `scripts/fla_torch.py` stands in for
them in pure torch when fla isn't installed. MXFP4 experts dequantize to BF16 on load, as the
engine patch does. `--offload-folder` streams one decoder layer at a time straight from the
checkpoint's safetensors and writes nothing. The vision tower isn't built.

```bash
python3 scripts/check_capture.py --model-path /mnt/data/kimi-k3 --trust-remote-code \
  --offload-folder /mnt/data/unused --reference-only k3-ref
```

A dry run of the first 4 layers on one 14-token prompt, c4-highcpu-96 in us-east5-a, 2026-09-26,
checkpoint on a 2,400 GB Hyperdisk Balanced at 1,200 MB/s: 4 minutes 13 seconds for both passes.
An MoE layer took 52 to 59 seconds to load in float32 and 13 to 18 seconds in BF16. Peak resident
memory was 169 GiB of the host's 188 GiB, during a float32 MoE layer.

Notes: no recorded run took this block as written; the dry run above cut it to 4 layers. Each
prompt streams every layer again. Projected from the dry run, untested: about 2 hours per prompt
for both passes over 93 layers, so about 8 hours for `--prompts-file` with the default three
prompts plus the joined one. Take a host with more than 188 GiB of RAM for margin.

`scripts/test_k3_reference.py` gates it on CPU: the stand-in kernels against a float64
recurrence, the modeling file on the stand-in against this page's float64 forward on a tiny
checkpoint, and streamed against plain bit for bit.

## Test

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/models/test_kimi_k3_model.py
```

CPU only, 8 simulated devices, and 32 for check 12. Twelve checks, each with negative controls that
have to fail. It pulls the patched code out of the patched files by AST, or
imports the patched checkout, so what runs is the patch. A mutation operator
that can't find its target fails the run, the same as a mutant that slips
through. The patches go on `eb061d8` with the capture and steering patches
under them, the tree `scripts/bootstrap_tpu_vm.sh` builds.

| Check | Reference |
|---|---|
| Both patches apply and every file compiles | `git apply --check` |
| MXFP4 dequantization, the sliced block reader, and the device decode | an E2M1 decoder written from the bit fields, and `dequantize_mxfp4` bit for bit |
| `situ` | float64, and `SituAndMul` from the checkpoint repo |
| `attn_res_mix` | float64, and `_apply_attn_res` from the checkpoint repo |
| The block-residual stack over 9 layers on the engine's Explicit mesh, and the capture hook | a float64 replay of the whole algorithm |
| The KDA decay layout, and the Mega KDA kernel on a per-head decay | the published `A_log` header and bytes, the reference modeling file, and a float64 recurrence |
| The layer split and pool sizes | the published `config.json` |
| The architecture name, `KimiK3Config`, and a layer-count override | the published `config.json` and the engine's own override code |
| The weight mappings in six shards, the runtime flags, and the MXFP4 expert stacks on both paths | the published safetensors headers, the bit-field decoder, and the file's own codes |
| The served model at `--ep-size` 1 and 2 and `--dp-size 2`, a layer-count override, both EPLB dispatch algorithms, and `NativeAttention`'s TPU branch at `--dp-size 2` | a float64 forward written from `modeling_kimi_linear.py`, and a float64 attention per rank |
| The router bias and `A_log` after a BF16 load, a dummy load, `--model-layer-nums` and a quantization config | the checkpoint file and the real loaders |
| The full model at `--tp-size 32` on 32 devices, weights and decode temporary a chip | shapes, dtypes and shardings under `nnx.eval_shape`, against 95.73 GiB |

Check 9 builds all 93 layers at full size under `nnx.eval_shape`, which
allocates nothing. It then measures 78 of the 2,460 mappings against the shapes
in the headers of shards 1, 2, 4, 94, 95 and 96. Those shards hold the dense
layer 0, KDA layer 1, MLA layer 3, the model-level weights, the vision tower
and the projector. Both directions have to close inside them: no mapping
without a tensor, and no tensor without a mapping or a predicate.

Check 10 writes a four-layer checkpoint in the published layout, with MXFP4
experts and a few vision tensors, and serves it through the engine's own path:
`ModelWorker` on the scheduler's Explicit mesh, which runs `JAXModelLoader` and
`ModelRunner`, then the startup precompile, then one prefill of two requests
and two decode steps through the scheduler's `PrefillAdder` and
`ScheduleBatch`. It serves float32 with a float32 short-convolution state. The
Mega KDA kernel takes BF16 only, so prefill runs the chunked KDA kernel, which
serving also takes whenever a 64-token tile holds more than two requests; check
6 runs the Mega kernel on its own. The logits, and every layer's input at
`--dp-size 1`, sit within 1.2e-05 relative error of the float64 forward at every
step. One `--ep-size 2` launch serves a file with a fifth layer, which
`--json-model-override-args '{"num_hidden_layers": 4}'` cuts back off. Two more
set `--ep-dispatch-algorithm` static and dynamic with
`--ep-num-redundant-experts 2`. Doubling half of layer 1's expert `w2` channels
in the file moves the served logits by 3.9e-01. A copy of the served result with
every logit and captured layer set to NaN has to fail all 30 gates.

The tiny checkpoint's `A_log` holds one value per head and zeros after them, as
the published one does, and the float64 forward reads it per head.

Check 10 then serves the tiny checkpoint in pairs, capture on: once with the
experts resident and once decoded at load. The tokens, every step's logits and
every captured layer have to agree bit for bit, in a float32 engine and a BF16
one, at `--ep-size` 1 and 2. Two mutants of `dequantize_mxfp4_jax` go into the
served resident path, the nibble order swapped and every scale one exponent
step high, and each has to break the agreement. Each pair runs in a child
process, because every launch keeps its compiled code mapped and a few more
capture launches in one process pass the default `vm.max_map_count`.

Check 9 loads the resident arrays at `--ep-size` 1, 2 and 8. They have to equal
the file's codes and scales, every callback has to return one device's shard
and no more, and their device decode has to equal the decode-at-load stack bit
for bit. Check 2 holds `dequantize_mxfp4_jax` to `dequantize_mxfp4` for all 256
byte values under all 256 E8M0 codes, in float32 and BF16. It also traces
`EPMoE._decode_mxfp4`: the first reader of the scales has to be the barrier
that takes the anchor, and the E8M0 lookup has to make 1/32 as many values as
the decode. Two source mutants, the scales moved back outside the barrier and a
per-element factor, each have to fail it.

Check 12 builds the full model at `--tp-size 32` on 32 simulated devices, in a
child process, and reports one chip's weights and the decoded-layer temporary.
Resident weights have to fit the `--mem-fraction-static 0.8` reservation, and
the decode-at-load weights have to exceed the chip, or the budget couldn't tell
the paths apart.

Off TPU the model serves only with `--attention-backend native`, the flag check
10 launches with. The default `fa` builds the absorbed-MLA latent pool, which
only the MLA Pallas kernel on TPU writes, so a CPU run refuses it at startup
and names the flag. Check 10 holds that refusal too.

Check 10 also runs `NativeAttention`'s TPU branch at `--dp-size 2` on CPU, with
the pool's TPU KV-write kernel in Pallas interpret mode. Each rank has to match
a float64 attention over its own requests, and a mutant that reads the batch as
one rank puts rank 1 off by 1.0.

Check 10 runs the engine's `ModelRunner`, which also imports `pybase64` and
`llguidance`, and the engine's `pyproject.toml` pins the second at `~=1.3.0`.

Set `SGLANG_JAX_REPO` to a local clone that holds `eb061d8` to skip the
download. The test reads `config.json`, `modeling_kimi_linear.py`, six
shard headers and two `A_log` tensors from one pinned revision of the Hub repo,
and each file has to match its pinned SHA-256 before the test reads it. They're
cached under `KIMI_K3_CACHE`, or `~/.cache/kimi-k3-published/<revision>` when
that isn't set. `LOG_PAYLOADS=1` prints the `ServerArgs` of every engine launch
and every request check 10 sends.

## Measured on a v5p-64

A `v5p-64` Spot slice (32 chips, 8 hosts) in us-east5-a, 2026-09-27. sglang-jax eb061d8 with
`sglang-jax-877.patch`, the steering patches, `multihost-hidden-states.patch` and
`models/kimi-k3-model.patch` (MXFP4-resident experts, scales barriered with the codes), built by
the `kimi-k3` row of `scripts/multihost_run.sh`: engine args `page_size=128
disable_radix_cache=True ep_size=32 watchdog_timeout=3600`, `mem_fraction_static=0.8`,
`--tp-size 32`. Every host read the weights from a GCS bucket in us-east5 through gcsfuse.

**Capture check**, `scripts/check_capture.py --trust-remote-code` against the float32 CPU
reference `refs/ref-kimi-k3.npz` (built from the checkpoint's own modeling code with the pure-torch
`fla` stand-in), 21:48 to 22:59Z:

| Prompt | Layers | Worst ratio to the BF16 floor | Worst Pearson | Control |
|---|---|---|---|---|
| 441 tokens | 93, all pass | 1.39 (layer 1) | 0.9816 (layer 84) | 6.58x, detected |
| 1,322 tokens, split across prefill passes | 93, all pass | 1.38 (layer 1) | 0.8859 (layer 92; the BF16 floor's own Pearson there is 0.9257) | 5.73x, detected |

The extend graph precompiled in 10.5 minutes and the decode graph in 8; the first capture request
compiled past sglang-jax's 300 s step watchdog on the first attempt, so the runner now passes
`watchdog_timeout=3600`. Two earlier launches failed: 196 GB of HLO temporaries from the MXFP4
scale expansion held for every layer (fixed by barriering the scales with the codes), then that
watchdog.

### Throughput on the `v5p-64`: 0.6 tokens a second

`scripts/multihost_run.sh NODE us-east5-a kimi-k3 measure` ran twice on the same slice and never
printed a capture-off RESULT line. Capture off prefills 2 warmup and 30 timed batches of 8 prompts
at 440 tokens, one output token each, so 112,640 tokens, and `serve_throughput.py` prints only
after the last batch.

| Run | Engine settings beyond the row | Precompile done | Stopped |
| --- | --- | --- | --- |
| 2026-09-27 | none | 00:08:37Z on 09-28 | 01:41Z, 93 min later |
| 2026-09-28 | `disable_overlap_schedule=True` | 02:34:47Z | 02:53Z, 18 min later |

In the first run all 8 hosts kept their engine processes (checked at 01:29Z). In both runs the
scheduler on host 0 sat in `device_get` on the batch result: in `resolve_last_batch_result` with
the overlap scheduler, in `run_batch` without it. `tpu-info` on host 0 read 100% duty cycle and
8.23% TensorCore use on every chip the whole time, with 81.83 of 95.73 GiB of HBM in use a chip.
The scheduler used 12 minutes of CPU in 2 hours 12 minutes, so the host wasn't compiling. The slow
part ran on the chips.

The capture check fits that. Its two requests were 441 and 1,322 tokens, the second split over two
1,024-token passes. Engine start at 21:48:46Z to exit at 22:59:21Z took 70.6 minutes. The precompile
took 18.5 of them, and loading took about 20 on earlier launches (not timed on this one), which
leaves about 30 minutes for 1,763 tokens. That's about 1 token a second, and at that rate
capture off alone needs about 31 hours, which fits both stopped runs. The suspects are the MXFP4
expert decode inside every MoE forward and the `mega` KDA prefill kernel the runs took by default.
Neither has been profiled. The standard `measure_model.sh` run would take over a day at that rate,
so a short run below stands in for it. HBM in use while serving, 81.83 GiB a chip at
`mem_fraction_static=0.8`, is from `tpu-info`, not the peak sampler.

A short run then measured the rate directly, 2026-09-28 03:21 to 04:44Z, same slice and engine
settings as the check, capture off, `serve_throughput.py --batch-size 1 --warmup-batches 1
--batches 2`:

```
RESULT {"stage": "capture_off", "tp_size": 32, "batches": 2, "tokens": 880, "secs": 1507.55, "tokens_per_s": 0.6, "window": "batches 2 to 3, first 1 discarded"}
```

That's 0.6 tokens a second, one 440-token prompt every 12.6 minutes, against 2,685.8 for GLM-5.3
on the same slice. The extend graph precompiled in 9.2 minutes and the decode graph in 8.4. Capture
on wasn't measured. At this rate the model can't serve a workload until the prefill is fixed.

Next step: take one xprof trace of a single 440-token prefill at batch 1, and split the time between
the MXFP4 decode, the grouped matmul and the KDA kernel before choosing a fix.
`scripts/serve_throughput.py --profile-dir DIR` takes that trace. [The roadmap](../docs/roadmap.md)
has the plan.

To reproduce the check, stage the checkpoint and its reference as
[Across hosts](../README.md#across-hosts) describes, then run from the repo root:

```bash
BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE kimi-k3 setup check
```

Notes: nobody has replayed that block as written from a clean start. The runs above came from the
same script and row. The batch-1 run discarded one warmup batch, where the other measured rates
discard two.

