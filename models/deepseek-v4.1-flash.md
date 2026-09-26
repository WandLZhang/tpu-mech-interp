# DeepSeek V4.1-Flash

`deepseek-ai/DeepSeek-V4.1-Flash`

**Status: not runnable.** No JAX implementation exists, and nothing here serves it. This page is
a design.

| | |
|---|---|
| Parameters | 552B backbone + 196.6B Engram. 8B active at prefill, 16B at decode |
| Layers | 40: 20 causal encoder + 20 decoder. 3 DSpark draft blocks follow |
| Hidden dim | 5,120, carried as 4 parallel residual copies |
| Attention | 64 query heads, 1 KV head, head dim 512. Sliding window 128, plus compressed sparse attention |
| Rotation | The last 64 of each head's 512 channels. The other 448 stay put |
| Global KV | 890 bytes a token, in 4 caches behind 40 layers |
| MLP | 384 routed experts top-6 plus 1 shared. The draft blocks route 128 top-3 |
| Positions | Two rotary tables. Compressed layers run YaRN over 65,536 at theta 160,000; window-only layers run theta 10,000 with YaRN off |
| Conditional memory | 2 n-gram hash tables, 384M rows each, at layers 1 and 14 |
| Vision | 32-layer ViT at 1,024 dim, two-layer MLP projector |
| Context | 1,048,576 |
| Weights | 1,394 GiB BF16. 475 GiB as shipped, at FP8 dense and FP4 experts |

[`scan/deepseek_v4_layers.py`](../scan/deepseek_v4_layers.py) parses the config into the plan a
port needs. The module builds no layers.

## Serve

Engine: none. `sglang-jax` has no `deepseek_v41` model
([survey](../upstream/capture-hook-survey.csv)).

Slice: `v5p-64`, 32 chips, 3,040 GiB. The backbone at BF16 is 1,028 GiB. Keep the Engram tables in
the FP8 the checkpoint ships and dequantize a row on lookup, the way `ParallelEngramEmbedding` does
in the reference runtime, and they add 189 GiB rather than 366 GiB. A lookup is a gather, so the
stored dtype costs no matmul throughput. That's 1,217 GiB of weights, 38.0 GiB a chip, and 57 GiB a
chip left for KV and activations.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in `/dev/shm`, which
defaults to half the host's RAM. That host has 440 GB of RAM, less than this 475 GiB checkpoint,
so the weights can't live there. Each of the 8 hosts in a `v5p-64` reads from the checkpoint. No
run here has loaded a checkpoint this size.

A port would confirm the chip count first, on all 8 hosts at once, as
[gpt-oss-120b](gpt-oss-120b.md#scaling-out-to-v5p-64) describes. Untested: no run here has used a
`v5p-64`. It reports 32.

```bash
python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count())"
```

The KV cache is small enough to stop being the constraint. 128 sequences at the full 1M context
cost 112 GiB, 4% of the slice.

Seven pieces of the engine decide what the port has to change, and each one has a specific site.

| What the model needs | What the engine has | The decision |
|---|---|---|
| A registry entry for `DeepseekV41ForCausalLM` | Nothing. The checkpoint is `image-text-to-text` with `image_token_id` and `vision.*` shards | Register text-only or multimodal. That choice sets the whole key map |
| A key map with no top-level prefix | `Qwen3ForCausalLM` maps `model.embed_tokens.weight` and `model.norm.weight` | The checkpoint stores `embed.weight`, `head.weight`, `norm.weight` and `aligner.w1/w2` at the root. Write the map against those |
| Four shared KV caches at two compression ratios | One `req_to_token` position-to-slot map and one `out_cache_loc` a token. `SWAKVPool.full_to_swa_index_mapping` is the only remap, and it handles one extra layer class | Carry one slot map per distinct compression ratio. Layers 2, 8 and 14 write one latent every two tokens, layer 20 writes one a token, and the ratio is static per layer |
| FP4 latents with a block scale | `MLATokenToKVPool` holds one float dtype, each half padded to 128 | At BF16 that layout is 1,280 bytes a latent slot and 256 an index key. Four owners cost 6,144 bytes a token and the default 40 layers cost 61,440, against the model's 890 |
| An indexer map with three types | `build_index_share_map` knows `full` and `shared` and raises on anything else. The config carries no `indexer_types` | Extend it. A `REINDEX` layer publishes its own top-k while reading somebody else's key buffer, which neither existing type describes. `layer_plans` derives all three from the two source lists |
| The Engram table read at FP8 | `WeightLoader.dequant_fp8_linear` returns BF16 | Add a gather path that dequantizes a row, not a tensor. Widening on load costs 366 GiB instead of 189 |
| SWA Bounded Replay | `SWAKVPool` persists the window | Either persist it, at 2.77 MiB a sequence, or rebuild it by replaying the last 128 tokens the way the model card describes |

The draft blocks are less new than they look. `models/dspark.py` already ships `VanillaMarkovHead`
with `markov_w1 [vocab_size, markov_rank]` and `markov_w2`, which match `mtp.2.markov_head.embed`
and `.head` at `[129280, 256]` against the config's `dspark_markov_rank: 256`, and it already logs
about `confidence_head.` keys it ignores. `models/dflash.py` builds its `fc` at
`num_context_features * target_hidden_size`, which is 3 by 5,120, the input width of
`mtp.0.main_proj.weight`. `speculative/multi_layer_draft_worker.py` runs one model runner per
`num_nextn_predict_layers`, so the three blocks arrive as three draft models with their own KV
pools.

## Layer plan

Four things break the assumption that a stack is 40 interchangeable decoder layers. The stack is 43
blocks and the last three sit in another namespace, which this section covers. The three sections
after it take the rest: the layers split into an encoder and a decoder, CSA2 makes each layer read
state another layer wrote, and Engram puts a table larger than one chip in front of two of them.

The snippets on this page run in sequence from `scan/`.

```python
from deepseek_v4_layers import (DEEPSEEK_V41_FLASH_CONFIG, check_causal_encoder_decoder,
                                engram_plan, stage_split, unit_runs)

for run in unit_runs(DEEPSEEK_V41_FLASH_CONFIG):
    print(run.stage, run.modes, run.start, run.repeats)
```

Nine runs cover the 43 blocks, so a scanned stack compiles nine bodies.

| Blocks | Prefix | Stage | Unit | Repeats |
|---|---|---|---|---|
| 0 | `layers.0.` | encoder | window | 1 |
| 1 | `layers.1.` | encoder | window, with an Engram table | 1 |
| 2-13 | `layers.2.` | encoder | Full + 5 Reuse | 2 |
| 14-19 | `layers.14.` | encoder | Full + 5 Reuse, with an Engram table | 1 |
| 20-23 | `layers.20.` | decoder | Full + 3 Reuse | 1 |
| 24-39 | `layers.24.` | decoder | Reindex + 3 Reuse | 4 |
| 40 | `mtp.0.` | draft | window, with the target projection | 1 |
| 41 | `mtp.1.` | draft | window | 1 |
| 42 | `mtp.2.` | draft | window, with the draft heads | 1 |

The run at layer 24 is the one that pays. Four identical units of four layers fold into one
`lax.scan` over stacked weights.

The three draft blocks share a stage, a mode and a ratio, and they still need three bodies. `mtp.0`
carries `main_proj` and `main_norm`, `mtp.1` carries nothing past a plain block, and `mtp.2` carries
`norm`, the Markov head and the confidence head. Weights that differ can't be stacked, so the run
key in `unit_runs` holds the tensor names beside the stage, the ratio and the modes. It also holds
each layer's role in the candidate pool. The builder indexes over every compressed position and a
reader indexes inside the pool, so a builder runs other code than a reader in the same mode.

`compress_ratios` carries 43 entries, one per block, and the last three are the DSpark draft blocks.
A runtime that sizes its stack from `num_hidden_layers` builds 40 blocks and three sliding-window
rings too few. The three extra blocks don't live in the `layers.` namespace either. That namespace
stops at 39, and the draft blocks are `mtp.0`, `mtp.1` and `mtp.2`, so a loader that builds keys
from a layer index resolves nothing for them. `LayerPlan.checkpoint_prefix` gives the right prefix
for all 43.

## Causal encoder-decoder

Layers 0 and 1 run a sliding window and pool nothing. Layers 2 through 19 pool two tokens into one
KV latent. Layers 20 through 39 keep one latent a token. The ratio change at layer 20 is the
boundary, and layer 20 is the only decoder layer that projects KV. It reads the hidden state
leaving the encoder, and the other 19 decoder layers attend over what it wrote.

```python
split = stage_split(DEEPSEEK_V41_FLASH_CONFIG)
split.encoder      # range(0, 20)
split.decoder      # range(20, 40)
check_causal_encoder_decoder(DEEPSEEK_V41_FLASH_CONFIG)
```

`check_causal_encoder_decoder` is the claim the boundary rests on. A config can change compression
ratio part way down and still be an ordinary stack, so the check confirms the decoder holds one KV
owner and that every decoder layer binds to it. A third property needs no check: an encoder layer
can't read the decoder's KV, because a consumer binds to a source at or before itself and an
encoder layer sits below the boundary.

This is where a port built on a uniform decoder goes wrong twice. It allocates 40 KV caches where
the model holds 4, and it projects each decoder layer's KV from that layer's own hidden state
instead of from the encoder output.

## CSA2

Each attention layer runs one of four modes.

| Mode | Compressed KV | Index keys | Top-k | Layers |
|---|---|---|---|---|
| `WINDOW` | none | none | none | 0, 1, and the 3 draft blocks |
| `FULL` | writes | writes | runs | 2, 8, 14, 20 |
| `REINDEX` | reads | reads | runs | 24, 28, 32, 36 |
| `REUSE` | reads | reads | reads | the other 30 |

`kv_source_layer_ids` and `index_source_layer_ids` name the owners. A consumer binds to the nearest
owner at or before it, because the reference runtime hands shared state down the stack in layer
order and every owner writes before its consumers read. Layers depend on each other, so grouping by
layer type alone gives no scan and the plan has to carry the bindings.

Two invariants hold the wiring together, and `layer_plans` raises on either. Every
`kv_source_layer_id` appears in `index_source_layer_ids`, because index keys project from the
compressed latent and only a layer that compresses its own KV can make them. A consumer and its
source run the same compression ratio, because the shared cache is sized `max_seq_len // ratio` and
both ends index it the same way.

Rotation is partial, and it runs both ways. `apply_rotary_emb` turns the last 64 channels of each
512-wide head and leaves the other 448 alone. The attention output is then rotated back by the
conjugate before the output projection, which is what keeps one shared cache usable by queries at
different positions. A port that skips the inverse reads the cache correctly and returns the wrong
activations.

The mode decides which tensors a layer carries, so the plan predicts the checkpoint.

| Mode | Tensors beyond a plain block |
|---|---|
| `FULL`, ratio 2 | `attn.compressor.{norm,wkv,wgate}`, `attn.indexer.{wk,k_norm,weights_proj,wq_b}` |
| `FULL`, ratio 1 | the same without `wgate`, because one token a latent needs no pooling gate |
| `REINDEX` | `attn.indexer.{weights_proj,wq_b}` |
| `REUSE` | none |
| `WINDOW`, backbone | none |
| `WINDOW`, draft | `main_proj` and `main_norm` at `mtp.0`, the heads at `mtp.2`, nothing at `mtp.1` |

`attn.indexer.wq_b` ships FP8 E4M3 with an E8M0 `.scale` beside it. The other three indexer tensors
and all three compressor tensors are BF16. Engram's `embed` and `wkv` are quantized, its `q_weight`
and `k_weight` aren't, and `mtp.0.main_proj` is. No suffix rule recovers that split, so
`LayerPlan.extra_weights` names each scale explicitly. A loader built from the weight names alone
misses a scale on ten of the 43 blocks.

Layer 20 also builds the candidate pool. It keeps the 2,048 highest-scoring blocks of 8 compressed
positions per query, and the four `REINDEX` layers below it score only inside that pool. The pool
bounds what the top-k can select, not what the scoring costs. `Indexer.forward` runs its einsum
over `index_k[:, :end_pos // ratio]`, the full compressed axis, and then masks the result with the
candidate mask, so every indexer's score pass grows with the context. A port that wants the bounded
cost the architecture describes has to gather the candidate blocks before the einsum instead of
masking after it.

Three of the four owners hold a second per-sequence state that has nothing to do with sequence
length. Layers 2, 8 and 14 pool two tokens into one latent, so between decode steps they hold the
tail of a group that hasn't closed. A cache manager that models the KV slot alone drops a token out
of every pair.

## Engram

Two hash tables, 196.6B parameters, sparsely read. Layer 1 owns 384,006,168 rows and layer 14 owns
384,016,682. A row is 256 values, stored FP8 E4M3 with one E8M0 scale per 32 channels, so 264 bytes
a row and 94.42 GiB a table.

A position is hashed as the 2-gram, 3-gram and 4-gram ending there, each split over 8 heads. That's
24 rows a token from each table, 48 in total, 12,672 packed bytes. Every (n-gram size, head) pair
owns a disjoint bucket range, sized by the next unused prime at or above 16,000,000, which is where
the row counts come from.

A row index is a hash of token ids alone, so every index for a whole chunk is known before layer 0
runs. The gather depends on no hidden state, which is what lets a prefetch work at all.

```python
plan = engram_plan(DEEPSEEK_V41_FLASH_CONFIG)
plan.placement((14,), tokens=8192, devices=32, ici_bytes_per_s=1.2e12,
               link_bytes_per_s=32e9, step_seconds=8.9e-3, num_blocks=40)
```

Costed on a 32-chip v5p at 1,200 GB/s of ICI a chip and 32 GB/s of host link a chip. The prefill
column is one 8,192-token chunk, timed at 8B active parameters and two flops a parameter against
the slice's peak BF16 rate. The decode column is one step of 256 tokens, timed at 16B active
parameters read from HBM once.

| Placement | HBM a chip | ICI, prefill chunk | Host link, prefill chunk | Left on the critical path |
|---|---|---|---|---|
| Both tables sharded over 32 chips | 5.90 GiB | 325 us | none | none |
| Both tables in host memory | none | none | 3.24 ms | 1.40 ms a chunk, 42 us a decode step |
| Layer 1 sharded, layer 14 in host memory | 2.95 GiB | 163 us | 1.62 ms | none |

Take the split. Layer 14 runs fourteen blocks into the forward pass, so its 1.62 ms fetch hides
under 3.12 ms of cover at prefill and its 51 us fetch hides under 127 us at decode. Layer 1 runs
one block in and gets a fortieth of the pass, 0.22 ms at prefill and 9 us at decode, which covers
neither. Sharding layer 1's table alone costs 2.95 GiB a chip, half of what both tables cost, and
adds nothing the prefetch can't absorb.

The sharded half works the way the reference runtime shards it. Rows split by device, each device
gathers against its own slice and zeros the rows it doesn't hold, and one all-reduce sums the
result. What crosses ICI is the dequantized rows at 12,288 BF16 bytes a token for one table, twice
the 6,336 packed bytes the gather read.

Host residency for both tables is the placement to reject. It saves 5.90 GiB a chip and buys a
16% longer prefill chunk and an 11% longer decode step, and the whole cost comes from layer 1.

## KV budget

| Tensor | Bytes | Format |
|---|---|---|
| Compressed KV latent | 288 | 512 FP4 E2M1 values, one E4M3 scale per 16 channels |
| Index key | 68 | 128 FP4 values, one E8M0 scale per 32 channels |
| Sliding-window slot | 528 | 512 FP8 E4M3 values, one E8M0 scale per 32 channels |

These are the packed formats that the model card's 890 bytes a token assumes. The reference
runtime rounds each value to that grid and stores it back in BF16, at 1,024, 256 and 1,024 bytes
an entry.

`num_key_value_heads` is 1. One 512-wide latent serves all 64 query heads, and the whole budget
below rests on that. A `FULL` layer at ratio `r` writes one latent and one index key every `r`
tokens, so it costs `(288 + 68) / r` bytes a token. Three encoder owners at ratio 2 contribute 178
each and the single decoder owner at ratio 1 contributes 356. That's 890 bytes a token, which is the
figure the model card publishes, and 890 MiB for one sequence at the full 1M context.

Nothing else grows. The sliding window is a ring of 128 slots in each of the 43 blocks, 2.77 MiB a
sequence at any length when packed. The reference runtime keeps the ring in memory. The model
card's SWA Bounded Replay rebuilds it from the last 128 tokens, so it never goes to SSD.

`padded_bf16_kv_bytes_per_token` prices the same four caches in a paged pool that has no FP4 path
and pads each half of a slot to 128. That's 6,144 bytes a token, 6.9 times the card's 890, and
61,440 if the pool also sizes its layer count from the stack rather than from the four owners.

## Weights the plan has to name

The mode table above lists what a block adds. Every block also carries 36 tensor names, counting
the routed experts once, and no upstream key map recognizes any of them. That set is where the
qwen3 and gemma4 conventions stop helping.

- `attn.wq_a [1280, 5120]` and `attn.wq_b [32768, 1280]`. The query projection is LoRA-factored, so
  there's no `q_proj`.
- `attn.wkv [512, 5120]`. One latent for every head, so there's no `k_proj` or `v_proj`.
- `attn.wo_a [8192, 4096]` and `attn.wo_b [5120, 8192]` behind `o_lora_rank: 1024` and
  `o_groups: 8`. `wo_a` is block-diagonal over the 8 groups and the reference runs it as an
  einsum, not a `Linear`.
- `attn.attn_sink [64]`, one learned sink per query head.
- `hc_attn_fn` and `hc_ffn_fn` at `[24, 20480]`, plus `hc_attn_base`, `hc_ffn_base`,
  `hc_attn_scale` and `hc_ffn_scale`. These produce the Hyper-Connections mixing coefficients.
- `ffn.gate.bias` and `ffn.gate.bias_vl`, both `[384]`. The second is the routing bias for image
  tokens, and no `EPMoE` path upstream models a second bias.

`WeightMapping._infer_default_sharding` keys off `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj` and `down_proj`. None of those names appears here, so every `kernel_axes`
choice is new work.

The router also needs `scoring_func: sqrtsoftplus`, `routed_scaling_factor: 1.5` and
`norm_topk_prob: true`, and the experts need `swiglu_limit: 10.0`, which clamps the up branch on
both sides and the gate branch from above. `rms_norm_eps` is `1e-20` and feeds every RMSNorm plus
the Hyper-Connections normalization.

## Capture

No model, and no hook to hang on one. Per-layer capture needs the JAX model written first, then
[`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) and a `layers_to_capture` hook.

The scan and the ordinary hook don't compose, so pick one before writing the model.

The ordinary hook is the None-residual append in
[`upstream/glm5-capture-hook.patch`](../upstream/glm5-capture-hook.patch). It needs an unrolled
Python loop over the blocks, because a scan body traces once and `layer_id` arrives as a tracer no
`in` test can read. An unrolled stack traces all 43 blocks into the program instead of 9 scan
bodies, and it's what every one of the 31 models in `sglang-jax` writes today. Nothing upstream
uses `lax.scan`, `nnx.scan` or `nnx.vmap`, `RadixAttention` stores `layer_id` as a Python int, and
`MLATokenToKVPool` indexes its buffer list with it.

The scanned stack needs the `ys` form instead. Each body returns one entry per layer in the unit,
so a run of `R` repeats over a unit of `U` layers hands back `[R, U, seq, hidden]`, and the reshape
and concatenation happen after the scan.
[`nemotron3-super.md`](nemotron3-super.md) writes that out. It also needs `layer_id` to stop being
a Python int in the two places above, which is the part with no precedent in the engine.

The residual stream carries four parallel copies. Each block collapses them twice, once into the
attention input and once into the feed-forward input, and expands twice on the way out, mixing the
residual in through a doubly stochastic matrix. So 40 layers offer 80 collapse points, 800 KiB a
token at 5,120 dim in bf16. Capture all four copies at both points and it's 3,200 KiB.

A block also hands the next block the mix coefficients its feed-forward produced. That carry
crosses the block boundary, so a scan body threads it alongside the stream.

The collapsed input is the tensor an SAE wants, because it's what the sublayer reads. The draft
head takes a different collapse of the same stream, the plain mean over the four copies, at layers
37, 38 and 39.

## Notes

`scan/test_deepseek_v4_layers.py` checks the plan on CPU. It needs no checkpoint download.

```bash
source .venv/bin/activate
uv pip install -r scan/requirements.txt
python3 scan/test_deepseek_v4_layers.py
```

Four references grade it, and none of them reads the parser. The shipped safetensors headers give
the tensor set and the prefix of all 43 blocks, which the mode assignment and the draft stage have
to predict. The size of the two Engram shards on disk pins the row counts to the byte. The
published bucket rule, re-run in the test with its own prime sieve, reproduces both row counts. The
model card gives 890 bytes a token, 196B Engram parameters and the 20/20 split.

Thirty-four controls corrupt the config, and each has to be rejected with the message that names
the fault. Six more make the parser return a plan that's wrong but well formed, or cost a plan the
wrong way, then confirm the checkpoint comparison, the prefix comparison, the run-key count and the
placement arithmetic catch them, so no check is grading itself.

A full implementation needs nine components the plan sizes but doesn't build: the Hyper-Connections
residual stream with its Sinkhorn mixing, the two-level sparse indexer, the FP4 KV path, the Engram
gather and its gate, the `sqrtsoftplus` router with its dual bias, the LoRA-factored query and the
grouped output projection, the per-head attention sink, the ViT and its projector, and the DSpark
draft loop with its Markov and confidence heads.
