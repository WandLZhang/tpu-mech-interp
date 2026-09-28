# Inkling

`thinkingmachines/Inkling`, `thinkingmachines/Inkling-Small`

**Status: captured on TPU**, `v5p-64`, 2026-09-27; every layer within the BF16 floor, one control
under its bar; not measured. Served and captured on 32 chips, 8 hosts, in us-east5-a. Throughput
and HBM aren't measured yet; [the roadmap](../docs/roadmap.md) has both. See
[Measured on a v5p-64](#measured-on-a-v5p-64).

| | Inkling | Inkling-Small |
|---|---|---|
| Parameters | 975B total / 41B active | 276B total / 12B active |
| Layers | 66: 55 local + 11 global | 42: 35 local + 7 global |
| Hidden dim | 6,144 | 4,096 |
| Attention | 64 query heads. 8 KV global, 16 KV local. 5:1 sliding window(512) to full | 32 query heads. 8 KV either way. 5:1 sliding window(512) to full |
| Attention scale | 1/128, with a per-head RMS norm on q and k | same |
| MLP | 2 dense, 64 routed. 256 experts, top-6, plus 2 shared inside the router | 2 dense, 40 routed. 256 experts, top-6, plus 2 shared inside the router |
| Expert width | 3,072 | 2,048 |
| Dense MLP width | 24,576 | 16,384 |
| Positions | learned relative bias, no RoPE | same |
| Per-layer state | 4 short convolutions, kernel 4 | same |
| Normalization | RMS, eps 1e-6, one of them on the embedding | same |
| Logit divide | 24.0 | 16.0 |
| Vocabulary | 201,024 head rows, 200,058 real | same |
| Context | 1,048,576 | 1,048,576 |
| Weights | 1,764.0 GiB BF16 served, 1,773.9 GiB on disk | 491.1 GiB BF16 served, 495.4 GiB on disk |

The model cards say 975B and 276B. The BF16 checkpoints hold 952.4B and
266.0B. `InklingForConditionalGeneration` loads only `model.llm`, so the MTP
head and the vision and audio towers stay on disk.

Expert width is the field the `transformers` config class gets wrong for
Inkling-Small. The published `config.json` writes it as `intermediate_size`
and writes the dense width as `dense_intermediate_size`.
The `transformers` config class moves the dense width onto `intermediate_size`
and leaves `moe_intermediate_size` at its class default of 3,072, which is
Inkling's expert width and not Inkling-Small's, so the patch registers
`InklingServingConfig` to read the published field before that rewrite.

## Serve

Engine: `sglang-jax` with
[`upstream/models/inkling-model.patch`](../upstream/models/inkling-model.patch), which adds
`python/sgl_jax/srt/models/inkling.py`, `configs/inkling.py` and the
per-request convolution pool in `mem_cache/short_conv_state_pool.py`.
`EntryClass` registers three names. `InklingForConditionalGeneration` matches
the architecture the published `config.json` advertises and serves the text
path of that checkpoint. `InklingForCausalLM` serves a text-only config.
`InklingMTPForCausalLM` builds one layer of the MTP head as a draft model,
selected with `mtp_layer_idx`. That layer's attention kind comes from
`mtp_config.local_layer_ids`, which is [0, 2, 4, 5, 6, 7] over 8 layers, so
layers 1 and 3 run full attention and the other six run the sliding window.
Nothing serves the draft yet. The worker refuses speculative decoding for
Inkling, because no convolution window rolls back past a rejected draft token.

The patch also teaches the native attention backend to read a per-token
position bias, because Inkling adds a learned bias to its logits in place of a
rotary table. `NativeAttention` reads it and sets
`supports_position_bias = True`. `--attention-backend` defaults to `fa`, which
doesn't. The native backend also maps a sliding-window layer's cache slots into
the smaller pool `SWAKVPool` keeps for those layers, as the FA backend does.

The usual native pass gathers k and v at every slot of the batch's `cache_loc` bucket and masks
the logits. The bucket is `--max-running-requests` times the page-aligned `max_req_len`, and
`max_req_len` follows the KV pool when `--context-length` is unset. For the capture check on a
`v5p-64` that's 8 x 666,176 = 5,329,408 slots. Each layer then builds `[1024, 64, 5,329,408]`
logits and `[1024, 5,329,408]` distance and mask matrices, the matrices whole on every chip. A call
with a position bias runs `blockwise_attention` instead. It walks the bucket 512 slots at a time,
stops after the batch's last real slot and carries an online softmax. So it reads each request's
own KV, as the FA backend's pages do, and holds one `[tokens, heads, 512]` tile at a time.

An AOT compile of the served forward shows the difference. It builds the full published config
with the engine's dummy loader and abstract pools, sizes the pool the way `init_memory_pool` does
for the check's flags, and compiles the extend shape (8 requests, 1,024 tokens) for a 32-chip
`v5p:2x4x4` topology through libtpu 0.0.46.1, on a CPU VM on 2026-09-27:

| | Whole-bucket pass | Blockwise pass |
|---|---|---|
| Temporaries per chip | 1.43 TB, over the 95.73 GiB HBM | 1.47 GiB |
| Weights and pools per chip | 73.1 GiB | 73.1 GiB |

Notes: the compile sizes the pool from 55.6 GiB of weights a chip, which gives 666,176 tokens. The
chip's own compile reported 1.71 TB, from a pool the engine sized from the HBM it measured.

Serve it with `--attention-backend native --disable-radix-cache`, at
`--dp-size 1`. Each request keeps its convolution windows in a pool slot, and
the radix cache can't snapshot them, so a prefix hit would open on the wrong
windows. Chunked prefill still works, because a request keeps its slot across
its chunks. Data parallelism splits the batch into per-rank sections, and the
convolutions and the relative bias read it as one packed list.

`ModelWorker` hands the flags to `InklingForCausalLM.check_server_args` before
it reads a weight. It refuses any attention backend but the native one, the
radix cache, `--dp-size` above 1, speculative decoding and
`--enable-recurrent-extra-buffer`. It refuses PD disaggregation in either mode,
`--pd-disaggregation` or `--disaggregation-mode`, because PD hands a request's
KV to the decode side and leaves its windows behind. It also refuses the three
EPLB flags, `--ep-dispatch-algorithm`, `--ep-num-redundant-experts` and
`--init-expert-location`. Inkling's router picks logical experts with its own
top-k, so they'd change nothing. One error names every flag at fault, so a
launch without the right flags stops in seconds rather than after it reads
1,773.9 GiB. The runner, the model and the first attention layer check again,
for a runner built without the worker.

The config class needs `transformers` 5.14 or later, and `sglang-jax` pins
5.12, so install a newer one into the serving environment first. On 5.12 the
patched tree still imports and serves other models, and an Inkling checkpoint
stops in `AutoConfig`.

Slice: `v5p-64`, 32 chips. A v5p chip carries 95 GiB of HBM,
so the slice holds 3,040 GiB. Take 1,764 GiB of weights and 1,276 GiB is left
for KV cache and activations. A `v5p-32` is 16 chips and holds 1,520 GiB, so
it's under the weights alone. Inkling-Small fits 8 chips on weight size, but
256 experts spread better over 32.

The [measured `v5p-8` runs](nemotron3-super.md#measured) keep weights in
`/dev/shm`, which defaults to half the host's RAM. That host has 440 GB of RAM,
less than either checkpoint, 1,773.9 GiB or 495.4 GiB, so neither can live
there. The 8 hosts of a `v5p-64` read the checkpoint from a gcsfuse mount of the
bucket, which `scripts/multihost_setup.sh` makes.

Each host reads only the blocks its 4 chips hold, once. `load_stacked_weights`
reads k and v, the dense MLP, the shared experts and the routed experts that
way, one `make_array_from_callback` per tensor, and chips holding copies of one
KV head share one read. The mapping table's tensors go through the loader's
sharded reads. The upstream loader also pre-reads every `.safetensors` file on
a fuse mount to warm the gcsfuse cache, and the patch skips that in a
multi-process load. On 2026-09-27, before these changes, that warm-up read
1,904 GB on each host, the whole checkpoint, before the first weight.

Mount the bucket with `RANGE_CACHE=false`, which the `inkling` row of `scripts/multihost_run.sh`
sets.
The experts shard over the tensor axis, so every host reads an eighth of every
expert file. With the gcsfuse file cache pulling a whole file on any read past
its first byte, each host pulls every file whole through a 120 GB cache, and
host 0 took in 31 TB for the 2026-09-27 load.

Confirm the chip count first, on all 8 hosts at once, as
[gpt-oss-120b](gpt-oss-120b.md#scaling-out-to-v5p-64) describes. It reported 32 on every host on
2026-09-27; `scripts/multihost_setup.sh` runs this check.

```bash
python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count())"
```

Both sizes run 8 KV heads on the global layers, so tensor parallel past 8 ways
replicates KV rather than splitting it. At `--tp-size 32` the cache is stored
four times, and the k and v weights are replicated on the way in to match.
`kv_head_padding` on the `attn.wk_dv.weight` and `attn.wv_dv.weight` mappings is
what asks for that, and `swa_head_dim` is what tells a 2,048-wide local tensor
from a 1,024-wide global one. The k and v convolutions run at the replicated
width too, so `load_weights` repeats each head's rows of `attn.k_sconv.weight`
and `attn.v_sconv.weight` the same way, and their windows grow with them.

Inkling's local layers run 16. Each layer keeps its own count.
`set_num_token_hybrid` walks `model.model.layers` and reads
`self_attn.attn.sliding_window_size` off each one, which `RadixAttention` holds
as None on a global layer and 512 on a local one. That sorts the stack into
`swa_attention_layer_ids` and `full_attention_layer_ids`, and the runner builds
an `SWAKVPool` taking `swa_head_num` from `swa_num_key_value_heads` and
`head_num` from the global count, so neither kind pads up to the other.

## Layer plan

The stack isn't uniform, so the port can't put all 66 layers in one `lax.scan`
over stacked weights. [`scan/inkling_layers.py`](../scan/inkling_layers.py)
parses `config.json` into the runs that can scan together. From `scan/`:

```python
from inkling_layers import INKLING

for group in INKLING.scan_groups():
    print(group.attention, group.mlp, group.start, group.length)
```

Three kinds of layer cover both models:

| Kind | Inkling | Inkling-Small |
|---|---|---|
| local attention, dense MLP | layers 0-1 | layers 0-1 |
| local attention, routed MLP | 53 layers in 11 runs | 33 layers in 7 runs |
| full attention, routed MLP | 11 layers, 1 run each | 7 layers, 1 run each |

That's 23 runs over 66 layers, and 15 over 42. Three kinds means XLA compiles
three bodies either way.

`dense_mlp_idx` is a boundary, not a layer number. The published value is 2, so
layers 0 and 1 run one dense MLP of width 24,576 and layers 2 and up route to
experts of width 3,072. `local_layer_ids` lists the sliding-window layers, and
the cycle it encodes puts a full-attention layer at every index of the form
`6k + 5`. The two breaks don't line up, which is why the first cycle splits
into three runs and the other ten split into two.

## Positions

There's no rotary table. Each attention layer adds a learned relative bias to
its logits. Attention runs four separate projections of the hidden state, and
the checkpoint stores four separate tensors for them: `attn.wq_du.weight`,
`attn.wk_dv.weight`, `attn.wv_dv.weight` and `attn.wr_du.weight`. The fourth is
a stream `r` of width `d_rel = 16` per head, and a `[d_rel, extent]` matrix
turns it into one bias per backward distance. Distances at or past the extent
get zero. Global layers run the extent at `rel_extent = 1024`, local layers at
the 512-token window, so a local layer biases every position it can reach.

Above `log_scaling_n_floor = 128000` tokens the global layers scale their
logits by `1 + 0.1 * ln(max(1, (pos + 1) / 128000))`. That's 1.07 at 256k and
1.21 at the full 1M context. The log is natural, so a base-10 reading gives
1.09 at 1M instead. Local layers never scale.

`RelativePosition` in `scan/inkling_layers.py` carries the fields and computes
the scale, for a sequence of `seq_len` tokens:

```python
import jax.numpy as jnp

pos = INKLING.positional()
pos.proj_shape("global")         # (16, 1024)
pos.tau(jnp.arange(seq_len))     # [T] float32, 1.0 below the floor
```

The bias rides into attention as an aux tensor of `[tokens, heads, extent]`. At
8,192 tokens, 64 heads and extent 1,024 that's 1 GiB in BF16 for one layer,
which is larger than the layer's activations. A TPU port wants it folded into
the flash kernel's score step rather than materialized.

## Normalization and scale

Every norm is RMS with `rms_norm_eps = 1e-6`. Three of them sit where a
Llama-shaped port has none, and two of the three decide what reaches the
logits.

`q_norm` and `k_norm` run per head. They take `[tokens, heads, head_dim]` after
the head reshape and before attention, so q and k both arrive at unit RMS.
That's why the attention scale is `1 / head_dim` rather than `1 / sqrt(head_dim)`.
At `head_dim = 128` the two differ by a factor of 11.3, so a TPU flash kernel
written with the usual `1 / sqrt(d)` serves logits 11.3 times too large.

`use_embed_norm` puts the third one on the embedding lookup. Layer 0 reads a
normalized hidden state, not a raw table row.

`logits_mup_width_multiplier` divides the last hidden state before the head,
24.0 on Inkling and 16.0 on Inkling-Small. The head holds `vocab_size = 201,024`
rows and only the first `unpadded_vocab_size = 200,058` are real, so the rest
have to leave the sampler at the floating-point floor.

## Router

`shared_expert_sink` puts the shared experts inside the gate. The gate weight
is `[n_routed_experts + n_shared_experts, hidden_size]`, which is [258, 6,144],
so it emits two logits past the 256 routed ones. The routed logits pick the top
6 through a sigmoid plus the per-expert `use_gate_bias` correction, then those 6
chosen logits and both shared logits normalize together, and the shared experts
take their gates from that same normalization. A shared expert therefore takes
weight away from the routed ones. The bias only selects; it never reaches a
weight. `route_scale = 8.0` multiplies everything the normalization produces,
alongside a trained `global_scale` scalar.

Read as a plain 256-way top-6 router with two always-on experts beside it, the
gate comes out two rows short and the six routed weights normalize over the
wrong set.

## Short convolution

`use_sconv` puts four depthwise causal convolutions in every layer: one on `k`,
one on `v`, one after attention, and one after the MLP. Kernel 4, so each holds
3 tokens of state. Each adds its input back, so it's a residual convolution
rather than a filter, and it runs in float32 whatever the model dtype.

A sequence shard reads 3 tokens from the shard on its left for each of the
four. Over the whole stack that's 6,204 KiB per shard boundary for Inkling and
2,520 KiB for Inkling-Small, at BF16.

### What the cache manager holds

The two convolutions on the residual stream run at `hidden_size`. The two on
`k` and `v` run at that layer's own KV width, which differs between the two
attention kinds. `get_conv_state_spec()` on the model reports the widths as one
`(name, channels, tokens)` list per layer, and
`ShortConvStatePool.bytes_per_device` turns that list into the bytes one device
spends on the pool.

| | Inkling | Inkling-Small |
|---|---|---|
| Channels per local layer | 16,384 | 10,240 |
| Channels per global layer | 14,336 | 10,240 |
| Float32 bytes per sequence | 12.12 MiB | 4.92 MiB |
| At 1,024 concurrent sequences | 12.12 GiB | 4.92 GiB |

A manager that models one convolution per layer at hidden width budgets 4.64
MiB for Inkling, which is 38% of the real figure. The state is per sequence
rather than per token, so it doesn't grow with context. Against the 220 MiB of
sliding-window KV one Inkling sequence already holds, it adds 5.5%.

The runner builds the pool from `get_conv_state_spec()`: a `ShortConvStatePool`
beside the KV pool, as `memory_pools.conv_state_pool`. It takes its slots from
`HybridReqToTokenPool`, the way `RecurrentStatePool` does for KDA state, so the
scheduler hands every batch `recurrent_indices` and a request keeps one slot
from its first prefill chunk to its last decode step. Each forward reads every
row's windows at its slot, zeros for a row with no prefix, and writes the new
ones back under the same name. The model raises when the pool or the slots are
missing, rather than serve logits computed on a history 3 tokens short.

The channel axis shards over `tensor`, because a depthwise convolution never
mixes channels. The pool holds one slot per request slot, 2,048 unless
`--max-running-requests` sets the count, and its bytes come out of the KV
budget. At `--tp-size 32` the k and v windows run at the replicated 4,096
width, so one Inkling sequence holds 15.47 MiB, 0.48 MiB per chip, and the
default pool 0.97 GiB per chip. Inkling-Small holds 7.88 MiB, 0.25 MiB per
chip, and 0.49 GiB.

`sconv_chunk_pairs` writes that state as affine `(A, B)` pairs, so
[`scan/`](../scan/) carries it across devices the same way it carries a Mamba-2
state. The window is replaced whole by any chunk of 3 tokens or more, so `A` is
zero and the composition collapses to the halo.

The sconv path takes that collapse. `affine_scan.incoming_state` spends
ceil(log2 D) rounds building a prefix product that a zero `A` throws away, and
only its closing shift decides the answer. `sconv_incoming_state` does the shift
in one `ppermute`. On a 32-device sequence mesh that drops 5 rounds per
convolution, four convolutions per layer, 66 layers.
`scan/test_inkling_layers.py` runs both routes and pins them to the same output.
`shard_map` gives every shard the same number of chunks, so pad the tokens until
the chunk count divides over the mesh. Right padding with zeros leaves every
real token's output as it was. The padded chunks sit after every real token, so
no real window reads them, and each still folds to `A = 0`. Padding does reach
the windows past the sequence end. Pass `sconv_chunk_pairs` the real token count
as `tokens`, and the window leaving the last shard is the last 3 real rows, the
one a decode step reads. `tokens` counts real rows across the whole sequence, so
that call runs before `shard_map` splits the pairs, the way the test runs it.

`causal_conv` runs on one chunk, so vmap it over the chunk axis. The lines
after `sconv_chunk_pairs` sit inside `shard_map` over the sequence axis:

```python
a_chunks, b_chunks = sconv_chunk_pairs(
    x, kernel=4, chunk_size=128, tokens=real_tokens
)

# Inside shard_map over "ctx", on one shard's chunks and rows:
h_in = sconv_incoming_state(compose_local(a_chunks, b_chunks)[1], h0, "ctx")
prefix = prefix_states(h_in, replay_local(a_chunks, b_chunks, h_in))

chunks = x.shape[0] // 128
y = jax.vmap(lambda xc, pc: causal_conv(xc, weight, bias, pc))(
    x.reshape(chunks, 128, x.shape[-1]), prefix
).reshape(x.shape)
```

One row of `prefix` is the argument `mamba2.causal_conv` takes. The block is a fragment: `x`,
`real_tokens`, `h0`, `weight` and `bias` come from the caller. `scan/test_inkling_layers.py` runs
this path.

## Capture

The `layers_to_capture` hook ships inside `inkling.py`, so per-layer capture
needs only [`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch)
for the `--enable-return-hidden-states` flag and the reshape.
`_setup_hidden_states_capture` reaches the list through
`getattr(self.model, "model")` and the flag on the outer class, which is where
`InklingTextModel` and `InklingForCausalLM` put them.

Inkling adds the residual inside the layer rather than carrying a
`(hidden, residual)` pair. The capture reads `hidden_states` directly, so it
skips the `hidden_states + residual` form the other hooks need.

66 layers at 6,144 dim is 792 KiB per token of host traffic. Inkling-Small is
42 layers at 4,096 dim, so 336 KiB. That's every slot. `capture_activations.py`
hands `--layers` to the engine as `--return-hidden-states-layers`, and one kept
slot moves 12 KiB per token, 8 KiB on Inkling-Small.

## Notes

The KV cache only grows on the global layers. Inkling holds 44 KiB per token
across its 11 full-attention layers, so a 1M-token sequence costs 44 GiB before
KV replication. The 55 local layers hold a fixed 220 MiB of window per
sequence, whatever the context length.

`scan/test_inkling_layers.py` checks the plan against the published fields of
both sizes on CPU, with an 8-device mesh for the convolution state. It pins the
numbers on this page: the group boundaries, the two MLP widths, the KV head
split, the tau curve, the halo, the capture traffic, the KV growth, the window,
the conv-state table, the 4.64 MiB a naive manager budgets and its 38% share,
the 5.5% the state adds to the window, and the 1 GiB one layer's bias holds at
8,192 tokens. The published fields live in `scan/inkling_layers.py` as literals
and a second time in the test, transcribed from `config.json`, and one check
compares the two copies. The run needs no checkpoint download.

```bash
source .venv/bin/activate
uv pip install -r scan/requirements.txt
python3 scan/test_inkling_layers.py
```

[`upstream/models/test_inkling.py`](../upstream/models/test_inkling.py) runs
the model through the engine's serving path on CPU. It builds the tree the TPU
VM serves from with this patch on top, and has `transformers` write a
four-layer random checkpoint with `save_pretrained`, carrying every
architecture flag the real config sets and both attention kinds. It serves that
checkpoint through `JAXModelLoader`, `ModelRunner` with its own pools and the
startup precompile, and batches `ScheduleBatch` builds, on a two-device CPU
mesh. Against `transformers` it compares every layer input, the prefill logits,
five decode steps, the prompt logprobs, two sequences packed into one prefill,
and a prompt split across two prefill passes and then decoded, on window slots
earlier requests held. Then a prefill and two decode steps at `--tp-size 8`,
past both kinds' KV head counts, and a prefill at `--ep-size 2`. Worst max
absolute error is 8.6e-06, correlation 1.000000000000.

Around those it checks the parts a comparison can't reach: the conv-state spec,
the pool the runner builds from it and the bytes it takes from the KV budget,
one allocator compile per window shape when a pool is built or cleared, every
plain prefill and decode step running a program the startup precompile built,
the interleaved gate and up rows, the dense and shared weights sharded over the
mesh, the padded vocabulary, a backend without the `position_bias` kwarg and a
step without the conv pool, which both have to raise, the ten launch settings
the worker refuses before the loader looks for a weight file, PD and the EPLB
flags among them, the quantization config and `--enable-dp-lm-head` reaching the
layers, a dummy-weight prefill, a dummy-weight load at `--tp-size 8` that never
holds a whole routed-expert stack on one device, an import on a `transformers`
without Inkling, all three entry classes, the router's sink, the published
expert width of both sizes, and every source key against the published
safetensors index of both checkpoints. A NaN logit has to fail the gate every
comparison uses.

Two checks cover the `v5p-64` failures. The first compiles one position-bias
attention call at an 8,192-slot and a 65,536-slot `cache_loc` bucket and needs
the same temporaries at both; the whole-bucket pass, its control, grows eightfold.
The second loads a checkpoint in two processes of 4 simulated devices each, one
8-device mesh, from a directory that reads as a gcsfuse mount. It counts every
byte each process pulls through `safetensors` or a plain file read, and each has
to stay within half of every sharded tensor plus each replicated tensor once.
Its controls, the single-process warm-up and every block cut from its whole
tensor, both have to go over.

A third section runs a config with every published field, sizes shrunk, at
`--tp-size 8`: 2 query heads a device, 4 copies of each global KV head and 2 of
each local one, the counts Inkling runs at `--tp-size 32`. A 24-token prompt,
past the 16-token bias extent and the 8-token window, has to match
`InklingForCausalLM` at every layer input and the prefill logits, and match
`scripts/check_capture.py`'s own reference, which the chip check gates on. That
reference's control norms the embeddings twice, as `transformers` 5.17's
multimodal class does, and has to miss layer 0. Two processes of 4 devices then
load the same checkpoint, and every block each holds has to equal the
one-process load. Its control repeats the KV heads in tiled order.

Sixteen negative controls follow, and each has to move the logits by more than
1e-3, about a hundred times the worst clean error.

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/models/test_inkling.py
```


## Measured on a v5p-64

A `v5p-64` Spot slice (32 chips, 8 hosts) in us-east5-a, 2026-09-27. sglang-jax eb061d8 with
`sglang-jax-877.patch`, the steering patches, `multihost-hidden-states.patch` and
`models/inkling-model.patch` (blockwise native attention over each request's own KV, per-host
reads), transformers 5.17.0 on the hosts, built by the `inkling` row of `scripts/multihost_run.sh`:
engine args `attention_backend=native disable_radix_cache=True`, `mem_fraction_static=0.8`,
`--tp-size 32`. Every host read the weights from a GCS bucket in us-east5 through gcsfuse with
range-read caching off.

**Capture check**, `scripts/check_capture.py` against the float32 CPU reference
`refs/ref-inkling.npz` rebuilt at 18:08Z with `embed_norm` run once (transformers 5.17's
`InklingForConditionalGeneration` runs it twice; upstream fixed that in transformers 3384908511,
not yet on PyPI), 18:35 to 20:40Z:

| Prompt | Layers | Worst ratio to the BF16 floor | Worst Pearson | Control (each slot against the next layer) | Result |
|---|---|---|---|---|---|
| 440 tokens | 66, all within the floor | 1.07 (layer 2); most layers 0.5 to 0.8 | 0.9937 (layer 65) | 3.58x, detected | pass |
| 1,321 tokens, split across prefill passes | 66, all within the floor | 0.99 (layer 2) | 0.9841 (layer 61) | 2.82x at layer 61, under the 3.0x bar | not detected |

`check_capture.py` exits FAILED because the second prompt's control reads 2.82x: at layer 61 the
stream entering layer 62 is almost the stream entering layer 61, so the per-token test can't tell
them apart by 3x. Every captured layer is within the floor on both prompts. The capture is saved,
so another control can gate it again without the engine, through `check_capture.py --capture-npz`.

Two earlier launches on the same slice failed: 1.71 TB of HLO temporaries from native attention over
the whole cache_loc bucket (fixed by the blockwise pass), then every layer off from slot 0 because
the reference ran `embed_norm` twice. Loading takes about 2 hours per launch (about 7 minutes of
dense weights, then the routed experts at 45 MB/s to 1.4 GB/s per host through gcsfuse).

To reproduce the check, stage the checkpoint and its reference as
[Across hosts](../README.md#across-hosts) describes, then run from the repo root:

```bash
BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE inkling setup check
```

The same command with `measure` in place of `check` takes throughput and HBM, about 4.5 hours for
two engine loads. Nobody has run it yet.

Notes: nobody has replayed the check block as written from a clean start. The run above came from
the same script and row. `check` exits 1 on the control above, and the script stops at the first
step that fails, so put `measure` in a run of its own. The 4.5 hours is projected from the 2-hour
loads above.

