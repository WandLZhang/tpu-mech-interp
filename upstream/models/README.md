# Model patches

Model implementations for [`sgl-project/sglang-jax`](https://github.com/sgl-project/sglang-jax)
that upstream doesn't carry, and the `layers_to_capture` hook for each. Every patch is a diff
against `eb061d8`. It applies there alone, on top of `sglang-jax-877.patch`, or on the tree
`scripts/bootstrap_tpu_vm.sh` builds, which adds both steering patches. A hook does nothing
without the flag 877 adds, so the block below applies 877 first. The patches carry the diff and
the rationale, not the commit trailers, so apply with `git apply` and write your own commit
message.

```bash
REPO=$PWD
git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
git checkout eb061d8
git -c user.name=you -c user.email=you@example.com am < "$REPO/upstream/sglang-jax-877.patch"
git apply "$REPO/upstream/models/inkling-model.patch"
```

| Patch | Adds | Serves | Test |
|---|---|---|---|
| [`inkling-model.patch`](inkling-model.patch) | `models/inkling.py`, `configs/inkling.py`, `mem_cache/short_conv_state_pool.py`, position bias and sliding-window slot mapping in the native backend, KV-head padding that reads the target path, a worker hook that refuses flags before the load | `thinkingmachines/Inkling`, `thinkingmachines/Inkling-Small` | [`test_inkling.py`](test_inkling.py) |
| [`kimi-k3-model.patch`](kimi-k3-model.patch) | `models/kimi_k3.py`, `utils/quantization/mxfp4.py`, `KimiK3Config`, `WeightMapping.narrow` for the padded per-head `A_log`, the `NativeAttention` fixes for CPU runs and data parallelism, and the output sharding of the EPLB dispatch gathers | `moonshotai/Kimi-K3` | [`test_kimi_k3_model.py`](test_kimi_k3_model.py) |
| [`kimi-k3-capture-hook.patch`](kimi-k3-capture-hook.patch) | the hook on `models/kimi_k3.py` | `moonshotai/Kimi-K3` | [`test_kimi_k3_model.py`](test_kimi_k3_model.py) |
| [`nemotron3-model.patch`](nemotron3-model.patch) | `models/nemotron_h.py`, `configs/nemotron_h.py`, `layers/attention/mamba/`, a build-time refusal of the expert-placement flags | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | [`test_nemotron_h_model.py`](test_nemotron_h_model.py) |
| [`nemotron3-capture-hook.patch`](nemotron3-capture-hook.patch) | the hook on `models/nemotron_h.py` | the same | [`test_nemotron_h_model.py`](test_nemotron_h_model.py) |
| [`gpt-oss-model.patch`](gpt-oss-model.patch) | `models/gpt_oss.py`, per-expert bias and the gpt-oss gate in `EPMoE`, YaRN `truncate`, the `gmm` v1 bias after the activation rescale, and the output sharding of the EPLB dispatch gathers | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | [`test_gpt_oss_model.py`](test_gpt_oss_model.py) |

One clone takes one model patch. `gpt-oss-model.patch`, `kimi-k3-model.patch` and
`nemotron3-model.patch` all edit `python/sgl_jax/srt/layers/moe.py`, and any two of them collide
there. The gpt-oss and Kimi K3 patches also carry the same `eplb/expert_location.py` hunk.
`inkling-model.patch` collides with the Kimi patch in `layers/attention/native_backend.py` and
with the Nemotron patch in `model_executor/model_runner_kv_cache_mixin.py`. With one patch
applied, `git apply --check` refuses each of the others, except gpt-oss with inkling in either
order. A serving instance runs one model, so clone per model. On a TPU VM,
`bash scripts/bootstrap_tpu_vm.sh --model NAME` builds the tree with one model's patches, `NAME`
being `gpt-oss`, `inkling`, `kimi-k3` or `nemotron3`. A rerun that asks for what the tree already
holds keeps it: the same name, or no `--model` on a tree built without one. A run that asks for
another model, or for none when the tree holds one, moves it aside and builds the tree it asks for.

Each test runs on CPU. Every test first builds the tree the TPU VM serves from, `eb061d8` with
`sglang-jax-877.patch` and both steering patches, and applies its patch on top. The gpt-oss,
Inkling and Nemotron 3 tests build it through `scripts/cpu_engine.build_tree`. Each test builds a
tiny random config that carries the architecture flags the real config sets, and gives the JAX
model and an independent reference the same weights. The section for each patch lists what its
test compares.
Point `SGLANG_JAX_REPO` at a local clone that holds `eb061d8` to skip the download. Every test
clones it with `--shared` before it patches, so the local clone stays clean. `SGL_COMMIT` picks
another commit, as it does for every gate.

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/models/test_inkling.py
```

`transformers` ships Inkling from 5.14 and `nemotron_h` from before 5.12. The requirements file
asks for 5.17, the release the CPU gates ran on. The packages after `xxhash` are import-time
dependencies of the `sglang-jax` modules the tests pull in. `pybase64` and `llguidance` come in
through the engine's `ModelRunner`, which `test_inkling.py` and Kimi check 10 run. The six after
them come in through the full `Engine` and its HTTP server, which the capture-hook and steering
tests and the `scripts/` gates start.

## `inkling-model.patch`

Nine files. `models/inkling.py`, `configs/inkling.py` and `mem_cache/short_conv_state_pool.py`
are new. The other six teach the native attention backend to read a per-token position bias and
to map sliding-window cache slots, register the config, let a checkpoint name its KV projections
something other than `k_proj` and `v_proj`, have the runner build a per-request pool for the
short-convolution windows, have precompile build the batches that address it, and have the
worker ask the model class about the flags before it loads a weight.

`EntryClass` registers three names. `InklingForCausalLM` is the decoder.
`InklingForConditionalGeneration` matches the architecture the published `config.json` advertises
and serves the text path of that checkpoint. `InklingMTPForCausalLM` builds one layer of the
multi-token-prediction head as a draft model, selected with `mtp_layer_idx`. The worker refuses
speculative decoding for Inkling, because no convolution window rolls back past a rejected draft
token, so nothing serves the draft yet.

Weight keys follow the published checkpoint, which stores the decoder under `model.llm.` and the
draft head under `model.mtp.`.

[`../../models/inkling.md`](../../models/inkling.md) carries the layer plan, the positional
scheme, the normalizations, the router, the short-convolution state the cache manager holds, and
the serving slice.

### How the bias reaches the logits

`InklingAttention` builds a `[tokens, heads, extent]` bias-vs-distance profile and hands it to
`RadixAttention` as a `position_bias` keyword, which `RadixAttention` already forwards to the
backend. `NativeAttention` reads it, picks the entry at each query-key distance, and adds it to
the logits. `AttentionBackend.supports_position_bias` says which backends do that, and the model
raises when the selected backend doesn't, so a run can't serve logits that quietly dropped the
bias. `--attention-backend` defaults to `fa`, which doesn't read the kwarg, so serving takes
`--attention-backend native`, and the worker refuses any other backend before it loads a weight.

The dense form materializes `[queries, heads, keys]`. At 8,192 tokens, 64 heads and a 1,024
extent that's 1 GiB per layer in BF16 for the profile alone, and the expansion to one entry per
query-key pair is larger again. A flash kernel wants the bias folded into its score step instead.

### Sliding-window slots

`SWAKVPool` keeps the sliding-window layers in a second, smaller pool, and
`SWATokenToKVPoolAllocator` maps each full-pool slot to a slot there. The FA backend maps its page
indices on the host in `get_forward_metadata`. `NativeAttention` does the same for `cache_loc`
and `out_cache_loc`, carries the mapped slots in its forward metadata, and writes and reads a
sliding-window layer at them, on the sub-pool. The mapping stops being the identity once the
window pool's free list wraps onto slots an evicted or finished request held.

The runner sets the mapping on the attention backend as a numpy array, whichever backend serves.
In a process that builds a second runner, JAX's `donation_vector` cache compares the new backend
with the old one, reaches the array, and raises. `jax.clear_caches()` leaves that cache alone, so
`test_inkling.py` clears it between workers. An Inkling server builds one runner, because the
worker refuses speculative decoding.

### The config class

`configs/inkling.py` registers `InklingServingConfig` for `model_type: inkling_mm_model`. The
published `text_config` writes the routed-expert width as `intermediate_size` and the dense width
as `dense_intermediate_size`. The stock `InklingTextConfig.__post_init__` moves the dense width
onto `intermediate_size` and leaves `moe_intermediate_size` at its class default of 3072, which
is Inkling's expert width and not Inkling-Small's 2048. `InklingServingConfig` reads the field
before that rewrite. `load_expert_weights` compares the checkpoint's own width against the one
the config reports and raises on a mismatch.

The class subclasses the `transformers` Inkling config, which ships from 5.14, and `sglang-jax`
pins 5.12. `hf_transformers_utils.py` imports it inside a `try`, so a tree carrying this patch
still serves every other model on 5.12, and an Inkling checkpoint stops in `AutoConfig`.

### Conv state through the pools

`get_conv_state_spec()` reports the widths as one `(name, channels, tokens)` list per layer. When
a model reports a spec, `ModelRunner` builds a `ShortConvStatePool` from it beside the KV pool:
one `[slots, channels, tokens]` buffer per window, the channel axis sharded over `tensor`. The
pool takes its slots from `HybridReqToTokenPool`, the way `RecurrentStatePool` does for KDA state,
so the scheduler hands every batch `recurrent_indices` and a request holds one slot from its first
prefill chunk to its last decode step. `tp_worker` gives the precompile batches the same indices.
`ShortConvStatePool.bytes_per_device` takes the pool's bytes out of the KV budget. The pool
allocates its buffers through the zero-allocator cache `RecurrentStatePool` keeps, one compiled
program per window shape, so building it and clearing it on `flush_cache` compile 3 programs for
Inkling's 264 windows, not 264.

Each forward reads every row's windows at its slot, zeros for a row with no prefix, runs each
convolution on its own channel shard in a `shard_map`, and returns the new buffers as
`conv_state_pool` beside `token_to_kv_pool`. The model raises when the pool or the slots are
missing.

The radix cache, speculative decoding and `--enable-recurrent-extra-buffer` would each open a
step on windows that don't match its prefix. PD disaggregation, `--pd-disaggregation` or
`--disaggregation-mode prefill|decode`, hands a request's KV to the decode side and leaves its
windows behind, so every decode step would convolve over zeros. Data parallelism splits the batch
into per-rank sections, and the convolutions read it as one packed list. `ModelWorker` calls the
model class's `check_server_args` before it builds the runner, and Inkling's refuses all of them
there, before the loader looks for a weight file. It also refuses any attention backend but the
native one, and the three EPLB flags, `--ep-dispatch-algorithm`, `--ep-num-redundant-experts` and
`--init-expert-location`. Inkling's router picks logical experts with its own top-k and the loader
fills one slot per expert, so those flags would change nothing. The runner refuses the pool's
problems again when it builds the pool, and the model refuses a `data` axis above 1 when it
builds, for a runner built without the worker.

### Loading the stacked tensors

Six families of tensor hold more than one target each, so `load_stacked_weights` reads them
outside the mapping table: the dense `w13_dn` and `w2_md` pair, the stacked routed experts, and
the stacked shared pair. Together they're 164 of the 878 `model.llm.*` keys on Inkling-Small and
nearly all of its parameter bytes, so a missing one raises rather than leaving a zeroed array
behind.

The fused `w13` tensors interleave gate and up: gate on the even rows, up on the odd ones. That's
how `transformers` reads them, through `Interleave` in its conversion mapping, and how
`save_pretrained` writes them. The routed experts come in through `jax.make_array_from_callback`
against the target `NamedSharding`, so each device reads only the block it owns and a 19 GB
tensor never lands whole on one device. The callback reads the file rows its block covers in one
contiguous slice, transposes the checkpoint's `[experts, out, in]` to the kernel's
`[experts, in, out]`, keeps its half and holds the other for `wi_1`. A safetensors slice with a
step would read fewer bytes, but the numpy reader in 0.6, the version `sglang-jax` pins, drops the
step. The dense and shared weights go through the same callback from a host array, into their
declared shardings, because `load_weights` runs outside the mesh. In dummy mode the model fills
the routed experts on `EPMoE`'s own mesh before the loader's dummy pass. That pass drops the
`expert` axis the model mesh doesn't carry, so a stack it filled would sit whole on every chip,
29.0 GB a layer on Inkling, and the `shard_map` would refuse it.

### KV heads

Each layer keeps its own KV head count, 16 on a local layer and 8 on a global one.
`set_num_token_hybrid` walks `model.model.layers` and reads `self_attn.attn.sliding_window_size`
off each one, which `RadixAttention` holds as None on a global layer, and sorts the stack into
`swa_attention_layer_ids` and `full_attention_layer_ids`. The runner then builds an `SWAKVPool`
taking `swa_head_num` from `swa_num_key_value_heads` and `head_num` from the global count, so
neither kind pads up to the other.

Tensor parallel wider than a layer's own KV count needs the checkpoint's heads replicated, which
is what `WeightMapping.kv_head_padding` asks for and what `_apply_kv_head_padding` does. It knew
the two head counts apart already, through `swa_head_dim` and `get_swa_weight_params`, but it
tested only the checkpoint key for `k_proj` and `v_proj`. Inkling names those tensors
`attn.wk_dv.weight` and `attn.wv_dv.weight`, so the patch has the test read the target path as
well. The k and v convolutions run at the replicated width too, and `widen_kv_convolutions`
repeats each checkpoint head's `head_dim` rows of `attn.k_sconv.weight` and
`attn.v_sconv.weight` in the order `_apply_kv_head_padding` repeats the heads of `wk_dv`.

### The runner's fields

`ModelRunner.load_model` writes `ep_size`, `moe_dp_size` and `enable_dp_lm_head` onto the config
it loads, and `ModelConfig` writes `quantization_config` there. The published config nests the
decoder under `text_config` and the model reads that block, so `text_config()` copies the four
across. `--ep-size` reaches every `EPMoE`, an int8 config quantizes the routed experts, and
`--enable-dp-lm-head` reaches the head.

### The head

`InklingForCausalLM` divides the last hidden state by `logits_mup_width_multiplier` before the
head. The head holds 201,024 rows and 200,058 are real. `InklingLogitsProcessor` pushes the rest
to the floor in `_get_logits`, so the sampled logits, the prompt logprobs and their top-k all read
the real vocabulary.

### Capture

The `layers_to_capture` hook is in `inkling.py` already, in the four parts
[`../capture-hooks/README.md`](../capture-hooks/README.md) describes: the list on
`InklingTextModel`, the append in the layer loop, `capture_aux_hidden_states` on every entry
class, and `aux_hidden_states` threaded into the logits processor.
`_setup_hidden_states_capture` in [`../sglang-jax-877.patch`](../sglang-jax-877.patch) reaches the
list through `getattr(self.model, "model")` and the flag on the outer class, so it needs no
further change.

Inkling adds the residual inside the layer rather than carrying a `(hidden, residual)` pair. The
capture reads `hidden_states` directly, so it skips the `hidden_states + residual` form the other
hooks need.

### What `test_inkling.py` covers

`transformers` owns the reference, and every number in the tables below comes out of it. It
writes the checkpoint too: a random `InklingForConditionalGeneration` saved with
`save_pretrained`, which lays the tensors out under the published names and with the published
interleaved gate and up rows. Nothing in the test renames or reorders a tensor. The reference
logits come from `InklingForCausalLM` holding the same language model and head, because in
transformers 5.17 the multimodal class applies `embed_norm` twice, once in `InklingModel.forward`
and again in the text model it calls. SGLang's PyTorch model applies it once, like this port.

The test builds the tree the TPU VM serves from through `scripts/cpu_engine.build_tree`,
`sglang-jax-877.patch` through `git am` and both steering patches after it, the steps
`bootstrap_tpu_vm.sh` runs, applies this patch on top, and serves the checkpoint through the
engine on a two-device CPU mesh:
`JAXModelLoader`, `ModelRunner` with its Explicit mesh and its own pools, the startup precompile,
and `ScheduleBatch` over the `SWAChunkCache` the scheduler builds under `--disable-radix-cache`.
The batches carry the scheduler's `SpeculativeAlgorithm.NONE`, as the precompile's dummies do, so
the prefills and decode steps it compares run the precompiled programs. The KV pool is small and
gives the sliding window a quarter of it, so the window pool's free list wraps and its slots stop
matching the full pool's.

| | Max abs error | Correlation |
|---|---|---|
| Layer 0 input, local and dense | 2.4e-07 | 1.000000000000 |
| Layer 1 input, global and routed | 3.4e-06 | 1.000000000000 |
| Layer 2 input, local and routed | 5.7e-06 | 1.000000000000 |
| Layer 3 input, local and routed | 8.1e-06 | 1.000000000000 |
| Prefill logits | 3.6e-07 | 1.000000000000 |
| Five decode steps, worst | 3.6e-07 | 1.000000000000 |
| Prompt logprobs | 4.8e-07 | 1.000000000000 |
| Two sequences packed into one prefill, worst | 3.1e-07 | 1.000000000000 |
| A 13-token prompt in two passes and three decode steps, worst | 3.3e-07 | 1.000000000000 |
| Prefill and two decode steps at `--tp-size 8`, worst | 3.9e-07 | 1.000000000000 |
| Prefill at `--ep-size 2` | 2.2e-07 | 1.000000000000 |

The layer inputs come out of the `--enable-return-hidden-states` capture on a second request for
the same prompt, and the engine's greedy tokens match `transformers` on every decode step. A
request that asks for hidden states or prompt logprobs compiles a variant of its own on first use,
as it does for any model, so the decoded request asks for neither. The packed prefill cuts 8
tokens into a 3-token and a 5-token sequence and compares each against its own reference run, so
the convolution, the mask and the relative bias all have to stop at the boundary. The chunked
prompt runs at `--chunked-prefill-size 8`, so its second pass opens on the windows and the KV the
first left, on window slots other requests held. At `--tp-size 8` both attention kinds replicate
their KV heads, four copies of each global head and two of each local one, the ratios Inkling
runs at `--tp-size 32`.

Around those it checks what a comparison can't reach.

| Check | What it pins |
|---|---|
| The pools | the runner builds `SWAKVPool`, `ShortConvStatePool` and `HybridReqToTokenPool`, and the scheduler's cache is `SWAChunkCache` |
| The precompile | every plain prefill and decode step at `--tp-size` 2 and 8 runs a program the startup precompile built, with no backend compile; a prefill whose batch carries `spec_algorithm=None` compiles, so the count can fail |
| `get_conv_state_spec()` | the widths against a table written out by hand, one pool slot per request slot, and the pool's bytes on device 0 against `bytes_per_device` and the hand count |
| Pool allocations | building a 48-window pool in two shapes compiles at most two allocators, and clearing it compiles none |
| The layout | `save_pretrained` puts gate on the even rows, and the dense gate, a routed gate and a shared up load equal to `transformers` |
| Placement | the dense and shared weights sit on the mesh in their declared shardings |
| The padded vocabulary | head rows past `unpadded_vocab_size` reach the sampler at the floor, and none reaches the prompt top-k |
| Contracts | a backend with no `supports_position_bias`, a step with no conv pool and a batch with no request slots each raise |
| Refusals | on a copy of the checkpoint without its weights, the worker refuses the radix cache, `--dp-size 2`, the `tt` backend, speculative decoding, the recurrent extra buffer, Pathways PD, a PD decode server and each of the three EPLB flags, each by name; flags Inkling serves stop in the loader on the same copy, so each refusal comes first; a runner built without a worker still refuses the radix cache |
| KV head replication | at `--tp-size 8` both kinds' k projections and k convolutions widen to 1,024 from the checkpoint's 256 and 512 |
| The runner's fields | `--ep-size 2` reaches every `EPMoE`, an int8 config reaches the experts, and `--enable-dp-lm-head` reaches the head |
| Dummy weights | a `--load-format dummy` prefill runs; at `--tp-size 8` the routed experts already sit in `EPMoE`'s layout right after the loader's dummy pass, and device 0 never holds more of what the load allocates than the loaded model holds there; control: with the model's expert fill off, the pass leaves the stacks whole on every device |
| The logit gate | a NaN logit fails the gate every comparison in the test uses |
| An older transformers | with `transformers.models.inkling` hidden, the engine imports and leaves Inkling unregistered |
| `EntryClass` | all three names build, and one draft model per MTP layer lands on the attention kind `mtp_local_layer_ids` gives that index |
| The sink | moving a shared gate row moves the routed weights |
| The expert width | 3072 on Inkling and 2048 on Inkling-Small, against the stock config's single 3072 |
| The checkpoint keys | every generated source key exists in the published index of both repositories, and the uncovered `model.llm.*` keys are the stacked tensors and nothing else |

Sixteen negative controls follow. Each breaks one thing, runs the same batch through the runner's
model on its own pools without writing them back, and has to move the logits by more than 1e-3,
about a hundred times the worst clean error. A control that moves nothing fails the run.

| Control | Moves logits by |
|---|---|
| muP width multiplier dropped | 7.9e-01 |
| Window KV read at the full-pool slots | 5.9e-01 |
| Sliding window removed | 2.5e-01 |
| `k_sconv` and `v_sconv` swapped | 2.2e-01 |
| Windows restarted from zeros | 1.6e-01 |
| Relative bias, one entry | 9.2e-02 |
| Log scaling turned off | 5.5e-02 |
| Routed gate and up read as halves | 4.3e-02 |
| `embed_norm` scaled | 3.8e-02 |
| Dense gate and up read as halves | 3.3e-02 |
| Attention scale at 1/sqrt(head_dim) | 2.5e-02 |
| Dense MLP gate and up swapped | 1.4e-02 |
| `mlp_sconv` removed | 1.2e-02 |
| `k_norm` skipped | 8.9e-03 |
| Shared gate row | 7.6e-03 |
| `q_norm` skipped | 4.1e-03 |

The two window controls run on the chunked prompt's second pass. `q_norm`, `k_norm` and the
attention scale go together: the norms bring q and k to unit RMS per head, which is why the scale
is `1 / head_dim`, and a port that drops the norms or reaches for `1 / sqrt(head_dim)` by habit
has to be caught.

The test takes about 4 minutes on CPU.

## `nemotron3-model.patch`

Twelve files. Five are new: `models/nemotron_h.py`, `configs/nemotron_h.py` and the three in
`layers/attention/mamba/`. Six are small edits to shared code, and one rewrites an upstream test
the patch would break.

`EntryClass` registers `NemotronHForCausalLM`, which both published sizes advertise. One file
serves Super and Ultra; every difference between them is a number in `config.json`.

[`../../models/nemotron3-super.md`](../../models/nemotron3-super.md) and
[`../../models/nemotron3-ultra.md`](../../models/nemotron3-ultra.md) carry the layer plan, the
state the cache manager holds, and the serving slice.

### The layer stack

A block is one norm, one mixer, one residual add. The mixer is a Mamba-2 layer, a GQA attention
layer, a LatentMoE layer or a dense MLP, so the stack never repeats a body and can't be one scan
over layers. `configs/nemotron_h.py` reads `layers_block_type` or `hybrid_override_pattern` and
hands the runner `linear_layer_ids`, `full_attention_layer_ids` and `linear_state_params` under
the names the hybrid recurrent path already reads. `num_hidden_layers` is the length of the block
list; a config that also ships the count and disagrees is rejected. A write to
`num_hidden_layers` on a built config cuts the block list to its first entries, which is how
`--model-layer-nums` and a `num_hidden_layers` override build a shorter stack. Stock
`transformers` 5.17 saves the two mixers as `linear_attention` and `full_attention`, and the parser
reads those as `mamba` and `attention`, so a re-saved checkpoint loads too.

That layer field has no default, so `NemotronHConfig()` raises. `PretrainedConfig.to_diff_dict`
builds a default instance to diff against, `to_json_string` calls it, `__repr__` calls that, and
`from_dict` logs the config it just built, so a class that raises on an empty construction can't
load its own `config.json` through `AutoConfig`. `has_no_defaults_at_init = True` is the
transformers answer and the class sets it. Check 9 of the test resolves both published sizes.

The stack builder emits one block per entry of the list. No two adjacent layers share a type on
either published size, so nothing stacks into a scan and the loop stays flat.
[`../../scan/nemotron3_layers.py`](../../scan/nemotron3_layers.py) carries the grouping analysis,
including the Nano runs that would fill a scan.

### No position encoding

The attention layers carry no rotary table and no learned bias. Both published `config.json` files
ship `rope_theta` and `partial_rotary_factor`, NVIDIA's `configuration_nemotron_h.py` declares
neither, the reference model reads neither, and this port does the same: they arrive on
`**kwargs`. Order reaches attention through the Mamba-2 layers below it. Applying RoPE anyway
moves the logits by 3.8e-04 against the reference, which is what the per-layer comparison caught.

### The recurrent state

`layers/attention/mamba/mamba2.py` is the served subset of
[`../../scan/mamba2.py`](../../scan/mamba2.py): the chunk fold, the per-token outputs and the two
layout helpers. The sequence-sharded composition stays in [`../../scan/`](../../scan/), where the
sharding study lives, so the patch carries nothing the engine doesn't call.
`mamba2_backend.py` is new and owns everything that touches the cache.

Prefill packs several requests onto one token axis, so the chunk grid has to cut at every request
start. `chunk_grid` builds it from `cu_q_lens`: chunk `c` holds up to `chunk_size` tokens of one
request, a request whose length isn't a multiple closes on a partial chunk, and padding slots
carry `dt = 0` so they add nothing and decay by one. A padding slot reads `x` and `B` as zero
too, because the gather clamps it onto the next request's first tokens. `open_requests` folds
each request's cached state into the `B` of its first chunk, and the scan restarts from that `B`
there, so one scan carries the whole batch.

The restart and the padding are both selects. A multiply by zero gives the same numbers on finite
values, but `0 * inf` is NaN. As multiplies, the restart would carry an overflow in one request's
state into every request after it, and the padding slots would carry a NaN in a request's first
tokens into the request before it.

`replay_scalar` runs that scan. `A` is a scalar times the identity, so it carries one number per
chunk per head where a dense `[K, K]` form would carry 16,384 at `ssm_state_size=128`.

`dt` is clamped to `mamba_dt_limit`, whose `(0.0, inf)` default neither published `config.json`
overrides. Megatron-LM trains with that pair, and NVIDIA's CUDA kernels and vLLM serve with it.
The torch paths floor `dt` at `time_step_min`, 0.001, instead: `transformers` 5.17 in prefill and
not in decode, and NVIDIA's `torch_forward` in both. On the published weights, 212 of Super's
5,120 Mamba-2 heads have `softplus(dt_bias)` below 0.001, so that floor binds there. The test's
random weights keep `dt` above 0.5, so its comparison with `transformers` never reaches the floor.

The conv runs through `short_convolution`, the same packed varlen conv the KDA and GDN backends
use.

### Track slots

With `--enable-unified-radix-tree --enable-recurrent-extra-buffer`, the scheduler hands the
backend a track slot for each request whose forward ends on a track boundary, and the radix tree
adopts that slot as the state of the prefix. The scheduler cuts every forward that crosses a
boundary so it ends there, so the backend writes the state the forward leaves to the track slot
as well as the running slot, the way the KDA backend does. A row whose mask is 0, and the padding
dummy at slot 0, keep what their slot held. With the flag off the metadata is None and the
backend compiles as it did before. The measured runs disable the radix cache, so this path has
run on CPU only.

### What the shared edits do

| File | Change |
|---|---|
| `layers/moe.py` | `EPMoE(gated=False)` and a `relu2` activation, for experts with one input matrix, which quantize those two matrices under a MoE quantization config |
| `mem_cache/recurrent_state_pool.py` | the SSM state buffer reads `head_k_dim` for its last axis, and the sharding asserts cover the axes that shard |
| `layers/attention/fla/group_rmsnorm.py` | a second layout for an RMS group that straddles shards |
| `model_executor/model_runner_kv_cache_mixin.py` | the same width in the per-request byte count, and `nemotron_h` resolves to its own recurrent config |
| `layers/attention/hybrid_linear_attn_backend.py` | `attn_backend_wrapper` builds `Mamba2AttnBackend` for `nemotron_h` |
| `hf_transformers_utils.py` | registers `NemotronHConfig` for `model_type: nemotron_h` |
| `test/layers/test_group_rmsnorm.py` | the test that asserted the old `GroupRMSNorm` error checks the split layout instead |

The pool allocated the SSM state square, `head_dim` by `head_dim`. KDA and GDN run those two
widths equal so it reads as one number there, but Mamba-2 runs `mamba_head_dim=64` by
`ssm_state_size=128` and a square slot holds half the state. Giving the last axis its own field
leaves both existing callers alone.

Two asserts capped the tensor axis at 8. The pool required `num_k_heads`, which carries the 8
Mamba-2 groups, to divide the recurrent axis, and the recurrent buffer shards `num_heads` and
never touches `num_k_heads`; the assert now covers the two segments of the conv buffer that do
shard. `GroupRMSNorm` required the tensor axis to divide `num_groups`, which is 8, and the gated
norm's RMS reduction does need whole groups; it now takes a second layout when a group straddles
shards, contracting the squares against a `[num_groups, hidden_size]` selector so one all-reduce
of a `[tokens, num_groups]` array closes the sum. A mesh where groups divide evenly keeps the old
path unchanged. Check 8 of the test runs a two-group norm on a four-way axis.

Upstream's `test_rejects_tp_not_dividing_num_groups` asserted the error the patch removes, so the
patch rewrites it into three tests: the split layout against an fp64 reference at tp=8 over 2
groups, with a per-shard RMS as the control; the same layer built and traced at tp=16 over 8
groups on an `AbstractMesh`, which runs on the one-chip CI runner; and the error the constructor
still raises when the tensor axis doesn't divide `hidden_size`. All six tests pass on 8 CPU
devices, where the unpatched file fails one.

Nemotron 3 experts aren't gated: one `up_proj`, `mlp_hidden_act`, one `down_proj`, and they run
in a latent of `moe_latent_size` rather than at `hidden_size`. The router reads the full-width
token first, then `fc1_latent_proj` drops it, which is the order the reference uses. The `gated`
flag drops `wi_1` and its GEMM; a gated layer compiles to the same HLO it did before.

The experts take the `QuantizationConfig` that `ModelConfig` resolves, and nothing else that sits
on the config, so `--quantization-config-path` reaches them and an ungated layer quantizes its
two matrices. `QuantizationConfig.from_yaml` requires linear rules as well, and two things keep
those rules from breaking the model. The depthwise conv kernel is a bare parameter rather than a
`LinearBase` that a rule would swap for a `QuantizedLinear` with no `weight` for the backend to
read. And `fc2_latent_proj` gets the experts' output split over the tensor axis, the only input
layout a row-parallel `QuantizedLinear` takes. A pre-quantized checkpoint is refused at build
time, since the mapping names no scale tensors.

The model refuses `--ep-dispatch-algorithm`, `--init-expert-location` and
`--ep-num-redundant-experts` before it builds. The engine builds expert-location metadata only for
a config that sets `num_experts`, and this one names the count `n_routed_experts`. The weight
mapping loads every expert in checkpoint order, so those flags would change nothing.
`NemotronHForCausalLM.check_server_args` holds the refusal, and `__init__` runs it on the server
args `ModelRunner.load_model` publishes before it builds the model.

### What `test_nemotron_h_model.py` covers

The test builds the tree the TPU VM serves from with `scripts/cpu_engine.build_tree`, which
applies `sglang-jax-877.patch` and both steering patches to `eb061d8`, or to `SGL_COMMIT` when
it's set. It applies both Nemotron 3 patches on top and compiles all 12 files
`git apply --numstat` lists for them.

`transformers` owns the reference. Nothing in the test restates the implementation, and every
comparison reads `NemotronHModel.__call__` rather than a walk over its blocks: the per-layer
streams come out of the capture hook, and the final hidden state comes out of the same return
tuple, so the last residual add and the final norm are under test too. A comparison reads a
non-finite value on either side as infinity, and every gate passes a number only when it compares
true, so a NaN fails the check that meets it.

| | Max abs error | Correlation |
|---|---|---|
| Layer 0 input, embedding | 0.0 | 1.000000000000 |
| Layer 1 input, after Mamba-2 | 1.5e-08 | 1.000000000000 |
| Layer 2 input, after LatentMoE | 1.5e-08 | 1.000000000000 |
| Layer 3 input, after attention | 3.0e-08 | 1.000000000000 |
| Final hidden state | 1.2e-07 | 1.000000000000 |
| Prefill logits | 7.5e-08 | 1.000000000000 |
| Two requests packed into one prefill | 7.5e-08 | 1.000000000000 |
| One decode step on the prefill state | 8.9e-08 | 1.000000000000 |
| Chunked scan against a token-by-token recurrence | 1.3e-07 | 1.000000000000 |
| A four-way tensor axis against a one-way one | 6.0e-08 | 1.000000000000 |

The packed prefill cuts 16 tokens into a 10-token and a 6-token request, so both end on a partial
chunk and the second one starts inside the grid. The decode step runs against the conv window and
the SSM state the prefill wrote back through the pool. Check 7 reads that pool: the second request
of a packed prefill has to leave the cached state it leaves when it runs alone, which is the only
thing the restart at a request's first chunk protects.

Check 5 compares `chunk_terms` and `replay_scalar` against a float64 token-by-token recurrence
written in the test file, so an error in the chunk fold or the intra-chunk decay can't cancel on
both sides.

The capture hook returns the same per-layer residual streams a hand-written walk over the blocks
produces, to the bit, and those streams differ layer to layer by far more than the tolerance, so
a hook reading the wrong layer shows up.

A second config runs the `layers_block_type` spelling, a dense MLP block, a tied head and an MTP
head in the config, which covers the branches the first config's `ME*E` pattern never reaches.
Check 8 writes a checkpoint under the published key names, loads it through
`_create_weight_mappings` and `WeightLoader`, and reads back the embeddings, the fused `in_proj`,
the `[conv_dim, 1, conv_kernel]` depthwise kernel, the router and the stacked experts. Check 9
resolves both published `config.json` files through `AutoConfig`, and a config stock
`transformers` saved under its own block names.

The forwards in checks 2 to 8 and 13 put a dense attention stand-in behind `RadixAttention`. The
engine's attention backend on CPU, `NativeAttention`, writes through `set_kv_buffer_legacy`,
which the `HybridLinearKVPool` a hybrid recurrent model gets doesn't have. Checks 10 to 12 run a
Mamba-2, LatentMoE, Mamba-2, MLP stack instead, which holds no attention block, through the
engine's own classes: `RecurrentStatePool`, `HybridReqToTokenPool` with the extra buffer,
`HybridLinearKVPool`, `MemoryPools`, the backend metadata `get_forward_metadata` builds from a
`ModelWorkerBatch`, and `make_jitted_run_model`, the jit the runner calls, which clones a matched
tree slot before the model reads it.

| Check | What it pins | Max abs error |
|---|---|---|
| 10, track slot | `_build_recurrent_track_entries` picks A's slot; it holds what A's running slot holds, and Z's two track slots, A's other one and slot 0 hold what they held | 0.0 |
| 10, prefix hit | `RecurrentComponent.prepare_for_caching_req` hands the tree that slot, and B resumes from a copy of it: 8 tokens against the same 24 prefilled from scratch | 0.0 |
| 11, non-finite state | one request's cached SSM state holds an inf and its conv window a NaN; the requests packed before and after it keep their clean logits and written state | 0.0 |

Check 12 loads `int8.yaml` through `ModelConfig`, `JAXModelLoader` and `apply_quantization`. The
experts come out int8 with a scale per output channel and no `wi_1`, at most 0.500 of a
quantization step off the checkpoint, the projections come out `QuantizedLinear`, and the conv
kernel stays float32. On CPU the `gmm` kernel's interpret path returns NaN for int8 weights with a
scale when an expert matrix is under 128 on either side, counted per shard. The test config's
experts are 32 by 24, so the loaded model stops at the weights. A latent MoE block built the same
way at 128 runs its int8 experts, and they have to land within 1e-5 of numpy on the dequantized
weights and more than 1e-3 off the float ones, relative. They land at 3.5e-07 and 8.4e-03. A
config that quantizes the linear layers alone runs a packed prefill 7.7e-04 off the float logits,
relative, through the conv kernel and `fc2_latent_proj`. A pre-quantized checkpoint is refused at
build time, and the same config without the static flag builds.

Check 13 writes a checkpoint under the published names and loads it through `ModelConfig` whole,
with `model_layer_nums=3`, and with `{"num_hidden_layers": 3}` in the override JSON. Both cuts
have to hold `mamba`, `moe`, `attention` and the cache plan they imply, and the cut model has to
match the whole model's first three blocks and its stream entering block 3 after the final norm.
A count above the block list has to raise. Check 14 publishes server args with each
expert-placement flag through `set_global_server_args` and builds the model. Each flag has to stop
the build with an error that names it, and the defaults have to build. Check 15 runs the runner's
own mixin code on a stand-in that holds only the model config. `linear_recurrent_config` has to
resolve to the model's config, `attn_backend_wrapper` has to build a `Mamba2AttnBackend` on its
layer plan, and `_per_req_state_bytes_from_config` has to count what one slot of a real
`RecurrentStatePool` holds, and 43,171,840 bytes for Super at `--tp-size 4`.

Twenty-three negative controls follow. Nine perturb one weight, fourteen rewrite the patched
source, and each has to move the check it targets past 1e-05. A control that returns NaN counts as
missed.

| Control | Check | Moves it by |
|---|---|---|
| Mamba-2 `D` skip zeroed | packed prefill | 1.6e-02 |
| One `dt_bias` entry nudged | packed prefill | 3.2e-04 |
| One `A_log` entry nudged | packed prefill | 3.3e-04 |
| One conv tap nudged | packed prefill | 2.6e-04 |
| Gated norm weight scaled | packed prefill | 1.6e-03 |
| One router correction bias nudged | packed prefill | 9.0e-03 |
| One expert's `up_proj` nudged | packed prefill | 1.1e-01 |
| Attention values scaled | packed prefill | 1.9e-02 |
| Final norm scaled | packed prefill | 5.8e-02 |
| Request boundary reset dropped | cross-request isolation | 2.7e-04 |
| Request boundary as a zero decay | non-finite state | inf |
| Padding slots read the next request's `B` | non-finite state | inf |
| Track slot write dropped | track slot | 3.1e+00 |
| Decode state carry weakened | decode | 3.4e-05 |
| Experts built with no quantization config | quantization | inf |
| The int8 down projection read without its scale | quantization | inf |
| Attention output turns NaN | packed prefill | inf |
| Mamba-2 decode step turns NaN | decode | inf |
| Split-group norm turns NaN | tp=4 against tp=1 | inf |
| A written layer count is ignored | cut stack | inf |
| Expert-placement flags let through | expert placement | inf |
| A square SSM slot in the runner's byte count | runner | inf |
| The runner misses the `nemotron_h` config | runner | inf |

The `D` control is the one the scan needs. `D` is the only term that reaches the output without
passing through the state, so zeroing it separates the skip from the recurrence. Check 1 carries
its own control: each patch gets one context line rewritten and has to be refused against a tree
it hasn't been applied to. Check 5 carries one too, a decay 10% off on the chunked side only.

The zero-decay control writes the restart as a multiply by `exp(-inf)`. Check 7 can't tell that
from the select, since on a finite carry the two give the same state; check 11 can. A check that
returns infinity found a non-finite value where the clean run has none, or a property that didn't
hold.

## `nemotron3-capture-hook.patch`

One file. The four parts [`../capture-hooks/README.md`](../capture-hooks/README.md) describes, on
`models/nemotron_h.py`. The stack carries `hidden_states` and `residual` as a pair and `residual`
is None on layer 0, so the append takes the None-residual form. Depends on
[`../sglang-jax-877.patch`](../sglang-jax-877.patch) for the flag and the output reshape.

## `kimi-k3-model.patch`

Eleven files. `models/kimi_k3.py` and `utils/quantization/mxfp4.py` are new. The rest add the
`situ` activation, the K3 config fields, an `externally_loaded` predicate and a `narrow` field for
`WeightLoader`, three engine fixes that let the model serve through `NativeAttention`, and the
output sharding of the EPLB dispatch gathers.

93 layers, 69 Kimi Delta Attention and 24 gated MLA with NoPE, hidden 7,168, 896 routed experts
top-16 plus 2 shared. [`../../models/kimi-k3.md`](../../models/kimi-k3.md) covers the
architecture, the slice arithmetic and what the two-pool split costs.

`models/kimi_linear.py` already serves Kimi-Linear-48B-A3B and shares the KDA recurrence and the
two-pool split. Four things separate K3 from it.

1. `hidden_act` is `situ`, in every MLP. It goes in `layers/activation.py` because the dense MLPs
   and the MoE experts both run it, and `EPMoE` takes it as `activation="situ"`. The dense and
   shared-expert MLPs subclass `kimi_linear.KimiMLP`, which builds the same three projections.
2. The routed experts run in a 3,584-wide latent. The router still scores the full 7,168 stream
   and the shared experts still run at full width. Both latent projections are row parallel: at
   49.0 MiB each across 92 MoE layers, replicating them would cost 8.80 GiB a chip.
3. `attn_res_block_size` is 12, so the residual stream is a learned softmax mixture over the
   running prefix sum and one snapshot per block rather than a single additive accumulator.
4. `use_full_rank_gate` puts one 7,168 x 12,288 `g_proj` in each KDA layer where Kimi-Linear-48B
   factors the output gate through a rank-128 pair.

The KDA layers keep Kimi-Linear's one log decay per head. The checkpoint ships each `A_log` as a
flat float32 `[128]`, which is `head_dim`, for 96 heads. In all 69 KDA layers the first 96 values
hold the decays and the last 32 are 0.0. The published `modeling_kimi_linear.py` declares `A_log`
as `torch.empty(self.num_heads)`, fla's KDA kernels read it at `A_log + i_h`, and vLLM and SGLang
keep its first `num_heads` entries. `WeightMapping` gains `narrow`, which keeps the first entries
along one axis before any reshape. The `A_log` mapping keeps 96 and holds them at `[1, 1, 96, 1]`,
the layout `kimi_linear.py` holds, so the KDA backend and kernels run unchanged.

The MLA layers add a sigmoid output gate, which fits the `_pre_o_proj` hook
`DeepseekV3Attention` already exposes.

### The config is nested

`config.json` keeps every language-model field under `text_config` and leaves `architectures`,
`model_type` and the token ids at the top. `AutoConfig` hands the outer object to the model class,
so `configs/kimi_linear.py` gains `KimiK3Config`. It subclasses `KimiLinearConfig` and copies the
text fields onto itself, which makes `hidden_size`, `linear_attn_config` and `is_kda_layer` read
the language model. Top-level keys win the merge, so `architectures` stays
`KimiK3ForConditionalGeneration` rather than the `KimiLinearForCausalLM` the text tower carries.
`hf_transformers_utils.py` registers it, which is what lets `AutoConfig` accept `model_type`
`kimi_k3`.

`KimiK3Config` is its own `text_config`. The model reads it, `ModelConfig` reads it as
`hf_text_config`, and the state pools read it through `get_kimi_linear_config`, so one set of
fields answers all three. `--json-model-override-args '{"num_hidden_layers": 4}'` then reaches
every reader, and so does the same override nested under `text_config`. A separate copy of the
text fields would build a four-layer model beside pools sized for 93, and the first forward
would fail.

`KimiK3ForCausalLM` keeps the config object the loader hands it. `ModelRunner.load_model` writes
`ep_size`, `moe_dp_size`, `ep_num_redundant_experts`, `moe_backend`, `use_absorbed_mla` and
`enable_sequence_parallel` onto `model_config.hf_config`, and `DefaultModelLoader` passes that
same object to the class.

`EntryClass` registers two names. `KimiK3ForCausalLM` is the decoder.
`KimiK3ForConditionalGeneration` matches the architecture the published `config.json` advertises
and runs the same code, because the language tower is the whole of what this file serves. Neither
joins `InModelMultimodalContract`, which is what marks a model multimodal, so the vision tower and
the projector load nowhere and a text request runs untouched.

### Mixed precision

`quantization_config.ignore` exempts attention, the shared experts, the dense layer and `lm_head`,
so those load BF16. Only the routed experts ship MXFP4: `weight_packed` uint8 with two E2M1 codes
per byte, `weight_scale` uint8 with one E8M0 exponent per 32 elements.
`utils/quantization/mxfp4.py` dequantizes and `KimiK3ForCausalLM._load_routed_experts` drives it
through `jax.make_array_from_callback`, so a host only holds the shard its own device owns.

The callback runs once per addressable device, so `read_mxfp4_block` cuts the rectangle before it
decodes: whole checkpoint rows on the output axis, and the input axis whenever the cut lands on a
32-element scale boundary. The expert index is logical, read through the expert-location map, the
way `create_moe_weights_mapping` does it for experts that load through `WeightLoader`.

With `--ep-dispatch-algorithm` set, `TopK` sends the router's logical ids through the same map. On
the engine's Explicit mesh the ids arrive sharded over the batch and the map unsharded, so the
stock gathers in `eplb/expert_location.py` raise at the startup precompile, for the static map and
the dynamic one alike. The patch gives each gather the ids' sharding, the same hunk
`gpt-oss-model.patch` carries.

`EPMoE` builds its stacks under `use_abstract_mesh`, so after the loader's `nnx.eval_shape` each
one names an `AbstractMesh`, which has no devices. `expert_sharding` moves the spec onto
`EPMoE.moe_mesh`, the concrete mesh with the same axes, before `make_array_from_callback` runs.
The stack comes out in the serving dtype; every MXFP4 value is exact in BF16. A dummy load reads
no file: `load_weights` fills the stacks with zeros on that mesh before the loader's dummy pass,
which would put each whole stack on every chip.

`WeightLoader` gains an `externally_loaded` predicate. A model that reads a tensor format the
loader has no mapping for names those keys, coverage validation counts them as handled, and a key
nobody reads still raises. Kimi K3 names the MXFP4 experts, the 168 `vision_tower.` and
`mm_projector.` tensors, and every layer at or past `--model-layer-nums`, which `WeightLoader`
skips only under a bare `model.layers.` prefix. `load_weights` turns the validation on.

The router keeps `e_score_correction_bias` in F32, the dtype the checkpoint ships, because top-16
picks on score plus bias. `lm_head` loads through `ParallelLMHead.weight_mapping`, so it takes the
data-and-tensor split the head declares. `EPMoE` takes a launch's `QuantizationConfig` and never
the checkpoint's own compressed-tensors dict, which it can't read. `ModelConfig` marks any
`--quantization-config-path` static on this checkpoint, and nothing here loads a statically
quantized checkpoint, so `KimiK3ForCausalLM` refuses one before the loader rewires its linears.

`KimiK3Model` starts the block stash as a zero-width slice of the embedding, so it carries the
embedding's `P("data", None)`. Under the engine's Explicit mesh, layer 0's concatenate refuses a
replicated operand beside a data-sharded one.

### The engine fixes

`NativeAttention` serves the model with `--attention-backend native`, the only backend that serves
off TPU. Three things kept it from serving K3.

| File | Change |
|---|---|
| `mem_cache/memory_pool.py` | `HybridLinearKVPool` forwards `set_kv_buffer_legacy`, which `NativeAttention` writes through off TPU, and refuses by name an inner pool that has none |
| `models/deepseek_v3.py` | `_forward_mha` reads V's width off the attention output; `NativeAttention` returns V at the KV pool's 128-aligned width, 256 for K3's 192-wide head |
| `layers/attention/native_backend.py` | under `--dp-size` above 1 each data rank attends on its own, on TPU and off it, and off TPU each rank's write indices move onto its own slice of the pool |

Under data parallelism each rank's allocator hands out slots from its own slice of the pool, so
two ranks' first tokens both land in slot 1. The legacy write off TPU took those slots as global
rows, so rank 1 overwrote rank 0. On TPU the pool's sharded kernel writes each rank's rows into
that rank's shard. On both, the read took the rank-local slots as global rows and masked the batch
as one packing, so rank 1 attended over rank 0's KV. Each rank now attends over its own slice.
Check 10 runs the TPU branch on CPU, with the TPU KV-write kernel in Pallas interpret mode.
Nothing here has run `NativeAttention` on a TPU at `--dp-size` above 1.

The default `--attention-backend fa` doesn't serve off TPU. A CPU run swaps `NativeAttention` in
for `fa`, but `server_args.attention_backend` stays `fa`, and `ModelRunner` reads it to pick the
absorbed MLA path and the latent `MLATokenToKVPool`. Only the MLA Pallas kernel writes that pool.
`HybridLinearKVPool.set_kv_buffer_legacy` refuses it at the startup precompile with a message
that names `--attention-backend native`.

### What `test_kimi_k3_model.py` covers

Eleven checks. No full HuggingFace forward runs, because `modeling_kimi_linear.py` needs Triton for
the KDA recurrence. Two pieces of that file don't: `SituAndMul` and `_apply_attn_res` compile and
run on their own, and checks 3 and 4 use them beside a NumPy float64 reference written from the
published equations. Check 10 runs the whole model against a float64 forward written from that
file, with the fla kernels it calls written out. Every architecture claim is measured against the
published `config.json`, safetensors headers and tensor bytes, all from one pinned Hub revision and
each held to a pinned SHA-256. The test pulls the patched code out of the patched files by AST, or
imports the patched checkout, so what runs is the patch. A mutation operator that can't find its
target fails the run, the same as a mutant that slips through, and a value that isn't finite
fails every gate.

1. Both patches apply to `eb061d8` with `sglang-jax-877.patch` and both steering patches on it,
   and every file they touch compiles. Control: the same patch with one context line rewritten is
   refused by the same tree that just accepted the real one.
2. `dequantize_mxfp4` against an E2M1 decoder written from the bit fields, and `read_mxfp4_block`
   on five rectangles against decoding the whole tensor and slicing after. Controls: one nibble
   flipped, four mutants of the decoder, three of the block reader.
3. `situ_and_mul`, against the published formula in float64 and against `SituAndMul` out of the
   modeling file the checkpoint repo ships. Controls: `linear_beta` dropped, gate and up swapped,
   `beta` set to 1.0, and four mutants of the function body.
4. `attn_res_mix`, against a float64 mixture with unfolded weights and an explicit per-row
   softmax, and against the shipped `_apply_attn_res`. Controls: either score weight dropped, the
   prefix sum left out, one stash doubled, and two mutants of the function body.
5. The patched `KimiK3DecoderLayer.__call__` and `KimiK3Model.__call__`, under `jit` and
   `jax.set_mesh` on a mesh with Explicit axes, the way `ModelRunner` runs them, with the input
   sharded `P("data", None)` the way `Embed` hands it over, against a float64 replay of the whole
   block-residual algorithm. The float32 run repeats on a one-way data axis. Reports max absolute
   error and correlation per layer. Ten mutants, each run against a full capture gate and a
   two-layer one, so a mutant confined to the hook still moves the answer. Then the entry class:
   the capture flag gates the list and the list reaches `LogitsProcessor` by keyword, with a
   mutant for each.
6. The published `A_log` of layers 0 and 1, read by range request: a float32 `[128]` whose first
   96 values are nonzero and whose last 32 are 0.0, and the reference `KimiDeltaAttention`
   declaring it `torch.empty(self.num_heads)`. Then the Mega KDA kernel in Pallas interpret mode,
   BF16, on one log decay per head and two requests sharing a 64-token tile, against a float64
   recurrence. Controls: the published vector read per channel disagrees with the per-head reading
   on 12,192 of 12,288 decays, and the Mega kernel handed each head its neighbor's decay moves by
   3.9e-01.
7. The layer split and the state each half costs, through the patched `is_kda_layer` reading the
   published `linear_attn_config`. Control: a 0-based reading moves 46 layers.
8. The name `EntryClass` has to register and the config class that reaches the language tower.
   `KimiK3Config` answers 20 published fields, and it's its own `text_config` and what
   `get_kimi_linear_config` returns. `{"num_hidden_layers": 4}`, at the top level and nested under
   `text_config`, has to reach the model, `hf_text_config` and the pools, through the engine's own
   `apply_model_config_overrides`. Controls: the text tower's `KimiLinearForCausalLM` used instead,
   the same scanner reading `kimi_linear.py` and the `_CONFIG_REGISTRY` on the unpatched tree, the
   top level read as the whole config, two mutants of `KimiK3Config`, and a third that keeps
   `text_config` as a separate copy.
9. All 93 layers built at full size under `nnx.eval_shape`, which allocates nothing, on the mesh
   `create_device_mesh` returns. `--ep-size 8` has to reach `EPMoE`, `use_absorbed_mla` has to
   reach MLA, and nine more published settings have to land on the layers that read them, the
   router bias dtype read off the shard 2 header among them. `_pre_o_proj` runs against a float64
   sigmoid gate. Then every weight mapping against the shapes in the headers of shards 1, 2, 4, 94,
   95 and 96, both directions: no mapping without a tensor, no tensor without a mapping or a
   predicate. `lm_head` has to map the way `ParallelLMHead` declares. Then a tiny model built the
   same way reads its MXFP4 experts out of a real safetensors file through
   `_load_routed_experts` at `--ep-size` 1, 2 and 8, and once more through a permuted
   expert-location map. Controls: the gate switched off, the vision tensors dropped from the
   predicate, `A_log` read per channel, `A_log` reshaped to the head count with its padding left
   on, and the expert-location map ignored.
10. The served model. A four-layer checkpoint in the published layout, MXFP4 experts and four
    vision tensors included, goes through `ModelWorker` on the scheduler's mesh: `JAXModelLoader`,
    `ModelRunner`, the startup precompile, then one prefill of an 11-token and a 6-token request
    and two decode steps through the scheduler's `PrefillAdder` and `ScheduleBatch`. It runs at
    `--ep-size 1` with the capture hook on, at `--ep-size 2`, at `--dp-size 2 --ep-size 2`, and
    twice at `--ep-size 2 --ep-num-redundant-experts 2`, once with `--ep-dispatch-algorithm static`
    and once with `dynamic`. Every step's logits have to match a float64 forward of the same
    weights. At `--dp-size 1` so does every layer's input the hook returns. The file's `A_log`
    holds one value per head and zeros after them, and the forward reads it per head. One
    `--ep-size 2` launch serves a file with a fifth layer, which
    `--json-model-override-args '{"num_hidden_layers": 4}'` cuts back off, and the model has to
    come out with four. The two redundant experts put copies of logical experts 0 and 1 in
    physical slots 8 and 9. The static map sends each logical id to its own slot, so that launch
    never reads a copy. The dynamic map picks among each expert's copies, so only that launch
    reads slots 8 and 9. `lm_head` has to load with the sharding it declares, and the default
    `--attention-backend fa` has to be refused at startup by a message that names
    `--attention-backend native`. Then `NativeAttention`'s TPU branch runs at
    `--dp-size 2`, with the pool's TPU KV-write kernel in Pallas interpret mode, and each rank has
    to match a float64 attention over its own requests. Controls: the tiny layout, sized up to the
    published config, reproduces all 10,830 tensors in four shard headers and refuses a flattened
    short convolution; a file with half of layer 1's expert `w2` channels doubled moves the served
    logits and still matches its own float64 forward; a copy of the served result with every
    logit and captured layer set to NaN fails all 30 gates; and a `NativeAttention` that reads the
    batch as one rank puts rank 1 off by 1.0.
11. The loader's other paths. A BF16 load has to keep the router bias F32 and bit-equal to the
    file. Check 10 can't show that, because at float32 every parameter is F32. The same load has
    to hold the file's first `num_heads` `A_log` values at `[1, 1, heads, 1]`, bit for bit, and a
    file whose padding holds 7.0 has to load the same parameter. A dummy load over the checkpoint,
    over `config.json` alone, and in the abstract mode AOT export uses has to read nothing and
    leave the experts zero on the expert mesh. `--model-layer-nums 2` has to load two layers, and
    a stray key in a served layer still has to stop it. `--quantization-config-path int8.yaml` has
    to be refused when the model builds, an online `QuantizationConfig` has to reach `EPMoE`, and
    the checkpoint's own dict mustn't. Control: the file's bias rounded through BF16 moves by
    2.4e-04.

Check 10 serves float32 with a float32 short-convolution state, so the engine's error is
rounding, not BF16. The Mega KDA kernel takes BF16 only, so its prefill runs the chunked KDA
kernel, the one serving falls back to when a 64-token tile holds more than two requests; check 6
covers the Mega kernel. It launches with `--attention-backend native`, the one backend that
serves off TPU.

| Path | float32 | bfloat16 | against the shipped torch |
|---|---|---|---|
| MXFP4 dequantization | 0.0 | | |
| `situ` | 3.1e-07 | 4.4e-03 | 3.1e-07 |
| `attn_res_mix` | 2.6e-07 | 3.9e-03 | 3.4e-07 |
| Mega KDA on one log decay per head | | 2.9e-03 | |
| 9-layer stack | 1.0e-06 | 2.1e-02 | |
| Served model, logits and layer inputs, worst of five launches | 1.2e-05 | | |
| `NativeAttention`'s TPU branch at `--dp-size 2`, worst rank | 2.4e-07 | | |

Relative error against the float64 reference. Correlation is 1.000000 at float32 on every path.
Mutants run at float32 against a gate of 1e-02, and the served model's gate is 1e-04. The
bfloat16 assertions carry a correlation floor of 1 - 1e-04 alongside the tolerance, because
bfloat16 rounding on the nine-layer stack is twice the mutant gate.

A permuted stash isn't among the controls. The mixture is a softmax over an unordered set, so a
reordered stash gives the same answer, and a control built on it would always pass.

The test reads `config.json`, `modeling_kimi_linear.py`, six shard headers and two `A_log` tensors
from Hub revision `f831ab66814297da540d832a5235f8e904f29d06`, and each file has to match its
pinned SHA-256 before the test reads it. They're cached under `KIMI_K3_CACHE`, or
`~/.cache/kimi-k3-published/<revision>` when that isn't set. Check 10 runs the engine's
`ModelRunner`, which imports `pybase64` and `llguidance`. Without them check 10 fails and the run
ends on the `uv pip install` line. `LOG_PAYLOADS=1` prints the `ServerArgs` of every engine launch
and every request check 10 sends. The whole run takes about 8 minutes on CPU, most of it in check
10's seven engine launches.

## `kimi-k3-capture-hook.patch`

One file. The four parts [`../capture-hooks/README.md`](../capture-hooks/README.md) describes, on
`models/kimi_k3.py`. Apply the model patch first. Depends on
[`../sglang-jax-877.patch`](../sglang-jax-877.patch) for the flag and the output reshape.

What lands is the prefix sum entering each layer. K3 carries no separate `residual`, so nothing is
None on layer 0 and the append reads the sum directly, where every other model's hook has to test
for it. The value a layer conditions on is one softmax away from the prefix sum, and the sum
restarts every 12 layers.

Check 10 of the test serves with `--enable-return-hidden-states` at `--dp-size 1` and holds every
layer's input the engine returns, at every prompt position and decode step, to the float64
forward.

## `gpt-oss-model.patch`

Five files. `models/gpt_oss.py` is new and serves both `openai/gpt-oss-120b` and
`openai/gpt-oss-20b`, which share an architecture and differ only in layer and expert count. The
hook is in the same patch, in the four parts
[`../capture-hooks/README.md`](../capture-hooks/README.md) describes. The layer carries a
`(hidden_states, residual)` pair, so the append uses the None-residual form.

The other four files are edits to shared code.

`layers/moe.py` gives `EPMoE` two things. `use_expert_bias` creates a per-expert bias on each of
the three GEMMs, in the `[experts, 1, n]` layout the `gmm` kernel already accepted but no caller
ever filled. `activation="swiglu_oai"` selects the gpt-oss gate, which clamps both halves and adds
one to the linear half:

```
gate = min(gate, limit)
up   = clip(up, -limit, limit)
out  = (up + 1) * gate * sigmoid(alpha * gate)
```

The second GEMM is row-parallel, so each `tensor` shard produces a partial sum that a later `psum`
finishes. Its bias divides by the shard count on the way in, and the `psum` reassembles one whole
bias.

`kernels/gmm/megablox_gmm_backend.py` adds those biases after the activation rescale on `gmm`'s v1
path, the one that runs off TPU. Under a config that quantizes the activations, such as
`int8_w8a8.yaml`, the v1 kernel would add each bias inside and the rescale after it would scale the
bias by the row's activation scale. v2 scales first and adds the bias after, so TPU runs don't
change. No caller passed a bias before this patch.

`eplb/expert_location.py` gives each EPLB dispatch gather the sharding of the expert ids it reads,
through `.at[...].get(out_sharding=...)`. On the engine's Explicit mesh the ids arrive sharded
over the batch and the map arrives unsharded, so the plain gather can't infer where its output
lives and raises at the first forward, for the static map and the dynamic one alike.

`layers/embeddings.py` gains `truncate` on the YaRN correction range. YaRN blends an extrapolated
and an interpolated inverse frequency across a linear ramp whose bounds come out fractional. Most
checkpoints round them to whole dimensions. gpt-oss sets `"truncate": false` and keeps the
fractional bounds, which moves the ramp and changes every inverse frequency inside it. At head_dim
64, base 150000 and a 4,096 pretraining window, truncating moves the bounds from (8.093, 17.398)
to (8, 18). The two settings then disagree by 2.2e-04 per position on the worst frequency, 4.6 full
rotations at the far end of the 131,072-token context.

### Attention sinks

Every query head carries one learned logit that joins the softmax and holds no value. The ragged
paged attention kernel takes it as an `attention_sink` argument and `RadixAttention` forwards
keywords to the backend, so the model passes `self.sinks` straight through. The checkpoint stores
the sinks in BF16. The parameter is float32, because that's what the kernel reads.

### Mixed precision

The two expert GEMMs ship MXFP4 as a pair of tensors. `*_blocks` is
`[experts, out, in / 32, 16]` uint8 holding two FP4 E2M1 codes per byte, low nibble first.
`*_scales` is `[experts, out, in / 32]` uint8 holding one E8M0 exponent per 32 elements, so the
block multiplier is `2 ** (s - 127)`. Everything else is BF16, so the loader runs the ordinary
mapping table first and decodes the experts in a second pass.

`gate_up_proj` interleaves the two halves of the gate along its output axis, even columns gate and
odd columns linear. `EPMoE` keeps them as separate `wi_0` and `wi_1`, so the deinterleave happens
at load. Both expert GEMMs transpose from the checkpoint's `[experts, out, in]` to the
`[experts, in, out]` the kernel wants.

`kimi-k3-model.patch` writes the same decode into `utils/quantization/mxfp4.py`. That patch and
this one both edit `layers/moe.py`, so a checkout holds one of them and neither can import the
other's module. gpt-oss carries its own copy.

### Load cost

`dequantize_mxfp4` decodes 2^18 blocks a pass, 8 passes at a time, straight into the serving
dtype. Each pass stages 32 MiB of float32, so the decode never holds a whole-tensor float32 copy.
Layer 0's `gate_up_proj` from `openai/gpt-oss-120b` is 0.99 GiB packed and 3.96 GiB decoded. On an
8-vCPU host its decode takes 8.9 s at 8 threads and 19.6 s at one, and peak RSS grows by 4.3 times
the packed tensor.

`_put` builds each parameter through `jax.make_array_from_callback`, so the host cuts the shards
and a chip only receives its own. `jnp.asarray` followed by a reshard puts the whole 1.98 GiB
tensor on one chip first, and gathers to do it on a multi-host slice.

Under `--load-format dummy` the model fills its expert parameters with zeros on `EPMoE`'s mesh
before the loader's dummy pass. That pass fills every parameter the mapping table leaves out on
the model mesh, which has no `expert` axis, so a stack it filled would sit whole on every chip,
6.37 GB a layer and 229 GB a chip over gpt-oss-120b's 36 layers.

### EPLB redundant experts

`EPMoE` sizes `wi_0`, `wi_1`, `wo` and the three biases at `num_physical_experts` whenever
expert-location metadata exists. `GptOssSparseMoeBlock` hands the metadata to `TopK`, which
rewrites logical expert ids into physical ones, the same call deepseek_v3 and qwen3_moe make, and
the patched gather in `eplb/expert_location.py` lets that call run on ids sharded over the batch.
`_load_mxfp4_experts` reads `physical_to_logical_map` and fills physical slot `p` from logical
expert `map[p]`, so `--ep-num-redundant-experts` gets weights in every slot it allocates. A slot
count that disagrees with the metadata raises.

### Attention type

`layer_types` decides which layers slide. `transformers.GptOssConfig` fills it in when the
checkpoint omits it, and the patch does the same, alternating from a sliding layer 0. A list that
is present but shorter than the stack describes a different model, so it raises rather than
turning the tail into full attention.

### Quantization configs

The published `config.json` carries its own `quantization_config`, the `mxfp4` dict that
describes the packed experts. `ModelConfig` doesn't convert it, so it resolves no quantization and
leaves the dict on the config. `EPMoE` calls methods on the config it gets, so the model hands it
the `QuantizationConfig` a user's `--quantization-config-path` resolves to and nothing else.
Without one, the experts stay in the serving dtype `_load_mxfp4_experts` decodes them to. With
one, `apply_quantization` quantizes the decoded experts after the load, the three expert biases
stay as they are, and the linear rules swap the attention projections for `QuantizedLinear`.

### What `test_gpt_oss_model.py` covers

`transformers` owns the reference on both sides of the checkpoint. The tiny checkpoint is a real
safetensors file with packed FP4 experts, beside a `config.json` that carries the published
`mxfp4` dict, and the model loads it through `ModelConfig` and `JAXModelLoader`, the way
`ModelRunner.load_model` does. So the config parsing, the shipped weight loader and the mapping
table all run on the way in. The HuggingFace model gets those same experts decoded by
`transformers.integrations.mxfp4.convert_moe_packed_tensors`, the routine `transformers` runs on a
real gpt-oss checkpoint, so the reference never sees the patch's decoder.

The decoder also has a second reference written from the OCP microscaling spec, and the two
references answer different questions. The spec fixes the 16 code points and the E8M0 bias of 127.
It says nothing about which element of a 32-wide block a byte's low nibble holds, and nothing about
the axis order, both of which are serialization choices `transformers` made. So the patch is
checked against the spec reference on every one of the 256 byte values, on E8M0 exponents 100
through 154 and on a random packed tensor, and against `convert_moe_packed_tensors` on a random
tensor spanning exponents 100 through 149 including the `[experts, out, in]` to
`[experts, in, out]` transpose. Two controls: reversing the nibble order has to disagree with the
spec reference, and swapping the two halves of every byte has to disagree with `transformers`.

Both sides have to agree on the rope before the forward runs. The inverse frequencies match to
0.0, and setting `truncate` back to true has to move them.

One forward at float32, four layers, 64 tokens, 8 experts top-4, alternating sliding window and
full attention. The logits row comes out of the entry class, so the shipped `LogitsProcessor` and
the `lm_head` selection are on the path.

| | Max abs error | Relative | Correlation |
|---|---|---|---|
| Layer 0 input | 0.0 | 0.0 | 1.000000000 |
| Layer 1 input | 1.1e-04 | 5.5e-07 | 1.000000000 |
| Layer 2 input | 2.1e-04 | 1.1e-06 | 1.000000000 |
| Layer 3 input | 9.5e-04 | 4.4e-06 | 1.000000000 |
| Final norm | 5.7e-05 | 1.0e-05 | 1.000000000 |
| Logits | 1.2e-04 | 1.3e-05 | 1.000000000 |

Six negative controls follow. Each breaks one thing and has to fail the 2e-4 gate the clean run
passes, so one number separates the two. The clean worst is 1.3e-05, 15 times under the gate, and
the weakest control is 6.5e-03, 32 times over it.

| Control | Moves the relative error to |
|---|---|
| One expert gate weight nudged | 1.3e-02 |
| Gate and linear halves swapped | 9.5e-01 |
| Down-projection bias dropped | 7.0e-03 |
| Attention sinks zeroed | 6.5e-03 |
| Expert reduction axis reversed | 1.3e+00 |
| Router bias dropped | 4.3e-01 |

The LM head check runs both branches of the entry class. Untied, the processor reads `lm_head` and
matches HuggingFace. Tied, it reads the embedding table instead. Control: the two branches have to
produce different logits.

The attention kernel is a Pallas TPU kernel, so the test puts a dense stand-in behind
`RadixAttention` that implements the contract upstream documents in
`ragged_paged_attention_v3.ref_ragged_paged_attention`: repeat the KV heads, scale, mask acausally
and outside the sliding window, prepend the per-head sink logit, softmax, drop the sink column.
Everything the patch owns on either side of that call stays under test. Upstream covers the kernel
itself, sinks and sliding windows included, in `test_flashattention_gqa.py`. `NativeAttention`,
the backend the engine falls back to on CPU, can't take the stand-in's place here: it returns the
KV pool's head width, which the runner pads to 128, so a 16-wide head breaks at `o_proj`. The
batch around the stand-in is the engine's own `ForwardBatch`, and the pools its `MemoryPools`.

The hook check runs the gate and both production setters. The gate captures nothing when empty,
two layers when given two, and every layer when marked. The layer 0 capture is the embedding
output. The layer 1 capture differs from it, so the None-residual form isn't reading a stale
residual. The flag on the entry class gates what the logits processor receives.
`set_eagle3_layers_to_capture` and `set_dflash_layers_to_capture` both shift the caller's ids by
one, matching llama and qwen3 upstream, and the check compares the tensors they capture against
the tensors a full-gate run produces at the shifted layers. Control: the inputs of layers 2 and 3
differ, so the shift is observable. The no-argument EAGLE3 default has to reproduce upstream's
`[2, n // 2, n - 3]`, and DFLASH has to refuse a missing layer list.

Check 8 turns on EPLB. Expert-location metadata with two redundant experts makes `EPMoE` allocate
10 physical slots for 8 logical experts, and the static dispatch map sends logical experts 0 and 1
to slots 8 and 9. All six expert parameters have to carry 10 slots, and slots 8 and 9 have to equal
logical experts 0 and 1. Control: slot 8 has to differ from logical expert 2, so the copy follows
the map rather than landing anywhere. On ids sharded over the batch, the way `TopK` hands them
over, the static map has to place logical 0 and 1 on slots 8 and 9, and the dynamic map has to
send every id to a copy of its own expert and logical 0 and 1 to both of theirs. Then slots 0 and 1
take a poison in every layer, and a forward that carries the static map in its `ForwardBatch` has
to match HuggingFace. Control: the same forward without the map routes logically, reads the
poisoned slots, and has to miss.

Check 9 reads quantization configs. The `mxfp4` dict has to stay on `hf_config` with no
resolved quantization, which is the config every forward above loaded under. Control: handing
`EPMoE` whatever sits on the config, the way the upstream MoE models do, has to break that load.
Under the built-in `int8.yaml` the three expert GEMMs have to come out int8 with a scale per
output channel, at most half a quantization step off the decoded MXFP4 weights, and the three
biases have to stay what the checkpoint holds. On CPU the `gmm` kernel's interpret path returns
NaN for int8 weights with a scale when an expert matrix is under 128 on either side, counted per
shard. `EPMoE` splits the experts 8 ways on the test's mesh, so the tiny model's 64-wide experts
stop at the weights. A block built the same way 1,024 wide, 128 a shard, runs its int8 experts,
and they have to land within 1e-5 of numpy on the dequantized weights and more than 1e-3 off the
float ones. They land at 3.4e-07 and 1.3e-02. Under `int8_w8a8.yaml` `gmm` quantizes the
activations too. The bias one `gmm` call adds has to land within 1e-5 of whole, for four groups
and for groups 2 and 3 through `group_offset`, and the down-projection bias of the same 1,024-wide
block has to reach its output within 1e-3 of the routed bias sum. They land at 9.5e-07 and
1.0e-05. A bias scaled by each row's activation scale misses by 2.0 and 0.84.

Check 10 loads the model with `--load-format dummy` at `--tp-size 8`. Right after the loader's
dummy pass every expert array has to sit in `EPMoE`'s layout, and the most device 0 holds of what
the load allocates has to stay within 1.25 times what the loaded model holds there. Device 0 peaks
at 552,592 bytes, what the loaded model holds. Control: with the model's own expert fill turned
off, the pass leaves 20 of the 24 expert arrays whole on every device, and device 0 peaks at
3,319,440 bytes.

> `test_gpt_oss_model.py` takes about 14 minutes on CPU. It loads the tiny model ten times through
> `JAXModelLoader` and twice through the dummy loader, and compares every layer against
> HuggingFace. It isn't hung.
