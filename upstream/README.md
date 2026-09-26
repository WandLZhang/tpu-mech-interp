# Patches

Against [`sgl-project/sglang-jax`](https://github.com/sgl-project/sglang-jax).

Upstream moves, so a hunk that applied last month can stop applying with no change on this side.
Check before trusting any of them:

```bash
bash scripts/verify_patches.sh
```

It applies all 14 patches to a clean checkout at `eb061d8`, in 12 checks, in the order each is
meant to be used, and names whatever fails. `SGL_COMMIT=main` runs the same check against upstream
main. The clone takes full history, because a `--depth 1` clone holds only the tip of `main` and
has no `eb061d8` to check out. Set `SGLANG_JAX_REPO` to a local checkout that holds `eb061d8` to
skip the clone.

When one does stop applying, three-way it, fix the conflict, and regenerate the patch from the
resolved tree. Two things drifted the last time. Upstream grew `enable_dp_lm_head=` on the
`LogitsProcessor` call, which moved the anchor every capture hook sits on, and it dropped
`GlmMoeDsaConfig`, which the Kimi K3 and Nemotron 3 config-registry hunks used to add alongside
their own entry.

## `sglang-jax-877.patch`

Per-layer hidden state extraction through the Engine API. Adds
`--enable-return-hidden-states` and `--return-hidden-states-layers`. With the first set, the model
runner marks every layer for capture, or the slots the second names, before `nnx.split()` bakes
`layers_to_capture` into `model_def`. The scheduler reshapes the result to
`[seq_len, slots, hidden_dim]`.

Slot `i` holds the residual stream entering block `i`, so slot 0 is the embedding output and slot
`i` is block `i - 1`'s output. That's the HuggingFace `output_hidden_states[i]` convention.

Carries [PR #877](https://github.com/sgl-project/sglang-jax/pull/877), plus these changes:

- `llama.py` captured `hidden_states + residual`, and `residual` is `None` on layer 0. EAGLE3
  captures layers `[2, n//2, n-3]` and never hits it; this flag captures every layer. Now uses the
  form `qwen3.py` already had.
- The prefill host transfer used `.astype(float)`, which is float64 in numpy and doubles the
  transfer. Now `.astype(np.float32)`.
- The scheduler put `req.hidden_states` into `output_hidden_states_for_mm` alone. The multimodal
  prompt-embed path reads that field and the detokenizer drops it, while `tokenizer_manager`
  builds `meta_info["hidden_states"]` out of `output_hidden_states`. The Engine API got no key.
  Both lists now carry it.
- The prefill slice took `len(req.origin_input_ids)` rows out of a buffer holding the forward
  pass's extend tokens. A cached prefix or a chunked prefill makes those differ, and the slice
  then read padding or the next request's rows. It now takes
  `extend_input_len_per_req[req_idx]`.
- The capture and the cursor ran only for requests whose prefill finished in the pass. A chunked
  prompt kept its last chunk alone, and every later request in the batch read from the wrong
  offset. Both now run once per request the batch visited.
- `scheduler.py` collected `extend_input_len_per_req` under `return_logprob` alone. It now
  collects it for `return_hidden_states` too, because `req.extend_input_len` already holds the
  next round's value by the time the result arrives.
- Drops an unrelated device-mesh log line and a superseded workflow edit.
- With data parallelism each rank's rows sit in their own padded block, and the prefill and
  decode paths read the hidden states as one packed run, so a request on rank 1 or later would
  get other requests' rows. Until the offsets follow the ranks, `--enable-return-hidden-states`
  refuses to start with `--dp-size` above 1, and the scheduler aborts any request that sets
  `return_hidden_states` when `dp_size` is above 1.
- A request for hidden states never reuses a cached radix prefix. `adjust_max_prefix_ids` caps
  its match at 0, the way `return_logprob` caps it at `logprob_start_len`. Before, a cached
  prefix skipped its forward pass, and the reply covered the uncached tail while
  `prompt_tokens` counted the whole prompt.
- On a model without the `layers_to_capture` hook, the flag refuses to start. `LogitsProcessor`
  would store the final normed hidden state, and the scheduler would cut it into per-layer
  slices that aren't layers, or stop the server when the layer count doesn't divide the hidden
  size.
- `output_hidden_states` holds one entry per request the scheduler sends, `[]` for one that
  didn't ask, because `tokenizer_manager` reads it by position in `rids`. A list of the askers
  alone handed one request another's rows, then ran off the end and stopped the server.
  `tokenizer_manager` sets `meta_info["hidden_states"]` only for a request that asked.
- An extend batch with a request for hidden states always takes the padded input-logprob path.
  The fallback raises on jax 0.11.1, which stopped the scheduler for a logprob request in the
  same batch. With `return_logprob` on, the sampler rebuilds the logits output without the
  hidden states, and `tp_worker` now carries them across instead of dropping those rows.
- The scheduler records where each request's rows start, and the output processor keeps only
  the positions a request doesn't hold yet. A retracted request prefills its prompt and its
  output again, and under the overlap scheduler a retracted request's in-flight prefill lands
  too, so both used to append duplicate rows. The start comes from `prefix_lens`, the positions
  the forward pass runs at. With `--enable-mixed-chunk` under the overlap scheduler, a running
  request's decode step rides in another request's prefill chunk before its input token reaches
  `output_ids`, and a start read from `fill_ids` dropped that row and filed every later row one
  position early. A pass that covers a position the request holds, outside a retraction, logs a
  warning.
- The flag refuses to start with `--speculative-algorithm`, because capture overwrites the
  layers EAGLE3 and DFLASH capture for their draft, and with `--pd-disaggregation`, whose
  prefill path asserts on a request for hidden states. It refuses `--disaggregation-mode`
  `prefill` and `decode` too. There the decode server takes the prompt's KV from the prefill
  server and runs only the last prompt position, so its reply held one prompt row while
  `prompt_tokens` counted every one. The scheduler also aborts such a request under either PD
  mode. `tokenizer_manager` refuses `return_hidden_states` on a server started without the flag,
  so the caller gets the error and the scheduler never sees the request.
- `filter_batch` derives `return_hidden_states` again from the requests it keeps, as it does
  `return_logprob`. The running batch used to stay in capture mode after the last request that
  asked left, and every later decode step concatenated every layer and copied it to the host.
- A request that doesn't stream gets its rows once, in the output that finishes it. The
  scheduler forces an output every 50 tokens, and each one sent the whole list again, only for
  the tokenizer manager to drop it. A streamed chunk still carries every row so far, the form
  upstream sglang streams.
- The non-stream `/generate` branch returns `ORJSONResponse`, the stream branch's encoder.
  FastAPI's default encoder refused the numpy arrays and answered with a 500.
- Upstream's `test_pd_prefill_overlap.py` drives `run_batch` with a stub batch that sets the
  fields `run_batch` reads. `run_batch` reads `return_hidden_states` now, so the stub sets it
  beside `return_logprob`.
- `test_hidden_states_alignment.py` and `_tp.py` share one base class in `test_utils.py`, and
  the three TPU tests take their Engine settings from one helper there. The base class reads
  every prefill chunk rather than the first.
- `--return-hidden-states-layers 15 20` names the capture slots to return, and each reply is then
  `[seq_len, 2, hidden_dim]`. The device concatenates and copies those slots alone, where it used
  to copy every layer whatever the caller kept: 36 slots on Qwen3-8B for an SAE that trains on
  one. Unset, it returns every slot, and slot `k` still holds the stream entering block `k`. The
  list takes separate integers, the way `--precompile-token-paddings` does. The hook appends in
  layer order, so `check_server_args` refuses a list out of ascending order. It also refuses an
  empty list, a slot listed twice, a negative slot, and the flag without
  `--enable-return-hidden-states`. The model runner refuses a slot at or past the layer count.

19 files, +1,859/−25.

```bash
REPO=$PWD
git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
git checkout eb061d8            # every patch in upstream/ applies here
git -c user.name=you -c user.email=you@example.com am < "$REPO/upstream/sglang-jax-877.patch"
```

`sglang-jax-877.patch` is a `git format-patch` export and carries its original author, so apply
it with `git am`. The rest carry the diff and the rationale without commit metadata, so apply
those with `git apply` and write your own message.

Tests: `test_return_hidden_states.py` checks shape and content.
`test_hidden_states_alignment.py` and `_tp.py` compare every layer against a HuggingFace CPU
float32 reference at `tp=1` and `tp=4`. All three need a chip.
`test_return_hidden_states_cpu.py` runs the real Engine on CPU with a tiny random Qwen3: a cached
prefix, a mixed batch, logprobs beside hidden states, HTTP `/generate`, what each output of a
streamed and an unstreamed request carries, both retraction routes, decode steps inside another
request's prefill chunks under the overlap scheduler, a request to a server without the flag, a
hookless model, and the settings that refuse to start. For the layer filter it checks the reply
shape, the order of slots, each refusal, the command-line syntax, and a kept slot's rows against
the same slot from a server that returns every slot. Where it compares values, it compares
against the same prompt run alone. On CPU the overlap scheduler sometimes computes a step of a
batched request wrong, with or without these patches, so under it the comparison stops where the
two runs' greedy tokens part. No test loads a Llama checkpoint, so the `llama.py` change goes
unrun. The scheduler's aborts sit behind `check_server_args`, which refuses the same settings
before the Engine starts, so no Engine test reaches them either.
`test_capture_mode_propagation.py` reads `schedule_batch.py` for the flags `init_new` and
`filter_batch` have to derive from the requests.
[`capture-hooks/test_capture_hooks.py`](capture-hooks/test_capture_hooks.py) runs
`test_return_hidden_states_cpu.py` as its check 6, on the tree `scripts/bootstrap_tpu_vm.sh`
builds, 877 and both steering patches. Its check 5 reads the four handoffs between
`req.hidden_states` and `meta_info["hidden_states"]` out of that tree, because filling the list
at one end says nothing about the other end.

The MAE thresholds it ships are 5e-2 for the first quarter of layers and 5e-1 for the rest. Those
suit speculative decoding. For interpretability, report Pearson alongside MAE so a systematic
shift shows up.

## Verified on hardware

`Qwen/Qwen3-0.6B` on a `v5litepod-4` in us-south1-a, `tp_size=4`, BF16, against a HuggingFace CPU
float32 reference on the same prompt. 28 layers captured, worst Pearson 0.99998586, layer 0
bit-identical. A control that shifts each captured layer against the wrong reference layer lands
at 6.5357e+03 and is detected. Gemma 4 26B-A4B's check on a `v5litepod-8`, one prompt alone and a
batch that splits a prompt across prefill passes, is in
[the capture guide](../docs/activation-capture.md#correctness).

Absolute error reaches the tens because a residual stream carries large magnitudes and the forward
is BF16.

## The capture-mode gap

`LogitsProcessor` only stores hidden states when `logits_metadata.capture_hidden_mode.need_capture()`
is true, and `ScheduleBatch` derives that from `self.return_hidden_states`. Nothing set that flag
from the requests in the batch, so it kept its `False` default, the mode stayed `NULL`, and
`meta_info["hidden_states"]` came back empty with every line of plumbing present.

`init_new` aggregates `return_logprob`, `return_output_logprob_only` and `return_routed_experts`
from `all_reqs`. `return_hidden_states` was the one missing from that list. The patch adds it.

Found on a v5litepod-4 in us-south1-a. Source inspection can't see it, because every file in the
chain reads correctly on its own. `test_capture_mode_propagation.py` checks the invariant instead:
every `return_*` flag that exists on both `Req` and `ScheduleBatch` has to be aggregated in
`init_new`, resolving a local binding before judging. It fails on the unfixed tree. Its negative
controls drop each flag from the parsed `init_new` call, or pin it to `False`, and run the same
judgment, so a judgment that accepts anything fails them.

The batch has to clear the flag too. `merge_batch` ORs it in from each batch it merges, and
`filter_batch` recomputed `return_logprob` from the requests it kept but not
`return_hidden_states`, so a running batch stayed in capture mode after the last request that
asked left. The test also requires `filter_batch` to derive every flag `merge_batch` ORs in, with
a control that drops each assignment from the parsed `filter_batch`.

## `glm5-capture-hook.patch`

Adds the `layers_to_capture` hook to `Glm5Model`, matching `gemma4.py`, `llama.py` and `qwen3.py`.
`GlmMoeDsaForCausalLM` inherits it. Depends on `sglang-jax-877.patch` for the flag and the
reshape.

## `steering-hook.patch`

Causal activation steering in the `gemma4` forward pass. Adds `--enable-steering`,
`--steering-bank` and `--steering-layer`. The server loads a bank of directions, a request names
one of its features, and the model adds `alpha * v` to the residual stream leaving the chosen
layer.

```bash
python -m sgl_jax.launch_server --model-path google/gemma-4-31B-it \
    --enable-steering --steering-bank steer_l20.npz --steering-layer 19
```

The hook fires after `layer(...)` returns, so it writes the stream leaving the layer you name.
Capture appends before `layer(...)` runs, so its slot `k` is the stream entering block `k`. A
bank trained on capture slot 20 steers at `--steering-layer 19`.

The bank carries that number, so nobody has to remember it. `sae/train.py` records the capture
slot in the checkpoint as `capture_layer`. With `--manifest`, `--layer` names the slot. With
`--activations` on a 3-D shard, `--layer` is a position on the shard's layer axis, so
`--capture-layer` names the slot. `from_sae.py` writes the slot and `steering_layer` into the
bank's meta and prints the flag, and `SteeringBank.load` refuses a bank whose recorded site isn't
the `--steering-layer` the server got. Width can't catch a site mismatch, because every layer of a
model is the same width. A bank with no recorded site loads with a warning.

```python
engine.generate(
    prompt=["..."],
    sampling_params={"max_new_tokens": 32},  # at eb061d8, leaving it out raises TypeError
    steering={"feature": 40977, "alpha": 1.5, "mode": "conditional"},
)
```

`steering` comes after `return_routed_experts`, the last parameter `Engine.generate` took by
position, so a call that passes the earlier ones by position binds as it did.

[`../steering/from_sae.py`](../steering/from_sae.py) writes the bank: a decoder row per feature as
the vector, the matching encoder column as the probe, and the fitted JumpReLU threshold converted
into the units the projection returns.

The hook site is the only thing that compiles in. It's a Python int on the inner model, set before
`nnx.split()` bakes it into `model_def`, the same way `layers_to_capture` works. The bank, the bank
row per token, the strength per token and the threshold per token all arrive as device arrays on
`ForwardBatch`. So one executable covers every feature, every `alpha`, every mix of static and
conditional, and every choice of steered tokens, and two requests in one batch can steer with
different features. Static steering is conditional steering at a threshold of `-inf`.

A server that loaded a bank builds a `SteeringBatch` on every forward pass, including the ones
`CompilationManager` precompiles and the ones nobody asked to steer, so the count stays at one.
Those carry alpha 0 and a threshold of `+inf` and fire on no token. `ForwardBatch` is a
registered pytree and `forward_batch` is a jit argument, so `None` against a `SteeringBatch` is a
second treedef, a second program, and a bf16 residual stream where the first has float32. The
bank places the all-off arrays once per token count, and a `lax.cond` on alpha in the hook skips
the two bank gathers and the projection when no token steers. Such a pass costs the float32
widening and one reduction over the token axis.

The stream widens to float32 at the hook and stays wide, because rounding back to bf16 after the
add loses 99% of a small shift. Matmuls downstream stay bf16: `LinearBase` asks for
`preferred_element_type=params_dtype` and the MoE dispatch casts its input, so what widens is the
residual accumulation alone. With `--enable-steering` on and `sglang-jax-877.patch` applied, the
per-layer capture comes back float32 for every captured slot, since `jnp.concat` promotes, which
doubles both the HBM and the wire figures in
[`../docs/activation-capture.md`](../docs/activation-capture.md). That holds when a captured slot
sits past `--steering-layer`. A `--return-hidden-states-layers` list with every slot at or below
it stays BF16. With the flag off nothing widens, and capture stays BF16 even with this patch
applied.

`gemma4` first, because its layer carries one residual stream rather than a `(hidden, residual)`
pair, so the hook is one call and nothing to unpack. `qwen3-steering-hook.patch` covers the paired
form. The patches are independent and apply in either order.

The steering is part of the radix-cache key. Steering changes the KV every layer above the hook
writes, so the tokenizer manager appends it to the request's `extra_key`, the way LoRA appends
`lora_id`. Two requests that steer the same way share a cached prefix, and a steered request and
an unsteered one with the same prompt never do. A position inside a reused prefix was steered
when that prefix ran, by a request with the same positions. Every steered namespace starts with
`steering:`, and a server with `--enable-steering` refuses a client `extra_key` that holds it, so
no request can name a steered request's namespace and share its KV.

The server refuses `--enable-steering` with `--speculative-algorithm`. The verify pass builds its
batch without the per-token steering arrays, so a steered request would steer its prefill and
none of the tokens after it. Three draft workers build a `ModelRunner` with the target's server
args and `is_draft_worker=True`, and `_setup_steering()` runs only outside that gate.

`SteeringBank.load` refuses a bank whose thresholds don't match its features, an empty bank, a
NaN or `-inf` threshold, a non-finite vector or probe, and a feature listed twice. A `+inf`
threshold marks a dead latent. The bank loads with a warning that names the feature, because
static steering reads no threshold and still adds the vector. The `from_sae.py` command line
refuses to write such a bank, and its `save_bank` warns. The request validator refuses an `alpha`
or a finite threshold past the float32 range, which the scheduler's float32 arrays would turn into
`inf`. It refuses a key outside `feature`, `alpha`, `mode`, `threshold` and `positions`, because
the scheduler reads a missing field as its default and a misspelled `positions` would steer every
token. It refuses a `threshold` beside `"mode": "static"`, which the scheduler would serve as
conditional steering. A `threshold` with no `mode` means conditional steering at that threshold.
A server started without `--enable-steering` refuses any request that sets `steering`, which
would otherwise come back unsteered with no error.

`ForwardBatch.init_new` places the per-token arrays with `device_array`, like every other array it
builds, so a mesh that spans hosts pays no cross-host check per forward pass. The steering dict is
logged at debug level where the tokenizer manager sends it, where the scheduler receives it and
lays it out per token, and where `init_new` resolves it against the bank.

10 files, +822.

The bank gather names its output sharding. The engine traces under a mesh whose axes are
Explicit, and there a gather with a replicated operand and token-sharded indices has no inferable
output sharding, so jax raises `ShardingTypeError` on the first steered request. It reuses the
sharding of `hidden_states`, which already has the shape the gather returns. Check 9 of
[`test_steering_hook.py`](test_steering_hook.py) runs that path and requires the un-named form to
fail.

## `qwen3-steering-hook.patch`

The same hook for `qwen3`. Needs `steering-hook.patch`, which carries the flags, the bank loader
and `layers/steering.py`. `_setup_steering()` looks for `steering_layer` on the inner model and
names the model in its error, so it stays model-agnostic and this patch only adds the site.

```bash
git apply "$REPO/upstream/steering-hook.patch"
git apply "$REPO/upstream/qwen3-steering-hook.patch"
```

A `qwen3` layer returns `(hidden_states, residual)` and defers the add, so the stream leaving the
layer is `hidden_states + residual`, which is what capture reports. The hook folds the pair before
it steers and passes `residual=None` down:

```python
if residual is not None:
    hidden_states = hidden_states + residual
    residual = None
hidden_states = apply_steering(hidden_states, forward_batch.steering)
```

Adding the shift to `hidden_states` alone would move the combined stream by the same amount, so
static steering would look right. Conditional steering wouldn't. It projects the stream onto a
probe to decide whether to fire, and half the stream is the wrong vector to project.

The hook runs after the layer's deepstack add. Qwen3-VL runs `QWen3Model` and adds a plane of
vision features to the image tokens of its first layers, and capture slot `N + 1` reads the stream
after that add. A probe read before it would compare image tokens against a threshold fit on a
different tensor.

1 file, +15.

### Test

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/test_steering_hook.py
```

CPU only, 8 simulated devices. Fourteen checks, each with a negative control that must fail. A
mutation that finds nothing to change fails the run too. The test fetches `eb061d8`, where the
patch applies. From upstream `5acff08a` on, the patch stops applying. Checks 11 and 12 start the
Engine, so the venv needs the packages `sgl_jax` imports at startup, which
[`models/requirements.txt`](models/requirements.txt) lists. Set `SGLANG_JAX_REPO` to a local
checkout that holds `eb061d8` to skip the fetch.

1. The patch applies to a clean `eb061d8`, every file it touches compiles, and it applies both
   before and after `sglang-jax-877.patch`. `qwen3-steering-hook.patch` applies on top, and
   `qwen3.py` compiles. Control: the same patch with one context line rewritten is refused.
2. The patched files carry the parts of the hook, read out of their ASTs: the module, the site on
   the inner model, the gate below the `layer(...)` call, the `apply_steering` call, the pytree
   child at an index `tree_unflatten` agrees with, the model runner setup inside the draft-worker
   gate, the three server flags, the request field and its validator, the per-token merge, the
   three fields on `ModelWorkerBatch`, the cache key the tokenizer manager appends, the per-token
   placement with `device_array`, and the refusal of speculative decoding. `Engine.generate` and
   `async_generate` keep every parameter a caller could pass by position where it was. Controls:
   the unpatched text carries none of the parts, and `steering` ahead of `return_routed_experts`
   reads as a moved parameter.
3. The patched `Gemma4Model.__call__` runs under `jit` on a token-sharded input, at float32 and
   bfloat16, against a float64 reference that steers one token at a time. A NaN layer counts as
   the worst error. Control: five mutants, each caught.
4. The shipped arithmetic on a batch that mixes static steering, conditional steering, a feature
   the bank doesn't hold and tokens nobody steers, plus a bank written by `from_sae.py`'s
   `save_bank` and read back by the server at the site it was fit on. That bank holds a dead
   latent, and `load` names it in a warning. `load` refuses a short thresholds array, an empty
   bank, a NaN or `-inf` threshold and a NaN in a vector. Control: six mutants, among them the
   steering branch swapped with the branch that only widens, a malformed bank, the same bank at
   three other sites, a bank of live latents that loads with no warning, and the same read at
   bfloat16.
5. The scheduler's per-token rules: positions map through the prefix, a position in another chunk
   drops, a repeat steers once, and the three ways a threshold is chosen.
6. Every key `_merge_steering` returns is a field of `ModelWorkerBatch`, against the field list
   read out of the patched source and rebuilt as a dataclass. The call spreads `**_steering`, so
   a key with no field raises on every forward pass, steered or not. Control: the unpatched field
   list refuses the same spread.
7. `ForwardBatch.init_new` builds a steering batch whether or not a request asked for one, and
   the steered and unsteered calls share one trace. A second all-off batch of the same size gets
   the placed arrays back, and the branch an all-off batch takes gathers nothing out of the bank.
   Controls: `None` in place of the all-off batch takes two traces and returns a bf16 stream, an
   all-off batch of another size gets its own arrays, and a hook with no branch gathers on every
   batch.
8. The request validator rejects a malformed feature, alpha, threshold, mode and positions list,
   an alpha or a threshold past float32, a key outside the five fields, and a threshold beside
   static mode, and its error names the field. Control: for each, what the scheduler thread makes
   of the value if it gets through, which is a raise on the thread that serves every request, an
   `inf`, or an answer the request didn't ask for. Each control has to show that failure.
9. `apply_steering` runs under a mesh whose axes are Explicit, like the engine's, and matches the
   float64 reference. Control: the same gather without `out_sharding` raises `ShardingTypeError`.
10. The cache key names every field that decides what the hook does, two spellings of one request
    share a key, and every key starts with `steering:` and holds it once. Controls: a key that
    forgets the positions fails the cases, and a key whose body holds the prefix fails the last.
11. The real Engine on CPU, with a tiny random Qwen3, the radix cache on and a bank at layer 1:
    a steered and an unsteered request with the same prompt reuse none of each other's KV, and
    each matches its fresh-cache output. HTTP `/generate` refuses an `extra_key` that names a
    steered namespace. The debug log shows the steering dict at every crossing. Controls: steering
    changes the output, a second request that steers the same way reuses the cache, and an
    `extra_key` that names no steered namespace is served.
12. The real Engine refuses `--enable-steering` with `--speculative-algorithm`, and a server
    started without `--enable-steering` refuses a steered request through the Engine and through
    HTTP. Controls: without `--enable-steering` the same speculative setting passes that check,
    and the server without the flag serves the same prompt unsteered.
13. On a mesh that spans two processes, `ForwardBatch.init_new` places the per-token arrays with
    no cross-process gather. Both processes write to files, and a pair that runs past its timeout
    gets killed. Control: `jax.device_put` on the same arrays gathers.
14. The patched `QWen3Model.__call__` steers after the deepstack add. A plane added to the image
    tokens at the hook's layer moves their projection past the threshold, so the hook fires on
    them only when it reads the stream after the add, and the captured layers match a float64
    reference that steers there. Controls: the same stack unsteered, and the hook moved above the
    add.

| Captured, 6 layers | Relative error | Pearson |
|---|---|---|
| float32 | 2.8e-07 | 1.000000000 |
| bfloat16 | 9.1e-03 | 0.999989 |

The bfloat16 read moves 3 of 4,096 tokens across the threshold against the float64 answer, and
the float32 read moves none. A token that flips takes the whole `alpha * v` with it.

## `capture-hooks/`

The same hook for `kimi_linear`, `qwen3_5`, `deepseek_v3` and `glm4_moe`, one patch each. All four
split the stream into `(hidden, residual)`, so each appends the pair's sum and takes
`hidden_states` alone on layer 0, the form the survey section below describes.
[`capture-hooks/test_capture_hooks.py`](capture-hooks/test_capture_hooks.py) runs each patched
model under `jit` against a float64 reference. See
[`capture-hooks/README.md`](capture-hooks/README.md).

## `models/`

A model the engine doesn't have, then its hook. `kimi-k3-model.patch` adds `models/kimi_k3.py` for
`moonshotai/Kimi-K3`, with the `situ` activation, an MXFP4 dequantizer for its routed experts, the
K3 fields in `configs/kimi_linear.py`, the `KimiK3Config` that reaches the language tower under
`text_config`, and a `narrow` field on `WeightMapping` that keeps the 96 per-head `A_log` values
at the front of the 128 the checkpoint ships. `kimi-k3-capture-hook.patch` adds the hook on top.

`inkling-model.patch` adds `models/inkling.py` for `thinkingmachines/Inkling` and
`thinkingmachines/Inkling-Small`, with the learned relative position bias that stands in for a
rotary table, the four short convolutions each layer holds state for, the shared-expert sink
router and the MTP head. It carries the hook already, so it needs no second patch. It also adds
`configs/inkling.py`, which keeps the published routed-expert width, and
`mem_cache/short_conv_state_pool.py`, which holds each request's convolution windows. Its edits to
upstream code make the runner take the window pool's bytes out of the KV budget, and make the
native attention backend read the bias and map each batch's slots into the sliding-window pool.
They let the KV-head padding fire on a checkpoint whose projections aren't named `k_proj` and
`v_proj`, and give the precompile batches their pool slots. Before it reads a weight, the worker
refuses a launch that sets `--dp-size` above 1, picks an attention backend other than `native`,
leaves the radix cache on, turns on speculative decoding, sets `--enable-recurrent-extra-buffer`,
asks for PD disaggregation, or sets one of the three EPLB flags.

`nemotron3-model.patch` adds `models/nemotron_h.py` for both Nemotron 3 sizes, with the config
that resolves their irregular layer stack, the Mamba-2 chunked scan and the backend that owns the
recurrent state, a non-gated squared-ReLU expert path in `EPMoE`, and a `GroupRMSNorm` layout for
the 8 RMS groups the gated norm runs on a 32-way tensor axis.
`nemotron3-capture-hook.patch` adds the hook on top.

`gpt-oss-model.patch` adds `models/gpt_oss.py` for `openai/gpt-oss-120b` and
`openai/gpt-oss-20b`, with the per-head attention sink, the clamped SwiGLU and per-expert biases
in `EPMoE`, the untruncated YaRN correction range the config asks for, and a loader that decodes
the MXFP4 expert GEMMs. It carries the hook already, so it needs no second patch.

See [`models/README.md`](models/README.md).

## `capture-hook-survey.csv`

Which `sglang-jax` models have the `layers_to_capture` hook upstream. Four of 31: `gemma4`,
`llama`, `qwen3`, and `qwen3_vl`, which runs a `QWen3Model` backbone and captures through its
hook. The patches here add five more, plus the models [`models/`](models/) writes.

Adding it to a model means a list on the inner model, an append in the layer loop, a flag on the
outer `ForCausalLM`, and threading `aux_hidden_states` into the logits processor. What to append
depends on how many streams the layer carries. A model that splits into `(hidden, residual)` adds
them back, and takes the residual only when there's one, because it's `None` on layer 0:

```python
if layer_id in self.layers_to_capture:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )
```

A model that carries one stream, `gemma4`, `inkling` and `kimi_k3` among them, has no `residual`
to read and appends `hidden_states` alone. `gemma4` is the model `steering-hook.patch` targets,
so that's the form to reason from when picking what an SAE trains on.

This form needs an unrolled Python layer loop, which every upstream model writes. A stack built
from `lax.scan` over stacked weights can't use it. A scan body traces once, so the append runs
once however many steps the scan takes, and `layer_id` arrives as a tracer that no `in` test can
read. Return the hidden states as the scan's `ys` instead and reshape after the scan. See
[`../models/nemotron3-super.md`](../models/nemotron3-super.md) for the worked form.
