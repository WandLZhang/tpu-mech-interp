# Sequence-sharded affine scan

Mamba-2 and KDA layers hold a recurrent state. A paged-attention cache manager has no slot for it,
and the scan that updates it runs sequentially, so a sequence shard can't split it.

The inter-chunk recurrence is affine in the state:

```
h' = A @ h + B
```

Affine maps compose associatively:

```
(A_r, B_r) o (A_l, B_l) = (A_r @ A_l,  A_r @ B_l + B_r)
```

So the scan becomes three steps. Each device folds its local chunks into one `(A, B)` pair. The
pairs compose across devices. Each device replays its chunks from the state it receives.

Each device sends ceil(log₂ D) + 1 pairs per layer over D devices: one per prefix round, then one
for the closing shift. A pair holds a K×K and a K×V matrix per head, so it grows with the head count
and not with the sequence.

## Three implementation choices

| Choice | Why |
|---|---|
| `lax.scan` for the local fold | `lax.associative_scan` materializes the running composition at every chunk |
| Hillis-Steele prefix scan over `ppermute` | It sends ceil(log₂ D) + 1 pairs per device per layer, where an all-gather sends D - 1, so 4 against 7 at D = 8 |
| `jax.checkpoint` on both scan bodies | A plain scan stacks the autodiff residuals per chunk |

## Test

The tests need jax and numpy, which `scan/requirements.txt` lists. On a workstation they run on CPU
in a venv of their own:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r scan/requirements.txt
python3 scan/test_affine_scan.py
python3 scan/test_mamba2.py
python3 scan/test_kda.py
python3 scan/test_nemotron3_layers.py
python3 scan/test_inkling_layers.py
python3 scan/test_deepseek_v4_layers.py
```

Four of them build an 8-device mesh. They add `--xla_force_host_platform_device_count=8` to
`XLA_FLAGS` beside any flag already there, and their first line names the platform, as in
`mesh: ctx=8 on cpu`. On a TPU VM, run `source ~/.tpu_env` first. The same commands then run in
`~/v312` on the 8 chips of a `v5litepod-8`. A host with fewer chips needs `JAX_PLATFORMS=cpu`.
`test_nemotron3_layers.py` runs jax on one device and names its backend on its first line.
`test_deepseek_v4_layers.py` runs no jax.

`transformers` is optional. Where it's installed, three tests also check against its reference
classes: `test_kda.py` against the GLM-5.3-Flash and Kimi-Linear forget gates, which need `torch`
too, `test_nemotron3_layers.py` against a save through `NemotronHConfig`, and
`test_inkling_layers.py` against `InklingTextConfig`. `transformers` 5.17 ships all four models.
5.12.1, the version in `~/v312`, ships `nemotron_h` alone, so the other checks skip, and each skip
names what's missing. The venv for the CPU gates in the top-level README has all of them.

`test_affine_scan.py` compares the sharded path against the recurrence applied chunk by chunk. On
CPU:

| Case | Chunks | Heads | K | V | max_abs | rel | cosine |
|---|---|---|---|---|---|---|---|
| small | 16 | 2 | 8 | 8 | 2.98e-08 | 5.82e-08 | 1.0 |
| gdn-like | 64 | 4 | 128 | 128 | 8.94e-08 | 1.51e-07 | 1.0 |
| deep | 256 | 2 | 64 | 64 | 8.94e-08 | 1.49e-07 | 1.0 |

The reference never builds `A` or `B`, so it checks the reformulation rather than restating it.

Each case also runs a negative control that breaks the cross-device chain, and asserts the error
is large. The test fails itself if a control ever passes. Two more checks pin claims above. The
pairs `incoming_state` sends are counted in its jaxpr at every mesh size up to 8, with a ring
all-gather as the control. And the device count lands in `XLA_FLAGS` beside another flag, with
`setdefault` as the control.

## Use

```python
a_tot, b_tot = compose_local(a_chunks, b_chunks)      # [C,...,K,K] -> [...,K,K]
h_in = incoming_state(a_tot, b_tot, h_init, "ctx")    # inside shard_map
states = replay_local(a_chunks, b_chunks, h_in)       # [C,...,K,V]
```

Producing `a_chunks` and `b_chunks` is per-architecture work. `mamba2.py` does it for Mamba-2 and
`kda.py` does it for the gated delta rule.

`shard_map` gives every device the same number of chunks, so the chunk count has to divide over the
mesh. A short sequence pads with whole chunks that fold to the identity pair. `pad_chunks` does
the padding, and `mamba2.pad_to_chunks` and `kda.pad_to_chunks` call it with the arrays their layer
folds. Both take `num_devices` and return the token count before padding.
`entering_states` shifts the states `replay_local` returns so each chunk gets the state before it.

All three steps run under `shard_map`'s default `check_vma=True`. A scan carry has to keep one type,
and inside `shard_map` that type records which mesh axes a value varies over, so `compose_local` and
`replay_local` cast their starting carry to vary where the chunks do. An initial state passed
replicated works too.

## Mamba-2

The SSD recurrence is affine in the state. Per token, per head:

```
h_t = exp(dt_t * A_h) * h_{t-1} + B_t (dt_t x_t)^T
```

`A_h` is one negative scalar per head, so the decay is a scalar. Fold a chunk of L tokens:

```
g_t     = dt_t * A_h
cum_t   = sum_{s <= t} g_s
A_chunk = exp(cum_{L-1}) * I
B_chunk = sum_t exp(cum_{L-1} - cum_t) B_t (dt_t x_t)^T
```

The terms of `B_chunk` don't depend on each other, so a chunk is one einsum. Only the chunk to
chunk step stays sequential, and that's the step `affine_scan.py` takes over.

```python
dt, log_decay = discretize(dt_raw, dt_bias, a_log, cfg.time_step_limit)
a_chunks, b_chunks = chunk_pairs(x, b, dt, log_decay, chunk_size=128)
```

`discretize` adds the bias, takes the softplus, clamps to `time_step_limit`, then multiplies by
`A = -exp(A_log)`. `config.json` spells that clamp `mamba_dt_limit`. Both published Nemotron 3
sizes leave it at `(0.0, inf)`, where neither end binds, and the Mamba-2 CUDA kernels, Megatron-LM
and vLLM apply that pair. The torch path of `NemotronHMamba2Mixer` in `transformers` 5.17 builds
its own pair, `(time_step_min, inf)`, and floors prefill `dt` at 0.001. NVIDIA's torch path in the
checkpoint repo floors prefill and decode alike. `mamba2.py` follows the CUDA kernels, so it
differs from either torch path on every step whose softplus lands under 0.001. On the published
weights, 212 of Super's 5,120 Mamba-2 heads have `softplus(dt_bias)` below 0.001. `time_step_min`
and `time_step_max` are a separate pair, the range the `dt_bias` initializer draws over. Both ends
of the clamp apply, so a config that ships a finite upper limit gets it.

`K` is `ssm_state_size` and `V` is `mamba_head_dim`, so the state index comes first. The reference
cache stores the transpose; `as_cache_layout` converts.

`A_chunk` is a scalar times the identity. `chunk_terms` returns that scalar, one number per chunk
per head, where `chunk_pairs` writes out the K×K form the generic scan takes. That's 16,384 times
as many numbers at K=128, or 16 GiB of `a_chunks` per layer against 1 MiB at the full 262,144
context, so take `chunk_terms` when the caller can apply the scalar itself.

### From chunk state to layer output

`chunk_outputs` turns the chunk boundary states into the per-token layer output. It adds two terms
and the `D` skip:

```
y_t = exp(cum_t) C_t . h_in
      + sum_{s <= t} exp(cum_t - cum_s) (C_t . B_s) dt_s x_s
      + D_h x_t
```

`h_in` is the state entering the chunk that holds token `t`, so a sequence shard needs nothing from
its neighbors beyond the pair the scan already carries. Inside a chunk the sum over `s` is one
masked einsum.

```python
states = replay_local(a_chunks, b_chunks, h_in)                        # [C,H,N,P]
y = chunk_outputs(x, b, c, dt, log_decay,
                  entering_states(states, h_in), chunk_size, d=d)      # [T,H,P]
```

`y` is the mixer output before the gated norm and the output projection.

### Padding

`pad_to_chunks` right pads the token axis with `dt = 0` and `log_decay = 0`, so a padded token adds
nothing and the state at the end of a padded chunk is the state at the last real token. A sharded
run also needs the chunk count to divide over the mesh, so pass `num_devices`:

```python
xp, bp, dtp, gp, tokens = pad_to_chunks(x, b, dt, log_decay, chunk_size=128, num_devices=16)
```

2,000 tokens then pad to 2,048, which is 16 chunks, one per chip, and `tokens` comes back as 2,000,
which says which chunk boundaries are real. Padding to whole chunks alone gives a 1,000 token prompt
8 chunks, which don't divide over 16 chips, and `shard_map` refuses the split.

### Dtypes and precision

`mamba_ssm_cache_dtype` is float32 in a BF16 model, so every function that accumulates upcasts and
returns float32, `causal_conv` included. `conv_halo`, `as_cache_layout` and `from_cache_layout`
only move data, so they pass the dtype through. Holding the state in BF16 instead costs between
1.5e4 and 9.7e4 times the float32 error over the three cases below, on CPU. The test gates that
ratio at 1e3 on CPU and 1e2 elsewhere, and prints "three orders" or "two orders" to match. On a TPU
the float32 side of the ratio already carries the platform's transcendental error.

Every matmul and einsum here asks for `Precision.HIGHEST`. A float32 `dot_general` at the default
precision rounds both operands to BF16 before the MXU multiplies them, which hands back the
accuracy the float32 cache dtype pays for. CPU always runs true float32, so the test reads the
jaxpr and fails on any `dot_general` that leaves the precision open.

The depthwise conv in front of the SSM reads `conv_kernel` tokens, so a sequence shard also takes
`conv_kernel - 1` tokens from its left neighbor. `conv_halo` returns that tail, and refuses a shard
too short to fill it.

Shapes for Nemotron 3 Super 120B-A12B, from `NEMOTRON_3_SUPER`:

| Field | Value |
|---|---|
| `ssm_state_size` (K) | 128 |
| `mamba_head_dim` (V) | 64 |
| `mamba_num_heads` | 128 |
| `n_groups` | 8 |
| `chunk_size` | 128 |
| `conv_kernel` | 4 |
| `conv_dim` | 10,240 |
| `mamba_ssm_cache_dtype` | float32 |

A Mamba-2 layer holds two slots per sequence, and `state_bytes_per_layer` adds them up. The SSM
state is `[128 heads, 64 head dim, 128 state]` in the cache layout, which is 4 MiB. The conv state
is `[conv_dim, conv_kernel - 1]`, which is 120 KiB: the current token arrives with the request and
the window needs the three before it. Neither one grows with the sequence. A cache manager that
allocates the SSM slot alone leaves decode reading a window the previous step never filled.

### Test

`test_mamba2.py` checks the chunk pairs against a float32 token-by-token reference fed the same
`dt`, on one device and over 8 shards. On CPU:

| Case | Chunks | Chunk | Heads | Groups | K | V | rel |
|---|---|---|---|---|---|---|---|
| small | 16 | 16 | 2 | 1 | 8 | 4 | 3.4e-07 |
| model shapes | 8 | 128 | 8 | 2 | 128 | 64 | 1.0e-05 |
| many chunks | 32 | 64 | 4 | 2 | 32 | 16 | 1.3e-05 |

`A` and `dt_bias` are drawn the way the reference initializer draws them, so `A` spans
`1 .. mamba_num_heads` and `dt` spans the `time_step` range. Heads at the top of that range forget
a chunk of state completely and heads at the bottom keep nearly all of it. `rel` therefore takes a
denominator per head: one global denominator hides a small head's error behind a large head's
magnitude. Every control has to clear the same bar the implementation passes, 1e-4 on CPU and 5e-4
elsewhere.

Three controls run per case, and each has to fail: the intra-chunk decay dropped, the segment sum
off by one, and the cross-device chain broken. Seven more cases follow.

| Case | Compares against | Controls |
|---|---|---|
| discretize | a float64 transcription of the mixer, and float32 `exp` and `log1p` against float64 | floor dropped, cap ignored, bias dropped, `A` not negated |
| layer output | a float64 token-by-token recurrence | causal mask dropped, `D` skip dropped, inherited state dropped |
| padding | 87 tokens padded to 96 on one device and 128 over 8 | left padded, no devices, chunks of zero tokens |
| conv | a float64 conv written from the definition | halo dropped, taps reversed, BF16 accumulation, shard shorter than the halo |
| cache layout | a recurrence written in the cache order | transpose dropped |
| config | `conv_halo`, `heads_per_group` and the three `state_bytes` properties against the shapes the cache allocates | the conv slot counted at the full kernel width, a square SSM slot, the state held in 2 bytes |
| matmul precision | the jaxpr of the whole path | a `dot_general` with the precision left open |

`exp` and `log1p` are the two float32 ops softplus runs through. The discretize case prints what
each costs on the platform it runs on: 7.4e-08 and 1.4e-07 on CPU.

## KDA

The gated delta rule is affine in the state too. The difference from Mamba-2 is that a token reads
the state before it writes, so the terms inside a chunk depend on each other. Per token, per head:

```
g_t = gate_lower_bound * sigmoid(exp(A_log) * (a_t + dt_bias))
u_t = beta_t * (v_t - (diag(exp(g_t)) h_{t-1})^T k_t)
h_t = diag(exp(g_t)) h_{t-1} + k_t u_t^T
```

The decay is per key channel, not per head. Unrolling a chunk of L tokens gives a triangular system
rather than a sum:

```
c_t       = sum_{s <= t} g_s
M[t, s]   = beta_t * sum_d k_t[d] k_s[d] exp(c_t[d] - c_s[d])      s < t
(I + M) U = diag(beta) (V - K_hat h_0)
A_chunk   = diag(exp(c_L)) - K_g^T w
B_chunk   = K_g^T u
```

`u` and `w` are the two right hand sides of one solve, and neither reads the incoming state, so a
whole chunk costs one triangular solve and two contractions.

```python
g = gate_log(a_raw, dt_bias, a_log, cfg.gate_lower_bound)
a_chunks, b_chunks = chunk_pairs(k, v, beta, g, cfg.chunk_size)
```

That's the gate every served KDA path computes once the config sets `gate_lower_bound`: the
sglang-jax prefill, Mega and decode kernels at `eb061d8`, and `Glm5NextTextForgetGate` in
`transformers`. The sigmoid keeps `g_t` between the bound and zero. It isn't the softplus gate
`-exp(A_log) * softplus(a_t + dt_bias)` with a clamp under it, and that softplus gate is what runs
with no bound. At `a_t + dt_bias = 0` and `A_log = 0` the two give -2.5 and -0.69. On the wide gate
the test draws, the two differ by more than 1e-3 on 100% of entries, and folding the softplus gate
instead moves the chunk states by 0.99 relative.

`gate_log` and `KDAConfig` take the bound with no default, and `None` picks the softplus gate. A
config without `gate_lower_bound` reads as `None` in sglang-jax, so Kimi-Linear serves the softplus
gate, while `Glm5NextTextConfig` in `transformers` fills in -5.0.

Every KDA checkpoint here stores `A_log` one value per head. Kimi-Linear ships `[1, 1, H, 1]`.
Kimi K3 ships a flat `[128]`, which holds its 96 per-head values and then 32 zeros. `gate_log` reads
a flat vector by its length, `[H]` as one value per head and `[K]` as one per channel, so pass
Kimi K3's as `A_log[:96]`. At a head count equal to the channel count a flat vector could run along
either axis, so `gate_log` refuses it there and takes `[H, 1]` or `[1, K]` instead.

`A_chunk` is a diagonal plus a rank-L update. `chunk_factors` returns the four factors and
`chunk_pairs` materializes the dense K×K form the generic scan takes. The factors hold fewer floats,
24,704 against 32,768 at the shipped shapes, and stay smaller up to `L = 84`. Replay goes the other
way: `apply_factored` costs `2 L K V` against `K^2 V`, so it only wins while `L < K / 2`. Both
models pair `chunk_size = 64` with `head_dim = 128`, where the two cost the same.

Every contraction asks for `Precision.HIGHEST`. A float32 `dot_general` at the default precision
rounds both operands to BF16 on the MXU, which takes the fold from 8e-07 to 3e-03. The one op left
open is the triangular solve, which takes no precision argument. `cond(I + M)` stays between 1.0
and 2.3 on these shapes, so the solve doesn't amplify what it's handed.

The state stays float32. Holding it in BF16 returns a finite answer that's wrong by 23%, because
`c_L` reaches -320 where one BF16 step is 2.

Shapes for the two KDA models, from `KIMI_K3` and `GLM_5_3_FLASH`:

| Field | Kimi K3 | GLM-5.3-Flash |
|---|---|---|
| `num_heads` | 96 | 64 |
| `head_dim` (K and V) | 128 | 128 |
| `chunk_size` | 64 | 64 |
| `short_conv_kernel_size` | 4 | 4 |
| `gate_lower_bound` | -5.0 | -5.0 |
| KDA layers | 69 of 93 | 34 of 45 |
| recurrent state per sequence, float32 | 414.0 MiB | 136.0 MiB |
| short-conv windows per sequence, BF16 | 14.6 MiB | 4.8 MiB |
| state per sequence | 428.6 MiB | 140.8 MiB |

The rest of the layers are gated MLA in Kimi K3 and sparse MLA in GLM-5.3-Flash. The short depthwise
conv in front of the projections is the same op Mamba-2 runs, so `mamba2.causal_conv` and
`mamba2.conv_halo` cover the shard halo. Its windows on `q`, `k` and `v` are the second slot a KDA
layer holds per sequence. sglang-jax's `RecurrentStatePool` keeps them in BF16 beside the float32
recurrent state, and `KDAConfig.state_bytes` adds both, the way `Mamba2Config` adds its two slots.

### Test

`test_kda.py` checks the gate first, against six values worked by hand, a float64 transcription of
the served formula in every `A_log` layout, and `Glm5NextTextForgetGate` and `KimiLinearForgetGate`
when `transformers` has them and `torch` is installed. Its controls are the softplus gate clamped at
the bound, a per-channel `A_log` read per head, and the bounded gate graded against Kimi-Linear. A
flat `A_log` has to be refused when the head count equals the channel count, and so does a
`gate_log` call or a `KDAConfig` that leaves the bound out.

Then it checks the chunk pairs against a float32 token-by-token reference fed the same gate, three
ways: folded on one device, replayed through `apply_factored` without ever building `A`, and run
through the sharded path over 8 shards. On CPU:

| Case | Chunks | Chunk | Heads | K | V | rel |
|---|---|---|---|---|---|---|
| small | 16 | 16 | 2 | 8 | 8 | 3.4e-07 |
| model shapes | 8 | 64 | 4 | 128 | 128 | 9.1e-07 |
| many chunks | 32 | 32 | 4 | 32 | 32 | 5.0e-07 |
| split gate | 8 | 64 | 4 | 32 | 32 | 5.2e-07 |

`rel` takes a denominator per head. One global denominator hides a small head's error behind a large
head's magnitude, and on the wide gate the worst head reads 1.5x the global figure.

The gate is tempered in the first three cases so a chunk keeps part of its incoming state. On a wide
gate a chunk forgets everything, `A` goes to zero, and a broken cross-device chain looks correct.
The `split gate` case gets both at once: half the key channels sit on the gate bound and half stay
open, so `c_L` reaches -320 while the state still survives the chunk. Every case asserts that `A`
carries something before it compares anything.

Four controls run per case and each has to fail: the intra-chunk decay dropped, the delta correction
dropped, the carry-out off by one, and the cross-device chain broken. A fifth build breaks nothing
and has to reproduce the implementation, which is what makes the other four attributable. Six more
cases follow.

| Case | Compares against | Controls |
|---|---|---|
| wide gate | a token-by-token reference at `A_log = log(Uniform(1e-9, 16))`, plus `I + M` finite and `cond(I + M)` under 10 per head | `exp(c_t) exp(-c_s)` for the pairwise decay, the softplus gate in place of the bounded one, BF16 accumulation |
| positive gate | the same reference with 75% of gate entries above zero | the pairwise decay clamped at zero instead of masked by position |
| padding | every chunk boundary of 121 tokens padded to 128, on one device and over 8, and 37 tokens padded out to one chunk per device | left padded, and a stride that doesn't divide the token count |
| config | the halo, the recurrent state and the conv windows against literals written from the shapes | the total without the conv windows, the windows counted at the full kernel width, the windows held in float32 |
| cost model | the two crossovers the module docstring quotes | the factored replay priced at `L < K` instead of `L < K / 2` |
| matmul precision | the jaxpr of the whole path, which has to ask for `HIGHEST` everywhere | a `dot_general` with the precision left open, at `DEFAULT`, at `HIGH`, as `'bfloat16'`, and as a BF16 algorithm preset |

## Inkling

`inkling_layers.py` parses `thinkingmachines/Inkling` and `Inkling-Small` into the runs of layers
that can share one `lax.scan`. The stack breaks twice and the breaks don't line up: `dense_mlp_idx`
splits dense MLPs from routed ones, and `local_layer_ids` gives a 5:1 sliding-window to full
attention cycle. That's 23 runs over 66 layers and 15 over 42, in three kinds.

```python
cfg = INKLING
cfg.scan_groups()        # 23 runs, each one lax.scan over stacked weights
cfg.positional()         # the relative scheme that stands in for RoPE
```

These layers hold no decaying state, but `use_sconv` puts four depthwise causal convolutions in
each one, kernel 4. `sconv_chunk_pairs` writes that 3-token window as affine `(A, B)` pairs, so the
same three steps carry it. A chunk of 3 tokens or more replaces the window whole, so `A` is zero
and the composition collapses to a halo exchange. `sconv_incoming_state` takes that collapse in one
`ppermute` where `affine_scan.incoming_state` would spend ceil(log₂ D) rounds before the same
shift.

Right padding never reaches a real token's output. It does reach the windows past the sequence end,
so pass `sconv_chunk_pairs` the real token count as `tokens`. The window leaving the last shard is
then the last three real rows, the window a decode step reads.

`test_inkling_layers.py` checks the runs against the group boundaries the model card gives, checks
`dense_mlp_idx` and `local_layer_ids` against the published fields, and checks the convolution over
8 shards against a conv written out tap by tap in the test file. It runs the real per-layer sconv
widths, 16,384 channels and 14,336, at one chunk per shard and at three. It also pads 37 and 33
tokens out to one chunk per shard. The real outputs have to match the unpadded conv, and the final
window has to be the last three real rows. It pins every sizing number `models/inkling.md` quotes,
including the conv-state table and the three shares beside it, and it pins `tau` to its closed form
rather than to the shape of the curve.

The published fields live twice, as literals in `inkling_layers.py` and as a transcription of
`config.json` in the test, and one check compares the two copies. Every other reference value
comes from the transcription, so nothing is compared to itself.

The parser reads two spellings. The published file writes `local_layer_ids`, `dense_mlp_idx`, the
dense width as `dense_intermediate_size` and the expert width as `intermediate_size`. A save through
`transformers` writes `layer_types`, `mlp_layer_types`, the dense width as `intermediate_size` and
the expert width as `moe_intermediate_size`, and renames `model_max_length` and `sconv_kernel_size`.
`InklingTextConfig` also reads `sliding_window` as `sliding_window_size` and `num_local_experts` as
`n_routed_experts`, so the parser does too. Where a file holds both spellings of one field they have
to agree, and `intermediate_size` with neither width field beside it is refused.

Defaults follow `InklingTextConfig` in `transformers`, the class the engine port reads a config
through, so a field a file leaves out takes the value the served model takes. A config without
`local_layer_ids` means the 5:1 cycle `{i for i in range(layers) if (i + 1) % 6}`, not zero local
layers, and a config without `use_sconv` means the four convolutions are on.

Eighteen controls run per size, and each has to fail: one transcribed field bent, layer 6 made
global, `dense_mlp_idx` moved, `local_layer_ids` set to an empty list, the log scaling floor
removed, `tau` measured against the same curve on a base-10 log, a `text_config` block that
disagrees with the outer dict, `layer_types` that disagrees with `local_layer_ids`, a dense MLP
after a routed one, `dense_mlp_idx` that disagrees with `mlp_layer_types`, `model_max_length` that
disagrees with `max_position_embeddings`, `intermediate_size` with no width field beside it, the
cross-device chain broken, the window read from the head of each chunk instead of its tail, every
layer made global, the kernel shortened to 3, `use_sconv` turned off, and the local KV heads halved.
Five more run once: `sliding_window_size` written into a config that leaves the rest out, and each
of the four aliases the parser reads written beside its field with a different value. The padding
check runs four more: the real count left out, at 37 tokens and at 33, and a real count shorter
than the window or past the padded rows, which `sconv_chunk_pairs` has to refuse. Each one perturbs
an input and requires the output to move or the parser to refuse it.

[`models/inkling.md`](../models/inkling.md) has the shapes and the slice sizing.

## DeepSeek V4.1-Flash

`deepseek_v4_layers.py` parses `deepseek-ai/DeepSeek-V4.1-Flash`. No layer here holds a decaying
state, so the affine scan sits this one out. What breaks is the assumption underneath every scan in
this directory, that a layer can run on its own.

The 40 backbone layers split 20/20 into a causal encoder and a decoder, and layer 20 is the only
decoder layer that projects KV. CSA2 then gives every attention layer one of four modes. A `FULL`
layer writes the compressed KV and the index keys, a `REINDEX` layer reads both and runs its own
top-k, a `REUSE` layer reads all three, and a `WINDOW` layer takes no part. `kv_source_layer_ids`
and `index_source_layer_ids` name the owners, and a consumer binds to the nearest owner at or
before it.

```python
layer_plans(DEEPSEEK_V41_FLASH_CONFIG)   # 43 blocks, each with its mode, its sources and its prefix
unit_runs(DEEPSEEK_V41_FLASH_CONFIG)     # 9 runs, one lax.scan each
check_causal_encoder_decoder(DEEPSEEK_V41_FLASH_CONFIG)
```

A unit is one owner plus the `REUSE` layers bound to it. A `REUSE` layer right after a `WINDOW`
layer still binds to its owner and opens a unit of its own. Nine runs cover the 43 blocks, and the
widest is four identical `(Reindex, Reuse, Reuse, Reuse)` units from layer 24. Two units fold into
one run only when their weight sets match and their layers play the same part in the candidate
pool. The weights separate the three DSpark draft blocks, which share a stage, a mode and a ratio
and carry three different parameter sets. The pool role keeps the layer that builds the pool out of
the run of the layers that index inside it, and an indexer inside the pool has to run the
builder's compression ratio.

Those three blocks also sit outside the `layers.` namespace. That namespace stops at 39 and they're
`mtp.0`, `mtp.1` and `mtp.2`, so `LayerPlan.checkpoint_prefix` gives the prefix and
`extra_weights` gives the names, `.scale` siblings included.

`engram_plan` describes the two n-gram hash tables, 196.6B parameters over 768 million rows, and
`placement` prices holding them in chip HBM against holding them in host memory behind a prefetch.
A row index is a hash of token ids alone, so every index for a chunk is known before layer 0 runs,
and the fetch for a table 14 layers down hides under the 14 layers above it. Hosted tables share one
link a chip, so with both hosted the second fetch waits for the first.

`test_deepseek_v4_layers.py` grades the plan against four references that don't read it: the tensor
set and prefix the shipped safetensors headers declare for all 43 blocks, the size of the two
Engram shards on disk, the published bucket rule re-run with its own prime sieve, and the byte
figures the model card publishes. Thirty-four controls corrupt the config, and each has to be
rejected with the message that names the fault. Six more return a plan that's wrong but well formed,
or cost it the wrong way, and confirm the tensor comparison, the prefix comparison, the run-key
count, the pool role in the run key, the shared host link and the placement arithmetic catch it.

[`models/deepseek-v4.1-flash.md`](../models/deepseek-v4.1-flash.md) has the shapes, the slice
sizing and the Engram placement proposal.
