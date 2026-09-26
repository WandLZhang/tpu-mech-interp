# gpt-oss-120b

`openai/gpt-oss-120b`

**Status: measured on TPU**, `v5p-8`, 2026-09-25, with the layer filter. [Measured](#measured)
has two runs of this page's [Reproduce](#reproduce) block, on one Spot slice in europe-west4-b.
An earlier run that day lost its Spot slice to preemption during the weight fetch.

| | |
|---|---|
| Parameters | 116.8B total / 5.1B active |
| Layers | 36 |
| Hidden dim | 2,880 |
| Attention | GQA, 64 query / 8 KV heads, head_dim 64. Alternating sliding window(128) and full, attention sinks |
| MoE | 128 experts, top-4, intermediate 2,880, clamped SwiGLU |
| Position | YaRN, factor 32 over a 4,096 pretraining window, `truncate: false` |
| Weights | 60.8 GiB on disk, 217.6 GiB once the experts decode to BF16 |
| Precision | MXFP4 experts. BF16 attention, router, norms, embeddings and LM head |

`openai/gpt-oss-20b` is the same architecture at 24 layers and 32 experts. The same patch covers
it. It hasn't served on a chip yet.

## Serve

Engine: `sglang-jax` with [`upstream/models/gpt-oss-model.patch`](../upstream/models/gpt-oss-model.patch).
Slice: `v5p-8`, 4 chips, one host.

```bash
REPO=$PWD
git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
git checkout eb061d8
git -c user.name=you -c user.email=you@example.com am < "$REPO/upstream/sglang-jax-877.patch"
git apply "$REPO/upstream/steering-hook.patch" "$REPO/upstream/qwen3-steering-hook.patch"
git apply "$REPO/upstream/models/gpt-oss-model.patch"
```

That's the tree [`scripts/bootstrap_tpu_vm.sh`](../scripts/bootstrap_tpu_vm.sh) builds with
`--model gpt-oss`. The measured runs below served it, and the CPU test builds it too. `git am` makes
a commit, so the `-c` pair gives it a committer on a machine with no git identity.

```bash
python3 -m sgl_jax.launch_server \
  --model-path SNAPSHOT_DIR \
  --dtype bfloat16 \
  --tp-size 4 \
  --mem-fraction-static 0.8 \
  --swa-full-tokens-ratio 0.05 \
  --enable-return-hidden-states
```

Untested: the measured runs drove the engine from Python through
[`scripts/measure_model.sh`](../scripts/measure_model.sh), and no run here has started this server
line.

Give `--model-path` the local snapshot directory, not the repo id. The repo also carries
`original/` and `metal/` copies of the weights, and the engine's own fetch matches `*.safetensors`
and `*.bin` at any depth, so from the repo id it pulls both copies too, about three times the
bytes. `HF_HUB_OFFLINE=1` doesn't help: the offline check wants those same files.
[`scripts/fetch_weights.py`](../scripts/fetch_weights.py) fetches the top level alone and prints
the directory on its `PATH` line. `measure_model.sh` runs the same fetch.

```bash
python3 scripts/fetch_weights.py openai/gpt-oss-120b
```

Confirm the chip count first, after `source ~/.tpu_env`: a `v5p-8` reports 4.

```bash
python3 -c "import jax; print(jax.device_count())"
```

The model runs 8 KV heads. At `--tp-size 4` they land 2 to a chip, with no copies.

`tpu-inference` also serves the model, through vLLM's TPU platform. It has no per-layer capture,
so use it when you only want tokens out.

### Why `--swa-full-tokens-ratio 0.05`

gpt-oss runs 18 sliding layers and 18 full ones. `set_num_token_hybrid` gives each sliding layer
`ratio` times the tokens it gives each full layer, and the default ratio is 0.8. That parks
`0.8 / 1.8`, 44%, of the KV bytes in pools whose window is 128 tokens, and the sliding pool
recycles a slot as soon as it leaves the window.

Each full layer gets `2 / (ratio + 1)` times `max_total_num_tokens`. Dropping the ratio from 0.8
to 0.05 raises that from 1.11x to 1.90x, so the context and the concurrency both rise 1.7x. The
floor is the sliding pool itself: it has to hold `window + page_size` tokens per running request,
so read `swa_layer_tokens` out of the startup log and check it against
`max_running_requests * 128`.

### Scaling out to `v5p-64`

`v5p-64` is 32 chips on 8 hosts. Nothing here has run on it, capture included. Run the
`launch_server` command on all 8 hosts with `--tp-size 32`, `--dp-size 4`, `--nnodes 8`,
`--node-rank $RANK` and `--dist-init-addr $HOST0:$PORT`. `$RANK` is the host's index, 0 to 7.
`$HOST0:$PORT` is host 0's internal IP and a free port. The engine calls
`jax.distributed.initialize` only when `--nnodes` is above 1. This repo has no multi-host recipe.
Capture needs `--dp-size 1`, so a captured run there takes `--tp-size 32` alone and stores every
KV head four times.
`sglang-jax` keeps its launch templates in
[`docs/deployment/`](https://github.com/sgl-project/sglang-jax/tree/eb061d8/docs/deployment).

Run the chip check on all 8 hosts at once, with `jax.distributed.initialize()` before the count,
as the [JAX multi-process guide](https://docs.jax.dev/en/latest/501/multiprocess.html) asks. It
reports 32.

```bash
python3 -c "import jax; jax.distributed.initialize(); print(jax.device_count())"
```

#### Why `--dp-size 4`

Attention shards the 8 KV heads over `tp_size / dp_size` ways, so `--tp-size 32` alone asks 32
chips to split 8 heads and `ModelConfig.needs_kv_head_replication(32)` turns on: each chip stores
a whole KV head and every head is stored four times. `--dp-size 4` brings the attention shard
count back to 8, one head per chip, no copies.

The tax is 4x on the full-attention half of the cache. The KV pool pads each head from 64 to 128
wide, so across the 18 full layers a token costs 4 KiB per layer, 72 KiB in total. Replicated it
costs 288 KiB. One 131,072-token sequence holds 9 GiB rather than 36 GiB.

The expert GEMMs still shard over all 32 chips, because `EPMoE` builds its own
`(expert, tensor)` mesh from the whole device list.

## Capture

The patch writes `models/gpt_oss.py` with the `layers_to_capture` hook already in it, in the four
parts [`upstream/capture-hooks/README.md`](../upstream/capture-hooks/README.md) describes: the list
on `GptOssModel`, the append in the layer loop, `capture_aux_hidden_states` on
`GptOssForCausalLM`, and `aux_hidden_states` threaded into the logits processor. The layer carries
a `(hidden_states, residual)` pair, so the append uses the None-residual form.

36 layers at 2,880 dim is 203 KiB per token of host traffic. That's every slot.
`capture_activations.py` hands `--layers` to the engine as `--return-hidden-states-layers`, and
one kept slot moves 5.6 KiB per token. Both runs in [Measured](#measured) pass that flag.

## Measured

The Reproduce block below ran twice on 2026-09-25, back to back on one `v5p-8` Spot slice in
europe-west4-b, with the layer filter. 4 chips, one host, `tp_size=4`, BF16 engine with the experts
decoded from MXFP4 at load. 400 wikitext passages cut at 440 tokens, 8 to a batch. Capture keeps
slot 18 of 36 and writes it to `/dev/shm` as float32.

| | Run 1 | Run 2 |
|---|---|---|
| Capture off, tokens/s | 7,279.2 | 7,271.9 |
| Capture off, window | batches 3 to 32, first two discarded, 105,600 tokens in 14.51 s | the same batches, 105,600 tokens in 14.52 s |
| Capture on, tokens/s | 5,859.4 | 5,870.3 |
| Capture on, window | 45.74 s to 61.96 s, 95,040 tokens | 17.62 s to 33.81 s, 95,040 tokens |
| Peak HBM per chip, capture off | 78.27 of 95.73 GiB, 80 samples over 189.3 s | 78.27 of 95.73 GiB, 63 samples over 153.9 s |
| Peak HBM per chip, capture on | 78.34 of 95.73 GiB, 87 samples over 204.4 s | 78.34 of 95.73 GiB, 73 samples over 176.1 s |
| Capture cost | 1.24x the throughput, 0.07 GiB of HBM per chip | 1.24x, 0.07 GiB |
| Wire in the window | 33.8 MB/s | 33.8 MB/s |
| End to end, capture on | 2,732.0 tokens/s over 64.42 s | 4,850.2 tokens/s over 36.29 s |
| Peak duty cycle, capture off / on | 99.4% / 95.3% | 99.3% / 95.2% |

The wire carries 5,760 bytes per token, the one kept slot at BF16, and disk takes 11,520, the same
slot at float32. Each run wrote 176,000 tokens into one 1.888 GiB shard, and both shards carry the
same SHA-256, `2021f3af...`.

Before the layer filter, on 2026-09-24, the engine copied all 36 slots, 207,360 bytes per token,
and the same command read 1,875.8 tokens/s with capture on, 3.88x, at 78.72 GiB of peak HBM.

HBM is the peak of what [`scripts/peak_hbm.py`](../scripts/peak_hbm.py) sampled every 2 s across
the load and the run. All 4 chips read the same. Duty cycle is the busiest chip's peak, and the
other 3 sit within 0.1 points of it. Both engine starts, capture off and on, set
`mem_fraction_static=0.8`, where the Gemma pages use the default 0.6. The engine sizes its KV pool
to fill that fraction, so compare the peaks with that in mind. The weights take 54.4 GiB of each
chip, which is 217.6 GiB over 4. That's arithmetic. `peak_hbm.py` reads only whole-chip totals.

Run 1's first capture reply came 34.9 s after the timer started, and run 2's after 6.8 s, which
sets the two end-to-end rates apart. Run 1 compiled in that first call, and run 2 found the result
in the compile cache.

No `RESULT` line carries the duty cycle, the end-to-end rate or the first reply. The duty cycle is
each chip's `duty_pct` in `hbm_off.json` and `hbm_on.json`. The end-to-end rate and the first reply
come from the last and first progress lines in `capture_on.out`.

The top-level weights, 65,248,893,184 bytes, came down in 290.5 s at 224.7 MB/s into `/dev/shm`
before run 1, and run 2 found them cached. On 2026-09-24 the same fetch read 299.0 MB/s.

### Reproduce

On a `v5p-8`, from the repo root. `bootstrap_tpu_vm.sh --model gpt-oss` builds `sglang-jax` in
`~/w` with the capture and steering patches, then the model, which carries its own hook. Run it
again and it keeps that tree, so the whole block runs twice on one VM.

```bash
bash scripts/bootstrap_tpu_vm.sh --model gpt-oss
source ~/.tpu_env
python3 -c "import jax; print(jax.device_count())"   # 4 on a v5p-8
TP=4 MEM_FRAC=0.8 bash scripts/measure_model.sh openai/gpt-oss-120b 18 ~/results/gpt-oss-120b
```

Run the last command inside `tmux`. Every `RESULT` line lands in
`~/results/gpt-oss-120b/results.txt`. A rerun moves the last run's files to `run-<time>/` in the
same directory and its shard to `/dev/shm/caps-gpt-oss-120b-<time>`. The THP,
unauthenticated-request and `libtpu metrics unavailable` messages are expected, and
[the capture guide](../docs/activation-capture.md#setting-up-a-tpu-vm) says why.

Each engine start also prints ``[transformers] `torch_dtype` is deprecated! Use `dtype` instead!``
and `Loading MoE Weights: 0it` twice. The first comes from `transformers`. The second means the
shared loader's expert pass found nothing to load, because the experts decode from MXFP4 in a pass
of their own ([Mixed-precision checkpoint](#mixed-precision-checkpoint)). Every recorded run exited
0 after both.

On a VM that served another model, bootstrap moves the old tree aside and builds this one. That
tree can't take this patch: the Nemotron 3 and Kimi K3 patches edit the same `layers/moe.py`. The
old model's weights still fill `/dev/shm`, and bootstrap lists them. Delete them before the
fetch, for example `rm -rf /dev/shm/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16`.

## What the patch adds

Five files.

`models/gpt_oss.py` is new. It holds the attention, router, MoE block, decoder layer, backbone and
`GptOssForCausalLM`, plus the MXFP4 decoder and the weight loader. Under `--load-format dummy` it
fills the experts on `EPMoE`'s mesh before the loader's dummy pass, because that pass places every
expert stack it fills whole on every chip.

`layers/moe.py` gains two things `EPMoE` needed for this model. `use_expert_bias` creates a
per-expert bias on each of the three GEMMs, in the `[experts, 1, n]` layout the `gmm` kernel
already accepted but no caller ever filled. `activation="swiglu_oai"` selects the gpt-oss gate.

`kernels/gmm/megablox_gmm_backend.py` adds those biases after the activation rescale on `gmm`'s v1
path, the path that runs off TPU. The v1 kernel adds a bias inside, and under a config such as
`int8_w8a8.yaml` the rescale after it would scale each row's bias by that row's activation scale.

`eplb/expert_location.py` gives each EPLB dispatch gather the sharding of the expert ids it reads.
On the engine's Explicit mesh the ids arrive sharded over the batch, and the gather can't infer
where its output lives, so `--ep-dispatch-algorithm` would raise at the first forward.

`layers/embeddings.py` gains `truncate` on the YaRN correction range, described below.

### Attention sinks

Every query head carries one learned logit that joins the softmax and holds no value. A head can
send probability mass there instead of spending all of it on the sequence. The ragged paged
attention kernel takes it as an `attention_sink` argument, `RadixAttention` forwards keywords to
the backend, so the model passes `self.sinks` straight through.

The checkpoint stores the sinks in BF16. The parameter is float32, because that's what the kernel
reads.

### Clamped SwiGLU

The expert gate clamps both halves, scales the sigmoid by alpha, and adds one to the linear half:

```
gate = min(gate, limit)
up   = clip(up, -limit, limit)
out  = (up + 1) * gate * sigmoid(alpha * gate)
```

with `alpha = 1.702` and `limit = config.swiglu_limit`, 7.0 in both released configs. The clamp
and the `+ 1` both change the result, so neither is optional.

### YaRN with the correction range untruncated

YaRN blends an extrapolated and an interpolated inverse frequency across a linear ramp. The ramp
bounds come out fractional, and most checkpoints round them to whole dimensions. gpt-oss sets
`"truncate": false` and keeps the fractional bounds, which moves the ramp and changes every
inverse frequency inside it.

At head_dim 64, base 150000 and a 4,096 pretraining window, truncating moves the bounds from
(8.093, 17.398) to (8, 18). The two settings then disagree by 2.2e-04 per position on the worst
frequency, which is 28.9 radians at the far end of the 131,072-token context, 4.6 full rotations.

### Mixed-precision checkpoint

The two expert GEMMs ship MXFP4. Each one is a pair of tensors:

| Tensor | Shape | Holds |
|---|---|---|
| `*_blocks` | `[experts, out, in / 32, 16]` uint8 | two FP4 E2M1 codes per byte, low nibble first |
| `*_scales` | `[experts, out, in / 32]` uint8 | one E8M0 exponent per 32 elements, value `2 ** (s - 127)` |

Everything else in the checkpoint is BF16, so the loader runs the ordinary mapping table for
attention, router, norms, embeddings and the LM head, then decodes the experts in a second pass.

`gate_up_proj` interleaves the two halves of the gate along its output axis. Even columns are the
gate, odd columns are the linear half. `EPMoE` keeps them as separate `wi_0` and `wi_1`, so the
deinterleave happens at load. Both expert GEMMs also transpose from the checkpoint's
`[experts, out, in]` to the `[experts, in, out]` the kernel wants.

The experts land in BF16. `--quantization-config-path` quantizes them again after the decode; the
checkpoint's own `mxfp4` dict stays out of `EPMoE`. Decoding at load turns 60.8 GiB on disk into
217.6 GiB in HBM. Each of the 18 full-attention layers then holds 4 KiB per token, 8 KV heads
padded to 128 wide, and each of the 18 sliding layers holds the same over a 128-token window, so a
131,072-token context takes 9 GiB of full-layer cache.

#### What the decode costs the host

Every process decodes every layer, because with `--ep-size 1` each chip carries all 128 experts
and only a slice of the intermediate axis. One 120b `gate_up_proj` layer is 0.99 GiB packed and
3.96 GiB decoded. `dequantize_mxfp4` runs the decode in 32 MiB passes across 8 threads and writes
straight into the serving dtype, so it never holds a whole-tensor float32 copy.

`_put` hands each decoded tensor to `jax.make_array_from_callback`, which cuts the shards on the
host. `jnp.asarray` would put the whole 1.98 GiB tensor on one chip first and leave the split to
a later reshard.

## Test

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/models/test_gpt_oss_model.py
```

CPU only, 8 simulated devices, float32 end to end, against HuggingFace
`transformers.GptOssForCausalLM` on a four-layer random config. The test builds the tree
`bootstrap_tpu_vm.sh` builds and applies the patch on top. The model loads through `ModelConfig`
and `JAXModelLoader` from a checkpoint whose `config.json` carries the published `mxfp4`
quantization dict. Every per-layer residual stream matches to 4.4e-06 relative or better, the
logits to 1.3e-05, correlation 1.000000000 throughout. The gate is 2e-4 relative, and the six
negative controls each have to fail it, the weakest at 6.5e-03. The run also drives the LM head
through the shipped `LogitsProcessor` on both the tied and the untied branch, both capture
setters, a load with two EPLB redundant experts whose forward reaches them through the dispatch
map, `int8.yaml`, which has to reach the routed experts while the `mxfp4` dict stays out of them,
`int8_w8a8.yaml`, whose expert biases have to add whole, and a dummy-weight load at
`--tp-size 8` that never holds a whole expert stack on one device. See
[`upstream/models/README.md`](../upstream/models/README.md) for the full table.
