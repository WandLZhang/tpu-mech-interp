# Nemotron 3 Super 120B-A12B

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`

**Status: measured on TPU**, `v5p-8`, 2026-09-25, with the layer filter. [Measured](#measured)
has two runs of this page's [Reproduce](#reproduce) block, on one Spot slice in europe-west4-b.

| | |
|---|---|
| Parameters | 123.6B total / 12B active, 120.7B in the served stack |
| Layers | 88: 40 Mamba-2 + 40 LatentMoE + 8 attention, plus a 2-block MTP head |
| Hidden dim | 4,096 |
| Attention | GQA, 32 query / 2 KV heads, head dim 128, no position encoding |
| Mamba-2 | 128 heads, head dim 64, state 128, 8 groups, conv kernel 4, chunk 128 |
| MoE | 512 routed experts top-22, 1 shared, latent dim 1,024 |
| Context | 262,144 |
| Weights | 224.8 GiB BF16 served, 230.2 GiB on disk |

The checkpoint holds 230.2 GiB, which `model.safetensors.index.json` reports as 247,222,108,160
bytes. The MTP head is 5.4 GiB of that and the engine doesn't build it, so 224.8 GiB lands in HBM.

## Layer plan

One layer is one mixer or one feed-forward, not both. The types interleave and the cycle length
changes part way down the stack, so a single `lax.scan` over 88 layers won't build it.

[`scan/nemotron3_layers.py`](../scan/nemotron3_layers.py) resolves the plan from a config dict.
Super and Nano spell the field `hybrid_override_pattern`, a string over `M`, `E`, `*` and `-`.
Ultra spells it `layers_block_type`, a list of `"mamba"`, `"moe"`, `"attention"` and `"mlp"`.
`block_types` reads either and returns the list. `num_hidden_layers` is derived from that length,
so a config where the two disagree is rejected. From `scan/`:

```python
from nemotron3_layers import NEMOTRON_3_SUPER_CONFIG, block_types, cache_plan, scan_groups, unit_runs

scan_groups(NEMOTRON_3_SUPER_CONFIG)               # 88 runs, every one a single layer
unit_runs(NEMOTRON_3_SUPER_CONFIG)                 # 17 runs, one lax.scan each
cache_plan(block_types(NEMOTRON_3_SUPER_CONFIG))   # 40 recurrent, 8 paged, 40 stateless
```

`scan_groups` returns contiguous runs of one block type. On Super every run holds one layer,
because no two adjacent layers share a type. Grouping by type alone never fills a scan.

`unit_runs` returns the grouping that does. Every feed-forward layer closes a unit, so the Super
stack cuts into `(mamba, moe)` pairs and `(mamba, attention, moe)` triples with nothing left over.
Super holds 40 units in two shapes, and adjacent identical units fold into 17 runs. Each run is
one `lax.scan` over stacked weights.

| | Super | Ultra 550B-A55B | Nano 4B |
|---|---|---|---|
| Layers | 88 | 108 | 42 |
| Mamba-2 / feed-forward / attention | 40 / 40 MoE / 8 | 48 / 48 MoE / 12 | 21 / 17 dense / 4 |
| Units | 40 | 48 | 17 |
| Scans after folding | 17 | 25 | 12 |
| Attention at | 7, 16, 25, 36, 47, 58, 69, 78 | 7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98 | 12, 17, 24, 32 |

Nano runs dense MLPs and mixer runs of two and three Mamba-2 layers, so it cuts into five unit
shapes and its `scan_groups` runs reach three layers wide. The same parser handles it.

Both MoE sizes carry one multi-token prediction head, two blocks of `(attention, moe)`. It's a
draft model for speculative decoding, it sits under the `mtp.` prefix in the checkpoint, and the
engine leaves it out. Read it with `mtp_block_types` and read main plus head with
`block_types_with_mtp`, which is what the config declares rather than what a runner builds.

### State

`nemotron3_layers.cache_plan` splits the layers the two cache managers own. Hand it the resolved
block type list of the stack the engine builds. Hand it the config dict instead and it plans main
plus MTP head, which is a larger stack than the one that runs. The served engine reads
`NemotronHConfig.linear_layer_ids` and `full_attention_layer_ids`, both of which cover the main
stack alone.

The 40 Mamba-2 layers hold two slots each, and neither one grows with the sequence.

| Slot | Shape | Per layer | 40 layers |
|---|---|---|---|
| SSM state | `[128 heads, 64 head dim, 128 state]` | 4 MiB | 160 MiB |
| conv state | `[10,240 channels, 3 tokens]` | 120 KiB | 4.69 MiB |

`conv_dim` is `mamba_num_heads * mamba_head_dim + 2 * n_groups * ssm_state_size`, so the conv
carries `B` and `C` alongside the SSM input. The conv slot holds `conv_kernel - 1` tokens: the
current token arrives with the request, and the window needs the three before it.
`Mamba2Config.state_bytes_per_layer` adds the two slots up, and `conv_state_bytes` returns the
second one on its own. A cache manager that allocates the SSM slot alone drops the conv slot, and
decode then reads a window the previous step never filled.

`mamba_ssm_cache_dtype` stays float32 while the weights ship BF16, so both slots cost twice what
the weight dtype suggests.

`RecurrentStatePool` allocated the SSM slot square, `head_dim` by `head_dim`, which is what KDA and
GDN need. Mamba-2 runs 64 by 128, so a square slot holds half the state. The model patch gives the
last axis its own field, `head_k_dim`, which leaves both existing callers alone.

The attention layers hold KV that grows with the sequence. 2 KV heads at head dim 128 is 1 KiB per
token per layer in BF16. The engine builds the 8 attention blocks of the main stack and no more, so
that's 8 KiB per token, or 2.0 GiB per sequence at the full 262,144 context.

[`scan/mamba2.py`](../scan/mamba2.py) turns a chunk of tokens into the `(A, B)` pair that
[`scan/affine_scan.py`](../scan/affine_scan.py) carries across a sequence shard. The layer plan
says which 40 layers run it.

## Serve

Engine: `sglang-jax` with
[`upstream/models/nemotron3-model.patch`](../upstream/models/nemotron3-model.patch), which adds
`python/sgl_jax/srt/models/nemotron_h.py`, `configs/nemotron_h.py` for the layer plan, and
`layers/attention/mamba/` for the chunked scan and the backend that owns the recurrent state. The
same patch serves [Ultra](nemotron3-ultra.md). Slice: `v5p-8`, 4 chips, one host, `tp_size=4`,
measured below.

The 8 attention layers carry no rotary table. `config.json` ships `rope_theta` and
`partial_rotary_factor`, the reference config class declares neither and the reference model reads
neither, and the port does the same. Order reaches attention through the Mamba-2 layers below it.

This model has run on a `v5p-8`, not yet on v6e. The Mamba-2 mixer in
[`scan/mamba2.py`](../scan/mamba2.py) is plain XLA, a cumsum, three einsums, a `lax.scan` and an
exponential, with no Pallas kernel and no chip generation check.

The measured runs drive the engine from Python through
[`scripts/measure_model.sh`](../scripts/measure_model.sh), with the settings in
[Reproduce](#reproduce). As a server, the engine settings those runs pass read as below.
Untested: no run here has started this server line.

```bash
python3 -m sgl_jax.launch_server \
  --model-path SNAPSHOT_DIR \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 4 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 1024 \
  --disable-radix-cache \
  --attention-backend fa \
  --page-size 64 \
  --skip-server-warmup \
  --max-running-requests 8 \
  --precompile-bs-paddings 8 \
  --precompile-token-paddings 1024 \
  --enable-return-hidden-states
```

A hybrid recurrent model needs `--disable-radix-cache` or `--enable-unified-radix-tree` to start,
and the measured runs disable the radix cache. Prefix caching at a page size above 1 also takes
`--enable-recurrent-extra-buffer`. The model test covers that path on CPU, and no chip has run it.

The model refuses `--ep-dispatch-algorithm`, `--init-expert-location` and
`--ep-num-redundant-experts` before it builds. It loads the routed experts in checkpoint order and
builds no expert-location metadata, so those flags would change nothing. `--ep-size` still sets
the expert axis.

`SNAPSHOT_DIR` is the path `scripts/fetch_weights.py` prints. Start the server after the first
three lines of [Reproduce](#reproduce), which build the tree, load `~/.tpu_env` and confirm the
chip count: a `v5p-8` reports 4.

### Larger slices, not yet run

Budget the slice per chip, not in aggregate. Each chip stores one of the 2 KV heads. A sequence pays
half its KV on every chip.

| | `v5p-64` | `v6e-16` |
|---|---|---|
| Chips | 32 | 16 |
| HBM per chip | 95 GiB | 32 GiB |
| Weights per chip | 7.0 GiB | 14.0 GiB |
| Free per chip | 88.0 GiB | 18.0 GiB |
| Full-context sequences | 87 | 17 |

A full-context sequence costs 1.0 GiB of KV on every chip. The recurrent state shards with the
Mamba-2 heads: 160 MiB of SSM state plus 4.69 MiB of conv state over 32 chips is 5.15 MiB per chip
per sequence, and 10.3 MiB over 16. Capture buffers come out of the same free space.

A `v5p-64` is 32 chips on 8 hosts, and nothing here has run across hosts. There the launch line
takes `--tp-size 32`.

Three counts don't divide over 32 chips. Each chip stores one of the 2 KV heads. The 8 Mamba-2
groups shard 8 ways and replicate four times, so `B` and `C` ride into the state path replicated
and each device slices the groups its heads read. The gated norm runs those same 8 groups, and its
RMS reduction covers a whole 1,024 channel group, so a 32-way tensor axis splits one group over
four shards; `GroupRMSNorm` sums a split group with one all-reduce of a `[tokens, 8]` array. The 32
query heads, the 128 Mamba-2 heads, the 10,240 conv channels, the 1,024 latent and the 512 routed
experts all shard cleanly.

Top-22 of 512 is a far wider all-to-all than the top-8 the existing MoE layers assume. Give the
expert axis its own mesh axis with `--ep-size`. The dispatch already runs in the latent:
`fc1_latent_proj` drops the token from 4,096 to 1,024 after the router reads it, so the wire
carries 22 copies of a 2 KiB token rather than 22 copies of an 8 KiB one.
[`nemotron3-ultra.md`](nemotron3-ultra.md#what-latentmoe-asks-of-the-mesh) works the numbers.

## Capture

Three patches and the flag.

1. [`upstream/models/nemotron3-model.patch`](../upstream/models/nemotron3-model.patch) for the
   model.
2. [`upstream/models/nemotron3-capture-hook.patch`](../upstream/models/nemotron3-capture-hook.patch)
   for the `layers_to_capture` hook on top of it.
3. [`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) for
   `--enable-return-hidden-states` and the output reshape.

The hook is the ordinary None-residual append, because the stack is a flat Python loop over
blocks. `scan_groups` returns 88 runs of one layer each on Super, so a scan over a run takes one
step and the loop is what a scan would unroll to.

```python
if layer_id in self.layers_to_capture:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )
```

A run longer than one layer can't use that form. Nano holds Mamba-2 runs of two and three layers,
and stacking their weights into one `lax.scan` is what makes those runs worth grouping. A scan
body traces once, so an append in the body runs once no matter how many steps the scan takes: a
4-repeat run leaves two tracers in the list and `jnp.stack` on them raises
`UnexpectedTracerError`. The `layer_id in self.layers_to_capture` test fails the same way, because
the layer index inside a scan is a tracer and a membership test on a tracer raises
`TracerBoolConversionError`.

Return the per-layer hidden states as the scan's second output instead. Each body returns one
entry per layer in the unit, so a run of `R` repeats over a unit of `U` layers hands back
`[R, U, seq, hidden]`. Reshape that to `[R * U, seq, hidden]`, concatenate the runs in layer
order, and the patch's reshape to `[seq_len, num_layers, hidden_dim]` takes it from there. The
sketch below is a design: the served model runs the flat loop, and no model here scans its stack.

```python
def body(carry, weights):
    hidden, residual = carry
    per_layer = []
    for block_type in run.types:          # a Python tuple, so it unrolls at trace time
        hidden, residual = apply(block_type, weights, hidden, residual)
        per_layer.append(hidden + residual if residual is not None else hidden)
    return (hidden, residual), jnp.stack(per_layer)

carry, captured = lax.scan(body, carry, stacked_weights)   # [R, U, seq, hidden]
run_states.append(captured.reshape(-1, *captured.shape[2:]))
```

The loop over `run.types` unrolls at trace time, so its appends land on real values. Select layers
after the concatenation, where the index is a Python integer again.

88 layers at 4,096 dim is 704 KiB per token of host traffic. That's every slot.
`capture_activations.py` hands `--layers` to the engine as `--return-hidden-states-layers`, and
one kept slot moves 8 KiB per token. Both runs in [Measured](#measured) pass that flag.

`upstream/models/test_nemotron_h_model.py` applies both patches on the stack the TPU VM serves and
runs them against the HuggingFace implementation on CPU. Every layer, the final hidden state and
the logits match to 1.2e-07 on a single request, on two packed requests and through a decode step,
and a four-way tensor axis matches a one-way one to 6.0e-08. It also loads a checkpoint written
under the published key names, resolves both published `config.json` files through `AutoConfig`,
and runs the engine's own pools and jit through a prefix hit on the extra buffer's track slot, a
poisoned request packed among clean ones, and `int8.yaml`. Twenty-three mutants each have to fail
the check they name, and a NaN fails every check it reaches.

The served model clamps `dt` to `(0, inf)`, as Megatron-LM, vLLM and NVIDIA's CUDA kernels do.
`transformers` floors prefill `dt` at 0.001 instead. The test's random weights keep `dt` above 0.5,
so the comparison never reaches that floor. On the published weights, 212 of Super's 5,120 Mamba-2
heads have `softplus(dt_bias)` below it.

## Measured

The Reproduce block below ran twice on 2026-09-25, back to back on one `v5p-8` Spot slice in
europe-west4-b, with the layer filter. 4 chips, one host, `tp_size=4`, BF16 engine,
`mem_fraction_static=0.8`. 400 wikitext passages cut at 440 tokens, sent 8 to a call. The engine
runs at most 6 at a time: it takes the smallest of the 8 asked for, its request pool size, and a
limit it computes from the 262,144 context and a page size of 64 (`tp_worker.py`). Capture keeps
slot 44 of 88 and writes it to `/dev/shm` as float32. Run 1 filled the compile cache, and run 2
read from it.

| | Run 1 | Run 2 |
|---|---|---|
| Capture off, tokens/s | 4,360.2 | 4,360.3 |
| Capture off, window | batches 3 to 32, first two discarded, 105,600 tokens in 24.22 s | the same batches, 105,600 tokens in 24.22 s |
| Capture on, tokens/s | 3,299.4 | 3,321.7 |
| Capture on, window | 106.58 s to 143.92 s, 123,200 tokens | 23.33 s to 60.42 s, 123,200 tokens |
| Peak HBM per chip, capture off | 78.75 of 95.73 GiB, 188 samples over 403.4 s | 78.75 of 95.73 GiB, 137 samples over 300.7 s |
| Peak HBM per chip, capture on | 78.89 of 95.73 GiB, 199 samples over 428.4 s | 78.89 of 95.73 GiB, 157 samples over 346.1 s |
| Capture cost | 1.32x the throughput, 0.14 GiB of HBM per chip | 1.31x, 0.14 GiB |
| Wire in the window | 27.0 MB/s | 27.2 MB/s |
| End to end, capture on | 1,186.9 tokens/s over 148.29 s | 2,718.8 tokens/s over 64.73 s |
| Peak duty cycle, capture off / on | 100.0% / 94.3% | 100.0% / 94.2% |

The wire carries 8,192 bytes per token, the one kept slot at BF16, and disk takes 16,384, the same
slot at float32. Each run wrote 176,000 tokens into one 2.686 GiB shard, and both shards carry the
same SHA-256, `027b52a6...`. Two runs before the filter, earlier on 2026-09-25, wrote different
SHA-256s from the same prompts, `c7aeda79...` and `dfe04ee2...`. The second of those deleted the
first one's shard, so nothing says why. `measure_model.sh` now keeps the last run's shard.

Before the layer filter, on 2026-09-24, the engine copied all 88 slots, 720,896 bytes per token,
and the same command read 514.0 and 503.8 tokens/s with capture on, 8.47x and 8.64x, at 80.28 GiB
of peak HBM.

HBM is the peak of what [`scripts/peak_hbm.py`](../scripts/peak_hbm.py) sampled every 2 s, and all
4 chips read the same. Duty cycle is the busiest chip's peak, and the other 3 sit within 0.1 points
of it. A warm cache moves the windows. Run 2's first capture line comes at 12.71 s against run 1's
95.92 s, and its capture-off sampler covers 300.7 s against 403.4 s.

No `RESULT` line carries the duty cycle, the end-to-end rate or the first capture line. The duty
cycle is each chip's `duty_pct` in `hbm_off.json` and `hbm_on.json`. The end-to-end rate and the
first capture line come from the last and first progress lines in `capture_on.out`.

The 50 safetensors, 247,227,650,480 bytes, came down in 668.3 s at 370.0 MB/s before run 1, with
the mount at `mpol=interleave`, and run 2 found them cached. On 2026-09-24 the same fetch read
363.3 MB/s, and `huggingface_hub` resumed two shards whose connections dropped inside that one
process. A fetch you kill leaves partial files that a new process fetches again from zero, and
`fetch_weights.py` deletes them before it starts. The checkpoint is more than the boot disk holds
and more than the default 221 GiB `/dev/shm` on a v5p host, so the Reproduce block grows the
mount to 340G first. The host has 440 GB of RAM.

That RAM sits on two NUMA nodes of about 220 GiB each. On one VM on 2026-09-25 the fetch put
every page on node 0 and stalled with the cache at 231,548,682,996 bytes: the kernel evicted and
reread page cache on node 0 while node 1 held 229 GB free. Moving 23 blobs, 114.9 GB, to node 1
and rerunning the fetch finished it. Two fetches on another VM the same day didn't stall, 666.8 s
on the default policy and 658.1 s with the mount at `mpol=interleave`, which split the weights
115.1 GiB to each node. To spread the pages from the start, remount with
`sudo mount -o remount,size=340G,mpol=interleave /dev/shm` before the fetch. On kernel
5.19.0-1027-gcp a later remount doesn't clear that policy. A fetch that stops growing while
`/proc/pressure/io` reads high and one node's `Shmem` line holds nearly all the weights is this
stall. The TPU image has no `numastat`, so read `grep Shmem /sys/devices/system/node/node*/meminfo`.

### Reproduce

On a `v5p-8`, from the repo root. `bootstrap_tpu_vm.sh --model nemotron3` builds `sglang-jax` in
`~/w` with the capture and steering patches, then the model and its hook. Run it again and it
keeps that tree, so the whole block runs twice on one VM.

```bash
bash scripts/bootstrap_tpu_vm.sh --model nemotron3
source ~/.tpu_env
python3 -c "import jax; print(jax.device_count())"   # 4 on a v5p-8
sudo mount -o remount,size=340G,mpol=interleave /dev/shm
TP=4 MEM_FRAC=0.8 bash scripts/measure_model.sh \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 44 ~/results/nemotron3-super
```

Run the last command inside `tmux`: the fetch alone takes about 11 minutes. Every `RESULT` line
lands in `~/results/nemotron3-super/results.txt`. A rerun moves the last run's files to
`run-<time>/` in the same directory and its shard to `/dev/shm/caps-nemotron3-super-<time>`.
The THP, unauthenticated-request and `libtpu metrics unavailable` messages are expected, and
[the capture guide](../docs/activation-capture.md#setting-up-a-tpu-vm) says why.

On a VM that served another model, bootstrap moves the old tree aside, builds this one and lists
the old model's weights. Delete them before this block's fetch, for example
`rm -rf /dev/shm/hf/hub/models--openai--gpt-oss-120b`. The old model's capture shards stay in
`/dev/shm/caps-*` as well; delete them once you've kept what you need.

To move the VM to another model afterwards, run bootstrap with that model's `--model`, or with
none for Gemma 4. It moves this tree aside and builds the other one. These weights still fill
230 GiB of `/dev/shm`, so delete them first:
`rm -rf /dev/shm/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16`.
