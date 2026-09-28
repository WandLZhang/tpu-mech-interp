# Gemma 4 31B

`google/gemma-4-31B-it`

**Status: measured on TPU**, `v5litepod-8`, 2026-09-25, with the layer filter.
[Measured](#measured) has two runs of this page's [Reproduce](#reproduce) command, on one Spot
slice in us-south1-a.

| | |
|---|---|
| Parameters | 31.3B dense |
| Layers | 60 |
| Hidden dim | 5,376 |
| Attention | GQA, 32 query heads, 5:1 sliding window to full |
| Weights | 58.3 GiB BF16 |

## Attention shapes

The two layer types have different head geometry, so one row can't describe both. `layer_types`
puts `full_attention` at indices 5, 11, 17, 23, 29, 35, 41, 47, 53 and 59, and
`sliding_attention` everywhere else.

| | Layers | Head dim | KV heads | Window | Config keys |
|---|---|---|---|---|---|
| Sliding | 50 | 256 | 16 | 1,024 | `head_dim`, `num_key_value_heads` |
| Full | 10 | 512 | 4 | none | `global_head_dim`, `num_global_key_value_heads` |

`attention_k_eq_v: true` applies to the full-attention layers, which reuse K as V and ship no
`v_proj`. The checkpoint index agrees: 50 of the 60 language-tower layers list
`self_attn.v_proj.weight`, and the 10 missing ones are the full-attention indices above.

Size a KV cache from the table, not from the sliding row alone. Per token at BF16 a sliding layer
holds 2 × 16 × 256 × 2 = 16 KiB and a full layer holds 2 × 4 × 512 × 2 = 8 KiB, so the fixed part
is 50 × 16 KiB across the 1,024-token window and the growing part is 10 × 8 KiB = 80 KiB per
token. Reading 16 KV heads at head dim 256 for every layer overstates the full layers by 4× on
head count and understates them by 2× on head dim. That puts a full layer at 16 KiB per token, 2×
the true 8 KiB.

`sglang-jax` reads this correctly. `Gemma4Config.__init__` moves the sliding values to
`swa_head_dim` and `swa_num_key_value_heads` and promotes the global values into `head_dim` and
`num_key_value_heads`, so the non-sliding branch of `Gemma4Attention.__init__` sizes the full
layers at 512 and 4 and the sliding branch reads the `swa_*` pair.

## Serve

Engine: `sglang-jax` as shipped, plus the capture patch that `bootstrap_tpu_vm.sh` applies. Slice:
`v5litepod-8`, 8 chips, one host, `tp_size=8`, measured below. The measured runs drive the engine
from Python through [`scripts/measure_model.sh`](../scripts/measure_model.sh). As a server, the
engine settings those runs pass read as below. Untested: no run here has started this server line.

```bash
python3 -m sgl_jax.launch_server \
  --model-path SNAPSHOT_DIR \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --mem-fraction-static 0.6 \
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

`SNAPSHOT_DIR` is the path `scripts/fetch_weights.py` prints. Confirm the chip count first, after
`source ~/.tpu_env`: a `v5litepod-8` reports 8.

```bash
python3 -c "import jax; print(jax.device_count())"
```

A `v5p-64` is 32 chips on 8 hosts. Gemma 4 31B hasn't run on one; six other models have, through
[Across hosts](../README.md#across-hosts).

## Capture

`gemma4.py` has the `layers_to_capture` hook, so this needs only
`upstream/sglang-jax-877.patch`. 60 layers at 5,376 dim is 630 KiB per token of host traffic.
That's every slot. `capture_activations.py` hands `--layers` to the engine as
`--return-hidden-states-layers`, and one kept slot moves 10.5 KiB per token. Both runs in
[Measured](#measured) pass that flag.

A run on `Qwen/Qwen3-0.6B`, which takes the same engine path at a fraction of the size, checked
the capture. Slice `v5litepod-4`, us-south1-a, 4 chips, `tp_size=4`, BF16 weights. Reference is
HuggingFace Transformers on CPU in float32, same prompt, compared layer by layer.

| | |
|---|---|
| Layers captured | 28, matching the reference |
| Captured shape | `[seq_len, num_layers, hidden_dim]` |
| Worst Pearson | 0.99998586 |
| Worst max abs | 3.2195e+01 |
| Layer 0 | 0.0000e+00, bit-identical |
| Negative control | layers shifted by one: 6.5357e+03, detected |

A residual stream carries large magnitudes, so the BF16 capture sits up to 32 from the float32
reference in absolute terms while the Pearson correlation stays at 0.99998586. Shifting each
captured layer against the wrong reference layer gives a max abs of 6.5357e+03, 203 times the
true pairing's.

## Measured

The Reproduce command below ran twice on 2026-09-25, on one `v5litepod-8` Spot slice in
us-south1-a, with the layer filter. 8 chips, `tp_size=8`, BF16 engine. 400 wikitext passages
re-encoded to 440 tokens, 441 with the BOS, 8 to a batch. Capture keeps slot 30 of 60 and writes
it to `/dev/shm` as float32.

| | Run 1 | Run 2 |
|---|---|---|
| Capture off, tokens/s | 6,942.0 | 6,937.4 |
| Capture off, window | batches 3 to 32, first two discarded, 105,840 tokens in 15.25 s | the same batches, 105,840 tokens in 15.26 s |
| Capture on, tokens/s | 5,469.8 | 5,477.7 |
| Capture on, window | 52.47 s to 73.11 s, 112,896 tokens | 16.76 s to 37.37 s, 112,896 tokens |
| Peak HBM per chip, capture off | 10.39 of 15.75 GiB, 71 samples over 171.1 s | 10.39 of 15.75 GiB, 28 samples over 85.1 s |
| Peak HBM per chip, capture on | 10.44 of 15.75 GiB, 57 samples over 147.9 s | 10.44 of 15.75 GiB, 37 samples over 108.6 s |
| Capture cost | 1.27x the throughput, 0.05 GiB of HBM per chip | 1.27x, 0.05 GiB |
| Wire in the window | 58.8 MB/s | 58.9 MB/s |
| End to end, capture on | 2,388.9 tokens/s over 73.84 s | 4,630.8 tokens/s over 38.09 s |
| Peak duty cycle, capture off / on | 97.5% / 78.7% | 97.7% / 78.8% |

The wire carries 10,752 bytes per token, the one kept slot at BF16, and disk takes 21,504, the
same slot at float32. Each run wrote 176,400 tokens into one 3.533 GiB shard, and the shard closed
0.73 and 0.72 s after the last progress line.

Before the layer filter, on 2026-09-24 and 2026-09-25, the engine copied all 60 slots, 645,120
bytes per token, and the same command read 676.6 and 652.4 tokens/s with capture on, 10.26x and
10.64x, at 11.66 GiB of peak HBM.

HBM is the peak of what [`scripts/peak_hbm.py`](../scripts/peak_hbm.py) sampled, every 2 s plus
the `tpu-info` call, across the load and the run. All 8 chips read the same. Duty cycle is the
busiest chip's peak, and the other 7 sit within 0.2 points of it.

Run 1's first capture reply came 42.13 s after the timer started, and run 2's after 6.52 s, which
sets the two end-to-end rates apart. Run 1 compiled in that first call, and run 2 found the result
in the compile cache. Both runs ran while the Gemma 4 26B-A4B weights downloaded on the same VM.

The weights, 62,546,338,248 bytes, came down with Xet off in 937.9 s at 66.7 MB/s to `/dev/shm`,
beside two other fetches, before the Reproduce command. Both runs found them cached. Keep the
weights and the shard in `/dev/shm`, and
[the disk note](../docs/activation-capture.md#setting-up-a-tpu-vm) says why.

### Reproduce

On a `v5litepod-8`, from the repo root. `sglang-jax` serves Gemma 4 as is, so the capture patch
that `bootstrap_tpu_vm.sh` applies is all it needs.

```bash
bash scripts/bootstrap_tpu_vm.sh
source ~/.tpu_env
python3 -c "import jax; print(jax.device_count())"   # 8 on a v5litepod-8
bash scripts/measure_model.sh google/gemma-4-31B-it 30 ~/results/gemma4-31b
```

Run the last command inside `tmux`. Every `RESULT` line lands in
`~/results/gemma4-31b/results.txt`. A rerun moves the last run's files to `run-<time>/` in the
same directory and its shard to `/dev/shm/caps-gemma4-31b-<time>`. The THP,
unauthenticated-request, `use_fast`, `Loading MoE Weights: 0it` and `libtpu metrics unavailable`
messages are expected, and [Run it](../README.md#run-it) says why.
