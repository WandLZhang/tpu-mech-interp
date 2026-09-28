# Gemma 4 26B-A4B

`google/gemma-4-26B-A4B-it`

**Status: measured on TPU**, `v5litepod-8`, 2026-09-25, with the layer filter.
[Measured](#measured) has two runs of this page's [Reproduce](#reproduce) command and one of the
README's Run-it chain, all on one Spot slice in us-south1-a.

| | |
|---|---|
| Parameters | 25.8B total / 3.8B active, 25.2B in the language model |
| Layers | 30: 25 sliding window, 5 full attention |
| Hidden dim | 2,816 |
| Attention | GQA, 16 query heads. Sliding layers 8 KV heads at head dim 256, full layers 2 KV heads at 512 |
| MoE | 128 experts top-8, plus 1 shared |
| Context | 262,144 |
| Weights | 48.1 GiB BF16 |

The checkpoint carries a vision encoder of about 550M parameters, which is why its 25.8B runs
above the model card's 25.2B for the language model.

## Attention shapes

`layer_types` puts full attention at layers 5, 11, 17, 23 and 29, and a sliding window everywhere
else.

| | Layers | Head dim | KV heads | Window | Config keys |
|---|---|---|---|---|---|
| Sliding | 25 | 256 | 8 | 1,024 | `head_dim`, `num_key_value_heads` |
| Full | 5 | 512 | 2 | none | `global_head_dim`, `num_global_key_value_heads` |

The full layers reuse K as V: `attention_k_eq_v` is true.

## Serve

Engine: `sglang-jax` as shipped, plus the capture patch that `bootstrap_tpu_vm.sh` applies. Slice:
`v5litepod-8`, 8 chips, one host, `tp_size=8`. The README's
[Run it](../README.md#run-it) chain serves, captures, trains and steers this model, and the
Reproduce command below measures it. Every number on this page comes from those commands on
2026-09-25, with the layer filter, unless it names another run.

Confirm the chip count first, after `source ~/.tpu_env`: a `v5litepod-8` reports 8.

```bash
python3 -c "import jax; print(jax.device_count())"
```

A `v5p-64` is 32 chips on 8 hosts. Gemma 4 26B-A4B hasn't run on one; six other models have,
through [Across hosts](../README.md#across-hosts). The grouped matmul flattens (batch, sequence).
Check that sequence sharding survives it by diffing the XLA buffer assignment between `ctx=1` and
`ctx=2`.

## Capture

`gemma4.py` has the `layers_to_capture` hook, so this needs only
[`upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch). The capture script hands
`--layers` to the engine as `--return-hidden-states-layers`, so `--layers 15` moves 2,816 × 2 =
5,632 bytes per token to the host at BF16. The captures below pass that flag.

Against a HuggingFace float32 forward, every layer's median per-token error stays within 1.23
times HuggingFace's own BF16 error, on one prompt alone and on a batch that splits a prompt across
prefill passes. [The capture guide](../docs/activation-capture.md#correctness) has the table.

## Measured

The Reproduce command below ran twice on 2026-09-25, back to back on one `v5litepod-8` Spot slice
in us-south1-a, with the layer filter. 8 chips, `tp_size=8`, BF16 engine,
`mem_fraction_static=0.6`. 400 wikitext passages re-encoded to 440 tokens, 441 with the BOS, 8 to
a call. Capture keeps slot 15 of 30 and writes it to `/dev/shm` as float32.

| | Run 1 | Run 2 |
|---|---|---|
| Capture off, tokens/s | 13,158.8 | 13,185.7 |
| Capture off, window | batches 3 to 32, first two discarded, 105,840 tokens in 8.04 s | the same batches, 105,840 tokens in 8.03 s |
| Capture on, tokens/s | 9,535.1 | 9,553.6 |
| Capture on, window | 18.7 s to 23.88 s, 49,392 tokens | 18.98 s to 24.15 s, 49,392 tokens |
| Peak HBM per chip, capture off | 10.34 of 15.75 GiB, 24 samples over 77.3 s | 10.14 of 15.75 GiB, 23 samples over 75.6 s |
| Peak HBM per chip, capture on | 10.37 of 15.75 GiB, 32 samples over 100.6 s | 10.37 of 15.75 GiB, 32 samples over 100.5 s |
| Capture cost | 1.38x the throughput, 0.03 GiB of HBM per chip | 1.38x, 0.23 GiB |
| Wire in the window | 53.7 MB/s | 53.8 MB/s |
| End to end, capture on | 6,652.0 tokens/s over 26.52 s | 6,588.3 tokens/s over 26.77 s |
| Peak duty cycle, capture off / on | 94.6% / 85.3% | 61.3% / 85.7% |

The wire carries 5,632 bytes per token, the one kept slot at BF16, and disk takes 11,264, the same
slot at float32. Each run wrote 176,400 tokens into one 1.851 GiB shard.

Before the layer filter, on 2026-09-24 and 2026-09-25, the engine copied all 30 slots, 168,960
bytes per token, and the same command read 2,088.1 and 2,201.9 tokens/s with capture on, 6.31x and
5.98x, at 10.69 and 10.50 GiB of peak HBM.

The capture timer starts after the engine loads and precompiles. The window runs from the third
progress line to the last one before the shard closes, the rule every model page uses. On both
runs it spans one progress step, 5.18 and 5.17 s, the shortest window the rule allows. HBM is the
peak of what [`scripts/peak_hbm.py`](../scripts/peak_hbm.py) sampled every 2 s, and all 8 chips
read the same. Duty cycle is the busiest chip's peak, and the other 7 sit within 0.2 points of it.
Capture off runs at full load for about 8 s, so the sampler can miss its peak. Run 2's capture-off
readings, 10.14 GiB and 61.3%, sit below run 1's 10.34 GiB and 94.6%.

The weights, 51,612,009,916 bytes, came down in 3,251.3 s at 15.9 MB/s in the Run-it chain's
fetch. The 49.9 GB first shard came down one HTTP connection. The rate varies. The same fetch read
52.4 MB/s on 2026-09-24 and 105.9 MB/s on 2026-09-25 before the layer filter. A killed fetch
starts over from zero.

### The Run-it chain

The same slice on 2026-09-25, with the layer filter, the README's commands as written:

| Step | Result |
|---|---|
| 1. Capture check, one 14-token prompt | 30 of 30 layers within 1.23x the BF16 error, shift control 8.9x |
| 2. Capture | 2,205,000 tokens of slot 15 in 12 shards, 23.131 GiB, in 277.57 s: 7,944.1 tokens/s end to end, 9,536.0 steady from 57.01 s to 274.55 s |
| 2. Capture check, a batch of four | every prompt passes, the 1,323-token one included. Worst ratio to BF16 1.00, 1.02, 1.00 and 1.00, shift control 6.3x, 5.9x, 6.6x and 7.6x |
| 3. SAE, d_sae 45,056, k=64, batch 512, 4,000 steps with 500 warmup | fvu 0.465 at step 3,999, 2,023 live and 43,033 dead latents over 64 calibration batches, MFU 0.3%, 336.4 s |
| 4. Feature | latent 32177, which fires on 47.7% of tokens |
| 4. `compare.py` exit code | 0 |

Step 4 steers the picked latent at a fraction of the stream's RMS norm, 107.49 at slot 15, beside
a random unit direction at the same length. The served model then judges each changed reply
coherent or broken, and [Run it](../README.md#run-it) gives the rule. Each cell counts prompts,
of 4:

| Fraction of the norm | Feature changed | Feature coherent | Random changed | Random coherent |
|---|---|---|---|---|
| 0.1 | 3 | 3 | 4 | 0 |
| 0.5 | 4 | 0 | 4 | 0 |
| 1.0 | 4 | 0 | 4 | 0 |

The judge labeled both of its controls right, the known-coherent reply coherent and the
known-broken one broken. The feature has more coherent replies than the random direction at 0.1,
so the run exits 0. At 0.1 the feature rewords three replies and keeps them coherent, while the
random direction breaks all four into repeated tokens. At 0.5 and 1.0 both break every reply: the
feature into runs of `**`, dots and pieces of words such as `much` and `ness`, the random direction
into fragments such as `disk` and `]++;` and tokens from other scripts.

Two earlier runs of the chain, on 2026-09-24 and 2026-09-25 before the layer filter, ran under the
old exit rule, which counted changed prompts. The random direction changed at least as many prompts
as the feature at every fraction, so both exited 1. They picked latents 1859 and 7987, second and
fifth in this run's ranking.

### Reproduce

On a `v5litepod-8`, from the repo root. `sglang-jax` serves Gemma 4 as is, so the capture patch
that `bootstrap_tpu_vm.sh` applies is all it needs. `measure_model.sh` measures capture off and on
with peak HBM beside each.

```bash
bash scripts/bootstrap_tpu_vm.sh
source ~/.tpu_env
python3 -c "import jax; print(jax.device_count())"   # 8 on a v5litepod-8
bash scripts/measure_model.sh google/gemma-4-26B-A4B-it 15 ~/results/gemma4-26b-a4b
```

Run the last command inside `tmux`. Every `RESULT` line lands in
`~/results/gemma4-26b-a4b/results.txt`. A rerun moves the last run's files to `run-<time>/` in
the same directory and its shard to `/dev/shm/caps-gemma4-26b-a4b-<time>`. The THP,
unauthenticated-request, `use_fast`, `Loading MoE Weights: 0it` and `libtpu metrics unavailable`
messages are expected, and [Run it](../README.md#run-it) says why.
