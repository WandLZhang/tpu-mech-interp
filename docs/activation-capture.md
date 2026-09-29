# Activation capture on TPU

Four operations, and what each one runs on.

| Operation | Support |
|---|---|
| Per-layer activation capture | `sglang-jax` with the patches in [`../upstream/`](../upstream/) |
| Sparse autoencoders | JAX on TPU, BatchTopK for training, JumpReLU for inference |
| Causal steering | [`../steering/`](../steering/), static and conditional, one program per mode |
| Steering a served model | `sglang-jax` with [`../upstream/steering-hook.patch`](../upstream/steering-hook.patch) |

## Setting up a TPU VM

[`scripts/bootstrap_tpu_vm.sh`](../scripts/bootstrap_tpu_vm.sh) does all of this. Run it on the
TPU VM; it's safe to run twice. The rest of this section is what it does and why.

A stock v5e TPU VM runs Python 3.10 with no numpy and no jax, and `sglang-jax` needs 3.12. Build a
`uv` venv and install the engine through its own `tpu` extra, which pins the jax it works with.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 ~/v312 && source ~/v312/bin/activate
git clone https://github.com/sgl-project/sglang-jax ~/w && git -C ~/w checkout eb061d8
git -C ~/w -c user.name=local -c user.email=local@localhost am < upstream/sglang-jax-877.patch
git -C ~/w apply "$PWD/upstream/steering-hook.patch"
git -C ~/w apply "$PWD/upstream/qwen3-steering-hook.patch"
uv pip install -e ~/w/python[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install "flax==0.12.9"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

`git -C` changes directory before it reads its arguments, so a patch path has to be absolute. A
fresh VM has no git identity, and `git am` makes a commit. No recorded run typed the block. The
measured runs took these steps through `bootstrap_tpu_vm.sh`, which also installs `tpu-info` and
writes `~/.tpu_env`.

| Pin | Why |
|---|---|
| `jax==0.11.1`, from the `[tpu]` extra | 0.11.2 exports `hijax.HiPrim`, flax asks for `hijax.HiPrimitive`, and every flax import dies |
| `flax==0.12.9` | 0.10.7 asks jax for `mutable_array`, which 0.11.1 doesn't have |
| `torch` and `torchvision`, CPU wheels | `transformers` builds the Gemma and Qwen processors through torchvision even for text |

Four more things cost an hour each if you meet them cold.

- **Keep weights and shards in `/dev/shm`.** The boot disk flushes at about 11 MB/s. A download
  lands in the page cache at 60 to 160 MB/s, so a 62.5 GB model leaves a backlog that takes most of
  an hour to reach the disk, and the kernel's writeback throttle holds every later read and write
  behind it. A process in `D` state at `rq_qos_wait` is this, not a deadlock:
  `echo 0 | sudo tee /sys/block/sda/queue/wbt_lat_usec` releases it. The boot disk is `sda` on v5e
  and `nvme0n1` on v5p, and `bootstrap_tpu_vm.sh` turns the throttle off on whichever it finds.
  `/dev/shm` is RAM, 189 GB on a `v5litepod-8`, and has no writeback. `bootstrap_tpu_vm.sh` puts
  the Hugging Face cache and the JAX compile cache there.
- **Turn Xet off**: `export HF_HUB_DISABLE_XET=1`. With Xet on, a large repo downloads every byte,
  then never finalizes. It leaves `<sha>.<tag>.incomplete` at the published size, writes a
  snapshot symlink to a blob it never materialized, and spins at 20 GB RSS. Plain HTTP pulled
  51.6 GB at 59.6 MB/s with no trouble.
- **Set `JAX_COMPILATION_CACHE_DIR`.** There's no cache by default, so a killed run throws away
  its XLA work.
- **Anything that builds an `Engine` must sit behind `if __name__ == "__main__":`** and live in a
  real file. The engine re-runs the main module in subprocesses, and a heredoc on stdin dies in
  `runpy` with `FileNotFoundError: <stdin>`.

One process holds the TPU at a time. After killing a run, wait for `/dev/vfio` to free and
`rm /tmp/libtpu_lockfile`, or the next start reports the chip already in use.

One VM can serve one model after another. `bootstrap_tpu_vm.sh --model NAME` builds a model from
[`../upstream/models/`](../upstream/models/) into `~/w`: `deepseek-v41`, `glm5-next`, `gpt-oss`,
`inkling`, `kimi-k3` or `nemotron3`. gpt-oss, Kimi K3 and Nemotron 3 all edit `layers/moe.py`,
DeepSeek V4.1 collides with Kimi K3 and Nemotron 3, and GLM-5.3-Flash carries DeepSeek V4.1's
runner hooks, so one tree takes one model. A rerun that asks for what `~/w` already holds keeps the
tree: the same name, or no `--model` on a tree built without one. A run that asks for another
model, or for none when `~/w` holds one, moves `~/w` aside to `~/w.stale-<time>` and builds the
tree it asks for. Each run prints which it did. On fresh `v5p-8` VMs on 2026-09-25, bootstrap took
42.1 s and 40.0 s. Into a fresh tree on a VM that had bootstrapped once, it took 22 s. The first
model's weights still fill `/dev/shm` when the second one fetches, and bootstrap lists them. Delete
the ones you're done with, for example
`rm -rf /dev/shm/hf/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16`.

A fetch you kill leaves `.incomplete` files in the cache. `huggingface_hub` resumes a dropped
connection inside one process, but a new process fetches those files from zero, so
`fetch_weights.py` deletes the partials first and says how many bytes they held.

Seven messages look alarming and aren't.

- Every JAX start prints
  `E... hugepage_text.cc:344] RAW: File offset incorrectly aligned for file-backed THP`, and every
  run here exited 0 after it.
- `huggingface_hub` warns about unauthenticated requests. None of the measured models is gated,
  so no `HF_TOKEN` is needed, and it resumes a shard whose connection drops. A gated model needs
  `HF_TOKEN` exported before the fetch.
- `huggingface_hub` can name its download bars "Downloading bytes" and "Reconstructing", Xet's
  words, with Xet off too. A fetch whose output goes to a file may show only its `Fetching N files`
  bar. `fetch_weights.py` prints an `XET off` line first when `HF_HUB_DISABLE_XET=1` holds.
- On Gemma 4, Kimi K3 and Inkling the engine prints `Loading MoE Weights: 0it`. Each reads its
  routed experts in a pass of its own, after the loader's, and that pass prints nothing. On a
  `v5p-64` Kimi K3's took up to 34 minutes and Inkling's about 2 hours.
- On a model with an image processor, `transformers` warns that `use_fast` is deprecated. The
  engine builds that processor, and a text prompt never reaches it.
- `peak_hbm.py` prints `libtpu metrics unavailable` until the engine holds the TPU, because
  libtpu serves its metrics only then. It says so once, then counts the empty rounds when the
  chips come back.
- At the end of a multi-host run every other host logs `Terminating process because the JAX
  distributed service detected fatal errors`, `Fatal Python error: Aborted` and a thread dump.
  Rank 0 shut its engine down, and the other ranks go with it. Rank 0's exit status and its RESULT
  lines tell whether the run passed.

## Capture

`sglang-jax` takes `return_hidden_states` as a request field, and `layers_to_capture` is a plain
Python list read at trace time. What the hook appends depends on how many streams the layer
carries. `gemma4` carries one, and it's the model the steering hook targets. `inkling` and
`kimi_k3` carry one too:

```python
if layer_id in self.layers_to_capture:
    aux_hidden_states.append(hidden_states)
```

`llama` and `qwen3` carry a `(hidden, residual)` pair and add them back together, with `residual`
`None` on layer 0:

```python
if layer_id in self.layers_to_capture:
    aux_hidden_states.append(
        hidden_states + residual if residual is not None else hidden_states
    )
```

Read the layer before porting the hook. On a one-stream model the fused form has no `residual`
to read, and on a two-stream model the plain form captures half the tensor.

`--enable-return-hidden-states` marks every layer and reshapes the output to
`[seq_len, num_layers, hidden_dim]`. The marking has to happen before `nnx.split()`, because
`layers_to_capture` is baked into `model_def` at split time.

`--return-hidden-states-layers 15 20` marks those slots alone, and the output is then
`[seq_len, 2, hidden_dim]` with slot 15 first. The engine concatenates and copies the two slots,
so HBM and the wire carry two slots and not every one. The list takes separate integers, the way
`--precompile-token-paddings` does, so argparse refuses `15,20` before anything loads. The hook
appends in layer order, so the list has to run in ascending order. The server refuses a list out
of order, an empty list, a slot listed twice, a negative slot, and the flag without
`--enable-return-hidden-states`. The model runner refuses a slot at or past the layer count.

With data parallelism each rank's rows sit in their own padded block, in prefill and in decode.
The patch starts each rank at its block, and above one rank it reads the batch's rows to the host
once, because JAX won't slice rows sharded over the data axis per request. The CPU test pins three
requests to two ranks at `--tp-size 2 --dp-size 2`, one of them split across prefill passes, and
holds slot 0 of every row, the embedding output, to the embedding of the token at its position.
`scripts/test_multihost_exec.py` sends the same three requests with each rank in a process of its
own, where `multihost-hidden-states.patch` gathers rank 1's rows from the other process's device.

On a chip, `check_capture.py` on Qwen3-8B at `--tp-size 8 --engine-arg dp_size=2` put five
requests in one engine call, four of 440 tokens and one of 1,322 split across passes, and the
scheduler spreads them over both ranks. Every one of the 36 layers passed on all five: worst ratio
to the BF16 floor 1.11, controls 26.7x to 29.0x (v5litepod-8, 2026-09-29).

The flag also refuses to start on a model without the hook, with `--speculative-algorithm`,
whose draft reads the capture layers the flag overwrites, with `--pd-disaggregation`, and with
`--disaggregation-mode prefill` or `decode`, where the decode server runs only the last prompt
position. A server started without the flag refuses a request for hidden states and tells the
caller why.

A request for hidden states never reuses a cached radix prefix, so its prefill returns one row
per prompt token with the radix cache on. A retraction under KV pressure leaves the rows as they
would have been without it: one row per position, prompt rows first, then one row per decode
step. So does `--enable-mixed-chunk`, which runs a request's decode steps inside another
request's prefill chunks.

### What a layer number means

The append runs at the top of the loop body, before `layer(...)`. Slot `i` therefore holds the
residual stream entering block `i`, which is block `i - 1`'s output. Slot 0 is the embedding
output. That's the HuggingFace `output_hidden_states[i]` convention, and the patch's
`assert_hidden_states_aligned` compares slot `i` against HF slot `i` on that basis.

Two consequences:

- `--layers 20` trains an SAE on block 19's output.
- A model with `N` blocks gives `N` slots, numbered 0 to `N - 1`. The output of the last block
  has no slot, at any `--layers` value. On DeepSeek V3, `--layers all` stops at block 59's output.

The steering hook adds its vector after `layer(...)` returns, so it writes the stream that
capture reports one slot later. Capture slot `k` and `--steering-layer k - 1` name the same
tensor.

Four models have the hook upstream: `gemma4`, `llama`, `qwen3`, and `qwen3_vl`, which captures
through its `QWen3Model` backbone.
[`../upstream/glm5-capture-hook.patch`](../upstream/glm5-capture-hook.patch) adds it to
`glm5_moe`, which serves GLM-5.3. [`../upstream/capture-hooks/`](../upstream/capture-hooks/)
adds it to `kimi_linear`, `qwen3_5`, `deepseek_v3` and `glm4_moe`, one per architecture family.

[`../upstream/models/`](../upstream/models/) writes the six models the engine has no file for,
and hooks each one. `gpt-oss-model.patch`, `inkling-model.patch`, `deepseek-v41-model.patch` and
`glm5-next-model.patch` carry the hook themselves. `kimi-k3-model.patch` and
`nemotron3-model.patch` take `kimi-k3-capture-hook.patch` and `nemotron3-capture-hook.patch` on
top. All six still need
[`../upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch) for the flag and the
reshape. On a model with no hook, the flag refuses to start.

### Writing shards

[`../scripts/capture_activations.py`](../scripts/capture_activations.py) drives the engine over a
prompt file and writes what [`../sae/train.py`](../sae/train.py) reads.

```bash
python3 scripts/capture_activations.py --model-path MODEL --prompts prompts.txt \
    --out caps --layers 10,20,30 --shard-bytes 2147483648 --dtype float16 --tp-size 8
```

Untested as written: no recorded run has taken this block. `MODEL` and `prompts.txt` are
placeholders, and the measured captures ran through
[`../scripts/measure_model.sh`](../scripts/measure_model.sh).

`--tp-size` defaults to 1, where `check_capture.py`, `serve_throughput.py` and `compare.py`
default to 8. Set it to the chips the host holds: 8 on a `v5litepod-8`, 4 on a `v5p-8`.

Each request's activations go to the open shard as soon as its batch returns, and get dropped.
`engine.generate` returns when the whole batch finishes, so the host holds `--batch-size`
requests at once, not one, in float32. At `--layers all`, Gemma 4 31B returns 60 slots of 5,376,
1.29 MB per token, so 8 requests of 2,048 tokens is about 21 GB in the host process. That's the
number to size the host against. A narrower `--layers` shrinks each reply to its slots, and
`--batch-size` is the other knob.

A shard closes at the first prompt boundary at or above `--shard-bytes`. Whole prompts per shard
is what lets a resume pick up where it stopped: rerunning against the same `--out` reads the
manifest, drops the shard the crash left behind, and continues from the prompt after the last
sealed one.

`--layers` takes `20`, `10,20,30`, `0-59`, `0-59:4` or `all`. One layer gives
`[tokens, d_model]` shards and several give `[tokens, len(layers), d_model]`, which are the two
shapes `activation_stream` accepts.

`--tokens` picks the rows. `prompt` keeps the prefill, `completion` keeps the decode steps, `all`
keeps both. `n` new tokens give `n - 1` completion rows, because the engine runs no forward pass
on the last token it generates, so `--tokens completion` needs `--max-new-tokens 2` or more and
refuses anything less.

`manifest.json` sits beside the shards:

| Field | Holds |
|---|---|
| `model`, `dtype`, `d_model` | what to construct the SAE against |
| `layers`, `num_model_layers`, `layer_axis` | which capture slots are on the shard axis, in what order, and how many slots the model has |
| `tokens`, `shards` | the token total, and per shard a name, a token count, a byte count and a SHA-256 |
| `prompts_done`, `prompts_sha256` | where a resume restarts, and which prompt list it belongs to |
| `tokens_kept`, `sampling_params`, `engine_dtype` | the token mode, the sampling settings and the serving dtype, which a resume has to match |

`sae/train.py --manifest caps/manifest.json --layer 20` reads it instead of globbing, so `--layer`
means the capture slot and `--d-model` comes along with it.

`--verify` re-reads the tree and exits non-zero on a mismatch. Per shard it checks the file
length, the `.npy` header's token count, the rank, the width of the layer axis against the
`layers` list, `d_model`, the dtype and the SHA-256. What it can't check is which layer a slot
came from: the bytes carry no label, so a `layers` list rewritten to another list of the same
length reads clean. A capture run ends with the same checks, the SHA-256 aside, since the writer
hashed each block as it went to disk. It then prints a `verified` line with the shard count, the
tokens, the bytes and the shape every shard reopens as.

The capture compares every reply against the `prompt_tokens` and `completion_tokens` the engine
sends beside it, and refuses a request that came back short, so a missing row becomes an error
instead of a short shard with a matching checksum.

The progress line reports wire bandwidth and disk bandwidth apart, and wire is the number to
compare against the NIC. `open_engine` starts the engine with `--return-hidden-states-layers` set
to `--layers`, so wire carries `len(layers) × d_model` elements a token. `--layers all` leaves the
flag unset, and wire carries every slot. The manifest records slot numbers either way:
`--layers 20` gets a reply one slot wide and writes `layers: [20]`. Wire counts the serving dtype,
2 bytes an element for bf16 and 4 for float32, not the float32 the patch widens to on the host.
On an engine started without the flag, the script takes the kept slots out of each chunk before
the chunks join, so its own copy holds those slots alone.

The capture prints the wire width once, on its first reply. On the CPU rig's tiny Qwen3, six
slots of 64 at bf16, `scripts/test_capture_activations.py` check 0 reads 768 bytes a token at
`--layers all` and 128 at `--layers 3`, in that line and in the progress counters.

`open_engine` pins what capture needs: the server flag, the kept slots and `chunked_prefill_size`
1024. It also turns the radix cache off. Capture doesn't need that, because a request for hidden
states never reuses a cached prefix. A model with a `linear_recurrent_config` needs it off,
because the engine won't start one with the cache on unless the unified radix tree is on too.
`open_engine` also sets `attention_backend="fa"`, `page_size=64` and one precompile bucket of
1,024 tokens. A batch longer than that prefills over several passes, and the patch keeps each
prompt's rows in order across them. `check_capture.py --prompts-file` checks that path.
`--engine-arg KEY=VALUE` is repeatable and overrides any of those defaults, which is how a model's
launch recipe reaches the engine. It refuses a key that a flag of the script sets, such as
`tp_size` or `return_hidden_states_layers`.
[`../scripts/measure_model.sh`](../scripts/measure_model.sh) passes
`--engine-arg mem_fraction_static=$MEM_FRAC`, and GLM-5.3 takes
`--engine-arg attention_backend=dsa_sparse` for its sparse indexer.

`open_engine` also sets `trust_remote_code=True`, and so do `check_capture.py`,
`serve_throughput.py` and `steering/compare.py`, which build on its defaults. `transformers` then
imports and runs the Python files a model repo's `auto_map` names for its config or tokenizer.
Kimi K3's tokenizer is such a file, and `fetch_weights.py` fetches a repo's top-level `*.py` for
it. Read those files before you serve a repo you don't trust, or pass
`--engine-arg trust_remote_code=False`. `build_corpus.py` loads the tokenizer the same way and
has no switch to turn it off.

[`../scripts/test_capture_activations.py`](../scripts/test_capture_activations.py) runs the real
patched engine on CPU over a tiny random-weight Qwen3, and checks every capture against
`transformers` in float32 on the same checkpoint. It covers the command line, the shard writer,
the manifest, the shard bound, the resume, `--verify`, the training loader and the engine
contract, with no chip. [`../scripts/cpu_engine.py`](../scripts/cpu_engine.py) builds the engine
tree at eb061d8 with the three patches, and the engine needs the packages in
[`../upstream/models/requirements.txt`](../upstream/models/requirements.txt).

### Cost on the device

Mark every layer and `aux_hidden_states` holds one full activation per layer, live until
`logits_processor.py` runs `jnp.concat(aux_hidden_states, axis=-1)`. The concat allocates the
same bytes a second time. The table computes each figure from the layer count and width, at the
engine's default `chunked_prefill_size` of 4,096 tokens. `capture_activations.py` pins 1,024, which
puts the last two columns at a quarter.

| Model | Layers × dim | Per token | Live at 4,096 tokens | Peak, with the concat |
|---|---|---|---|---|
| Kimi-Linear-48B-A3B | 27 × 2,304 | 121 KiB | 0.47 GiB | 0.95 GiB |
| Qwen3.5-35B-A3B | 40 × 2,048 | 160 KiB | 0.62 GiB | 1.25 GiB |
| DeepSeek V3 | 61 × 7,168 | 854 KiB | 3.34 GiB | 6.67 GiB |
| GLM-4.5 | 92 × 5,120 | 920 KiB | 3.59 GiB | 7.19 GiB |

Those figures are BF16, which is what a server serves with `--enable-steering` off, the steering
patch applied or not. Turn the flag on and they double. The hook returns float32 and the stream
stays wide, so `aux_hidden_states` holds BF16 entries up to the hook site and float32 entries
after it, and `jnp.concat` promotes the whole tensor to float32. On a server that steers, read
every number in this section and the next at 2×. A `--return-hidden-states-layers` list with no
slot past `--steering-layer` holds no float32 entry, so it stays BF16.

HBM decides whether the run starts, and the table scales with two levers. The chunk size is one,
and `capture_activations.py` takes it as `--engine-arg chunked_prefill_size=N`. The slot count is
the other. `capture_activations.py` hands `--layers` to the engine as
`--return-hidden-states-layers`, so `_setup_hidden_states_capture` marks the kept slots alone. One
slot of DeepSeek V3 is 14 KiB per token, 56 MiB live at 4,096 tokens.

### Cost on the wire

Capture then moves one BF16 residual stream per captured slot per token to the host.

Measured, with the rate each model captured at. A row with two figures gives two runs, run 1
first:

| Model | Slots copied × dim | Wire bytes per token | Capture tokens/s | Wire MB/s | Run |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B | 1 × 2,816 | 5,632 | 9,535.1 and 9,553.6 | 53.7 and 53.8 | 2026-09-25, with the layer filter |
| gpt-oss-120b | 1 × 2,880 | 5,760 | 5,859.4 and 5,870.3 | 33.8 and 33.8 | 2026-09-25, with the layer filter |
| Qwen3-8B | 1 × 4,096 | 8,192 | 16,210.5 | 132.8 | 2026-09-25, with the layer filter, `PROMPTS=1000` |
| Nemotron 3 Super | 1 × 4,096 | 8,192 | 2,979.4 and 2,999.0 | 24.4 and 24.6 | 2026-09-28, with the layer filter, after the patch's float32 shard sums |
| Gemma 4 31B | 1 × 5,376 | 10,752 | 5,469.8 and 5,477.7 | 58.8 and 58.9 | 2026-09-25, with the layer filter |

The capture script counts wire bytes as returned elements times the serving dtype's width, which
is 2 for the BF16 every run above served, so each figure is `slots × dim × 2` by construction. It
shows which slots came back for every token, and isn't a reading of the link. Every run above
passed `--return-hidden-states-layers` and moved one slot. The Gemma 4 and Qwen3-8B runs served
on a `v5litepod-8` Spot slice in us-south1-a, and gpt-oss-120b and Nemotron 3 Super on a `v5p-8`
Spot slice in europe-west4-b. The `v5p-64` runs are on their model pages. Each rate runs from the
third progress line to the last one before the final shard closes. The model pages give each
window, and Qwen3-8B's is in [The whole chain, measured](#the-whole-chain-measured).

With `--enable-steering` on, the hook widens the stream to float32 and the concat promotes every
captured slot to match, so the wire doubles. Slot `steering_layer + 1` is the first wide one, so
a capture whose kept slots all sit at or below `--steering-layer` stays at 2 bytes, and
`capture_activations.py` counts it that way. The patch alone changes nothing: every run above had
the steering patch applied with the flag off and moved 2 bytes per element.

### Correctness

[`check_capture.py`](../scripts/check_capture.py) sends the same token ids through the engine and
through `transformers` on the host CPU in float32, and compares every layer. It also runs
`transformers` in BF16 as a noise floor. A token's error is its row's L2 distance from the float32
row, over the float32 row's norm, and a token past 0.25 has diverged. A layer passes two tests.
The capture's median per-token error is at most twice the BF16 forward's. Its diverged tokens
number at most twice the BF16 forward's, plus 2% of the prompt's tokens, or plus 3 on a prompt
short enough that 2% is fewer. A shift control compares each slot against the next layer's
reference, and at every layer the capture has to sit at least three times closer to its own.
`test/srt/test_hidden_states_alignment.py` in the engine patch compares every layer too, but gates
on mean absolute error alone, which hides a systematic shift.

Measured on Gemma 4 26B-A4B, `v5litepod-8` Spot slice in us-south1-a, `tp_size=8`, on the patch as
shipped, in the Run-it chain on 2026-09-25 with the layer filter:

| Prompt | Tokens | Worst ratio to BF16 | Control | Diverged tokens, capture / BF16 |
|---|---|---|---|---|
| Step 1, one prompt alone | 14 | 1.23, layer 28 | 8.9x | 2 / 2 |
| Batch of four, prompt 0 | 441 | 1.00 | 6.3x | 58 / 68 |
| Batch of four, prompt 1 | 441 | 1.02, layer 29 | 5.9x | 59 / 57 |
| Batch of four, prompt 2 | 441 | 1.00 | 6.6x | 41 / 39 |
| Batch of four, joined prompt | 1,323 | 1.00 | 7.6x | 199 / 186 |

"Diverged" counts tokens past relative error 0.25, at the layer where the most do.

BF16 drifts from float32 with depth. On the 14-token prompt, HF's own BF16 forward reads Pearson
0.9909 against float32 at layer 27, so a fixed Pearson gate fails a sound capture. The model routes
each token to 8 of 128 experts, and a near-tie resolves differently under slightly different
arithmetic. The BF16 forward and the capture both send a handful of tokens to other experts, a
different handful each time. In one batch, prompt 1's token 400 flipped at layer 9 and ended at
relative error 2.0. One such token decides a Pearson over the whole prompt, and the median doesn't
move. `--no-floor` skips the BF16 forward and gates each layer on its median per-token Pearson
instead, so no single row decides it there either.

`--prompts-file` sends several corpus prompts in one engine call plus one joined from the next
three, which runs past the engine's 1,024-token prefill pass and always splits. The check stops
before any forward when the file holds too few prompts to join, or when the joined prompt fits one
pass. Whether the others share a pass depends on when they reach the scheduler. Before the fix now
in [`../upstream/sglang-jax-877.patch`](../upstream/sglang-jax-877.patch), a split prompt kept its
last pass's rows plus the next request's, and every request after it in the pass read from a
shifted offset. One such prompt read Pearson 0.094 at layer 0, with the row count still right, so
nothing raised. Captures taken before the fix, the Qwen3-8B chain below among them, hold rows filed
under the wrong tokens. A chunk filed that way diverges at every layer from the embedding on, so
the diverged-token test fails it. The median alone passes it until close to half the prompt's rows
go wrong.

## Sparse autoencoders

An SAE is two matmuls and a top-k. The work is the activation shuffle.

DeepMind trained most [Gemma Scope](https://arxiv.org/abs/2408.05147) SAEs on TPUv3 in a 4x2
configuration. One batch for a 131K-width SAE took 45 ms on 8 chips, about 50.8% MFU. The
[Gemma Scope 2 report](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
gives the JAX specifics in Appendix B: sparse decoding written for JAX, JAX's approximate TopK,
and training with BatchTopK then converting to JumpReLU for inference.

Weights ship as `params.npz` for Gemma Scope and `params.safetensors` for Gemma Scope 2. Training
code was never released, so Appendix B is the specification.

## Steering

Static steering adds a fixed vector at a fixed layer, picking tokens with a device-side mask.

```python
h = h + alpha * v
```

Build the mask inside the compiled function from a positions array padded to a fixed length with
-1. Then the positions are an operand, and one executable covers every set of steered positions
and every value of alpha. Pass the positions unpadded and the array's shape changes with the set,
which costs one executable per set.

Conditional steering reads the activation and intervenes only on the tokens a predicate selects.
Written as a masked select, or as a `lax.cond` that skips the update when no token in the batch
fires, it stays inside the same compiled program. A Python branch on the read is what cuts the
graph: the positions become trace-time constants, so each set of them is another executable. Each
of the three device-side modes is one program across requests; the three modes are three programs.

### Precision

The engine serves a BF16 residual stream, and a BF16 add throws the steering vector away. At a
per-element residual RMS of 300 with `alpha = 1` and 5,376 dim, 99% of the delta's components come
back bit-identical, and the shift that lands has 0.18 of the norm you asked for. So the hook
widens the stream to float32, adds there, and leaves it widened for the rest of the stack.
Rounding back to BF16 after the add loses the same 99%.

Reads have the same problem. A BF16 projection puts 512 tokens on about 400 distinct values, so
the fired set drifts off the float64 answer and a flipped token moves the output by the whole
`alpha * v`. Normalize the probe and contract in float32. Ask for `HIGHEST` matmul precision too,
because a float32 matmul at TPU's default runs a single BF16 pass.

[`../steering/`](../steering/) implements both modes and tests them on CPU.

### Where the vector comes from

An SAE latent is two directions. `w_dec[f]` is what the feature writes into the residual stream,
so it's the steering vector. `w_enc[:, f]` is what the feature reads out of it, so it's the probe.
The two start tied at initialization, move apart over training, and nothing re-ties them. How far
they move depends on the run. Gemma 4 26B-A4B's latent 32177 reads cosine 0.620 after 4,000 steps
on 2026-09-25, and Qwen3-8B's latent 23394 reads 0.993 after 5,000 on 2026-09-23. Steering along
the encoder row steers along something else.

The threshold carries over too. `sae/train.py` folds the pre-encoder bias in before it fits the
thresholds, so the `b_enc` it writes is `b_enc - b_dec @ w_enc` and the encoder reads `x` as it
stands. JumpReLU fires latent `f` when `w_enc[:, f] . x + b_enc[f] >= theta[f]`, and the steering
predicate fires a token when `x . (p / ||p||) > t`. Those are the same predicate at

```
t = (theta[f] - b_enc[f]) / ||w_enc[:, f]||
```

so the fitted threshold becomes a steering threshold by dividing out the probe norm and moving
the encoder bias across. The SAE compares inclusively and the predicate compares strictly, so the
threshold comes down one float32 ulp. Parameters that still subtract `b_dec` at the encoder need
`from_sae.py --subtract-pre-bias`, which folds the bias in first.

[`../steering/from_sae.py`](../steering/from_sae.py) does the three conversions and stacks
features into a bank a server loads. Normalizing the vector is what makes `alpha` portable.
Training keeps every decoder row at unit norm, and `train.unscale_params` multiplies each one by
the input scale, so every raw row is as long as the stream's RMS token norm and says nothing
about the feature.

The layer travels with it. `sae/train.py` records `--layer` in the checkpoint as `capture_layer`,
and `from_sae.py` writes both that and `steering_layer = capture_layer - 1` into the bank's meta,
then prints the flag to pass. Without the record the two numbers look like the same number and
the bank steers one block late.

### In the engine

[`../upstream/steering-hook.patch`](../upstream/steering-hook.patch) puts both modes inside
`gemma4`'s forward pass. The hook site is a Python int on the inner model, set from
`--steering-layer` before `nnx.split()` bakes it in. Everything a request chooses reaches the
model as a device array on `ForwardBatch`: the bank, a bank row per token, `alpha` per token and a
threshold per token. One executable covers every steering configuration, and two requests in one
batch can steer with different features.

The steering is part of the radix-cache key, so a steered request and an unsteered one never
share cached KV, and two requests that steer the same way do. The server refuses
`--enable-steering` with speculative decoding, whose verify pass drops the per-token steering.
A server without `--enable-steering` refuses a steered request, and a steering server refuses a
client `extra_key` that holds `steering:`. The request validator refuses a key outside `feature`,
`alpha`, `mode`, `threshold` and `positions`, and a `threshold` beside `"mode": "static"`.

## The whole chain, measured

Capture, SAE and steering run end to end on `Qwen/Qwen3-8B`, 36 layers at 4,096 dim. Slice
`v5litepod-8`, 8 chips, `tp_size=8`, us-south1-a, BF16 engine. The capture figures come from this
line, run on 2026-09-25 with the layer filter, after the README's bootstrap and inside `tmux`:

```bash
PROMPTS=1000 bash scripts/measure_model.sh Qwen/Qwen3-8B 18 ~/results/qwen3-8b
```

At the default of 400 prompts, Qwen3-8B finishes its capture before the four progress lines the
script reads its steady window from, and the script exits 1 with "no steady window". The SAE and
steering figures come from a chain on 2026-09-23, before the chunked-prefill fix and the layer
filter.

**Capture**, `PROMPTS=1000` on a `v5litepod-8` Spot slice, 2026-09-25, with the layer filter.
1,000 wikitext passages re-encoded to 440 tokens, 8 to a call. Capture keeps slot 18 of 36 and
writes it to `/dev/shm` as float32:

| | Capture off | Capture on |
|---|---|---|
| Tokens/s | 32,778.7 | 16,210.5 |
| Window | batches 3 to 32, first two discarded, 105,600 tokens in 3.22 s | 17.6 s to 32.61 s, 243,320 tokens |
| Peak HBM per chip | 9.84 of 15.75 GiB, 11 samples over 47.5 s | 10.20 of 15.75 GiB, 27 samples over 83.2 s |
| Peak duty cycle | 36.8% | 56.1% |

Capture costs 2.02x the throughput. The wire carries 8,192 bytes per token, the one kept slot at
BF16, at 132.8 MB/s in the window, and disk takes 16,384, the same slot at float32. The run wrote
440,000 tokens into two shards, 6.714 GiB, and read 12,710.0 tokens/s end to end over 34.62 s.

The HBM figures come from this `PROMPTS=1000` run. Duty cycle is the busiest chip's peak. Two runs
at the default 400 prompts exited 1 with no steady window, and their samplers read 9.72 and
9.84 GiB off and 10.19 and 10.20 on. All three ran while other models' weights downloaded on the
same VM.

Before the layer filter, the 2026-09-23 chain moved all 36 slots, 294,912 bytes per token, and
captured at 1,184.2 tokens/s, 27.7x.

**SAE**, trained on the 2026-09-23 chain's own capture of slot 18: 6,400 wikitext-103 passages
cut at 440 tokens, 2,700,102 tokens in 11 shards, 41.2 GiB. That capture ran before the
chunked-prefill fix, so its rows from prompts that split across prefill passes sit under the wrong
tokens, and each batch's last pass added padding rows. BatchTopK, expansion 16 so d_sae 65,536,
k=64, batch 512, 5,000 steps with the first 500 as warmup, 641.69 s:

| | |
|---|---|
| fvu at steps 4,700 / 4,800 / 4,900 / 4,999 | 0.0098 / 0.0051 / 0.0063 / 0.0280 |
| L0 | 64.0, flat |
| Live / dead latents | 5,302 / 60,234 over 64 calibration batches |
| Thresholds min / median / max | 0.00105 / 0.00129 / 0.02212 |

Most latents die. A dead latent's threshold is infinite, so conditional steering never fires it.
Static steering reads no threshold and adds its row to every token it steers. `from_sae.py`
refuses a dead latent, and a bank that holds one warns when it's saved and when a server loads it.

**Steering**, feature 23394 from that SAE, static, applied after layer 17:

| alpha | prompts changed, of 4 |
|---|---|
| 0.25 | 1 |
| 1.0 | 3 |
| 4.0 | 3, and one collapses into a repetition loop |

`from_sae` reports encoder-decoder cosine 0.99342 and decoder norm 4,484.59 for that feature. The
decoder norm is the SAE's input scale, the RMS token norm at slot 18, and every latent shares it.
The bank vector is unit length, so alpha is the shift's length in activation units: at alpha 1 a
steered token moves by 1/4,485 of the RMS token norm, and at alpha 4 by 1/1,121. That chain had
no random-direction arm, so the changed prompts show the hook moves the output, and nothing about
feature 23394. `compare.py` sets alpha as a share of the norm and steers a random direction
beside the feature. The served model then labels each changed reply coherent or broken, and the
run exits 0 when the feature has more coherent changed replies than the random direction at some
alpha, and 1 when it doesn't. A judge that mislabels one of its two known replies, or gives no
label, exits 2. A crash, such as a `--model-path` with no tokenizer, also exits 1, with a
traceback. [The Gemma 4 26B-A4B page](../models/gemma4-26b-a4b.md#the-run-it-chain) has a chip
run under that rule that exits 0, on 2026-09-25.

The 2026-09-23 chain measured no HBM. `Device.memory_stats()` needs the calling process to hold
the TPU, and the engine's scheduler child holds it for the whole run, so the parent gets nothing.

[`scripts/peak_hbm.py`](../scripts/peak_hbm.py) reads it from outside instead. `tpu-info` pulls the
same counters out of `libtpu` from its own process, so it runs alongside the job:

```bash
uv pip install tpu-info
python3 scripts/peak_hbm.py --seconds 600 --out peak_hbm.json    # in a second shell
```

Untested as written: `measure_model.sh` runs the same sampler with `--seconds 14400 --every 2`
beside every measured engine run.

The runtime keeps no high-water mark, so this is the peak of what it sampled. Sample often enough
to catch the load and report the sample count beside the number.

## Engines

| Engine | Models | Capture |
|---|---|---|
| `sglang-jax` | JAX implementations, TPU only | yes, with the patches |
| `tpu-inference` | broad, via torchax | no hook |

`tpu-inference` is what vLLM's TPU platform delegates to. Its `aux_hidden_states` path is tied to
speculative decoding with fixed layer indices and no route to the API.

`sglang-jax` is TPU-only in practice, so using it means running a different engine on TPU than on
GPU.

## PyTorch tooling

Captures are `.npy` shards with a JSON manifest. SAEs and steering banks are `.npz`. PyTorch code
reads all three with `numpy.load` and `torch.from_numpy` on any host.

Numerics need checking on any path.
[pytorch/xla#7050](https://github.com/pytorch/xla/issues/7050) records `torch.einsum` producing
different SGLD dynamics on TPU against CPU and CUDA, reproducible across seeds and two
architectures. Validate against a CPU reference.
