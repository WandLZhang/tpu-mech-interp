# Mechanistic interpretability on Cloud TPU

Serving large language models on Google Cloud TPU with per-layer activation capture, for sparse
autoencoder training and causal steering.

## 1. Hardware

A 32-chip TPU v5p slice holds 3,040 GiB, enough for a frontier model in BF16 with room for KV
cache and activations.

Slice names count TensorCores. A v5p chip holds two, so 32 chips is `v5p-64`. Topology 2x4x4,
8 hosts, machine type `ct5p-hightpu-4t`. A v5e chip holds one, so `v5litepod-8` is 8 chips on one
host, 16 GiB of HBM each.

| | Per chip | 32 chips |
|---|---|---|
| HBM | 95 GiB | 3,040 GiB |
| HBM bandwidth | 2,765 GB/s | 88.5 TB/s |
| BF16 dense | 459 TFLOPS | 14.7 PFLOPS |
| ICI | 1,200 GB/s bidirectional | 3D mesh; slices below a 64-chip cube have no wraparound links |

Two KDA models ran on a `v5p-64`: Kimi K3 through the engine's Pallas KDA kernels, and
GLM-5.3-Flash through the port's plain-XLA KDA. The KDA kernels in `sglang-jax` never read the chip
generation, and no KDA model has run on v6e here, so which generation serves one best stays open.
Mamba-2 needs no kernel at all: [`scan/mamba2.py`](scan/mamba2.py) is plain XLA, and both Nemotron
3 sizes served on v5p.

### Rates

Per chip-hour, and per day at 32 chips.

| Vehicle | Region | $/chip-hr | $/day |
|---|---|---|---|
| Spot | us-south1 | 0.6338 | 487 |
| Spot | us-east5 | 1.0034 | 771 |
| 3-year committed use discount | us-east5 | 1.89 | 1,452 |
| Dynamic Workload Scheduler, Flex Start | us-east5 | 2.10 | 1,613 |
| Dynamic Workload Scheduler, Calendar | us-east5 | 2.94 | 2,258 |
| On demand | us-east5 | 4.20 | 3,226 |

Spot rates come from the Cloud Billing Catalog. The rest come from the
[TPU pricing page](https://cloud.google.com/tpu/pricing). Spot prices change monthly.

Google can reclaim a Spot slice at any time. A Flex Start slice runs for up to 7 days, then Google
deletes it. Both bill for as long as the machine is up. Calendar mode reserves a block ahead of
time and bills for all of it.

### Getting a slice

You need an authenticated `gcloud`, a project with the Cloud TPU API on
(`gcloud services enable tpu.googleapis.com`), and quota for the chips you ask for.
[`scripts/spray_tpu_spot.sh`](scripts/spray_tpu_spot.sh) submits to every zone that offers the
accelerator type, keeps the first slice to reach READY, and deletes the rest. Set `ZONES` to
narrow the spray and `MODE=ondemand` to skip Spot. It doesn't book Flex Start or Calendar.

```bash
PROJECT=your-project ACCEL=v5p-64 bash scripts/spray_tpu_spot.sh
```

The recorded sprays asked for `v5litepod-8`, `v5p-8` and `v5p-64`, and [Run it](#run-it) gives
the `v5litepod-8` line. Each model page names the zone and date of the slice its runs got. A slice
whose hosts read a GCS bucket needs the cloud-platform scope:
`TPU_CREATE_FLAGS="--scopes=https://www.googleapis.com/auth/cloud-platform"`.

The spray prints the `ssh` command for the slice it kept. That `ssh` needs a firewall rule on the
slice's network that admits TCP port 22 from your address. The recorded runs added one for each
run and deleted it at teardown. On the VM,
[`scripts/bootstrap_tpu_vm.sh`](scripts/bootstrap_tpu_vm.sh) installs the engine, and its smoke
test prints the chips the host sees: `DEVICES 8 TPU v5 lite` on a `v5litepod-8`, 4 on a `v5p-8`.
A `v5p-64` spans 8 hosts; [Across hosts](#across-hosts) covers it.

### Tear down

A slice bills while it's up, busy or idle. The spray prints the delete command for the slice it
kept, with the name and zone filled in:

```bash
gcloud compute tpus queued-resources delete NAME --zone=ZONE --project=PROJECT --force
```

Then list what's left in every zone. `--zone=-` covers them all, so a slice the spray lost track
of still shows. Both lists should come back empty.

```bash
gcloud compute tpus queued-resources list --zone=- --project=PROJECT --format="value(name)"
gcloud compute tpus tpu-vm list --zone=- --project=PROJECT --format="value(name)"
```

The spray also prints the zones it tried, as a `ZONES="..."` line under its spraying line. On its
way out it prints both list commands as a loop over those zones, with the project filled in.

### Across hosts

A `v5p-64` is 8 hosts of 4 chips, and every model here over 380 GiB needs one. Six models ran that
way in us-east5-a on 2026-09-27 and 28, each through its row of
[`scripts/multihost_run.sh`](scripts/multihost_run.sh). A run needs four things first:

1. The checkpoint in a GCS bucket in the slice's region, one folder per model, named as the
   script's row expects. A 1 TB-class checkpoint doesn't fit a host's RAM, so every host reads it
   through gcsfuse, which [`scripts/multihost_setup.sh`](scripts/multihost_setup.sh) mounts.
2. A capture reference in the bucket's `refs/` folder, built on a CPU VM that holds one decoder
   layer at a time: `scripts/check_capture.py --offload-folder DIR --reference-only`, with the
   flags the script's header gives.
3. The slice, created with the cloud-platform scope above so its hosts can read the bucket.
4. A firewall rule that lets the hosts reach each other on their internal IPs, and SSH from your
   machine to each host.

Then, from the repo root:

```bash
BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE MODEL setup check measure
```

`setup` bootstraps every host, mounts the bucket and checks that every host sees all 32 chips.
`check` runs the capture check against the reference. `measure` runs `measure_model.sh` with the
engine spread over every host by [`scripts/multihost_exec.sh`](scripts/multihost_exec.sh). Each
model page gives its row and its results.

Notes: nobody has replayed this section as written from a clean start; the model pages' runs came
from the same script. When a host's external IP stops answering, `SLICE_SSH_CONFIG` takes an ssh
config that reaches it through host 0.

## 2. Models

| Family | Repo | Total / active | Architecture | BF16 GiB | Status | Code |
|---|---|---|---|---|---|---|
| [Gemma 4 31B](models/gemma4-31b.md) | `google/gemma-4-31B-it` | 31.3B | GQA, 5:1 SWA | 58.3 | **measured on TPU**, `v5litepod-8`, 2026-09-25, with the layer filter | upstream |
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | `google/gemma-4-26B-A4B-it` | 25.8B / 3.8B | MoE, 128 experts top-8 plus 1 shared, 5:1 SWA | 48.1 | **measured on TPU**, `v5litepod-8`, 2026-09-25, with the layer filter | upstream |
| Qwen3-8B, no page; the [capture guide](docs/activation-capture.md#the-whole-chain-measured) has its numbers | `Qwen/Qwen3-8B` | 8.2B | GQA | 15.3 | **measured on TPU**, `v5litepod-8`, 2026-09-25, with the layer filter | upstream |
| [gpt-oss-120b](models/gpt-oss-120b.md) | `openai/gpt-oss-120b` | 116.8B / 5.1B | MoE 128 experts top-4, 1:1 SWA, sinks | 217.6 | **measured on TPU**, `v5p-8`, 2026-09-25, with the layer filter | our patch |
| [Nemotron 3 Super](models/nemotron3-super.md) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 123.6B / 12B | Mamba-2 + LatentMoE | 230.2 | **measured on TPU**, `v5p-8`, 2026-09-28, with the layer filter | our patch |
| [Nemotron 3 Ultra](models/nemotron3-ultra.md) | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 560.5B / 55B | Mamba-2 + LatentMoE | 1,044.1 | **measured on TPU**, `v5p-64`, 2026-09-28, with the layer filter | our patch, shared with Super |
| [Inkling](models/inkling.md) | `thinkingmachines/Inkling` | 975B / 41B | MoE, no RoPE, short conv | 1,773.9 | **captured on TPU**, `v5p-64`, 2026-09-27; every layer within the BF16 floor, one control under its bar; not measured ([roadmap](docs/roadmap.md)) | our patch |
| [Kimi K3](models/kimi-k3.md) | `moonshotai/Kimi-K3` | 2.78T / 104B | 69 KDA + 24 gated MLA | 5,178 | **captured on TPU**, `v5p-64`, 2026-09-27; the capture check passes, prefill runs at 0.6 tokens/s; not measured ([roadmap](docs/roadmap.md)) | our patch |
| [GLM-5.3](models/glm5.3.md) | `zai-org/GLM-5.3` | 753B / 40B | MLA + sparse top-2048 indexer | 1,403 | **measured on TPU**, `v5p-64`, 2026-09-27 and 28, with the layer filter | upstream, our capture hook and two fixes |
| [GLM-5.3-Flash](models/glm5.3-flash.md) | `zai-org/GLM-5.3-Flash` | 320B / 18B | 34 KDA + 11 sparse MLA, mHC | 598.5 | **measured on TPU**, `v5p-64`, 2026-09-28, with the layer filter | our patch |
| [DeepSeek V4.1-Flash](models/deepseek-v4.1-flash.md) | `deepseek-ai/DeepSeek-V4.1-Flash` | 552B + 196.6B Engram | encoder-decoder, n-gram memory | 1,394 | **measured on TPU**, `v5p-64`, 2026-09-27, with the layer filter | our patch |

**Status** says what has been shown, strongest first. Every model page opens with the same status.

| Status | Meaning |
|---|---|
| **measured on TPU**, slice, dates | served and captured on that slice on those dates; the model page has the numbers and the command behind them |
| **captured on TPU**, slice, date | served on that slice and captured for the capture check; the model page gives the check's result, and the roadmap says what the measure still needs |

**With the layer filter** means the capture passed `--return-hidden-states-layers`, so the engine
copied only the kept slot to the host.

Open work, Kimi K3 and Inkling first, sits in [the roadmap](docs/roadmap.md).

**Code** says where the model lives. `upstream` ships in `sglang-jax` as is. `our patch` lives in
[`upstream/models/`](upstream/models/), and the capture hooks live in [`upstream/`](upstream/).
`bash scripts/verify_patches.sh` checks that every patch still applies; it doesn't say a model
runs.

Kimi K3, GLM-5.3-Flash and both Nemotron 3 sizes carry a recurrent state. The Kimi K3 and
Nemotron 3 patches keep it in a state pool beside the paged KV cache, which is how Kimi K3 and both
Nemotron 3 sizes serve. Splitting one long sequence across chips takes the sharded scan in
[`scan/`](scan/).

A 32-chip v5p slice holds 3,040 GiB, so every row except Kimi K3 fits one slice at BF16. The
engine shards a model within one slice, so Kimi K3's 5,178 GiB of BF16 weights would take a
64-chip `v5p-128`, 6,080 GiB, or a `v6e-256`, 8,192 GiB. Its routed experts ship MXFP4, and kept
packed they bring the weights to 1,453 GiB, which is how it runs on one `v5p-64`: the patch decodes
each expert stack to BF16 just before its grouped matmul. That decode is one suspect for its slow
prefill.

Both Gemma 4 rows count a vision encoder of about 550M parameters. The model cards give the
language model alone: 30.7B for 31B and 25.2B for 26B-A4B.

gpt-oss-120b also ships MXFP4 routed experts, 60.8 GiB on disk. It decodes to BF16 at load, so
the 217.6 GiB in the table is what lands in HBM.

The BF16 column counts each checkpoint's parameters at 2 bytes, and several checkpoints carry parts
the engine doesn't build: both Nemotron 3 rows and GLM-5.3 count a multi-token-prediction head, and
Inkling counts its head plus vision and audio towers. Each model page gives what lands in HBM.

## 3. Operations

| Operation | Support |
|---|---|
| Per-layer activation capture | `sglang-jax` with [`upstream/sglang-jax-877.patch`](upstream/sglang-jax-877.patch) |
| Sparse autoencoders | [`sae/`](sae/), BatchTopK to train, JumpReLU to serve |
| Causal steering | [`steering/`](steering/), static and conditional, one program per mode |
| Steering a served model | `sglang-jax` with [`upstream/steering-hook.patch`](upstream/steering-hook.patch) for `gemma4`, plus [`upstream/qwen3-steering-hook.patch`](upstream/qwen3-steering-hook.patch) for `qwen3` |

[`docs/activation-capture.md`](docs/activation-capture.md) covers all four.

All four run on hardware. On a `v5litepod-8` Spot slice in us-south1-a on 2026-09-25, with the
layer filter, the Run-it chain below went the whole way on Gemma 4 26B-A4B. Against a float32
forward, every captured layer stays within 1.23 times a BF16 forward's error, on one prompt and on
a batch that splits a prompt across prefill passes.
Step 2 captures 2.2M tokens, step 3 trains an SAE for 4,000 steps, and step 4 steers one of its
features beside a random direction of the same length. At a tenth of the stream's norm the
feature rewords three of four replies and the served model judges all three coherent. The random
direction breaks all four. [The model page](models/gemma4-26b-a4b.md#the-run-it-chain) has the
numbers. [`docs/activation-capture.md`](docs/activation-capture.md) has the setup, and an earlier
chain on Qwen3-8B that ran on 2026-09-23, before the chunked-prefill fix. Qwen3-8B has no model
page, and the capture guide holds its numbers.

### Run it

The whole chain on Gemma 4 26B-A4B, which fits one `v5litepod-8`. Get the slice from your
workstation:

```bash
PROJECT=your-project ACCEL=v5litepod-8 bash scripts/spray_tpu_spot.sh
```

The spray prints the `ssh` command for the slice it kept. On the VM, run
`git clone https://github.com/WandLZhang/tpu-mech-interp && cd tpu-mech-interp` and start `tmux`.
Run every line below from the repo root in that one `tmux` shell. `SNAP` and `FEATURE` exist only
in that shell, and a dropped `ssh` session leaves a `tmux` shell running.

```bash
bash scripts/bootstrap_tpu_vm.sh && source ~/.tpu_env
python3 scripts/fetch_weights.py google/gemma-4-26B-A4B-it | tee fetch.log
export SNAP=$(awk '/^PATH /{print $2}' fetch.log)

# 1. every captured layer sits as close to a float32 forward as a BF16 forward does
python3 scripts/check_capture.py --model-path "$SNAP" --tp-size 8

# 2. 2.2M tokens of one middle layer, then the same check on a batch that splits a prompt
python3 scripts/build_corpus.py --model "$SNAP" --prompts 5000 --tokens 440 --out prompts.jsonl
python3 scripts/capture_activations.py --model-path "$SNAP" --prompts prompts.jsonl \
    --out /dev/shm/caps --layers 15 --tp-size 8 --batch-size 8 --dtype float32
python3 scripts/check_capture.py --model-path "$SNAP" --tp-size 8 --prompts-file prompts.jsonl

# 3. an SAE, 4,000 steps with the first 500 as warmup
python3 sae/train.py --manifest /dev/shm/caps/manifest.json --layer 15 \
    --expansion-factor 16 --k 64 --steps 4000 --warmup-steps 500 --batch-size 512 \
    --out sae_l15.npz

# 4. steer with a latent that fires, beside a random direction as control; the model judges
#    each changed reply coherent or broken
python3 steering/pick_feature.py --sae sae_l15.npz --manifest /dev/shm/caps/manifest.json \
    --layer 15 | tee pick.log
export "$(tail -n 1 pick.log)"
python3 steering/from_sae.py --sae sae_l15.npz --feature "$FEATURE" --out steer_l15.npz
python3 steering/compare.py --model-path "$SNAP" --bank steer_l15.npz --steering-layer 14 \
    --feature "$FEATURE" --tp-size 8
```

`pick.log` keeps the ranking `pick_feature.py` prints. Its last line, `FEATURE=<id>`, names the
latent that `from_sae.py` and `compare.py` take.

On a `v5litepod-8` Spot slice in us-south1-a on 2026-09-25, with the layer filter, the chain took
76 minutes from bootstrap to `compare.py`. The fetch took 54 of them at 15.9 MB/s, because the
49.9 GB first shard came down one HTTP connection. The steps from the first check to `compare.py`
took 21.5 minutes: the first check 3, the capture 6, the second check 3, the SAE 6 and
`compare.py` 3. Every other step took under a minute. The fetch rate varies. The record holds
52.4 MB/s on 2026-09-24, 105.9 MB/s on 2026-09-25 before the layer filter, 15.9 MB/s on this run,
and 123.7 MB/s on a replay in us-east1-c on 2026-09-28, whose steps from the first check to
`compare.py` took 21.7 minutes. A killed fetch starts over from zero.

`--steering-layer` is the capture slot minus one. `compare.py` sends each prompt as a chat turn,
because the `-it` model continues raw text badly, and sends the turn as token ids, so it opens on
one BOS as the captured prompts did. It sets alpha at 0.1, 0.5 and 1.0 of the stream's RMS norm,
and steers a random unit direction at each length as a control.

The served model, unsteered, then labels each changed reply `coherent` or `broken`: it gets the
prompt and the reply as a question, and the label whose first token it rates more likely wins. A
known-coherent reply and a known-broken reply go through the same judge in the same batch. A
judge that mislabels either one makes the run exit 2 and print its answers. Otherwise the run
exits 0 when the feature has more coherent changed replies than the random direction at some
length, and 1 when it doesn't. It prints each changed reply with its label and one count line per
length, and its last line gives the exit code and the counts behind it. A crash in `compare.py`
itself, such as a `--model-path` with no tokenizer, also exits 1 and prints a traceback, so read
the output before you trust a 1.

On 2026-09-25, with the layer filter, `compare.py` exited 0. At 0.1 of the norm the feature
changed 3 of 4 replies, and the judge called all 3 coherent. The random direction broke all 4 into
repeated tokens such as "to to to", and the judge called none coherent. Both judge controls came
back right. At 0.5 and 1.0 both directions broke every reply.

The SAE and the latent it picks change from run to run. This run kept 2,023 live latents of
45,056 and picked latent 32177. [The model page](models/gemma4-26b-a4b.md#the-run-it-chain) has
the earlier runs' picks.

Some output looks like an error and isn't:

- `hugepage_text.cc:344] RAW: File offset incorrectly aligned for file-backed THP` at every JAX
  start, and `huggingface_hub`'s warning about unauthenticated requests. No measured model is
  gated, so no `HF_TOKEN` is needed.
  [The capture guide](docs/activation-capture.md#setting-up-a-tpu-vm) covers both.
- The `Downloading bytes` and `Reconstructing (incomplete total...)` bars during a fetch.
  `huggingface_hub` 1.33 can draw them with Xet off, and `~/.tpu_env` turns Xet off with
  `HF_HUB_DISABLE_XET=1`. A fetch whose output goes to a file may show only `Fetching N files`.
- `Loading MoE Weights: 0it` at each engine start. The shared loader always runs its expert pass.
  On Gemma 4 that pass finds nothing, because 31B has no experts and `gemma4.py` loads 26B-A4B's
  stacked expert tensors itself once the shared loader finishes.
- A `transformers` warning that `use_fast` is deprecated. The engine loads Gemma 4's image
  processor with `use_fast=True`.
- `libtpu metrics unavailable (StatusCode.UNAVAILABLE)`, and later `libtpu metrics came back
  after N empty round(s)`, in `measure_model.sh`'s `hbm_*.log` files. `peak_hbm.py` starts
  before the engine holds the TPU, and `libtpu` serves its metrics only once a process holds it.

Each measured model page gives the command behind its throughput and HBM figures. A single-host
page calls [`scripts/measure_model.sh`](scripts/measure_model.sh) after the same bootstrap, inside
`tmux`. A `v5p-64` page runs the same script through the `measure` step of
`scripts/multihost_run.sh`. Qwen3-8B has no page, and on a `v5litepod-8` its call reads:

```bash
PROMPTS=1000 bash scripts/measure_model.sh Qwen/Qwen3-8B 18 ~/results/qwen3-8b
```

`PROMPTS` sets how many corpus prompts the capture runs, 400 by default. The script reads the
capture-on rate over progress lines it prints every 5 s, and it needs four of them before the last
shard closes. Qwen3-8B captures one slot fast enough that 400 prompts finish first, so the script
exits 1 with "no steady window". Gemma 4 26B-A4B cleared the rule on 2026-09-25 with four lines,
the fewest it takes. The script's header also documents `TP` and `MEM_FRAC`.

The capture guide's Qwen3-8B throughput and HBM figures come from that line, run on 2026-09-25
with the layer filter. Its Qwen3-8B SAE and steering figures come from a chain on 2026-09-23.

Delete the slice when you're done. [Tear down](#tear-down) has the commands.

The CPU gates need no TPU. Run them from the repo root on any Linux machine, in a Python 3.12 venv
of their own, because the TPU VM's `~/v312` carries `transformers` 5.12 and
`upstream/models/requirements.txt` pins `transformers>=5.17` for the model tests. On a 90-vCPU
`c3d-highcpu-90` in us-central1-a on 2026-09-28, all 46 gates passed in 2 hours 3 minutes, most of
it in the model tests under `upstream/models/`: `test_deepseek_v41_model.py` took 27 minutes and
`test_glm5_next_model.py` 21. The capture, check and throughput
gates start the real engine on CPU, on the patched `sglang-jax`, against a tiny random-weight
Qwen3. The steering gates do the same against a cut-down Gemma 4 26B-A4B. So the venv also takes
the engine's own imports, which `upstream/models/requirements.txt` lists, and `tpu-info` from
`scripts/requirements.txt`. Each gate prints its line when it ends, so the terminal stays quiet for
up to 27 minutes while a model test runs. Its log grows meanwhile in the `/tmp/test_all-logs.*`
folder the script makes. A gate that fails prints its whole log, and the logs stay on disk. A clean
run deletes the folder, so run one test file alone to read its checks.

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv if it's missing
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --torch-backend=cpu -r scan/requirements.txt -r sae/requirements.txt \
    -r upstream/models/requirements.txt -r scripts/requirements.txt
bash scripts/test_all.sh
```

`--torch-backend=cpu` takes `torch` from PyTorch's CPU index. Without it, `uv` pulls the CUDA build
and about 20 `nvidia-*` wheels, several GB that a CPU-only machine never uses.

On a machine whose hostname runs 64 characters or more, the two-process check in
`upstream/test_steering_hook.py` fails inside gloo, at `device.cc:151`. Give the machine a short
name with `sudo hostnamectl set-hostname NAME` and rerun. A GCE VM hit this on 2026-09-25.

### Capture

`--enable-return-hidden-states` returns a list of chunks for each request. Each prefill pass adds
one `[rows, num_layers, hidden_dim]` chunk, and each decode step adds one
`[num_layers, hidden_dim]` row. A prompt longer than one pass, or one that shares a pass with
other prompts, arrives in several chunks, so join the 3-D chunks to get
`[seq_len, num_layers, hidden_dim]`. Any slot is then `hs[token, slot, :]`. Slot `i` holds the
residual stream entering block `i`, which is block `i - 1`'s output, and slot 0 is the embedding
output. An `N`-block model gives `N` slots, so the last block's output has no slot.
`--return-hidden-states-layers 15 20`, or `return_hidden_states_layers=[15, 20]` on the `Engine`,
returns those slots alone in ascending order, so `hs[token, 0, :]` is slot 15.

`bootstrap_tpu_vm.sh` builds the engine with this patch and the steering patches. To build the
capture tree by hand, run this from the repo root. `git am` makes a commit, so the `-c` pair gives
it a committer on a machine with no git identity. No recorded run typed these lines. The measured
runs built their trees through `bootstrap_tpu_vm.sh`, which runs the same `git am`.

```bash
REPO=$PWD
git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
git checkout eb061d8            # every patch in upstream/ applies here
git -c user.name=you -c user.email=you@example.com am < "$REPO/upstream/sglang-jax-877.patch"
```

`bash scripts/verify_patches.sh` applies all 21 patches to a clean checkout at `eb061d8`, in 17
checks, and names what fails. Main moves, so a hunk that applies at `eb061d8` can stop applying
on main with no change on this side. `SGL_COMMIT=main` runs the same check against main.

The loop a capture runs, as a sketch with two `...` to fill in:

```python
import numpy as np
from sgl_jax.srt.entrypoints.engine import Engine

# Save this as a .py file. The engine re-runs the main module in its subprocesses.
if __name__ == "__main__":
    engine = Engine(model_path=..., enable_return_hidden_states=True)
    request = {
        "prompt": [...],
        "sampling_params": {"max_new_tokens": 1},  # at eb061d8, leaving it out raises TypeError
        "return_hidden_states": True,
    }
    print("generate", request)
    for out in engine.generate(**request):
        chunks = out["meta_info"]["hidden_states"]
        hs = np.concatenate([np.asarray(c) for c in chunks if np.ndim(c) == 3])
        # hs is [seq_len, num_layers, hidden_dim], one row per prompt token
```

[`scripts/capture_activations.py`](scripts/capture_activations.py) runs that loop over a prompt
file and streams the layers you keep to `.npy` shards with a JSON manifest, which is the format
[`sae/train.py`](sae/train.py) reads.

```bash
python3 scripts/capture_activations.py --model-path MODEL --prompts prompts.txt \
    --out caps --layers 20 --shard-bytes 2147483648 --tp-size 8
python3 sae/train.py --manifest caps/manifest.json --layer 20 \
    --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz
python3 steering/from_sae.py --sae sae_l20.npz --feature 40977 --out steer_l20.npz
python3 -m sgl_jax.launch_server --model-path MODEL --tp-size 8 \
    --enable-steering --steering-bank steer_l20.npz --steering-layer 19
```

Untested as written: no recorded run has taken this block, and the tested chain is
[Run it](#run-it). `MODEL`, `prompts.txt` and feature 40977 are placeholders.

At the default batch of 4,096, `--steps 100000` reads about 410M tokens, so size `prompts.txt` to
match. `train.py` stops before training when the shards hold fewer than it needs.

Compiled for a `v5litepod-8` on 2026-09-25, that command's training step takes 15.33 GiB a chip
at `d_model` 4,096. At 5,376 its temporaries alone need 20.42 GiB against the chip's 15.75 GiB,
and the compile fails.

`--steering-layer 19`, not 20. Capture takes the stream entering block 20 and steering writes the
stream leaving the block you name, so the two flags land on the same tensor one apart.

The host holds one batch, not one request. `engine.generate` returns when the whole batch
finishes, so `--batch-size` requests are live at once. At `--layers all`, Gemma 4 31B returns
1.29 MB per token in float32, so 2,048-token prompts come to about 21 GB at the default 8.
`--layers 20` returns one slot of it. A shard closes at the first prompt boundary at or above
`--shard-bytes`, so a crash costs one shard and rerunning against the same `--out` continues from
there. Every shard carries a SHA-256 in the manifest, and `--verify` re-reads the tree against it.

Three model files ship the per-layer hook: `gemma4`, `llama` and `qwen3`
([survey](upstream/capture-hook-survey.csv)). `qwen3_vl` builds its text model from the `qwen3`
one, so it carries the hook too. On a model without it, `--enable-return-hidden-states` refuses
to start.
[`upstream/glm5-capture-hook.patch`](upstream/glm5-capture-hook.patch) adds it to `glm5_moe`,
which serves [GLM-5.3](models/glm5.3.md). [GLM-5.3-Flash](models/glm5.3-flash.md) is a different
architecture, `glm5_next`, which `sglang-jax` doesn't implement; this repo's model patch adds it
with the hook.

[`upstream/capture-hooks/`](upstream/capture-hooks/) adds it to four more: `kimi_linear`,
`qwen3_5`, `deepseek_v3` and `glm4_moe`. Those are Kimi-Linear-48B, Qwen3.5, DeepSeek V3 and
GLM-4.5, so they exercise the pattern on each architecture family rather than on the checkpoints
above. None of the four has a model page, because nothing here serves them. A CPU test runs each
hook's patched layer loop against a float64 reference.

[`upstream/models/`](upstream/models/) writes a model the engine doesn't have, then hooks it.
`kimi_k3` lands there as two patches, model then hook, and covers `moonshotai/Kimi-K3`.
`nemotron_h` lands the same way and covers both Nemotron 3 sizes. `gpt_oss` and `inkling` each
land as one patch that carries the hook already, covering `openai/gpt-oss-120b` and
`openai/gpt-oss-20b`, and `thinkingmachines/Inkling` and `thinkingmachines/Inkling-Small`.
gpt-oss-20b shares the [gpt-oss-120b page](models/gpt-oss-120b.md) and has never served on a chip.
Inkling-Small shares the [Inkling page](models/inkling.md). `deepseek_v41` and `glm5_next` land as
one patch each with the hook, covering `deepseek-ai/DeepSeek-V4.1-Flash` and
`zai-org/GLM-5.3-Flash`.

gpt-oss, Kimi K3 and Nemotron 3 all edit `layers/moe.py`, DeepSeek V4.1 collides with Kimi K3 and
Nemotron 3, and GLM-5.3-Flash carries the runner hooks DeepSeek V4.1 does, so one clone takes one
model patch. A serving instance runs one model anyway.
[`upstream/models/README.md`](upstream/models/README.md) lists which patches share a file.

Capture moves one BF16 residual stream per kept slot per token to the host.
`capture_activations.py` starts the engine with `--return-hidden-states-layers` set to its
`--layers`, so the engine copies those slots alone. `--layers all` leaves the flag unset, and the
engine copies every slot.

That's `len(layers) × dim × 2` bytes per token. One slot of Qwen3-8B is 8,192 bytes, and all 36
are 294,912. One slot of Gemma 4 26B-A4B is 5,632 and all 30 are 168,960, at 2,816 dim. One slot
of Gemma 4 31B is 10,752 and all 60 are 645,120, at 5,376. The capture script counts wire bytes as
returned elements times the engine dtype's width, 2 for BF16, so its figure shows which slots came
back for every token. It isn't a reading of the link. On the CPU rig's tiny Qwen3, six slots of 64
at BF16, `scripts/test_capture_activations.py` check 0 prints 768 wire bytes a token at
`--layers all` and 128 at `--layers 3`. Run the file alone to see those lines; `test_all.sh` keeps
a gate's log only when something fails.

Capture costs throughput. On a `v5litepod-8` Spot slice in us-south1-a on 2026-09-25, with the
layer filter and one kept slot, Gemma 4 26B-A4B served 13,158.8 and 13,185.7 tokens/s with capture
off on two runs, and 9,535.1 and 9,553.6 with it on, 1.38x both times. Gemma 4 31B served 6,942.0
and 6,937.4 off and 5,469.8 and 5,477.7 on, 1.27x both times. Qwen3-8B, at `PROMPTS=1000`, served
32,778.7 off and 16,210.5 on, 2.02x. The same day, on a `v5p-8` Spot slice in europe-west4-b with
the same filter, gpt-oss-120b served 7,279.2 and 7,271.9 off and 5,859.4 and 5,870.3 on, 1.24x
both times; a replay on 2026-09-28 matched within 0.1%. On a `v5p-8` Spot slice in europe-west4-b
on 2026-09-28, Nemotron 3 Super served 3,916.3 and 3,917.1 off and 2,979.4 and 2,999.0 on, 1.31x
both times, 10% under its 2026-09-25 rates from before its patch summed tensor shards in float32.
On a `v5p-64` Spot slice in us-east5-a on 2026-09-27 and 28, with the same filter and one
run each, Nemotron 3 Ultra served 3,095.3 off and 2,218.2 on, 1.40x. GLM-5.3 served 2,685.8 and
2,417.1, 1.11x. GLM-5.3-Flash served 1,791.0 and 1,293.5, 1.38x. DeepSeek V4.1-Flash served
3,414.5 and 2,570.2, 1.33x. Each rate reads a steady window, and the model pages give each window.
The single-host pages set each rate beside that model's rate from before the filter, and
[`docs/activation-capture.md`](docs/activation-capture.md) tabulates the single-host runs.

## 4. Sharded scan

Mamba-2 and KDA layers hold a recurrent state. A paged-attention cache manager has no slot for it,
and the scan that updates it can't split across a sequence shard.

The inter-chunk recurrence is affine, so it composes associatively:

```
h' = A @ h + B
```

[`scan/`](scan/) implements the sharded form and tests it on CPU, no chip needed. At 64 chunks,
4 heads, K=V=128 it matches a sequential reference to relative error 1.5e-07.

The six test files run unchanged on a TPU VM and on a workstation. On the VM, run them in the
bootstrap venv after `source ~/.tpu_env`. That venv carries `libtpu`, so jax takes the chips. On a
workstation, build the venv that [`scan/README.md`](scan/README.md#test) gives, and jax runs them
on 8 simulated CPU devices. The four tests that build a mesh print which backend they got.
`test_nemotron3_layers.py` prints the backend its one-device check runs on, and
`test_deepseek_v4_layers.py` runs no jax.

The first line runs all six from the repo root and names any that fails. The second shows which
backend the mesh took. On a TPU VM one process holds the chips at a time, so run them when no
engine step is running; a scan test that overlaps an engine start makes the engine fail with "The
TPU is already in use".

```bash
for t in scan/test_*.py; do python3 "$t" || echo "FAIL $t"; done
python3 scan/test_affine_scan.py | grep mesh   # mesh: ctx=8 on tpu
```

On eight real chips, a `v5litepod-8`, the sharded scan matches one device to 3.7e-08 to 2.5e-07
relative on four runs, one on 2026-09-24, two on 2026-09-25 and one in us-east1-c on 2026-09-28.
That's the interconnect claim, and on CPU the same check reads 3.7e-08 to 4.8e-07. All six tests passed on the first run. On the
second, `test_nemotron3_layers` failed the three checks that save a config through the VM's
`transformers` 5.12.1, which writes the old layer names. Its chip checks and the other five tests
passed. The test now reads either set of names. It passes under 5.12.1 on CPU, and all six passed
on the third run.

Against a float32 token-by-token reference fed the same gate, the KDA and Mamba-2 states land at
9.7e-06 to 2.0e-04 on the v5e and 2.3e-07 to 1.3e-05 on CPU. Against a float64 oracle, Mamba-2's
discretized `dt` lands at 2.2e-04 and 2.4e-07. The matmuls all ask for `HIGHEST`. The error comes
from the float32 `exp` and `log1p` that Mamba-2's softplus runs through. `test_mamba2.py` prints
what each costs on the platform it runs on, 7.4e-08 and 1.4e-07 on CPU. The tests hold both bounds
at 1e-4 on CPU and 5e-4 elsewhere, and hold the shard check at 1e-5 everywhere. KDA's gate has its
own check, against a float64 transcription of the gate the served kernels compute.

[`scan/kda.py`](scan/kda.py) folds the gated delta rule into those `(A, B)` pairs, which covers
Kimi K3 and GLM-5.3-Flash. [`scan/nemotron3_layers.py`](scan/nemotron3_layers.py),
[`scan/inkling_layers.py`](scan/inkling_layers.py) and
[`scan/deepseek_v4_layers.py`](scan/deepseek_v4_layers.py) resolve the irregular layer stacks into
scan groups.

[`scan/mamba2.py`](scan/mamba2.py) folds the Mamba-2 selective scan into those `(A, B)` pairs and
turns the chunk boundary states back into per-token outputs. At Nemotron 3 Super shapes, K=128 and
V=64 with a 128-token chunk, it matches a token-by-token reference to 1.0e-05 per head on CPU and
3.5e-05 on the v5e, on one device and over 8 shards.
