# Mechanistic interpretability on Cloud TPU

Serve large open models on Cloud TPU with [`sglang-jax`](https://github.com/sgl-project/sglang-jax),
capture the residual stream at any layer, train sparse autoencoders on it, and steer the served
model with a learned feature.

## Models

Prefill tokens a second with capture off and on, keeping one layer. Each page gives the command,
the capture check and the run behind its figures.

| Model | Params, total / active | Slice | Capture off | Capture on |
|---|---|---|---|---|
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b.md) | 25.8B / 3.8B | `v5litepod-8` | 13,159 | 9,535 |
| [Gemma 4 31B](models/gemma4-31b.md) | 31.3B | `v5litepod-8` | 6,942 | 5,470 |
| [Qwen3-8B](docs/activation-capture.md#the-whole-chain-measured) | 8.2B | `v5litepod-8` | 32,779 | 16,211 |
| [gpt-oss-120b](models/gpt-oss-120b.md) | 116.8B / 5.1B | `v5p-8` | 7,279 | 5,859 |
| [Nemotron 3 Super](models/nemotron3-super.md) | 123.6B / 12B | `v5p-8` | 3,916 | 2,979 |
| [Nemotron 3 Ultra](models/nemotron3-ultra.md) | 560.5B / 55B | `v5p-64` | 3,095 | 2,218 |
| [GLM-5.3](models/glm5.3.md) | 753B / 40B | `v5p-64` | 2,686 | 2,417 |
| [GLM-5.3-Flash](models/glm5.3-flash.md) | 320B / 18B | `v5p-64` | 1,791 | 1,294 |
| [DeepSeek V4.1-Flash](models/deepseek-v4.1-flash.md) | 749B / 8B | `v5p-64` | 3,415 | 2,570 |
| [Kimi K3](models/kimi-k3.md) | 2.78T / 104B | `v5p-64` | 1,010 | 1,010 |
| [Inkling](models/inkling.md) | 975B / 41B | `v5p-64` | 548 | 518 |

Every captured layer stays within twice the error a BF16 forward makes against float32.
[The roadmap](docs/roadmap.md) holds the open work.

A `v5litepod-8` is 8 v5e chips on one host, a `v5p-8` is 4 v5p chips on one host, and a `v5p-64`
is 32 v5p chips on 8 hosts. A `v5p-64` costs $771 a day on Spot in us-east5 and $3,226 on demand.

## What's where

| Folder | Holds |
|---|---|
| [`models/`](models/) | A page per model: how it serves, how its capture was checked, what it measured |
| [`scripts/`](scripts/) | Every step you run, from getting a slice to measuring throughput, and the float32 references the capture check compares against |
| [`upstream/`](upstream/) | The patches to `sglang-jax`: capture, steering, and six models it doesn't ship |
| [`sae/`](sae/) | Sparse autoencoder training |
| [`steering/`](steering/) | Steering with an SAE feature, judged against a random direction |
| [`docs/`](docs/) | [The capture guide](docs/activation-capture.md), which explains each step, and [the roadmap](docs/roadmap.md) |
| [`scan/`](scan/) | The layer plans and recurrent-layer math the model pages walk through, and a sequence-sharded scan for long prompts. Serving doesn't import it |

Each folder keeps its tests beside its code, as `test_*.py`.

## Run it

The whole chain on Gemma 4 26B-A4B, on one `v5litepod-8`: check the capture, capture 2.2M tokens,
train an SAE, and steer with one of its features. It took 39 minutes on 2026-09-29, 8 of them to
fetch the weights.

You need `gcloud`, a project with the Cloud TPU API on, and quota for 8 v5e chips. Get a slice:

```bash
PROJECT=your-project ACCEL=v5litepod-8 bash scripts/spray_tpu_spot.sh
```

It prints an `ssh` command, which needs a firewall rule that admits port 22 from your address. On
the VM, run `git clone https://github.com/WandLZhang/tpu-mech-interp && cd tpu-mech-interp`, start
`tmux`, and run every line below in that one shell:

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

Step 4 exits 0 when the feature leaves more changed replies coherent than the random direction
does. On 2026-09-25 and again on 2026-09-29 the feature reworded 3 of 4 replies at a tenth of the
stream's norm, all judged coherent, and the random direction broke all 4.
[The capture guide](docs/activation-capture.md#setting-up-a-tpu-vm) lists the warnings that look
like errors and aren't.

Delete the slice when you're done, and check that nothing is left:

```bash
gcloud compute tpus queued-resources delete NAME --zone=ZONE --project=PROJECT --force
gcloud compute tpus queued-resources list --zone=- --project=PROJECT --format="value(name)"
gcloud compute tpus tpu-vm list --zone=- --project=PROJECT --format="value(name)"
```

## Larger models

A model on one host runs [`scripts/measure_model.sh`](scripts/measure_model.sh) after the same
bootstrap. A `v5p-64` model reads its weights from a GCS bucket on all 8 hosts, through its row of
[`scripts/multihost_run.sh`](scripts/multihost_run.sh), whose header lists what it needs first:

```bash
PROJECT=your-project BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE MODEL setup check measure
```

Each model page gives its own line.

## Tests

The tests need no TPU. [`scripts/test_all.sh`](scripts/test_all.sh) runs all 46, most of them the
patched engine on CPU with tiny random weights, and
[`scripts/verify_patches.sh`](scripts/verify_patches.sh) checks that every patch still applies.
From the repo root, in a Python 3.12 venv of their own:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv if it's missing
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --torch-backend=cpu -r scan/requirements.txt -r sae/requirements.txt \
    -r upstream/models/requirements.txt -r scripts/requirements.txt
bash scripts/test_all.sh
```

All 46 passed in 2 hours 8 minutes on a 90-vCPU `c3d-highcpu-90` on 2026-09-29.

Notes: a host whose name runs 64 characters or more fails the two-process steering test inside
gloo; `sudo hostnamectl set-hostname NAME` fixes it. Spot prices change monthly; the two above
come from the Cloud Billing Catalog and the [TPU pricing page](https://cloud.google.com/tpu/pricing)
in September 2026.
