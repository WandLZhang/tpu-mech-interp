# Sparse autoencoders

An SAE reads one activation site and writes a sparse code over a learned dictionary. It's two
matmuls with a sparsifying activation between them:

```
f(x)  = sigma(W_enc x + b_enc)
xhat  = W_dec f + b_dec
```

Training uses BatchTopK, which keeps the `k * batch` largest pre-activations across the whole
batch and zeros the rest. The active count is then fixed, so XLA compiles a sparse decoder with
static shapes. An L1 or L0 penalty gives a count that moves every step, and every step recompiles
or pads.

BatchTopK can't serve. It reads the whole batch to decide what fires, and inference has one token.
The conversion fits one JumpReLU threshold per latent from what BatchTopK kept during training:

```
f(x) = JumpReLU_theta(W_enc x + b_enc) = z * (z >= theta)
```

That's elementwise. Serving needs no top-k, no batch and no pre-encoder bias.

The method is Appendix B of the
[Gemma Scope 2 technical report](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/).

| File | Holds |
|---|---|
| `sae.py` | parameters, BatchTopK, sparse decode, sharding, threshold fit |
| `train.py` | optax loop, activation stream, `.npz` output |
| `test_sae.py` | CPU test on synthetic activations with a known answer |

## Four implementation choices

| Choice | Instead of | Why |
|---|---|---|
| Exact `lax.top_k` | `lax.approx_max_k` | The threshold fit reads the kept set as everything at or above the cutoff, and only an exact selection gives that |
| Gather plus segment sum | `f @ w_dec` | The dense decode multiplies through a matrix that's `1 - k/d_sae` zeros. At k=100 and a 256k dictionary that's a factor of 2,560 |
| Sharding the latent axis | data parallel | The encoder matmul runs per shard, and the decode runs in a `shard_map`, where each shard gathers and segment-sums the rows it holds. A step all-gathers the pre-activations for the global selection and all-reduces `[batch, d_model]` partial sums, where a decode without the mesh all-reduces all `k * batch` gathered rows |
| Projecting the decoder gradient | plain Adam | The component parallel to a latent vector only changes its length, and renormalization undoes it. Left in, it distorts the second moment for the directions that do move |

### Recall target

`recall_target=1.0` is the default and runs `lax.top_k`. Set it lower and `batch_topk` runs
`lax.approx_max_k`, which partitions instead of sorting and is faster on TPU for a large flat
array. That kernel only exists on TPU. XLA lowers `approx_top_k` to sort-and-slice on every other
backend, so a CPU run of an approximate selection returns the exact set and predicts nothing about
the TPU.

Approximate selection can miss members of the true top set and keep lower values in their place,
so its cutoff sits lower and admits latents BatchTopK never kept. `threshold_stats` therefore runs
an exact selection whatever `recall_target` says, so the calibration holds its derivation when
training runs approximate.

Both kernels index the flattened `[batch, d_sae]` array in int32. `lax.top_k` takes up to 2**31
pre-activations and `lax.approx_max_k` up to 2**31 - 1, so a 1M dictionary trains at a batch of
2,048 at most on the exact kernel and 2,047 on the approximate one. The report's batch is 4,096.
`batch_topk` refuses a wider batch and names the widest that fits, and `train.py` says so before
it reads a shard.

## The conversion

Each calibration batch gives a window for latent `i`. The bottom is the cutoff BatchTopK applied
to the whole batch, since the latent cleared it. The top is the smallest value BatchTopK kept for
that latent, since anything below that got dropped. The threshold is the geometric mean of the
two.

Over batches, the cutoff is averaged and the kept minimum is taken as a median. A latent that
fires rarely and hard has a long right tail on its kept minimum, and the median ignores it. A
latent that never fires gets `inf`, so JumpReLU holds it off.

The fit runs in float32 and returns float32 thresholds whatever `cfg.dtype` is. Its encoder
matmul asks for `HIGHEST` precision, as every matmul in `sae.py` does. At TPU's default precision
a float32 matmul runs one BF16 pass, and the steering probe that reads a threshold computes in
float32.

`jump_relu` compares inclusively. A bfloat16 pre-activation grid holds 256 values per octave, so
the window collapses to zero width for about 6% of firing latents, and a strict `>` against a
threshold that landed on the grid point would drop values BatchTopK kept.

## Training

Hyperparameters are the ones in the report.

| Setting | Value |
|---|---|
| Learning rate | 7e-5, cosine warmup from 0.1 of that over 1,000 steps |
| Optimizer | Adam, betas (0, 0.999), eps 1e-8 |
| Batch | 4,096 |
| Loss | reconstruction error; `--auxk-coef` adds the AuxK term |
| Input | scaled by one fixed scalar so `E[||x/c||^2] == 1` |
| `W_dec` init | He-uniform, rows rescaled to unit norm |
| `W_enc` init | the transpose of `W_dec`, untied afterward |
| Biases | zero |
| Parameters and Adam state | float32, set by `param_dtype` |
| Forward pass | `dtype`, cast from the master weights on the way in, every matmul at `HIGHEST` |

Keep `param_dtype` at float32. Adam sizes its moments from the parameters, and at bfloat16 the
second moment increments by 0.001 of itself per step, which falls under half an ulp and stalls:
3,000 identical gradients of 1e-3 leave it at 4.77e-7 against a true value of 1e-6, so the
effective learning rate runs 1.45x high for the whole run. A peak learning rate of 7e-5 is also
1.15 ulp of a unit-norm decoder element at `d_model=5376`, which makes most of every update
quantization. `dtype` picks the forward-pass precision on its own and costs nothing.

Raw activation norms move over orders of magnitude between layers and sites. The input scale is
what lets one learning rate work everywhere. Training folds it back into the parameters, so what
ships reads raw activations.

BatchTopK gives a latent that never makes the selection no gradient, so once it stops firing it
stays dead. `--auxk-coef` turns on the AuxK term (Gao et al., 2024), which counts the batches
since each latent last fired, marks it dead after `--dead-batches`, and has the dead latents
reconstruct what the live ones leave unexplained: their `--k-aux` largest pre-activations a token
decode toward the residual, and that error enters the loss at the coefficient. The gradient
reaches the dead latents and no others. The count rides in the optimizer state.

It's off by default. At the BatchTopK reference implementation's settings, `--auxk-coef 0.03125
--k-aux 512 --dead-batches 5`, the README chain's SAE on Gemma 4 26B-A4B kept 455 of 45,056
latents live at fvu 0.4657, against 2,111 at fvu 0.4670 without it, on the same activations
(v5litepod-8, 2026-09-29).

```bash
python3 sae/train.py --activations 'caps/*.npy' --layer 1 --capture-layer 20 \
    --d-model 5376 --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz
```

This recipe doesn't fit a `v5litepod-8`. At `d_model` 5,376, expansion 16, k 100 and the default
batch of 4,096, a compile for that slice on 2026-09-25 needed 20.42 GiB of temporaries against
15.75 GiB per chip, and failed. The same step takes 15.33 GiB a chip at `d_model` 4,096, which
fits. No run here has trained this
recipe on any slice. The README's [Run it](../README.md#run-it) chain trains at `d_model` 2,816,
batch 512 and k 64.

Shards are `[tokens, d_model]` or `[tokens, layers, d_model]`. On a 3-D shard `--layer` picks a
position on the layer axis, and `--capture-layer` names the capture slot the checkpoint records.
Without `--capture-layer` the checkpoint records no slot. On a 2-D shard `--layer` names the slot.

[`../scripts/capture_activations.py`](../scripts/capture_activations.py) writes those shards, plus
a manifest that `--manifest` reads in place of the glob. The manifest carries the shard list in
capture order and `--d-model`, and it maps `--layer` from a capture slot to its position on the
shard:

```bash
python3 sae/train.py --manifest caps/manifest.json --layer 20 \
    --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz
```

This command takes its width from the manifest. No run here has taken it as written. On a
5,376-wide capture it's the recipe above, and it doesn't fit a `v5litepod-8` either.

Slot 20 is the residual stream entering block 20, which is block 19's output. `save` records the
slot in the checkpoint as `capture_layer`, and
[`../steering/from_sae.py`](../steering/from_sae.py) turns it into the `--steering-layer` a
serving hook takes, which is one lower. Capture writes tokens in sequence order and neighboring
tokens correlate hard, so `activation_stream` holds a shuffle pool and draws each batch from a
random position in it.

The pool is sized in bytes, `--shuffle-bytes`, default 4 GiB. Sizing it in tokens is what makes it
blow up at a realistic width: 1M tokens at `d_model=5376` is 21.0 GiB of float32. The pool is one
preallocated array filled a slice at a time off the mmap and shuffled in place, so the peak is one
copy rather than two.

## Test

Runs on CPU with a forced 8-device mesh.

```bash
source .venv/bin/activate
uv pip install -r sae/requirements.txt
python3 sae/test_sae.py
```

The data is synthetic with a known answer: a fixed dictionary of 64 atoms, three positive
coefficients per token, and a little noise. Check 3 measures the learned dictionary against that
generating dictionary. The SAE is `d_model=32`, `d_sae=128`, `k=3`, trained for 4,000 steps.

| Check | Number | Control | Control gives |
|---|---|---|---|
| The documented identities hold | four allclose, each under 1.5e-6, plus an empty window | each identity with one term wrong | 0.94 to 8.6, and 12,213 occurrences in the window |
| Training improves reconstruction | fvu 1.11 to 0.068 | the untrained SAE | fvu 1.11 |
| JumpReLU reproduces BatchTopK | jaccard 0.983, reconstruction difference 0.044, L0 3.00 to 2.96 | thresholds permuted across latents | jaccard 0.49, difference 0.66 |
| | | thresholds rescaled by 0.75 | jaccard 0.75, L0 3.97 |
| | | dead latents opened to 0 | jaccard 0.64, L0 4.61 |
| The learned dictionary recovers the atoms | best cosine mean 0.9944, min 0.9775, 64 of 64 above 0.9 | a random dictionary | mean 0.44 |
| Sharding changes nothing | loss equal to 8 figures | shard contents rotated on the mesh | loss 25.3 against 0.94 |
| The checkpoint records the capture slot, never an axis position | `--layer 1` on a 3-D shard through `--activations` records no slot | `--capture-layer 20` on that shard, `--manifest` at `--layer 20`, and `--layer 20` on a 2-D shard | slot 20 each time, and the first two train the same parameters |
| | | `--layer 20` with `--capture-layer 21` on a 2-D shard | exit 1 |
| `step_flops` counts the compiled step | 1.612e9 against XLA's 1.653e9, ratio 0.975 | the decoder counted as a dense matmul | ratio 1.948 |
| A batch too wide for one selection fails early | at d_sae 1,048,576, batch 2,049 raises on `lax.top_k` and 2,048 on `lax.approx_max_k`; `train.py` exits 1 at batch 4,096 and d_sae 524,320 | the widest batch that fits | 2,048 and 2,047 trace, and d_sae 524,288 passes at batch 4,096 |
| A bfloat16 forward pass trains float32 parameters | after 2 updates, parameters, gradients and Adam state float32, and a first loss of 13.0996 against 13.1077 at float32 | a step that returns bfloat16 parameters, and one that stores the Adam state in bfloat16 | bfloat16 in each |
| The step on the mesh all-reduces partial sums | all-reduces of `[128, 256]`, `[2048]` and `[128, 256]` at batch 128, k=16, d_model 256; loss and gradients within 1e-6 of one device | the step built without the mesh | an all-reduce of `[2048, 256]` |
| Every matmul asks for `HIGHEST` | 7 dot_generals over the training step, the threshold fit, the JumpReLU forward and the bias fold | the encoder matmul with no precision | `DEFAULT` |
| A NaN or an infinity stops `train` | a ValueError for each of the scale sample, a training batch and a calibration batch | the same stream with none | trains, 126 live latents |
| `train.py` counts the scale sample | `--steps 4` on 68 batches exits 1, where training reads 80 | `--steps 4` on 80 batches | trains and calibrates on 64 |
| AuxK trains the dead latents and no others | its gradient reaches all 64 latents marked dead and none of the other 64; a first step with nothing dead equals the plain step bit for bit | every latent marked dead | the gradient reaches all 64 of the other half |
| AuxK brings dead latents back | 32 latents shrunk until 2 fire: 500 AuxK steps bring 17 back, fvu 0.0711 against 0.0761 before the shrink | 500 plain steps | 0 fire, fvu 0.0842 |

Each control has to fail. If one passes, the test fails itself.

Check 0 covers the identities the module documents and nothing else asserts: sparse decode equals
dense decode, the `shard_map` decode on the 8-device mesh equals the one-device decode, folding
the pre-encoder bias leaves pre-activations alone, the projected decoder gradient is orthogonal to
every latent vector, and the BatchTopK window holds no occurrence of the latent it belongs to.

The training loop in the test runs on one device, and the sharding checks cover all eight. XLA's
CPU backend runs every device program as a thread in one pool, and a few hundred collective
executions in a row starve it. On TPU the loop runs on the full mesh.

## Use

```python
from sae import SAEConfig, forward_jumprelu
from train import TrainConfig, activation_stream, manifest_inputs, train

paths, d_model, axis = manifest_inputs("caps/manifest.json", layer=20)
cfg = SAEConfig(d_model=d_model, expansion_factor=16, k=100)
result = train(cfg, TrainConfig(), activation_stream(paths, batch_size=4096, layer=axis))
recon, codes = forward_jumprelu(result.params, x, result.threshold)
```

A sketch: `x` is the activations you encode. At a 5,376-wide capture these settings don't fit a
`v5litepod-8`, for the reason [Training](#training) gives.

`manifest_inputs` maps capture slot 20 to its position on the shards' layer axis, the `layer` that
`activation_stream` slices. A capture of `--layers 10,20,30` holds slot 20 at position 1.
`train` returns parameters that read raw activations, so nothing else has to know the input scale.
