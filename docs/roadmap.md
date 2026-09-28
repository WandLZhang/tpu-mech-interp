# Roadmap

What's open after 2026-09-28, in the order it gets worked. Each item says what's known, what comes
next and when it's done. The model pages hold the measured runs behind each one.

## 1. Kimi K3 prefill

Kimi K3 serves and passes the capture check on one `v5p-64`, but prefill runs at 0.6 tokens/s: 880
tokens in 1,507.55 s at batch 1
([model page](../models/kimi-k3.md#throughput-on-the-v5p-64-06-tokens-a-second)). `tpu-info` read
100% duty cycle and 8.23% TensorCore use on every chip. GLM-5.3 prefills 2,685.8 tokens/s on the
same slice.

Two suspects, neither profiled:

- The routed experts stay packed MXFP4 in HBM, and every forward decodes each chip's whole stack to
  BF16 through a 16-entry table gather just before its grouped matmul.
- KDA prefill runs the engine's `mega` Pallas kernel by default.
  `SGLANG_JAX_KDA_PREFILL_KERNEL=chunked` picks the other one, and nobody has timed the two apart.

A 4-layer cut of the model fits a `v5p-8`, so the next step traces a single 440-token prefill there
with `scripts/serve_throughput.py --profile-dir`, in four arms: packed or decoded-at-load experts,
each with `mega` or `chunked`. The trace splits by layer type, projects to 93 layers, and has to
land near the 753.8 s per prompt the `v5p-64` read before it names a fix. The fix is then a kernel
default, a bit-op decode in place of the gather, or a grouped matmul that reads the MXFP4 codes and
E8M0 scales directly. [Inferact/tpu-megakernels](https://github.com/Inferact/tpu-megakernels)
decodes MXFP4 tile by tile inside its Kimi K3 kernel, which is a design reference for the last.

Done when the model page carries capture-off and capture-on throughput and peak HBM on a `v5p-64`,
with the capture check still passing.

## 2. Inkling

Inkling serves on one `v5p-64`, and every captured layer sits within the BF16 floor on both
prompts. The capture check still exits FAILED, because on the 1,321-token prompt its control reads
2.82x at layer 61, under the 3.0x bar ([model page](../models/inkling.md#measured-on-a-v5p-64)).
At that layer the stream entering layer 62 is close to the stream entering layer 61, so a one-layer
shift can't separate them by 3x. The bar stays at 3.0x.

Throughput and HBM need one `measure` run, about 4.5 hours for two engine loads.

Done when the model page carries capture-off and capture-on throughput and peak HBM.

Notes: the 4.5 hours is projected from the Inkling page's loads of about 2 hours each.

## 3. Capture under data parallelism

With `--dp-size` above 1 the engine lays out each rank's rows in their own padded block, and the
capture patch reads the hidden states as one packed run. A request on rank 1 would get other
requests' rows, so the patch refuses the pairing. That keeps capture at one data-parallel rank,
which holds Kimi K3 to 32 chips: attention splits its 96 heads `tp / dp` ways, and 64 chips at one
rank split them 64 ways. With the fix, Kimi K3 can hold its experts decoded to BF16 on a
`v5p-128` at tp 64 and dp 2, or on a `v6e-256` at dp 8, and gpt-oss-120b stops paying 4x KV on a
`v5p-64`.

The fix starts each rank's rows at `dp_rank * (rows // dp_size)`, the way `tp_worker.py` walks
logprobs, in prefill and in decode.

Done when a CPU test at `dp_size=2` shows every request gets its own rows, across decode steps,
uneven rank batches, a prompt split across prefill passes and two processes, and the refusal comes
out.

## 4. Packed FP4 experts

DeepSeek V4.1-Flash and gpt-oss-120b ship FP4 routed experts, and both ports widen them to BF16 at
load. Kept packed with an in-kernel decode, DeepSeek's routed experts would take about 8.4 GiB a
chip on a `v5p-64` in place of 31.64 GiB, room for more KV and larger batches. The grouped matmul
from item 1 serves all three models.

Done when one model serves with packed experts, passes the capture check, and its page shows HBM
and throughput beside the widened run.

Notes: the 8.4 GiB is projected at 4.25 bits a weight.

## 5. Replay the `v5p-64` pages from a clean start

Every `v5p-64` result came from `scripts/multihost_run.sh`, run by the people who wrote it. Each
single-host page passed a clean-context replay that followed it as written.

Done when one `v5p-64` page passes that replay too.

## 6. Move the `sglang-jax` pin

Every patch here applies to `eb061d8`. Upstream fixed a bug after it, where batched requests with
the same grammar shared one llguidance matcher (sgl-project/sglang-jax#1704, fixed by #1710 as
294611c). `steering/compare.py`'s judge reads label log probabilities, so nothing here depends on
the fix. Moving the pin means reapplying 21 patches.

Done when `scripts/verify_patches.sh` and `scripts/test_all.sh` pass at a commit at or after
294611c.

## 7. Capture in vLLM on TPU

`tpu-inference`, which vLLM's TPU platform delegates to, has no per-layer hook. Its
`aux_hidden_states` path belongs to speculative decoding, with fixed layer indices and no route to
the API. A vLLM path lets a team keep one engine across GPU and TPU. The `sglang-jax` hook has three
parts to copy: a `layers_to_capture` list read at trace time, one append per layer, and one reshape
at the output.

Done when a vLLM TPU server returns per-layer hidden states that pass `scripts/check_capture.py`.

## 8. Keep SAE latents alive

Every SAE run loses most of its latents. The Gemma 4 26B-A4B chain on 2026-09-25 kept 2,023 of
45,056 live. `sae/train.py` has no auxiliary loss and no resampling, so a latent that stops firing
early never comes back. The standard fix is an AuxK loss, which reconstructs the residual error from
the top dead latents at a small weight.

Done when a run of the README chain keeps most latents live at the same fvu, with the before and
after on the model page.
