# Roadmap

What's open after 2026-09-29, in the order it gets worked. Each item says what's known, what comes
next and when it's done. The model pages hold the measured runs behind each one.

## 1. Kimi K3: fewer expert decodes

Kimi K3 prefills 1,010 tokens/s with capture off and on, on one `v5p-64` with the routed experts in
MXFP4 ([model page](../models/kimi-k3.md#measured-on-a-v5p-64)). GLM-5.3 prefills 2,685.8 on the
same slice. Every forward decodes each chip's whole expert stack to BF16 before its grouped
matmul. The scripts prefill in 1,024-token passes (`engine_settings` in
`scripts/capture_activations.py`), so a batch of eight 440-token prompts decodes every stack four
times.

Three changes cut the decode. None has run on the model yet:

- Longer passes: `--token-padding 4096 --engine-arg chunked_prefill_size=4096` decodes once a
  batch. The chips peak at 80.48 of 95.73 GiB, and a longer pass needs more activation memory.
- A Pallas decode that writes BF16 straight from the packed codes. A prototype decodes one chip's
  28-expert `wi_0` stack in 0.786 ms, against 1.123 ms for XLA's planar decode, with every value
  bit-exact (host 0 of the `v5p-64`, 2026-09-29).
- A grouped matmul that reads the MXFP4 codes and E8M0 scales directly and never writes a BF16
  stack. [Inferact/tpu-megakernels](https://github.com/Inferact/tpu-megakernels) decodes MXFP4
  tile by tile inside its Kimi K3 kernel, a design reference.

Done when the model page shows a faster prefill on a `v5p-64` with the capture check passing.

Notes: the Pallas timing averages 20 calls after a warm call.

## 2. Inkling: load in minutes, not hours

Inkling's engine takes about 2 hours to load on a `v5p-64`
([model page](../models/inkling.md#measured-on-a-v5p-64)). Its loader asks each host only for the
blocks its chips hold, but every host took in 1.94 to 1.99 TB, the whole checkpoint, in the first
27 minutes of a load, and 6.4 to 7.4 TB over the two loads of one measure (2026-09-29). The mount
keeps range reads out of the gcsfuse file cache, yet a read at byte 0 still pulls a whole file, and
the loader reads every file's header: each host's cache held six whole 19.5 GB files.

The next run mounts Inkling with no file cache, so each read fetches only its range.

Done when an Inkling load on a `v5p-64` takes under 30 minutes and the measure reads the same
throughput.

Notes: the byte counts are each host's received bytes in `/proc/net/dev`, read by the clean-context
run that took the page's throughput. The 30 minutes is a target, not a projection.

## 3. Keep SAE latents alive

The README chain's SAE on Gemma 4 26B-A4B keeps 2,111 of 45,056 latents live at fvu 0.4670
(v5litepod-8, 2026-09-29), because BatchTopK gives a latent that stops firing no gradient.
`sae/train.py` carries the AuxK term (Gao et al., 2024), which trains the dead latents on what the
live ones leave unexplained. At the BatchTopK reference settings, `--auxk-coef 0.03125 --k-aux 512
--dead-batches 5`, the same activations kept 455 live at fvu 0.4657, and training took 876 s
against 337 s. The term stays off by default. In the CPU test it brings 17 of 32 shrunk latents
back in 500 steps.

The next run logs the dead count and the AuxK term's gradient norm every step on those
activations, with the term on and off, to find where the live set shrinks.

Done when the README chain keeps most latents live at the same fvu, with the before and after on
the [model page](../models/gemma4-26b-a4b.md).

## 4. Packed FP4 experts

DeepSeek V4.1-Flash and gpt-oss-120b ship FP4 routed experts, and both ports widen them to BF16 at
load. Kept packed with an in-kernel decode, DeepSeek's routed experts would take about 8.4 GiB a
chip on a `v5p-64` in place of 31.64 GiB, room for more KV and larger batches. The grouped matmul
from item 1 serves all three models.

Done when one model serves with packed experts, passes the capture check, and its page shows HBM
and throughput beside the widened run.

Notes: the 8.4 GiB is projected at 4.25 bits a weight.

## 5. Move the `sglang-jax` pin

Every patch here applies to `eb061d8`. Upstream fixed a bug after it, where batched requests with
the same grammar shared one llguidance matcher (sgl-project/sglang-jax#1704, fixed by #1710 as
294611c). `steering/compare.py`'s judge reads label log probabilities, so nothing here depends on
the fix. Moving the pin means reapplying 21 patches.

At 294611c, `SGL_COMMIT=294611c bash scripts/verify_patches.sh` fails 9 of its 17 checks
(2026-09-28): the capture patch itself, in `engine.py`, `schedule_batch.py`,
`scheduler_output_processor_mixin.py`, `tokenizer_manager.py` and `server_args.py`; the steering
hook; the `qwen3_5` and `deepseek_v3` capture hooks; the multi-host patch; and the Inkling, Kimi K3
and Nemotron 3 model patches. Upstream's 302b081a (#1708) aligns hidden states under data
parallelism with the layout the capture patch uses, a host copy per batch and each rank starting at
`rank * (rows // dp_size)`, so the capture patch can keep upstream's version and carry its own
fixes on top.

Done when `scripts/verify_patches.sh` and `scripts/test_all.sh` pass at a commit at or after
294611c.

## 6. Capture in vLLM on TPU

`tpu-inference`, which vLLM's TPU platform delegates to, has no per-layer hook. Its
`aux_hidden_states` path belongs to speculative decoding, with fixed layer indices and no route to
the API. A vLLM path lets a team keep one engine across GPU and TPU. The `sglang-jax` hook has three
parts to copy: a `layers_to_capture` list read at trace time, one append per layer, and one reshape
at the output.

Done when a vLLM TPU server returns per-layer hidden states that pass `scripts/check_capture.py`.
