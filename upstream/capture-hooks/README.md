# Capture hooks

`layers_to_capture` hooks for four `sglang-jax` models that don't have one. Without the hook,
`--enable-return-hidden-states` refuses to start. On a model with no hook, `LogitsProcessor`
would store the final layer's normed `hidden_states`, and the scheduler would cut that into
per-layer slices that aren't layers, so `_setup_hidden_states_capture` in
[`../sglang-jax-877.patch`](../sglang-jax-877.patch) raises instead.

Each patch is a diff against `eb061d8`, and each touches one model family, so they apply in any
order and independently. They go on the tree
[`../../scripts/bootstrap_tpu_vm.sh`](../../scripts/bootstrap_tpu_vm.sh) builds, which is the tree
the TPU VM serves: `sglang-jax-877.patch` through `git am`, then both steering patches. They carry
the rationale but no commit trailers, so use `git apply` and write your own commit message.

| Patch | Model file | Serves |
|---|---|---|
| [`kimi-linear-capture-hook.patch`](kimi-linear-capture-hook.patch) | `models/kimi_linear.py` | `moonshotai/Kimi-Linear-48B-A3B-Instruct` |
| [`qwen3_5-capture-hook.patch`](qwen3_5-capture-hook.patch) | `models/qwen3_5.py` | Qwen3.5 hybrid, text and vision, plus `Qwen/Qwen3.8-27B` |
| [`deepseek-v3-capture-hook.patch`](deepseek-v3-capture-hook.patch) | `models/deepseek_v3.py`, `multimodal/models/kimi_k25/kimi_k25_vl_generation.py` | DeepSeek V2, V3, R1, and Kimi K2.5 VL |
| [`glm4-moe-capture-hook.patch`](glm4-moe-capture-hook.patch) | `models/glm4_moe.py` | GLM-4.5, GLM-4.6 |

```bash
REPO=$PWD
git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
git checkout eb061d8
git -c user.name=you -c user.email=you@example.com am < "$REPO/upstream/sglang-jax-877.patch"
git apply "$REPO/upstream/steering-hook.patch"
git apply "$REPO/upstream/qwen3-steering-hook.patch"
git apply "$REPO/upstream/capture-hooks/kimi-linear-capture-hook.patch"
```

The flag itself comes from [`../sglang-jax-877.patch`](../sglang-jax-877.patch). Apply that too,
or the hook sits idle with an empty `layers_to_capture`. No recorded run typed the block above.
`scripts/verify_patches.sh` applies the same patches in its checks.

## What each patch does

Four parts, the same four `gemma4.py`, `llama.py` and `qwen3.py` already carry.

1. `self.layers_to_capture = []` on the inner model.
2. An append in the layer loop, which turns `for layer in self.layers` into
   `for layer_id, layer in enumerate(self.layers)`.
3. `self.capture_aux_hidden_states = False` on the outer entry class.
4. `aux_hidden_states=` threaded into every `logits_processor` call, on both the tied and untied
   head paths.

The inner model returns `aux_hidden_states` as the second element of its tuple, which matches the
position `gemma4.py` and `qwen3.py` use. The outer class sets it to `None` when the flag is off,
so a run that doesn't ask for hidden states pays nothing.

## The None-residual form

All four models carry `residual` alongside `hidden_states`, and it's `None` on layer 0. The
capture reads the residual stream entering the layer, so it has to tolerate that:

```python
aux_hidden_states.append(
    hidden_states + residual if residual is not None else hidden_states
)
```

EAGLE3 captures layers `[2, n // 2, n - 3]` and never reaches layer 0.
`--enable-return-hidden-states` marks every layer by default, and a
`--return-hidden-states-layers` list can name slot 0, so it does.

## Two models need more than the four parts

**Qwen3.5** nests its text backbone one level deeper. The entry class is
`Qwen3_5MoeForConditionalGeneration`, and the backbone sits at `language_model.model`.
`_setup_hidden_states_capture` reaches it with `getattr(self.model, "model", None)`, so the patch
adds a read-only `model` property that returns the backbone. The test calls that property, writes
`layers_to_capture` through it, and confirms the backbone sees the write. A property lives on the
class, so it adds nothing to the instance dict `nnx.split` walks. The test checks that too.

**DeepSeek V3** has a subclass that overrides `__call__`.
`KimiK25ForConditionalGeneration` unpacks the backbone tuple itself, so it has to move with the
model. The patch updates it, which gives the Kimi K2.5 VL generation path the hook too. The test
runs that subclass's `__call__` against the patched backbone, so a stale unpack shows up as the
`ValueError` a real request would raise.

## Test

```bash
source .venv/bin/activate
uv pip install -r upstream/models/requirements.txt
python3 upstream/capture-hooks/test_capture_hooks.py
```

CPU only, 8 simulated devices. Six checks, each with a negative control that must fail.

1. Every patch applies to the tree `bootstrap_tpu_vm.sh` builds, and every file it touches
   compiles. The file list comes from the patch. Control: the same patch with one context line
   rewritten must be refused.
2. Every file each patch touches carries its part of the hook, read out of its AST: the list, the
   gate, the gate above the `layer(...)` call, `aux_hidden_states` second in the return tuple, the
   flag on every entry class, and the keyword on every `logits_processor` call. Control: the
   unpatched text carries none of them.
3. The patched methods run, and a capture with a NaN in it fails. Controls: the same run with
   every captured value times NaN, which has to fail each comparison; nine mutants of the same
   patched source, each caught; and each mutation operator on the unpatched text, where it has
   to report that it found no target.
4. Without the hook the flag refuses to start. The 877 patch's CPU test starts the real Engine on
   a hookless Qwen2 with the flag. Control: the same checkpoint starts and serves without it.
5. [`../sglang-jax-877.patch`](../sglang-jax-877.patch) connects all four handoffs between
   `req.hidden_states` and `meta_info["hidden_states"]`: the scheduler puts it into
   `output_hidden_states`, `BatchTokenIDOut` carries it, the detokenizer forwards it into
   `BatchStrOut`, and `tokenizer_manager` reads it into `meta_info`. Control: the unpatched tree
   carries three of the four and breaks the one the patch owns.
6. The 877 patch passes its own CPU test, `test/srt/test_return_hidden_states_cpu.py`, which runs
   the real Engine with a tiny random Qwen3: a cached prefix, a mixed batch, logprobs beside
   hidden states, HTTP `/generate`, both retraction routes, decode steps inside another
   request's prefill chunks under the overlap scheduler, a request to a server without the flag,
   the settings that refuse to start, and `--return-hidden-states-layers`: the reply shape, the
   slot order, each refusal and the command-line syntax. Each test carries its own control, and
   a test that goes missing fails the check. The patched tree also has to pass upstream's unit
   tests for the scheduler, batch and tokenizer manager code the patch changes. Control: with
   `process_batch_result_prefill` reading `server_args` on every batch, the chunked-ownership
   tests fail.

Checks 4 and 6 start the Engine, so the venv needs the packages `sgl_jax` imports at startup,
which [`../models/requirements.txt`](../models/requirements.txt) lists. The CPU test's process
gets the checkout's `python/` ahead of your `PYTHONPATH`, so packages on `PYTHONPATH` import too.
The six checks take about seven minutes.

`SGL_COMMIT` picks the `sglang-jax` commit, `eb061d8` by default. Set `SGLANG_JAX_REPO` to a local
checkout that holds it to skip the fetch.

### How check 3 runs the shipped code

The test applies the patch, pulls `__call__` out of the patched file by AST, strips the signature
annotations, and compiles the body as written. It binds that function to toy pre-norm layers and
runs it under `jit` on a token-sharded input. The layers mirror `rmsnorm_forward`: they add the
residual in the layer dtype and reduce in float32, the way every RMSNorm in these four models
does. So the code under test is the patch, and only the surrounding block is a stand-in.

The reference carries the residual stream as one running value in float64, rather than the
`(hidden, residual)` pair the models split it into. The two paths share no operand order, so the
tolerance measures rounding rather than agreeing with itself.

| Captured, 12 layers | Relative error |
|---|---|
| float32 | 2.1e-07 |
| bfloat16 | 9.2e-03 |

bfloat16 is what all four models default to. The gate for a mutant is float32, where the signal
clears the rounding floor by five orders of magnitude.

The check also runs the gate at three settings: empty, which must capture nothing; EAGLE3's
`[2, n // 2, n - 3]`; and every layer. It calls each entry class's `__call__` with the flag off
and on, and reads what the logits processor received.

The nine mutants: the append moved below the `layer(...)` call, the gate inverted to `not in`,
the None check dropped, the None check inverted, `aux_hidden_states` dropped from the return tuple,
`aux_hidden_states` moved to the end of it, the `capture_aux_hidden_states` gate removed, the
`aux_hidden_states=` keyword removed, and the append removed with the gate kept.

Every comparison passes only on `rel < tol`. A NaN in a capture makes `rel` NaN, and NaN compares
False with everything, so the capture fails. The NaN control reruns the three comparisons with
every captured value times NaN, and all three have to fail.

Each mutation operator finds its target through a locator that raises when the target isn't
there, and the run counts that as a failure. The last control points each operator at the
unpatched text, and all nine have to report that they found no target.
