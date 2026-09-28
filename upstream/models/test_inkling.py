# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check `inkling.py` against the HuggingFace `transformers` reference on CPU, through the engine.

The patch in this directory adds `python/sgl_jax/srt/models/inkling.py` and the
engine pieces it needs to `sglang-jax`. This script builds the tree the TPU VM
serves from through `scripts/cpu_engine.build_tree`, `sglang-jax-877.patch`
through `git am` and both steering patches after it, applies this patch on top,
and runs the engine's own serving path on tiny random checkpoints:
`JAXModelLoader`, `ModelRunner` with its Explicit mesh and the pools it builds
(`SWAKVPool`, `ShortConvStatePool`, `HybridReqToTokenPool`), the startup
precompile, and batches `ScheduleBatch` builds over the `SWAChunkCache` the
scheduler picks under `--disable-radix-cache`. Every engine logit here comes out
of `ModelWorker.forward_batch_generation`, and every reference number out of
`transformers`. The batches carry the scheduler's `SpeculativeAlgorithm.NONE`,
so a plain prefill or decode step runs a program the startup precompile built,
and the test counts backend compiles to hold it there. The flags Inkling can't
serve go to a worker on a copy of the checkpoint without its weights, so each
refusal has to come before the loader.

`transformers` writes the checkpoints: `InklingForConditionalGeneration` with
random weights and `save_pretrained`, which lays the tensors out the way the
published checkpoint does, fused gate and up rows interleaved. The reference
logits come from `transformers.InklingForCausalLM` holding the same language
model and head. The multimodal class isn't the reference: in transformers 5.17
its `InklingModel.forward` applies `embed_norm` and then calls the text model,
which applies it again. SGLang's PyTorch model applies it once, and so does
this port.

The engine path imports `pybase64` and `llguidance`, beside the packages the
other model tests need. Without them the engine sections fail at once, and the
rest still run.

    SGLANG_JAX_REPO=/path/to/sglang-jax python3 upstream/models/test_inkling.py

Without `SGLANG_JAX_REPO` the script clones `sgl-project/sglang-jax` into a
scratch directory. `SGL_COMMIT` picks the commit, eb061d8 by default, as it
does for every gate.
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Eight simulated devices. The main run takes two, so the tensor axis is real
# and every `kernel_axes` and `out_sharding` annotation has to hold. The
# replication run takes all eight, past both attention kinds' KV head counts.
# The device count goes in beside any flag XLA_FLAGS already holds, where
# setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()
warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "inkling-model.patch")
# `scripts/cpu_engine.py` builds the tree bootstrap_tpu_vm.sh builds.
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "scripts"))
import cpu_engine  # noqa: E402

# Every architecture flag the published `config.json` sets, at toy widths. The
# flags that carry no number are here so a divergence in how either side reads
# them shows up, even where both currently agree. Layer 0 is local and dense,
# layer 1 global and routed, layers 2 and 3 local and routed.
CONFIG = dict(
    vocab_size=48,
    unpadded_vocab_size=40,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=128,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=4,
    swa_head_dim=128,
    sliding_window_size=3,
    d_rel=4,
    rel_extent=5,
    log_scaling_n_floor=2,
    log_scaling_alpha=0.75,
    local_layer_ids=[0, 2, 3],
    dense_mlp_idx=1,
    dense_intermediate_size=96,
    intermediate_size=96,
    moe_intermediate_size=32,
    n_routed_experts=6,
    num_experts_per_tok=2,
    n_shared_experts=2,
    route_scale=8.0,
    conv_kernel_size=4,
    use_sconv=True,
    logits_mup_width_multiplier=2.0,
    rms_norm_eps=1e-6,
    use_embed_norm=True,
    shared_expert_sink=True,
    use_gate_bias=True,
    gate_activation="sigmoid",
    norm_after_topk=True,
    use_global_scale=True,
    q_bias=False,
    o_bias=False,
    final_logit_softcapping=None,
    model_max_length=64,
    num_mtp_layers=4,
    mtp_local_layer_ids=[0, 2],
    chain_hidden_post_norm=False,
    mtp_hidden_states_first=True,
)
# Inkling at --tp-size 32 runs four copies of each of its 8 global KV heads and
# two of each of its 16 local ones. Eight query heads on eight devices give this
# config the same ratios over its 2 global and 4 local KV heads.
REPLICATION_CONFIG = dict(CONFIG, num_attention_heads=8, swa_num_attention_heads=8)
TP = 2
REPLICATION_TP = 8
# --chunked-prefill-size. The chunked prompt below is longer, so it runs as two
# passes, and the second one opens on the windows and the KV the first left.
CHUNK = 8
MAX_ABS_ERROR = 1e-3
MIN_CORRELATION = 1.0 - 1e-9
# A control must move the logits by more than this to count as detected. It sits
# about a hundred times above the worst clean error.
CONTROL_FLOOR = 1e-3
# The two published repositories, for the checkpoint-key check.
PUBLISHED_REPOS = ("thinkingmachines/Inkling", "thinkingmachines/Inkling-Small")


def patched_checkout(scratch: str) -> str:
    """The tree the TPU VM serves from, with this patch applied on top.

    `cpu_engine.build_tree` runs the steps `bootstrap_tpu_vm.sh` runs: it checks
    out `SGL_COMMIT`, takes `sglang-jax-877.patch` through `git am` and both
    steering patches after it.
    """
    target = cpu_engine.build_tree(
        os.path.join(scratch, "sglang-jax"), os.environ.get("SGLANG_JAX_REPO") or None
    )
    subprocess.run(["git", "-C", target, "apply", PATCH], check=True)
    return os.path.join(target, "python")


def install(path: str) -> None:
    sys.path.insert(0, path)


# --- the reference and its checkpoint -------------------------------------


def write_checkpoint(text_config: dict, path: str):
    """A random `InklingForConditionalGeneration` written by `save_pretrained`.

    Returns the `transformers.InklingForCausalLM` that holds the same language
    model and head, which is the reference every comparison reads.
    """
    import torch
    from transformers.models.inkling import (
        InklingConfig,
        InklingForCausalLM,
        InklingForConditionalGeneration,
    )

    config = InklingConfig(
        text_config=dict(text_config),
        vision_config=dict(
            patch_size=2, temporal_patch_size=2, num_channels=3, num_hidden_layers=1
        ),
        audio_config=dict(n_mel_bins=4, mel_vocab_size=8),
    )
    torch.manual_seed(0)
    multimodal = InklingForConditionalGeneration(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, param in multimodal.named_parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.25)
        for _, buffer in multimodal.named_buffers():
            if buffer.dtype.is_floating_point:
                buffer.copy_(torch.randn(buffer.shape, generator=generator) * 0.25)
    multimodal.save_pretrained(path)

    reference = InklingForCausalLM(multimodal.config.text_config).to(torch.float32).eval()
    reference.model.load_state_dict(multimodal.model.language_model.state_dict())
    reference.lm_head.load_state_dict(multimodal.lm_head.state_dict())
    return reference


def reference_logits(reference, ids) -> "np.ndarray":
    """`[len(ids), unpadded_vocab_size]` logits for one sequence."""
    import torch

    with torch.no_grad():
        return reference(input_ids=torch.tensor([list(ids)]), use_cache=False).logits[0].numpy()


def reference_layer_inputs(reference, ids) -> list:
    import torch

    with torch.no_grad():
        output = reference.model(
            input_ids=torch.tensor([list(ids)]), output_hidden_states=True, use_cache=False
        )
    return [state[0].numpy() for state in output.hidden_states]


def log_softmax(logits):
    import numpy as np

    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


# --- the engine -------------------------------------------------------------


class Step:
    """What one forward returned, cut to the real requests."""

    def __init__(self, model_worker_batch, output, next_ids):
        import numpy as np

        real = model_worker_batch.real_bs
        self.batch = model_worker_batch
        self.output = output
        self.logits = np.asarray(output.next_token_logits)[:real]
        self.next_ids = np.asarray(next_ids)[:real]


class Compiles:
    """Backend compiles in this process, counted through `jax.monitoring`."""

    count = 0
    listening = False

    @classmethod
    def start(cls) -> None:
        import jax

        if not cls.listening:
            jax.monitoring.register_event_duration_secs_listener(cls.listen)
            cls.listening = True

    @classmethod
    def listen(cls, event, duration, **kwargs):
        from jax._src.dispatch import BACKEND_COMPILE_EVENT

        if event == BACKEND_COMPILE_EVENT:
            cls.count += 1


class Forward:
    """One served forward: its label, its mode, and what it compiled."""

    def __init__(self, label, model_worker_batch, compiles):
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

        self.label = label
        self.mode = model_worker_batch.forward_mode.name
        self.compiles = compiles
        # The precompile's dummies ask for no logprobs and no hidden states. A
        # batch that does compiles its own variant on first use, for any model.
        self.plain = (
            not model_worker_batch.return_logprob
            and model_worker_batch.capture_hidden_mode == CaptureHiddenMode.NULL
        )


class Serving:
    """One `ModelWorker` and the request machinery the scheduler runs around it.

    The worker, the tree cache and the batches are the engine's own. What the
    `Scheduler` process adds on top, turning a queue into rounds, is the loop in
    `extend` and `decode`, which calls the same methods in the same order.
    """

    def __init__(self, model_path: str, tp: int, *, precompile: bool = True, **overrides):
        from sgl_jax.srt.managers.tp_worker import ModelWorker
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.server_args import ServerArgs
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        Compiles.start()
        self.forwards: list[Forward] = []

        settings = dict(
            model_path=model_path,
            device="cpu",
            tp_size=tp,
            dtype="float32",
            skip_tokenizer_init=True,
            attention_backend="native",
            disable_radix_cache=True,
            max_running_requests=4,
            max_prefill_tokens=CHUNK,
            chunked_prefill_size=CHUNK,
            # A small pool with a quarter of it for the sliding window, so the
            # window's free list wraps onto slots a finished request held and
            # the full-to-window slot mapping stops being the identity.
            max_total_tokens=64,
            swa_full_tokens_ratio=0.25,
            precompile_token_paddings=[CHUNK],
            precompile_bs_paddings=[1, 2, 4],
            disable_overlap_schedule=True,
            enable_return_hidden_states=True,
            random_seed=0,
        )
        settings.update(overrides)
        self.server_args = ServerArgs(**settings)
        # Scheduler.__init__ builds its mesh and its algorithm the same way.
        # The precompile's dummies carry SpeculativeAlgorithm.NONE too, so a
        # batch holding None would compile a second program.
        self.mesh = create_device_mesh(
            ici_parallelism=[1, tp], dcn_parallelism=[1, 1], device_indexes=list(range(tp))
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            self.server_args.speculative_algorithm
        )
        self.worker = ModelWorker(server_args=self.server_args, mesh=self.mesh)
        self.runner = self.worker.model_runner
        self.model = self.runner.model
        self.precompile_seconds = None
        if precompile:
            started = time.time()
            self.worker.run_precompile()
            self.precompile_seconds = time.time() - started
        self.tree_cache = build_kv_cache(
            server_args=self.server_args,
            model_config=self.worker.model_config,
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            page_size=self.server_args.page_size,
            is_hybrid=self.runner.is_hybrid,
            is_hybrid_recurrent=isinstance(self.runner.req_to_token_pool, HybridReqToTokenPool),
            sliding_window_size=self.runner.sliding_window_size,
            tp_size=tp,
            spec_algorithm=self.spec_algorithm,
            mesh=self.mesh,
        )
        self.paddings = self.worker.get_precompile_paddings()

    def request(self, rid: str, ids, *, logprobs: bool = False, hidden: bool = False):
        from sgl_jax.srt.managers.schedule_batch import Req
        from sgl_jax.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[int(i) for i in ids],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
            return_logprob=logprobs,
            top_logprobs_num=5 if logprobs else 0,
            vocab_size=int(CONFIG["vocab_size"]),
            dp_rank=0,
            return_hidden_states=hidden,
        )
        # handle_generate_request's rule: a logprob request scores from its
        # start, every other request from its last prompt token.
        req.logprob_start_len = 0 if logprobs else len(ids) - 1
        return req

    def prepare_extend(self, reqs, *, chunk: int | None = None):
        """One prefill round, the way `get_new_batch_prefill` builds it.

        A request that already holds a prefix continues it, the way the
        scheduler's chunked request does. `chunk` truncates a request the way
        `PrefillAdder` does at `--chunked-prefill-size`.
        """
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        chunked = None
        for req in reqs:
            if req.kv_committed_len:
                req.init_next_round_input()
            else:
                req.init_next_round_input(self.tree_cache)
            if chunk is not None and req.extend_input_len > chunk:
                req.extend_input_len = chunk
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + chunk]
                req.is_chunked += 1
                chunked = req
        batch = ScheduleBatch.init_new(
            reqs=[list(reqs)],
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.worker.model_config,
            enable_overlap=False,
            dp_size=1,
            chunked_reqs=[chunked],
            mesh=self.mesh,
            spec_algorithm=self.spec_algorithm,
        )
        batch.prepare_for_extend()
        return batch, self.worker_batch(batch)

    def worker_batch(self, batch):
        return batch.get_model_worker_batch(*self.paddings, self.server_args.page_size, False)

    def forward(self, model_worker_batch, label: str) -> Step:
        import numpy as np

        before = Compiles.count
        output, next_ids, _ = self.worker.forward_batch_generation(model_worker_batch)
        step = Step(model_worker_batch, output, np.asarray(next_ids))
        self.forwards.append(Forward(label, model_worker_batch, Compiles.count - before))
        return step

    def finish_extend(self, reqs, step: Step) -> None:
        """What `process_batch_result_prefill` does to each request."""
        for req, token in zip(reqs, step.next_ids):
            if req.is_chunked > 0:
                req.is_chunked -= 1
                self.tree_cache.cache_unfinished_req(req)
            else:
                req.output_ids.append(int(token))

    def extend(self, reqs, *, chunk: int | None = None):
        batch, model_worker_batch = self.prepare_extend(reqs, chunk=chunk)
        step = self.forward(model_worker_batch, "prefill " + "+".join(req.rid for req in reqs))
        self.finish_extend(reqs, step)
        return batch, step

    def decode(self, batch) -> Step:
        """One decode step on the requests `batch` last extended or decoded."""
        import numpy as np

        info = batch.reqs_info[0]
        info.output_ids = np.array([req.output_ids[-1] for req in info.reqs], dtype=np.int32)
        batch.prepare_for_decode()
        label = "decode " + "+".join(req.rid for req in info.reqs)
        step = self.forward(self.worker_batch(batch), label)
        for req, token in zip(info.reqs, step.next_ids):
            req.output_ids.append(int(token))
        return step

    def release(self, req) -> None:
        from sgl_jax.srt.mem_cache.common import release_kv_cache

        release_kv_cache(req, self.tree_cache)

    def trial(self, model_worker_batch, *, memory_pools=None, metadata=None, attn_backend=None):
        """A forward that writes nothing back: the runner's model on its pools.

        It runs the model the runner holds, as it stands now, through a jit of
        its own that returns every prompt row's logits and the pool updates
        instead of writing them. A control can break one part of the model and
        run the same batch again.
        """
        import jax
        from flax import nnx

        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

        backend = self.runner.attn_backend
        backend.forward_metadata = (
            backend.get_forward_metadata(model_worker_batch) if metadata is None else metadata
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.runner)
        if attn_backend is not None:
            forward_batch.attn_backend = attn_backend
        pools = self.runner.memory_pools if memory_pools is None else memory_pools
        with jax.set_mesh(self.mesh):
            graphdef, state = nnx.split(self.model)
            return jitted_all_rows()(graphdef, state, forward_batch, pools)


def all_rows(model, forward_batch, memory_pools):
    """Logits for every token of the batch, and the conv buffers the step would write."""
    conv_pool = model.conv_pool(memory_pools, forward_batch)
    hidden, _, _, conv_buffers, _, _ = model.model(
        forward_batch, memory_pools.token_to_kv_pool, conv_pool
    )
    logits = model.logits_processor._get_logits(
        hidden / model.mup_width_multiplier, model.lm_head
    )
    return logits, conv_buffers


_JITTED_ALL_ROWS = []


def jitted_all_rows():
    """`all_rows` under jit, keyed on the model's graph, so a changed weight reuses it."""
    import jax
    from flax import nnx

    if not _JITTED_ALL_ROWS:

        def run(graphdef, state, forward_batch, memory_pools):
            return all_rows(nnx.merge(graphdef, state), forward_batch, memory_pools)

        _JITTED_ALL_ROWS.append(jax.jit(run))
    return _JITTED_ALL_ROWS[0]


def retire() -> None:
    """Free a finished worker, compiled programs included.

    A serving process holds one runner. This one builds several, and a cache
    that outlives its runner compares the next runner's backend against the old
    one's, down to the host array of window slots the runner hangs on it, which
    has no equality to compare. `donation_vector` keeps a `functools` cache of
    its own that `clear_caches` leaves alone.
    """
    import jax
    from jax._src import api_util

    gc.collect()
    jax.clear_caches()
    api_util.donation_vector.cache_clear()


def statistics(left, right):
    import numpy as np

    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    return float(np.max(np.abs(left - right))), float(np.corrcoef(left, right)[0, 1])


def report(label, left, right, failures):
    error, correlation = statistics(left, right)
    print(f"  {label:<34} max abs error {error:.3e}   correlation {correlation:.12f}")
    # A comparison with NaN is False, so the gate asks for a pass. A NaN error or
    # correlation then fails.
    if not (error <= MAX_ABS_ERROR and correlation >= MIN_CORRELATION):
        failures.append(f"{label}: max abs error {error:.3e}, correlation {correlation:.12f}")
    return error


def note(label, ok, detail, failures):
    """One pass/fail line, in the shape the numeric reports use."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def control(label, moved, failures):
    detected = moved > CONTROL_FLOOR
    print(f"  {label:<44} moves logits by {moved:.3e}  {'detected' if detected else 'MISSED'}")
    if not detected:
        failures.append(f"control {label!r} changed nothing, so the check proves nothing")


# --- the engine run at --tp-size 2 ------------------------------------------


def check_serving(serving, reference, failures):
    """Prefill, decode, layer inputs, prompt logprobs and a packed prefill, against `transformers`."""
    import numpy as np

    from sgl_jax.srt.managers.tp_worker import _iter_padded_input_logprob_reqs
    from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, SWAKVPool
    from sgl_jax.srt.mem_cache.short_conv_state_pool import ShortConvStatePool

    keep = CONFIG["unpadded_vocab_size"]
    runner = serving.runner
    pools = runner.memory_pools
    note(
        "the runner builds its own pools",
        isinstance(pools.token_to_kv_pool, SWAKVPool)
        and isinstance(pools.conv_state_pool, ShortConvStatePool)
        and isinstance(runner.req_to_token_pool, HybridReqToTokenPool)
        and type(serving.tree_cache).__name__ == "SWAChunkCache",
        f"{type(pools.token_to_kv_pool).__name__}, {type(pools.conv_state_pool).__name__},"
        f" {type(runner.req_to_token_pool).__name__}, {type(serving.tree_cache).__name__}",
        failures,
    )
    note(
        "the startup precompile runs",
        serving.precompile_seconds is not None,
        f"EXTEND and DECODE buckets in {serving.precompile_seconds:.0f} s",
        failures,
    )

    rng = np.random.default_rng(3)

    # One prompt, then decoded.
    prompt = rng.integers(0, keep, size=7).tolist()
    req = serving.request("prompt", prompt)
    batch, step = serving.extend([req])
    want = reference_logits(reference, prompt)
    print("prefill, one sequence of", len(prompt), "tokens")
    report("logits", step.logits[0, :keep], want[-1], failures)
    note(
        "the padded head rows sit at the floor",
        bool(np.all(step.logits[0, keep:] < -1e30)),
        f"{CONFIG['vocab_size'] - keep} rows past unpadded_vocab_size",
        failures,
    )

    print("decode, five steps on top of the prefill")
    sequence = list(prompt) + [req.output_ids[-1]]
    greedy = [int(np.argmax(want[-1])) == req.output_ids[-1]]
    for index in range(5):
        step = serving.decode(batch)
        want = reference_logits(reference, sequence)
        report(f"step {index} logits", step.logits[0, :keep], want[-1], failures)
        greedy.append(int(np.argmax(want[-1])) == int(step.next_ids[0]))
        sequence.append(int(step.next_ids[0]))
    note(
        "the engine's greedy tokens match transformers",
        all(greedy),
        f"{sum(greedy)} of {len(greedy)}: {sequence[len(prompt):]}",
        failures,
    )
    serving.release(req)

    # The same prompt with every layer captured.
    req = serving.request("captured", prompt, hidden=True)
    _, step = serving.extend([req])
    print("layer inputs, the same sequence through --enable-return-hidden-states")
    for index, (got, expect) in enumerate(
        zip(
            np.asarray(step.output.hidden_states)[: len(prompt)].reshape(
                len(prompt), CONFIG["num_hidden_layers"], CONFIG["hidden_size"]
            ).transpose(1, 0, 2),
            reference_layer_inputs(reference, prompt),
        )
    ):
        kind = "local" if index in CONFIG["local_layer_ids"] else "global"
        mlp = "dense" if index < CONFIG["dense_mlp_idx"] else "routed"
        report(f"layer {index} input ({kind}/{mlp})", got, expect, failures)
    serving.release(req)

    # The same prompt scored from its first token. Prompt logprobs normalize
    # over the real vocabulary, as the sampled token does, and no padded row
    # reaches their top-k.
    req = serving.request("scored", prompt, logprobs=True)
    _, step = serving.extend([req])
    want = reference_logits(reference, prompt)
    token_logprobs = np.asarray(step.output.input_token_logprobs)
    ((_, offset, length),) = list(
        _iter_padded_input_logprob_reqs(step.batch, len(token_logprobs))
    )
    scored = token_logprobs[offset : offset + length - 1]
    expected = log_softmax(want)[np.arange(len(prompt) - 1), prompt[1:]]
    print("prompt logprobs, the same sequence")
    report("prompt logprobs", scored, expected, failures)
    top_ids = step.output.input_top_logprobs_idx[0]
    note(
        "prompt top-k ids stay in the real vocabulary",
        all(i < keep for row in top_ids for i in row),
        f"{sum(len(row) for row in top_ids)} ids, largest {max(max(row) for row in top_ids)}",
        failures,
    )
    serving.release(req)

    # Two sequences packed into one prefill. The convolution, the mask and the
    # relative bias all have to stop at the boundary between them.
    first = rng.integers(0, keep, size=3).tolist()
    second = rng.integers(0, keep, size=CHUNK - 3).tolist()
    reqs = [serving.request("first", first), serving.request("second", second)]
    _, step = serving.extend(reqs)
    print(f"packed prefill, {len(first)} tokens then {len(second)}")
    for row, (label, ids) in enumerate((("first", first), ("second", second))):
        report(
            f"{label} sequence logits",
            step.logits[row, :keep],
            reference_logits(reference, ids)[-1],
            failures,
        )
    for finished in reqs:
        serving.release(finished)

    # NEGATIVE CONTROL: every comparison in this file goes through report(), and it
    # has to fail a NaN. One NaN in the second sequence's row stands in for a request
    # boundary that leaks one.
    poisoned = np.array(step.logits[1, :keep], dtype=np.float64)
    poisoned[0] = np.nan
    caught = []
    report("control: one NaN logit", poisoned, reference_logits(reference, second)[-1], caught)
    print(f"      control (one NaN logit)  -> {'detected' if caught else 'NOT DETECTED'}")
    if not caught:
        failures.append("the logit gate passes a NaN, so no comparison here can catch one")


def check_chunked(serving, reference, failures):
    """A prompt split across prefill passes, then decode, on recycled window slots.

    The second pass has to open every convolution on the window the first pass
    left in the request's `ShortConvStatePool` slot, and every sliding-window
    layer has to read its KV at the slots `SWATokenToKVPoolAllocator` mapped,
    which by now are slots earlier requests held.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sgl_jax.srt.layers.attention.native_backend import NativeAttentionMetadata
    from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    keep = CONFIG["unpadded_vocab_size"]
    rng = np.random.default_rng(5)
    allocator = serving.runner.token_to_kv_pool_allocator

    # Earlier requests come and go until the window pool, a quarter the size of
    # the full one, has handed out every slot once. From there its free list
    # holds recycled slots while the full pool's still holds fresh ones.
    for index in range(-(-(allocator.size_swa + 1) // CHUNK)):
        filler = serving.request(f"filler-{index}", rng.integers(0, keep, size=CHUNK).tolist())
        serving.extend([filler])
        serving.release(filler)

    prompt = rng.integers(0, keep, size=CHUNK + 5).tolist()
    want = reference_logits(reference, prompt)
    req = serving.request("chunked", prompt)
    print(f"chunked prompt, {len(prompt)} tokens at --chunked-prefill-size {CHUNK}")

    _, step = serving.extend([req], chunk=CHUNK)
    report("first pass, last row", step.logits[0, :keep], want[CHUNK - 1], failures)

    batch, worker_batch = serving.prepare_extend([req], chunk=CHUNK)
    full_slots = serving.runner.req_to_token_pool.req_to_token[req.req_pool_idx, : len(prompt)]
    window_slots = allocator.full_to_swa_index_mapping[full_slots]
    # Slot 0 marks a position the window left behind, so only live slots count.
    live = window_slots > 0
    note(
        "the live window slots differ from the full slots",
        bool(np.any(window_slots[live] != full_slots[live])),
        f"full {full_slots.tolist()} -> window {window_slots.tolist()}",
        failures,
    )

    # The same second pass as a trial, once as served and once with each
    # of the two things the pass depends on broken.
    served, _ = serving.trial(worker_batch)
    served_row = np.asarray(served)[len(prompt) - CHUNK - 1, :keep]
    report("second pass as a trial", served_row, want[-1], failures)

    pools = serving.runner.memory_pools
    with jax.set_mesh(serving.mesh):
        zeroed = jax.tree.map(jnp.zeros_like, pools.conv_state_pool)
    restarted, _ = serving.trial(
        worker_batch,
        memory_pools=MemoryPools(token_to_kv_pool=pools.token_to_kv_pool, conv_state_pool=zeroed),
    )
    moved_windows = float(
        np.max(np.abs(np.asarray(restarted)[len(prompt) - CHUNK - 1, :keep] - served_row))
    )

    backend = serving.runner.attn_backend
    mapped = backend.get_forward_metadata(worker_batch)
    raw = ForwardBatch.init_new(worker_batch, serving.runner).cache_loc
    misread, _ = serving.trial(
        worker_batch,
        metadata=NativeAttentionMetadata(
            swa_cache_loc=raw, swa_out_cache_loc=mapped.swa_out_cache_loc
        ),
    )
    moved_read = float(
        np.max(np.abs(np.asarray(misread)[len(prompt) - CHUNK - 1, :keep] - served_row))
    )

    step = serving.forward(worker_batch, "prefill chunked, second pass")
    serving.finish_extend([req], step)
    report("second pass, last row", step.logits[0, :keep], want[-1], failures)

    sequence = list(prompt) + [req.output_ids[-1]]
    for index in range(3):
        step = serving.decode(batch)
        want = reference_logits(reference, sequence)
        report(f"step {index} logits", step.logits[0, :keep], want[-1], failures)
        sequence.append(int(step.next_ids[0]))
    serving.release(req)

    print("  controls on the second pass")
    control("windows restarted from zeros", moved_windows, failures)
    control("window KV read at the full-pool slots", moved_read, failures)


def check_precompiled(serving, label, failures):
    """Every plain batch served so far ran a program the startup precompile built.

    The scheduler's batches and the precompile's dummies have to flatten to the
    same tree, with the same shapes and shardings, or the first request of each
    shape compiles a second program: dropped `recurrent_indices`, window slots
    placed differently, an algorithm of None where the scheduler passes NONE.
    """
    plain = [forward for forward in serving.forwards if forward.plain]
    modes = {mode: sum(forward.mode == mode for forward in plain) for mode in ("EXTEND", "DECODE")}
    compiled = [forward for forward in plain if forward.compiles]
    note(
        f"{label} runs on the precompiled programs",
        all(modes.values()) and not compiled,
        f"{counted(modes['EXTEND'], 'prefill')} and {counted(modes['DECODE'], 'decode step')},"
        f" {counted(sum(forward.compiles for forward in plain), 'backend compile')}"
        + (f", the first in {compiled[0].label}" if compiled else ""),
        failures,
    )


def counted(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def control_precompiled(serving, failures):
    """NEGATIVE CONTROL: a batch that flattens differently has to show up as a compile.

    A test that builds its batches with `spec_algorithm=None` puts every forward
    it compares on a program of its own, and the count has to see it.
    """
    import numpy as np

    req = serving.request("stray-algorithm", np.arange(1, CHUNK - 2).tolist())
    _, worker_batch = serving.prepare_extend([req])
    worker_batch.spec_algorithm = None
    step = serving.forward(worker_batch, "prefill stray-algorithm")
    serving.finish_extend([req], step)
    serving.release(req)
    compiles = serving.forwards[-1].compiles
    detected = compiles > 0
    print(
        f"      control (a prefill with spec_algorithm None): {compiles} backend compiles"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        failures.append("a batch unlike the precompile's compiled nothing, so the count proves nothing")


# --- what the cache manager holds -------------------------------------------


def expected_conv_widths(config) -> list[list[tuple[str, int, int]]]:
    """The four windows per layer, written out from the config by hand."""
    tokens = config["conv_kernel_size"] - 1
    spec = []
    for index in range(config["num_hidden_layers"]):
        local = index in config["local_layer_ids"]
        heads = config["swa_num_key_value_heads"] if local else config["num_key_value_heads"]
        head_dim = config["swa_head_dim"] if local else config["head_dim"]
        kv = heads * head_dim
        spec.append(
            [
                ("k_sconv", kv, tokens),
                ("v_sconv", kv, tokens),
                ("attn_sconv", config["hidden_size"], tokens),
                ("mlp_sconv", config["hidden_size"], tokens),
            ]
        )
    return spec


def check_conv_state(serving, failures):
    """`get_conv_state_spec()`, the pool the runner built from it, and what the pool costs.

    `ShortConvStatePool.bytes_per_device` is what the runner takes out of the KV
    budget, so it has to equal what the pool's buffers hold on device 0 and what
    the spec written out by hand gives. A pool's create and its clear, which every
    flush_cache runs, compile one allocator per window shape, not one per window.
    """
    from sgl_jax.srt.mem_cache.short_conv_state_pool import ShortConvStatePool

    want = expected_conv_widths(CONFIG)
    got = serving.model.get_conv_state_spec()
    note(
        "conv state spec",
        got == want,
        f"{len(got)} layers, {got[0][0][1]} channels on k in layer 0 against {want[0][0][1]}",
        failures,
    )
    pool = serving.runner.memory_pools.conv_state_pool
    live = [
        sorted((name, int(array.shape[1]), int(array.shape[2])) for name, array in layer.items())
        for layer in pool.buffers
    ]
    slots = {int(array.shape[0]) for layer in pool.buffers for array in layer.values()}
    note(
        "the pool holds one slot per request slot",
        live == [sorted(layer) for layer in want]
        and slots == {serving.runner.req_to_token_pool.size + 1},
        f"{sorted(slots)} slots for {serving.runner.req_to_token_pool.size} request slots"
        " and the dummy",
        failures,
    )

    device = serving.mesh.devices.flat[0]
    held = sum(
        shard.data.nbytes
        for layer in pool.buffers
        for array in layer.values()
        for shard in array.addressable_shards
        if shard.device == device
    )
    budget = ShortConvStatePool.bytes_per_device(got, pool.size, serving.mesh, pool.dp_size)
    # Every window here splits over both devices of the tensor axis.
    by_hand = (
        (pool.size + 1) * sum(channels * tokens for layer in want for _, channels, tokens in layer)
        * 4
        // TP
    )
    note(
        "the pool's bytes per device",
        held == budget == by_hand,
        f"{held} on device 0, {budget} in the KV budget, {by_hand} written out",
        failures,
    )

    # Two widths no other pool in this process uses, 12 layers of four windows.
    spec = [[("a", 24, 3), ("b", 24, 3), ("c", 40, 3), ("d", 40, 3)] for _ in range(12)]
    windows = sum(len(layer) for layer in spec)
    shapes = len({channels for layer in spec for _, channels, _ in layer})
    before = Compiles.count
    extra = ShortConvStatePool(spec, size=6, mesh=serving.mesh)
    created = Compiles.count - before
    before = Compiles.count
    extra.clear()
    cleared = Compiles.count - before
    note(
        "window allocations compile once per shape",
        0 < created <= shapes and cleared == 0,
        f"{created} compiles to create {windows} windows in {shapes} shapes, {cleared} to clear"
        " them",
        failures,
    )


# --- contracts --------------------------------------------------------------


def check_contracts(serving, worker_batch, failures):
    """The pool, its slots and the backend kwarg each have to stop a run that lacks them."""
    from flax import nnx

    from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

    class BlindBackend(nnx.Module):
        """An attention backend that never reads the kwarg."""

        def __call__(self, *args, **kwargs):
            raise AssertionError("the model reached attention on a blind backend")

    note(
        "the native backend advertises the kwarg",
        bool(getattr(serving.runner.attn_backend, "supports_position_bias", False)),
        "supports_position_bias is True",
        failures,
    )
    try:
        serving.trial(worker_batch, attn_backend=BlindBackend())
    except NotImplementedError as error:
        raised, detail = "position_bias" in str(error), "NotImplementedError names the kwarg"
    except AssertionError:
        raised, detail = False, "the model served the request instead of raising"
    else:
        raised, detail = False, "no raise"
    note("a backend without the kwarg raises", raised, detail, failures)

    kv_only = MemoryPools(token_to_kv_pool=serving.runner.memory_pools.token_to_kv_pool)
    try:
        serving.trial(worker_batch, memory_pools=kv_only)
    except ValueError as error:
        raised, detail = "conv_state_pool" in str(error), "ValueError names the missing pool"
    else:
        raised, detail = False, "the prefill ran on zeroed windows"
    note("a prefill without the conv pool raises", raised, detail, failures)

    worker_batch.recurrent_indices, saved = None, worker_batch.recurrent_indices
    try:
        serving.trial(worker_batch)
    except ValueError as error:
        raised, detail = "recurrent_indices" in str(error), "ValueError names recurrent_indices"
    else:
        raised, detail = False, "the prefill ran with no slots"
    finally:
        worker_batch.recurrent_indices = saved
    note("a batch without request slots raises", raised, detail, failures)


def first_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {(str(error).splitlines() or [''])[0]}"


def check_refusals(model_path, scratch, failures):
    """Flags Inkling can't serve stop the worker before it reads a weight.

    The worker gets a copy of the checkpoint with every file but the weights, so
    a launch that got past the check would stop in the loader instead of naming
    the flag. The runner checks again when it builds the pool, for a runner
    built without a worker.
    """
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.tp_worker import ModelWorker
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    weightless = os.path.join(scratch, "weightless")
    shutil.copytree(model_path, weightless, ignore=shutil.ignore_patterns("*.safetensors*"))

    def launch(path, dp=1, **overrides):
        settings = dict(
            model_path=path,
            device="cpu",
            tp_size=TP,
            dp_size=dp,
            dtype="float32",
            skip_tokenizer_init=True,
            attention_backend="native",
            disable_radix_cache=True,
            max_running_requests=4,
            max_prefill_tokens=CHUNK,
            chunked_prefill_size=CHUNK,
            max_total_tokens=64,
            disable_overlap_schedule=True,
            random_seed=0,
        )
        settings.update(overrides)
        mesh = create_device_mesh(
            ici_parallelism=[dp, TP // dp], dcn_parallelism=[1, 1], device_indexes=list(range(TP))
        )
        return ServerArgs(**settings), mesh

    cases = (
        ("the radix cache", dict(disable_radix_cache=False), "--disable-radix-cache"),
        ("data parallelism", dict(dp=2), "--dp-size 1"),
        ("the tt attention backend", dict(attention_backend="tt"), "--attention-backend native"),
        ("speculative decoding", dict(speculative_algorithm="NEXTN"), "speculative decoding"),
        (
            "the recurrent extra buffer",
            dict(enable_recurrent_extra_buffer=True, page_size=2),
            "--enable-recurrent-extra-buffer",
        ),
        ("Pathways PD disaggregation", dict(pd_disaggregation="pathways"), "--pd-disaggregation"),
        (
            "a PD decode server",
            dict(
                disaggregation_mode="decode",
                disaggregation_bootstrap_url="http://127.0.0.1:8998",
                page_size=128,
            ),
            "--disaggregation-mode decode",
        ),
        ("EPLB dispatch", dict(ep_dispatch_algorithm="static"), "--ep-dispatch-algorithm"),
        ("redundant experts", dict(ep_num_redundant_experts=2), "--ep-num-redundant-experts"),
        (
            "an expert location file",
            dict(init_expert_location=os.path.join(scratch, "no-such-map.json")),
            "--init-expert-location",
        ),
    )
    for label, overrides, flag in cases:
        args, mesh = launch(weightless, **overrides)
        try:
            ModelWorker(server_args=args, mesh=mesh)
        except ValueError as error:
            if flag in str(error):
                refused, detail = True, f"ValueError names {flag}"
            else:
                refused, detail = False, f"stopped somewhere else first, {first_line(error)}"
        except Exception as error:  # noqa: BLE001 - where it stopped is the finding
            refused, detail = False, f"stopped somewhere else first, {first_line(error)}"
        else:
            refused, detail = False, "the worker started"
        note(f"{label} is refused before any weight loads", refused, detail, failures)

    # NEGATIVE CONTROL: flags Inkling serves have to get past the check on the
    # same copy and stop in the loader, or the copy proves nothing about order.
    args, mesh = launch(weightless)
    try:
        ModelWorker(server_args=args, mesh=mesh)
    except Exception as error:  # noqa: BLE001 - the loader's error is the point
        reached, detail = "can't serve" not in str(error), first_line(error)
    else:
        reached, detail = False, "the worker started with no weights"
    print(
        f"      control (served flags on the weightless copy): {detail}"
        f"  -> {'detected' if reached else 'NOT DETECTED'}"
    )
    if not reached:
        failures.append("the weightless copy stops nothing, so the refusals prove no order")

    args, mesh = launch(model_path, disable_radix_cache=False)
    try:
        ModelRunner(
            model_config=ModelConfig.from_server_args(args),
            mem_fraction_static=args.mem_fraction_static,
            tp_size=TP,
            dp_size=1,
            server_args=args,
            mesh=mesh,
        )
    except ValueError as error:
        refused, detail = "--disable-radix-cache" in str(error), "ValueError names the flag"
    else:
        refused, detail = False, "the runner started with the radix cache on"
    note("a runner built without a worker refuses the radix cache", refused, detail, failures)
    retire()


# --- the weights the loader read --------------------------------------------


def check_layout(serving, reference, model_path, failures):
    """The fused gate and up tensors are interleaved, and the loader reads them so."""
    import jax
    import numpy as np
    import safetensors

    with safetensors.safe_open(
        os.path.join(model_path, "model.safetensors"), framework="np"
    ) as handle:
        fused = handle.get_tensor("model.llm.layers.0.mlp.w13_dn.weight")
    hf_gate = reference.model.layers[0].mlp.gate_proj.weight.detach().numpy()
    width = hf_gate.shape[0]
    note(
        "save_pretrained interleaves gate and up",
        bool(np.array_equal(fused[0::2], hf_gate) and not np.array_equal(fused[:width], hf_gate)),
        "gate on the even rows of w13_dn, not the first half",
        failures,
    )

    layers = serving.model.model.layers
    dense = layers[0].mlp
    moe = layers[1].mlp
    loaded = {
        "dense gate": (dense.gate_proj.weight.value, hf_gate.T),
        "routed gate": (
            np.transpose(np.asarray(moe.experts.wi_0.value), (0, 2, 1)),
            reference.model.layers[1].mlp.experts.gate_up_proj.detach().numpy()[
                :, : CONFIG["moe_intermediate_size"]
            ],
        ),
        "shared up": (
            np.transpose(np.asarray(moe.shared_experts.up_proj.value), (0, 2, 1)),
            reference.model.layers[1].mlp.shared_experts.up_proj.detach().numpy(),
        ),
    }
    for label, (got, want) in loaded.items():
        error = float(np.max(np.abs(np.asarray(jax.device_get(got)) - want)))
        note(f"{label} matches transformers", error == 0.0, f"max abs {error:.1e}", failures)

    # The dense and shared weights land on the mesh in their declared layouts,
    # not whole on device 0.
    placed = {
        "dense gate": dense.gate_proj.weight.value,
        "dense down": dense.down_proj.weight.value,
        "shared gate": moe.shared_experts.gate_proj.value,
        "shared down": moe.shared_experts.down_proj.value,
    }
    for label, array in placed.items():
        sharding = array.sharding
        note(
            f"{label} is sharded over the mesh",
            len(sharding.device_set) == TP and "tensor" in str(getattr(sharding, "spec", "")),
            f"{type(sharding).__name__} {getattr(sharding, 'spec', '')}"
            f" on {len(sharding.device_set)} devices",
            failures,
        )


# --- tensor parallel past the KV head count ---------------------------------


def check_replication(scratch, failures):
    """Both attention kinds replicate their KV heads at --tp-size 8, k and v convolutions too."""
    import numpy as np
    import safetensors

    model_path = os.path.join(scratch, "replication")
    reference = write_checkpoint(REPLICATION_CONFIG, model_path)
    serving = Serving(model_path, REPLICATION_TP)
    width = REPLICATION_TP * REPLICATION_CONFIG["head_dim"]
    keep = REPLICATION_CONFIG["unpadded_vocab_size"]
    path = os.path.join(model_path, "model.safetensors")
    with safetensors.safe_open(path, framework="np") as handle:
        stored = {
            index: handle.get_slice(f"model.llm.layers.{index}.attn.k_sconv.weight").get_shape()[0]
            for index in range(REPLICATION_CONFIG["num_hidden_layers"])
        }
    for index, layer in enumerate(serving.model.model.layers):
        attention = layer.self_attn
        kind = "local" if attention.is_local else "global"
        note(
            f"layer {index} ({kind}) k and its convolution",
            attention.k_proj.weight.value.shape[1] == width
            and attention.k_sconv.weight.value.shape[0] == width
            and stored[index] < width,
            f"checkpoint {stored[index]} wide, k_proj {attention.k_proj.weight.value.shape[1]},"
            f" k_sconv {attention.k_sconv.weight.value.shape[0]}",
            failures,
        )
    rng = np.random.default_rng(7)
    prompt = rng.integers(0, keep, size=7).tolist()
    req = serving.request("replicated", prompt)
    batch, step = serving.extend([req])
    want = reference_logits(reference, prompt)
    report("prefill logits", step.logits[0, :keep], want[-1], failures)
    sequence = list(prompt) + [req.output_ids[-1]]
    for index in range(2):
        step = serving.decode(batch)
        want = reference_logits(reference, sequence)
        report(f"step {index} logits", step.logits[0, :keep], want[-1], failures)
        sequence.append(int(step.next_ids[0]))
    serving.release(req)
    check_precompiled(serving, f"serving at --tp-size {REPLICATION_TP}", failures)
    del serving
    retire()


# --- the published structure at the v5p-64's ratios -------------------------

# Every field of the published `text_config`, sizes shrunk. At --tp-size 8 it
# runs 2 query heads a device, 4 copies of each global KV head and 2 of each
# local one, the counts Inkling runs at --tp-size 32. The layer plan is the
# published one cut to 6 layers: 5 local then 1 global, the first 2 dense.
# `head_dim`, `d_rel`, the conv kernel, `log_scaling_*`, `route_scale`, the
# top-k and the logit multiplier keep their published values. The vocabulary pads 6 rows,
# as the published one pads 966.
PUBLISHED_TINY_CONFIG = dict(
    CONFIG,
    vocab_size=64,
    unpadded_vocab_size=58,
    hidden_size=64,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=128,
    swa_num_attention_heads=16,
    swa_num_key_value_heads=4,
    swa_head_dim=128,
    sliding_window_size=8,
    d_rel=16,
    rel_extent=16,
    log_scaling_n_floor=128000,
    log_scaling_alpha=0.1,
    local_layer_ids=[i for i in range(6) if i % 6 != 5],
    dense_mlp_idx=2,
    dense_intermediate_size=96,
    intermediate_size=96,
    moe_intermediate_size=32,
    n_routed_experts=16,
    num_experts_per_tok=6,
    n_shared_experts=2,
    route_scale=8.0,
    logits_mup_width_multiplier=24.0,
    model_max_length=128,
)
PUBLISHED_TINY_TP = 8
# Past the 16-token bias extent and the 8-token window, in one prefill pass.
PUBLISHED_TINY_PROMPT = 24


def check_capture_reference(model_path: str, ids, workdir: str, *, single_norm: bool):
    """`check_capture.reference_forward` on `ids`, float32, run in a process of its own.

    The capture check on the slice gates against this function's output. It runs
    apart from the engine because importing `sgl_jax` registers its own Inkling
    config with `AutoConfig`, which `transformers` can't build a model from.
    `single_norm=False` drops the embed-norm fix, which rebuilds the reference the
    v5p-64 check of 2026-09-27 read.
    """
    import json

    import numpy as np

    out = os.path.join(workdir, f"check-capture-reference-{int(single_norm)}.npz")
    script = (
        "import json, sys, numpy as np\n"
        f"sys.path.insert(0, {os.path.join(HERE, os.pardir, os.pardir, 'scripts')!r})\n"
        "import check_capture\n"
        + (
            ""
            if single_norm
            else "check_capture.norm_embeddings_once = lambda model: {'modules': 0}\n"
        )
        + f"states = check_capture.reference_forward({model_path!r}, [{json.dumps(list(ids))}],"
        " 'float32')[0]\n"
        f"np.savez({out!r}, *states)\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=900
    )
    if run.returncode != 0:
        raise RuntimeError(f"check_capture.reference_forward exited {run.returncode}: {run.stderr}")
    for line in run.stdout.splitlines():
        if "embed_norm ran" in line:
            print("      " + line)
    data = np.load(out)
    return [data[f"arr_{index}"] for index in range(len(data.files))]


def check_published_structure(scratch, tree, failures):
    """The published structure at --tp-size 8, in one process and in two, against transformers.

    One process serves a prompt past the bias extent and the window and captures
    every layer input. Each has to match `transformers.InklingForCausalLM` and
    `check_capture.reference_forward`, the reference the v5p-64 check gates
    against. That reference's control drops its embed-norm fix and has to miss at
    layer 0 by more than CONTROL_FLOOR. Then two processes of 4 devices each load
    the same checkpoint over one 8-device mesh, and every block of every weight
    each process holds has to equal the one-process load's. Its control repeats
    the KV heads in the tiled order, `h0 h1 h0 h1`, in place of `h0 h0 h1 h1`.
    """
    import json

    import jax
    import numpy as np
    from flax import nnx

    cfg = PUBLISHED_TINY_CONFIG
    model_path = os.path.join(scratch, "published-tiny")
    reference = write_checkpoint(cfg, model_path)
    serving = Serving(
        model_path,
        PUBLISHED_TINY_TP,
        precompile=False,
        # 64 padded tokens times the top 6 is 384 rows, three of the grouped
        # matmul's 128-row tiles.
        chunked_prefill_size=64,
        max_prefill_tokens=64,
        max_total_tokens=256,
        precompile_token_paddings=[64],
    )
    for index, layer in enumerate(serving.model.model.layers):
        attention = layer.self_attn
        kind = "local" if attention.is_local else "global"
        heads = attention.q_head_num // PUBLISHED_TINY_TP
        stored = cfg["swa_num_key_value_heads"] if attention.is_local else cfg["num_key_value_heads"]
        copies = attention.k_proj.weight.value.shape[1] // (stored * attention.head_dim)
        note(
            f"layer {index} ({kind}) runs the v5p-64's head counts",
            heads == 2 and copies == (2 if attention.is_local else 4),
            f"{heads} query heads a device, {copies} copies of each of {stored} KV heads",
            failures,
        )

    keep = cfg["unpadded_vocab_size"]
    prompt = np.random.default_rng(17).integers(0, keep, size=PUBLISHED_TINY_PROMPT).tolist()
    req = serving.request("published", prompt, hidden=True)
    _, step = serving.extend([req])
    report("prefill logits", step.logits[0, :keep], reference_logits(reference, prompt)[-1], failures)
    captured = (
        np.asarray(step.output.hidden_states)[: len(prompt)]
        .reshape(len(prompt), cfg["num_hidden_layers"], cfg["hidden_size"])
        .transpose(1, 0, 2)
    )
    serving.release(req)
    for index, (got, want) in enumerate(zip(captured, reference_layer_inputs(reference, prompt))):
        kind = "local" if index in cfg["local_layer_ids"] else "global"
        mlp = "dense" if index < cfg["dense_mlp_idx"] else "routed"
        report(f"layer {index} input ({kind}/{mlp})", got, want, failures)

    workdir = os.path.join(scratch, "published-tiny-logs")
    os.makedirs(workdir, exist_ok=True)
    print("  against check_capture.reference_forward, what the capture check gates on")
    fixed = check_capture_reference(model_path, prompt, workdir, single_norm=True)
    for index, got in enumerate(captured):
        report(f"slot {index} against check_capture", got, fixed[index], failures)
    unfixed = check_capture_reference(model_path, prompt, workdir, single_norm=False)
    control(
        "check_capture's reference with embed_norm twice",
        float(np.max(np.abs(captured[0] - unfixed[0]))),
        failures,
    )

    one_process = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(serving.model)):
        array = getattr(leaf, "value", leaf)
        if isinstance(array, jax.Array):
            one_process[jax.tree_util.keystr(path)] = np.asarray(jax.device_get(array))
    del serving
    retire()

    def compare(mode):
        results = host_reads(tree, model_path, workdir, mode)
        if results is None:
            return None, None
        blocks = mismatched = 0
        first = None
        for pid in sorted(results):
            dump = np.load(os.path.join(workdir, f"blocks-{mode}-{pid}.npz"))
            for key in dump.files:
                name, bounds = key.split("|", 1)
                index = tuple(slice(start, stop) for start, stop in json.loads(bounds))
                blocks += 1
                want = one_process.get(name)
                if want is None or not np.array_equal(dump[key], want[index]):
                    mismatched += 1
                    first = first or f"{name} {bounds} on process {pid}"
        return blocks, (mismatched, first)

    blocks, outcome = compare("dump")
    if blocks is None:
        failures.append("the two-process load of the published structure didn't report")
        return
    mismatched, first = outcome
    names = {key.split("|", 1)[0] for key in one_process}
    note(
        f"{HOST_READS_PROCESSES} processes x {HOST_READS_DEVICES} devices load what one process loads",
        blocks > 0 and mismatched == 0,
        f"{blocks} blocks of {len(names)} weights, {mismatched} differ"
        + (f", the first {first}" if first else ""),
        failures,
    )
    control_blocks, control_outcome = compare("tiled-heads")
    caught = control_blocks is not None and control_outcome[0] > 0
    print(
        f"      control (KV heads repeated in the tiled order): "
        f"{control_outcome[0] if control_outcome else 'no'} blocks differ"
        f"  -> {'detected' if caught else 'NOT DETECTED'}"
    )
    if not caught:
        failures.append("the tiled-heads control loaded what one process loads, so the check proves nothing")


# --- the runner's own fields ------------------------------------------------


def check_expert_parallel(model_path, reference, failures):
    """`--ep-size` reaches every `EPMoE` through the nested config, and the logits hold."""
    import numpy as np

    serving = Serving(model_path, TP, precompile=False, ep_size=2)
    moe = [layer.mlp.experts for layer in serving.model.model.layers if layer.is_moe]
    note(
        "--ep-size 2 reaches EPMoE",
        all(experts.ep_size == 2 and experts.experts_per_device == 3 for experts in moe),
        f"ep_size {[experts.ep_size for experts in moe]},"
        f" experts per device {[experts.experts_per_device for experts in moe]}",
        failures,
    )
    keep = CONFIG["unpadded_vocab_size"]
    prompt = np.random.default_rng(11).integers(0, keep, size=7).tolist()
    _, step = serving.extend([serving.request("expert-parallel", prompt)])
    want = reference_logits(reference, prompt)
    report("prefill logits at --ep-size 2", step.logits[0, :keep], want[-1], failures)
    del serving
    retire()


def check_runtime_fields(model_path, failures):
    """The quantization config and `--enable-dp-lm-head` reach the layers that read them.

    `ModelRunner.load_model` and `ModelConfig` write both onto the outer config,
    and the model reads the nested text block. Dummy weights keep the load
    short; nothing here runs a forward.
    """
    import jax.numpy as jnp

    serving = Serving(
        model_path,
        TP,
        precompile=False,
        load_format="dummy",
        quantization_config_path="int8.yaml",
        enable_dp_lm_head=True,
    )
    moe = [layer.mlp.experts for layer in serving.model.model.layers if layer.is_moe]
    dtypes = [
        jnp.dtype(experts.quantized_dtype).name if experts.quantized_dtype else None
        for experts in moe
    ]
    note(
        "the quantization config reaches EPMoE",
        all(experts.quantized_dtype == jnp.int8 for experts in moe),
        f"quantized_dtype {dtypes}",
        failures,
    )
    note(
        "--enable-dp-lm-head reaches the head",
        bool(serving.model.lm_head.enable_dp_lm_head)
        and bool(serving.model.logits_processor.enable_dp_lm_head),
        f"lm_head {serving.model.lm_head.enable_dp_lm_head},"
        f" logits processor {serving.model.logits_processor.enable_dp_lm_head}",
        failures,
    )
    del serving
    retire()


def check_dummy_load(model_path, failures):
    """`--load-format dummy` serves a prefill: the routed experts land on `EPMoE`'s mesh."""
    import numpy as np

    serving = Serving(model_path, TP, precompile=False, load_format="dummy")
    try:
        _, step = serving.extend([serving.request("dummy", [1, 2, 3, 4, 5])])
        logits = step.logits[:, : CONFIG["unpadded_vocab_size"]]
        ran, detail = bool(np.all(np.isfinite(logits))), f"finite logits, shape {logits.shape}"
    except Exception as error:  # noqa: BLE001 - the failure is the finding
        ran, detail = False, f"{type(error).__name__}: {str(error).splitlines()[0]}"
    note("a dummy-weight prefill runs", ran, detail, failures)
    del serving
    retire()


def dummy_load_record(model_path, mesh, fill_experts: bool = True) -> dict:
    """`--load-format dummy` through `JAXDummyModelLoader`, watched from device 0.

    Records the routed-expert shards right after the loader's dummy pass beside the
    ones `EPMoE`'s layout gives them, the most device 0 holds of the arrays the load
    allocates while it runs, and what the loaded model holds on device 0. Bytes come
    off each array's sharding, so no buffer gets touched. `fill_experts=False` turns
    the model's own expert fill off, for the control.
    """
    import jax
    from flax import nnx
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.model_loader.loader import get_model_loader
    from sgl_jax.srt.models import inkling
    from sgl_jax.srt.utils.weight_utils import WeightLoader

    # The layouts `EPMoE.__init__` declares on its (expert, tensor) mesh.
    layout = {
        "wi_0": P("expert", None, "tensor"),
        "wi_1": P("expert", None, "tensor"),
        "wo": P("expert", "tensor", None),
    }
    device = mesh.devices.flat[0]

    def device_bytes(array) -> int:
        if device not in array.sharding.device_set:
            return 0
        return math.prod(array.sharding.shard_shape(array.shape)) * array.dtype.itemsize

    def expert_shards(model) -> list:
        rows = []
        for index, layer in enumerate(model.model.layers):
            if not layer.is_moe:
                continue
            experts = layer.mlp.experts
            for name, spec in layout.items():
                value = getattr(experts, name).get_value()
                want = NamedSharding(experts.moe_mesh, spec).shard_shape(value.shape)
                got = value.sharding.shard_shape(value.shape)
                rows.append((f"layer {index} {name}", tuple(got), tuple(want)))
        return rows

    made = {}  # id of each array the load allocates -> the bytes device 0 holds of it
    record = {"peak": 0}
    dummy_pass = WeightLoader.load_weights_from_safetensors
    allocate = WeightLoader._dummy_array
    fill = inkling.InklingForCausalLM.place_dummy_experts

    def watched_pass(self, *args, **kwargs):
        result = dummy_pass(self, *args, **kwargs)
        record["after pass"] = expert_shards(self.model)
        return result

    def watched_allocate(self, *args, **kwargs):
        array = allocate(self, *args, **kwargs)
        made[id(array)] = device_bytes(array)
        live = {id(held) for held in jax.live_arrays()}
        record["peak"] = max(record["peak"], sum(v for k, v in made.items() if k in live))
        return array

    WeightLoader.load_weights_from_safetensors = watched_pass
    WeightLoader._dummy_array = watched_allocate
    if not fill_experts:
        inkling.InklingForCausalLM.place_dummy_experts = lambda *args, **kwargs: None
    try:
        gc.collect()
        model_config = ModelConfig(model_path=model_path, trust_remote_code=False, dtype="float32")
        loader = get_model_loader(LoadConfig(load_format=LoadFormat.DUMMY), mesh)
        model = loader.load_model(model_config=model_config)
        record["held"] = sum(
            device_bytes(leaf)
            for leaf in jax.tree.leaves(nnx.state(model))
            if isinstance(leaf, jax.Array)
        )
    finally:
        WeightLoader.load_weights_from_safetensors = dummy_pass
        WeightLoader._dummy_array = allocate
        inkling.InklingForCausalLM.place_dummy_experts = fill
    del model
    return record


def check_dummy_placement(scratch, failures):
    """At --tp-size 8 a dummy-weight load never holds a whole routed-expert stack on one device.

    The loader's dummy pass fills every parameter the mapping table leaves out, on
    the model mesh, which has no `expert` axis, so a stack it filled would sit whole
    on every device, 29.0 GB a layer at Inkling's shapes. The model fills its routed
    experts on `EPMoE`'s mesh before that pass. Control: with that fill turned off,
    the pass leaves the stacks whole on every device.
    """
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    model_path = os.path.join(scratch, "dummy-placement")
    write_checkpoint(REPLICATION_CONFIG, model_path)
    mesh = create_device_mesh(
        ici_parallelism=[1, REPLICATION_TP],
        dcn_parallelism=[1, 1],
        device_indexes=list(range(REPLICATION_TP)),
    )

    record = dummy_load_record(model_path, mesh)
    wrong = [row for row in record["after pass"] if row[1] != row[2]]
    note(
        "after the loader's pass the routed experts sit in EPMoE's layout",
        bool(record["after pass"]) and not wrong,
        f"{len(record['after pass'])} expert arrays"
        + ("" if not wrong else f"; {len(wrong)} don't, the first {wrong[0][0]} at {wrong[0][1]}"),
        failures,
    )
    note(
        "device 0 never holds more than the loaded model",
        record["peak"] <= 1.25 * record["held"],
        f"at most {record['peak']:,} bytes of what the load allocates, {record['held']:,} in the"
        " loaded model",
        failures,
    )

    control = dummy_load_record(model_path, mesh, fill_experts=False)
    whole = [row for row in control["after pass"] if row[1] != row[2]]
    verdict = "detected" if whole else "NOT DETECTED"
    print(
        f"      control (the model's expert fill off): {len(whole)} expert arrays sit whole on"
        f" every device, peak {control['peak']:,} bytes  -> {verdict}"
    )
    if not whole:
        failures.append("the dummy load without the expert fill left nothing whole")
    retire()


# --- the attention's temporaries -------------------------------------------


def attention_temporaries(bucket: int, *, bias: bool) -> int:
    """Temporary bytes XLA plans for one extend `forward_attention` over a `bucket`-slot cache_loc.

    Four requests of 16 tokens, 8 query heads on 2 KV heads, a 4,096-slot pool.
    The bucket is the engine's `cache_loc` length: `max_running_requests` times
    the page-aligned `max_req_len`. `bias=False` runs the whole-bucket pass the
    other models take, `bias=True` the blockwise one Inkling's position bias
    takes.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.layers.attention.native_backend import forward_attention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    mesh = create_device_mesh(ici_parallelism=[1, 1], dcn_parallelism=[1, 1], device_indexes=[0])
    tokens, heads, kv_heads, dim, pool, extent = 64, 8, 2, 128, 4096, 16

    def spec(*shape, dtype=jnp.float32, axes=None):
        axes = axes if axes is not None else P()
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, axes))

    def run(q, k, v, seq_lens, loc, prefix, extend, position_bias):
        return forward_attention(
            q,
            k,
            v,
            seq_lens,
            loc,
            prefix,
            extend,
            heads,
            kv_heads,
            page_size=16,
            scale=1.0 / dim,
            mode=ForwardMode.EXTEND,
            kv_sharding=NamedSharding(mesh, P(None, "tensor", None)),
            mesh=mesh,
            sliding_window_size=None,
            softmax_dtype=jnp.float32,
            position_bias=position_bias,
        )

    ints = jnp.int32
    args = (
        spec(tokens, heads, dim, axes=P(None, "tensor", None)),
        spec(pool, kv_heads, dim, axes=P(None, "tensor", None)),
        spec(pool, kv_heads, dim, axes=P(None, "tensor", None)),
        spec(4, dtype=ints),
        spec(bucket, dtype=ints),
        spec(4, dtype=ints),
        spec(4, dtype=ints),
        spec(tokens, heads, extent, axes=P(None, "tensor", None)) if bias else None,
    )
    with jax.set_mesh(mesh):
        compiled = jax.jit(run).lower(*args).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes)


def check_attention_memory(failures):
    """The position-bias pass holds the same temporaries whatever the cache_loc bucket.

    Inkling's check launch on a v5p-64 hands the native backend a 5,329,408-slot
    bucket. An AOT compile of a pass that spans it asks for 1.43 TB of temporaries
    per chip, and the chip's own compile asked for 1.71 TB.
    Control: the whole-bucket pass, which grows with the bucket.
    """
    small, large = 8192, 65536
    bias_small = attention_temporaries(small, bias=True)
    bias_large = attention_temporaries(large, bias=True)
    note(
        "position-bias temporaries don't grow with the bucket",
        bias_large <= 1.05 * bias_small + (large - small) * 16,
        f"{bias_small:,} bytes at {small:,} slots, {bias_large:,} at {large:,}",
        failures,
    )
    whole_small = attention_temporaries(small, bias=False)
    whole_large = attention_temporaries(large, bias=False)
    grew = whole_large >= 4 * whole_small
    print(
        f"      control (the whole-bucket pass): {whole_small:,} bytes at {small:,} slots,"
        f" {whole_large:,} at {large:,}  -> {'detected' if grew else 'NOT DETECTED'}"
    )
    if not grew:
        failures.append(
            "the whole-bucket pass didn't grow with the bucket, so the check proves nothing"
        )


# --- what each host reads ---------------------------------------------------

HOST_READS_PROCESSES = 2
HOST_READS_DEVICES = 4  # per process, so the mesh spans 8 devices like REPLICATION_TP


def host_reads_child(argv) -> int:
    """One of two processes loading one checkpoint over an 8-device mesh, 4 devices each.

    Counts the bytes each checkpoint key gives up to this process, through
    `safetensors` and through plain file reads of a `.safetensors` file. The
    checkpoint directory reads as a gcsfuse mount, so the loader's warm-up
    decides on the process count alone. `mode` is `fixed`, or a control:
    `warm-up` forces the single-process warm-up, `whole` reads every block out
    of its whole tensor.
    """
    import io
    import json
    import types

    import jax

    pid, port, tree, model_path, mode = int(argv[1]), argv[2], argv[3], argv[4], argv[5]
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{port}",
        num_processes=HOST_READS_PROCESSES,
        process_id=pid,
        initialization_timeout=120,
    )
    install(tree)
    import builtins

    import numpy as np
    import safetensors
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.model_loader import loader as loader_module
    from sgl_jax.srt.model_loader.loader import JAXModelLoader, get_model_loader
    from sgl_jax.srt.models import inkling
    from sgl_jax.srt.utils import weight_utils

    reads: dict[str, int] = {}

    def count(key, nbytes):
        reads[key] = reads.get(key, 0) + int(nbytes)

    real_safe_open = safetensors.safe_open

    class CountedSlice:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key

        def __getitem__(self, index):
            block = self.inner[index]
            count(self.key, np.asarray(block).nbytes)
            return block

        def __getattr__(self, name):
            return getattr(self.inner, name)

    class CountedHandle:
        def __init__(self, *args, **kwargs):
            self.inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

        def get_slice(self, key):
            return CountedSlice(self.inner.get_slice(key), key)

        def get_tensor(self, key):
            tensor = self.inner.get_tensor(key)
            count(key, np.asarray(tensor).nbytes)
            return tensor

        def __getattr__(self, name):
            return getattr(self.inner, name)

    safetensors.safe_open = CountedHandle
    if hasattr(weight_utils, "safe_open"):
        weight_utils.safe_open = CountedHandle
    loader_module.safe_open = CountedHandle

    class CountedFile:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.inner.close()

        def readinto(self, buffer):
            size = self.inner.readinto(buffer)
            count("whole-file reads", size or 0)
            return size

        def read(self, *args):
            data = self.inner.read(*args)
            count("whole-file reads", len(data))
            return data

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def mounted_open(path, *args, **kwargs):
        if str(path) == "/proc/mounts":
            return io.StringIO(f"gcsfuse {model_path} fuse.gcsfuse ro,nosuid 0 0\n")
        handle = builtins.open(path, *args, **kwargs)
        return CountedFile(handle) if str(path).endswith(".safetensors") else handle

    loader_module.open = mounted_open

    if mode == "warm-up":
        warm_up = JAXModelLoader._warmup_safetensors_cache

        def single_process_warm_up(model_config):
            saved = loader_module.jax
            loader_module.jax = types.SimpleNamespace(process_count=lambda: 1)
            try:
                return warm_up(model_config)
            finally:
                loader_module.jax = saved

        JAXModelLoader._warmup_safetensors_cache = staticmethod(single_process_warm_up)
    elif mode == "tiled-heads":
        # The control for `dump`: each device reads the KV head its columns
        # would hold if the copies went h0 h1 h0 h1 rather than h0 h0 h1 h1.
        def tiled_kv_array(self, weight_info, key, param, attention):
            rows, hidden = self.tensor_shape(weight_info, key)
            width = param.shape[1]
            head_dim = attention.head_dim
            stored_heads = rows // head_dim

            def fetch(start, stop):
                columns = np.arange(start[1], stop[1])
                source = (columns // head_dim) % stored_heads * head_dim + columns % head_dim
                block = self.read_block(weight_info, key, (slice(0, rows), slice(start[0], stop[0])))
                return block[source].T

            return self.sharded_array((hidden, width), P(None, "tensor"), fetch)

        inkling.InklingForCausalLM.replicated_kv_array = tiled_kv_array
    elif mode == "whole":

        def whole_block(self, weight_info, key, index):
            path = self.tensor_file(weight_info, key)
            with safetensors.safe_open(path, framework="np", device="cpu") as handle:
                return handle.get_tensor(key)[index]

        inkling.InklingForCausalLM.read_block = whole_block

    devices = HOST_READS_PROCESSES * HOST_READS_DEVICES
    mesh = Mesh(
        np.array(jax.devices()).reshape(1, devices),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    model_config = ModelConfig(model_path=model_path, trust_remote_code=False, dtype="float32")
    model_config.validate_tensor_parallel_config(devices)
    model_config.configure_for_tensor_parallel(devices)
    with jax.set_mesh(mesh):
        model = get_model_loader(LoadConfig(load_format="auto"), mesh).load_model(
            model_config=model_config
        )
    mappings = model.create_weight_mappings()
    replicated = sorted(
        key
        for key, mapping in mappings.items()
        if mapping.sharding is not None and all(axis is None for axis in mapping.sharding)
    )
    # The loaded k of the first layer, this process's shards, against the checkpoint.
    attention = model.model.layers[0].self_attn
    with real_safe_open(os.path.join(model_path, "model.safetensors"), framework="np") as handle:
        stored = handle.get_tensor(f"{model.source_root}.layers.0.attn.wk_dv.weight")
    heads = stored.reshape(-1, attention.head_dim, stored.shape[-1])
    copies = attention.k_proj.weight.value.shape[1] // stored.shape[0]
    want = np.repeat(heads, copies, axis=0).reshape(-1, stored.shape[-1]).T
    k_matches = all(
        np.array_equal(np.asarray(shard.data), want[shard.index])
        for shard in attention.k_proj.weight.value.addressable_shards
    )
    if mode in ("dump", "tiled-heads"):
        # Every block of every weight this process holds, keyed by the weight's
        # path and the block's bounds, for the parent to hold against a
        # one-process load of the same checkpoint.
        from flax import nnx

        blocks = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(model)):
            array = getattr(leaf, "value", leaf)
            if not isinstance(array, jax.Array):
                continue
            name = jax.tree_util.keystr(path)
            for shard in array.addressable_shards:
                bounds = [list(part.indices(size)[:2]) for part, size in zip(shard.index, array.shape)]
                blocks[f"{name}|{json.dumps(bounds)}"] = np.asarray(shard.data)
        np.savez(os.path.join(os.environ["INKLING_BLOCKS_DIR"], f"blocks-{mode}-{pid}.npz"), **blocks)
    print(
        "RESULT "
        + json.dumps(
            {
                "pid": pid,
                "mode": mode,
                "reads": reads,
                "replicated": replicated,
                "mapped": sorted(mappings),
                "k_matches": bool(k_matches),
            }
        ),
        flush=True,
    )
    jax.distributed.shutdown()
    return 0


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def host_reads(tree, model_path, workdir, mode, timeout=900) -> dict | None:
    """Both processes' RESULT lines for one mode, keyed by process index."""
    import json

    port = free_port()
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={HOST_READS_DEVICES}",
        INKLING_BLOCKS_DIR=workdir,
    )
    logs = []
    procs = []
    files = [
        open(os.path.join(workdir, f"host-reads-{mode}-{pid}.log"), "w+")
        for pid in range(HOST_READS_PROCESSES)
    ]
    try:
        for pid, fh in enumerate(files):
            argv = [
                sys.executable,
                os.path.abspath(__file__),
                "--host-reads-child",
                str(pid),
                str(port),
                tree,
                model_path,
                mode,
            ]
            procs.append(subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + timeout
        for proc in procs:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.1))
    except subprocess.TimeoutExpired:
        pass
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for proc in procs:
            proc.wait()
        for fh in files:
            fh.seek(0)
            logs.append(fh.read())
            fh.close()
    results = {}
    for log in logs:
        for line in log.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
                results[result["pid"]] = result
    if len(results) != HOST_READS_PROCESSES:
        print(f"  [FAIL] {len(results)} of {HOST_READS_PROCESSES} processes reported ({mode}):")
        for log in logs:
            print(log)
        return None
    return results


def over_share(results, sizes) -> list[str]:
    """Each process's reads beyond its share of the sharded tensors plus every replicated one.

    A sharded tensor's share is its bytes times the process's fraction of the
    devices. A replicated tensor may be read whole once. A key the model never
    maps or stacks may not be read at all.
    """
    fraction = HOST_READS_DEVICES / (HOST_READS_PROCESSES * HOST_READS_DEVICES)
    problems = []
    for pid, result in sorted(results.items()):
        replicated = set(result["replicated"])
        mapped = set(result["mapped"])
        allowed = sum(
            size if key in replicated else size * fraction
            for key, size in sizes.items()
            if key in mapped or is_stacked_key(key)
        )
        total = sum(result["reads"].values())
        over = [
            f"{key} {read:,} of {sizes.get(key, 0):,}"
            for key, read in sorted(result["reads"].items())
            if read > (sizes.get(key, 0) if key in replicated else sizes.get(key, 0) * fraction)
        ]
        if total > allowed or over:
            problems.append(
                f"process {pid} read {total:,} bytes against a share of {allowed:,.0f}"
                + (f"; over on {len(over)} key(s), the first {over[0]}" if over else "")
            )
    return problems


def is_stacked_key(key: str) -> bool:
    """The tensors `load_stacked_weights` reads beside the mapping table."""
    return key.endswith(
        (
            ".mlp.w13_dn.weight",
            ".mlp.w2_md.weight",
            ".experts.w13_weight",
            ".experts.w2_weight",
            ".shared_experts.shared_w13_weight",
            ".shared_experts.shared_w2_weight",
        )
    ) and ".mtp." not in key


def check_host_reads(scratch, tree, failures):
    """Two processes of 4 devices each load one checkpoint, and each reads only its share.

    Its share is its half of every sharded tensor plus each replicated tensor
    whole, once, and nothing the model doesn't load. On the v5p-64 every host
    read the whole 1,773.9 GiB checkpoint through the loader's GCSFuse warm-up
    before it read its own slices. Controls: the single-process warm-up, and
    every block read out of its whole tensor. Both have to go over.
    """
    import safetensors

    model_path = os.path.join(scratch, "host-reads")
    write_checkpoint(REPLICATION_CONFIG, model_path)
    with safetensors.safe_open(os.path.join(model_path, "model.safetensors"), framework="np") as fh:
        sizes = {key: fh.get_tensor(key).nbytes for key in fh.keys()}
    workdir = os.path.join(scratch, "host-reads-logs")
    os.makedirs(workdir, exist_ok=True)

    results = host_reads(tree, model_path, workdir, "fixed")
    if results is None:
        failures.append("the two-process load didn't report")
        return
    for pid, result in sorted(results.items()):
        total = sum(result["reads"].values())
        print(
            f"      process {pid} read {total:,} of the checkpoint's {sum(sizes.values()):,} bytes"
        )
    problems = over_share(results, sizes)
    note(
        "each process reads its share once",
        not problems,
        "; ".join(problems) or f"{len(sizes)} keys, {sum(sizes.values()):,} bytes, two processes",
        failures,
    )
    note(
        "the k each process loaded matches the checkpoint",
        all(result["k_matches"] for result in results.values()),
        f"layer 0 k_proj shards on {len(results)} processes",
        failures,
    )
    for mode in ("warm-up", "whole"):
        control_results = host_reads(tree, model_path, workdir, mode)
        caught = control_results is not None and bool(over_share(control_results, sizes))
        totals = (
            [sum(r["reads"].values()) for _, r in sorted(control_results.items())]
            if control_results
            else []
        )
        print(
            f"      control ({mode}): processes read {totals} bytes"
            f"  -> {'detected' if caught else 'NOT DETECTED'}"
        )
        if not caught:
            failures.append(
                f"the {mode} control stayed within the share, so the check proves nothing"
            )


# --- the three entry classes ------------------------------------------------


def check_entry_classes(serving, failures):
    """Every name `EntryClass` registers builds, and the MTP head picks a layer."""
    import jax
    import jax.numpy as jnp

    from sgl_jax.srt.models import inkling

    config = serving.model.config
    mesh = serving.mesh
    names = [cls.__name__ for cls in inkling.EntryClass]
    note(
        "EntryClass registers three names",
        names
        == ["InklingForCausalLM", "InklingForConditionalGeneration", "InklingMTPForCausalLM"],
        ", ".join(names),
        failures,
    )
    note(
        "the published architecture serves the text path",
        type(serving.model).__name__ == "InklingForConditionalGeneration"
        and len(serving.model.model.layers) == config.num_hidden_layers,
        f"{type(serving.model).__name__}, {len(serving.model.model.layers)} layers under"
        f" source root {serving.model.source_root}",
        failures,
    )

    # One draft model per MTP layer. The published head mixes the two attention
    # kinds, so the layer the config marks local has to come out local and the
    # rest global, each with its own KV width and relative extent.
    kinds = []
    with jax.set_mesh(mesh):
        for index in range(config.num_mtp_layers):
            draft = inkling.InklingMTPForCausalLM(
                config, mesh=mesh, dtype=jnp.float32, mtp_layer_idx=index
            )
            attention = draft.model.layers[0].self_attn
            local = config.mtp_layer_types[index] == "hybrid_sliding"
            heads = config.swa_num_key_value_heads if local else config.num_key_value_heads
            extent = config.sliding_window_size if local else config.rel_extent
            kinds.append("local" if attention.is_local else "global")
            source = f"model.mtp.layers.{index}.transformer_block.attn.wk_dv.weight"
            ok = (
                attention.is_local == local
                and attention.kv_head_num == heads
                and attention.rel_logits_proj.proj.value.shape == (config.d_rel, extent)
                and not draft.model.layers[0].is_moe
                and source in draft.create_weight_mappings()
            )
            note(
                f"MTP layer {index} draft model",
                ok,
                f"{kinds[-1]}, {attention.kv_head_num} KV heads, rel projection"
                f" {tuple(attention.rel_logits_proj.proj.value.shape)}, reads {source}",
                failures,
            )

    # NEGATIVE CONTROL: the head isn't uniform. A draft config that read a
    # fixed index would build the same layer every time.
    detected = len(set(kinds)) > 1
    print(
        f"      control (the MTP head mixes both kinds): {kinds}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        failures.append("every MTP layer built the same kind, so the index is unread")


# --- the router -------------------------------------------------------------


def check_shared_expert_sink(serving, failures):
    """The shared logits normalize beside the routed ones.

    Without the sink the routed weights normalize among themselves and a shared
    gate row can't reach them, so this reads the router's routed output either
    side of a change to a shared row.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    router = serving.model.model.layers[1].mlp.router
    with jax.set_mesh(serving.mesh):
        probe = jax.random.normal(jax.random.PRNGKey(7), (16, CONFIG["hidden_size"]), jnp.float32)
        before = np.asarray(router(probe)[0])
        kernel = router.gate.kernel.value
        router.gate.kernel.value = kernel.at[:, CONFIG["n_routed_experts"]].add(1.0)
        after = np.asarray(router(probe)[0])
        router.gate.kernel.value = kernel

    moved = float(np.max(np.abs(after - before)))
    note(
        "a shared gate row moves the routed weights",
        moved > CONTROL_FLOOR,
        f"routed weights move by {moved:.3e}, gate width {router.gate.kernel.value.shape[-1]}"
        f" for {CONFIG['n_routed_experts']} routed and {CONFIG['n_shared_experts']} shared",
        failures,
    )


# --- the published expert width ---------------------------------------------


def check_expert_width(failures):
    """The serving config keeps the routed-expert width the checkpoint carries.

    The published `text_config` writes the expert width as `intermediate_size`
    and the dense width as `dense_intermediate_size`. `transformers` moves the
    dense width onto `intermediate_size` and leaves `moe_intermediate_size` at
    its class default of 3072, which is Inkling's width and not
    Inkling-Small's.
    """
    from transformers.models.inkling import InklingConfig as StockConfig

    from sgl_jax.srt.configs.inkling import InklingServingConfig

    # The two published shapes, transcribed.
    shapes = {
        "thinkingmachines/Inkling": (6144, 24576, 3072),
        "thinkingmachines/Inkling-Small": (4096, 16384, 2048),
    }
    stock_widths = []
    for repo, (hidden, dense, expert) in shapes.items():
        block = {
            "hidden_size": hidden,
            "num_hidden_layers": 4,
            "dense_intermediate_size": dense,
            "intermediate_size": expert,
        }
        served = InklingServingConfig(text_config=dict(block)).text_config
        stock = StockConfig(text_config=dict(block)).text_config
        stock_widths.append(stock.moe_intermediate_size)
        note(
            f"expert width, {repo}",
            served.moe_intermediate_size == expert and served.intermediate_size == dense,
            f"experts {served.moe_intermediate_size} wide, dense"
            f" {served.intermediate_size} wide, config says {expert} and {dense}",
            failures,
        )

    # NEGATIVE CONTROL: the stock config reads one width for both sizes, so it
    # can't be carrying the published field.
    detected = len(set(stock_widths)) == 1
    print(
        f"      control (stock config on both shapes): {stock_widths}"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        failures.append("the stock config already reads the published expert width")


# --- an older transformers --------------------------------------------------

# Hides `transformers.models.inkling` the way transformers before 5.14 lacks it.
HIDE_INKLING = """
import importlib.abc
import sys


class HideInkling(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "transformers.models.inkling" or name.startswith("transformers.models.inkling."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, HideInkling())
"""


def check_old_transformers(tree, failures):
    """A tree carrying this patch still serves other models on a transformers without Inkling."""
    env = dict(os.environ, PYTHONPATH=tree, JAX_PLATFORMS="cpu")

    def run(code):
        return subprocess.run(
            [sys.executable, "-c", HIDE_INKLING + code], env=env, capture_output=True, text=True
        )

    result = run(
        "import sgl_jax.srt.hf_transformers_utils as utils\n"
        "import sgl_jax.srt.configs.model_config\n"
        "print('inkling_mm_model' in utils._CONFIG_REGISTRY)\n"
    )
    lines = result.stdout.split()
    note(
        "the engine imports without transformers.models.inkling",
        result.returncode == 0 and lines[-1:] == ["False"],
        "hf_transformers_utils and model_config import, Inkling left unregistered"
        if result.returncode == 0
        else (result.stderr.strip().splitlines() or ["no output"])[-1],
        failures,
    )

    # NEGATIVE CONTROL: the Inkling config module itself has to fail in the
    # same environment, or the hook hides nothing.
    detected = run("import sgl_jax.srt.configs.inkling\n").returncode != 0
    print(
        f"      control (configs.inkling under the same hook)"
        f"  -> {'detected' if detected else 'NOT DETECTED'}"
    )
    if not detected:
        failures.append("the import hook hides nothing, so the check proves nothing")


# --- the checkpoint keys ----------------------------------------------------


def published_mappings(config, source_root="model.llm"):
    """Every source key `create_weight_mappings()` builds, at published widths."""
    import types

    from sgl_jax.srt.models import inkling

    holder = types.SimpleNamespace(config=config, source_root=source_root)
    return set(inkling.InklingForCausalLM.create_weight_mappings(holder))


def published_mtp_mappings(config, index):
    """The same for one layer of the MTP head."""
    import types

    from sgl_jax.srt.models import inkling

    holder = types.SimpleNamespace(config=config, source_root="model.mtp", mtp_layer_idx=index)
    holder.mtp_source_prefix = lambda: f"model.mtp.layers.{index}"
    return set(inkling.InklingMTPForCausalLM.create_weight_mappings(holder))


# Tensors that hold more than one target, so `load_stacked_weights` reads them
# instead of the mapping table.
STACKED_SUFFIXES = (
    ".mlp.w13_dn.weight",
    ".mlp.w2_md.weight",
    ".mlp.experts.w13_weight",
    ".mlp.experts.w2_weight",
    ".mlp.shared_experts.shared_w13_weight",
    ".mlp.shared_experts.shared_w2_weight",
)


def check_checkpoint_keys(failures):
    """Generated source keys against the published safetensors index. It needs the Hub."""
    import json

    from sgl_jax.srt.configs.inkling import InklingServingConfig

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [SKIP] checkpoint keys: huggingface_hub isn't installed")
        return

    for repo in PUBLISHED_REPOS:
        try:
            config_path = hf_hub_download(repo, "config.json")
            index_path = hf_hub_download(repo, "model.safetensors.index.json")
        except Exception as error:  # noqa: BLE001 - the Hub is optional here
            print(f"  [SKIP] checkpoint keys, {repo}: {type(error).__name__}")
            continue

        with open(config_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        raw.pop("model_type", None)
        text = InklingServingConfig(**raw).text_config
        with open(index_path, encoding="utf-8") as handle:
            published = set(json.load(handle)["weight_map"])

        generated = published_mappings(text)
        missing = sorted(generated - published)
        note(
            f"decoder keys, {repo}",
            not missing,
            f"{len(generated)} generated keys, all in the index"
            if not missing
            else f"{len(missing)} generated keys the index has no entry for: {missing[:3]}",
            failures,
        )

        decoder_keys = {key for key in published if key.startswith("model.llm.")}
        uncovered = decoder_keys - generated
        stacked = {key for key in uncovered if key.endswith(STACKED_SUFFIXES)}
        note(
            f"uncovered decoder keys are the stacked tensors, {repo}",
            uncovered == stacked,
            f"{len(uncovered)} uncovered, {len(stacked)} of them stacked",
            failures,
        )

        head_layers = int(raw["mtp_config"]["num_nextn_predict_layers"])
        mtp_missing = sorted(
            key
            for index in range(head_layers)
            for key in published_mtp_mappings(text, index) - published
            if key.startswith("model.mtp.")
        )
        note(
            f"MTP keys, {repo}",
            not mtp_missing,
            f"all {head_layers} head layers map onto index entries"
            if not mtp_missing
            else f"{len(mtp_missing)} missing: {mtp_missing[:3]}",
            failures,
        )

        # NEGATIVE CONTROL: bend one generated name. The comparison has to see
        # a key the index doesn't carry.
        bent = {key.replace("attn.wk_dv", "attn.wk_dvv") for key in generated}
        detected = bool(bent - published)
        print(
            f"      control (wk_dv renamed, {repo}):"
            f" -> {'detected' if detected else 'NOT DETECTED'}"
        )
        if not detected:
            failures.append(f"the index check for {repo} accepts a name that isn't there")


# --- negative controls ------------------------------------------------------


def run_controls(serving, worker_batch, model_path, baseline, failures):
    """Break one thing at a time and measure how far the prompt's logits move.

    Each runs the same prefill as a trial on the runner's pools. A control
    that moves nothing means the comparison reads past that part of the model,
    so the numbers above would prove nothing about it. The edits run on the
    engine's Explicit mesh, like every array they replace.
    """
    import jax

    with jax.set_mesh(serving.mesh):
        break_one_at_a_time(serving, worker_batch, model_path, baseline, failures)


def break_one_at_a_time(serving, worker_batch, model_path, baseline, failures):
    import jax
    import numpy as np
    import safetensors
    from flax import nnx
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    keep = slice(0, CONFIG["unpadded_vocab_size"])
    rows = slice(0, len(baseline))
    model = serving.model
    layers = model.model.layers

    def measure(label):
        logits, _ = serving.trial(worker_batch)
        moved = float(np.max(np.abs(np.asarray(logits)[rows, keep] - baseline[:, keep])))
        control(label, moved, failures)

    # One entry of one relative-bias bank.
    attention = layers[1].self_attn
    original = attention.rel_logits_proj.proj.value
    attention.rel_logits_proj.proj.value = original.at[0, 0].add(1.0)
    measure("relative bias, one entry")
    attention.rel_logits_proj.proj.value = original

    # The convolution after the MLP, disconnected.
    saved = layers[0].mlp_sconv
    layers[0].mlp_sconv = None
    measure("mlp_sconv removed")
    layers[0].mlp_sconv = saved

    # Gate and up swapped in the dense MLP, which a symmetric split hides.
    dense = layers[0].mlp
    gate, up = dense.gate_proj.weight.value, dense.up_proj.weight.value
    dense.gate_proj.weight.value, dense.up_proj.weight.value = up, gate
    measure("dense MLP gate/up swapped")
    dense.gate_proj.weight.value, dense.up_proj.weight.value = gate, up

    # The fused dense and routed tensors read as contiguous halves, the layout
    # the published checkpoint doesn't use.
    with safetensors.safe_open(
        os.path.join(model_path, "model.safetensors"), framework="np"
    ) as handle:
        dense_fused = handle.get_tensor("model.llm.layers.0.mlp.w13_dn.weight")
        routed_fused = handle.get_tensor("model.llm.layers.1.mlp.experts.w13_weight")
    half = dense_fused.shape[0] // 2
    dense.gate_proj.weight.value = model.put(dense_fused[:half].T, P(None, "tensor"))
    dense.up_proj.weight.value = model.put(dense_fused[half:].T, P(None, "tensor"))
    measure("dense gate/up read as halves")
    dense.gate_proj.weight.value, dense.up_proj.weight.value = gate, up

    experts = layers[1].mlp.experts
    wi_0, wi_1 = experts.wi_0.value, experts.wi_1.value
    sharding = NamedSharding(experts.moe_mesh, P("expert", None, "tensor"))
    half = routed_fused.shape[1] // 2
    for param, block in (
        (experts.wi_0, routed_fused[:, :half]),
        (experts.wi_1, routed_fused[:, half:]),
    ):
        array = np.ascontiguousarray(np.transpose(block, (0, 2, 1)).astype(np.float32))
        param.value = jax.make_array_from_callback(
            array.shape, sharding, lambda index, a=array: a[index]
        )
    measure("routed gate/up read as halves")
    experts.wi_0.value, experts.wi_1.value = wi_0, wi_1

    # Long-context logit scaling turned off on the global layer.
    floor = attention.log_scaling_n_floor
    attention.log_scaling_n_floor = None
    measure("log scaling turned off")
    attention.log_scaling_n_floor = floor

    # The sliding window widened to the whole prefix.
    local_attention = layers[0].self_attn
    window = local_attention.attn.sliding_window_size
    local_attention.attn.sliding_window_size = None
    measure("sliding window removed")
    local_attention.attn.sliding_window_size = window

    # The two attention convolutions swapped, which an order mistake in the
    # packed conv state would also produce.
    k_weight = local_attention.k_sconv.weight.value
    v_weight = local_attention.v_sconv.weight.value
    local_attention.k_sconv.weight.value = v_weight
    local_attention.v_sconv.weight.value = k_weight
    measure("k_sconv and v_sconv swapped")
    local_attention.k_sconv.weight.value = k_weight
    local_attention.v_sconv.weight.value = v_weight

    # The muP divide on the final hidden state.
    multiplier = model.mup_width_multiplier
    model.mup_width_multiplier = 1.0
    measure("muP width multiplier dropped")
    model.mup_width_multiplier = multiplier

    # The per-head query norm, which is what makes the 1/head_dim scale right.
    # Skipping it is what a port written from a Llama-shaped page does.
    class Skipped(nnx.Module):
        """Stands in for a norm a port left out."""

        def __call__(self, x):
            return x

    q_norm = attention.q_norm
    attention.q_norm = Skipped()
    measure("q_norm skipped")
    attention.q_norm = q_norm

    k_norm = attention.k_norm
    attention.k_norm = Skipped()
    measure("k_norm skipped")
    attention.k_norm = k_norm

    # The attention scale that goes with those two norms. 1/sqrt(head_dim) is
    # the value a port reaches for by habit.
    scaling = attention.attn.scaling
    attention.attn.scaling = 1.0 / math.sqrt(attention.head_dim)
    measure("attention scale 1/sqrt(head_dim)")
    attention.attn.scaling = scaling

    # The norm on the embedding, which layer 0 reads before anything else.
    embed_norm = model.model.embed_norm.scale.value
    model.model.embed_norm.scale.value = embed_norm * 1.5
    measure("embed_norm scaled")
    model.model.embed_norm.scale.value = embed_norm

    # One shared expert's gate row. With the sink the shared logits normalize
    # beside the chosen routed ones, so a shared row reaches the routed weights
    # and the logits with them.
    router = layers[1].mlp.router
    kernel = router.gate.kernel.value
    router.gate.kernel.value = kernel.at[:, CONFIG["n_routed_experts"]].add(1.0)
    measure("shared gate row")
    router.gate.kernel.value = kernel


def section(title: str, failures: list[str], check, *args):
    """Run one check. A crash inside it is a failure, and the run goes on."""
    print(title)
    try:
        return check(*args)
    except Exception as error:  # noqa: BLE001 - the crash is the finding
        import traceback

        traceback.print_exc()
        message = str(error).splitlines()[0] if str(error) else ""
        print(f"  [FAIL] {title}: {type(error).__name__}: {message}")
        failures.append(f"{title}: {type(error).__name__}: {message}")
        return None


def controls_batch(serving, reference, failures):
    """One fresh prompt through the scheduler's batch, run as a trial.

    It carries the contracts and the controls, so it has to match the
    reference on every prompt row first.
    """
    import numpy as np

    rng = np.random.default_rng(13)
    prompt = rng.integers(0, CONFIG["unpadded_vocab_size"], size=CHUNK).tolist()
    req = serving.request("controls", prompt)
    _, worker_batch = serving.prepare_extend([req])
    baseline, _ = serving.trial(worker_batch)
    baseline = np.asarray(baseline)[: len(prompt)]
    report(
        "every prompt row",
        baseline[:, : CONFIG["unpadded_vocab_size"]],
        reference_logits(reference, prompt),
        failures,
    )
    return req, worker_batch, baseline


# Base dependencies of sglang-jax that the engine imports and the other model
# tests don't need.
ENGINE_PACKAGES = ("pybase64", "llguidance")


def missing_engine_packages() -> list[str]:
    missing = []
    for name in ENGINE_PACKAGES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def run_engine(scratch, model_path, reference, failures, tree) -> None:
    """Every section that builds a `ModelWorker`."""
    serving = section(
        f"engine at --tp-size {TP}, checkpoint written by save_pretrained",
        failures,
        Serving,
        model_path,
        TP,
    )
    if serving is not None:
        section("serving", failures, check_serving, serving, reference, failures)
        section("chunked prompt", failures, check_chunked, serving, reference, failures)
        section(
            "the startup precompile",
            failures,
            check_precompiled,
            serving,
            f"serving at --tp-size {TP}",
            failures,
        )
        section("a batch unlike the precompile's", failures, control_precompiled, serving, failures)
        section("what the cache manager holds", failures, check_conv_state, serving, failures)
        section(
            "the weights the loader read",
            failures,
            check_layout,
            serving,
            reference,
            model_path,
            failures,
        )
        batch = section(
            "one prefill as a trial", failures, controls_batch, serving, reference, failures
        )
        if batch is not None:
            req, worker_batch, baseline = batch
            section("contracts", failures, check_contracts, serving, worker_batch, failures)
            section(
                "negative controls",
                failures,
                run_controls,
                serving,
                worker_batch,
                model_path,
                baseline,
                failures,
            )
            serving.release(req)
        section("entry classes", failures, check_entry_classes, serving, failures)
        section("the router", failures, check_shared_expert_sink, serving, failures)
        del serving
        retire()

    section("flags Inkling refuses", failures, check_refusals, model_path, scratch, failures)
    section(
        f"replicated KV heads at --tp-size {REPLICATION_TP}",
        failures,
        check_replication,
        scratch,
        failures,
    )
    section(
        f"the published structure at --tp-size {PUBLISHED_TINY_TP}, one process and two",
        failures,
        check_published_structure,
        scratch,
        tree,
        failures,
    )
    section(
        "expert parallel", failures, check_expert_parallel, model_path, reference, failures
    )
    section("the runner's own fields", failures, check_runtime_fields, model_path, failures)
    section("dummy weights", failures, check_dummy_load, model_path, failures)
    section(
        f"dummy weights at --tp-size {REPLICATION_TP}",
        failures,
        check_dummy_placement,
        scratch,
        failures,
    )
    section("the position-bias pass's temporaries", failures, check_attention_memory, failures)
    section(
        f"what each of {HOST_READS_PROCESSES} processes reads",
        failures,
        check_host_reads,
        scratch,
        tree,
        failures,
    )


def main() -> int:
    """Every section, in a scratch directory the run removes when it ends."""
    scratch = tempfile.mkdtemp(prefix="inkling-test-")
    try:
        return run_all(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_all(scratch: str) -> int:
    tree = patched_checkout(scratch)
    install(tree)

    failures: list[str] = []
    missing = missing_engine_packages()
    if missing:
        message = (
            f"the engine imports {' and '.join(missing)}, which this environment lacks."
            " Both are base dependencies of sglang-jax, and upstream/models/requirements.txt"
            " lists them: uv pip install -r upstream/models/requirements.txt"
        )
        print(f"engine\n  [FAIL] {message}")
        failures.append(message)
    else:
        model_path = os.path.join(scratch, "inkling-tiny")
        reference = write_checkpoint(CONFIG, model_path)
        run_engine(scratch, model_path, reference, failures, tree)

    section("published expert width", failures, check_expert_width, failures)
    section("an older transformers", failures, check_old_transformers, tree, failures)
    section("published checkpoint keys", failures, check_checkpoint_keys, failures)

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(" ", failure)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--host-reads-child"]:
        raise SystemExit(host_reads_child(sys.argv[1:]))
    raise SystemExit(main())
