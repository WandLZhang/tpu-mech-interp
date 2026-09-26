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

"""Correctness gate for the Nemotron 3 model patch and its capture hook.

Runs on CPU with a forced 8-device mesh. No TPU needed.

    python3 upstream/models/test_nemotron_h_model.py

The reference is the HuggingFace `transformers` implementation of
`nemotron_h`, which is independent code: a different framework, a different
chunked-scan formulation, and a dense attention rather than a paged one. The
test builds a tiny random config, instantiates both models, copies one set of
random weights into both, and runs one forward in float32.

Fifteen checks. Every one drives the patched source.

1. `scripts/cpu_engine.py` builds the tree the TPU VM serves, `sglang-jax` at
   `SGL_COMMIT` (default `eb061d8`) with `sglang-jax-877.patch` and both
   steering patches. Both patches apply on it, and every file they touch
   compiles.
2. Prefill, one request: every layer's residual stream, the final hidden state
   and the logits against the reference, read out of `NemotronHModel.__call__`
   itself. A hand-written walk over the same blocks pins the capture hook.
3. Prefill, two packed requests of different lengths, which exercises the
   chunk grid and the partial last chunk.
4. Decode: one token on top of the prefill state, against the reference's full
   forward over the same tokens.
5. The chunked scan against a token-by-token recurrence written in this file.
6. A second config in the other spelling of the layer field, with a dense MLP
   block, a tied head and an MTP head the served stack leaves out.
7. Cross-request isolation: one request decoded on top of a packed prefill
   against the same request run on its own.
8. The published checkpoint key names load through `_create_weight_mappings`
   and `WeightLoader`, and a four-way tensor axis matches a one-way one.
9. The published `config.json` of both sizes resolves through `AutoConfig`,
   and so does a config stock `transformers` saved under its own block names.
10. `--enable-recurrent-extra-buffer`: the track slot the scheduler picks
    holds the state the forward leaves, and a prefix hit that resumes from it
    matches a fresh prefill.
11. An inf or NaN in one request's cached state reaches none of the requests
    packed around it.
12. `int8.yaml` reaches the routed experts, int8 experts 128 wide match
    numpy, and a linear-only int8 config runs a packed prefill.
13. `--model-layer-nums` and a `num_hidden_layers` override cut the stack to
    its first three blocks, and the cut model matches the whole one there.
14. `--ep-dispatch-algorithm`, `--init-expert-location` and
    `--ep-num-redundant-experts` stop the model build, and the defaults build.
15. The runner resolves `nemotron_h` to its own recurrent config, builds
    `Mamba2AttnBackend` for it, and counts the state bytes a pool slot holds.

The forwards in checks 2 to 8 and 13 put a dense attention stand-in behind
`RadixAttention`. Checks 10 to 12 run a stack with no attention block through
the engine's own pools, metadata and jit, the path the runner takes.

Every comparison reads a non-finite value on either side as infinity, and
every gate passes a number only when it compares true, so a NaN fails the
check that meets it.

Every check carries a negative control. Check 1 refuses a patch with one
context line rewritten, against a tree that patch hasn't touched. Check 5
feeds the chunked side a decay 10% off. The capture check compares the
captured streams one layer out of line. Twenty-three mutants follow, each
breaking one weight, one connection or one line of the patched source, and
each naming the check it has to move; a check no mutant names fails the run.
Three of them turn the attention output, the Mamba-2 decode step and the
split-group norm of the tp=4 path into NaN.

Point `SGLANG_JAX_REPO` at a clone that holds `eb061d8` to skip the download.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import types

# Must be set before jax initializes. The device count goes in beside any flag XLA_FLAGS already
# holds, where setdefault would drop it, and a count the caller set wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=8".strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
PATCHES = ["nemotron3-model.patch", "nemotron3-capture-hook.patch"]


def load_by_path(name: str, path: str):
    """A module loaded from its file, whatever `sys.path` holds."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# `build_tree` builds the stack `bootstrap_tpu_vm.sh` serves from, and
# `corrupt` rewrites one context line of a patch for the check 1 control.
cpu_engine = load_by_path(
    "scripts_cpu_engine", os.path.join(REPO_ROOT, "scripts", "cpu_engine.py")
)
capture_hooks = load_by_path(
    "test_capture_hooks",
    os.path.join(REPO_ROOT, "upstream", "capture-hooks", "test_capture_hooks.py"),
)

# One threshold decides both directions: every clean number has to sit below
# it and every mutant above it. Both sides run float32. The clean run lands at
# 1.3e-07, which is 75x under, and the weakest mutant at 3.4e-05, which is
# 3.4x over. The floor is float32 roundoff; the ceiling is how small a change
# the weakest mutant makes, so a tighter threshold would buy margin above at
# the cost of margin below.
TOL = 1e-5

# Four layers of 64 hidden, with the flags the published configs set: the same
# block types, one group of experts, a latent narrower than hidden, a
# squared-ReLU expert, two KV heads, and a chunk the prompt doesn't fill.
TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=48,
    hybrid_override_pattern="ME*E",
    mamba_num_heads=4,
    mamba_head_dim=8,
    ssm_state_size=16,
    n_groups=2,
    conv_kernel=4,
    chunk_size=8,
    num_attention_heads=4,
    head_dim=16,
    num_key_value_heads=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    moe_intermediate_size=24,
    moe_latent_size=32,
    moe_shared_expert_intermediate_size=40,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=5.0,
    norm_topk_prob=True,
    num_nextn_predict_layers=0,
    tie_word_embeddings=False,
)

# The other spelling of the layer field, a dense MLP block, a tied head, and
# an MTP head the served stack leaves out. Ultra ships the list spelling and
# both published sizes ship the head.
TINY_MLP = dict(
    TINY,
    hybrid_override_pattern=None,
    layers_block_type=["mamba", "moe", "attention", "mlp"],
    num_nextn_predict_layers=1,
    mtp_layers_block_type=["attention", "moe"],
    tie_word_embeddings=True,
)
del TINY_MLP["hybrid_override_pattern"]

# Four KV heads, which is what `configure_for_tensor_parallel` leaves behind
# when it replicates the published two up to a four-way tensor axis.
TINY_TP = dict(TINY, num_key_value_heads=4)

# Mamba-2, LatentMoE, Mamba-2, dense MLP. No attention block, so a forward
# needs no KV and the recurrent pool is the only state one request can hand
# another. The second Mamba-2 layer reads what the first one wrote.
TINY_RECURRENT = dict(TINY, hybrid_override_pattern="MEM-")

# `--enable-recurrent-extra-buffer` wants a page size above 1 and a track
# interval that's a multiple of it.
TRACK_PAGE = 16
TRACK_INTERVAL = 16

# Check 13 cuts TINY's `ME*E` to its first three blocks, `ME*`, so the cut
# stack still holds an attention block.
CUT = 3


# ---------------------------------------------------------------------------
# Check 1: the stack builds, the patches apply on it, and a rewritten context
# line doesn't.
# ---------------------------------------------------------------------------


def touched_files(repo: str, patch: str) -> list[str]:
    """Every path `patch` touches, as `git apply --numstat` lists it."""
    listing = subprocess.run(
        ["git", "apply", "--numstat", patch], cwd=repo, capture_output=True, text=True
    )
    if listing.returncode != 0:
        raise AssertionError(f"git apply --numstat {patch} failed:\n{listing.stderr}")
    # One "added<TAB>deleted<TAB>path" row per file.
    return [row.split("\t")[2] for row in listing.stdout.splitlines() if row.strip()]


def apply_patches(repo: str, workdir: str, patches) -> list[str]:
    """Apply each patch in order and return every path they touch.

    Before a patch goes on, the tree carries the patches before it and nothing
    else, so `git apply --check` has to accept it and refuse a copy with one
    context line rewritten.
    """
    touched = []
    for path in patches:
        name = os.path.basename(path)
        clean = subprocess.run(
            ["git", "apply", "--check", path], cwd=repo, capture_output=True, text=True
        )
        if clean.returncode != 0:
            raise AssertionError(f"{name} doesn't apply: {clean.stderr}")

        with open(path) as handle:
            broken = capture_hooks.corrupt(handle.read())
        broken_path = os.path.join(workdir, f"broken-{name}")
        with open(broken_path, "w") as handle:
            handle.write(broken)
        refused = subprocess.run(
            ["git", "apply", "--check", broken_path], cwd=repo, capture_output=True
        )
        if refused.returncode == 0:
            raise AssertionError(f"control: {name} with a rewritten context line applied")

        touched += [rel for rel in touched_files(repo, path) if rel not in touched]
        subprocess.run(["git", "apply", path], cwd=repo, check=True)
        print(f"  applied {name}, and its rewritten context line was refused")
    return touched


def prepare_repo(workdir: str, patches=None) -> str:
    """Build the served stack, apply both patches on it, and compile what they touch.

    `cpu_engine.build_tree` clones `SGLANG_JAX_REPO` with `--shared`, or
    `sglang-jax` from GitHub, checks out `SGL_COMMIT`, and applies 877 and both
    steering patches the way `bootstrap_tpu_vm.sh` does. A model patch that
    collides with that stack fails here. Every `.py` file either patch touches
    has to compile, the runner files no later check imports among them.
    """
    if patches is None:
        patches = [os.path.join(HERE, name) for name in PATCHES]
    source = os.environ.get("SGLANG_JAX_REPO") or None
    repo = cpu_engine.build_tree(os.path.join(workdir, "sglang-jax"), source)
    stack = subprocess.run(
        ["git", "-C", repo, "log", "--format=%h %s", "-3"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    print(f"  built sglang-jax at {cpu_engine.SGL_COMMIT} with the capture and steering patches:")
    for line in stack.splitlines():
        print(f"    {line}")

    touched = apply_patches(repo, workdir, patches)
    sources = sorted(rel for rel in touched if rel.endswith(".py"))
    for rel in sources:
        subprocess.run([sys.executable, "-m", "py_compile", os.path.join(repo, rel)], check=True)
    print(f"  {len(sources)} touched files compile:")
    for rel in sources:
        print(f"    {rel}")
    return repo


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class DenseCausalAttention:
    """Stand-in for the TPU flash backend: dense causal GQA in float32.

    Holds its own per-request KV so the decode step attends over the prefill.
    The paged cache is engine plumbing, not model math. Everything the model
    file owns around it, the projections and the head layout, is under test.
    """

    def __init__(self, history=None):
        # {layer_id: [(k, v) per request]}
        self.history = {} if history is None else history

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        full = lambda x: jax.sharding.reshard(x, P(None, None, None))
        q, k, v = full(q), full(k), full(v)
        decode = forward_batch.forward_mode == ForwardMode.DECODE
        seq_lens = forward_batch.request_lens
        past = self.history.setdefault(layer.layer_id, [None] * len(seq_lens))

        outs = []
        start = 0
        for request, length in enumerate(seq_lens):
            qs = q[start : start + length].astype(jnp.float32)
            ks = k[start : start + length].astype(jnp.float32)
            vs = v[start : start + length].astype(jnp.float32)
            if decode and past[request] is not None:
                ks = jnp.concatenate([past[request][0], ks], axis=0)
                vs = jnp.concatenate([past[request][1], vs], axis=0)
            past[request] = (ks, vs)

            repeat = layer.q_head_num // layer.kv_head_num
            kr = jnp.repeat(ks, repeat, axis=1, out_sharding=P(None, None, None))
            vr = jnp.repeat(vs, repeat, axis=1, out_sharding=P(None, None, None))
            scores = jnp.einsum("qhd,khd->hqk", qs, kr) * layer.scaling
            keys = kr.shape[0]
            offset = keys - length
            rows = jnp.arange(length)[:, None] + offset
            mask = rows >= jnp.arange(keys)[None, :]
            probs = jax.nn.softmax(jnp.where(mask[None], scores, -jnp.inf), axis=-1)
            outs.append(jnp.einsum("hqk,khd->qhd", probs, vr).reshape(length, -1))
            start += length
        return jnp.concatenate(outs, axis=0), None

    def get_forward_metadata(self, batch):
        return None


class Harness:
    """A tiny Nemotron 3 on one mesh, plus the reference beside it."""

    def __init__(self, repo: str, seed: int = 0, config: dict | None = None, tp: int = 1):
        sys.path.insert(0, os.path.join(repo, "python"))
        from sgl_jax.srt.configs.nemotron_h import NemotronHConfig

        self.settings = TINY if config is None else config
        self.cfg = NemotronHConfig(**self.settings)
        self.tp = tp
        self.mesh = Mesh(
            np.array(jax.devices()[:tp]).reshape(1, tp),
            axis_names=("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        self.reference = build_reference(seed, self.settings)
        self.model = self.build_model()
        self.copy_weights()

    def build_model(self):
        from sgl_jax.srt.models.nemotron_h import NemotronHForCausalLM

        with jax.set_mesh(self.mesh):
            return NemotronHForCausalLM(self.cfg, mesh=self.mesh, dtype=jnp.float32)

    @property
    def head(self):
        """The output embedding, tied or not."""
        if getattr(self.cfg, "tie_word_embeddings", False):
            return self.model.model.embed_tokens.embedding
        return self.model.lm_head.embedding

    def new_pool(self, size: int = 8):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        params = self.cfg.linear_state_params
        return RecurrentStatePool(
            linear_recurrent_layer_ids=params.layers,
            size=size,
            num_heads=params.num_heads,
            head_dim=params.head_dim,
            conv_kernel_size=params.conv_kernel_size,
            mesh=self.mesh,
            temporal_dtype=params.dtype.temporal,
            conv_dtype=params.dtype.conv,
            num_k_heads=params.num_k_heads,
            head_k_dim=params.head_k_dim,
        )

    def batch(
        self,
        ids,
        seq_lens,
        pool,
        decode=False,
        carry_state=False,
        attention_history=None,
        slots=None,
    ):
        """A forward batch, its pools, and the backend, for one call."""
        from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
            HybridLinearAttnBackendMetadata,
            LinearRecurrentAttnBackendMetadata,
        )
        from sgl_jax.srt.layers.attention.mamba.mamba2_backend import Mamba2AttnBackend
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        backend = HybridLinearAttnBackend(
            full_attn_backend=DenseCausalAttention(attention_history),
            linear_attn_backend=Mamba2AttnBackend(
                mamba_config=self.cfg.mamba_config(),
                mesh=self.mesh,
                activation=self.cfg.mamba_hidden_act,
            ),
            full_attn_layers=self.cfg.full_attention_layer_ids,
        )
        count = len(seq_lens)
        cumulative = np.zeros(count + 1, np.int32)
        cumulative[1:] = np.cumsum(seq_lens)
        on_data = lambda x: jax.device_put(
            jnp.asarray(x), jax.sharding.NamedSharding(self.mesh, P("data"))
        )
        # Slot 0 is the per-rank dummy, so requests start at 1.
        if slots is None:
            slots = list(range(1, count + 1))
        backend.forward_metadata = HybridLinearAttnBackendMetadata(
            full_attn_metadata=None,
            linear_attn_metadata=LinearRecurrentAttnBackendMetadata(
                cu_q_lens=on_data(cumulative),
                recurrent_indices=on_data(np.asarray(slots, dtype=np.int32)),
                has_initial_state=on_data(np.full((count,), carry_state, bool)),
            ),
        )
        positions = np.concatenate([np.arange(n) for n in seq_lens]).astype(np.int32)
        forward_batch = types.SimpleNamespace(
            input_ids=jnp.asarray(ids, jnp.int32),
            positions=jnp.asarray(positions),
            input_embedding=None,
            attn_backend=backend,
            expert_location_metadata=None,
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            # The stand-in attention reads the request split from here; the
            # real backend reads it out of its own metadata.
            request_lens=list(seq_lens),
        )
        pools = types.SimpleNamespace(recurrent_state_pool=pool, token_to_kv_pool=None)
        return forward_batch, pools

    def run(self, ids, seq_lens, pool, **kwargs):
        """Per-layer residual streams, the final hidden state, and the logits.

        Everything comes out of `NemotronHModel.__call__`: the capture hook
        returns the stream entering each block, and the first return value is
        the state after the last residual add and the final norm.
        """
        forward_batch, pools = self.batch(ids, seq_lens, pool, **kwargs)
        model = self.model.model
        model.layers_to_capture = list(range(self.cfg.num_hidden_layers))
        with jax.set_mesh(self.mesh):
            hidden, captured, _, (recurrent, conv), topk_ids = model(forward_batch, pools)
            # ParallelLMHead shards the vocab axis over ("data", "tensor"), and the tokens already
            # sit on "data", so a bare product would map "data" twice. These logits only meet the
            # HuggingFace reference, so replicate the head first.
            head = jax.sharding.reshard(self.head[...], P(None, None))
            logits = hidden @ head.T
        model.layers_to_capture = []
        if len(topk_ids) != self.cfg.num_hidden_layers:
            raise AssertionError(
                f"the model returned {len(topk_ids)} routing entries for "
                f"{self.cfg.num_hidden_layers} layers; the capturer indexes "
                "that list by layer id"
            )
        pool.replace_buffer((recurrent, conv))
        return (
            [np.asarray(state) for state in captured],
            np.asarray(hidden),
            np.asarray(logits),
        )

    def walk(self, ids, seq_lens, pool, **kwargs):
        """The same stack, stepped block by block from outside the model.

        This is the independent path the capture hook is pinned against. It
        reads no aux output, so a hook that captures the wrong value shows up
        as a mismatch.
        """
        forward_batch, pools = self.batch(ids, seq_lens, pool, **kwargs)
        model = self.model.model
        with jax.set_mesh(self.mesh):
            hidden = model.embed_tokens(forward_batch.input_ids)
            residual = None
            per_layer = []
            for layer in model.layers:
                per_layer.append(
                    np.asarray(hidden + residual if residual is not None else hidden)
                )
                hidden, residual, _, _ = layer(
                    hidden, forward_batch, pools, residual, dispatch_info=None
                )
        return per_layer

    def copy_weights(self):
        reference = dict(self.reference.named_parameters())
        reference.update(dict(self.reference.named_buffers()))
        take = lambda key: jnp.asarray(reference[key].detach().numpy())
        model = self.model

        put(model.model.embed_tokens.embedding, take("model.embeddings.weight"))
        put(model.model.norm.scale, take("model.norm_f.weight"))
        if not getattr(self.cfg, "tie_word_embeddings", False):
            put(model.lm_head.embedding, take("lm_head.weight"))

        for index, block in enumerate(self.cfg.layers_block_type):
            layer = model.model.layers[index]
            source = f"model.layers.{index}"
            put(layer.norm.scale, take(f"{source}.norm.weight"))
            mixer = layer.mixer
            if block == "mamba":
                put(mixer.in_proj.weight, take(f"{source}.mixer.in_proj.weight").T)
                put(mixer.conv1d_weight, take(f"{source}.mixer.conv1d.weight")[:, 0, :])
                put(mixer.conv1d_bias, take(f"{source}.mixer.conv1d.bias"))
                put(mixer.a_log, take(f"{source}.mixer.A_log"))
                put(mixer.dt_bias, take(f"{source}.mixer.dt_bias"))
                put(mixer.d, take(f"{source}.mixer.D"))
                put(mixer.norm.weight, take(f"{source}.mixer.norm.weight"))
                put(mixer.out_proj.weight, take(f"{source}.mixer.out_proj.weight").T)
            elif block == "attention":
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    put(getattr(mixer, name).weight, take(f"{source}.mixer.{name}.weight").T)
            elif block == "moe":
                put(mixer.moe_gate.kernel, take(f"{source}.mixer.gate.weight").T)
                put(
                    mixer.moe_gate.bias,
                    take(f"{source}.mixer.gate.e_score_correction_bias"),
                )
                put(
                    mixer.fc1_latent_proj.weight,
                    take(f"{source}.mixer.fc1_latent_proj.weight").T,
                )
                put(
                    mixer.fc2_latent_proj.weight,
                    take(f"{source}.mixer.fc2_latent_proj.weight").T,
                )
                put(
                    mixer.shared_experts.up_proj.weight,
                    take(f"{source}.mixer.shared_experts.up_proj.weight").T,
                )
                put(
                    mixer.shared_experts.down_proj.weight,
                    take(f"{source}.mixer.shared_experts.down_proj.weight").T,
                )
                # The reference stacks experts as [E, out, in]; EPMoE wants
                # [E, in, out].
                put(mixer.experts.wi_0, jnp.swapaxes(take(f"{source}.mixer.experts.up_proj"), 1, 2))
                put(mixer.experts.wo, jnp.swapaxes(take(f"{source}.mixer.experts.down_proj"), 1, 2))
            else:
                put(mixer.up_proj.weight, take(f"{source}.mixer.up_proj.weight").T)
                put(mixer.down_proj.weight, take(f"{source}.mixer.down_proj.weight").T)


def put(param, array):
    """Write a value without dropping the layout the parameter was built on."""
    current = param[...]
    param.set_value(jax.device_put(jnp.asarray(array, current.dtype), current.sharding))


def build_reference(seed: int, settings: dict):
    """The HuggingFace model, with random weights rather than the shipped init."""
    from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

    torch.manual_seed(seed)
    model = NemotronHForCausalLM(NemotronHConfig(dtype="float32", **settings))
    model = model.to(torch.float32).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        for name, buffer in model.named_buffers():
            if "e_score_correction_bias" in name:
                buffer.copy_(torch.randn(buffer.shape, generator=generator) * 0.1)
        if settings.get("tie_word_embeddings"):
            # This reference keeps `lm_head` as its own parameter whatever the
            # flag says, so a tied checkpoint has to be written by hand.
            model.lm_head.weight.copy_(model.model.embeddings.weight)
    return model


def reference_forward(model, ids):
    with torch.no_grad():
        batch = torch.tensor(np.asarray(ids), dtype=torch.long)[None]
        out = model(batch, output_hidden_states=True)
    hidden = [state[0].numpy() for state in out.hidden_states]
    # The reference appends the post-norm state last; the ones before it are
    # the residual stream entering each block.
    return hidden[:-1], hidden[-1], out.logits[0].numpy()


def gap(mine, theirs) -> float:
    """Max absolute difference, and infinity when either side isn't finite."""
    mine, theirs = np.asarray(mine), np.asarray(theirs)
    if not (np.all(np.isfinite(mine)) and np.all(np.isfinite(theirs))):
        return math.inf
    return float(np.abs(mine - theirs).max())


def worst_of(errors) -> float:
    """The largest error, and infinity when any of them isn't a number.

    Python's `max` drops a NaN that isn't its first argument, since every
    comparison with NaN is false, so a fold over errors needs this instead.
    """
    values = [float(error) for error in errors]
    if any(math.isnan(value) for value in values):
        return math.inf
    return max(values)


def report(name, mine, theirs, quiet=False):
    """Max absolute error, infinity when either side isn't finite, and the correlation."""
    error = gap(mine, theirs)
    if not quiet:
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = float(np.corrcoef(np.ravel(mine), np.ravel(theirs))[0, 1])
        print(f"    {name:<28} max abs {error:.3e}   corr {correlation:.10f}")
    return error


# ---------------------------------------------------------------------------
# Checks 2 to 4
# ---------------------------------------------------------------------------


def compare(harness, ids, seq_lens, label, quiet=False):
    """One packed forward against the reference, layer by layer."""
    pool = harness.new_pool()
    mine_layers, mine_final, mine_logits = harness.run(ids, seq_lens, pool)

    their_layers, their_final, their_logits = [], [], []
    start = 0
    for length in seq_lens:
        layers, final, logits = reference_forward(harness.reference, ids[start : start + length])
        their_layers.append(layers)
        their_final.append(final)
        their_logits.append(logits)
        start += length
    their_layers = [np.concatenate(group, axis=0) for group in zip(*their_layers)]
    their_final = np.concatenate(their_final, axis=0)
    their_logits = np.concatenate(their_logits, axis=0)

    if not quiet:
        print(f"  {label}")
    errors = []
    for index, (mine, theirs) in enumerate(zip(mine_layers, their_layers)):
        block = harness.cfg.layers_block_type[index] if index else "embedding"
        errors.append(report(f"layer {index} in ({block})", mine, theirs, quiet))
    errors.append(report("final norm", mine_final, their_final, quiet))
    errors.append(report("logits", mine_logits, their_logits, quiet))
    return worst_of(errors)


def compare_decode(harness, ids, label, quiet=False):
    """Prefill, then one more token through the decode path."""
    pool = harness.new_pool()
    history = {}
    harness.run(ids[:-1], [len(ids) - 1], pool, attention_history=history)
    _, _, mine_logits = harness.run(
        ids[-1:], [1], pool, decode=True, carry_state=True, attention_history=history
    )
    _, _, their_logits = reference_forward(harness.reference, ids)

    if not quiet:
        print(f"  {label}")
    return report("logits", mine_logits[-1], their_logits[-1], quiet)


def capture_hook_matches(harness, ids, seq_lens, quiet=False):
    """The hook returns the stream the hand-written walk sees entering a block."""
    pool = harness.new_pool()
    captured, _, _ = harness.run(ids, seq_lens, pool)
    expected = harness.walk(ids, seq_lens, harness.new_pool())

    if len(captured) != len(expected):
        raise AssertionError(f"hook returned {len(captured)} layers, expected {len(expected)}")
    worst = worst_of(gap(a, b) for a, b in zip(captured, expected))
    # The control: the streams have to differ layer to layer, or matching them
    # proves nothing about which layer the hook read.
    shifted = min(gap(a, b) for a, b in zip(captured[1:], expected[:-1]))
    if not quiet:
        print(
            f"  capture hook: {len(captured)} layers, max abs {worst:.3e}, "
            f"one layer off {shifted:.3e}"
        )
    if not shifted > TOL:
        raise AssertionError("control: the residual streams don't differ layer to layer")
    return worst


# ---------------------------------------------------------------------------
# Check 5: the chunked scan against a token-by-token recurrence.
# ---------------------------------------------------------------------------


def token_by_token(x, b, dt, log_decay, heads):
    """The Mamba-2 recurrence, one token at a time, in plain numpy.

    Written from the recurrence rather than from the engine, so it shares no
    chunking, no decay weighting and no einsum with the thing it checks:

        h_t = exp(dt_t A_h) h_{t-1} + B_t (dt_t x_t)^T

    Args:
      x: [T, H, P] the SSM input.
      b: [T, G, N] per group.
      dt: [T, H]
      log_decay: [T, H]
      heads: H.

    Returns:
      [T, H, N, P] the state after every token.
    """
    tokens, _, head_dim = x.shape
    groups, state = b.shape[1], b.shape[2]
    per_group = heads // groups
    h = np.zeros((heads, state, head_dim), np.float64)
    out = []
    for t in range(tokens):
        for head in range(heads):
            row = b[t, head // per_group]
            h[head] = math.exp(log_decay[t, head]) * h[head] + np.outer(
                row, dt[t, head] * x[t, head]
            )
        out.append(h.copy())
    return np.stack(out)


def chunked_scan_matches(scale_decay: float = 1.0, quiet: bool = False):
    """`chunk_terms` and `replay_scalar` against the token loop above.

    `scale_decay` moves the decay the chunked side sees and leaves the token
    loop alone, which is the control: an error inside the chunk has to show.
    """
    from sgl_jax.srt.layers.attention.mamba.mamba2 import chunk_terms, discretize
    from sgl_jax.srt.layers.attention.mamba.mamba2_backend import replay_scalar

    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    tokens, heads, head_dim, groups, state, chunk = 32, 4, 8, 2, 16, 8
    x = jax.random.normal(keys[0], (tokens, heads, head_dim))
    b = jax.random.normal(keys[1], (tokens, groups, state))
    dt_raw = jax.random.normal(keys[2], (tokens, heads))
    dt_bias = jax.random.normal(keys[3], (heads,))
    a_log = jax.random.normal(keys[4], (heads,))
    dt, log_decay = discretize(dt_raw, dt_bias, a_log)
    h0 = jnp.zeros((heads, state, head_dim))

    log_total, b_chunks = chunk_terms(x, b, dt, log_decay * scale_decay, chunk)
    mine = np.asarray(replay_scalar(log_total, b_chunks, h0))

    theirs = token_by_token(
        np.asarray(x, np.float64),
        np.asarray(b, np.float64),
        np.asarray(dt, np.float64),
        np.asarray(log_decay, np.float64),
        heads,
    )[chunk - 1 :: chunk]
    # The state runs an order of magnitude above the residual streams the
    # other checks read, so measure it against its own size.
    scale = float(np.abs(theirs).max())
    return report("chunk boundary states", mine / scale, theirs / scale, quiet)


def discretize_honors_the_limit(quiet=False):
    """`discretize` clamps `dt` to both ends of the pair it's handed.

    The published checkpoints leave `mamba_dt_limit` at `(0.0, inf)`, where
    neither end binds. The reference floors prefill `dt` at `time_step_min`
    instead, but this file's random weights keep `dt` above that floor, so the
    model comparison can't tell one clamp from another. This reads the clamp
    directly.
    """
    from sgl_jax.srt.layers.attention.mamba.mamba2 import discretize

    raw = jnp.array([[-20.0, 0.0, 20.0]])
    zeros = jnp.zeros((3,))
    free, _ = discretize(raw, zeros, zeros, (0.0, math.inf))
    bound, _ = discretize(raw, zeros, zeros, (0.5, 2.0))
    free, bound = np.asarray(free)[0], np.asarray(bound)[0]
    softplus = np.log1p(np.exp(-np.abs(np.asarray(raw)[0]))) + np.maximum(
        np.asarray(raw)[0], 0.0
    )
    if not np.allclose(free, softplus, atol=1e-6):
        raise AssertionError(f"an open limit changed dt: {free} against {softplus}")
    if not np.allclose(bound, np.clip(softplus, 0.5, 2.0), atol=1e-6):
        raise AssertionError(f"a finite limit gave {bound}")
    if not quiet:
        print(f"    dt at (0, inf) {free.round(4)}, at (0.5, 2) {bound.round(4)}")


# ---------------------------------------------------------------------------
# Check 7: one request's state doesn't leak into the next.
# ---------------------------------------------------------------------------


def cached_state(pool, slot):
    """Every recurrent slot one request owns, as one flat array."""
    pieces = [np.asarray(buf)[slot].ravel() for buf in pool.recurrent_buffers]
    pieces += [np.asarray(pair[0])[slot].ravel() for pair in pool.conv_buffers]
    return np.concatenate(pieces)


def request_isolation(harness, ids, quiet=False):
    """The second request of a packed prefill against the same request alone.

    The Mamba-2 scan runs one carry over the whole packed batch, so the
    request before this one is in that carry until `replay_scalar` drops it
    at this request's first chunk. A carry that stayed would reach only the
    state written back to the pool. `h_entering` takes the cache for the
    outputs of a request's first chunk, and this request's 6 tokens fit in
    one chunk. So this reads the pool as well as the logits a decode step on
    top of it produces.
    """
    together = harness.new_pool()
    history = {}
    harness.run(ids, [10, 6], together, attention_history=history)

    alone = harness.new_pool()
    solo_history = {}
    harness.run(ids[10:16], [6], alone, attention_history=solo_history)

    state = report(
        "request 2 cached state", cached_state(together, 2), cached_state(alone, 1), quiet
    )

    _, _, packed = harness.run(
        ids[[9, 15]],
        [1, 1],
        together,
        decode=True,
        carry_state=True,
        attention_history=history,
    )
    _, _, single = harness.run(
        ids[15:16],
        [1],
        alone,
        decode=True,
        carry_state=True,
        attention_history=solo_history,
    )
    return worst_of([state, report("request 2 logits", packed[1], single[0], quiet)])


# ---------------------------------------------------------------------------
# Check 8: the published key names load, and a wider mesh agrees.
# ---------------------------------------------------------------------------


def checkpoint_shapes(harness) -> dict:
    """Reference-side shape for every key `_create_weight_mappings` names."""
    cfg = harness.cfg
    shapes = {
        "backbone.embeddings.weight": (cfg.vocab_size, cfg.hidden_size),
        "backbone.norm_f.weight": (cfg.hidden_size,),
    }
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (cfg.vocab_size, cfg.hidden_size)
    for layer_id, block in enumerate(cfg.layers_block_type):
        prefix = f"backbone.layers.{layer_id}"
        shapes[f"{prefix}.norm.weight"] = (cfg.hidden_size,)
        if block == "mamba":
            shapes.update(
                {
                    f"{prefix}.mixer.in_proj.weight": (
                        cfg.mamba_in_proj_size,
                        cfg.hidden_size,
                    ),
                    # The checkpoint ships the depthwise kernel channel-first
                    # with a width-one input axis.
                    f"{prefix}.mixer.conv1d.weight": (cfg.conv_dim, 1, cfg.conv_kernel),
                    f"{prefix}.mixer.conv1d.bias": (cfg.conv_dim,),
                    f"{prefix}.mixer.A_log": (cfg.mamba_num_heads,),
                    f"{prefix}.mixer.dt_bias": (cfg.mamba_num_heads,),
                    f"{prefix}.mixer.D": (cfg.mamba_num_heads,),
                    f"{prefix}.mixer.norm.weight": (cfg.mamba_intermediate_size,),
                    f"{prefix}.mixer.out_proj.weight": (
                        cfg.hidden_size,
                        cfg.mamba_intermediate_size,
                    ),
                }
            )
        elif block == "attention":
            width = cfg.num_attention_heads * cfg.head_dim
            kv_width = cfg.num_key_value_heads * cfg.head_dim
            shapes.update(
                {
                    f"{prefix}.mixer.q_proj.weight": (width, cfg.hidden_size),
                    f"{prefix}.mixer.k_proj.weight": (kv_width, cfg.hidden_size),
                    f"{prefix}.mixer.v_proj.weight": (kv_width, cfg.hidden_size),
                    f"{prefix}.mixer.o_proj.weight": (cfg.hidden_size, width),
                }
            )
        elif block == "moe":
            latent = cfg.expert_input_size
            shapes.update(
                {
                    f"{prefix}.mixer.gate.weight": (
                        cfg.n_routed_experts,
                        cfg.hidden_size,
                    ),
                    f"{prefix}.mixer.gate.e_score_correction_bias": (
                        cfg.n_routed_experts,
                    ),
                    f"{prefix}.mixer.fc1_latent_proj.weight": (latent, cfg.hidden_size),
                    f"{prefix}.mixer.fc2_latent_proj.weight": (cfg.hidden_size, latent),
                    f"{prefix}.mixer.shared_experts.up_proj.weight": (
                        cfg.moe_shared_expert_intermediate_size,
                        cfg.hidden_size,
                    ),
                    f"{prefix}.mixer.shared_experts.down_proj.weight": (
                        cfg.hidden_size,
                        cfg.moe_shared_expert_intermediate_size,
                    ),
                }
            )
            for expert in range(cfg.n_routed_experts):
                shapes[f"{prefix}.mixer.experts.{expert}.up_proj.weight"] = (
                    cfg.moe_intermediate_size,
                    latent,
                )
                shapes[f"{prefix}.mixer.experts.{expert}.down_proj.weight"] = (
                    latent,
                    cfg.moe_intermediate_size,
                )
        else:
            shapes[f"{prefix}.mixer.up_proj.weight"] = (
                cfg.intermediate_size,
                cfg.hidden_size,
            )
            shapes[f"{prefix}.mixer.down_proj.weight"] = (
                cfg.hidden_size,
                cfg.intermediate_size,
            )
    return shapes


def write_checkpoint(harness, path: str, scale: float = 1.0) -> dict:
    """A random checkpoint under the published key names, with its config.json.

    Returns the tensors it wrote.
    """
    from safetensors.numpy import save_file

    os.makedirs(path, exist_ok=True)
    settings = dict(harness.settings)
    settings["model_type"] = "nemotron_h"
    settings["architectures"] = ["NemotronHForCausalLM"]
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(settings, handle)

    rng = np.random.default_rng(7)
    source = {
        name: (rng.standard_normal(shape) * scale).astype(np.float32)
        for name, shape in checkpoint_shapes(harness).items()
    }
    save_file(source, os.path.join(path, "model.safetensors"))
    return source


def weight_loader_reads_the_checkpoint(repo, harness, workdir):
    """Write a checkpoint under the published names and load it.

    `copy_weights` writes every parameter by hand, so on its own it never
    touches `_create_weight_mappings`, the `backbone.` prefix or the
    `[conv_dim, 1, conv_kernel]` conv layout the checkpoint ships.
    """
    from sgl_jax.srt.configs.model_config import ModelConfig

    path = os.path.join(workdir, "checkpoint")
    source = write_checkpoint(harness, path)

    model_config = ModelConfig(model_path=path, trust_remote_code=False, dtype="float32")
    fresh = Harness.__new__(Harness)
    fresh.settings = harness.settings
    fresh.cfg = harness.cfg
    fresh.tp = harness.tp
    fresh.mesh = harness.mesh
    fresh.model = fresh.build_model()
    fresh.model.load_weights(model_config)

    mapped = set(fresh.model._create_weight_mappings())
    missing = {key for key in source if ".experts." not in key} - mapped
    if missing:
        raise AssertionError(f"the mapping names no target for {sorted(missing)[:4]}")

    cfg = harness.cfg
    moe = next(i for i, b in enumerate(cfg.layers_block_type) if b == "moe")
    model = fresh.model.model
    landed = {
        "embeddings": (
            model.embed_tokens.embedding,
            source["backbone.embeddings.weight"],
        ),
        "mamba in_proj": (
            model.layers[0].mixer.in_proj.weight,
            source["backbone.layers.0.mixer.in_proj.weight"].T,
        ),
        "mamba conv1d": (
            model.layers[0].mixer.conv1d_weight,
            source["backbone.layers.0.mixer.conv1d.weight"].reshape(
                cfg.conv_dim, cfg.conv_kernel
            ),
        ),
        "router kernel": (
            model.layers[moe].mixer.moe_gate.kernel,
            source[f"backbone.layers.{moe}.mixer.gate.weight"].T,
        ),
        "expert 3 up_proj": (
            model.layers[moe].mixer.experts.wi_0,
            np.stack(
                [
                    source[f"backbone.layers.{moe}.mixer.experts.{e}.up_proj.weight"].T
                    for e in range(cfg.n_routed_experts)
                ]
            ),
        ),
    }
    for name, (param, want) in landed.items():
        got = np.asarray(param[...])
        if got.shape != want.shape:
            raise AssertionError(f"{name} loaded as {got.shape}, want {want.shape}")
        error = gap(got, want)
        if not error <= TOL:
            raise AssertionError(f"{name} loaded {error:.3e} away from the checkpoint")
    print(f"  loaded {len(source)} checkpoint tensors, {len(landed)} spot-checked")
    return fresh


def wider_mesh_agrees(repo, ids, quiet=False):
    """The same weights on a four-way tensor axis and on a one-way one.

    A one-device mesh normalizes every `kernel_axes` entry away, so nothing
    else here compiles a `shard_map` with more than one shard, exercises
    `_local_groups`, or splits an RMS group of the gated norm across shards.
    Two groups on four shards is the layout Super runs at `--tp-size 32`.

    The engine replicates KV heads up to the tensor axis before it builds the
    model, so this config carries the four heads that replication produces
    rather than the two the checkpoint ships.
    """
    narrow = Harness(repo, config=TINY_TP, tp=1)
    wide = Harness(repo, config=TINY_TP, tp=4)
    pool_narrow, pool_wide = narrow.new_pool(), wide.new_pool()
    _, _, thin = narrow.run(ids, [10, 6], pool_narrow)
    _, _, thick = wide.run(ids, [10, 6], pool_wide)
    return report("tp=4 against tp=1", thick, thin, quiet)


# ---------------------------------------------------------------------------
# Check 9: the published configs resolve.
# ---------------------------------------------------------------------------


PUBLISHED = {
    "Super": dict(
        hybrid_override_pattern=(
            "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
            "EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
        ),
        num_hidden_layers=88,
        hidden_size=4096,
        mamba_num_heads=128,
        moe_latent_size=1024,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        expect_layers=88,
        expect_attention=[7, 16, 25, 36, 47, 58, 69, 78],
        expect_recurrent=40,
    ),
    "Ultra": dict(
        layers_block_type=None,  # filled in below
        hidden_size=8192,
        mamba_num_heads=256,
        moe_latent_size=2048,
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
        expect_layers=108,
        expect_attention=[7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98],
        expect_recurrent=48,
    ),
}
def _ultra_stack() -> list[str]:
    """Ultra's 108 entries: attention at the published indices, the rest
    alternating mamba and moe straight through the inserts."""
    attention = set(PUBLISHED["Ultra"]["expect_attention"])
    out, toggle = [], 0
    for index in range(108):
        if index in attention:
            out.append("attention")
        else:
            out.append(("mamba", "moe")[toggle])
            toggle ^= 1
    return out


PUBLISHED["Ultra"]["layers_block_type"] = _ultra_stack()


def published_configs_resolve(workdir, quiet=False):
    """Both published sizes load through `AutoConfig` and resolve their stack.

    `AutoConfig.from_dict` formats the config it built into a log line, which
    reaches `to_diff_dict`, which builds a default instance of the class. A
    config class whose layer field has no default has to say so, or the
    published `config.json` never loads.
    """
    from sgl_jax.srt.hf_transformers_utils import get_config

    for name, spec in PUBLISHED.items():
        spec = dict(spec)
        expect_layers = spec.pop("expect_layers")
        expect_attention = spec.pop("expect_attention")
        expect_recurrent = spec.pop("expect_recurrent")
        path = os.path.join(workdir, f"config-{name}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(dict(spec, model_type="nemotron_h"), handle)

        config = get_config(path, trust_remote_code=False)
        if type(config).__name__ != "NemotronHConfig":
            raise AssertionError(f"{name} resolved to {type(config).__name__}")
        if config.num_hidden_layers != expect_layers:
            raise AssertionError(
                f"{name} resolved {config.num_hidden_layers} layers, want {expect_layers}"
            )
        if config.full_attention_layer_ids != expect_attention:
            raise AssertionError(
                f"{name} put attention at {config.full_attention_layer_ids}, "
                f"want {expect_attention}"
            )
        if len(config.linear_layer_ids) != expect_recurrent:
            raise AssertionError(
                f"{name} found {len(config.linear_layer_ids)} recurrent layers, "
                f"want {expect_recurrent}"
            )
        # The MTP head is in the config and out of the served stack, so the
        # KV budget counts the main stack's attention blocks and no more.
        if config.mtp_layers_block_type != ["attention", "moe"]:
            raise AssertionError(f"{name} lost its MTP block list")
        if len(config.full_attention_layer_ids) != (8 if name == "Super" else 12):
            raise AssertionError(f"{name} counted the MTP attention block in the KV plan")
        if tuple(config.time_step_limit) != (0.0, math.inf):
            raise AssertionError(
                f"{name} clamps dt to {config.time_step_limit}; neither published "
                "config.json overrides the reference default"
            )
        repr(config)
        if not quiet:
            print(
                f"    {name:<8} {config.num_hidden_layers} layers, "
                f"{len(config.linear_layer_ids)} recurrent, "
                f"{len(config.full_attention_layer_ids)} attention, "
                f"dt limit {tuple(config.time_step_limit)}"
            )

    # A checkpoint re-saved through stock `transformers` carries its names for
    # the two mixers, `linear_attention` and `full_attention`.
    from transformers.models.nemotron_h import NemotronHConfig as StockConfig

    path = os.path.join(workdir, "config-resaved")
    settings = dict(TINY, hybrid_override_pattern="ME*-", num_nextn_predict_layers=1)
    StockConfig(**settings, mtp_hybrid_override_pattern="*E").save_pretrained(path)
    with open(os.path.join(path, "config.json")) as handle:
        written = json.load(handle)
    if "linear_attention" not in written.get("layers_block_type", []):
        raise AssertionError(
            f"control: stock transformers wrote {written.get('layers_block_type')}, "
            "so this case tests nothing"
        )
    config = get_config(path, trust_remote_code=False)
    found = (config.layers_block_type, config.mtp_layers_block_type)
    want = (["mamba", "moe", "attention", "mlp"], ["attention", "moe"])
    if found != want:
        raise AssertionError(f"a re-saved config resolved to {found}, want {want}")
    if not quiet:
        print(f"    re-saved {written['layers_block_type']} resolves to {found[0]}")


# ---------------------------------------------------------------------------
# Checks 10 and 11: the engine's own pools, metadata and jit.
# ---------------------------------------------------------------------------


class Serving:
    """The serving state around one test model.

    Every piece is the class the runner uses for this model family:
    `RecurrentStatePool`, `HybridLinearKVPool` and `MemoryPools` hold the
    state, `LinearRecurrentAttnBackend.get_forward_metadata` builds the backend
    metadata from a `ModelWorkerBatch`, and `make_jitted_run_model` is the jit
    the runner calls, which clones a matched tree slot before the model reads
    it and donates the pools.
    """

    def __init__(self, harness, slots: int = 12):
        from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sgl_jax.srt.layers.attention.mamba.mamba2_backend import Mamba2AttnBackend
        from sgl_jax.srt.layers.attention.native_backend import NativeAttention
        from sgl_jax.srt.mem_cache.memory_pool import HybridLinearKVPool, MemoryPools
        from sgl_jax.srt.model_executor.model_forward import make_jitted_run_model

        self.harness = harness
        cfg, mesh = harness.cfg, harness.mesh
        self.recurrent = harness.new_pool(slots)
        self.kv = HybridLinearKVPool(
            size=256,
            page_size=TRACK_PAGE,
            dtype=jnp.float32,
            full_attention_layer_ids=cfg.full_attention_layer_ids,
            mesh=mesh,
            head_num=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )
        self.pools = MemoryPools(token_to_kv_pool=self.kv, recurrent_state_pool=self.recurrent)
        self.backend = HybridLinearAttnBackend(
            full_attn_backend=NativeAttention(
                cfg.num_attention_heads, cfg.num_key_value_heads, mesh
            ),
            linear_attn_backend=Mamba2AttnBackend(
                mamba_config=cfg.mamba_config(), mesh=mesh, activation=cfg.mamba_hidden_act
            ),
            full_attn_layers=cfg.full_attention_layer_ids,
        )
        self.run_model = make_jitted_run_model(self.backend)

    def seed(self, seed: int):
        """Random values in every slot, so a write that shouldn't land shows."""
        rng = np.random.default_rng(seed)
        fill = lambda buf: jax.device_put(
            jnp.asarray(rng.standard_normal(buf.shape), buf.dtype), buf.sharding
        )
        pool = self.recurrent
        pool.recurrent_buffers = [fill(buf) for buf in pool.recurrent_buffers]
        pool.conv_buffers = [[fill(buf) for buf in pair] for pair in pool.conv_buffers]

    def slot(self, index: int) -> list[np.ndarray]:
        """Every array one recurrent slot holds, SSM then conv, per layer."""
        pool = self.recurrent
        return [np.asarray(buf)[index] for buf in pool.recurrent_buffers] + [
            np.asarray(pair[0])[index] for pair in pool.conv_buffers
        ]

    def poke(self, which: str, index: tuple, value: float):
        """Write one element of the first layer's SSM or conv buffer."""
        pool = self.recurrent
        buf = pool.recurrent_buffers[0] if which == "ssm" else pool.conv_buffers[0][0]
        host = np.asarray(buf).copy()
        host[index] = value
        placed = jax.device_put(jnp.asarray(host), buf.sharding)
        if which == "ssm":
            pool.recurrent_buffers[0] = placed
        else:
            pool.conv_buffers[0][0] = placed

    def extend(self, token_lists, slots, has_initial, prefix_lens=None, track=None, cow=None):
        """One EXTEND through the serving jit. Returns every token's logits.

        `track` is the `(indices, mask)` pair `_build_recurrent_track_entries`
        returns, and `cow` the tree slot each request resumes from. The stack
        holds no attention block, so the KV locations stay zero and nothing
        reads them.
        """
        from flax import nnx

        from sgl_jax.srt.layers.logits_processor import LogitsMetadata
        from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
        from sgl_jax.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sgl_jax.srt.utils.jax_utils import device_array

        lens = [len(tokens) for tokens in token_lists]
        prefix_lens = [0] * len(lens) if prefix_lens is None else list(prefix_lens)
        ids = np.concatenate(token_lists).astype(np.int32)
        positions = np.concatenate(
            [np.arange(p, p + n) for p, n in zip(prefix_lens, lens)]
        ).astype(np.int32)
        seq_lens = np.asarray([p + n for p, n in zip(prefix_lens, lens)], np.int32)
        batch = ModelWorkerBatch(
            bid=0,
            forward_mode=ForwardMode.EXTEND,
            input_ids=ids,
            real_input_ids_len=len(ids),
            seq_lens=seq_lens,
            out_cache_loc=np.zeros(len(ids), np.int32),
            req_pool_indices=np.arange(len(lens), dtype=np.int32),
            sampling_info=None,
            positions=positions,
            cache_loc=np.zeros(int(seq_lens.sum()), np.int32),
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_seq_lens=np.asarray(lens, np.int32),
            extend_prefix_lens=np.asarray(prefix_lens, np.int32),
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            logits_indices=None,
            real_bs=len(lens),
            real_bs_per_dp=[len(lens)],
            dp_size=1,
            per_dp_bs_size=len(lens),
            recurrent_indices=np.asarray(slots, np.int32),
            recurrent_cow_src_indices=None if cow is None else np.asarray(cow, np.int32),
            recurrent_track_indices=None if track is None else track[0],
            recurrent_track_mask=None if track is None else track[1],
            has_initial_state=np.asarray(has_initial, bool),
        )

        # The same placement `ForwardBatch.init_new` gives every per-token and
        # per-request array: sharded over "data".
        on_data = jax.sharding.NamedSharding(self.harness.mesh, P("data"))
        place = lambda value: None if value is None else device_array((value,), on_data)[0]
        forward_batch = ForwardBatch(
            bid=batch.bid,
            forward_mode=batch.forward_mode,
            batch_size=len(lens),
            input_ids=place(batch.input_ids),
            req_pool_indices=place(batch.req_pool_indices),
            seq_lens=place(batch.seq_lens),
            out_cache_loc=place(batch.out_cache_loc),
            positions=place(batch.positions),
            attn_backend=self.backend,
            cache_loc=place(batch.cache_loc),
            extend_prefix_lens=place(batch.extend_prefix_lens),
            extend_seq_lens=place(batch.extend_seq_lens),
            recurrent_indices=place(batch.recurrent_indices),
            recurrent_cow_src_indices=place(batch.recurrent_cow_src_indices),
            recurrent_track_indices=place(batch.recurrent_track_indices),
            recurrent_track_mask=place(batch.recurrent_track_mask),
        )
        self.backend.forward_metadata = self.backend.get_forward_metadata(batch)

        # DECODE metadata with no logprob request keeps one logits row per
        # token, which is what a comparison position by position wants.
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.DECODE, capture_hidden_mode=CaptureHiddenMode.NULL
        )
        model_def, model_state = nnx.split(self.harness.model)
        leaves, state_def = jax.tree_util.tree_flatten(model_state)
        with jax.set_mesh(self.harness.mesh):
            output, updates, _, _ = self.run_model(
                model_def, state_def, leaves, forward_batch, self.pools, metadata
            )
        self.pools.replace_all(updates)
        return np.asarray(output.next_token_logits)


def track_slots_resume_a_prefix(harness, ids, quiet=False):
    """`--enable-recurrent-extra-buffer`, from the scheduler's pick to a prefix hit.

    Request A prefills one track interval beside request Z, which crosses no
    boundary. `_build_recurrent_track_entries`, the scheduler's bookkeeping,
    picks A's track slot and leaves Z's row dormant. After the forward the
    track slot has to hold what A's running slot holds, and no other slot may
    change: not Z's two track slots, not A's other one, not the dummy slot 0.
    `RecurrentComponent.prepare_for_caching_req` then names the slot the radix
    tree adopts, and request B, A's tokens plus 8 more, resumes from a copy of
    it through the serving jit. B's 8 rows have to match the same tokens
    prefilled from scratch.
    """
    from sgl_jax.srt.managers.schedule_batch import Req, _build_recurrent_track_entries
    from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from sgl_jax.srt.mem_cache.base_prefix_cache import InsertParams
    from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    serving = Serving(harness)
    serving.seed(11)
    slots = HybridReqToTokenPool(
        size=4,
        max_context_len=64,
        dtype=np.int32,
        recurrent_state_pool=serving.recurrent,
        enable_recurrent_extra_buffer=True,
        ping_pong_slots=2,
    )
    tree = UnifiedRadixCache(
        req_to_token_pool=slots,
        token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(
            size=256, page_size=TRACK_PAGE, kvcache=serving.kv
        ),
        page_size=TRACK_PAGE,
        tree_components=(ComponentType.FULL, ComponentType.RECURRENT),
        enable_recurrent_extra_buffer=True,
        recurrent_track_interval=TRACK_INTERVAL,
    )

    a_ids, z_ids = ids[:TRACK_INTERVAL], ids[TRACK_INTERVAL : TRACK_INTERVAL + 10]
    a = Req("a", "", a_ids.tolist(), SamplingParams(max_new_tokens=1))
    z = Req("z", "", z_ids.tolist(), SamplingParams(max_new_tokens=1))
    slots.alloc([a, z])
    for req in (a, z):
        req.extend_input_len = len(req.origin_input_ids)
    track = _build_recurrent_track_entries(
        [a, z], [len(a_ids), len(z_ids)], interval=TRACK_INTERVAL, pool=slots, is_extend=True
    )
    if track[1].tolist() != [1, 0]:
        raise AssertionError(f"the scheduler marked boundaries {track[1].tolist()}, want [1, 0]")
    track_slot = int(track[0][0])
    idle = [0, *z.recurrent_ping_pong_track_buffer]
    idle += [s for s in a.recurrent_ping_pong_track_buffer if s != track_slot]
    before = {s: serving.slot(s) for s in idle}

    serving.extend(
        [a_ids, z_ids],
        [a.recurrent_pool_idx, z.recurrent_pool_idx],
        [False, False],
        track=track,
    )
    written = worst_of(
        gap(x, y) for x, y in zip(serving.slot(track_slot), serving.slot(a.recurrent_pool_idx))
    )
    moved = [
        s
        for s in idle
        if not all(np.array_equal(x, y) for x, y in zip(before[s], serving.slot(s)))
    ]
    if moved:
        raise AssertionError(f"slots {moved} changed, and nothing wrote them this forward")

    donated = InsertParams()
    cache_len = tree.components[ComponentType.RECURRENT].prepare_for_caching_req(
        a, donated, len(a_ids), is_finished=True
    )
    tree_slot = int(donated.recurrent_value[0])
    if (cache_len, tree_slot) != (TRACK_INTERVAL, track_slot):
        raise AssertionError(
            f"the tree adopts slot {tree_slot} at {cache_len} tokens; the forward "
            f"wrote slot {track_slot} at {TRACK_INTERVAL}"
        )

    suffix = ids[TRACK_INTERVAL : TRACK_INTERVAL + 8]
    b = Req("b", "", np.concatenate([a_ids, suffix]).tolist(), SamplingParams(max_new_tokens=1))
    slots.alloc([b])
    hit = serving.extend(
        [suffix], [b.recurrent_pool_idx], [True], prefix_lens=[TRACK_INTERVAL], cow=[tree_slot]
    )
    fresh = Serving(harness).extend([np.concatenate([a_ids, suffix])], [1], [False])
    resumed = gap(hit, fresh[TRACK_INTERVAL:])

    if not quiet:
        print(
            f"    track slot {track_slot} against A's running slot   max abs {written:.3e}\n"
            f"    slots {idle} untouched, the tree adopts slot {tree_slot}\n"
            f"    prefix hit against a fresh prefill      max abs {resumed:.3e}"
        )
    return worst_of([written, resumed])


def nonfinite_state_stays_in_its_request(harness, ids, quiet=False):
    """One request's inf or NaN reaches no other request of a packed prefill.

    Four requests of 5, 10, 6 and 12 tokens, chunk size 8, each resuming from
    a cached state in its own slot. The second one's cached SSM state holds an
    inf and its conv window a NaN. Every other request's logits and written
    state have to match the run where nothing is poisoned. The request after
    it would take the NaN through a carry multiplied by zero; the one before
    it through the padding slots of its partial chunk, which the gather
    clamps onto the poisoned request's first tokens.
    """
    lens = [5, 10, 6, 12]
    starts = np.concatenate([[0], np.cumsum(lens)])
    tokens = [ids[s : s + n] for s, n in zip(starts, lens)]
    slots = [1, 2, 3, 4]

    def run(poison):
        serving = Serving(harness)
        serving.seed(3)
        if poison:
            serving.poke("ssm", (2, 1, 2, 3), np.inf)
            serving.poke("conv", (2, 5, 1), np.nan)
        logits = serving.extend(tokens, slots, [True] * len(lens))
        return logits, {s: serving.slot(s) for s in slots}

    clean_logits, clean_state = run(False)
    dirty_logits, dirty_state = run(True)
    if np.all(np.isfinite(dirty_logits[starts[1] : starts[2]])):
        raise AssertionError("control: the poisoned request came out finite")

    errors = []
    for request in (0, 2, 3):
        rows = slice(starts[request], starts[request + 1])
        slot = slots[request]
        logits = gap(dirty_logits[rows], clean_logits[rows])
        state = worst_of(gap(x, y) for x, y in zip(dirty_state[slot], clean_state[slot]))
        if not quiet:
            print(
                f"    request {request} of 4: logits max abs {logits:.3e}, "
                f"written state max abs {state:.3e}"
            )
        errors += [logits, state]
    return worst_of(errors)


# ---------------------------------------------------------------------------
# Check 12: a quantization config reaches the experts.
# ---------------------------------------------------------------------------


def int8_experts_against_numpy(harness, width):
    """Run the int8 experts of a latent MoE block `width` wide against numpy.

    Builds `NemotronHLatentMoE` the way the model does, from the test config
    with the latent and the expert width set to `width` and `int8.yaml` on it.
    Random experts go through `quantize_weights`, the call
    `apply_quantization` makes, and run on 16 random tokens routed to distinct
    experts. Returns the max error relative to the largest reference output,
    against numpy on the dequantized weights and on the float weights, and
    infinity when the experts return a non-finite value.
    """
    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.models.nemotron_h import NemotronHLatentMoE

    config = type(harness.cfg)(
        **dict(harness.settings, moe_latent_size=width, moe_intermediate_size=width)
    )
    config.quantization_config = QuantizationConfig.from_path("int8.yaml")
    with jax.set_mesh(harness.mesh):
        block = NemotronHLatentMoE(config, mesh=harness.mesh, layer_id=1, dtype=jnp.float32)
    experts = block.experts
    rng = np.random.default_rng(7)
    count, top_k = config.n_routed_experts, config.num_experts_per_tok
    floats = {
        name: (rng.standard_normal((count, width, width)) * 0.2).astype(np.float32)
        for name in ("wi_0", "wo")
    }
    for name, value in floats.items():
        put(getattr(experts, name), value)
    experts.quantize_weights()
    dequantized = {
        name: np.asarray(getattr(experts, name)[...], np.float64)
        * np.asarray(getattr(experts, f"{name}_scale")[...], np.float64)[:, 0]
        for name in floats
    }

    tokens = rng.standard_normal((16, width)).astype(np.float32)
    ids = np.argsort(rng.random((16, count)), axis=1)[:, :top_k].astype(np.int32)
    weights = rng.random((16, top_k)).astype(np.float32)
    with jax.set_mesh(harness.mesh):
        out = experts(jnp.asarray(tokens), jnp.asarray(weights), jnp.asarray(ids))
    out = np.asarray(out, np.float64)

    def routed(matrices):
        # One input matrix and a squared ReLU, summed over the top-k by weight.
        up = np.einsum("th,tkhi->tki", tokens, matrices["wi_0"][ids])
        down = np.einsum("tki,tkih->tkh", np.maximum(up, 0.0) ** 2, matrices["wo"][ids])
        return np.einsum("tk,tkh->th", weights, down)

    errors = {}
    for label, matrices in (("dequantized", dequantized), ("float", floats)):
        reference = routed(matrices)
        errors[label] = gap(out, reference) / float(np.abs(reference).max())
    return errors


def quantization_reaches_the_experts(harness, workdir, ids, quiet=False):
    """`--quantization-config-path`, the way the runner applies it.

    `ModelConfig` resolves the config and puts it on `hf_config`,
    `JAXModelLoader` builds the model and loads a checkpoint written under the
    published names, and `apply_quantization` quantizes online, which is what
    `ModelRunner.load_model` does next.

    Under the built-in `int8.yaml` the routed experts have to come out int8
    with one scale per output channel and no `wi_1`, and hold the checkpoint's
    numbers to within half a quantization step. The linear layers have to come
    out `QuantizedLinear`, and the conv kernel has to stay the float parameter
    the backend reads.

    On CPU the `gmm` kernel's interpret path returns NaN for int8 weights
    with a scale when an expert matrix is under 128 on either side, counted
    per shard. The test config's experts read a 32-wide latent and are 24
    wide, so the loaded model stops at the weights. A latent MoE block built
    the same way at 128 runs its int8 experts, and they have to land within
    1e-5 of numpy on the dequantized weights and more than 1e-3 off the float
    ones.

    A config that quantizes the linear layers alone runs the whole model, so
    a packed prefill under it has to land near the float logits and off
    them. That forward crosses `fc2_latent_proj`, which takes the experts'
    output, and the conv kernel.

    Returns 0.0 when all of that holds and infinity when any of it doesn't,
    so a mutant reads the same way it does on the other checks.
    """
    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.layers.linear import QuantizedLinear
    from sgl_jax.srt.model_loader.loader import get_model_loader
    from sgl_jax.srt.models.nemotron_h import NemotronHForCausalLM
    from sgl_jax.srt.utils.quantization.quantization_utils import apply_quantization

    path = os.path.join(workdir, "checkpoint-recurrent")
    if not os.path.exists(path):
        write_checkpoint(harness, path, scale=0.1)
    linear_only = os.path.join(workdir, "linear-int8.yaml")
    with open(linear_only, "w") as handle:
        handle.write(
            "quantization:\n"
            "  linear:\n"
            "    rules:\n"
            "      - module_path: '.*'\n"
            "        weight_dtype: 'int8'\n"
            "  moe:\n"
            "    weight_dtype: null\n"
        )

    def load(quantization):
        model_config = ModelConfig(
            model_path=path,
            trust_remote_code=False,
            dtype="float32",
            quantization_config_path=quantization,
        )
        # `ModelRunner.load_model` makes both calls outside any mesh context;
        # the loader sets its own.
        model = get_model_loader(LoadConfig(), harness.mesh).load_model(model_config=model_config)
        if model_config.quantization_config is not None:
            model = apply_quantization(model_config, model)
        loaded = Harness.__new__(Harness)
        loaded.settings, loaded.cfg = harness.settings, model_config.hf_config
        loaded.tp, loaded.mesh, loaded.model = harness.tp, harness.mesh, model
        return loaded

    plain = load(None)
    quantized = load("int8.yaml")
    stack = quantized.model.model.layers
    moe = next(i for i, b in enumerate(quantized.cfg.layers_block_type) if b == "moe")
    experts = stack[moe].mixer.experts
    found = {
        "expert weights": (str(experts.wi_0[...].dtype), str(experts.wo[...].dtype)),
        "expert scales": (experts.wi_0_scale is not None, experts.wo_scale is not None),
        "wi_1": experts.wi_1,
        "in_proj": type(stack[0].mixer.in_proj).__name__,
        "fc2_latent_proj": type(stack[moe].mixer.fc2_latent_proj).__name__,
        "conv kernel": str(stack[0].mixer.conv1d_weight[...].dtype),
    }
    want = {
        "expert weights": ("int8", "int8"),
        "expert scales": (True, True),
        "wi_1": None,
        "in_proj": QuantizedLinear.__name__,
        "fc2_latent_proj": QuantizedLinear.__name__,
        "conv kernel": "float32",
    }
    if found != want:
        if not quiet:
            print(f"    int8.yaml left the model at {found}, want {want}")
        return math.inf

    # Per output channel, `quantize_tensor` rounds w / scale to the nearest
    # integer with scale = max|w| / 127, so a dequantized weight sits within
    # half a scale of the checkpoint's.
    reference = plain.model.model.layers[moe].mixer.experts
    offsets = []
    for name in ("wi_0", "wo"):
        weights = np.asarray(getattr(experts, name)[...], np.float32)
        scale = np.asarray(getattr(experts, f"{name}_scale")[...])[:, 0]
        original = np.asarray(getattr(reference, name)[...])
        # |w * scale - original| / scale, in quantization steps.
        offsets.append(gap(weights, original / scale))
    steps = worst_of(offsets)
    if not quiet:
        print(f"    experts {found['expert weights']}, {steps:.3f} of a step off the checkpoint")
    if not steps <= 0.5 + 1e-3:
        return math.inf

    errors = int8_experts_against_numpy(harness, 128)
    if not quiet:
        print(
            f"    int8 experts 128 wide against numpy, max relative {errors['dequantized']:.3e} "
            f"on the dequantized weights; control: {errors['float']:.3e} on the float ones"
        )
    if not (errors["dequantized"] < 1e-5 and errors["float"] > 1e-3):
        return math.inf

    tokens = [ids[:10], ids[10:16]]
    before = Serving(plain).extend(tokens, [1, 2], [False, False])
    after = Serving(load(linear_only)).extend(tokens, [1, 2], [False, False])
    relative = gap(after, before) / float(np.abs(before).max())
    if not quiet:
        print(f"    int8 linear layers against float32 logits, max relative {relative:.3e}")
    if not 1e-6 < relative < 5e-2:
        return math.inf

    # A pre-quantized checkpoint names scale tensors the mapping doesn't read,
    # so the model refuses it at build time. Control: the same config without
    # the static flag builds.
    static = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[{"module_path": ".*", "weight_dtype": "float8_e4m3fn"}],
        moe_weight_dtype=jnp.float8_e4m3fn,
    )
    config = type(harness.cfg)(**harness.settings)
    config.quantization_config = static
    try:
        with jax.set_mesh(harness.mesh):
            NemotronHForCausalLM(config, mesh=harness.mesh, dtype=jnp.float32)
    except NotImplementedError:
        pass
    else:
        if not quiet:
            print("    a pre-quantized checkpoint built")
        return math.inf
    static.is_static_checkpoint = False
    with jax.set_mesh(harness.mesh):
        NemotronHForCausalLM(config, mesh=harness.mesh, dtype=jnp.float32)
    if not quiet:
        print("    a pre-quantized checkpoint is refused at build time; an online config builds")
    return 0.0


# ---------------------------------------------------------------------------
# Check 13: a stack cut to its first blocks.
# ---------------------------------------------------------------------------


def loaded_from(harness, model_config):
    """A `Harness` around a model built from `model_config` and loaded from its checkpoint."""
    loaded = Harness.__new__(Harness)
    loaded.settings, loaded.cfg = harness.settings, model_config.hf_config
    loaded.tp, loaded.mesh = harness.tp, harness.mesh
    loaded.model = loaded.build_model()
    loaded.model.load_weights(model_config)
    return loaded


def cut_stack_matches(harness, workdir, ids, quiet=False):
    """`--model-layer-nums` and a `num_hidden_layers` override build the first `CUT` blocks.

    `ModelConfig` writes `num_hidden_layers` on the config for
    `--model-layer-nums`, and `get_config` writes it for a `num_hidden_layers`
    entry in `--json-model-override-args`. Either write has to cut the block
    list and the cache plan to the first `CUT` blocks. The model a cut config
    builds holds `CUT` blocks and loads from the whole checkpoint. Its
    captured streams have to match the whole model's first `CUT`, and its final
    hidden state the whole model's stream entering block `CUT` after the same
    final norm. A count above the block list has to raise, and so does a
    config whose count disagrees with its block list at construction.

    Returns the worst error, and infinity when a shape is wrong, a cut raises,
    or a count that has to raise doesn't.
    """
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.configs.nemotron_h import NemotronHConfig

    path = os.path.join(workdir, "checkpoint-cut")
    if not os.path.exists(path):
        write_checkpoint(harness, path, scale=0.1)

    def model_config(**kwargs):
        return ModelConfig(model_path=path, trust_remote_code=False, dtype="float32", **kwargs)

    whole = loaded_from(harness, model_config())
    plan = whole.cfg
    want = (
        CUT,
        list(plan.layers_block_type[:CUT]),
        [i for i in plan.linear_layer_ids if i < CUT],
        [i for i in plan.full_attention_layer_ids if i < CUT],
    )
    problems = []
    cut_configs = {}
    for label, kwargs in (
        ("--model-layer-nums", dict(model_layer_nums=CUT)),
        ("the override", dict(model_override_args=json.dumps({"num_hidden_layers": CUT}))),
    ):
        try:
            config = model_config(**kwargs)
        except ValueError as exc:
            problems.append(f"{label} raised {exc}")
            continue
        cfg = config.hf_config
        found = (
            config.num_hidden_layers,
            list(cfg.layers_block_type),
            cfg.linear_layer_ids,
            cfg.full_attention_layer_ids,
        )
        if found != want:
            problems.append(f"{label} gave {found}, want {want}")
        cut_configs[label] = config

    too_many = json.dumps({"num_hidden_layers": plan.num_hidden_layers + 1})
    try:
        model_config(model_override_args=too_many)
        problems.append(f"a count of {plan.num_hidden_layers + 1} loaded")
    except ValueError:
        pass
    try:
        NemotronHConfig(**dict(harness.settings, num_hidden_layers=CUT))
        problems.append("a config whose count disagrees with its block list built")
    except ValueError:
        pass

    error = math.inf
    if "--model-layer-nums" in cut_configs:
        cut = loaded_from(harness, cut_configs["--model-layer-nums"])
        if len(cut.model.model.layers) != CUT:
            problems.append(f"the cut config built {len(cut.model.model.layers)} blocks")
        else:
            whole_streams, _, _ = whole.run(ids, [len(ids)], whole.new_pool())
            cut_streams, cut_final, _ = cut.run(ids, [len(ids)], cut.new_pool())
            with jax.set_mesh(cut.mesh):
                entering = jnp.asarray(whole_streams[CUT])
                expected = np.asarray(cut.model.model.norm(entering))
            error = worst_of(
                [gap(a, b) for a, b in zip(cut_streams, whole_streams[:CUT])]
                + [gap(cut_final, expected)]
            )
    if not quiet:
        for problem in problems:
            print(f"    {problem}")
        print(
            f"    {plan.layers_block_type} cut to {want[1]}: cache plan {want[2]} recurrent, "
            f"{want[3]} paged; the cut model against the whole one, max abs {error:.3e}"
        )
    return math.inf if problems else error


# ---------------------------------------------------------------------------
# Check 14: the expert-placement flags stop the build.
# ---------------------------------------------------------------------------


PLACEMENT_DEFAULTS = dict(
    ep_dispatch_algorithm=None, ep_num_redundant_experts=0, init_expert_location="trivial"
)
PLACEMENT_FLAGS = {
    "--ep-dispatch-algorithm": dict(ep_dispatch_algorithm="static"),
    "--ep-num-redundant-experts": dict(ep_num_redundant_experts=8),
    "--init-expert-location": dict(init_expert_location="expert-map.json"),
}


def placement_flags_stop_the_build(harness, quiet=False):
    """Each expert-placement flag stops `NemotronHForCausalLM` before it builds.

    `ModelRunner.load_model` publishes its server args with
    `set_global_server_args`, then builds the model. With any one of the three
    flags set the build has to raise a `ValueError` that names the flag, and
    with the defaults it has to build. The global server args go back to what
    they held, so no later check sees them.

    Returns 0.0 when all of that holds and infinity when any of it doesn't.
    """
    from sgl_jax.srt.eplb import expert_location
    from sgl_jax.srt.models.nemotron_h import NemotronHForCausalLM

    def build(**settings):
        server_args = types.SimpleNamespace(**dict(PLACEMENT_DEFAULTS, **settings))
        expert_location.set_global_server_args(server_args)
        with jax.set_mesh(harness.mesh):
            NemotronHForCausalLM(harness.cfg, mesh=harness.mesh, dtype=jnp.float32)

    previous = expert_location.get_global_server_args()
    refused = {}
    try:
        for flag, settings in PLACEMENT_FLAGS.items():
            try:
                build(**settings)
                refused[flag] = "built"
            except ValueError as exc:
                refused[flag] = "refused" if flag in str(exc) else f"raised without the flag: {exc}"
        build()
    finally:
        expert_location.set_global_server_args(previous)
    if not quiet:
        for flag, outcome in refused.items():
            print(f"    {flag:<28} {outcome}")
        print("    the defaults build")
    return 0.0 if all(outcome == "refused" for outcome in refused.values()) else math.inf


# ---------------------------------------------------------------------------
# Check 15: the runner's pool sizing and backend choice.
# ---------------------------------------------------------------------------


# Super at --tp-size 4, the measured slice: 40 Mamba-2 layers, each with a
# float32 SSM slot of [128 heads, 64, 128] and a float32 conv slot of [10,240
# channels, 3 tokens], split four ways by head and by channel.
SUPER_STATE_BYTES_TP4 = 40 * ((128 // 4) * 64 * 128 + 3 * (10240 // 4)) * 4


def runner_sizes_the_state_pool(harness, quiet=False):
    """The runner's view of the state pool matches what `RecurrentStatePool` allocates.

    `handle_recurrent_cache` splits HBM between the recurrent pool and the KV
    pages with `_per_req_state_bytes_from_config`, on the config
    `linear_recurrent_config` resolves, before the pool exists, and
    `attn_backend_wrapper` picks the linear backend from the same config. This
    runs that mixin code on a runner that holds only a model config. The
    config has to resolve to the model's own, the backend to a
    `Mamba2AttnBackend` on its layer plan, and the byte count to what one slot
    of a real pool holds, and to Super's published shapes at `--tp-size 4`.

    Returns 0.0 when all of that holds and infinity when any of it doesn't.
    """
    from sgl_jax.srt.configs.nemotron_h import NemotronHConfig
    from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import attn_backend_wrapper
    from sgl_jax.srt.model_executor import model_runner_kv_cache_mixin as mixin

    class Runner(mixin.ModelRunnerKVCacheMixin):
        def __init__(self, hf_config, mesh):
            self.model_config = types.SimpleNamespace(hf_config=hf_config)
            self.mesh = mesh

    cfg = harness.cfg
    runner = Runner(cfg, harness.mesh)
    problems = []
    if runner.linear_recurrent_config is not cfg:
        problems.append(f"linear_recurrent_config is {runner.linear_recurrent_config!r}")
    else:
        if runner._kv_pool_layer_count() != len(cfg.full_attention_layer_ids):
            problems.append(f"the KV pool counts {runner._kv_pool_layer_count()} layers")
        wrapped = attn_backend_wrapper(runner, DenseCausalAttention())
        linear = getattr(wrapped, "linear_attn_backend", None)
        built = type(linear).__name__
        if built != "Mamba2AttnBackend" or linear.mamba_config != cfg.mamba_config():
            problems.append(f"attn_backend_wrapper built {built}")
        elif wrapped.full_attn_layers != frozenset(cfg.full_attention_layer_ids):
            problems.append(f"the backend routes {sorted(wrapped.full_attn_layers)} to attention")

    pool = harness.new_pool(size=2)
    buffers = list(pool.recurrent_buffers) + [buf for pair in pool.conv_buffers for buf in pair]
    allocated = sum(math.prod(buf.shape[1:]) * buf.dtype.itemsize for buf in buffers)
    counted = mixin._per_req_state_bytes_from_config(cfg, harness.tp)
    if counted != allocated:
        problems.append(f"counted {counted} bytes a request, and a pool slot holds {allocated}")

    super_spec = {k: v for k, v in PUBLISHED["Super"].items() if not k.startswith("expect_")}
    super_counted = mixin._per_req_state_bytes_from_config(NemotronHConfig(**super_spec), 4)
    if super_counted != SUPER_STATE_BYTES_TP4:
        problems.append(f"Super at tp=4 counted {super_counted}, want {SUPER_STATE_BYTES_TP4}")
    if not quiet:
        for problem in problems:
            print(f"    {problem}")
        print(
            f"    state bytes a request: {counted} counted, {allocated} a pool slot holds; "
            f"Super at tp=4 {super_counted}"
        )
    return math.inf if problems else 0.0


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


CHECKS = {
    "packed": lambda run: compare(run.main, run.ids, [10, 6], "", quiet=True),
    "decode": lambda run: compare_decode(run.main, run.ids[:10], "", quiet=True),
    "isolation": lambda run: request_isolation(run.main, run.ids, quiet=True),
    "wide": lambda run: wider_mesh_agrees(run.repo, run.ids, quiet=True),
    "track": lambda run: track_slots_resume_a_prefix(run.recurrent, run.long_ids, quiet=True),
    "nonfinite": lambda run: nonfinite_state_stays_in_its_request(
        run.recurrent, run.long_ids, quiet=True
    ),
    "quantization": lambda run: quantization_reaches_the_experts(
        run.recurrent, run.workdir, run.long_ids, quiet=True
    ),
    "cut": lambda run: cut_stack_matches(run.main, run.workdir, run.ids[:10], quiet=True),
    "placement": lambda run: placement_flags_stop_the_build(run.main, quiet=True),
    "pool": lambda run: runner_sizes_the_state_pool(run.main, quiet=True),
}


def weight_mutants(harness):
    """Each entry perturbs one weight or breaks one connection."""
    model = harness.model.model

    def scale(param, factor):
        def apply():
            before = param[...]
            put(param, before * factor)
            return lambda: put(param, before)

        return apply

    def bump(param, index, delta):
        def apply():
            before = param[...]
            after = np.asarray(before).copy()
            after[index] += delta
            put(param, after)
            return lambda: put(param, before)

        return apply

    mamba = model.layers[0].mixer
    moe = model.layers[1].mixer
    attention = model.layers[2].mixer
    return {
        "mamba D skip zeroed": (scale(mamba.d, 0.0), "packed"),
        "mamba dt_bias nudged": (bump(mamba.dt_bias, 0, 0.5), "packed"),
        "mamba A_log nudged": (bump(mamba.a_log, 1, 0.5), "packed"),
        "one conv tap nudged": (bump(mamba.conv1d_weight, (0, 0), 0.5), "packed"),
        "gated norm weight scaled": (scale(mamba.norm.weight, 1.1), "packed"),
        "one router bias nudged": (bump(moe.moe_gate.bias, 3, 1.0), "packed"),
        "one expert nudged": (bump(moe.experts.wi_0, 2, 0.5), "packed"),
        "attention values scaled": (scale(attention.v_proj.weight, 1.1), "packed"),
        # Only the model's own final block reads this one. A walk that stops
        # at the last layer would miss it.
        "final norm scaled": (scale(model.norm.scale, 1.1), "packed"),
    }


def rewrite_source(owner, name: str, function, old: str, new: str):
    """Recompile `function` with one line of its source replaced, bind it as
    `owner.name`, and return the undo.

    A target that isn't in the source once raises, so a mutant can't pass by
    rewriting nothing. The source keeps its decorators, so a rewritten
    property comes back a property.
    """
    if isinstance(owner, type) and name not in owner.__dict__:
        undo = lambda: delattr(owner, name)  # noqa: E731
    else:
        original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        undo = lambda: setattr(owner, name, original)  # noqa: E731
    source = textwrap.dedent(inspect.getsource(function))
    if source.count(old) != 1:
        raise AssertionError(f"mutant target isn't in {owner.__name__}.{name} once: {old!r}")
    namespace = {}
    module = sys.modules[function.__module__]
    code = compile(source.replace(old, new), f"<mutant of {name}>", "exec")
    exec(code, module.__dict__, namespace)
    setattr(owner, name, namespace[name])
    return undo


def rewrite_method(owner, name: str, old: str, new: str):
    """Rebuild one method, or one module function, with one line replaced."""
    return rewrite_source(owner, name, getattr(owner, name), old, new)


def rewrite_property(cls, name: str, old: str, new: str):
    """Rebuild one property's getter with one line replaced."""
    return rewrite_source(cls, name, cls.__dict__[name].fget, old, new)


def source_mutants():
    """Each entry rewrites one line of the patched source."""
    from sgl_jax.srt.configs import nemotron_h as nemotron_config
    from sgl_jax.srt.layers import moe
    from sgl_jax.srt.layers.attention.fla import group_rmsnorm
    from sgl_jax.srt.layers.attention.mamba import mamba2_backend
    from sgl_jax.srt.model_executor import model_runner_kv_cache_mixin as mixin
    from sgl_jax.srt.models import nemotron_h

    backend = mamba2_backend.Mamba2AttnBackend

    def drop_boundary_reset():
        original = mamba2_backend.replay_scalar

        def without_reset(log_total, b_chunks, h_in, first=None):
            return original(log_total, b_chunks, h_in)

        mamba2_backend.replay_scalar = without_reset
        return lambda: setattr(mamba2_backend, "replay_scalar", original)

    def boundary_as_zero_decay():
        # The restart as a multiply: a request start takes a decay of
        # exp(-inf) = 0 and the scan multiplies the carry by it.
        original = mamba2_backend.replay_scalar

        def multiply(log_total, b_chunks, h_in, first=None):
            if first is not None:
                log_total = jnp.where(first[:, None], -jnp.inf, log_total)
            return original(log_total, b_chunks, h_in)

        mamba2_backend.replay_scalar = multiply
        return lambda: setattr(mamba2_backend, "replay_scalar", original)

    def padding_reads_the_next_request():
        return rewrite_method(
            backend,
            "_extend",
            "b_g = jnp.where(keep[..., None], gather(b_proj), 0.0)",
            "b_g = gather(b_proj)",
        )

    def drop_track_write():
        original = backend._set_track_state
        backend._set_track_state = lambda self, full_buffer, *args: full_buffer
        return lambda: setattr(backend, "_set_track_state", original)

    def weaken_decode_carry():
        original = backend._decode

        def weaker(self, x, b_proj, c_proj, dt, log_decay, h_in, d):
            return original(self, x, b_proj, c_proj, dt, log_decay, h_in * 0.9, d)

        backend._decode = weaker
        return lambda: setattr(backend, "_decode", original)

    def drop_expert_quantization():
        # EPMoE built with no quantization config, whatever the user asked for.
        original = nemotron_h._quantization_config
        nemotron_h._quantization_config = lambda config: None
        return lambda: setattr(nemotron_h, "_quantization_config", original)

    def drop_down_projection_scale():
        # The int8 down projection read as raw integers.
        return rewrite_method(
            moe.EPMoE,
            "__call__",
            "self.wo_scale.value if self.wo_scale is not None else None,",
            "None,",
        )

    def attention_turns_nan():
        # A NaN out of the attention mixer, which every later layer carries.
        return rewrite_method(
            nemotron_h.NemotronHAttention,
            "__call__",
            "out, _ = self.o_proj(attn_output)",
            "out, _ = self.o_proj(attn_output * jnp.nan)",
        )

    def decode_step_turns_nan():
        # A NaN out of the Mamba-2 decode step alone; prefill stays finite.
        return rewrite_method(
            backend,
            "_decode",
            'y = jnp.einsum("bhn,bhnp->bhp", ch, h, precision=PRECISION)',
            'y = jnp.einsum("bhn,bhnp->bhp", ch, h, precision=PRECISION) * jnp.nan',
        )

    def split_group_norm_turns_nan():
        # A NaN out of the layout an RMS group takes when it straddles shards,
        # which only the tp=4 side of the wide check runs.
        return rewrite_method(
            group_rmsnorm.GroupRMSNorm,
            "_split_groups",
            "scale = lax.rsqrt(group_sq / self.group_size + self.epsilon)",
            "scale = lax.rsqrt(group_sq / self.group_size + self.epsilon) * jnp.nan",
        )

    def ignore_layer_cut():
        # The stock transformers setter: it takes the write and keeps every block.
        cls = nemotron_config.NemotronHConfig
        original = cls.__dict__["num_hidden_layers"]
        cls.num_hidden_layers = property(original.fget, lambda self, value: None)
        return lambda: setattr(cls, "num_hidden_layers", original)

    def let_placement_through():
        cls = nemotron_h.NemotronHForCausalLM
        original = cls.__dict__["check_server_args"]
        cls.check_server_args = classmethod(lambda klass, server_args, model_config=None: None)
        return lambda: setattr(cls, "check_server_args", original)

    def square_state_bytes():
        # The count before the patch: a square SSM slot, head_dim by head_dim.
        return rewrite_method(
            mixin,
            "_compute_recurrent_per_req_bytes",
            "* head_dim * head_k_dim *",
            "* head_dim * head_dim *",
        )

    def runner_misses_the_config():
        return rewrite_property(
            mixin.ModelRunnerKVCacheMixin,
            "linear_recurrent_config",
            'if getattr(self.model_config.hf_config, "model_type", None) == "nemotron_h":',
            "if False:",
        )

    return {
        "request boundary reset dropped": (drop_boundary_reset, "isolation"),
        "request boundary as a zero decay": (boundary_as_zero_decay, "nonfinite"),
        "padding slots read the next request": (padding_reads_the_next_request, "nonfinite"),
        "track slot write dropped": (drop_track_write, "track"),
        "decode state carry weakened": (weaken_decode_carry, "decode"),
        "experts built with no quantization": (drop_expert_quantization, "quantization"),
        "int8 down projection loses its scale": (drop_down_projection_scale, "quantization"),
        "attention output turns NaN": (attention_turns_nan, "packed"),
        "Mamba-2 decode step turns NaN": (decode_step_turns_nan, "decode"),
        "split-group norm turns NaN": (split_group_norm_turns_nan, "wide"),
        "a written layer count is ignored": (ignore_layer_cut, "cut"),
        "expert-placement flags let through": (let_placement_through, "placement"),
        "square SSM slot in the byte count": (square_state_bytes, "pool"),
        "runner misses the nemotron_h config": (runner_misses_the_config, "pool"),
    }


# ---------------------------------------------------------------------------


def main():
    ids = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59], np.int32)

    with tempfile.TemporaryDirectory() as workdir:
        print("check 1: the served stack builds, both patches apply, and what they touch compiles")
        repo = prepare_repo(workdir)

        harness = Harness(repo)
        print(f"  config: {harness.cfg.layers_block_type}")

        print("check 2: prefill, one request of 10 tokens")
        single = compare(harness, ids[:10], [10], "one request")
        hook = capture_hook_matches(harness, ids[:10], [10])

        print("check 3: prefill, two packed requests of 10 and 6 tokens")
        packed = compare(harness, ids, [10, 6], "two requests")

        print("check 4: decode one token on top of a 9 token prefill")
        decode = compare_decode(harness, ids[:10], "decode step")

        print("check 5: the chunked scan against a token-by-token recurrence")
        scan = chunked_scan_matches()
        loose = chunked_scan_matches(scale_decay=0.9, quiet=True)
        print(f"    control: a decay 10% off lands at {loose:.3e}")
        if not loose > TOL:
            raise AssertionError("control: the chunk comparison ignores the decay")
        discretize_honors_the_limit()

        print("check 6: the list spelling, a dense MLP, a tied head, an MTP head")
        other = Harness(repo, config=TINY_MLP)
        if other.cfg.num_hidden_layers != 4:
            raise AssertionError("the MTP head landed in the served stack")
        if other.cfg.mtp_layers_block_type != ["attention", "moe"]:
            raise AssertionError("the config dropped the MTP block list")
        if len(other.model.model.layers) != 4:
            raise AssertionError("the model built a block per MTP entry")
        mlp = compare(other, ids[:10], [10], f"blocks {other.cfg.layers_block_type}")

        print("check 7: one request's state doesn't leak into the next")
        isolation = request_isolation(harness, ids)

        print("check 8: the published key names load, and tp=4 matches tp=1")
        weight_loader_reads_the_checkpoint(repo, harness, workdir)
        wide = wider_mesh_agrees(repo, ids)

        print("check 9: the published configs resolve through AutoConfig")
        published_configs_resolve(workdir)

        recurrent = Harness(repo, config=TINY_RECURRENT)
        long_ids = np.random.default_rng(5).integers(3, TINY["vocab_size"], size=40)
        print(f"check 10: the extra buffer's track slot, on {recurrent.cfg.layers_block_type}")
        track = track_slots_resume_a_prefix(recurrent, long_ids)

        print("check 11: a non-finite state stays in its own request")
        nonfinite = nonfinite_state_stays_in_its_request(recurrent, long_ids)

        print("check 12: int8.yaml reaches the experts")
        quantization = quantization_reaches_the_experts(recurrent, workdir, long_ids)

        print(f"check 13: --model-layer-nums and a num_hidden_layers override keep {CUT} blocks")
        cut = cut_stack_matches(harness, workdir, ids[:10])

        print("check 14: the expert-placement flags stop the build")
        placement = placement_flags_stop_the_build(harness)

        print("check 15: the runner resolves the config, the backend and the state bytes")
        pool = runner_sizes_the_state_pool(harness)

        clean = {
            "single": single,
            "packed": packed,
            "decode": decode,
            "scan": scan,
            "hook": hook,
            "mlp": mlp,
            "isolation": isolation,
            "wide": wide,
            "track": track,
            "nonfinite": nonfinite,
            "quantization": quantization,
            "cut": cut,
            "placement": placement,
            "pool": pool,
        }
        worst = worst_of(clean.values())
        if not worst <= TOL:
            failed = {name: error for name, error in clean.items() if not error <= TOL}
            raise AssertionError(f"max abs error {worst:.3e} is above {TOL:.0e}: {failed}")
        print(f"  clean run worst error {worst:.3e}, tolerance {TOL:.0e}")

        print("controls: every mutant must move the check it targets")
        run = types.SimpleNamespace(
            main=harness,
            recurrent=recurrent,
            repo=repo,
            ids=ids,
            long_ids=long_ids,
            workdir=workdir,
        )
        mutants = dict(weight_mutants(harness))
        mutants.update(source_mutants())
        uncovered = set(CHECKS) - {target for _, target in mutants.values()}
        if uncovered:
            raise AssertionError(f"no mutant targets {sorted(uncovered)}")
        for name, (apply, target) in mutants.items():
            undo = apply()
            try:
                error = CHECKS[target](run)
            finally:
                undo()
            # NaN compares false either way, so it reads as missed.
            caught = error > TOL
            status = "caught" if caught else "MISSED"
            print(f"    {name:<36} {target:<12} max abs {error:.3e}   {status}")
            if not caught:
                raise AssertionError(f"control passed: {name} changed nothing {target} reads")
        print(f"  {len(mutants)} mutants, each caught by the check it names")

    print("all checks passed")


if __name__ == "__main__":
    main()
