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

"""The capture reference for DeepSeek V4.1-Flash, from DeepSeek's own runtime.

`check_capture.py --deepseek-inference` comes here. transformers has no V4.1 class, and the
checkpoint ships no modeling file for `trust_remote_code`. It ships a readable runtime instead,
`inference/model.py` and `inference/engram.py`, and this module runs that on the host CPU:

1. `deepseek_cpu_kernels.install()` stands in for `inference/kernel.py`, which is tilelang and
   CUDA-only. Each of the four Python files the reference imports has to match the SHA-256 in
   `upstream/models/deepseek-v41-published.sha256`, the revision the port was checked against.
2. `ModelArgs` come from `inference/config.json`, with the linears as float weights (`dtype:
   bf16`, no FP4 experts), one sequence, the vision tower and the DSpark blocks off. The Engram
   hash builds its token map from the checkpoint's tokenizer, as the runtime does.
3. The model builds with its parameters on the meta device and its buffers real, and every
   weight loads from the safetensors by name. FP8 E4M3 dense weights dequantize with one E8M0
   scale per 32x32 tile, packed E2M1 experts with one E8M0 scale per 32 inputs, both to BF16,
   the values `convert.py` feeds the runtime. The float32 pass upcasts each weight. The BF16 pass
   builds under a BF16 default dtype, so the parameters the runtime declares in float32 (the
   Hyper-Connections tensors, the sinks, the router bias, the ratio-2 compressor, the head) stay
   float32.
4. Two changes to the runtime, both named where they happen. `Indexer.forward` publishes an
   owner's index keys only on a step that closes a group, so on a decode step a ratio-2 owner
   reads another layer's keys; the reference makes each owner publish its own cache first. A
   prefill closes every group it can, so a prefill-only capture runs the same either way. The
   Engram table reads its rows from the memory-mapped safetensors instead of holding 98 GB per
   table: the same gather, the same E8M0 factor, the same BF16 cast as
   `ParallelEngramEmbedding.forward` at one rank.
5. Hidden states come from forward hooks at the slots the engine captures: the input of each
   layer's `attn_norm`, which is the four copies collapsed with the incoming `pre` mix, and the
   final norm's output last.

`streamed=True` keeps every block on meta, and a hook on each block loads its weights just before
it runs and frees them after, so RAM holds one block: about 25 GiB of BF16 experts, 51 GiB in
float32. `num_layers` builds only the first blocks, for a dry run.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PINS = os.path.join(HERE, os.pardir, "upstream", "models", "deepseek-v41-published.sha256")
# The code the reference imports. `inference/config.json` is read, not pinned: a tiny snapshot
# carries its own.
REFERENCE_FILES = ("inference/model.py", "inference/engram.py", "inference/vision.py",
                   "inference/image_processor.py")
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def check_pins(model_path: str, pins_path: str = PINS) -> None:
    """Each reference file in the checkpoint has to match its pinned SHA-256."""
    pins = {}
    with open(pins_path) as fp:
        for line in fp:
            digest, name = line.split()
            pins[name] = digest
    for name in REFERENCE_FILES:
        with open(os.path.join(model_path, name), "rb") as fp:
            digest = hashlib.sha256(fp.read()).hexdigest()
        if digest != pins[name]:
            raise SystemExit(f"{model_path}/{name} has SHA-256 {digest}; the port was checked "
                             f"against {pins[name]}")


def import_runtime(model_path: str):
    """`inference/model.py` and `engram.py` from the checkpoint, on the CPU kernel stand-ins,
    with the index-key handoff made per owner."""
    import deepseek_cpu_kernels

    deepseek_cpu_kernels.install()
    inference = os.path.join(model_path, "inference")
    if inference not in sys.path:
        sys.path.insert(0, inference)
    for name in ("model", "engram", "vision", "image_processor"):
        sys.modules.pop(name, None)
    model = importlib.import_module("model")
    fix_index_handoff(model)
    return model, importlib.import_module("engram")


def fix_index_handoff(model) -> None:
    """Make an index-key owner publish its own cache before it scores.

    `Indexer.forward` sets `shared_attn.index_k = self.k_cache` only when `latent is not None`,
    so on a step that closes no group the owner scores against the last publisher's cache.
    """
    if getattr(model.Indexer.forward, "_publishes_first", False):
        return
    original = model.Indexer.forward

    def forward(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            model.shared_attn.index_k = self.k_cache
        return original(self, x, qr, latent, start_pos, offset)

    forward._publishes_first = True
    model.Indexer.forward = forward


class Checkpoint:
    """The safetensors of one snapshot, read one tensor at a time by memory map."""

    def __init__(self, model_path: str):
        self.path = model_path
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as fp:
                self.files = dict(json.load(fp)["weight_map"])
        else:
            from safetensors import safe_open

            self.files = {}
            for name in sorted(os.listdir(model_path)):
                if name.endswith(".safetensors"):
                    with safe_open(os.path.join(model_path, name), "pt") as fh:
                        self.files.update({k: name for k in fh.keys()})
        self._handles = {}

    def __contains__(self, key):
        return key in self.files

    def keys(self):
        return self.files.keys()

    def get(self, key):
        from safetensors import safe_open

        name = self.files[key]
        if name not in self._handles:
            self._handles[name] = safe_open(os.path.join(self.path, name), "pt")
        return self._handles[name].get_tensor(key)

    def rows(self, key):
        """A read-only numpy memory map of a 2-D tensor, for gathering rows without loading it."""
        import numpy as np

        path = os.path.join(self.path, self.files[key])
        with open(path, "rb") as fp:
            n = int.from_bytes(fp.read(8), "little")
            info = json.loads(fp.read(n))[key]
        start, _ = info["data_offsets"]
        dtype = {"F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "BF16": np.uint16, "F32": np.float32}[info["dtype"]]
        return np.memmap(path, dtype=dtype, mode="r", offset=8 + n + start, shape=tuple(info["shape"])), info["dtype"]


def dequant_fp8(weight, scale, block: int = 32):
    """E4M3 `[out, in]` with one E8M0 scale per `block x block` tile, the last tile partial, to
    BF16 through float32."""
    import torch

    out, inn = weight.shape
    factor = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:out, :inn]
    return (weight.float() * factor).to(torch.bfloat16)


def dequant_fp4(packed, scale, block: int = 32):
    """Packed E2M1 `[out, in/2]`, even element in the low nibble, one E8M0 scale per `block`
    inputs, to BF16 through float32. Every value is exact in BF16."""
    import torch

    p = packed.view(torch.uint8)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).reshape(p.shape[0], -1)
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32)[codes.long()]
    factor = scale.float().repeat_interleave(block, 1)
    return (values * factor).to(torch.bfloat16)


class Reference:
    """DeepSeek's runtime with its weights read from one snapshot."""

    def __init__(self, model_path: str, dtype_name: str, streamed: bool = False,
                 num_layers: int | None = None, max_seq_len: int = 4096, tokenizer=None,
                 mmap_engram: bool | None = None, check_files: bool = True):
        import torch
        from accelerate import init_empty_weights

        if check_files:
            check_pins(model_path)
        self.model_path = model_path
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
        self.streamed = streamed
        self.mmap_engram = streamed if mmap_engram is None else mmap_engram
        runtime, engram = import_runtime(model_path)
        self.runtime = runtime
        with open(os.path.join(model_path, "inference", "config.json")) as fp:
            cfg = json.load(fp)
        fields = runtime.ModelArgs.__dataclass_fields__
        kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in fields}
        kwargs.update(dtype="bf16", expert_dtype=None, max_batch_size=1, max_seq_len=max_seq_len,
                      vision_n_layers=0, dspark_block_size=0, temperature=0.0)
        self.full_layers = kwargs["n_layers"]
        if num_layers is not None:
            if not 0 < num_layers <= kwargs["n_layers"]:
                raise ValueError(f"--reference-layers {num_layers}: the model has {kwargs['n_layers']}")
            kwargs["n_layers"] = num_layers
            kwargs["engram_layer_ids"] = tuple(i for i in kwargs.get("engram_layer_ids", ()) if i < num_layers)
            kept = [i for i, lid in enumerate(cfg.get("engram_layer_ids", [])) if lid < num_layers]
            kwargs["engram_num_embeddings"] = tuple(cfg["engram_num_embeddings"][i] for i in kept)
            for key in ("kv_source_layers", "index_source_layers"):
                kwargs[key] = tuple(i for i in kwargs[key] if i < num_layers)
            if kwargs.get("candidate_source_layer", -1) >= num_layers:
                kwargs["candidate_source_layer"] = -1
        self.num_layers = kwargs["n_layers"]
        args = runtime.ModelArgs(**kwargs)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.checkpoint = Checkpoint(model_path)
        torch.set_default_dtype(torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float32)
        try:
            with init_empty_weights():  # parameters on meta, buffers real
                model = runtime.Transformer(args, tokenizer)
        finally:
            torch.set_default_dtype(torch.float32)
        model.eval()
        self.model = model
        self.args = args
        self.shapes = {n: (tuple(t.shape), t.dtype) for n, t in model.named_parameters()}
        if self.dtype == torch.float32:
            # the float32 pass upcasts every floating parameter, as `.float()` would
            self.shapes = {n: (s, torch.float32 if d.is_floating_point else d) for n, (s, d) in self.shapes.items()}
        if self.mmap_engram:
            for layer in model.layers:
                if layer.engram is not None:
                    self._mmap_engram(layer)
        self.check_coverage()
        self._load(streamed)
        print(f"deepseek reference: {self.num_layers} of {self.full_layers} layers, {dtype_name}, "
              f"{'streamed' if streamed else 'resident'}, Engram rows "
              f"{'memory-mapped' if self.mmap_engram else 'resident'}", flush=True)

    # --- weights ------------------------------------------------------------------------

    def source(self, name: str):
        if name not in self.checkpoint:
            raise KeyError(f"{name}: the checkpoint holds no {name}")
        return name

    def check_coverage(self):
        """Every parameter has a tensor. The tensors left over are the scales the loader reads
        beside their weights, the image-token bias, the vision tower and DSpark, the layers past
        a cut, and the Engram tables when they're memory-mapped."""
        missing = [n for n in self.shapes if n not in self.checkpoint]
        if missing:
            raise ValueError(f"{len(missing)} parameters have no checkpoint tensor: {missing[:5]}")

    def value(self, name: str):
        import torch

        shape, dtype = self.shapes[name]
        t = self.checkpoint.get(name)
        scale_key = name[: -len("weight")] + "scale" if name.endswith(".weight") else None
        if t.dtype == torch.float8_e4m3fn and ".engram.embed." not in name:
            t = dequant_fp8(t, self.checkpoint.get(scale_key))
        elif t.dtype == torch.int8 and scale_key in self.checkpoint:
            t = dequant_fp4(t, self.checkpoint.get(scale_key))
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} is {tuple(t.shape)}, and the parameter wants {shape}")
        if t.is_floating_point() and dtype.is_floating_point:
            if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                return t  # the resident Engram table stays FP8, as ParallelEngramEmbedding keeps it
            t = t.to(dtype)
        return t

    def _put(self, module, name, value):
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(module, name, "cpu", value=value,
                                    dtype=value.dtype if value.is_floating_point() else None)

    def _load(self, streamed):
        started = time.time()
        for name in self.shapes:
            if self.mmap_engram and ".engram.embed." in name:
                continue
            if streamed and name.startswith("layers."):
                continue
            self._put(self.model, name, self.value(name))
        print(f"loaded {'the non-layer' if streamed else 'every'} weight in "
              f"{time.time() - started:.1f}s", flush=True)
        if not streamed:
            return

        def load_layer(i, block):
            # The runtime calls a block's Engram before the block itself, so the Engram's own
            # pre-hook loads the block first, and the block's hook finds it loaded.
            def hook(module, args):
                if getattr(block, "_stream_loaded", False):
                    return
                t0 = time.time()
                for local, _ in list(block.named_parameters()):
                    full = f"layers.{i}.{local}"
                    if self.mmap_engram and ".engram.embed." in full:
                        continue
                    self._put(block, local, self.value(full))
                block._stream_load_seconds = time.time() - t0
                block._stream_loaded = True
            return hook

        def drop(layer, args, output):
            from accelerate.utils import set_module_tensor_to_device

            for local, t in list(layer.named_parameters()):
                if t.device.type == "cpu" and ".engram.embed." not in local:
                    set_module_tensor_to_device(layer, local, "meta")
            layer._stream_loaded = False
            gc.collect()
            from check_capture import release_freed_memory

            release_freed_memory()

        for i, layer in enumerate(self.model.layers):
            if layer.engram is not None:
                layer.engram.register_forward_pre_hook(load_layer(i, layer))
            layer.register_forward_pre_hook(load_layer(i, layer))
            layer.register_forward_hook(drop)

    def _mmap_engram(self, layer):
        """Gather a layer's Engram rows from the memory-mapped safetensors, the same arithmetic as
        `ParallelEngramEmbedding.forward` at one rank."""
        import numpy as np
        import torch

        embed = layer.engram.embed
        key = f"layers.{layer.layer_id}.engram.embed."
        table, tag = self.checkpoint.rows(key + "weight")
        scale, _ = self.checkpoint.rows(key + "scale")
        if tag != "F8_E4M3":
            raise ValueError(f"{key}weight ships {tag}")
        block = embed.block_size

        def forward(indices):
            flat = indices.reshape(-1).numpy()
            values = torch.from_numpy(np.ascontiguousarray(table[flat])).view(torch.float8_e4m3fn)
            codes = torch.from_numpy(np.ascontiguousarray(scale[flat])).view(torch.float8_e8m0fnu)
            v = values.float().unflatten(-1, (-1, block)) * codes.float().unsqueeze(-1)
            return v.flatten(-2).to(torch.bfloat16).reshape(*indices.shape, -1)

        embed.forward = forward

    # --- forward ------------------------------------------------------------------------

    def hidden_states(self, ids, log_layers: bool = False) -> list:
        """`[seq, d]` float64 arrays: the collapsed attention input of each layer, then the final
        norm's output. One prefill of the whole prompt."""
        import torch

        states, handles, timing = [], [], {}
        layers = list(self.model.layers)

        def collapsed(module, args):
            states.append(args[0][0].detach().to(torch.float64).numpy())

        def entering(layer, args):
            timing[layer] = time.time()

        def leaving(layer, args, output):
            if log_layers:
                loaded = getattr(layer, "_stream_load_seconds", 0.0)
                print(f"  layer {layer.layer_id}: {time.time() - timing[layer]:.1f}s "
                      f"({loaded:.1f}s loading)", flush=True)

        for layer in layers:
            handles.append(layer.attn_norm.register_forward_pre_hook(collapsed))
            handles.append(layer.register_forward_pre_hook(entering, prepend=True))
            handles.append(layer.register_forward_hook(leaving))
        handles.append(self.model.norm.register_forward_hook(
            lambda m, a, out: states.append(out[0].detach().to(torch.float64).numpy())))
        # the Engram embed casts to BF16 for the FP8 GEMM behind it; with float weights the wkv
        # after it runs in the pass's dtype. A dequantized row is exact in BF16, so this loses
        # nothing.
        for layer in layers:
            if layer.engram is not None:
                wdtype = self.shapes[f"layers.{layer.layer_id}.engram.wkv.weight"][1]
                handles.append(layer.engram.wkv.register_forward_pre_hook(
                    lambda m, a, d=wdtype: (a[0].to(d),)))
        try:
            with torch.inference_mode():
                self.model(torch.tensor([list(ids)]), 0)
        finally:
            for h in handles:
                h.remove()
        if len(states) != len(layers) + 1:
            raise RuntimeError(f"{len(layers)} layers gave {len(states)} hidden states")
        return states


def reference_forward(model_path: str, batch: list, dtype_name: str, streamed: bool = False,
                      num_layers: int | None = None) -> list:
    """Per prompt, the hidden states `Reference.hidden_states` returns. Each prompt runs alone."""
    longest = max(len(ids) for ids in batch)
    ref = Reference(model_path, dtype_name, streamed, num_layers,
                    max_seq_len=-(-longest // 128) * 128)
    out = []
    for n, ids in enumerate(batch):
        started = time.time()
        out.append(ref.hidden_states(ids, log_layers=streamed))
        print(f"prompt {n}: {len(ids)} tokens, {dtype_name} forward in {time.time() - started:.1f}s",
              flush=True)
    del ref
    gc.collect()
    return out
