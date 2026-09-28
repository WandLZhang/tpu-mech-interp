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

"""The capture reference for a checkpoint that ships its own modeling code, such as Kimi K3.

`check_capture.py --trust-remote-code` comes here. transformers has no Kimi K3 class, so the
reference runs the checkpoint's `modeling_kimi_linear.py`, the language model that
`modeling_kimi_k3.py` wraps. The steps:

1. `fla_torch.install()` gives the modeling file the fla symbols it imports, in pure torch, when
   fla isn't importable. transformers 5.17 moved `OutputRecorder` out of
   `transformers.utils.generic`, where the file imports it from, so it goes back there. Nothing
   in 5.17 reads it: `check_model_inputs` there only merges config defaults.
2. The config loads with `trust_remote_code`, and the text model's class comes from the text
   config's `auto_map`. A text prompt never reaches the vision tower, so it isn't built, and its
   168 tensors stay on disk.
3. The model builds on the meta device, and every weight loads straight from the checkpoint's
   safetensors. Nothing converts through `from_pretrained`, so nothing is offloaded or written.
   Three rules map the checkpoint onto the module:
   - A `mxfp4-pack-quantized` tensor, `weight_packed` and `weight_scale`, dequantizes to BF16 as
     the engine's `utils/quantization/mxfp4.py` does: E2M1 codes low nibble first, times
     `2 ** (scale - 127)` in float32, then BF16. Every E2M1 value times a power of two is exact in
     BF16.
   - `A_log` ships `[head_dim]` for `num_heads` heads, zero after them; the module declares
     `[num_heads]`. The load keeps the first `num_heads` and refuses a nonzero tail.
   - Everything else loads by name under the text model's prefix, `language_model.` for K3.
   Every tensor under the prefix has to feed a parameter and every parameter needs a tensor.
4. The float32 pass upcasts each weight; the BF16 pass casts each floating weight to BF16, which
   is what `from_pretrained(dtype=bfloat16)` does for a model with no `_keep_in_fp32_modules`.
5. Attention runs the file's own `eager_attention_forward`. The file sets `flash_attention_2`,
   which needs flash_attn and a GPU.
6. Hidden states come from forward hooks: the input to layer 0, each later layer's input (the
   earlier layer's output), and the final norm's output last. That's the list
   `output_hidden_states=True` returns under transformers 4.56, where the file was written; 5.17
   returns none for this file.

`streamed=True` keeps every decoder layer on meta, and a hook on each layer loads its weights
just before it runs and frees them after, so RAM holds one layer: 55 GiB of BF16 experts for a
Kimi K3 MoE layer, 110 GiB in float32. `num_layers` builds only the first layers, for a dry run.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MXFP4_FORMAT = "mxfp4-pack-quantized"
MXFP4_EXPONENT_BIAS = 127
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def patch_transformers_compat() -> None:
    """Put `OutputRecorder` back where transformers 4.56 kept it, for the file's import line."""
    from transformers.utils import generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder


def patch_module_compat(module) -> None:
    """Fit the remote module's calls into transformers 5, where it was written against 4.56.

    `create_causal_mask` took `input_embeds` and `cache_position` in 4.56. 5.17 takes
    `inputs_embeds` and reads positions from the cache. The wrapper renames the first. It drops
    `cache_position` only when there's no cache, where the positions start at 0 either way, and
    refuses it otherwise.
    """
    import inspect

    original = getattr(module, "create_causal_mask", None)
    if original is None or getattr(original, "_renames_input_embeds", False):
        return
    accepted = inspect.signature(original).parameters
    if "input_embeds" in accepted:
        return

    def create_causal_mask(*args, **kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" in kwargs and "cache_position" not in accepted:
            if kwargs.get("past_key_values") is not None:
                raise NotImplementedError("cache_position with a cache under transformers 5")
            kwargs.pop("cache_position")
        return original(*args, **kwargs)

    create_causal_mask._renames_input_embeds = True
    module.create_causal_mask = create_causal_mask


def dequantize_mxfp4(packed, scale, group_size: int = 32, dtype=None):
    """One `weight_packed` / `weight_scale` pair to a dense tensor, BF16 unless `dtype` says.

    The same steps as the engine's `dequantize_mxfp4`: low nibble first, the E2M1 table, a float32
    factor of `2 ** (scale - 127)` per group, then the cast.
    """
    import torch

    dtype = dtype or torch.bfloat16
    packed = packed.to(torch.uint8)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1)
    groups = scale.shape[-1]
    if groups * group_size != codes.shape[-1]:
        raise ValueError(f"{groups} scales at group_size {group_size} cover {groups * group_size} "
                         f"elements, but the packed array holds {codes.shape[-1]}")
    if tuple(scale.shape[:-1]) != tuple(codes.shape[:-1]):
        raise ValueError(f"scale shape {tuple(scale.shape)} doesn't match packed shape {tuple(codes.shape)}")
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32)[codes.long()]
    exponent = scale.to(torch.int32) - MXFP4_EXPONENT_BIAS
    factor = torch.ldexp(torch.ones(exponent.shape, dtype=torch.float32), exponent)
    grouped = values.reshape(*values.shape[:-1], groups, group_size) * factor[..., None]
    return grouped.reshape(values.shape).to(dtype)


class Checkpoint:
    """The safetensors of one snapshot directory, read one tensor at a time by memory map."""

    def __init__(self, model_path: str):
        self.path = model_path
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as fh:
                self.files = dict(json.load(fh)["weight_map"])
        else:
            from safetensors import safe_open

            single = "model.safetensors"
            with safe_open(os.path.join(model_path, single), "pt") as fh:
                self.files = {key: single for key in fh.keys()}
        self._handles = {}

    def __contains__(self, key: str) -> bool:
        return key in self.files

    def keys(self):
        return self.files.keys()

    def get(self, key: str):
        from safetensors import safe_open

        name = self.files[key]
        handle = self._handles.get(name)
        if handle is None:
            handle = safe_open(os.path.join(self.path, name), "pt")
            self._handles[name] = handle
        return handle.get_tensor(key)


class RemoteModel:
    """The text model out of a checkpoint's remote code, on meta, with a reader for its weights."""

    def __init__(self, model_path: str, num_layers: int | None = None):
        import transformers
        from accelerate import init_empty_weights
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        import fla_torch

        self.kernels = fla_torch.install()
        patch_transformers_compat()
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text = getattr(config, "text_config", None) or config
        self.full_layers = text.num_hidden_layers
        if num_layers is not None:
            if not 0 < num_layers <= text.num_hidden_layers:
                raise ValueError(f"--reference-layers {num_layers}: the model has {text.num_hidden_layers}")
            text.num_hidden_layers = num_layers
        self.num_layers = text.num_hidden_layers
        auto_map = getattr(text, "auto_map", None) or getattr(config, "auto_map", {})
        reference = auto_map.get("AutoModelForCausalLM")
        if reference is None:
            raise ValueError(f"{model_path}: the text config's auto_map names no AutoModelForCausalLM")
        cls = get_class_from_dynamic_module(reference, model_path)
        patch_module_compat(sys.modules[cls.__module__])
        quant = getattr(text, "quantization_config", None) or {}
        if not isinstance(quant, dict):
            quant = quant.to_dict()
        self.quant_format = quant.get("format")
        groups = quant.get("config_groups", {})
        self.group_size = next((g.get("weights", {}).get("group_size") for g in groups.values()), 32) or 32
        with init_empty_weights():
            model = cls(text)
        model.config._attn_implementation = "eager"
        model.eval()
        self.model = model
        self.text_config = text
        self.checkpoint = Checkpoint(model_path)
        self.prefix = self._find_prefix()
        self.shapes = {n: tuple(t.shape) for n, t in
                       list(model.named_parameters()) + list(model.named_buffers())}
        print(f"remote code: {cls.__module__}.{cls.__name__}, {self.num_layers} of "
              f"{self.full_layers} layers, fla kernels from {self.kernels}, weights under "
              f"{self.prefix!r}, {self.quant_format or 'no'} quantization", flush=True)
        self.check_coverage()

    def _find_prefix(self) -> str:
        names = [n for n, _ in self.model.named_parameters()]
        probe = "model.embed_tokens.weight" if "model.embed_tokens.weight" in names else names[0]
        found = [k[: -len(probe)] for k in self.checkpoint.keys() if k.endswith(probe)
                 and (k == probe or k[-len(probe) - 1] == ".")]
        if not found:
            raise ValueError(f"no checkpoint tensor ends in {probe}")
        return min(found, key=len)

    def sources(self, name: str) -> list[str]:
        """The checkpoint keys one parameter reads."""
        key = self.prefix + name
        if key in self.checkpoint:
            return [key]
        if name.endswith(".weight"):
            base = key[: -len("weight")]
            if base + "weight_packed" in self.checkpoint:
                return [base + "weight_packed", base + "weight_scale"]
        raise KeyError(f"{name}: the checkpoint holds no {key}")

    def check_coverage(self) -> None:
        """Every parameter has a tensor, and every tensor under the prefix feeds a parameter.

        With `num_layers` the layers past the cut stay unread. Tensors outside the prefix, the
        vision tower's for K3, stay unread and get counted.
        """
        used = set()
        for name in self.shapes:
            used.update(self.sources(name))
        under = [k for k in self.checkpoint.keys() if k.startswith(self.prefix)]
        layer_root = f"{self.prefix}model.layers."
        cut = []
        unused = []
        for key in under:
            if key in used:
                continue
            if key.startswith(layer_root) and int(key[len(layer_root):].split(".")[0]) >= self.num_layers:
                cut.append(key)
            else:
                unused.append(key)
        if unused:
            raise ValueError(f"{len(unused)} checkpoint tensors feed no parameter: {sorted(unused)[:5]}")
        outside = len(self.checkpoint.files) - len(under)
        print(f"coverage: {len(self.shapes)} parameters from {len(used)} tensors; "
              f"{len(cut)} tensors past the layer cut and {outside} outside {self.prefix!r} unread",
              flush=True)

    def value(self, name: str, dtype):
        """One parameter's value, from the checkpoint, in `dtype` when it's floating."""
        import torch

        keys = self.sources(name)
        if len(keys) == 2:
            if self.quant_format != MXFP4_FORMAT:
                raise ValueError(f"{name} ships packed, but the config's format is {self.quant_format}")
            value = dequantize_mxfp4(self.checkpoint.get(keys[0]), self.checkpoint.get(keys[1]),
                                     self.group_size, torch.bfloat16)
        else:
            value = self.checkpoint.get(keys[0])
        want = self.shapes[name]
        if tuple(value.shape) != want:
            if name.endswith("A_log") and value.dim() == 1 and len(want) == 1 and value.numel() > want[0]:
                tail = value[want[0]:]
                if torch.count_nonzero(tail):
                    raise ValueError(f"{keys[0]} holds nonzero values past its first {want[0]}")
                value = value[: want[0]].clone()
            else:
                raise ValueError(f"{keys[0]} is {tuple(value.shape)}, and {name} wants {want}")
        if value.is_floating_point():
            value = value.to(dtype)
        return value


def load(model_path: str, dtype_name: str, streamed: bool, num_layers: int | None = None):
    """The text model, every weight loaded, or with its decoder layers streamed one at a time."""
    import torch
    from accelerate.utils import set_module_tensor_to_device

    from check_capture import decoder_layers, release_freed_memory

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    remote = RemoteModel(model_path, num_layers)
    model = remote.model
    prefix, layers = decoder_layers(model)

    def put(module, name, value):
        # Without dtype=, accelerate casts the value to the meta tensor's dtype, float32 here,
        # and the BF16 pass would run float32 weights.
        set_module_tensor_to_device(module, name, "cpu", value=value,
                                    dtype=value.dtype if value.is_floating_point() else None)

    started = time.time()
    for name in remote.shapes:
        if streamed and name.startswith(prefix + "."):
            continue
        put(model, name, remote.value(name, dtype))
    print(f"loaded {'the non-layer' if streamed else 'every'} weight in {time.time() - started:.1f}s, "
          f"{dtype_name}", flush=True)
    if streamed:
        def load_layer(i):
            def hook(layer, args, kwargs=None):
                t0 = time.time()
                for local, _ in list(layer.named_parameters()) + list(layer.named_buffers()):
                    put(layer, local, remote.value(f"{prefix}.{i}.{local}", dtype))
                layer._stream_load_seconds = time.time() - t0
            return hook

        def drop(layer, args, output):
            for local, _ in list(layer.named_parameters()) + list(layer.named_buffers()):
                set_module_tensor_to_device(layer, local, "meta")
            gc.collect()
            release_freed_memory()

        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(load_layer(i))
            layer.register_forward_hook(drop)
        print(f"streamed reference: {len(layers)} layers under {prefix!r} load one at a time from "
              f"{model_path}, {dtype_name}", flush=True)
    model._remote = remote
    return model


def hidden_states(model, ids, log_layers: bool = False) -> list:
    """`[seq, d]` float64 arrays: the stream entering each decoder layer, then the final norm's output.

    The final norm's hook reads the one call the text model makes to it, after its last layer.
    """
    import torch

    from check_capture import decoder_layers

    _, layers = decoder_layers(model)
    states = []
    handles = []
    timing = {}

    def entering(layer, args, kwargs):
        timing[layer] = time.time()
        if layer is layers[0]:
            states.append(args[0] if args else kwargs["hidden_states"])

    def leaving(layer, args, kwargs, output):
        i = len(states) - 1
        if layer is not layers[-1]:
            states.append(output[0] if isinstance(output, tuple) else output)
        if log_layers:
            loaded = getattr(layer, "_stream_load_seconds", 0.0)
            print(f"  layer {i}: {time.time() - timing[layer]:.1f}s ({loaded:.1f}s loading)", flush=True)

    for layer in layers:
        handles.append(layer.register_forward_pre_hook(entering, with_kwargs=True, prepend=True))
        handles.append(layer.register_forward_hook(leaving, with_kwargs=True))
    handles.append(model.model.norm.register_forward_hook(lambda m, a, out: states.append(out)))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(states) != len(layers) + 1:
        raise RuntimeError(f"{len(layers)} layers gave {len(states)} hidden states, not {len(layers) + 1}")
    return [s[0].to(torch.float64).numpy() for s in states]


def reference_forward(model_path: str, batch: list, dtype_name: str, streamed: bool = False,
                      num_layers: int | None = None) -> list:
    """Per prompt, the hidden states `hidden_states` returns. Each prompt runs alone."""
    model = load(model_path, dtype_name, streamed, num_layers)
    out = []
    for n, ids in enumerate(batch):
        started = time.time()
        out.append(hidden_states(model, ids, log_layers=streamed))
        print(f"prompt {n}: {len(ids)} tokens, {dtype_name} forward in {time.time() - started:.1f}s",
              flush=True)
    del model
    gc.collect()
    return out
