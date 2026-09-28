#!/usr/bin/env python3
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

"""Capture per-layer activations from a served model into shards `sae/train.py` reads.

`sglang-jax` with `--enable-return-hidden-states` returns `[seq_len, slots, hidden_dim]` per
request. This drives the engine over a prompt file and streams the slots you ask for to `.npy`
shards plus a JSON manifest. It starts the engine with `--return-hidden-states-layers` set to
those slots, so the engine copies them alone to the host.

    python3 scripts/capture_activations.py --model-path MODEL --prompts prompts.txt \
        --out caps --layers 20 --shard-bytes 2147483648 --tp-size 8

    python3 sae/train.py --manifest caps/manifest.json --layer 20 \
        --expansion-factor 16 --k 100 --steps 100000 --out sae_l20.npz

Shards hold `[tokens, d_model]` for one layer and `[tokens, len(layers), d_model]` for several,
which are the two shapes `sae.train.activation_stream` accepts.

What a layer number means. The engine hook appends the residual stream entering block `i`, so
slot `i` holds the input to block `i`, which is the output of block `i - 1`. Slot 0 is the
embedding output. That's the HuggingFace `output_hidden_states[i]` convention, and it's what
`test_hidden_states_alignment.py` in the patch compares against. A model with `N` blocks gives
`N` slots, numbered 0 to `N - 1`, so the output of the last block has no slot.

Design points:

- The host holds one batch. Each request's activations go to the open shard as soon as the batch
  returns, and get dropped. `engine.generate` returns when the whole batch finishes, so
  `--batch-size` requests are live at once, in float32. At `--layers all`, Gemma 4 31B returns 60
  slots at 5,376 dim, which is 1.29 MB per token, so 8 requests of 2,048 tokens is about 21 GB in
  the host process. A narrower `--layers` shrinks the reply to its slots, and a lower
  `--batch-size` lowers the ceiling too. On an engine started without
  `--return-hidden-states-layers`, each reply gives up its kept slots chunk by chunk before the
  chunks join, so the copy this script makes holds those slots alone.
- A shard closes at the first prompt boundary at or above `--shard-bytes`. Whole prompts per
  shard is what lets a resume pick up where it stopped, so a crash costs one shard.
- A prompt counts once all its rows are on disk. A Ctrl-C or a failed write partway through a
  prompt leaves the shard cut back to the last whole prompt, and the resume captures that prompt
  again.
- Every shard carries its byte count and a SHA-256 of its data in the manifest, taken as the rows
  go to disk. A finished run reopens every shard, checks it against the manifest and prints a
  `verified` line with the shard count, the tokens and the bytes. `--verify` re-reads the tree,
  re-hashes it, and reports any shard that moved.
- The progress line reports wire bandwidth and disk bandwidth apart. Wire counts what the engine
  copied to the host: `len(layers) x d_model` elements a token when this script starts the
  engine, and every slot on an engine started without `--return-hidden-states-layers`. It counts
  the engine's serving dtype, 2 bytes an element for bf16 and 4 for float32. A server that steers
  widens the stream to float32 after its steering layer, so it counts 4 when a returned slot sits
  past that layer.
- The manifest records slot numbers, whatever position a slot takes in the reply. `--layers 20`
  on a 36-slot model gets a reply one slot wide, and the manifest says `layers: [20]` and
  `num_model_layers: 36`.
- `--dtype float16` holds values up to 65,504. A capture that holds a larger value, or an inf or
  a NaN from the engine, stops with an error instead of writing it.

Resuming a run points it at the same `--out` with the same prompt file. It reads the manifest,
drops any shard the crash left behind, skips the prompts already on disk, and appends. It refuses
a run whose model, layers, dtypes, token mode, sampling settings or prompt list differ.
`--layers all` stands for every slot the model has, the `num_model_layers` the first run recorded.

`--log-payloads`, or `LOG_PAYLOADS=1` in the environment, logs every request this script sends the
engine and a summary of every reply to stderr at debug level: prompts, sampling settings, token
counts, and the shape and dtype of each hidden state chunk. It never logs the activations.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "MANIFEST_NAME",
    "PAYLOAD_LOG",
    "CaptureConfig",
    "ShardWriter",
    "Progress",
    "capture",
    "check_layer_count",
    "check_row_counts",
    "enable_payload_log",
    "hidden_states_from_output",
    "layer_index",
    "log_replies",
    "log_request",
    "manifest_path",
    "manifest_shards",
    "engine_settings",
    "engine_slots",
    "open_engine",
    "parse_engine_args",
    "parse_layers",
    "payload_log_requested",
    "read_manifest",
    "read_prompts",
    "reply_positions",
    "reply_shape",
    "reply_summary",
    "verify_manifest",
    "verified_line",
    "write_manifest",
]

MANIFEST_NAME = "manifest.json"
SHARD_PREFIX = "shard-"

# Reserved `.npy` header, a multiple of 64 so the data starts aligned. The writer stamps a
# placeholder count, appends rows, then rewrites the count in place at close. Padding the header
# to a fixed width is what lets that second write land on the same bytes.
HEADER_BYTES = 128

DTYPES = {"float32": np.float32, "float16": np.float16}

# Bytes per element on the device-to-host hop, by what the engine serves in. `half` and `float` are
# the engine's own names for float16 and float32. The patch widens to float32 on the host after the
# copy, so the array this file sees is wider than the wire was.
ENGINE_ITEMSIZE = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4}

# Keys the capture sets itself, and the flag that sets each. `--engine-arg` refuses them, because
# the Engine would get the keyword twice.
RESERVED_ENGINE_ARGS = {
    "model_path": "use --model-path",
    "enable_return_hidden_states": "the capture always turns it on",
    "return_hidden_states_layers": "use --layers, which the capture passes to the engine",
    "tp_size": "use --tp-size",
    "dtype": "use --engine-dtype",
    "batch_size": "use --batch-size",
    "token_padding": "use --token-padding",
}

# `--engine-arg` values typed the Python way. JSON reads `true`, `false` and `null` and refuses
# these, and the string "False" would reach the Engine as a true value.
PYTHON_LITERALS = {"True": True, "False": False, "None": None}


# --- payload log ----------------------------------------------------------------------------

# Requests this script hands the engine and summaries of what comes back. Off unless a script
# turns it on, because a capture sends thousands of requests.
PAYLOAD_LOG = logging.getLogger("capture_activations.payload")


def payload_log_requested(flag: bool = False) -> bool:
    """True when `--log-payloads` is set or `LOG_PAYLOADS` holds anything but empty or 0."""
    return bool(flag) or os.environ.get("LOG_PAYLOADS", "").strip() not in ("", "0")


def enable_payload_log(stream=None) -> None:
    """Send the payload log to stderr, or to `stream`, at debug level."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    PAYLOAD_LOG.addHandler(handler)
    PAYLOAD_LOG.setLevel(logging.DEBUG)
    # The engine configures the root logger for its own lines. This log keeps its own handler.
    PAYLOAD_LOG.propagate = False


def _describe(value: Any) -> Any:
    """A value for the log: arrays become their shape and dtype, everything else stays."""
    if isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        return {"shape": [int(d) for d in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    return value


def log_request(**payload: Any) -> None:
    """Log one engine call: what goes in, whole, at debug level."""
    if PAYLOAD_LOG.isEnabledFor(logging.DEBUG):
        described = _describe(payload)
        PAYLOAD_LOG.debug(
            "request %s", json.dumps(described, default=str), extra={"payload": {"request": described}}
        )


def reply_summary(output: dict) -> dict:
    """What came back for one request, with every array reduced to its shape and dtype."""
    summary = {key: _describe(value) for key, value in output.items() if key != "meta_info"}
    meta = output.get("meta_info") or {}
    summary["meta_info_keys"] = sorted(meta)
    summary["meta_info"] = {key: _describe(value) for key, value in meta.items()}
    return summary


def log_replies(outputs: Sequence[dict], first_index: int = 0) -> None:
    """Log a summary of each reply at debug level. `first_index` numbers them within the run."""
    if not PAYLOAD_LOG.isEnabledFor(logging.DEBUG):
        return
    for offset, output in enumerate(outputs):
        summary = dict(reply_summary(output), prompt_index=first_index + offset)
        PAYLOAD_LOG.debug(
            "reply %s", json.dumps(summary, default=str), extra={"payload": {"reply": summary}}
        )


# --- npy shard writing ----------------------------------------------------------------------


def _npy_header(shape: tuple, dtype: np.dtype, total: int = HEADER_BYTES) -> bytes:
    """A version 2.0 `.npy` header padded to `total` bytes."""
    dims = ", ".join(str(int(d)) for d in shape)
    if len(shape) == 1:
        dims += ","
    text = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (
        np.lib.format.dtype_to_descr(np.dtype(dtype)),
        dims,
    )
    body = total - 12  # 6 magic bytes, 2 version bytes, 4 length bytes.
    if len(text) + 1 > body:
        raise ValueError(f"header needs {len(text) + 1} bytes, the reserved region holds {body}")
    text = text.ljust(body - 1) + "\n"
    return b"\x93NUMPY" + bytes([2, 0]) + int(body).to_bytes(4, "little") + text.encode("latin1")


def _non_finite(source: np.ndarray, narrowed: np.ndarray) -> str:
    """Why a block can't go on disk: the engine sent inf or NaN, or the dtype can't hold it."""
    bad = int(np.size(narrowed) - np.count_nonzero(np.isfinite(narrowed)))
    if not np.isfinite(source).all():
        return (
            f"the engine returned {bad} inf or NaN value(s) in this block. A capture that holds "
            f"them trains nothing, so check the model and the serving dtype."
        )
    peak = float(np.max(np.abs(source)))
    top = float(np.finfo(narrowed.dtype).max)
    return (
        f"{bad} value(s) overflow {narrowed.dtype}, which holds up to {top:g}; this block reaches "
        f"{peak:g}. Capture at --dtype float32."
    )


class ShardWriter:
    """One `.npy` file, appended a block of tokens at a time.

    The row shape is everything past the token axis, so `(d_model,)` for a single layer and
    `(layers, d_model)` for several.

    A block counts once `append` returns. A Ctrl-C or a failed write partway through a block can
    leave some of its bytes on disk, so `close` cuts the file back to the last block that counts,
    and the header and the SHA-256 cover those rows and no others. The count, the digest and the
    prompt tally sit in one tuple that one assignment replaces, so an interrupt lands before a
    block counts or after it, never between.
    """

    def __init__(self, path: str, row_shape: Sequence[int], dtype, checksum: bool = True):
        self.path = path
        self.row_shape = tuple(int(d) for d in row_shape)
        self.dtype = np.dtype(dtype)
        self.row_bytes = int(np.prod(self.row_shape)) * self.dtype.itemsize
        self._committed = (0, hashlib.sha256() if checksum else None, 0)
        self._record = None
        # Unbuffered, so a failed write leaves bytes on disk and none in a Python buffer that a
        # later flush would push out.
        self._fp = open(path, "wb", buffering=0)
        self._write(_npy_header((0,) + self.row_shape, self.dtype))

    @property
    def rows(self) -> int:
        """Rows that count."""
        return self._committed[0]

    @property
    def prompts(self) -> int:
        """Prompts whose rows all count."""
        return self._committed[2]

    @property
    def nbytes(self) -> int:
        """Data bytes that count, header excluded."""
        return self.rows * self.row_bytes

    def _write(self, data) -> None:
        view = memoryview(data).cast("B")
        while view.nbytes:
            view = view[self._fp.write(view) :]

    def append(self, block: np.ndarray, prompts: int = 1) -> int:
        """Write `[tokens, *row_shape]` for `prompts` prompt(s) and return the bytes it cost."""
        source = np.asarray(block)
        if source.ndim != len(self.row_shape) + 1 or tuple(source.shape[1:]) != self.row_shape:
            raise ValueError(f"block is {source.shape}, the shard takes rows of {self.row_shape}")
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.ascontiguousarray(source, dtype=self.dtype)
        if not np.isfinite(narrowed).all():
            raise ValueError(f"{os.path.basename(self.path)}: {_non_finite(source, narrowed)}")
        rows, digest, done = self._committed
        raw = memoryview(narrowed).cast("B")
        if digest is not None:
            digest = digest.copy()
            digest.update(raw)
        self._fp.seek(HEADER_BYTES + rows * self.row_bytes)
        self._write(raw)
        self._committed = (rows + int(narrowed.shape[0]), digest, done + int(prompts))
        return len(raw)

    def credit(self, prompts: int = 1) -> None:
        """Count prompts that kept no rows as part of this shard."""
        rows, digest, done = self._committed
        self._committed = (rows, digest, done + int(prompts))

    def close(self) -> dict:
        """Cut the file to the rows that count, stamp their count, and return the manifest record.

        A second call returns the same record, so a close that an interrupt cut short can run
        again.
        """
        if self._record is None:
            rows, digest, _ = self._committed
            data_end = HEADER_BYTES + rows * self.row_bytes
            if not self._fp.closed:
                self._fp.truncate(data_end)
                self._fp.seek(0)
                self._write(_npy_header((rows,) + self.row_shape, self.dtype))
                os.fsync(self._fp.fileno())
                self._fp.close()
            self._record = {
                "path": os.path.basename(self.path),
                "tokens": rows,
                "bytes": data_end,
                "sha256": digest.hexdigest() if digest is not None else None,
            }
        return self._record


# --- manifest -------------------------------------------------------------------------------


def manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, MANIFEST_NAME)


def write_manifest(out_dir: str, payload: dict) -> None:
    """Write the manifest through a temporary file, so a crash never leaves a half one."""
    target = manifest_path(out_dir)
    staging = target + ".partial"
    with open(staging, "w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(staging, target)


def read_manifest(out_dir: str) -> dict | None:
    """The manifest in `out_dir`, or None when there isn't one. Takes the file path too."""
    path = out_dir if out_dir.endswith(".json") else manifest_path(out_dir)
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        return json.load(fp)


def manifest_shards(out_dir: str) -> list:
    """Shard paths in write order, so `sae.train.activation_stream` needs no glob."""
    path = out_dir if out_dir.endswith(".json") else manifest_path(out_dir)
    manifest = read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"no manifest at {path}")
    root = os.path.dirname(os.path.abspath(path))
    return [os.path.join(root, shard["path"]) for shard in manifest["shards"]]


def layer_index(manifest: dict, model_layer: int) -> int | None:
    """Where a capture slot sits on the shard's layer axis, or None for a single-layer shard.

    `model_layer` is a slot number, and slot `i` holds the residual stream entering block `i`.
    """
    layers = list(manifest["layers"])
    if model_layer not in layers:
        raise ValueError(f"layer {model_layer} isn't in the capture, which holds {layers}")
    if not manifest.get("layer_axis", len(layers) > 1):
        return None
    return layers.index(model_layer)


def verify_manifest(out_dir: str, checksum: bool = True) -> list:
    """Re-read the tree and return one string per problem. An empty list means it's clean.

    Per shard it checks that the file is there, that its length matches, that its `.npy` header
    reads and holds the token count the manifest claims, that its rank matches `layer_axis`, that
    its layer axis is as wide as the `layers` list, that its last axis is `d_model` wide, that its
    dtype matches, and that its data hashes to the recorded SHA-256. A shard that fails one check
    doesn't stop the checks on the next. What stays outside reach is which layer each slot came
    from. The bytes carry no label, so a `layers` list rewritten to another list of the same
    length reads clean.
    """
    path = out_dir if out_dir.endswith(".json") else manifest_path(out_dir)
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError) as exc:
        return [f"{path} doesn't read as JSON: {exc}"]
    if manifest is None:
        return [f"no manifest at {path}"]
    root = os.path.dirname(os.path.abspath(path))
    problems = []
    layers = list(manifest.get("layers") or [])
    want_ndim = 3 if manifest.get("layer_axis") else 2
    for record in manifest["shards"]:
        shard = os.path.join(root, record["path"])
        if not os.path.exists(shard):
            problems.append(f"{record['path']}: missing")
            continue
        size = os.path.getsize(shard)
        if size != record["bytes"]:
            problems.append(f"{record['path']}: {size} bytes on disk, manifest says {record['bytes']}")
            continue
        try:
            array = np.load(shard, mmap_mode="r")
        except (OSError, ValueError, EOFError, OverflowError) as exc:
            # OverflowError is what np.load raises on a negative count or one past a C long.
            problems.append(f"{record['path']}: the .npy header doesn't read: {exc}")
            continue
        if array.ndim == 0:
            problems.append(f"{record['path']}: the .npy header holds a scalar, with no token axis")
            del array
            continue
        if array.shape[0] != record["tokens"]:
            problems.append(
                f"{record['path']}: header holds {array.shape[0]} tokens, "
                f"manifest says {record['tokens']}"
            )
        if array.ndim != want_ndim:
            problems.append(
                f"{record['path']}: {array.ndim} dimension(s), "
                f"layer_axis={bool(manifest.get('layer_axis'))} calls for {want_ndim}"
            )
        elif want_ndim == 3 and array.shape[1] != len(layers):
            problems.append(
                f"{record['path']}: {array.shape[1]} slot(s) on the layer axis, "
                f"manifest lists {len(layers)} ({layers})"
            )
        if array.shape[-1] != manifest["d_model"]:
            problems.append(
                f"{record['path']}: {array.shape[-1]} wide, manifest says {manifest['d_model']}"
            )
        if str(array.dtype) != manifest["dtype"]:
            problems.append(f"{record['path']}: dtype {array.dtype}, manifest says {manifest['dtype']}")
        del array
        if checksum and record.get("sha256"):
            digest = hashlib.sha256()
            with open(shard, "rb") as fp:
                fp.seek(HEADER_BYTES)
                for block in iter(lambda: fp.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                problems.append(f"{record['path']}: sha256 mismatch, the shard is corrupt")
    # Summed over every record, so a shard that failed a check above doesn't also read as a
    # manifest total that's off.
    total = sum(record["tokens"] for record in manifest["shards"])
    if total != manifest["tokens"]:
        problems.append(f"shards hold {total} tokens, manifest says {manifest['tokens']}")
    return problems


def verified_line(manifest: dict, out_dir: str) -> str:
    """What a finished capture prints once `verify_manifest` finds nothing wrong with it."""
    row = [len(manifest["layers"]), manifest["d_model"]] if manifest.get("layer_axis") else [
        manifest["d_model"]]
    shape = ", ".join(["tokens"] + [str(d) for d in row])
    size = sum(record["bytes"] for record in manifest["shards"])
    hashed = all(record.get("sha256") for record in manifest["shards"])
    return (
        f"verified {len(manifest['shards'])} shard(s) in {out_dir}: {manifest['tokens']} "
        f"token(s), {size} byte(s). Each shard reopens as [{shape}] {manifest['dtype']}, at the "
        f"size and token count manifest.json lists. "
        + (
            "Each SHA-256 was taken as its rows went to disk; --verify reads them back and "
            "re-hashes them."
            if hashed
            else "No shard carries a SHA-256 (--no-checksum)."
        )
    )


# --- engine ---------------------------------------------------------------------------------


def engine_settings(batch_size: int = 8, token_padding: int = 1024, **kwargs) -> dict:
    """The keyword arguments `open_engine` hands the Engine.

    `chunked_prefill_size` serves the capture rather than the deployment. It bounds the
    device-side cost of the captured slots. `disable_radix_cache` costs a capture nothing,
    because a request for hidden states never reuses a cached prefix. It also lets a model with a
    `linear_recurrent_config` start, since the engine refuses one with the cache on unless the
    unified radix tree is on too.

    `precompile_bs_paddings` and `precompile_token_paddings` each pin one bucket. Left unset the
    engine compiles its whole default ladder, and on a `v5litepod-8` that sat over 40 minutes on
    the first `[EXTEND]` shape at `bs=3, tokens=64` and produced nothing. One bucket each takes
    12.4 seconds. A batch longer than the token bucket prefills over several passes of
    `chunked_prefill_size`, which caps the bucket: a `token_padding` above 1024 has no effect.

    `trust_remote_code` is on, so `transformers` imports and runs the Python files a model repo's
    `auto_map` names for its config or tokenizer. Kimi K3's tokenizer is such a file. `--engine-arg trust_remote_code=False`
    turns it off for a repo you haven't read.

    `log_requests` stays off. The engine's request log prints each reply's `meta_info` whole, and
    with capture on that's every hidden state array. `--log-payloads` logs the same traffic with
    each array reduced to its shape and dtype.

    Everything else is a plain default, and `--engine-arg KEY=VALUE` replaces any of them. That's
    how a model's own launch recipe reaches the engine. `scripts/measure_model.sh` passes
    `--engine-arg mem_fraction_static=$MEM_FRAC`, and GLM-5.3 takes
    `--engine-arg attention_backend=dsa_sparse` for its sparse indexer. A key a flag sets, such as
    `tp_size` or the capture's `dtype`, goes through the flag, and `parse_engine_args` refuses it.
    """
    settings = {
        "trust_remote_code": True,
        "tp_size": 1,
        "device": "tpu",
        "dtype": "bfloat16",
        "mem_fraction_static": 0.6,
        "chunked_prefill_size": 1024,
        "disable_radix_cache": True,
        "log_requests": False,
        "attention_backend": "fa",
        "page_size": 64,
        "skip_server_warmup": True,
        "max_running_requests": batch_size,
        "precompile_bs_paddings": [batch_size],
        "precompile_token_paddings": [token_padding],
    }
    settings.update(multinode_settings())
    settings.update(kwargs)
    return settings


def multinode_settings(env: dict | None = None) -> dict:
    """Engine keywords for one host of a multi-host slice, read from the environment.

    `scripts/multihost_exec.sh` runs the same command on every host and sets `SGL_NNODES`,
    `SGL_NODE_RANK` and `SGL_DIST_INIT_ADDR` (host 0's internal address and a port) on each. Rank 0
    serves the requests; every other rank's Engine starts its scheduler and blocks there, which
    is how sglang-jax runs a non-zero rank. With `SGL_NNODES` unset or 1 this returns nothing, so a
    single-host run is unchanged.
    """
    env = os.environ if env is None else env
    nnodes = int(env.get("SGL_NNODES", "1") or 1)
    if nnodes <= 1:
        return {}
    rank = int(env["SGL_NODE_RANK"])
    addr = env["SGL_DIST_INIT_ADDR"]
    if not 0 <= rank < nnodes:
        raise ValueError(f"SGL_NODE_RANK={rank} is outside 0..{nnodes - 1}")
    return {"nnodes": nnodes, "node_rank": rank, "dist_init_addr": addr}


def parse_engine_args(items: Iterable[str], reserved: dict | None = None) -> dict:
    """`--engine-arg KEY=VALUE` strings to Engine keywords.

    VALUE goes through JSON, then `PYTHON_LITERALS`, and falls back to a string. `reserved` maps a
    key the script sets itself to what to do instead, and such a key raises ValueError before
    anything loads. So does an item with no `=`, which would hand the Engine an empty string.

    `json_model_override_args` stays a JSON string. `ServerArgs` types it as one and runs
    `json.loads` on it, so the dict JSON decoding makes of `{"num_hidden_layers": 4}` would raise
    TypeError inside the engine.
    """
    reserved = RESERVED_ENGINE_ARGS if reserved is None else reserved
    extra = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--engine-arg takes KEY=VALUE, got {item!r}")
        if key in reserved:
            raise ValueError(f"--engine-arg {key}: this script sets {key} itself; {reserved[key]}")
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = PYTHON_LITERALS.get(value.strip(), value)
        if key == "json_model_override_args" and isinstance(extra[key], dict):
            extra[key] = json.dumps(extra[key], sort_keys=True)
    return extra


# Printed before an Engine loads, because the load prints two lines that look wrong and aren't.
ENGINE_LOAD_NOTE = (
    'loading the engine. On Gemma 4 it prints "Loading MoE Weights: 0it", because gemma4.py loads '
    "the experts in a pass of its own after that one. On a model with an image processor, "
    "transformers warns that `use_fast` is deprecated; a text prompt never reaches that processor."
)


def open_engine(model_path: str, layers: Sequence[int] | None = None, **kwargs) -> Any:
    """A patched `sglang-jax` Engine with per-layer capture turned on.

    `layers` names the capture slots the engine returns, in ascending order, and the engine
    copies those alone to the host. None returns every slot.

    The import lives here so the rest of this file loads on a host with no engine installed.
    """
    from sgl_jax.srt.entrypoints.engine import Engine

    settings = engine_settings(**kwargs)
    if layers is not None:
        settings["return_hidden_states_layers"] = [int(slot) for slot in layers]
    log_request(call="Engine", model_path=model_path, enable_return_hidden_states=True, **settings)
    print(ENGINE_LOAD_NOTE, flush=True)
    return Engine(model_path=model_path, enable_return_hidden_states=True, **settings)


def check_row_counts(meta_info: dict, rows: int, prompt_rows: int) -> None:
    """Compare what came back against the token counts the engine reports beside it.

    The engine returns `prompt_tokens + completion_tokens - 1` rows: one per prompt token from
    prefill, then one per decode step except the last generated token, which no forward pass
    reads. An engine built with the current `upstream/sglang-jax-877.patch` returns them all,
    because a request for hidden states matches no cached prefix and each prefill chunk appends
    its rows. This check catches an engine that doesn't. Without it a short array is written,
    checksummed and reported clean.
    """
    want_prompt = meta_info.get("prompt_tokens")
    want_completion = meta_info.get("completion_tokens")
    if want_prompt is None or want_completion is None:
        return
    want_prompt, want_completion = int(want_prompt), int(want_completion)
    want_rows = want_prompt + max(want_completion, 1) - 1
    if prompt_rows != want_prompt:
        raise ValueError(
            f"prefill returned {prompt_rows} row(s) and the engine counts {want_prompt} prompt "
            f"token(s). The current upstream/sglang-jax-877.patch returns one row per prompt "
            f"token, so check that the engine tree carries it. scripts/bootstrap_tpu_vm.sh "
            f"builds one that does."
        )
    if rows != want_rows:
        raise ValueError(
            f"the request returned {rows} row(s). The engine counts {want_prompt} prompt "
            f"token(s) and {want_completion} completion token(s), which is {want_rows} row(s)."
        )


UNHOOKED = (
    "the reply carries no [seq_len, slots, d_model] prefill chunk, which is what a server "
    "without per-layer capture sends. Start it with --enable-return-hidden-states on a tree with "
    "sglang-jax-877.patch, and use a model with the capture hook."
)


def _chunks(output: dict) -> list:
    chunks = output["meta_info"]["hidden_states"]
    if not chunks:
        raise ValueError("the request came back with no hidden states; is the flag on?")
    return chunks


def reply_shape(output: dict) -> tuple:
    """`(slots, d_model)` of one reply, read off its first prefill chunk without copying it."""
    for chunk in _chunks(output):
        shape = np.shape(chunk)
        if len(shape) == 3:
            return int(shape[1]), int(shape[2])
    raise ValueError(UNHOOKED)


def hidden_states_from_output(output: dict, positions: Sequence[int] | None = None) -> tuple:
    """Flatten one request's chunk list to `[tokens, slots, d_model]`.

    Prefill chunks arrive as `[seq_len, slots, d_model]` and each decode step adds a
    `[slots, d_model]` chunk. A prompt longer than `chunked_prefill_size` arrives as several
    prefill chunks, so the prefill rows accumulate rather than replace. Returns the array and the
    prefill row count, so a caller can keep prompt tokens and completion tokens apart.

    `positions` keeps those places on the reply's slot axis, from each chunk before the chunks
    join, so the host copies them alone and returns `[tokens, len(positions), d_model]`. On an
    engine that returns every slot, a position is the slot number. `reply_positions` maps slot
    numbers to positions on an engine that returns a subset.

    A server without per-layer capture sends `[seq_len, d_model]` for the prefill and `[d_model]`
    per step: the last layer alone. That reply has no prefill chunk with a layer axis, and it
    raises here.
    """
    keep = None if positions is None else list(positions)
    stacked = []
    prompt_rows = 0
    for chunk in _chunks(output):
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim != 3:
            raise ValueError(f"a hidden state chunk is {array.shape}; {UNHOOKED}")
        else:
            prompt_rows += array.shape[0]
        stacked.append(array if keep is None else array[:, keep, :])
    if prompt_rows == 0:
        raise ValueError(UNHOOKED)
    return np.concatenate(stacked, axis=0), prompt_rows


# --- progress -------------------------------------------------------------------------------


class Progress:
    """Tokens per second, plus the two bandwidths, on a wall-clock interval.

    Wire counts everything the engine moved to the host: the slots it returns, which are the
    kept slots when `open_engine` started it with them. Disk counts what goes to the shards. When
    wire sits near the NIC's rate, the host link bounds the capture and not the chips.
    """

    def __init__(self, log: Callable[[str], None] = print, every: float = 5.0):
        self.log = log
        self.every = every
        self.tokens = 0
        self.wire_bytes = 0
        self.disk_bytes = 0
        self.shards = 0
        self.started = time.time()
        self._last = self.started

    def update(self, tokens: int, wire_bytes: int, disk_bytes: int) -> None:
        self.tokens += tokens
        self.wire_bytes += wire_bytes
        self.disk_bytes += disk_bytes

    def line(self) -> str:
        elapsed = max(time.time() - self.started, 1e-9)
        return json.dumps(
            {
                "tokens": self.tokens,
                "shards": self.shards,
                "elapsed_s": round(elapsed, 2),
                "tokens_per_s": round(self.tokens / elapsed, 1),
                "wire_mb_per_s": round(self.wire_bytes / elapsed / 1e6, 1),
                "disk_mb_per_s": round(self.disk_bytes / elapsed / 1e6, 1),
                "wire_bytes": self.wire_bytes,
                "disk_bytes": self.disk_bytes,
                "disk_gib": round(self.disk_bytes / (1 << 30), 3),
            }
        )

    def maybe_log(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last >= self.every:
            self._last = now
            self.log(self.line())


# --- capture --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CaptureConfig:
    """What to keep and how to cut it up. `layers=None` keeps every layer.

    `tokens="completion"` keeps the decode rows. `n` new tokens give `n - 1` rows, because the
    engine never runs a forward pass on the last one it generates, so `max_new_tokens` has to be
    2 or more for that mode to write anything.

    `engine_dtype` is what the engine serves in. The manifest records it, and a resume refuses a
    different one. None reads it off the engine's `server_args` when there's one to read. The
    wire count takes its bytes per element from it: 4 for float32, and 2 for bf16, float16 or a
    dtype it can't read. A server started with `--enable-steering` counts 4 whatever it serves
    when a returned slot sits past its steering layer.
    """

    model: str
    layers: tuple | None = None
    dtype: str = "float32"
    shard_bytes: int = 2 << 30
    tokens: str = "all"  # all, prompt, or completion
    layer_axis: bool | None = None  # None keeps the axis only when there's more than one layer
    checksum: bool = True
    batch_size: int = 8
    max_new_tokens: int = 1
    token_budget: int | None = None
    sampling_params: dict | None = None
    engine_dtype: str | None = None


def parse_layers(spec: str) -> tuple | None:
    """`20`, `10,20,30`, `0-5`, `0-59:4` or `all`. Returns None for `all`."""
    if spec.strip().lower() == "all":
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            span, _, stride = part.partition(":")
            low, _, high = span.partition("-")
            out.extend(range(int(low), int(high) + 1, int(stride) if stride else 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"no layers in {spec!r}")
    return tuple(sorted(set(out)))


def read_prompts(path: str, text_field: str = "text") -> list:
    """One prompt per line, or one JSON object per line when the name ends in `.jsonl`."""
    prompts = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                prompts.append(json.loads(line)[text_field])
            else:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"{path} holds no prompts")
    return prompts


def _prompt_fingerprint(prompts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _select_rows(hidden: np.ndarray, prompt_rows: int, which: str) -> np.ndarray:
    """The rows one mode keeps. `completion` holds every generated token but the last."""
    if which == "all":
        return hidden
    if which == "prompt":
        return hidden[:prompt_rows]
    if which == "completion":
        return hidden[prompt_rows:]
    raise ValueError(f"--tokens takes all, prompt or completion, got {which!r}")


def _sampling(cfg: CaptureConfig) -> dict:
    """The sampling settings every request carries, as the manifest stores them."""
    sampling = dict(cfg.sampling_params or {"temperature": 0})
    sampling.setdefault("max_new_tokens", cfg.max_new_tokens)
    return json.loads(json.dumps(sampling, sort_keys=True))


def _engine_dtype(engine: Any, cfg: CaptureConfig) -> str | None:
    if cfg.engine_dtype is not None:
        return str(cfg.engine_dtype)
    dtype = getattr(getattr(engine, "server_args", None), "dtype", None)
    return None if dtype is None else str(dtype)


def _wire_itemsize(
    engine: Any, engine_dtype: str | None, slots: Iterable[int] | None = None
) -> int:
    """Bytes per element on the device-to-host hop.

    The wire carries the serving dtype, 2 bytes when it's unknown. A server started with
    `--enable-steering` widens the stream to float32 after its `--steering-layer`, so slot
    `steering_layer + 1` is the first wide one. The capture's concat widens every returned slot to
    match when one of them sits past that layer, and leaves them in the serving dtype when none
    does. `slots` names the slots the engine returns. A steering server counts 4 when `slots` or
    its steering layer is unknown.
    """
    args = getattr(engine, "server_args", None)
    if getattr(args, "enable_steering", False) is True:
        layer = getattr(args, "steering_layer", None)
        if slots is None or layer is None or any(int(s) > int(layer) for s in slots):
            return 4
    return ENGINE_ITEMSIZE.get(str(engine_dtype), 2)


def engine_slots(engine: Any) -> list | None:
    """The slots the engine returns, in reply order, or None when it returns every slot.

    It reads the `return_hidden_states_layers` the engine started with.
    """
    layers = getattr(getattr(engine, "server_args", None), "return_hidden_states_layers", None)
    return None if layers is None else [int(slot) for slot in layers]


def _model_slot_count(engine: Any) -> int | None:
    """The model's slot count, off the engine's model config, or None when there's none to read."""
    config = getattr(getattr(engine, "tokenizer_manager", None), "model_config", None)
    count = getattr(config, "num_hidden_layers", None)
    return None if count is None else int(count)


def reply_positions(layers: Sequence[int], returned: Sequence[int] | None) -> list:
    """Where each slot in `layers` sits on the reply's slot axis.

    `returned` is what `engine_slots` gives: the slots the engine returns, or None for every
    slot, where a slot's position is its number.
    """
    if returned is None:
        return [int(slot) for slot in layers]
    returned = [int(slot) for slot in returned]
    missing = [int(slot) for slot in layers if slot not in returned]
    if missing:
        raise ValueError(
            f"the capture keeps slot(s) {missing}, and the engine returns slots {returned}. "
            f"Start the engine with --return-hidden-states-layers set to the capture's --layers, "
            f"as open_engine does."
        )
    return [returned.index(slot) for slot in layers]


def _resume_state(
    out_dir: str, cfg: CaptureConfig, fingerprint: str, sampling: dict, engine_dtype, log
) -> dict | None:
    """Check an existing manifest against this run and return it, or None to start fresh."""
    manifest = read_manifest(out_dir)
    if manifest is None:
        return None
    mismatches = []
    if manifest["model"] != cfg.model:
        mismatches.append(f"model {manifest['model']!r} against {cfg.model!r}")
    if manifest["dtype"] != cfg.dtype:
        mismatches.append(f"dtype {manifest['dtype']} against {cfg.dtype}")
    if manifest["tokens_kept"] != cfg.tokens:
        mismatches.append(f"--tokens {manifest['tokens_kept']} against {cfg.tokens}")
    # `--layers all` asks for every slot the model has, which the manifest records once the first
    # reply lands. Until then there's no list to hold it against.
    if cfg.layers is not None:
        wanted, asked = list(cfg.layers), str(list(cfg.layers))
    elif manifest.get("num_model_layers") is not None:
        wanted = list(range(int(manifest["num_model_layers"])))
        asked = f"all {len(wanted)} slots (--layers all)"
    else:
        wanted = asked = None
    if wanted is not None and manifest["layers"] is not None and list(manifest["layers"]) != wanted:
        mismatches.append(f"layers {manifest['layers']} against {asked}")
    if (
        cfg.layer_axis is not None
        and manifest["layer_axis"] is not None
        and manifest["layer_axis"] != cfg.layer_axis
    ):
        mismatches.append(f"layer_axis {manifest['layer_axis']} against {cfg.layer_axis}")
    if manifest["prompts_sha256"] != fingerprint:
        mismatches.append("a different prompt list")
    # A different generation length or sampling setting gives each prompt a different set of
    # rows, and a different serving dtype gives them a different precision. Both would mix into
    # one capture with nothing on disk to tell them apart.
    if "sampling_params" not in manifest:
        mismatches.append(
            "no sampling_params in the manifest, so the run it came from can't be matched on "
            "--max-new-tokens"
        )
    else:
        stored = manifest["sampling_params"]
        for key in sorted(set(stored) | set(sampling)):
            if stored.get(key) != sampling.get(key):
                name = "--max-new-tokens" if key == "max_new_tokens" else f"sampling {key}"
                mismatches.append(f"{name} {stored.get(key)!r} against {sampling.get(key)!r}")
    if manifest.get("engine_dtype") != engine_dtype:
        mismatches.append(f"engine dtype {manifest.get('engine_dtype')} against {engine_dtype}")
    if mismatches:
        raise ValueError(
            f"{manifest_path(out_dir)} was written by another run: "
            + "; ".join(mismatches)
            + ". Point --out somewhere else or delete it."
        )
    log(
        f"resuming: {len(manifest['shards'])} shard(s), {manifest['tokens']} token(s), "
        f"{manifest['prompts_done']} prompt(s) done"
    )
    return manifest


def _drop_orphans(out_dir: str, keep: Iterable[str], log) -> None:
    """Delete shards the manifest doesn't list. That's what a crash leaves behind."""
    keep = set(keep)
    for name in sorted(os.listdir(out_dir)):
        if name.startswith(SHARD_PREFIX) and name.endswith(".npy") and name not in keep:
            os.remove(os.path.join(out_dir, name))
            log(f"dropped {name}, which no manifest entry covers")


def _with_shape(manifest: dict, num_layers: int | None, d_model: int) -> dict:
    """The manifest with the fields the first reply fixes: slot count, width, layers, layer axis.

    `num_layers` is the model's slot count, None when an engine that returns a subset gives no
    way to read it.
    """
    if manifest["layers"] is not None:
        layers = manifest["layers"]
    elif num_layers is not None:
        layers = list(range(num_layers))
    else:
        raise ValueError(
            "--layers all needs the model's slot count, and this engine returns a subset of "
            "slots with no model config to read the count off"
        )
    if num_layers is not None and max(layers) >= num_layers:
        raise ValueError(
            f"asked for slot {max(layers)}, the capture holds {num_layers} slot(s), "
            f"0 to {num_layers - 1}"
        )
    layer_axis = manifest["layer_axis"]
    if layer_axis is None:
        layer_axis = len(layers) > 1
    return dict(
        manifest, num_model_layers=num_layers, d_model=d_model, layers=list(layers),
        layer_axis=layer_axis,
    )


def check_layer_count(manifest: dict, slots: int, returned: Sequence[int] | None = None) -> None:
    """Refuse a reply whose slot count differs from what the engine returns.

    `returned` is what `engine_slots` gives. An engine started with `--return-hidden-states-layers`
    returns that many slots, and one started without it returns the model's slot count, which the
    manifest records.
    """
    want = manifest["num_model_layers"] if returned is None else len(returned)
    if slots != want:
        raise ValueError(f"a request returned {slots} slot(s), and the engine returns {want}")


def capture(
    engine: Any,
    prompts: Sequence[str],
    cfg: CaptureConfig,
    out_dir: str,
    log: Callable[[str], None] = print,
    log_every: float = 5.0,
) -> dict:
    """Drive the engine over `prompts` and write shards plus a manifest into `out_dir`.

    Args:
      engine: anything with `generate(prompt, sampling_params, return_hidden_states=True)`.
      prompts: the prompt list. A resume needs the same list in the same order.
      cfg: layers, dtype and shard size.
      out_dir: destination. An existing manifest here resumes.
      log: line sink.
      log_every: seconds between progress lines.

    Returns:
      the manifest it wrote.
    """
    if cfg.dtype not in DTYPES:
        raise ValueError(f"--dtype takes {sorted(DTYPES)}, got {cfg.dtype!r}")
    sampling = _sampling(cfg)
    steps = int(sampling["max_new_tokens"])
    if cfg.tokens == "completion" and steps < 2:
        raise ValueError(
            f"--tokens completion with --max-new-tokens {steps} writes nothing. The engine runs "
            f"no forward pass on the last token it generates, so n new tokens give n - 1 rows. "
            f"Ask for 2 or more."
        )
    os.makedirs(out_dir, exist_ok=True)
    fingerprint = _prompt_fingerprint(prompts)
    engine_dtype = _engine_dtype(engine, cfg)
    returned = engine_slots(engine)
    # The patch widens to float32 on the host after the copy, so the array this file sees is wider
    # than what the NIC moved. The first reply fixes it, because on a steering server it depends
    # on which slots come back, and an engine that returns every slot says how many there are
    # only then.
    wire_itemsize = None
    manifest = _resume_state(out_dir, cfg, fingerprint, sampling, engine_dtype, log)

    if manifest is None:
        manifest = {
            "model": cfg.model,
            "layers": list(cfg.layers) if cfg.layers is not None else None,
            "layer_axis": cfg.layer_axis,
            "dtype": cfg.dtype,
            "engine_dtype": engine_dtype,
            "sampling_params": sampling,
            "d_model": None,
            "num_model_layers": None,
            "tokens": 0,
            "tokens_kept": cfg.tokens,
            "prompts_total": len(prompts),
            "prompts_done": 0,
            "prompts_sha256": fingerprint,
            "shard_bytes": cfg.shard_bytes,
            "shards": [],
        }
    _drop_orphans(out_dir, [shard["path"] for shard in manifest["shards"]], log)
    # Where each kept slot sits in a reply. It's known up front unless `--layers all` waits on
    # the first reply for the slot count, and a slot the engine doesn't return stops the run
    # before the first call.
    positions = None
    if manifest["layers"] is not None:
        positions = reply_positions(manifest["layers"], returned)

    done = int(manifest["prompts_done"])
    if done >= len(prompts):
        log(f"every prompt is already captured, {manifest['tokens']} token(s) on disk")
        return manifest
    if cfg.token_budget is not None and manifest["tokens"] >= cfg.token_budget:
        log(f"{manifest['tokens']} token(s) on disk already meet the {cfg.token_budget} budget")
        return manifest

    progress = Progress(log=log, every=log_every)
    progress.shards = len(manifest["shards"])

    # The run's state is one tuple: the manifest as last published, the open shard, and prompts
    # that kept no rows while no shard was open. Each step replaces the whole tuple, so a Ctrl-C
    # leaves the state from before the step or after it, and the `finally` seals that.
    state = (manifest, None, 0)

    def close_shard(state: tuple) -> tuple:
        """Seal the open shard, credit its prompts, publish the manifest, return the new state.

        It builds the new manifest beside the old one and publishes it in one replace. Run again
        on the same state, it publishes the same manifest.
        """
        manifest, writer, pending = state
        if writer is not None and writer.prompts == 0:
            # The shard's first block never landed. Nothing in it counts, so it isn't listed.
            writer.close()
            with contextlib.suppress(FileNotFoundError):
                os.remove(writer.path)
            writer = None
        if writer is None and pending == 0:
            return (manifest, None, 0)
        shards, tokens, credited = list(manifest["shards"]), manifest["tokens"], pending
        record = None
        if writer is not None:
            record = writer.close()
            shards.append(record)
            tokens += record["tokens"]
            credited += writer.prompts
        published = dict(
            manifest, shards=shards, tokens=tokens, prompts_done=manifest["prompts_done"] + credited
        )
        write_manifest(out_dir, published)
        progress.shards = len(shards)
        if record is not None:
            log(f"closed {record['path']}: {record['tokens']} token(s), {record['bytes']} byte(s)")
        return (published, None, 0)

    try:
        for batch_start in range(done, len(prompts), cfg.batch_size):
            batch = list(prompts[batch_start : batch_start + cfg.batch_size])
            log_request(
                call="generate",
                first_prompt_index=batch_start,
                prompt=batch,
                sampling_params=sampling,
                return_hidden_states=True,
            )
            outputs = engine.generate(
                prompt=batch, sampling_params=sampling, return_hidden_states=True
            )
            log_replies(outputs, first_index=batch_start)
            for output in outputs:
                slots, width = reply_shape(output)
                if wire_itemsize is None:
                    wire_itemsize = _wire_itemsize(
                        engine, engine_dtype, range(slots) if returned is None else returned
                    )
                manifest, writer, pending = state
                if manifest["d_model"] is None:
                    model_slots = slots if returned is None else _model_slot_count(engine)
                    manifest = _with_shape(manifest, model_slots, width)
                    state = (manifest, writer, pending)
                    log(
                        f"the engine returns {slots} slot(s) at d_model={width}, "
                        + ("every slot" if returned is None else f"slots {returned}")
                        + f"; keeping {manifest['layers']} of {manifest['num_model_layers']} "
                        f"slot(s); the wire carries {slots * width * wire_itemsize} bytes a "
                        f"token at {wire_itemsize} bytes an element for engine dtype "
                        f"{engine_dtype}"
                    )
                if positions is None:
                    positions = reply_positions(manifest["layers"], returned)
                check_layer_count(manifest, slots, returned)
                # Each chunk gives up its kept slots before the chunks join, so the host copies
                # those alone.
                hidden, prompt_rows = hidden_states_from_output(output, positions=positions)
                check_row_counts(output["meta_info"], hidden.shape[0], prompt_rows)
                wire_bytes = hidden.shape[0] * slots * width * wire_itemsize
                kept = _select_rows(hidden, prompt_rows, cfg.tokens)
                if not manifest["layer_axis"]:
                    kept = kept[:, 0, :]
                disk_bytes = 0
                if kept.shape[0]:
                    if writer is None:
                        name = f"{SHARD_PREFIX}{len(manifest['shards']):05d}.npy"
                        writer = ShardWriter(
                            os.path.join(out_dir, name),
                            kept.shape[1:],
                            DTYPES[cfg.dtype],
                            checksum=cfg.checksum,
                        )
                        state = (manifest, writer, pending)
                    # One step writes the prompt's rows and counts the prompt.
                    disk_bytes = writer.append(kept, prompts=1)
                elif writer is not None:
                    writer.credit(1)
                else:
                    state = (manifest, None, pending + 1)
                progress.update(kept.shape[0], wire_bytes, disk_bytes)
                if writer is not None and writer.nbytes >= cfg.shard_bytes:
                    state = close_shard(state)
                progress.maybe_log()
                budget = cfg.token_budget
                manifest, writer, _ = state
                if budget is not None and manifest["tokens"] + (
                    writer.rows if writer is not None else 0
                ) >= budget:
                    log(f"hit the {budget} token budget")
                    state = close_shard(state)
                    progress.maybe_log(force=True)
                    return state[0]
    finally:
        state = close_shard(state)
        progress.maybe_log(force=True)
    return state[0]


# --- command line ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="destination directory for shards and manifest")
    ap.add_argument("--model-path", help="what the engine loads; required unless --verify")
    ap.add_argument("--prompts", help="prompt file, one per line or .jsonl")
    ap.add_argument("--text-field", default="text", help="field to read from a .jsonl prompt file")
    ap.add_argument(
        "--layers",
        default="all",
        help="20, or 10,20,30, or 0-59, or 0-59:4, or all. Slot i is the stream entering block i",
    )
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument(
        "--shard-bytes",
        type=int,
        default=2 << 30,
        help="close a shard at the first prompt boundary at or above this",
    )
    ap.add_argument(
        "--tokens",
        default="all",
        choices=["all", "prompt", "completion"],
        help="completion gives n - 1 rows for n new tokens, so it needs --max-new-tokens 2 or more",
    )
    ap.add_argument(
        "--layer-axis",
        action="store_true",
        help="keep the layer axis even for a single layer, so shards stay 3-D",
    )
    ap.add_argument("--no-checksum", action="store_true", help="skip the per-shard SHA-256")
    ap.add_argument("--batch-size", type=int, default=8, help="prompts per engine call")
    ap.add_argument(
        "--token-padding",
        type=int,
        default=1024,
        help="the one prefill token bucket to compile; chunked_prefill_size caps it, and a longer "
        "batch prefills over several passes",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1, help="1 captures the prompt alone")
    ap.add_argument("--token-budget", type=int, default=None, help="stop after this many tokens")
    ap.add_argument("--log-every", type=float, default=5.0, help="seconds between progress lines")
    ap.add_argument(
        "--tp-size", type=int, default=1, help="the chips to shard over; 8 on a v5litepod-8"
    )
    ap.add_argument("--engine-dtype", default="bfloat16", help="what the engine serves in")
    ap.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Engine keyword, repeatable; VALUE reads as JSON, True, False, None or a string",
    )
    ap.add_argument(
        "--log-payloads",
        action="store_true",
        help="log each engine request and a summary of each reply to stderr; LOG_PAYLOADS=1 too",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check the shards in --out against the manifest, then exit",
    )
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # progress reaches a redirected log as it happens
        sys.stdout.reconfigure(line_buffering=True)

    if args.verify:
        problems = verify_manifest(args.out, checksum=not args.no_checksum)
        for problem in problems:
            print(problem)
        print("clean" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    if not args.model_path or not args.prompts:
        ap.error("--model-path and --prompts are required unless you pass --verify")
    try:
        extra = parse_engine_args(args.engine_arg)
    except ValueError as exc:
        ap.error(str(exc))
    if payload_log_requested(args.log_payloads):
        enable_payload_log()

    # A non-zero rank of a multi-host slice only starts its engine and blocks in the scheduler, so
    # only rank 0 reads the prompts; the file lives on host 0 alone.
    if int(os.environ.get("SGL_NODE_RANK", "0") or 0) > 0:
        prompts = []
    else:
        prompts = read_prompts(args.prompts, args.text_field)
    print(f"{len(prompts)} prompt(s) from {args.prompts}")
    cfg = CaptureConfig(
        model=args.model_path,
        layers=parse_layers(args.layers),
        dtype=args.dtype,
        shard_bytes=args.shard_bytes,
        tokens=args.tokens,
        layer_axis=True if args.layer_axis else None,
        checksum=not args.no_checksum,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        token_budget=args.token_budget,
        engine_dtype=args.engine_dtype,
    )
    engine = open_engine(
        args.model_path,
        layers=cfg.layers,
        tp_size=args.tp_size,
        dtype=args.engine_dtype,
        batch_size=args.batch_size,
        token_padding=args.token_padding,
        **extra,
    )
    try:
        manifest = capture(engine, prompts, cfg, args.out, log_every=args.log_every)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    print(
        f"wrote {len(manifest['shards'])} shard(s), {manifest['tokens']} token(s) "
        f"to {args.out}"
    )
    if manifest["tokens"] == 0:
        print("the run kept no tokens, so there's nothing to train on")
        return 1
    # The writer hashed each block as it went to disk. This pass checks every shard against the
    # manifest and leaves the SHA-256 out, which would read every byte back; --verify does that.
    problems = verify_manifest(args.out, checksum=False)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s) in {args.out}")
        return 1
    print(verified_line(read_manifest(args.out), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
