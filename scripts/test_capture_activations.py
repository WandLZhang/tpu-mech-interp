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

"""Correctness gate for `capture_activations.py`, on the real patched engine, on CPU.

    python3 scripts/test_capture_activations.py

`cpu_engine.py` builds `sglang-jax` at eb061d8 with the capture and steering patches, writes a
tiny random-weight Qwen3 with `save_pretrained`, and starts the real Engine on it. Every capture
here goes through that engine: its tokenizer, its scheduler, its chunked prefill, and the reply
the 877 patch builds. Nothing stands in for it. The engine needs the packages in
`upstream/models/requirements.txt`, and `cpu_engine.py` says where the tree comes from.

The reference is independent of the capture code. `transformers` runs the same checkpoint in
float32 over each prompt's ids plus the tokens the engine generated, and `reference_stream`
slices its hidden states with plain NumPy. It calls nothing in `capture_activations.py`, so an
agreement says the chunk flattening, the layer slice, the shard boundaries, the `.npy` headers
and the manifest all land where they should. The engine also serves in float32 here, and the two
sides agree to about 1e-6 per row.

Nine checks:

0. The command line in its own process, where the engine spawns its scheduler the way it does
   on a chip and serves in bf16, against the reference and a bf16 `transformers` floor, and the
   `verified` line it ends on. The engine returns the slots `--layers` keeps, and the wire bytes
   a token the capture prints drop from every slot's to one slot's at `--layers 3`.
1. A capture reproduces the reference, per layer.
2. The shard writer round-trips real rows at float32 and float16, cuts back a block a failed
   write tore, and refuses what float16 can't hold.
3. The manifest describes what's on disk.
4. Shards stay under the byte bound, and there's more than one of them.
5. An interrupted run resumes and leaves finished shards alone, wherever the interrupt landed.
   `--layers all` resumes an every-slot capture and nothing narrower.
6. Every branch of `verify_manifest` catches the thing it's there for, and keeps going.
7. `sae.train.activation_stream` reads the output, at both shard ranks.
8. The engine contract: chunked prefill, the token modes, the row-count check, the kept slots
   each chunk gives up before the chunks join, the wire count off a float32 engine, a server with
   the capture flag off, the payload log, the settings the scripts hand the engine, and an engine
   that returns a subset of slots, bare and seen as a steering server, whose wire widens only
   when a returned slot sits past its steering layer.

Each detection control breaks one thing and has to be caught. If a control passes, this file
fails itself, because a check that survives a broken input measures nothing.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import hashlib
import io
import json
import logging
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import types

# The gate runs on CPU, on a TPU host too.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "sae"))

import capture_activations  # noqa: E402
import cpu_engine  # noqa: E402
from capture_activations import (  # noqa: E402
    HEADER_BYTES,
    PAYLOAD_LOG,
    CaptureConfig,
    Progress,
    ShardWriter,
    _npy_header,
    capture,
    check_layer_count,
    check_row_counts,
    engine_settings,
    hidden_states_from_output,
    layer_index,
    manifest_shards,
    parse_engine_args,
    parse_layers,
    read_manifest,
    read_prompts,
    reply_positions,
    reply_shape,
    verified_line,
    verify_manifest,
    write_manifest,
)
from capture_activations import main as cli  # noqa: E402

NUM_LAYERS = cpu_engine.NUM_LAYERS
D_MODEL = cpu_engine.D_MODEL
KEEP = (1, 3, 4)
MAX_NEW_TOKENS = 3
# Three float32 slots of 64 are 768 bytes a row, so a shard closes after a prompt or two.
SHARD_BYTES = 16384
# Per-row relative error allowed between the float32 engine and float32 `transformers`. They
# measure about 1.3e-06 at worst; the shifted-slot control measures above 0.3.
TOL = 1e-4

_rng = random.Random(0)
# One prompt runs past the 64-token prefill pass on its own, and every batch of 8 does too.
LENGTHS = [3, 9, 17, 5, 26, 7, 12, 4, 15, 8, 21, 6, 11, 60, 5, 14]
PROMPTS = [" ".join(_rng.choice(cpu_engine.WORDS) for _ in range(n)) for n in LENGTHS]
MASSIVE_PROMPT = f"the city {cpu_engine.MASSIVE_TEXT} of the world"


# --- the engine, the checkpoint and the payload log -----------------------------------------

# The checkpoint, the engine, token ids and reference rows by sequence, and the captures later
# checks reuse: "golden" from check 1, "flat" and "axis" from check 5.
_state = {"model": None, "engine": None, "ids": {}, "rows": {}}


def model_path():
    """The tiny checkpoint, written once per process."""
    if _state["model"] is None:
        root = tempfile.mkdtemp(prefix="capture-model-")
        atexit.register(shutil.rmtree, root, True)
        _state["model"] = cpu_engine.write_checkpoint(os.path.join(root, "qwen3-tiny"))
    return _state["model"]


def engine():
    """The real engine, with capture on, started once per process."""
    if _state["engine"] is None:
        _state["engine"] = cpu_engine.open_engine(model_path(), batch_size=8, token_padding=64)
    return _state["engine"]


class Payloads(logging.Handler):
    """Keeps the structured records `capture_activations` writes to its payload log."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        payload = getattr(record, "payload", None)
        if payload is not None:
            self.records.append(payload)

    def replies_since(self, start):
        """Prompt text to reply summary, for the engine calls logged after record `start`."""
        out, queue = {}, []
        for record in self.records[start:]:
            if "request" in record and record["request"].get("call") == "generate":
                queue = list(record["request"]["prompt"])
            elif "reply" in record and queue:
                out[queue.pop(0)] = record["reply"]
        return out


PAYLOADS = Payloads()
PAYLOAD_LOG.addHandler(PAYLOADS)
PAYLOAD_LOG.setLevel(logging.DEBUG)
PAYLOAD_LOG.propagate = False
LAST = {"replies": {}}


def run_capture(out_dir, prompts=PROMPTS, engine=None, log=None, **overrides):
    """A capture through the real engine. `LAST["replies"]` keeps what the engine sent back."""
    # No engine_dtype: capture() reads float32 off the CPU engine, and the wire count follows it.
    settings = dict(
        model=model_path(),
        layers=KEEP,
        dtype="float32",
        shard_bytes=SHARD_BYTES,
        batch_size=8,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    settings.update(overrides)
    cfg = CaptureConfig(**settings)
    start = len(PAYLOADS.records)
    try:
        return capture(engine or globals()["engine"](), prompts, cfg, out_dir, log=log or quiet)
    finally:
        LAST["replies"] = PAYLOADS.replies_since(start)


# --- independent reference ------------------------------------------------------------------


def token_ids(prompt):
    if prompt not in _state["ids"]:
        _state["ids"][prompt] = cpu_engine.tokenize(model_path(), [prompt])[0]
    return _state["ids"][prompt]


def reference_rows(prompt, reply):
    """`transformers` float32 rows for one prompt, and its prompt length.

    The rows cover the prompt, then every token the engine generated but the last, which is
    what a capture holds. `output_ids` leaves out a stop token and `completion_tokens` counts
    it, so the count decides how many generated ids go in.
    """
    prompt_ids = token_ids(prompt)
    done = int(reply["meta_info"]["completion_tokens"])
    sequence = tuple(prompt_ids) + tuple(reply["output_ids"][: max(done - 1, 0)])
    if sequence not in _state["rows"]:
        _state["rows"][sequence] = cpu_engine.reference_hidden_states(model_path(), [sequence])[0]
    return _state["rows"][sequence], len(prompt_ids)


def reference_stream(prompts, replies, keep=KEEP, which="all"):
    """The token stream a capture should write, built with `transformers` and NumPy alone."""
    pieces = []
    for prompt in prompts:
        full, n = reference_rows(prompt, replies[prompt])
        if which == "prompt":
            full = full[:n]
        elif which == "completion":
            full = full[n:]
        pieces.append(full)
    return np.concatenate(pieces, axis=0)[:, list(keep), :]


def row_error(got, want):
    """The worst row's L2 distance from its reference row, over the reference row's norm."""
    if got.shape != want.shape:
        return float("inf")
    g = np.asarray(got, np.float64).reshape(-1, got.shape[-1])
    w = np.asarray(want, np.float64).reshape(-1, want.shape[-1])
    norms = np.maximum(np.linalg.norm(w, axis=1), 1e-30)
    return float(np.max(np.linalg.norm(g - w, axis=1) / norms))


# --- reporting ------------------------------------------------------------------------------


def report(failures, ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    return failures + (not ok)


def control(failures, detected, text):
    print(f"      control ({text}) -> {'detected' if detected else 'NOT DETECTED'}", flush=True)
    if not detected:
        print("      FAIL: the control didn't fail, so this check detects nothing.")
        return failures + 1
    return failures


def quiet(_):
    pass


def load_capture(out_dir):
    """Every shard in manifest order, concatenated."""
    return np.concatenate([np.load(p) for p in manifest_shards(out_dir)], axis=0)


def row_counts(array):
    """A multiset of rows, so a loader can't pass by handing one row out many times."""
    return collections.Counter(row.tobytes() for row in np.ascontiguousarray(array))


def flip_one_value(path, row=0):
    """Corrupt a single float in a shard's data region, leaving the file length alone."""
    array = np.load(path, mmap_mode="r")
    offset = HEADER_BYTES + row * int(np.prod(array.shape[1:])) * array.dtype.itemsize
    del array
    with open(path, "r+b") as fp:
        fp.seek(offset)
        original = fp.read(4)
        fp.seek(offset)
        fp.write(bytes(b ^ 0xFF for b in original))
    return original


def rewrite_header_shape(path, shape):
    """Stamp another shape into a shard header, leaving the dtype and the file length alone."""
    array = np.load(path, mmap_mode="r")
    dtype = array.dtype
    del array
    with open(path, "r+b") as fp:
        fp.write(_npy_header(shape, dtype))


def rewrite_header_tokens(path, tokens):
    """Stamp a different token count into a shard header, leaving the file length alone."""
    array = np.load(path, mmap_mode="r")
    row_shape = array.shape[1:]
    del array
    rewrite_header_shape(path, (tokens,) + row_shape)


def file_state(path):
    with open(path, "rb") as fp:
        return os.path.getsize(path), hashlib.sha256(fp.read()).hexdigest()


def amended(manifest, **fields):
    """A copy of a manifest with some fields replaced."""
    out = json.loads(json.dumps(manifest))
    out.update(fields)
    return out


def refuses(fn, *args, **kwargs):
    """True when the call raises ValueError, which is how capture reports a bad input."""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


class InterruptAt:
    """Sends this process a real SIGINT from inside a capture callback, on the Nth call."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.n:
            os.kill(os.getpid(), signal.SIGINT)


# --- checks ---------------------------------------------------------------------------------


def check_command_line(root):
    """0. The CLI in its own process, the engine at bf16, spawning its scheduler as on a chip."""
    print("\n0. The command line, end to end")
    failures = 0
    tree = cpu_engine.stacked_tree()
    prompts = os.path.join(root, "cli_prompts.txt")
    with open(prompts, "w") as fp:
        fp.write("\n".join(PROMPTS) + "\n")
    out = os.path.join(root, "cli_caps")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(tree, "python"), HERE]),
               JAX_PLATFORMS="cpu", LOG_PAYLOADS="1")
    env.pop("XLA_FLAGS", None)

    def command(out_dir, *extra):
        """The capture command line, as the README runs it, with the CPU engine settings."""
        return [
            sys.executable, os.path.join(HERE, "capture_activations.py"),
            "--model-path", model_path(), "--prompts", prompts, "--out", out_dir,
            "--layers", "1,3,4", "--batch-size", "8", "--token-padding", "64",
            "--max-new-tokens", str(MAX_NEW_TOKENS), "--shard-bytes", str(SHARD_BYTES),
            "--engine-dtype", "bfloat16",
            "--engine-arg", "device=cpu", "--engine-arg", "chunked_prefill_size=64",
            "--engine-arg", "max_total_tokens=4096", "--engine-arg", "log_level=error",
            "--engine-arg", "disable_overlap_schedule=true", "--engine-arg", "random_seed=0",
            *extra,
        ]

    run = subprocess.run(command(out), capture_output=True, text=True, env=env, timeout=1800)
    ok = run.returncode == 0
    failures = report(failures, ok, f"the capture command exits {run.returncode}")
    if not ok:
        print("      its stdout:\n" + run.stdout + "\n      its stderr:\n" + run.stderr)
        return failures

    replies = {}
    for line in run.stderr.splitlines():
        marker = " capture_activations.payload reply "
        if marker in line:
            summary = json.loads(line.split(marker, 1)[1])
            replies[PROMPTS[summary["prompt_index"]]] = summary
    failures = report(
        failures,
        len(replies) == len(PROMPTS)
        and all(s["meta_info"]["hidden_states"][0]["shape"][1:] == [len(KEEP), D_MODEL]
                for s in replies.values()),
        f"LOG_PAYLOADS=1 logged {len(replies)} reply summaries, each chunk as a shape and dtype, "
        f"and each reply holds the {len(KEEP)} slots --layers keeps",
    )
    engine_calls = []
    for line in run.stderr.splitlines():
        marker = " capture_activations.payload request "
        if marker in line and '"Engine"' in line:
            engine_calls.append(json.loads(line.split(marker, 1)[1]))
    failures = report(
        failures,
        len(engine_calls) == 1
        and engine_calls[0].get("return_hidden_states_layers") == list(KEEP),
        f"the engine starts with return_hidden_states_layers "
        f"{engine_calls[0].get('return_hidden_states_layers') if engine_calls else None}",
    )
    got = load_capture(out)
    want = reference_stream(PROMPTS, replies)
    shifted = reference_stream(PROMPTS, replies, keep=(2, 4, 5))
    sequences = [
        tuple(token_ids(p))
        + tuple(replies[p]["output_ids"][: max(replies[p]["meta_info"]["completion_tokens"] - 1, 0)])
        for p in PROMPTS
    ]
    floor = np.concatenate(
        cpu_engine.reference_hidden_states(model_path(), sequences, dtype="bfloat16"), axis=0
    )[:, list(KEEP), :]

    def median_error(a, b):
        a = np.asarray(a, np.float64).reshape(-1, D_MODEL)
        b = np.asarray(b, np.float64).reshape(-1, D_MODEL)
        return float(np.median(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))

    if got.shape != want.shape:
        return report(failures, False, f"the capture holds {got.shape}, the reference {want.shape}")
    engine_err, floor_err = median_error(got, want), median_error(floor, want)
    failures = report(
        failures,
        engine_err <= 2.0 * floor_err,
        f"the bf16 capture's median row error {engine_err:.2e} sits within twice a bf16 "
        f"transformers forward's {floor_err:.2e}",
    )
    failures = control(
        failures, median_error(got, shifted) > 10 * floor_err,
        f"the reference one slot late, median error {median_error(got, shifted):.2e}",
    )
    last = json.loads([line for line in run.stdout.splitlines() if line.startswith("{")][-1])
    rows = got.shape[0]
    failures = report(
        failures,
        last["wire_bytes"] == rows * len(KEEP) * D_MODEL * 2
        and last["disk_bytes"] == rows * len(KEEP) * D_MODEL * 4,
        f"the progress line counts {last['wire_bytes']} wire bytes at bf16, {len(KEEP)} slots a "
        f"row, and {last['disk_bytes']} disk bytes at float32 over {rows} rows",
    )
    failures = control(
        failures, last["wire_bytes"] != rows * len(KEEP) * D_MODEL * 4,
        "wire bytes measured on the float32 array the patch widens to on the host",
    )
    failures = control(
        failures, last["wire_bytes"] != rows * NUM_LAYERS * D_MODEL * 2,
        f"wire bytes for all {NUM_LAYERS} slots, what the engine copied before the filter",
    )
    on_disk = read_manifest(out)
    failures = report(
        failures,
        on_disk["layers"] == list(KEEP) and on_disk["num_model_layers"] == NUM_LAYERS,
        f"the manifest records slots {on_disk['layers']} of {on_disk['num_model_layers']}, "
        f"the model's numbers, where the reply held them at positions 0 to {len(KEEP) - 1}",
    )

    # The wire width, as the capture prints it, with one kept slot and with every slot.
    few = os.path.join(root, "cli_few_prompts.txt")
    with open(few, "w") as fp:
        fp.write("\n".join(PROMPTS[:4]) + "\n")
    per_token = {}
    for spec in ("3", "all"):
        width_run = subprocess.run(
            [*command(os.path.join(root, f"cli_width_{spec}"), "--layers", spec),
             "--prompts", few],
            capture_output=True, text=True, env=env, timeout=1800)
        said = [line for line in width_run.stdout.splitlines() if "the wire carries" in line]
        progress = [json.loads(line) for line in width_run.stdout.splitlines()
                    if line.startswith("{")]
        print(f"      --layers {spec}: {said[0] if said else width_run.stdout + width_run.stderr}")
        if width_run.returncode == 0 and said and progress and progress[-1]["tokens"]:
            per_token[spec] = (said[0].split("the wire carries ", 1)[1].split(" ", 1)[0],
                               progress[-1]["wire_bytes"] / progress[-1]["tokens"])
    failures = report(
        failures,
        per_token.get("3") == (str(D_MODEL * 2), D_MODEL * 2)
        and per_token.get("all") == (str(NUM_LAYERS * D_MODEL * 2), NUM_LAYERS * D_MODEL * 2),
        f"wire bytes a token at bf16, printed and counted: --layers 3 {per_token.get('3')}, "
        f"--layers all {per_token.get('all')}",
    )
    verify = subprocess.run([sys.executable, os.path.join(HERE, "capture_activations.py"),
                             "--out", out, "--verify"], capture_output=True, text=True, env=env)
    failures = report(failures, verify.returncode == 0 and verify.stdout.strip() == "clean",
                      f"--verify on it exits {verify.returncode} and says {verify.stdout.strip()!r}")
    on_disk = read_manifest(out)
    verified = [line for line in run.stdout.splitlines() if line.startswith("verified ")]
    failures = report(
        failures,
        verified == [verified_line(on_disk, out)]
        and f"{len(on_disk['shards'])} shard(s)" in verified[0]
        and f"{on_disk['tokens']} token(s)" in verified[0]
        and f"{sum(s['bytes'] for s in on_disk['shards'])} byte(s)" in verified[0]
        and f"[tokens, {len(KEEP)}, {D_MODEL}] float32" in verified[0],
        f"the run ends on a line a reader can check: {verified[0] if verified else 'none'!r}",
    )
    failures = report(failures, capture_activations.ENGINE_LOAD_NOTE in run.stdout,
                      "the run says which of the engine's load messages to expect")
    first_shard = os.path.join(out, on_disk["shards"][0]["path"])
    good_header = open(first_shard, "rb").read(HEADER_BYTES)
    rewrite_header_tokens(first_shard, on_disk["shards"][0]["tokens"] - 1)
    failures = control(
        failures, any("header holds" in p for p in verify_manifest(out, checksum=False)),
        "the end-of-run check, which skips the SHA-256, over a shard header one token short",
    )
    with open(first_shard, "r+b") as fp:
        fp.write(good_header)

    bad = subprocess.run(command(os.path.join(root, "cli_bad"), "--engine-arg", "tp_size=2"),
                         capture_output=True, text=True, env=env, timeout=600)
    failures = control(
        failures, bad.returncode == 2 and "--tp-size" in bad.stderr
        and not os.path.exists(os.path.join(root, "cli_bad")),
        f"--engine-arg tp_size=2 beside --tp-size, which exits {bad.returncode} before the "
        f"engine loads: {bad.stderr.strip().splitlines()[-1] if bad.stderr.strip() else ''!r}",
    )
    return failures


def check_round_trip(root):
    """1. What lands on disk matches the `transformers` reference, layer by layer."""
    print("\n1. Capture against an independent reference")
    failures = 0
    out = os.path.join(root, "caps")
    lines = []
    manifest = run_capture(out, log=lines.append)
    replies = LAST["replies"]
    _state["golden"] = (out, replies, lines)
    failures = report(
        failures,
        sorted(replies) == sorted(PROMPTS)
        and all(replies[p]["meta_info"]["prompt_tokens"] == len(token_ids(p)) for p in PROMPTS),
        "the engine counts the same prompt tokens the checkpoint's tokenizer gives",
    )
    got = load_capture(out)
    want = reference_stream(PROMPTS, replies)
    failures = report(
        failures, got.shape == want.shape,
        f"the capture holds {got.shape}, the reference holds {want.shape}",
    )
    if got.shape != want.shape:
        return failures
    worst = 0.0
    for axis, model_layer in enumerate(KEEP):
        err = row_error(got[:, axis, :], want[:, axis, :])
        worst = max(worst, err)
        failures = report(failures, err <= TOL, f"slot {model_layer}: worst row error {err:.2e}")
    print(f"      worst slot: {worst:.2e} over {got.shape[0]} tokens")

    flip_one_value(manifest_shards(out)[2])
    failures = control(failures, row_error(load_capture(out), want) > TOL,
                       "one float flipped in shard 2")
    flip_one_value(manifest_shards(out)[2])  # the same flip again puts it back
    failures = report(failures, verify_manifest(out) == [], "the flip undone, the tree verifies")
    failures = control(failures, row_error(got, reference_stream(PROMPTS, replies, keep=(0, 2, 5)))
                       > TOL, "the reference built from slots 0, 2 and 5")
    rotated = PROMPTS[1:] + PROMPTS[:1]
    failures = control(failures, row_error(got, reference_stream(rotated, replies)) > TOL,
                       "the reference built from a rotated prompt order")
    print(f"      manifest: {len(manifest['shards'])} shard(s), {manifest['tokens']} token(s)")
    return failures


def check_writer(root):
    """2. The shard writer, fed real captured rows."""
    print("\n2. Shard writer")
    failures = 0
    out, replies, _ = _state["golden"]
    data = load_capture(out)[:37]
    path = os.path.join(root, "writer.npy")
    writer = ShardWriter(path, data.shape[1:], np.float32)
    for start in range(0, data.shape[0], 8):
        writer.append(data[start : start + 8])
    record = writer.close()
    loaded = np.load(path)
    failures = report(failures, loaded.shape == data.shape and np.array_equal(loaded, data),
                      f"np.load returns {loaded.shape}, bit for bit equal")
    failures = report(
        failures, record["tokens"] == 37 and record["bytes"] == os.path.getsize(path),
        f"the record says {record['tokens']} token(s) and {record['bytes']} byte(s), the file is "
        f"{os.path.getsize(path)}",
    )
    failures = report(failures, writer.close() is record, "a second close returns the same record")
    half = os.path.join(root, "half.npy")
    shutil.copyfile(path, half)
    with open(half, "r+b") as fp:
        fp.truncate(os.path.getsize(half) - 400)
    try:
        detected = np.load(half).shape != data.shape
    except Exception:
        detected = True
    failures = control(failures, detected, "the shard truncated by 400 bytes")

    float16 = os.path.join(root, "half_precision.npy")
    writer = ShardWriter(float16, (D_MODEL,), np.float16)
    writer.append(data[:, 0, :])
    writer.close()
    narrow = np.load(float16)
    failures = report(
        failures,
        narrow.dtype == np.float16 and np.array_equal(narrow, data[:, 0, :].astype(np.float16)),
        "a float16 shard holds each float32 row rounded to float16",
    )
    failures = control(failures, not np.array_equal(narrow.astype(np.float32), data[:, 0, :]),
                       "float32 rows read back out of the float16 shard unchanged")

    # A write that fails partway: RLIMIT_FSIZE lets the kernel write up to the cap and then
    # refuses the rest, so the block lands torn. The writer has to cut it back at close.
    torn = os.path.join(root, "torn.npy")
    row_bytes = int(np.prod(data.shape[1:])) * 4
    cap = HEADER_BYTES + 20 * row_bytes + 100
    writer = ShardWriter(torn, data.shape[1:], np.float32)
    raised, before_close = None, 0
    with file_size_cap(cap):
        try:
            for start in range(0, data.shape[0], 8):
                writer.append(data[start : start + 8])
        except OSError as exc:
            raised = exc
            before_close = os.path.getsize(torn)
    record = writer.close()
    kept = np.load(torn)
    failures = report(failures, raised is not None and raised.errno == 27,
                      f"the append past the cap raised {raised!r}")
    failures = report(
        failures,
        record["tokens"] == 16 and os.path.getsize(torn) == record["bytes"]
        and np.array_equal(kept, data[:16])
        and record["sha256"] == hashlib.sha256(np.ascontiguousarray(data[:16]).tobytes()).hexdigest(),
        f"close cut the file to the {record['tokens']} rows that count, and the header and the "
        f"SHA-256 cover them and no more",
    )
    failures = control(failures, before_close > record["bytes"],
                       f"the failed append left {before_close - record['bytes']} byte(s) past the "
                       f"last whole block before close")

    massive = massive_rows()
    wide = os.path.join(root, "massive16.npy")
    writer = ShardWriter(wide, massive.shape[1:], np.float16)
    try:
        writer.append(massive)
        refused = ""
    except ValueError as exc:
        refused = str(exc)
    record = writer.close()
    failures = report(
        failures, "float16" in refused and "--dtype float32" in refused and record["tokens"] == 0,
        f"a float16 shard refuses a real block that reaches {float(np.abs(massive).max()):g}: "
        f"{refused!r}",
    )
    writer = ShardWriter(os.path.join(root, "massive32.npy"), massive.shape[1:], np.float32)
    writer.append(massive)
    writer.close()
    failures = control(
        failures, float(np.abs(np.load(os.path.join(root, "massive32.npy"))).max()) >= 65520,
        "the same block at float32 holds its values past float16's range",
    )
    poisoned = data[:4].copy()
    poisoned[1, 0, 3] = np.nan
    writer = ShardWriter(os.path.join(root, "nan.npy"), poisoned.shape[1:], np.float32)
    failures = control(failures, refuses(writer.append, poisoned),
                       "a block with a NaN in it, which the float32 shard refuses too")
    writer.close()
    return failures


@contextlib.contextmanager
def file_size_cap(limit):
    """RLIMIT_FSIZE at `limit` bytes, with SIGXFSZ ignored so a write past it fails instead."""
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)


def massive_rows():
    """The engine's rows for a prompt whose one token carries a value past float16's range."""
    if "massive" not in _state:
        outputs = engine().generate(
            prompt=[MASSIVE_PROMPT], sampling_params={"temperature": 0, "max_new_tokens": 1},
            return_hidden_states=True,
        )
        hidden, _ = hidden_states_from_output(outputs[0])
        _state["massive"] = hidden[:, list(KEEP), :]
    return _state["massive"]


def check_manifest(root):
    """3. The manifest says what the files hold."""
    print("\n3. Manifest against the files")
    failures = 0
    golden, replies, _ = _state["golden"]
    out = os.path.join(root, "manifest_caps")
    shutil.copytree(golden, out)
    on_disk = read_manifest(out)
    want = reference_stream(PROMPTS, replies)
    failures = report(
        failures,
        on_disk["model"] == model_path()
        and on_disk["layers"] == list(KEEP)
        and on_disk["dtype"] == "float32"
        and on_disk["d_model"] == D_MODEL
        and on_disk["num_model_layers"] == NUM_LAYERS,
        "model, layers, dtype, d_model and layer count all read back",
    )
    failures = report(
        failures,
        on_disk["engine_dtype"] == "float32"
        and on_disk["sampling_params"] == {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0},
        f"the engine dtype and the sampling settings read back: {on_disk['engine_dtype']}, "
        f"{on_disk['sampling_params']}",
    )
    failures = report(
        failures,
        on_disk["tokens"] == want.shape[0]
        and sum(s["tokens"] for s in on_disk["shards"]) == want.shape[0],
        f"the token count is {on_disk['tokens']}, the reference has {want.shape[0]}",
    )
    failures = report(
        failures, on_disk["prompts_done"] == len(PROMPTS) == on_disk["prompts_total"],
        f"{on_disk['prompts_done']} of {on_disk['prompts_total']} prompt(s) recorded done",
    )
    listed = [s["path"] for s in on_disk["shards"]]
    present = sorted(f for f in os.listdir(out) if f.endswith(".npy"))
    failures = report(
        failures, sorted(listed) == present and listed == sorted(listed),
        f"the shard list is every .npy in the directory, in write order ({len(listed)} of them)",
    )
    failures = report(
        failures,
        all(np.load(os.path.join(out, s["path"]), mmap_mode="r").shape
            == (s["tokens"], len(KEEP), D_MODEL) for s in on_disk["shards"]),
        "every shard header matches its manifest entry",
    )
    failures = report(failures, verify_manifest(out) == [], "verify_manifest reports nothing")
    write_manifest(out, amended(on_disk, tokens=on_disk["tokens"] + 1))
    failures = control(failures, verify_manifest(out) != [], "the manifest token count off by one")
    write_manifest(out, on_disk)
    dropped = on_disk["shards"][1]["path"]
    os.remove(os.path.join(out, dropped))
    failures = control(failures, verify_manifest(out) != [], f"{dropped} deleted")
    return failures


def check_bound(root):
    """4. A shard closes at the bound, so a crash costs one shard."""
    print("\n4. Shard size bound")
    failures = 0
    golden, replies, _ = _state["golden"]
    manifest = read_manifest(golden)
    row_bytes = len(KEEP) * D_MODEL * 4
    longest = max(reference_rows(p, replies[p])[0].shape[0] for p in PROMPTS) * row_bytes
    sizes = [s["bytes"] - HEADER_BYTES for s in manifest["shards"]]
    failures = report(failures, len(sizes) > 3,
                      f"the bound split the run into {len(sizes)} shards, so it did something")
    failures = report(
        failures, max(sizes) < SHARD_BYTES + longest,
        f"the largest shard is {max(sizes)} bytes, under the {SHARD_BYTES} bound plus one prompt "
        f"at {longest} bytes",
    )
    failures = report(failures, all(size >= SHARD_BYTES for size in sizes[:-1]),
                      "every shard but the last one reached the bound before closing")

    few = PROMPTS[:8]
    one_each = os.path.join(root, "tiny_bound_caps")
    run_capture(one_each, prompts=few, shard_bytes=1)
    failures = report(
        failures, len(read_manifest(one_each)["shards"]) == len(few),
        "a one-byte bound gives one shard per prompt, since a shard closes at a prompt boundary",
    )
    total = read_manifest(one_each)["tokens"]
    budgeted = os.path.join(root, "budget_caps")
    stopped = run_capture(budgeted, prompts=few, token_budget=total // 3)
    budget_replies = LAST["replies"]
    failures = report(
        failures,
        total // 3 <= stopped["tokens"] < total and stopped["prompts_done"] < len(few),
        f"--token-budget {total // 3} stopped at {stopped['tokens']} of {total} token(s) and "
        f"{stopped['prompts_done']} prompt(s)",
    )
    failures = report(
        failures,
        verify_manifest(budgeted) == []
        and row_error(load_capture(budgeted),
                      reference_stream(few[: stopped["prompts_done"]], budget_replies)) <= TOL,
        "what the budgeted run wrote is a prefix of the reference",
    )
    unbounded = os.path.join(root, "unbounded_caps")
    run_capture(unbounded, prompts=few, shard_bytes=1 << 30)
    failures = report(
        failures, len(read_manifest(unbounded)["shards"]) == 1,
        "the same capture at a 1 GiB bound gives one shard, so the split came from the bound",
    )
    return failures


def check_resume(root):
    """5. An interrupted run resumes, rewrites nothing, and duplicates nothing."""
    print("\n5. Interrupt and resume")
    failures = 0

    # A real Ctrl-C, sent the moment the second shard closes.
    out = os.path.join(root, "resume_caps")
    sent = []

    def log(line):
        if line.startswith("closed shard-00001") and not sent:
            sent.append(line)
            os.kill(os.getpid(), signal.SIGINT)

    interrupted = False
    try:
        run_capture(out, log=log)
    except KeyboardInterrupt:
        interrupted = True
    first = LAST["replies"]
    failures = report(failures, interrupted, "the Ctrl-C reached the caller")
    partial = read_manifest(out)
    before = {s["path"]: file_state(os.path.join(out, s["path"])) for s in partial["shards"]}
    failures = report(
        failures,
        0 < partial["prompts_done"] < len(PROMPTS) and verify_manifest(out) == [],
        f"the interrupt left {partial['prompts_done']} prompt(s) and {len(partial['shards'])} "
        f"verified shard(s)",
    )
    orphan = os.path.join(out, "shard-09999.npy")
    shutil.copyfile(os.path.join(out, partial["shards"][0]["path"]), orphan)
    resumed = run_capture(out)
    replies = dict(first, **LAST["replies"])
    failures = report(failures, not os.path.exists(orphan),
                      "the resume dropped the shard no manifest covered")
    after = {s["path"]: file_state(os.path.join(out, s["path"])) for s in resumed["shards"]}
    failures = report(
        failures, all(after[name] == state for name, state in before.items()),
        f"the {len(before)} shard(s) from the first run are byte for byte unchanged",
    )
    failures = report(failures, resumed["shards"][: len(partial["shards"])] == partial["shards"],
                      "the resumed manifest keeps the old records and appends")
    want = reference_stream(PROMPTS, replies)
    got = load_capture(out)
    failures = report(
        failures, row_error(got, want) <= TOL and verify_manifest(out) == [],
        f"the finished capture matches the reference, {got.shape[0]} of {want.shape[0]} rows",
    )
    again = run_capture(out)
    failures = report(failures, again["shards"] == resumed["shards"],
                      "a third run over a finished capture writes nothing")

    conflicts = {
        "different layers": dict(layers=(0, 1)),
        "every slot, through --layers all": dict(layers=None),
        "a different prompt list": dict(prompts=PROMPTS[:-1] + ["something else"]),
        "a different dtype": dict(dtype="float16"),
        "a different model": dict(model="other/model"),
        "another slice of the tokens": dict(tokens="prompt"),
        "another --max-new-tokens": dict(max_new_tokens=MAX_NEW_TOKENS + 2),
        "another sampling temperature": dict(sampling_params={"temperature": 0.7}),
        "another engine dtype": dict(engine_dtype="bfloat16"),
    }
    for name, overrides in conflicts.items():
        failures = control(failures, refuses(run_capture, out, **overrides),
                           f"a resume asking for {name}")
    legacy = amended(read_manifest(out))
    del legacy["sampling_params"]
    write_manifest(out, legacy)
    failures = control(failures, refuses(run_capture, out),
                       "a resume over a manifest that predates sampling_params")
    write_manifest(out, again)

    # `--layers all` against its own capture: the manifest's list is every slot, so it resumes.
    few = PROMPTS[:4]
    every = os.path.join(root, "resume_every_caps")
    run_capture(every, prompts=few[:2], layers=None)
    first_every = read_manifest(every)
    resumed_every = None
    try:
        resumed_every = run_capture(every, prompts=few[:2], layers=None)
    except ValueError as exc:
        print(f"      the resume raised: {exc}")
    failures = report(
        failures,
        first_every["layers"] == list(range(NUM_LAYERS)) and resumed_every is not None
        and not refuses(run_capture, every, prompts=few[:2], layers=tuple(range(NUM_LAYERS))),
        f"a --layers all capture records {first_every['layers']} and resumes under --layers all "
        f"and under the same list spelled out",
    )
    failures = control(failures, refuses(run_capture, every, prompts=few[:2], layers=(3,)),
                       "a resume asking for slot 3 alone over that every-slot capture")

    # The layer axis on its own. Both captures ask for one layer, so `layers` matches and the
    # only thing left to refuse is the axis.
    flat = os.path.join(root, "resume_flat_caps")
    run_capture(flat, prompts=few, layers=(3,))
    _state["flat"] = flat
    failures = report(failures, read_manifest(flat)["layer_axis"] is False,
                      "a one-layer capture records layer_axis false")
    failures = control(failures, refuses(run_capture, flat, prompts=few, layers=(3,),
                                         layer_axis=True),
                       "a resume forcing the layer axis onto a 2-D capture, layers unchanged")
    kept = os.path.join(root, "resume_axis_caps")
    run_capture(kept, prompts=few, layers=(3,), layer_axis=True)
    _state["axis"] = kept
    failures = control(failures, refuses(run_capture, kept, prompts=few, layers=(3,),
                                         layer_axis=False),
                       "a resume dropping the layer axis from a 3-D capture, layers unchanged")

    # A Ctrl-C after a prompt's rows land and before the loop moves on. The prompt has to count
    # with the shard it sits in, or the resume captures it a second time.
    mid = os.path.join(root, "mid_prompt_caps")
    real_update, hit = Progress.update, InterruptAt(5)

    def interrupting_update(self, *args, **kwargs):
        hit()
        return real_update(self, *args, **kwargs)

    Progress.update = interrupting_update
    try:
        run_capture(mid, shard_bytes=1 << 30)
    except KeyboardInterrupt:
        pass
    finally:
        Progress.update = real_update
    mid_first = LAST["replies"]
    stopped = read_manifest(mid)
    run_capture(mid, shard_bytes=1 << 30)
    replies = dict(mid_first, **LAST["replies"])
    got, want = load_capture(mid), reference_stream(PROMPTS, replies)
    failures = report(
        failures,
        stopped["prompts_done"] == 5 and got.shape == want.shape and row_error(got, want) <= TOL,
        f"a Ctrl-C inside the fifth prompt's progress update left {stopped['prompts_done']} "
        f"prompt(s) done, and the resumed capture holds {got.shape[0]} rows against the "
        f"reference's {want.shape[0]}",
    )

    # A Ctrl-C after a shard is sealed and before the manifest that lists it lands. The seal
    # runs again in the cleanup and publishes the same record once.
    sealed = os.path.join(root, "sealed_caps")
    real_write, knock = capture_activations.write_manifest, InterruptAt(2)

    def interrupting_write(out_dir, payload):
        knock()
        return real_write(out_dir, payload)

    capture_activations.write_manifest = interrupting_write
    try:
        run_capture(sealed, prompts=PROMPTS[:8])
    except KeyboardInterrupt:
        pass
    finally:
        capture_activations.write_manifest = real_write
    sealed_first = LAST["replies"]
    held = read_manifest(sealed)
    names = [s["path"] for s in held["shards"]]
    failures = report(
        failures,
        verify_manifest(sealed) == [] and len(names) == len(set(names)) == 2,
        f"a Ctrl-C between sealing the second shard and publishing it left {names}, each once, "
        f"verified",
    )
    run_capture(sealed, prompts=PROMPTS[:8])
    replies = dict(sealed_first, **LAST["replies"])
    failures = report(
        failures,
        row_error(load_capture(sealed), reference_stream(PROMPTS[:8], replies)) <= TOL,
        "the resume after it matches the reference",
    )

    # A write that fails partway through a prompt, on the real disk path.
    torn = os.path.join(root, "torn_caps")
    first_shard = read_manifest(_state["golden"][0])["shards"][0]["bytes"]
    raised = None
    with file_size_cap(first_shard - 50):
        try:
            run_capture(torn)
        except OSError as exc:
            raised = exc
    torn_first = LAST["replies"]
    problems = verify_manifest(torn)
    failures = report(failures, raised is not None and not problems,
                      f"a capture whose shard write hit RLIMIT_FSIZE raised {raised!r} and left "
                      f"a tree that verifies: {problems}")
    run_capture(torn)
    replies = dict(torn_first, **LAST["replies"])
    failures = report(
        failures,
        verify_manifest(torn) == [] and row_error(load_capture(torn),
                                                 reference_stream(PROMPTS, replies)) <= TOL,
        "the resume after it finishes clean and matches the reference",
    )

    # The engine refuses a prompt past the model's context. The batch before it is kept.
    long_prompt = " ".join(["the"] * (cpu_engine.CONTEXT + 50))
    failed = os.path.join(root, "engine_error_caps")
    error = None
    try:
        run_capture(failed, prompts=PROMPTS[:8] + [long_prompt] + PROMPTS[8:10])
    except ValueError as exc:
        error = exc
    left = read_manifest(failed)
    failures = report(
        failures,
        error is not None and "context length" in str(error)
        and left["prompts_done"] == 8 and verify_manifest(failed) == [],
        f"the engine's error reached the caller ({str(error)[:60]!r}...), and the batch before "
        f"it sits verified, {left['prompts_done']} prompt(s)",
    )
    return failures


def check_verify(root):
    """6. Each branch of `verify_manifest` catches the thing it's there for."""
    print("\n6. verify_manifest, branch by branch")
    failures = 0
    out = os.path.join(root, "corrupt_caps")
    shutil.copytree(_state["golden"][0], out)
    manifest = read_manifest(out)
    clean = verify_manifest(out)
    failures = report(failures, clean == [], f"a fresh capture verifies clean, {clean}")

    def fires(word, text, repair=None, where=None):
        """Run verify, require a problem naming `word`, then put the tree back."""
        nonlocal failures
        tree = where or out
        problems = verify_manifest(tree)
        hit = any(word in problem for problem in problems)
        first = problems[0] if problems else "nothing"
        extra = f" and {len(problems) - 1} more" if len(problems) > 1 else ""
        failures = control(failures, hit, f"{text}, which gives {first!r}{extra}")
        if repair is not None:
            repair()
        left = verify_manifest(tree)
        if left:
            failures = report(failures, False, f"the repair left {left}")
        return problems

    target = manifest["shards"][3]["path"]
    path = os.path.join(out, target)
    good = read_manifest(out)
    original = open(path, "rb").read()

    def restore():
        with open(path, "wb") as fp:
            fp.write(original)

    flip_one_value(path, row=2)
    failures = report(failures, os.path.getsize(path) == manifest["shards"][3]["bytes"],
                      "a flipped float leaves the file length alone, so the size check can't see it")
    fires("sha256", f"one float flipped in {target}", restore)
    flip_one_value(path, row=2)
    failures = report(failures, verify_manifest(out, checksum=False) == [],
                      "the same tree reads clean without the checksum, so the checksum caught it")
    restore()

    with open(path, "r+b") as fp:
        fp.truncate(os.path.getsize(path) - 64)
    fires("bytes on disk", f"{target} truncated by 64 bytes", restore)
    with open(path, "ab") as fp:
        fp.write(b"\0" * 64)
    fires("bytes on disk", f"64 bytes appended to {target}", restore)
    rewrite_header_tokens(path, manifest["shards"][3]["tokens"] - 1)
    fires("header holds", f"the {target} header claiming one token fewer", restore)

    # A header that claims more rows than the file holds, a negative count, or a count past a C
    # long can't be mapped at all. A scalar header leaves no token axis to compare. Each time,
    # verify has to say so and go on to the next shard, where a flipped float waits.
    later = os.path.join(out, manifest["shards"][5]["path"])
    later_original = open(later, "rb").read()
    tokens, row_shape = manifest["shards"][3]["tokens"], (len(KEEP), D_MODEL)
    flip_one_value(later, row=1)
    for shape, word, claim in (
        ((tokens + 1,) + row_shape, "header", "one token more"),
        ((-1,) + row_shape, "header", "-1 tokens"),
        ((2**63,) + row_shape, "header", "2**63 tokens"),
        ((), "scalar", "a scalar shape"),
    ):
        rewrite_header_shape(path, shape)
        try:
            problems, raised = verify_manifest(out), None
        except Exception as exc:
            problems, raised = [], exc
        failures = report(
            failures,
            raised is None and any(target in p and word in p for p in problems)
            and any(manifest["shards"][5]["path"] in p and "sha256" in p for p in problems),
            f"a header claiming {claim} is reported, and the corrupt shard after it too: "
            f"{problems if raised is None else raised!r}",
        )
        if raised is None:
            code, said = verify_cli(out)
            failures = report(failures, code == 1 and word in said,
                              f"main --verify over it exits {code} and says {said!r}")
        restore()
    with open(later, "wb") as fp:
        fp.write(later_original)

    write_manifest(out, amended(good, d_model=D_MODEL + 1))
    fires("wide", "the manifest d_model off by one", lambda: write_manifest(out, good))
    write_manifest(out, amended(good, dtype="float16"))
    fires("dtype", "the manifest dtype changed to float16", lambda: write_manifest(out, good))
    write_manifest(out, amended(good, layers=list(KEEP) + [5]))
    fires("slot(s) on the layer axis", "a fourth layer added to a three-wide capture",
          lambda: write_manifest(out, good))
    write_manifest(out, amended(good, layer_axis=False))
    fires("dimension(s)", "layer_axis flipped to false on a 3-D capture",
          lambda: write_manifest(out, good))
    flat = os.path.join(root, "flat_verify_caps")
    shutil.copytree(_state["flat"], flat)
    flat_good = read_manifest(flat)
    write_manifest(flat, amended(flat_good, layer_axis=True))
    fires("dimension(s)", "layer_axis flipped to true on a 2-D capture",
          lambda: write_manifest(flat, flat_good), where=flat)
    with open(os.path.join(out, "manifest.json"), "w") as fp:
        fp.write('{"shards": [')
    fires("JSON", "a manifest cut off mid-write", lambda: write_manifest(out, good))
    os.remove(path)
    fires("missing", f"{target} deleted", restore)
    return failures


def verify_cli(out, *flags):
    """Run the command line's --verify and hand back its exit code and what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli(["--out", out, "--verify", *flags])
    return code, buffer.getvalue().strip().replace("\n", "; ")


def check_training_stream(root):
    """7. The training loader reads the shards at both ranks."""
    print("\n7. sae.train.activation_stream over the capture")
    failures = 0
    from train import activation_stream, manifest_inputs  # noqa: E402

    out = _state["golden"][0]
    manifest = read_manifest(out)
    axis = layer_index(manifest, 3)
    failures = report(failures, axis == 1, f"slot 3 sits at shard index {axis}")
    paths, width, wired = manifest_inputs(os.path.join(out, "manifest.json"), 3)
    failures = report(
        failures, paths == manifest_shards(out) and width == D_MODEL and wired == axis,
        f"train.py --manifest resolves {len(paths)} shard(s), d_model={width}, layer={wired}",
    )
    refused = sum(refuses(manifest_inputs, out, bad) for bad in (None, 2))
    failures = control(failures, refused == 2,
                       f"--manifest with no layer and with layer 2, which isn't captured, refused "
                       f"{refused} of 2")
    stored = load_capture(out)[:, KEEP.index(3), :]
    wanted = row_counts(stored)
    batches = list(activation_stream(manifest_shards(out), batch_size=8, layer=axis,
                                     shuffle_tokens=64))
    seen = np.concatenate(batches, axis=0)
    served = row_counts(seen)
    extra, dropped = served - wanted, wanted - served
    failures = report(
        failures, seen.shape[1] == D_MODEL and seen.shape[0] >= stored.shape[0] - 8,
        f"the loader yielded {seen.shape[0]} of {stored.shape[0]} token(s) at width {seen.shape[1]}",
    )
    failures = report(
        failures, not extra and sum(dropped.values()) <= 8,
        f"every row it handed out is a stored row of slot 3, counted as a multiset: "
        f"{sum(extra.values())} row(s) it invented or repeated, {sum(dropped.values())} short",
    )
    other = np.concatenate(list(activation_stream(manifest_shards(out), batch_size=8, layer=0,
                                                  shuffle_tokens=64)), axis=0)
    hits = sum((row_counts(other) & wanted).values())
    failures = control(failures, hits == 0, f"the loader pointed at layer index 0, {hits} hits")
    doubled = np.concatenate([seen[:1]] * seen.shape[0], axis=0)
    failures = control(failures, bool(row_counts(doubled) - wanted),
                       "one stored row handed out as the whole stream")

    flat = _state["flat"]
    flat_manifest = read_manifest(flat)
    shard = np.load(manifest_shards(flat)[0], mmap_mode="r")
    failures = report(failures, shard.ndim == 2 and layer_index(flat_manifest, 3) is None,
                      f"a single-layer capture writes 2-D shards, {shard.shape}")
    failures = report(
        failures, manifest_inputs(flat, None)[2] is None and manifest_inputs(flat, 3)[2] is None,
        "--manifest on a 2-D capture hands the loader no layer index, with or without --layer",
    )
    flat_rows = np.concatenate(list(activation_stream(manifest_shards(flat), batch_size=4,
                                                      shuffle_tokens=64)), axis=0)
    failures = report(failures, not (row_counts(flat_rows) - row_counts(load_capture(flat))),
                      "the 2-D shards need no --layer and hand out their own rows")
    failures = report(failures, np.load(manifest_shards(_state["axis"])[0], mmap_mode="r").ndim == 3,
                      "--layer-axis keeps a single-layer capture 3-D")
    failures = report(
        failures,
        parse_layers("all") is None and parse_layers("20") == (20,)
        and parse_layers("10,20,30") == (10, 20, 30)
        and parse_layers("0-5") == (0, 1, 2, 3, 4, 5) and parse_layers("0-9:3") == (0, 3, 6, 9),
        "the layer spec parses every form the help text claims",
    )
    return failures


def check_engine_contract(root):
    """8. Chunked prefill, token modes, the row-count check, the flag off, the payload log."""
    print("\n8. Engine contract and command line")
    failures = 0
    golden, replies, lines = _state["golden"]

    # The long prompt runs past one 64-token prefill pass, so its reply holds two prefill chunks.
    long_prompt = PROMPTS[LENGTHS.index(max(LENGTHS))]
    chunks = replies[long_prompt]["meta_info"]["hidden_states"]
    prefill = [c["shape"] for c in chunks if len(c["shape"]) == 3]
    failures = report(
        failures,
        len(prefill) >= 2 and sum(s[0] for s in prefill) == len(token_ids(long_prompt)),
        f"a {len(token_ids(long_prompt))}-token prompt came back as prefill chunks {prefill}",
    )
    prompt_only = os.path.join(root, "prompt_caps")
    run_capture(prompt_only, tokens="prompt")
    failures = report(
        failures,
        row_error(load_capture(prompt_only),
                  reference_stream(PROMPTS, LAST["replies"], which="prompt")) <= TOL,
        "--tokens prompt, chunked prefill included, matches the reference",
    )
    done = os.path.join(root, "completion_caps")
    run_capture(done, tokens="completion")
    completion_replies = LAST["replies"]
    want_rows = sum(max(int(r["meta_info"]["completion_tokens"]) - 1, 0)
                    for r in completion_replies.values())
    failures = report(
        failures,
        row_error(load_capture(done),
                  reference_stream(PROMPTS, completion_replies, which="completion")) <= TOL
        and read_manifest(done)["tokens"] == want_rows,
        f"--tokens completion keeps {read_manifest(done)['tokens']} decode row(s), one per "
        f"generated token but the last, and they match the reference",
    )
    failures = control(
        failures,
        refuses(run_capture, os.path.join(root, "empty_caps"), tokens="completion",
                max_new_tokens=1),
        "--tokens completion at --max-new-tokens 1, which would write nothing and exit clean",
    )

    # The progress line, from the golden capture.
    rows = read_manifest(golden)["tokens"]
    last = json.loads([line for line in lines if line.startswith("{")][-1])
    failures = report(
        failures,
        last["wire_bytes"] == rows * NUM_LAYERS * D_MODEL * 4
        and last["disk_bytes"] == rows * len(KEEP) * D_MODEL * 4,
        f"the progress line reports {last['wire_bytes']} wire and {last['disk_bytes']} disk "
        f"byte(s) for {rows} rows off a float32 engine, whose dtype the capture config left "
        f"unset",
    )
    real_args = server_args(engine())
    widths = {
        "this float32 engine": capture_activations._wire_itemsize(engine(), real_args.dtype),
        "bf16": capture_activations._wire_itemsize(None, "bfloat16"),
        "float": capture_activations._wire_itemsize(None, "float"),
        "an unknown dtype": capture_activations._wire_itemsize(None, None),
        "bf16 with steering on": capture_activations._wire_itemsize(
            types.SimpleNamespace(server_args=types.SimpleNamespace(enable_steering=True)),
            "bfloat16"),
    }
    failures = report(
        failures,
        widths == {"this float32 engine": 4, "bf16": 2, "float": 4, "an unknown dtype": 2,
                   "bf16 with steering on": 4},
        f"the wire counts bytes per element by what the stream carries: {widths}",
    )
    # A steering server widens the stream after its steering layer, so slot 20 is float32 at
    # --steering-layer 19 and slot 15 isn't. The concat keeps a list of narrow slots narrow.
    steering_19 = types.SimpleNamespace(
        server_args=types.SimpleNamespace(enable_steering=True, steering_layer=19))
    steered = {
        "slot 15": capture_activations._wire_itemsize(steering_19, "bfloat16", [15]),
        "slots 0 to 19": capture_activations._wire_itemsize(steering_19, "bfloat16", range(20)),
        "slot 20": capture_activations._wire_itemsize(steering_19, "bfloat16", [20]),
        "slots 15 and 20": capture_activations._wire_itemsize(steering_19, "bfloat16", [15, 20]),
        "every slot of 30": capture_activations._wire_itemsize(steering_19, "bfloat16", range(30)),
    }
    failures = report(
        failures,
        steered == {"slot 15": 2, "slots 0 to 19": 2, "slot 20": 4, "slots 15 and 20": 4,
                    "every slot of 30": 4},
        f"at --steering-layer 19 the wire widens only for a slot past 19: {steered}",
    )
    # Control: the rule this replaced counted 4 on any steering server, slot 15 among them.
    failures = control(
        failures,
        capture_activations._wire_itemsize(steering_19, "bfloat16") != steered["slot 15"],
        "the width read with no slot list, which counts a narrow slot 15 as float32",
    )

    # The row-count check, on a real reply and on the same reply one row short.
    reply = engine().generate(prompt=[PROMPTS[4]], sampling_params={
        "temperature": 0, "max_new_tokens": MAX_NEW_TOKENS}, return_hidden_states=True)[0]
    hidden, prompt_rows = hidden_states_from_output(reply)
    failures = report(
        failures,
        check_row_counts(reply["meta_info"], hidden.shape[0], prompt_rows) is None
        and check_row_counts({}, 1, 1) is None,
        "the row-count check passes a real reply and skips one with no counts",
    )
    short = dict(reply, meta_info=dict(reply["meta_info"]))
    short["meta_info"]["hidden_states"] = [reply["meta_info"]["hidden_states"][0][:-1]] + list(
        reply["meta_info"]["hidden_states"][1:])
    h, p = hidden_states_from_output(short)
    failures = control(failures, refuses(check_row_counts, short["meta_info"], h.shape[0], p),
                       "the same reply with its last prefill row dropped")
    no_step = dict(reply, meta_info=dict(reply["meta_info"]))
    no_step["meta_info"]["hidden_states"] = list(reply["meta_info"]["hidden_states"][:-1])
    h, p = hidden_states_from_output(no_step)
    failures = control(failures, refuses(check_row_counts, no_step["meta_info"], h.shape[0], p),
                       "the same reply with its last decode row dropped")

    # The two error paths inside the capture loop.
    failures = control(
        failures,
        refuses(run_capture, os.path.join(root, "range_caps"), prompts=PROMPTS[:2],
                layers=(NUM_LAYERS + 1,)),
        f"a capture asking for slot {NUM_LAYERS + 1} from a {NUM_LAYERS}-slot model",
    )
    manifest = read_manifest(golden)
    failures = report(failures, check_layer_count(manifest, reply_shape(reply)[0]) is None,
                      "a real reply's slot count matches the capture's")
    failures = control(failures, refuses(check_layer_count, manifest, hidden.shape[1] - 1),
                       "a reply one slot short of the capture's first")
    failures = report(
        failures,
        check_layer_count(manifest, len(KEEP), returned=KEEP) is None
        and reply_positions((3,), KEEP) == [1] and reply_positions(KEEP, KEEP) == [0, 1, 2]
        and reply_positions((3,), None) == [3],
        "an engine that returns slots 1, 3 and 4 holds slot 3 at position 1, and one that "
        "returns every slot holds it at position 3",
    )
    failures = control(failures, refuses(check_layer_count, manifest, NUM_LAYERS, returned=KEEP),
                       f"a reply of all {NUM_LAYERS} slots from an engine that returns {KEEP}")
    failures = control(failures, refuses(reply_positions, (2,), KEEP),
                       f"slot 2 from an engine that returns {KEEP}")

    # The host copy: each chunk gives up the kept slots before the chunks join.
    kept, kept_rows = hidden_states_from_output(reply, positions=KEEP)
    failures = report(
        failures,
        kept.shape == (hidden.shape[0], len(KEEP), D_MODEL) and kept_rows == prompt_rows
        and np.array_equal(kept, hidden[:, list(KEEP), :])
        and kept.nbytes == hidden.shape[0] * len(KEEP) * D_MODEL * 4,
        f"hidden_states_from_output(positions={KEEP}) returns {kept.shape}, the same rows as the "
        f"whole reply's slots {KEEP}, in {kept.nbytes} bytes where the whole reply takes "
        f"{hidden.nbytes}",
    )
    failures = control(
        failures,
        not np.array_equal(hidden_states_from_output(reply, positions=(0, 2, 5))[0], kept),
        "the same call for slots 0, 2 and 5",
    )
    failures = report(failures, reply_shape(reply) == (NUM_LAYERS, D_MODEL),
                      f"reply_shape reads {reply_shape(reply)} off the first prefill chunk")

    # float16 on a real capture whose one token runs past 65,504. The three prompts before it
    # land first, so the stop leaves them sealed.
    massive = PROMPTS[:3] + [MASSIVE_PROMPT]
    narrow = os.path.join(root, "float16_caps")
    try:
        run_capture(narrow, prompts=massive, dtype="float16")
        refused = ""
    except ValueError as exc:
        refused = str(exc)
    left = read_manifest(narrow)
    on_disk = load_capture(narrow) if left and left["shards"] else np.zeros((0,), np.float16)
    failures = report(
        failures,
        "float16" in refused and "--dtype float32" in refused
        and verify_manifest(narrow) == [] and np.isfinite(on_disk).all(),
        f"a float16 capture stops at the value it can't hold, with nothing non-finite on disk: "
        f"{refused!r}",
    )
    wide = os.path.join(root, "float32_massive_caps")
    run_capture(wide, prompts=massive)
    got = load_capture(wide)
    failures = control(failures, np.isfinite(got).all() and float(np.abs(got).max()) >= 65520,
                       f"the same capture at float32 holds {float(np.abs(got).max()):g}")

    # The payload log, as the golden capture wrote it.
    requests = [r["request"] for r in PAYLOADS.records if "request" in r
                and r["request"].get("call") == "generate"]
    summaries = [r["reply"] for r in PAYLOADS.records if "reply" in r]

    def has_array_values(value):
        if isinstance(value, dict):
            return any(has_array_values(v) for v in value.values())
        if isinstance(value, list):
            return any(isinstance(v, float) for v in value) or any(has_array_values(v)
                                                                    for v in value)
        return False

    failures = report(
        failures,
        requests and all(r["sampling_params"]["max_new_tokens"] >= 1 and r["prompt"]
                         and r["return_hidden_states"] is True for r in requests)
        and any(r["prompt"] == PROMPTS[:8] for r in requests),
        f"the payload log holds {len(requests)} engine call(s), each with its prompts whole, its "
        f"sampling settings and its flags",
    )
    failures = report(
        failures,
        summaries and all(isinstance(c, dict) and set(c) == {"shape", "dtype"}
                          for s in summaries for c in s["meta_info"]["hidden_states"])
        and not any(has_array_values(s) for s in summaries),
        f"its {len(summaries)} reply summaries give each hidden state chunk as a shape and a "
        f"dtype, and carry no activation values",
    )

    # Engine settings. A launch recipe in models/ overrides any of them through --engine-arg.
    settings = engine_settings()
    failures = report(
        failures,
        settings["precompile_bs_paddings"] == [8] and settings["precompile_token_paddings"] == [1024]
        and settings["disable_radix_cache"] is True and settings["log_requests"] is False,
        "the defaults pin one precompile bucket each, keep the radix cache off, and keep the "
        "engine's own request log off",
    )
    sized = engine_settings(batch_size=4, token_padding=2048)
    failures = report(
        failures,
        sized["precompile_bs_paddings"] == [4] and sized["precompile_token_paddings"] == [2048]
        and sized["max_running_requests"] == 4,
        "the buckets follow --batch-size and --token-padding",
    )
    args = server_args(engine())
    failures = report(
        failures,
        args.enable_return_hidden_states is True and args.model_path == model_path()
        and args.disable_radix_cache is True and args.tp_size == 1,
        "the running engine got the capture flag, the model path, the radix cache off and tp 1",
    )
    items = ["attention_backend=dsa_sparse", "recurrent_state_memory_ratio=0.5",
             "mem_fraction_static=0.8", "skip_server_warmup=false",
             "disable_overlap_schedule=False", "quantization=None"]
    recipe = parse_engine_args(items)
    failures = report(
        failures,
        recipe == {"attention_backend": "dsa_sparse", "recurrent_state_memory_ratio": 0.5,
                   "mem_fraction_static": 0.8, "skip_server_warmup": False,
                   "disable_overlap_schedule": False, "quantization": None}
        and all(recipe[key] is value for key, value in (("skip_server_warmup", False),
                                                        ("disable_overlap_schedule", False),
                                                        ("quantization", None)))
        and engine_settings(**recipe)["attention_backend"] == "dsa_sparse",
        f"--engine-arg values go through JSON, read True, False and None the Python way, fall "
        f"back to strings, and override a default: {recipe}",
    )
    literals = capture_activations.PYTHON_LITERALS
    capture_activations.PYTHON_LITERALS = {}
    try:
        unmapped = parse_engine_args(items)
    finally:
        capture_activations.PYTHON_LITERALS = literals
    failures = control(
        failures, unmapped != recipe and unmapped["disable_overlap_schedule"] == "False",
        f"the parser with no Python spellings, which hands the Engine "
        f"{unmapped['disable_overlap_schedule']!r}, a true value",
    )
    for item in ("tp_size=4", "dtype=float32", "batch_size=4", "token_padding=2048",
                 "model_path=x", "enable_return_hidden_states=false",
                 "return_hidden_states_layers=[1]", "recurrent_state_memory_ratio"):
        failures = control(failures, refuses(parse_engine_args, [item]),
                           f"--engine-arg {item}, which would collide or hand the Engine ''")

    # read_prompts, both file shapes.
    plain = os.path.join(root, "prompts.txt")
    with open(plain, "w") as fp:
        fp.write("first prompt\n\nsecond prompt\n")
    jsonl = os.path.join(root, "prompts.jsonl")
    with open(jsonl, "w") as fp:
        for text in ("first prompt", "second prompt"):
            fp.write(json.dumps({"text": text, "id": 1}) + "\n")
    failures = report(
        failures,
        read_prompts(plain) == ["first prompt", "second prompt"]
        and read_prompts(jsonl) == ["first prompt", "second prompt"]
        and read_prompts(jsonl, text_field="id") == [1, 1],
        "read_prompts takes one prompt per line, one JSON object per line, and --text-field",
    )
    blank = os.path.join(root, "blank.txt")
    open(blank, "w").close()
    failures = control(failures, refuses(read_prompts, blank), "a prompt file with no prompts")
    same_bytes = os.path.join(root, "prompts_as_text.txt")
    shutil.copyfile(jsonl, same_bytes)
    failures = control(failures, read_prompts(same_bytes) != read_prompts(jsonl),
                       "the same JSON lines under a .txt name, which come back as raw JSON")

    progress = Progress(log=quiet)
    progress.update(tokens=10, wire_bytes=1000, disk_bytes=10)
    progress.update(tokens=5, wire_bytes=500, disk_bytes=5)
    line = json.loads(progress.line())
    failures = report(
        failures,
        progress.tokens == 15 and progress.wire_bytes == 1500 and progress.disk_bytes == 15
        and line["wire_mb_per_s"] > line["disk_mb_per_s"],
        "Progress keeps tokens, wire bytes and disk bytes in their own counters",
    )

    checked = os.path.join(root, "cli_verify_caps")
    shutil.copytree(golden, checked)
    code, said = verify_cli(checked)
    failures = report(failures, code == 0 and said == "clean",
                      f"main --verify exits {code} and says {said!r}")
    flip_one_value(os.path.join(checked, read_manifest(checked)["shards"][0]["path"]))
    code, said = verify_cli(checked)
    failures = control(failures, code == 1 and "sha256" in said,
                       f"main --verify over one flipped float, which exits {code}: {said!r}")
    code, _ = verify_cli(checked, "--no-checksum")
    failures = report(failures, code == 0,
                      "main --verify --no-checksum passes the same tree, so the checksum failed it")

    # Last, a real engine with the capture flag off. What comes back is the last layer alone.
    shutdown(engine())
    _state["engine"] = None
    plain_engine = cpu_engine.open_engine(model_path(), capture=False, batch_size=8,
                                          token_padding=64)
    try:
        message = ""
        try:
            run_capture(os.path.join(root, "flag_off_caps"), prompts=PROMPTS[:2],
                        engine=plain_engine)
        except ValueError as exc:
            message = str(exc)
        failures = control(failures, "--enable-return-hidden-states" in message,
                           f"a real engine started without the capture flag: {message[:90]!r}...")
    finally:
        shutdown(plain_engine)

    # A real engine that returns slots 1, 3 and 4. A capture of slot 3 reads position 1 of each
    # reply, matches the reference's slot 3, and records slot 3 of the model's six.
    subset_engine = cpu_engine.open_engine(model_path(), batch_size=8, token_padding=64,
                                           layers=KEEP)
    try:
        subset_out = os.path.join(root, "subset_caps")
        subset = run_capture(subset_out, prompts=PROMPTS[:8], engine=subset_engine, layers=(3,))
        subset_replies = LAST["replies"]
        got = load_capture(subset_out)
        want = reference_stream(PROMPTS[:8], subset_replies, keep=(3,))[:, 0, :]
        shapes = sorted({tuple(c["shape"][-2:]) for r in subset_replies.values()
                         for c in r["meta_info"]["hidden_states"]})
        failures = report(
            failures,
            shapes == [(len(KEEP), D_MODEL)] and row_error(got, want) <= TOL
            and subset["layers"] == [3] and subset["num_model_layers"] == NUM_LAYERS,
            f"an engine started with slots {list(KEEP)} replies {shapes}, and a capture of slot 3 "
            f"matches the reference's slot 3 to {row_error(got, want):.1e}, recorded as layers "
            f"{subset['layers']} of {subset['num_model_layers']}",
        )
        others = {slot: row_error(got, reference_stream(PROMPTS[:8], subset_replies,
                                                          keep=(slot,))[:, 0, :])
                  for slot in KEEP if slot != 3}
        failures = control(failures, min(others.values()) > 100 * TOL,
                           f"the reference's slots at the reply's other two positions: "
                           f"{ {slot: f'{err:.1e}' for slot, err in others.items()} }")
        failures = control(
            failures,
            refuses(run_capture, os.path.join(root, "subset_missing_caps"), prompts=PROMPTS[:2],
                    engine=subset_engine, layers=(2,)),
            f"a capture of slot 2 from the engine that returns {list(KEEP)}",
        )
        # capture() reads the wire width off the slots the engine returns. The same engine,
        # labeled bf16 and seen as a steering server, returns slots 1, 3 and 4: all narrow at
        # --steering-layer 4, and slots 3 and 4 wide at --steering-layer 2.
        def wire_per_row(steering_layer, name):
            args = types.SimpleNamespace(**vars(server_args(subset_engine)))
            args.enable_steering, args.steering_layer = True, steering_layer
            view = types.SimpleNamespace(server_args=args, generate=subset_engine.generate,
                                         tokenizer_manager=subset_engine.tokenizer_manager)
            lines = []
            done = run_capture(os.path.join(root, name), prompts=PROMPTS[:2], engine=view,
                               layers=(3,), engine_dtype="bfloat16", log=lines.append)
            last = json.loads([line for line in lines if line.startswith("{")][-1])
            return last["wire_bytes"] / done["tokens"]

        widths = {layer: wire_per_row(layer, f"steer_{layer}_caps") for layer in (4, 2)}
        failures = report(
            failures,
            widths == {4: len(KEEP) * D_MODEL * 2, 2: len(KEEP) * D_MODEL * 4},
            f"a steering server returning slots {list(KEEP)} moves {widths[4]:g} wire bytes a row "
            f"at --steering-layer 4 and {widths[2]:g} at 2",
        )
        # Control: the rule this replaced counted 4 bytes on any steering server, which gives both
        # layers one width.
        failures = control(failures, widths[4] != widths[2],
                           "one width at both steering layers, as 4 bytes on any steering server "
                           "would give")
    finally:
        shutdown(subset_engine)
    return failures


def server_args(engine):
    return engine.server_args


def shutdown(engine):
    stop = getattr(engine, "shutdown", None)
    if callable(stop):
        stop()


def main():
    # A job started in the background of a non-interactive shell starts with SIGINT ignored, and
    # Python then installs no KeyboardInterrupt handler. Check 5 sends real Ctrl-Cs, so put
    # Python's handler back before any check runs.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    root = tempfile.mkdtemp(prefix="capture-test-")
    print(f"activation capture test, scratch under {os.path.basename(root)}", flush=True)
    failures = 0
    try:
        failures += check_command_line(root)
        failures += check_round_trip(root)
        failures += check_writer(root)
        failures += check_manifest(root)
        failures += check_bound(root)
        failures += check_resume(root)
        failures += check_verify(root)
        failures += check_training_stream(root)
        failures += check_engine_contract(root)
    finally:
        if _state["engine"] is not None:
            shutdown(_state["engine"])
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'PASS' if failures == 0 else f'FAIL: {failures} problem(s)'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
