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

"""Check a model's captured residual stream against a HuggingFace float32 forward, every layer.

    python3 scripts/check_capture.py --model-path SNAPSHOT_DIR --tp-size 8

Both sides read the same token ids. The engine captures on the TPU in BF16; `transformers` runs
the same checkpoint on the host CPU in float32 with `output_hidden_states=True`. Slot `i` of the
capture is the stream entering block `i`, which is HuggingFace's `hidden_states[i]`, so the two
line up index for index. The last HuggingFace entry is the final norm's output and has no slot.
The engine starts without `--return-hidden-states-layers`, so each reply holds every slot, and
`--engine-arg return_hidden_states_layers` is refused.

Each layer line prints Pearson correlation beside the readings the gate takes, and the RESULT
line carries the worst max absolute error. A residual stream carries large values, so BF16 against
float32 differs by tens in absolute terms while the vectors stay collinear to many decimal places.

BF16 drifts from float32 with depth, and how far depends on the model. So the gate is relative:
the check also runs `transformers` in BF16, and each layer has to pass two tests against that
forward. A token's error is its row's L2 distance from the reference row, as a fraction of the
reference row's norm. The capture's median per-token error can be at most `--floor-factor` times
the BF16 forward's. The capture's diverged tokens can number at most `--floor-factor` times the
BF16 forward's, plus a slack of 2% of the prompt's tokens. The slack is 3 tokens on a prompt short
enough that 2% is fewer. On Gemma 4 26B-A4B the BF16 forward alone reads Pearson 0.9909 at layer
27, and a fixed 0.999 would fail a sound capture.

A diverged token sits past relative error 0.25. In an MoE a routing near-tie sends a handful of
tokens to other experts, on the BF16 side and on the capture side, a different handful each time:
12 to 14 of 441 tokens on Gemma 4 26B-A4B, one of them at relative error 2.0 by the last layer. One
such token decides a Pearson over the whole prompt, so Pearson gates nothing. The median holds
until close to half a prompt's rows go wrong. A prefill chunk filed under another request's rows,
or zeroed, puts each of its rows past 0.25 at every layer from the embedding on, and the count
catches it where the median doesn't.

`--no-floor` skips the BF16 forward. Each layer then needs a median per-token Pearson of at least
`--min-pearson`, and no more diverged tokens than the slack. Both read one value per token, so a
large-norm row such as Gemma's BOS doesn't decide the layer.

The control compares each slot against the next layer's reference, one token at a time: each row's
error as a fraction of its own norm, then the median over tokens, so a large-norm token such as
Gemma's BOS doesn't decide it. At every layer the capture has to sit `--control-factor` times
closer to its own layer than to the next, or the check can't tell layers apart and proves nothing.
On 441-token Gemma 4 prompts a BF16 forward sits at least 5.3 times closer, layer by layer.

The ids get the tokenizer's BOS token when the tokenizer doesn't add one itself, as Gemma's
doesn't. A Gemma forward without BOS runs off its trained distribution. `--no-bos` sends the ids
as they are.

`--prompts-file prompts.jsonl` sends the first `--num-prompts` prompts to the engine in one call,
plus one prompt joined from the next three, and compares each against its own forward. On
`build_corpus.py`'s 440-token prompts the joined prompt runs 1,323 tokens with BOS, past the
engine's 1,024-token prefill pass, so it splits across passes however the scheduler groups the
requests. Whether the others share a pass depends on when they reach the scheduler, so they may or
may not split. The joined prompt is the one that tests the chunked-prefill path, so the check stops
before any forward when the file holds fewer than `--num-prompts` + 3 prompts, or when the joined
prompt fits one pass of `--chunked-prefill-size` tokens. That size reaches the engine through its
own flag, and `--engine-arg chunked_prefill_size` is refused.

The float32 reference needs host RAM of about 4 bytes per parameter: 103 GB for Gemma 4 26B-A4B.
The BF16 forward loads after the float32 model is freed.

Both HuggingFace forwards run before the engine starts. Importing `sgl_jax` registers its own
config classes with `AutoConfig`, Gemma 4's among them, and `transformers` can't build the model
from those. Engine() re-imports this module in its subprocesses, so everything sits behind
`__main__`. `--engine-arg` gets parsed first, so a bad one stops the run before the forwards.

`--save-npz caps` writes `caps.npz`, since `np.savez` adds the suffix, and prints that name.
`--log-payloads`, or `LOG_PAYLOADS=1`, logs the request the engine gets, token ids and all, and a
summary of each reply with every array reduced to its shape and dtype.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROMPT = "The capital of France is Paris, and the capital of Japan is"

# Keys this script sets itself, and the flag that sets each. `--engine-arg` refuses them.
RESERVED_ENGINE_ARGS = {
    "model_path": "use --model-path",
    "enable_return_hidden_states": "the check always turns it on",
    "return_hidden_states_layers": "the check compares every slot, so the engine returns them all",
    "tp_size": "use --tp-size",
    "batch_size": "use --batch-size",
    "token_padding": "use --token-padding",
    "chunked_prefill_size": "use --chunked-prefill-size, which the joined prompt is checked against",
}

# The engine_settings default, which splits the joined prompt of 440-token corpus prompts.
CHUNKED_PREFILL_SIZE = 1024

# A per-token error below this counts as a match whatever the floor says. It keeps a layer that
# both sides reproduce to float rounding, such as the embedding, from failing on noise.
ERROR_EPSILON = 1e-6

# A token whose error passes this has left its reference. In an MoE that's a routing near-tie that
# resolved to other experts, which BF16 does on both sides at different tokens.
DIVERGED = 0.25

# The diverged tokens a capture may hold beyond `floor_factor` times the BF16 forward's count: this
# share of the prompt's tokens, and never fewer than DIVERGED_MIN_SLACK tokens. The measured
# Gemma 4 26B-A4B runs held at most 7 against the BF16 forward's 2 on a 441-token prompt. A chunk
# filed under the wrong rows diverges on every row it holds.
DIVERGED_SLACK = 0.02
DIVERGED_MIN_SLACK = 3


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def token_pearsons(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Each row's Pearson correlation with its reference row, one value per token.

    A row with no spread reads 1.0 when it equals its reference and 0.0 otherwise, so a zeroed
    row counts as a miss.
    """
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    g = got - got.mean(axis=-1, keepdims=True)
    w = want - want.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1)
    same = np.all(got == want, axis=-1)
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, np.sum(g * w, axis=-1) / safe, np.where(same, 1.0, 0.0))


def token_errors(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Each row's L2 error as a fraction of the reference row's norm, one value per token."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    norms = np.linalg.norm(want, axis=-1)
    norms = np.where(norms > 0, norms, 1.0)
    return np.linalg.norm(got - want, axis=-1) / norms


def token_error(got: np.ndarray, want: np.ndarray) -> float:
    """The median over tokens of `token_errors`: every token counts the same."""
    return float(np.median(token_errors(got, want)))


def diverged(errors: np.ndarray) -> int:
    """Tokens past DIVERGED. A NaN or inf error counts as diverged too."""
    return int(np.sum(~(np.asarray(errors) <= DIVERGED)))


def diverged_slack(tokens: int, share: float = DIVERGED_SLACK) -> float:
    """The diverged tokens a prompt of `tokens` may hold beyond what the floor accounts for."""
    return max(float(DIVERGED_MIN_SLACK), share * tokens)


def compare_layers(
    captured: np.ndarray,
    reference: list,
    min_pearson: float,
    control_factor: float,
    floor: list | None = None,
    floor_factor: float = 2.0,
    slack_share: float = DIVERGED_SLACK,
):
    """Aligned and shifted comparisons of `[seq, layers, d]` against per-layer `[seq, d]` arrays.

    `floor` holds a BF16 forward's per-layer arrays. With it, a layer passes two tests: the
    capture's median per-token error is at most `floor_factor` times the floor's, and its diverged
    tokens number at most `floor_factor` times the floor's plus `diverged_slack`. Without it, a
    layer passes when its median per-token Pearson reaches `min_pearson` and its diverged tokens
    stay within `diverged_slack`. `slack_share` sets that slack's share of the prompt.

    Returns `(rows, summary)`. `rows` holds one dict per layer, with `failed_tests` naming the
    tests it failed. `summary` holds the worst readings, whether the aligned check passed, which
    layers failed which test, and whether the control was detected.
    """
    layers = min(captured.shape[1], len(reference))
    if floor is not None:
        layers = min(layers, len(floor))
    if layers == 0:
        raise ValueError("nothing to compare: the capture or the reference has no layers")
    slack = diverged_slack(captured.shape[0], slack_share)
    rows = []
    for i in range(layers):
        got = captured[:, i, :]
        want = np.asarray(reference[i], np.float64)
        if got.shape != want.shape:
            raise ValueError(f"layer {i}: capture is {got.shape}, reference is {want.shape}")
        errors = token_errors(got, want)
        row = {
            "layer": i,
            "pearson": pearson(got, want),
            "max_abs": float(np.max(np.abs(got - want))),
            "token_error": float(np.median(errors)),
            "diverged": diverged(errors),
        }
        if floor is None:
            row["token_pearson"] = float(np.median(token_pearsons(got, want)))
            row["diverged_allowed"] = slack
            first = ("pearson", row["token_pearson"] >= min_pearson)
        else:
            floor_errors = token_errors(floor[i], want)
            row["floor_pearson"] = pearson(floor[i], want)
            row["floor_token_error"] = float(np.median(floor_errors))
            row["floor_diverged"] = diverged(floor_errors)
            error, floor_error = row["token_error"], row["floor_token_error"]
            row["ratio"] = error / floor_error if floor_error > 0 else (0.0 if error <= 0 else float("inf"))
            row["diverged_allowed"] = floor_factor * row["floor_diverged"] + slack
            first = ("median", error <= floor_factor * floor_error + ERROR_EPSILON)
        tests = (first, ("diverged", row["diverged"] <= row["diverged_allowed"]))
        row["failed_tests"] = [name for name, ok in tests if not ok]
        row["passed"] = not row["failed_tests"]
        rows.append(row)
    ratios = []
    for i in range(layers - 1):
        if i + 1 >= len(reference):
            break
        aligned = rows[i]["token_error"]
        shifted = token_error(captured[:, i, :], reference[i + 1])
        if aligned > 0:
            rows[i]["shift_ratio"] = shifted / aligned
        else:  # an exact match: the layers are apart only if the next one differs at all
            rows[i]["shift_ratio"] = float("inf") if shifted > 0 else 1.0
        ratios.append((rows[i]["shift_ratio"], i))
    worst = min(rows, key=lambda r: r["pearson"])
    worst_abs = max(r["max_abs"] for r in rows)
    control_ratio, control_layer = min(ratios) if ratios else (0.0, None)
    failed = {}
    for r in rows:
        for name in r["failed_tests"]:
            failed.setdefault(name, []).append(r["layer"])
    summary = {
        "layers": layers,
        "gate": "floor" if floor is not None else "min_pearson",
        "worst_pearson": worst["pearson"],
        "worst_pearson_layer": worst["layer"],
        "worst_max_abs": worst_abs,
        "diverged_tokens": max(r["diverged"] for r in rows),
        "diverged_slack": slack,
        "control_ratio": control_ratio,
        "control_ratio_layer": control_layer,
        "passed": all(r["passed"] for r in rows),
        "failed_layers": failed,
        "control_detected": bool(ratios) and control_ratio >= control_factor,
        "control_factor": control_factor,
    }
    if floor is None:
        worst_token = min(rows, key=lambda r: r["token_pearson"])
        summary["min_pearson"] = min_pearson
        summary["worst_token_pearson"] = worst_token["token_pearson"]
        summary["worst_token_pearson_layer"] = worst_token["layer"]
    else:
        worst_ratio = max(rows, key=lambda r: r["ratio"])
        summary["floor_factor"] = floor_factor
        summary["worst_ratio"] = worst_ratio["ratio"]
        summary["worst_ratio_layer"] = worst_ratio["layer"]
        summary["floor_diverged_tokens"] = max(r["floor_diverged"] for r in rows)
    return rows, summary


def engine_capture(args, batch: list, extra: dict) -> list:
    """The engine's prefill capture for each request in one call, `[seq, layers, d]` apiece.

    `extra` holds the parsed `--engine-arg` keywords.
    """
    from capture_activations import hidden_states_from_output, log_replies, log_request, open_engine

    engine = open_engine(
        args.model_path,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        token_padding=args.token_padding,
        chunked_prefill_size=args.chunked_prefill_size,
        **extra,
    )
    try:
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        log_request(call="generate", input_ids=batch, sampling_params=sampling,
                    return_hidden_states=True)
        out = engine.generate(input_ids=batch, sampling_params=sampling, return_hidden_states=True)
        log_replies(out)
        captured = []
        for ids, reply in zip(batch, out):
            array, prompt_rows = hidden_states_from_output(reply)
            if prompt_rows != len(ids):
                raise ValueError(f"a {len(ids)}-token prompt came back with {prompt_rows} prefill rows")
            captured.append(array[:prompt_rows])
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return captured


def reference_forward(model_path: str, batch: list, dtype_name: str) -> list:
    """HuggingFace hidden states on the host CPU: per prompt, one `[seq, d]` float64 array per entry.

    Each prompt runs alone, so no padding or attention mask enters the reference.
    """
    import torch
    import transformers

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.eval()
    states = []
    with torch.no_grad():
        for ids in batch:
            out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
            states.append([h[0].to(torch.float64).numpy() for h in out.hidden_states])
            del out
    del model
    gc.collect()
    return states


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", required=True, help="the local snapshot directory")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompts-file", help="a prompt file; its first --num-prompts go in one engine call")
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--no-bos", action="store_true", help="send the tokenizer's ids with no BOS added")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="the engine's precompile bucket")
    ap.add_argument("--token-padding", type=int, default=1024)
    ap.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=CHUNKED_PREFILL_SIZE,
        help="tokens per prefill pass; the joined --prompts-file prompt has to hold more",
    )
    ap.add_argument("--floor-factor", type=float, default=2.0)
    ap.add_argument("--no-floor", action="store_true", help="skip the BF16 forward; gate on --min-pearson")
    ap.add_argument(
        "--min-pearson",
        type=float,
        default=0.999,
        help="the gate with --no-floor, on each layer's median per-token Pearson",
    )
    ap.add_argument("--control-factor", type=float, default=3.0)
    ap.add_argument("--save-npz", help="write every capture and reference array here, for a closer look")
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
        help="log the engine request and a summary of each reply to stderr; LOG_PAYLOADS=1 too",
    )
    args = ap.parse_args(argv)

    from capture_activations import (
        enable_payload_log,
        parse_engine_args,
        payload_log_requested,
        read_prompts,
    )

    try:
        extra = parse_engine_args(args.engine_arg, reserved=RESERVED_ENGINE_ARGS)
    except ValueError as exc:
        ap.error(str(exc))
    if payload_log_requested(args.log_payloads):
        enable_payload_log()
    save_npz = args.save_npz
    if save_npz and not save_npz.endswith(".npz"):
        save_npz += ".npz"

    from transformers import AutoTokenizer

    if args.prompts_file:
        corpus = read_prompts(args.prompts_file)
        need = args.num_prompts + 3
        if len(corpus) < need:
            raise SystemExit(
                f"{args.prompts_file} holds {len(corpus)} prompt(s). --prompts-file needs "
                f"--num-prompts + 3 = {need}: the last three join into the prompt that has to split "
                f"across prefill passes."
            )
        prompts = corpus[: args.num_prompts]
        prompts.append("\n".join(corpus[args.num_prompts : need]))
    else:
        prompts = [args.prompt]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    bos = tokenizer.bos_token_id
    batch = []
    for text in prompts:
        ids = tokenizer(text)["input_ids"]
        added = not args.no_bos and bos is not None and (not ids or ids[0] != bos)
        batch.append([bos] + ids if added else ids)
        print(f"prompt {len(batch) - 1}: {len(batch[-1])} tokens{', BOS added' if added else ''}")
    if args.prompts_file and len(batch[-1]) <= args.chunked_prefill_size:
        raise SystemExit(
            f"prompt {len(batch) - 1}, joined from three corpus prompts, holds {len(batch[-1])} "
            f"tokens and fits one {args.chunked_prefill_size}-token prefill pass, so nothing "
            f"splits and the check can't reach the chunked-prefill path. Use longer prompts, such "
            f"as build_corpus.py's 440-token ones, or a lower --chunked-prefill-size."
        )
    if len(batch) > args.batch_size:
        raise SystemExit(f"{len(batch)} prompts need --batch-size {len(batch)} or more")

    reference = reference_forward(args.model_path, batch, "float32")
    print(f"float32 reference: {len(reference[0])} entries per prompt")
    floor = None
    if not args.no_floor:
        floor = reference_forward(args.model_path, batch, "bfloat16")
        print("bf16 floor: done")
    captured = engine_capture(args, batch, extra)
    print("engine capture: " + ", ".join(str(c.shape) for c in captured))
    if save_npz:
        arrays = {}
        for n, got in enumerate(captured):
            arrays[f"capture{n}"] = got.astype(np.float32)
            arrays[f"reference{n}"] = np.stack(reference[n]).astype(np.float32)
            if floor:
                arrays[f"floor{n}"] = np.stack(floor[n]).astype(np.float32)
        np.savez(save_npz, **arrays)
        print(f"wrote {save_npz}")

    ok = True
    for n, got in enumerate(captured):
        rows, summary = compare_layers(
            got, reference[n], args.min_pearson, args.control_factor,
            floor[n] if floor else None, args.floor_factor,
        )
        summary["prompt"] = n
        summary["tokens"] = got.shape[0]
        if len(captured) > 1:
            print(f"prompt {n}, {got.shape[0]} tokens")
        for r in rows:
            mark = "PASS" if r["passed"] else "FAIL"
            why = f"  <- fails on {' and '.join(r['failed_tests'])}" if r["failed_tests"] else ""
            if floor:
                print(
                    f"  [{mark}] layer {r['layer']:3d}  token err {r['token_error']:.5f}  "
                    f"bf16 {r['floor_token_error']:.5f}  ratio {r['ratio']:5.2f}  "
                    f"pearson {r['pearson']:.6f} (bf16 {r['floor_pearson']:.6f})  "
                    f"diverged {r['diverged']} (bf16 {r['floor_diverged']}, "
                    f"allowed {r['diverged_allowed']:g}){why}"
                )
            else:
                print(
                    f"  [{mark}] layer {r['layer']:3d}  token pearson {r['token_pearson']:.8f}  "
                    f"pearson {r['pearson']:.8f}  max_abs {r['max_abs']:.4e}  "
                    f"diverged {r['diverged']} (allowed {r['diverged_allowed']:g}){why}"
                )
        print(
            f"      control (each slot against the next layer, per token): closest layer "
            f"{summary['control_ratio_layer']} sits {summary['control_ratio']:.1f}x closer to its own -> "
            f"{'detected' if summary['control_detected'] else 'NOT DETECTED'}"
        )
        for name, failed_at in summary["failed_layers"].items():
            print(f"      {len(failed_at)} layer(s) fail on {name}: {failed_at}")
        print("RESULT " + json.dumps(summary))
        ok = ok and summary["passed"] and summary["control_detected"]
    print("All checks passed." if ok else "FAILED")
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
