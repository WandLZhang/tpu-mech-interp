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

"""Prefill throughput with capture off: the number capture's cost is measured against.

    python3 scripts/serve_throughput.py --model-path SNAPSHOT_DIR --prompts prompts.jsonl \
        --tp-size 8

Builds the engine with the same settings `capture_activations.py` uses, minus the capture flag, so
the two rates compare. Runs two batches and discards them, because first-step timers read 7x to
11x high, then times 30. Prints one `RESULT` line of JSON with the window it read.

`--log-payloads`, or `LOG_PAYLOADS=1`, logs every request the engine gets, prompts and all, and a
summary of each reply: its token counts and the shape and dtype of any array in it.

Engine() re-imports this module in its subprocesses, so everything sits behind `__main__`. Run it
from a file; a heredoc on stdin dies in `runpy`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Keys this script sets itself, and the flag that sets each. `--engine-arg` refuses them.
RESERVED_ENGINE_ARGS = {
    "model_path": "use --model-path",
    "enable_return_hidden_states": "this script measures capture off",
    "return_hidden_states_layers": "this script measures capture off, which returns no slots",
    "tp_size": "use --tp-size",
    "batch_size": "use --batch-size",
    "token_padding": "use --token-padding",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", required=True, help="the local snapshot directory")
    ap.add_argument("--prompts", required=True, help=".jsonl from build_corpus.py")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--token-padding", type=int, default=4096)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--batches", type=int, default=30)
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
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    from capture_activations import (
        ENGINE_LOAD_NOTE,
        enable_payload_log,
        engine_settings,
        log_replies,
        log_request,
        parse_engine_args,
        payload_log_requested,
    )

    try:
        extra = parse_engine_args(args.engine_arg, reserved=RESERVED_ENGINE_ARGS)
    except ValueError as exc:
        ap.error(str(exc))
    if payload_log_requested(args.log_payloads):
        enable_payload_log()

    with open(args.prompts, encoding="utf-8") as fh:
        prompts = [json.loads(line)["text"] for line in fh if line.strip()]
    need = (args.warmup_batches + args.batches) * args.batch_size
    if len(prompts) < need:
        print(f"need {need} prompts, got {len(prompts)}", file=sys.stderr)
        return 1

    from sgl_jax.srt.entrypoints.engine import Engine

    settings = engine_settings(
        batch_size=args.batch_size, token_padding=args.token_padding, tp_size=args.tp_size, **extra
    )
    log_request(call="Engine", model_path=args.model_path, **settings)
    print(ENGINE_LOAD_NOTE, flush=True)
    engine = Engine(model_path=args.model_path, **settings)
    sampling = {"max_new_tokens": 1, "temperature": 0.0}
    b = args.batch_size

    def generate(i):
        batch = prompts[i * b : (i + 1) * b]
        log_request(call="generate", first_prompt_index=i * b, prompt=batch, sampling_params=sampling)
        out = engine.generate(prompt=batch, sampling_params=sampling)
        log_replies(out, first_index=i * b)
        return out

    for i in range(args.warmup_batches):
        generate(i)

    started, tokens, done = time.time(), 0, 0
    for i in range(args.warmup_batches, args.warmup_batches + args.batches):
        out = generate(i)
        tokens += sum(o["meta_info"]["prompt_tokens"] for o in out)
        done += 1
    secs = time.time() - started

    first = args.warmup_batches + 1
    print(
        "RESULT "
        + json.dumps(
            {
                "stage": "capture_off",
                "model": args.model_path,
                "tp_size": args.tp_size,
                "batches": done,
                "tokens": tokens,
                "secs": round(secs, 2),
                "tokens_per_s": round(tokens / secs, 1),
                "window": f"batches {first} to {first + done - 1}, "
                f"first {args.warmup_batches} discarded",
            }
        ),
        flush=True,
    )
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    # Engine subprocesses can keep the interpreter alive after the work is done.
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
