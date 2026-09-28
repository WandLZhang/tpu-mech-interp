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

`--profile-dir DIR` traces one more batch, between the warmup and the timed window, with the JAX
profiler in the engine's scheduler process, which holds the chips. The trace lands under DIR as
`plugins/profile/<run>/<host>.xplane.pb`, which xprof and TensorBoard read. On a multi-host slice
it covers host 0's chips. The timed window leaves the traced batch out.

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
    ap.add_argument(
        "--profile-dir",
        help="trace one more batch after the warmup into this folder with the JAX profiler; the "
        "timed window leaves it out",
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

    # A non-zero rank of a multi-host slice starts its engine and blocks in the scheduler; only rank
    # 0 sends prompts, so only rank 0 reads them (the file lives on host 0 alone).
    peer = int(os.environ.get("SGL_NODE_RANK", "0") or 0) > 0
    prompts = []
    if not peer:
        with open(args.prompts, encoding="utf-8") as fh:
            prompts = [json.loads(line)["text"] for line in fh if line.strip()]
    profiled = 1 if args.profile_dir else 0
    need = (args.warmup_batches + profiled + args.batches) * args.batch_size
    if not peer and len(prompts) < need:
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

    if args.profile_dir:
        # The scheduler process holds the chips, so the trace starts there. The tokenizer manager's
        # start_profile takes the output folder, which Engine.start_profile() doesn't pass, and
        # python_tracer_level=0 keeps Python events out of the trace. Both calls raise
        # RuntimeError when the scheduler refuses.
        profile_dir = os.path.abspath(args.profile_dir)
        os.makedirs(profile_dir, exist_ok=True)
        log_request(call="start_profile", output_dir=profile_dir, python_tracer_level=0)
        engine.loop.run_until_complete(
            engine.tokenizer_manager.start_profile(output_dir=profile_dir, python_tracer_level=0)
        )
        generate(args.warmup_batches)
        log_request(call="stop_profile")
        engine.loop.run_until_complete(engine.tokenizer_manager.stop_profile())
        print(f"PROFILE batch {args.warmup_batches + 1} traced into {profile_dir}", flush=True)

    first_timed = args.warmup_batches + profiled
    started, tokens, done = time.time(), 0, 0
    for i in range(first_timed, first_timed + args.batches):
        out = generate(i)
        tokens += sum(o["meta_info"]["prompt_tokens"] for o in out)
        done += 1
    secs = time.time() - started

    first = first_timed + 1
    traced = f", batch {args.warmup_batches + 1} traced" if profiled else ""
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
                f"first {args.warmup_batches} discarded{traced}",
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
