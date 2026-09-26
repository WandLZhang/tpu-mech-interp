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

"""CPU gate for `serve_throughput.py`, on the real patched engine.

    python3 scripts/test_serve_throughput.py

Runs the script end to end on `cpu_engine.py`'s tiny Qwen3, the way `measure_model.sh` runs it,
and reads its `RESULT` line and its payload log. Every check carries a control. A control that
passes fails the run.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_engine  # noqa: E402

failures = 0


def report(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    failures += not ok


def control(detected, text):
    global failures
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}", flush=True)
    if not detected:
        print("      FAIL: the control didn't fail, so this check proves nothing.")
        failures += 1


BATCH, WARMUP, TIMED = 4, 1, 2
_rng = random.Random(2)
PROMPTS = [" ".join(_rng.choice(cpu_engine.WORDS) for _ in range(_rng.randint(4, 30)))
           for _ in range((WARMUP + TIMED) * BATCH)]

root = tempfile.mkdtemp(prefix="serve-throughput-test-")
try:
    print("serve_throughput.py end to end", flush=True)
    tree = cpu_engine.stacked_tree()
    model = cpu_engine.write_checkpoint(os.path.join(root, "qwen3-tiny"))
    prompts = os.path.join(root, "prompts.jsonl")
    with open(prompts, "w") as fp:
        for text in PROMPTS:
            fp.write(json.dumps({"text": text}) + "\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(tree, "python"), HERE]),
               JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)

    def command(*extra):
        return [
            sys.executable, os.path.join(HERE, "serve_throughput.py"), "--model-path", model,
            "--prompts", prompts, "--tp-size", "1", "--batch-size", str(BATCH),
            "--token-padding", "64", "--warmup-batches", str(WARMUP), "--batches", str(TIMED),
            "--engine-arg", "device=cpu", "--engine-arg", "chunked_prefill_size=64",
            "--engine-arg", "max_total_tokens=4096", "--engine-arg", "log_level=error",
            "--engine-arg", "disable_overlap_schedule=true", *extra,
        ]

    run = subprocess.run(command("--log-payloads"), capture_output=True, text=True, env=env,
                         timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in run.stdout.splitlines()
               if line.startswith("RESULT ")]
    ok = run.returncode == 0 and len(results) == 1
    report(ok, f"the script exits {run.returncode} with {len(results)} RESULT line(s)")
    if not ok:
        print("      its stdout:\n" + run.stdout + "\n      its stderr:\n" + run.stderr)
        raise SystemExit(1)
    result = results[0]
    timed = PROMPTS[WARMUP * BATCH :]
    want = sum(len(ids) for ids in cpu_engine.tokenize(model, timed))
    report(
        result["stage"] == "capture_off" and result["batches"] == TIMED
        and result["tokens"] == want and result["tokens_per_s"] > 0,
        f"the RESULT line counts {result['tokens']} prompt tokens over {result['batches']} timed "
        f"batches, and the tokenizer gives {want}: {result['window']}",
    )
    control(result["tokens"] != sum(len(ids) for ids in cpu_engine.tokenize(model, PROMPTS)),
            "the warmup batch's tokens, which the count leaves out")

    marker = " capture_activations.payload "
    logged = [line.split(marker, 1)[1] for line in run.stderr.splitlines() if marker in line]
    calls = [json.loads(line[len("request "):]) for line in logged if line.startswith("request ")]
    replies = [json.loads(line[len("reply "):]) for line in logged if line.startswith("reply ")]
    generates = [c for c in calls if c.get("call") == "generate"]
    engine_call = [c for c in calls if c.get("call") == "Engine"]
    report(
        len(generates) == WARMUP + TIMED
        and [p for c in generates for p in c["prompt"]] == PROMPTS
        and all(c["sampling_params"] == {"max_new_tokens": 1, "temperature": 0.0} for c in generates),
        f"--log-payloads logged {len(generates)} engine calls, each with its prompts whole and its "
        f"sampling settings",
    )
    report(
        len(engine_call) == 1 and engine_call[0]["log_requests"] is False
        and "enable_return_hidden_states" not in engine_call[0]
        and "return_hidden_states_layers" not in engine_call[0]
        and len(replies) == len(PROMPTS)
        and all(r["meta_info"]["prompt_tokens"] > 0 for r in replies),
        "it logged the Engine settings, capture off, and one reply summary per prompt",
    )

    for item, flag in (("tp_size=2", "--tp-size"), ("batch_size=2", "--batch-size"),
                       ("token_padding=128", "--token-padding"),
                       ("enable_return_hidden_states=true", "capture off"),
                       ("return_hidden_states_layers=[1]", "capture off")):
        bad = subprocess.run(command("--engine-arg", item), capture_output=True, text=True,
                             env=env, timeout=600)
        control(bad.returncode == 2 and flag in bad.stderr,
                f"--engine-arg {item} beside {flag}, which exits {bad.returncode} before the engine "
                f"loads")
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
