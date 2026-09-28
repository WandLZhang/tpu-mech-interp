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

"""CPU gate for `check_capture.py`, on the real patched engine and a real `transformers` forward.

    python3 scripts/test_check_capture.py

Runs `check_capture.py` end to end on `cpu_engine.py`'s tiny Qwen3: the engine captures in bf16,
and `transformers` runs the same ids in float32 and in bf16. One prompt carries a token whose row
runs tens of thousands of times the median norm, the way a sink token does, and the check prints
the ratio. The run saves every array with `--save-npz`, and the checks on `compare_layers` read
those arrays and derive each control from them. Among them: the joined prompt with its last quarter
filed under another prompt's rows, which the median passes and the diverged count has to fail, and
a capture that keeps only the sink row, which a Pearson over the whole block passes and the
per-token `--no-floor` gate has to fail. The engine starts without `--return-hidden-states-layers`
and every reply holds every slot. Five runs stop before any forward: a prompt file one prompt
short of the joined prompt, a joined prompt that fits one prefill pass, and `tp_size`,
`chunked_prefill_size` or `return_hidden_states_layers` passed through `--engine-arg`.

The same run writes its capture with `--save-capture`, and `--capture-npz` gates it again against
the saved reference with no engine, to the same RESULT lines. Its controls: the capture with prompt
0's slots one layer off has to fail, and a capture holding other token ids is refused.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpu_engine  # noqa: E402
from check_capture import compare_layers, diverged_slack  # noqa: E402

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


_rng = random.Random(1)
PLAIN = " ".join(_rng.choice(cpu_engine.WORDS) for _ in range(12))
SINK = f"the {cpu_engine.MASSIVE_TEXT} " + " ".join(_rng.choice(cpu_engine.WORDS) for _ in range(10))
# Three more lines, which check_capture joins into one prompt past a 64-token prefill pass.
JOINED = [" ".join(_rng.choice(cpu_engine.WORDS) for _ in range(n)) for n in (20, 25, 18)]


def command(model, prompts, *extra, chunk="64"):
    return [
        sys.executable, os.path.join(HERE, "check_capture.py"), "--model-path", model,
        "--tp-size", "1", "--batch-size", "8", "--token-padding", "64",
        "--prompts-file", prompts, "--num-prompts", "2", "--chunked-prefill-size", chunk,
        "--engine-arg", "device=cpu",
        "--engine-arg", "max_total_tokens=4096", "--engine-arg", "log_level=error",
        "--engine-arg", "disable_overlap_schedule=true", "--engine-arg", "random_seed=0",
        *extra,
    ]


root = tempfile.mkdtemp(prefix="check-capture-test-")
try:
    print("check_capture.py end to end, engine at bf16", flush=True)
    tree = cpu_engine.stacked_tree()
    model = cpu_engine.write_checkpoint(os.path.join(root, "qwen3-tiny"))
    prompts = os.path.join(root, "prompts.txt")
    with open(prompts, "w") as fp:
        fp.write("\n".join([PLAIN, SINK] + JOINED) + "\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(tree, "python"), HERE]),
               JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)
    saved = os.path.join(root, "arrays")  # no suffix, so np.savez adds one
    capture_file = os.path.join(root, "capture.npz")
    run = subprocess.run(command(model, prompts, "--save-npz", saved, "--log-payloads",
                                 "--save-capture", capture_file),
                         capture_output=True, text=True, env=env, timeout=1800)
    ok = run.returncode == 0 and "All checks passed." in run.stdout
    report(ok, f"check_capture.py exits {run.returncode} and passes every layer of every prompt")
    if not ok:
        print("      its stdout:\n" + run.stdout + "\n      its stderr:\n" + run.stderr)
        raise SystemExit(1)
    results = [json.loads(line[len("RESULT "):]) for line in run.stdout.splitlines()
               if line.startswith("RESULT ")]
    report(
        len(results) == 3 and all(r["passed"] and r["control_detected"] and r["gate"] == "floor"
                                  for r in results),
        f"three RESULT lines, each on the bf16 floor gate: worst ratios "
        f"{[round(r['worst_ratio'], 2) for r in results]}, control ratios "
        f"{[round(r['control_ratio'], 1) for r in results]}",
    )
    printed = [line[len("wrote "):] for line in run.stdout.splitlines() if line.startswith("wrote ")]
    report(printed == [saved + ".npz"], f"--save-npz {os.path.basename(saved)} prints {printed}")
    control(bool(printed) and os.path.exists(printed[0]),
            "the path it printed names a file that exists")
    requests = [line for line in run.stderr.splitlines() if " capture_activations.payload request " in line
                and '"input_ids"' in line]
    replies = [json.loads(line.split(" capture_activations.payload reply ", 1)[1])
               for line in run.stderr.splitlines() if " capture_activations.payload reply " in line]
    report(
        len(requests) == 1 and len(replies) == 3
        and all(c.keys() == {"shape", "dtype"} for r in replies
                for c in r["meta_info"]["hidden_states"]),
        f"--log-payloads logged the request with its token ids and {len(replies)} reply summaries "
        f"with shapes in place of arrays",
    )
    joined_chunks = [c["shape"] for c in replies[2]["meta_info"]["hidden_states"]
                     if len(c["shape"]) == 3]
    report(len(joined_chunks) >= 2,
           f"the joined prompt came back over prefill passes {joined_chunks}")
    engine_calls = [json.loads(line.split(" capture_activations.payload request ", 1)[1])
                    for line in run.stderr.splitlines()
                    if " capture_activations.payload request " in line and '"Engine"' in line]
    slot_counts = sorted({c["shape"][1] for r in replies for c in r["meta_info"]["hidden_states"]
                          if len(c["shape"]) == 3})
    report(
        len(engine_calls) == 1 and "return_hidden_states_layers" not in engine_calls[0]
        and slot_counts == [cpu_engine.NUM_LAYERS],
        f"the engine starts without --return-hidden-states-layers, and every prefill chunk holds "
        f"{slot_counts} slots",
    )

    # The saved capture gates again without the engine, against the saved reference.
    regate = subprocess.run(command(model, prompts, "--reference-npz", saved + ".npz",
                                    "--capture-npz", capture_file),
                            capture_output=True, text=True, env=env, timeout=600)
    again = [json.loads(line[len("RESULT "):]) for line in regate.stdout.splitlines()
             if line.startswith("RESULT ")]
    report(
        regate.returncode == 0 and "All checks passed." in regate.stdout
        and "engine capture:" not in regate.stdout and "capture read from" in regate.stdout
        and again == results,
        f"--capture-npz re-gates the saved capture with no engine: exit {regate.returncode}, "
        f"{len(again)} RESULT lines equal to the engine run's",
    )
    if regate.returncode != 0:
        print("      its stdout:\n" + regate.stdout + "\n      its stderr:\n" + regate.stderr)
    held = dict(np.load(capture_file))
    shifted = dict(held)
    # Prompt 0's slots moved up one layer: each slot holds the stream one layer later.
    shifted["capture0"] = np.concatenate([held["capture0"][:, 1:], held["capture0"][:, -1:]], axis=1)
    shifted_file = os.path.join(root, "capture-shifted.npz")
    np.savez(shifted_file, **shifted)
    moved = subprocess.run(command(model, prompts, "--reference-npz", saved + ".npz",
                                   "--capture-npz", shifted_file),
                           capture_output=True, text=True, env=env, timeout=600)
    control(moved.returncode == 1 and "FAILED" in moved.stdout,
            f"a saved capture with prompt 0's slots one layer off fails the re-gate, exit "
            f"{moved.returncode}")
    other = dict(held)
    other["ids0"] = held["ids0"][::-1].copy()
    other_file = os.path.join(root, "capture-other-ids.npz")
    np.savez(other_file, **other)
    refused = subprocess.run(command(model, prompts, "--reference-npz", saved + ".npz",
                                     "--capture-npz", other_file),
                             capture_output=True, text=True, env=env, timeout=600)
    control(refused.returncode != 0 and "other token ids" in refused.stderr,
            f"a saved capture whose prompt 0 holds other token ids is refused, exit "
            f"{refused.returncode}")

    bad = subprocess.run(command(model, prompts, "--engine-arg", "tp_size=8"),
                         capture_output=True, text=True, env=env, timeout=600)
    control(bad.returncode == 2 and "--tp-size" in bad.stderr and "reference" not in bad.stdout,
            f"--engine-arg tp_size=8 stops the run before the reference forwards, exit "
            f"{bad.returncode}")
    subset = subprocess.run(
        command(model, prompts, "--engine-arg", "return_hidden_states_layers=[1]"),
        capture_output=True, text=True, env=env, timeout=600)
    control(subset.returncode == 2 and "compares every slot" in subset.stderr
            and "reference" not in subset.stdout,
            f"--engine-arg return_hidden_states_layers=[1], which would leave one slot to "
            f"compare, exits {subset.returncode} before the reference forwards")
    chunk_arg = subprocess.run(
        command(model, prompts, "--engine-arg", "chunked_prefill_size=4096"),
        capture_output=True, text=True, env=env, timeout=600)
    control(chunk_arg.returncode == 2 and "--chunked-prefill-size" in chunk_arg.stderr
            and "reference" not in chunk_arg.stdout,
            f"--engine-arg chunked_prefill_size=4096, which would let the joined prompt fit one "
            f"pass, exits {chunk_arg.returncode} before the reference forwards")
    short = os.path.join(root, "short.txt")
    with open(short, "w") as fp:
        fp.write("\n".join([PLAIN, SINK] + JOINED[:2]) + "\n")
    one_short = subprocess.run(command(model, short), capture_output=True, text=True, env=env,
                               timeout=600)
    control(one_short.returncode != 0 and "--num-prompts + 3 = 5" in one_short.stderr
            and "reference" not in one_short.stdout,
            f"a prompt file one prompt short of the joined prompt exits {one_short.returncode} "
            f"before the reference forwards, where it used to drop the joined prompt and pass")
    fits = subprocess.run(command(model, prompts, chunk="4096"), capture_output=True, text=True,
                          env=env, timeout=600)
    control(fits.returncode != 0 and "fits one 4096-token prefill pass" in fits.stderr
            and "reference" not in fits.stdout,
            f"--chunked-prefill-size 4096, where the joined prompt fits one pass, exits "
            f"{fits.returncode} before the reference forwards")

    arrays = np.load(saved + ".npz")

    def case(n):
        """Prompt n's capture `[seq, layers, d]`, float32 reference and bf16 floor, per layer."""
        capture = arrays[f"capture{n}"].astype(np.float64)
        reference = list(arrays[f"reference{n}"].astype(np.float64))
        floor = list(arrays[f"floor{n}"].astype(np.float64))
        return capture, reference, floor

    captured, reference, floor = case(0)
    seq, layers, _ = captured.shape
    print(f"compare_layers on the saved arrays: {seq} tokens, {layers} slots", flush=True)

    rows, summary = compare_layers(captured, reference, min_pearson=0.999, control_factor=3.0)
    report(summary["layers"] == layers == cpu_engine.NUM_LAYERS,
           f"compares {layers} slots and leaves the final norm out: {summary['layers']}")
    report(summary["passed"],
           f"the real capture passes the --no-floor gate at 0.999: worst median per-token Pearson "
           f"{summary['worst_token_pearson']:.6f}")
    report(summary["control_detected"],
           f"every slot sits far closer to its own layer than the next: "
           f"{summary['control_ratio']:.1f}x at the closest")

    rolled = captured.copy()
    rolled[:, 3, :] = np.roll(captured[:, 3, :], 1, axis=0)
    _, s2 = compare_layers(rolled, reference, 0.999, 3.0)
    control(not s2["passed"] and s2["worst_pearson_layer"] == 3,
            "slot 3's rows moved one token over fails, and at that layer")
    late = captured[:, 1:, :]
    _, s3 = compare_layers(late, reference, 0.999, 3.0)
    control(not s3["passed"], "a capture one slot late fails the aligned check")
    same = np.stack([reference[0]] * layers, axis=1)
    _, s4 = compare_layers(same, [reference[0]] * (layers + 1), 0.999, 3.0)
    control(not s4["control_detected"], "identical layers leave the control blind, and the check says so")

    sink_capture, sink_reference, sink_floor = case(1)
    norms = np.linalg.norm(sink_reference[3], axis=-1)
    _, s10 = compare_layers(sink_capture, sink_reference, 0.999, 3.0, floor=sink_floor)
    report(
        s10["passed"] and s10["control_detected"],
        f"a prompt whose sink row runs {norms.max() / np.median(norms):.0f} times the median norm "
        f"passes and keeps the control: {s10['control_ratio']:.1f}x",
    )
    sink_only = np.zeros_like(sink_capture)
    sink_row = int(np.argmax(norms))
    sink_only[sink_row] = sink_capture[sink_row]
    rows13, s13 = compare_layers(sink_only, sink_reference, 0.999, 3.0)
    report(min(r["pearson"] for r in rows13) >= 0.999,
           f"a capture that keeps the sink row and zeroes the other {sink_capture.shape[0] - 1} "
           f"reads a Pearson over the whole block of at least 0.999 at every layer: worst "
           f"{min(r['pearson'] for r in rows13):.6f}")
    control(not s13["passed"] and len(s13["failed_layers"].get("pearson", [])) == layers,
            f"the --no-floor gate reads one Pearson per token and fails that capture at "
            f"{len(s13['failed_layers'].get('pearson', []))} of {layers} layers")
    try:
        compare_layers(captured[:, :, :-1], reference, 0.999, 3.0)
        control(False, "a width mismatch raises")
    except ValueError:
        control(True, "a width mismatch raises")

    print("floor gate")
    rows_f, s5 = compare_layers(captured, reference, 0.999999, 10.0, floor=floor, floor_factor=2.0)
    report(s5["gate"] == "floor" and all("ratio" in r for r in rows_f),
           "the floor gate reports a ratio per layer")
    report(s5["passed"], f"the real bf16 capture passes the floor gate: worst ratio {s5['worst_ratio']:.2f}")
    report(min(r["pearson"] for r in rows_f) < 0.999999,
           "and it passes where the fixed gate at the same setting would fail")
    noisy = np.asarray(reference[:layers]).transpose(1, 0, 2)
    noisy = noisy + 5.0 * (captured - noisy)
    _, s6 = compare_layers(noisy, reference, 0.999, 3.0, floor=floor, floor_factor=2.0)
    control(not s6["passed"] and s6["worst_ratio"] > 2.0,
            f"the capture's own error times five fails: worst ratio {s6['worst_ratio']:.2f}")
    _, s7 = compare_layers(rolled, reference, 0.999, 3.0, floor=floor, floor_factor=2.0)
    control(not s7["passed"] and s7["worst_ratio_layer"] == 3,
            "slot 3 moved one token over fails the floor gate, at that layer")
    _, s8 = compare_layers(late, reference, 0.999, 3.0, floor=floor, floor_factor=2.0)
    control(not s8["passed"], "a capture one slot late fails the floor gate")

    flip = captured.copy()
    flip[5, 3:, :] = captured[6, 3:, :]  # token 5 carries token 6's rows from slot 3 on
    _, s11 = compare_layers(flip, reference, 0.999, 3.0, floor=floor, floor_factor=2.0)
    report(s11["passed"] and s11["diverged_tokens"] == 1,
           f"one diverged token passes both tests and shows in the count: {s11['diverged_tokens']} "
           f"of the {diverged_slack(seq):g} the slack allows")
    other, joined_reference, joined_floor = case(2)
    n = min(other.shape[0], seq)
    _, s12 = compare_layers(other[:n], [r[:n] for r in reference], 0.999, 3.0,
                            floor=[f[:n] for f in floor], floor_factor=2.0)
    control(not s12["passed"] and "median" in s12["failed_layers"],
            "the joined prompt's rows against this prompt's reference fail the median test")

    # The joined prompt split across prefill passes. A later pass whose rows land under another
    # request, or come back zeroed, leaves the median inside the rows that are right.
    total = other.shape[0]
    quarter = total // 4
    donor = captured[np.arange(quarter) % seq]
    for label, filler in (("another prompt's rows", donor), ("zeros", np.zeros_like(donor))):
        misfiled = other.copy()
        misfiled[total - quarter:] = filler
        _, s14 = compare_layers(misfiled, joined_reference, 0.999, 3.0, floor=joined_floor,
                                floor_factor=2.0)
        report("median" not in s14["failed_layers"],
               f"the joined prompt with its last {quarter} of {total} rows replaced by {label} "
               f"passes the median test at every layer")
        control(not s14["passed"] and len(s14["failed_layers"].get("diverged", [])) == layers,
                f"the diverged count fails that capture at "
                f"{len(s14['failed_layers'].get('diverged', []))} of {layers} layers: "
                f"{s14['diverged_tokens']} diverged against a slack of "
                f"{s14['diverged_slack']:g}")

    exact = np.stack(reference[:layers], axis=1)
    _, s9 = compare_layers(exact, reference, 0.999, 3.0, floor=[r.copy() for r in reference],
                           floor_factor=2.0)
    report(s9["passed"], "a layer both sides reproduce bit for bit passes, with no divide by zero")
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
