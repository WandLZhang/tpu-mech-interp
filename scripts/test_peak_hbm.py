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

"""Gate for peak_hbm.py. No TPU needed, and none gets faked.

    python3 scripts/test_peak_hbm.py

On a host with no TPU, the real `tpu-info` API finds no chip. That's the failure a run on a slice
hits when nothing holds the TPU or the package is missing, and it has to fail loudly: exit 1, an
`error` in the JSON, no peak of 0, and no `RESULT` line from `--result`. The first check fails
when `tpu-info` isn't installed here, because the zero-chip checks would then never reach its API.

The peak arithmetic runs on HBM figures read on real slices. A v6e host with four chips read
18.45 GiB of 31.25 on chip 0 and 10.40 on the other three, at duty cycles of 100, 97.5, 100 and
100 percent. On a `v5litepod-8` on 2026-09-24, before the layer filter, every chip of Gemma 4
26B-A4B read 10.34 of 15.75 GiB with capture off and 10.69 with capture on. Here those two
readings are two rounds of one run.

Each check carries a negative control. A control that passes fails the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from peak_hbm import NoChips, Outage, Reader, fold, read_usage, result_line, summarize  # noqa: E402

GIB = 1 << 30
failures = 0


def report(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def control(detected, text):
    global failures
    print(f"      control ({text}): {'detected' if detected else 'NOT DETECTED'}")
    if not detected:
        print("      FAIL: the control didn't fail, so this check proves nothing.")
        failures += 1


def reading(used_gib, total_gib, duty_pct):
    """One round in read_usage's form, from figures in GiB, in bytes."""
    return [
        {"device": i, "used_bytes": round(u * GIB), "total_bytes": round(total_gib * GIB),
         "duty_pct": d}
        for i, (u, d) in enumerate(zip(used_gib, duty_pct))
    ]


V6E = reading([18.45, 10.40, 10.40, 10.40], 31.25, [100.0, 97.5, 100.0, 100.0])
# The model page never gave a duty cycle for that run, so these two carry none.
V5E_CAPTURE_OFF = reading([10.34] * 8, 15.75, [None] * 8)
V5E_CAPTURE_ON = reading([10.69] * 8, 15.75, [None] * 8)


print("zero chips, through the real tpu-info API on this host")
try:
    import tpu_info  # noqa: F401
    importable = "imports, so the checks below reach its API"
except ImportError as exc:
    importable = f"doesn't import ({exc}); uv pip install -r scripts/requirements.txt"
report(importable.startswith("imports"), f"tpu-info {importable}")
try:
    rows = read_usage()
    found = f"read {len(rows)} device(s)"
except NoChips as exc:
    found = f"NoChips: {exc}"
report(found.startswith("NoChips"), f"read_usage on a host with no TPU raises: {found}")
try:
    Reader()()
    through = "returned"
except NoChips as exc:
    through = f"NoChips: {exc}"
report(through.startswith("NoChips"), f"the same error comes out of the sampling thread: {through}")

with tempfile.TemporaryDirectory() as root:
    out = os.path.join(root, "hbm_off.json")
    run = subprocess.run(
        [sys.executable, os.path.join(HERE, "peak_hbm.py"), "--seconds", "3", "--every", "0.5",
         "--out", out],
        capture_output=True, text=True, timeout=120,
    )
    saved = json.load(open(out)) if os.path.exists(out) else {}
    report(
        run.returncode == 1 and "peak_hbm:" in run.stderr and saved.get("chips") == 0
        and saved.get("error") and saved.get("peak_used_gib_max") is None,
        f"a sampling run that reads no chip exits {run.returncode}, says so on stderr, and writes "
        f"chips 0 with an error and no peak: {saved.get('error')!r}",
    )
    stage = subprocess.run(
        [sys.executable, os.path.join(HERE, "peak_hbm.py"), "--result", out, "--stage", "off"],
        capture_output=True, text=True, timeout=120,
    )
    control(
        stage.returncode == 1 and "RESULT" not in stage.stdout and "no HBM reading" in stage.stderr,
        f"--result on that file, which exits {stage.returncode} with no RESULT line: "
        f"{stage.stderr.strip()!r}",
    )
    missing = subprocess.run(
        [sys.executable, os.path.join(HERE, "peak_hbm.py"), "--result",
         os.path.join(root, "never_written.json"), "--stage", "on"],
        capture_output=True, text=True, timeout=120,
    )
    control(missing.returncode == 1 and "RESULT" not in missing.stdout,
            f"--result on a file the sampler never wrote, exit {missing.returncode}")

    print("peak arithmetic on readings from real slices")
    summary = summarize(fold({}, V6E), samples=1, window_s=2.0, every_s=2.0)
    chips = summary["peak_per_chip"]
    report(summary["chips"] == 4 and "error" not in summary,
           f"the v6e round gives four chips and no error: {sorted(chips)}")
    report(chips["0"]["used_gib"] == 18.45 and chips["0"]["total_gib"] == 31.25,
           f"chip 0 reads 18.45 of 31.25: {chips['0']['used_gib']} of {chips['0']['total_gib']}")
    report(chips["1"]["duty_pct"] == 97.5, f"duty cycle stays per chip: {chips['1']['duty_pct']}")
    report(summary["peak_used_gib_max"] == 18.45, "the peak is the max across chips")

    peaks = fold({}, V5E_CAPTURE_OFF)
    peaks = fold(peaks, [])  # a round the metrics server didn't answer
    peaks = fold(peaks, V5E_CAPTURE_ON)
    later = summarize(peaks, samples=2, window_s=6.0, every_s=2.0)
    report(later["chips"] == 8 and later["peak_used_gib_max"] == 10.69
           and all(v["used_gib"] == 10.69 for v in later["peak_per_chip"].values()),
           f"over the v5e rounds, capture off then capture on, every chip peaks at 10.69: "
           f"{later['peak_used_gib_max']}")
    backwards = summarize(fold(fold({}, V5E_CAPTURE_ON), V5E_CAPTURE_OFF), samples=2,
                          window_s=6.0, every_s=2.0)
    control(backwards["peak_used_gib_max"] == 10.69,
            "the same rounds in the other order, where keeping the last reading would give 10.34")
    control(summarize(fold({}, []), samples=0, window_s=6.0, every_s=2.0).get("error") is not None,
            "a run whose rounds all came back empty carries an error, not a peak")

    good = os.path.join(root, "hbm_on.json")
    with open(good, "w") as fh:
        json.dump(later, fh)
    line = result_line(good, "on")
    report(line.startswith("RESULT ") and json.loads(line[7:])["peak_used_gib_max"] == 10.69
           and json.loads(line[7:])["total_gib"] == 15.75,
           f"--result prints the RESULT line for a run that read chips: {line}")

    print("what the sampler says while libtpu's metrics server is down")
    outage = Outage()
    # A v5litepod-8 run of Gemma 4 26B-A4B: 16 empty rounds while the engine loaded, then chips.
    said = [outage.down("StatusCode.UNAVAILABLE") for _ in range(16)] + [outage.up(), outage.up()]
    said += [outage.down("StatusCode.UNAVAILABLE"), outage.up()]
    notes = [s for s in said if s]
    report(
        len(notes) == 4 and "holds the TPU" in notes[0] and "StatusCode.UNAVAILABLE" in notes[0]
        and notes[1] == "libtpu metrics came back after 16 empty round(s)"
        and "holds the TPU" in notes[2] and notes[3].endswith("after 1 empty round(s)"),
        f"16 empty rounds print the reason once and the count when the chips come back, and a "
        f"later outage explains itself again: {len(notes)} lines for {len(said)} rounds",
    )
    report(sum(bool(outage.down("StatusCode.UNAVAILABLE")) for _ in range(16)) == 1,
           "sixteen more empty rounds print one line, where the old sampler printed sixteen")

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
