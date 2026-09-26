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

"""Gate for `measure_model.sh`: its start, steady-window step, rerun and DOWNLOAD line.

    python3 scripts/test_measure_model.py

The step runs as `measure_model.sh` holds it: the test cuts the lines from the script, from the
comment that opens the step to the `free_tpu` after it, and runs them in bash with the script's
own `keep_results`. The input is the progress log of a real capture: Gemma 4 26B-A4B on a
`v5litepod-8` before the layer filter, on 2026-09-25, whose RESULT line read 2,201.9 tokens/s over
19.82 s to 85.49 s. Three cuts of that log have no window to report: a run that ends before four
progress lines, a window in which the token count stands still, and a log whose progress lines
all fail to parse. Each has to print why and set the exit status. A fourth has an engine log line
on the same line as a progress line, and has to give the same RESULT as the clean log. The
steady-window step is the one behind every model page's capture rate.

The script starts in a fresh `HOME`. With fewer than three arguments it has to exit 2 on its usage
line, where it used to die on an unbound variable, and with no `~/.tpu_env` it has to stop before
the fetch. Controls: the script without each of those two blocks.

A rerun has to keep the last run: `keep_last_run`, cut from the script, runs on a results
directory and a shard directory laid out as a run leaves them. Run 2 has to move run 1's files to
`run-<start time>` and its shard to `<capture directory>-<start time>`, and run 3 has to keep run 2
beside run 1. Control: the two lines the script ran before, which delete run 1's results and shard.

The DOWNLOAD line comes from `fetch_weights.py`, which fetches
`hf-internal-testing/tiny-random-gpt2` into a fresh `HF_HOME` three times: cold, warm, and after a killed fetch that took the weights'
blob and left a tagged `.incomplete` partial of it. Each has to count the bytes that run wrote,
and the third has to delete the partial. This part needs network access to the Hugging Face Hub.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "measure_model.sh")

# capture_on.out from that run, the progress bars left out.
REAL_LOG = [
    "400 prompt(s) from /home/admin/results/gemma4-26b-a4b/prompts.jsonl",
    "capture holds 30 slot(s) at d_model=2816; keeping [15]",
    '{"tokens": 441, "shards": 0, "elapsed_s": 9.61, "tokens_per_s": 45.9, "wire_mb_per_s": 7.8, "disk_mb_per_s": 0.5, "wire_bytes": 74511360, "disk_bytes": 4967424, "disk_gib": 0.005}',
    '{"tokens": 11020, "shards": 0, "elapsed_s": 14.74, "tokens_per_s": 747.7, "wire_mb_per_s": 126.3, "disk_mb_per_s": 8.4, "wire_bytes": 1861939200, "disk_bytes": 124129280, "disk_gib": 0.116}',
    '{"tokens": 21600, "shards": 0, "elapsed_s": 19.82, "tokens_per_s": 1090.1, "wire_mb_per_s": 184.2, "disk_mb_per_s": 12.3, "wire_bytes": 3649536000, "disk_bytes": 243302400, "disk_gib": 0.227}',
    '{"tokens": 32182, "shards": 0, "elapsed_s": 25.42, "tokens_per_s": 1266.1, "wire_mb_per_s": 213.9, "disk_mb_per_s": 14.3, "wire_bytes": 5437470720, "disk_bytes": 362498048, "disk_gib": 0.338}',
    '{"tokens": 46290, "shards": 0, "elapsed_s": 31.69, "tokens_per_s": 1460.6, "wire_mb_per_s": 246.8, "disk_mb_per_s": 16.5, "wire_bytes": 7821158400, "disk_bytes": 521410560, "disk_gib": 0.486}',
    '{"tokens": 60399, "shards": 0, "elapsed_s": 37.72, "tokens_per_s": 1601.4, "wire_mb_per_s": 270.6, "disk_mb_per_s": 18.0, "wire_bytes": 10205015040, "disk_bytes": 680334336, "disk_gib": 0.634}',
    '{"tokens": 74505, "shards": 0, "elapsed_s": 43.77, "tokens_per_s": 1702.3, "wire_mb_per_s": 287.6, "disk_mb_per_s": 19.2, "wire_bytes": 12588364800, "disk_bytes": 839224320, "disk_gib": 0.782}',
    '{"tokens": 88616, "shards": 0, "elapsed_s": 50.12, "tokens_per_s": 1767.9, "wire_mb_per_s": 298.7, "disk_mb_per_s": 19.9, "wire_bytes": 14972559360, "disk_bytes": 998170624, "disk_gib": 0.93}',
    '{"tokens": 101836, "shards": 0, "elapsed_s": 55.13, "tokens_per_s": 1847.1, "wire_mb_per_s": 312.1, "disk_mb_per_s": 20.8, "wire_bytes": 17206210560, "disk_bytes": 1147080704, "disk_gib": 1.068}',
    '{"tokens": 113297, "shards": 0, "elapsed_s": 61.47, "tokens_per_s": 1843.1, "wire_mb_per_s": 311.4, "disk_mb_per_s": 20.8, "wire_bytes": 19142661120, "disk_bytes": 1276177408, "disk_gib": 1.189}',
    '{"tokens": 123875, "shards": 0, "elapsed_s": 66.54, "tokens_per_s": 1861.6, "wire_mb_per_s": 314.5, "disk_mb_per_s": 21.0, "wire_bytes": 20929920000, "disk_bytes": 1395328000, "disk_gib": 1.3}',
    '{"tokens": 137986, "shards": 0, "elapsed_s": 72.71, "tokens_per_s": 1897.9, "wire_mb_per_s": 320.7, "disk_mb_per_s": 21.4, "wire_bytes": 23314114560, "disk_bytes": 1554274304, "disk_gib": 1.448}',
    '{"tokens": 152094, "shards": 0, "elapsed_s": 78.91, "tokens_per_s": 1927.4, "wire_mb_per_s": 325.6, "disk_mb_per_s": 21.7, "wire_bytes": 25697802240, "disk_bytes": 1713186816, "disk_gib": 1.596}',
    '{"tokens": 166201, "shards": 0, "elapsed_s": 85.49, "tokens_per_s": 1944.1, "wire_mb_per_s": 328.5, "disk_mb_per_s": 21.9, "wire_bytes": 28081320960, "disk_bytes": 1872088064, "disk_gib": 1.744}',
    "closed shard-00000.npy: 176342 token(s), 1986316416 byte(s)",
    '{"tokens": 176342, "shards": 1, "elapsed_s": 88.67, "tokens_per_s": 1988.7, "wire_mb_per_s": 336.0, "disk_mb_per_s": 22.4, "wire_bytes": 29794744320, "disk_bytes": 1986316288, "disk_gib": 1.85}',
    "wrote 1 shard(s), 176342 token(s) to /dev/shm/caps-gemma4-26b-a4b",
]
# The RESULT line that run wrote to its results.txt.
REAL_RESULT = {"stage": "capture_on_steady", "window": "19.82 s to 85.49 s", "tokens": 144601,
               "secs": 65.67, "tokens_per_s": 2201.9, "wire_mb_per_s": 372.0,
               "wire_bytes_per_token": 168960, "run_tokens": 176342}
# An engine log line, from the same slice, written onto a progress line.
ENGINE_LINE = "[2026-09-25 07:52:10] INFO scheduler: Prefill batch. #new-seq: 8"

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


def step_lines(script: str) -> tuple:
    """`keep_results` and the steady-window step, cut from the script as it stands."""
    lines = open(script, encoding="utf-8").read().splitlines()
    keep = [line for line in lines if line.startswith("keep_results()")]
    start = next(n for n, line in enumerate(lines) if line.startswith("# The steady window"))
    end = next(n for n in range(start, len(lines)) if lines[n].startswith("free_tpu"))
    return keep, lines[start:end]


def run_step(root: str, log: list, name: str) -> tuple:
    """Run the step over one capture log. Returns (RC after it, RESULT lines, whole output)."""
    out = os.path.join(root, name)
    os.makedirs(out)
    with open(os.path.join(out, "capture_on.out"), "w") as fp:
        fp.write("\n".join(log) + "\n")
    keep, step = step_lines(SCRIPT)
    driver = os.path.join(out, "step.sh")
    with open(driver, "w") as fp:
        fp.write("set -uo pipefail\n")
        fp.write(f"OUT={out}\nRC=0\n")
        fp.write('log() { echo "### $*"; }\n')
        fp.write("\n".join(keep + step) + "\n")
        fp.write('echo "RC=$RC"\n')
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    done = subprocess.run(["bash", driver], capture_output=True, text=True, env=env, timeout=120)
    text = done.stdout + done.stderr
    rc = [line for line in done.stdout.splitlines() if line.startswith("RC=")]
    results_file = os.path.join(out, "results.txt")
    results = []
    if os.path.exists(results_file):
        results = [json.loads(line[len("RESULT "):]) for line in open(results_file)
                   if line.startswith("RESULT ")]
    return (int(rc[-1][3:]) if rc else None), results, text


# ---------------------------------------------------------------------------------------------
# A rerun keeps the last run.

def keep_lines(script: str) -> list:
    """`RUN_FILES` and `keep_last_run`, cut from the script as it stands."""
    lines = open(script, encoding="utf-8").read().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("RUN_FILES=("))
    end = next(n for n in range(start, len(lines)) if lines[n] == "}")
    return lines[start:end + 1]


def run_keep(out: str, caps: str, body: list) -> str:
    """Run `body` in bash with OUT and CAPS set. Returns its whole output."""
    driver = os.path.join(os.path.dirname(out), "keep.sh")
    with open(driver, "w") as fp:
        fp.write("set -uo pipefail\n")
        fp.write(f"OUT={out}\nCAPS={caps}\n")
        fp.write('log() { echo "### $*"; }\n')
        fp.write("\n".join(body) + "\n")
    done = subprocess.run(["bash", driver], capture_output=True, text=True, timeout=60)
    return done.stdout + done.stderr


def lay_out_run(out: str, caps: str, tag: str, started: str | None) -> dict:
    """Write one run's files and shard, each holding `tag`. Returns {relative path: text}."""
    os.makedirs(out, exist_ok=True)
    os.makedirs(caps, exist_ok=True)
    files = {name: f"{tag} {name}\n" for name in
             ("results.txt", "fetch.log", "capture_off.out", "capture_on.out", "manifest.json",
              "hbm_on.json")}
    if started:
        files["started"] = started + "\n"
    for name, text in files.items():
        with open(os.path.join(out, name), "w") as fp:
            fp.write(text)
    with open(os.path.join(caps, "shard-00000.npy"), "w") as fp:
        fp.write(f"{tag} shard\n")
    return files


def read(path: str) -> str | None:
    return open(path).read() if os.path.exists(path) else None


def check_keep_last_run(root: str) -> None:
    script = open(SCRIPT, encoding="utf-8").read().splitlines()
    report("keep_last_run" in script and 'rm -f "$OUT/results.txt"' not in script
           and 'rm -rf "$CAPS"' not in script,
           "measure_model.sh calls keep_last_run and no longer deletes results.txt or the shards")
    body = keep_lines(SCRIPT)
    out = os.path.join(root, "keep", "results", "nemotron3-super")
    caps = os.path.join(root, "keep", "caps-nemotron3-super")
    run1 = lay_out_run(out, caps, "run 1", "20260925T083109Z")
    with open(os.path.join(out, "prompts.jsonl"), "w") as fp:
        fp.write("corpus\n")
    text = run_keep(out, caps, body + ["keep_last_run"])
    kept = os.path.join(out, "run-20260925T083109Z")
    moved_ok = all(read(os.path.join(kept, name)) == body_text for name, body_text in run1.items())
    report(moved_ok and read(caps + "-20260925T083109Z/shard-00000.npy") == "run 1 shard\n"
           and not os.path.exists(caps) and not os.path.exists(os.path.join(out, "results.txt"))
           and read(os.path.join(out, "prompts.jsonl")) == "corpus\n"
           and len((read(os.path.join(out, "started")) or "").strip()) == 16,
           f"run 2 keeps run 1: its {len(run1)} files in run-20260925T083109Z, its shard in"
           f" caps-nemotron3-super-20260925T083109Z, the corpus in place, a new start time."
           f" Output: {text.strip()}")

    # Run 2 then writes its own files, and run 3 keeps run 2 beside run 1.
    started2 = read(os.path.join(out, "started")).strip()
    run2 = lay_out_run(out, caps, "run 2", None)
    run2["started"] = started2 + "\n"
    run_keep(out, caps, body + ["keep_last_run"])
    report(read(os.path.join(out, f"run-{started2}", "results.txt")) == "run 2 results.txt\n"
           and read(os.path.join(kept, "results.txt")) == "run 1 results.txt\n"
           and read(f"{caps}-{started2}/shard-00000.npy") == "run 2 shard\n"
           and read(caps + "-20260925T083109Z/shard-00000.npy") == "run 1 shard\n",
           f"run 3 keeps run 2 in run-{started2} and leaves run 1 where it was")

    # A results directory from before `started` existed dates run 1 from its results.
    old = os.path.join(root, "keep-old", "results", "gpt-oss-120b")
    old_caps = os.path.join(root, "keep-old", "caps-gpt-oss-120b")
    lay_out_run(old, old_caps, "old run", None)
    os.utime(os.path.join(old, "results.txt"), (1790000000, 1790000000))
    run_keep(old, old_caps, body + ["keep_last_run"])
    stamp = "20260921T141320Z"
    report(read(os.path.join(old, f"run-{stamp}", "results.txt")) == "old run results.txt\n"
           and read(f"{old_caps}-{stamp}/shard-00000.npy") == "old run shard\n",
           f"a run with no start time keeps its files under its results' time, run-{stamp}")

    # Control: the two lines the script ran before, on a copy of run 1.
    ctl = os.path.join(root, "keep-control", "results", "x")
    ctl_caps = os.path.join(root, "keep-control", "caps-x")
    lay_out_run(ctl, ctl_caps, "run 1", "20260925T083109Z")
    run_keep(ctl, ctl_caps, ['rm -f "$OUT/results.txt"', 'rm -rf "$CAPS"'])
    lost = [p for p in (os.path.join(ctl, "results.txt"), ctl_caps) if not os.path.exists(p)]
    control(len(lost) == 2, f"the old script's two lines delete run 1's {lost}")


# ---------------------------------------------------------------------------------------------
# A short command line, and a VM with no ~/.tpu_env.

def run_script(path: str, args: list, home: str) -> subprocess.CompletedProcess:
    """Run a copy of measure_model.sh with HOME at `home`, so nothing reads the real one."""
    env = dict(os.environ, HOME=home)
    return subprocess.run(["bash", path, *args], capture_output=True, text=True, env=env,
                          timeout=60)


def check_start(root: str) -> None:
    home = os.path.join(root, "start-home")
    os.makedirs(home)
    out = os.path.join(root, "start-out")
    for args in ([], ["google/gemma-4-26B-A4B-it", "15"]):
        done = run_script(SCRIPT, args, home)
        report(done.returncode == 2 and done.stderr.startswith("usage:")
               and "unbound variable" not in done.stderr and not os.path.exists(out),
               f"{len(args)} argument(s) exit 2 on the usage line: {done.stderr.strip()}")
    # Control: the same script without the usage block dies on the first missing argument.
    lines = open(SCRIPT, encoding="utf-8").read().splitlines()
    start = next(n for n, line in enumerate(lines) if line.startswith("if (($# != 3))"))
    bare = os.path.join(root, "no-usage.sh")
    with open(bare, "w") as fp:
        fp.write("\n".join(lines[:start] + lines[start + 4:]) + "\n")
    done = run_script(bare, [], home)
    control("unbound variable" in done.stderr,
            f"the script without its usage block: {done.stderr.strip()}")

    # With every argument and no ~/.tpu_env, it stops before the fetch.
    done = run_script(SCRIPT, ["google/gemma-4-26B-A4B-it", "15", out], home)
    report(done.returncode == 1 and "run scripts/bootstrap_tpu_vm.sh" in done.stderr
           and "FETCH" not in done.stdout,
           f"a VM with no ~/.tpu_env stops before the fetch: {done.stderr.strip()}")
    # Control: without the check, and cut off where the fetch starts, the script goes on.
    start = next(n for n, line in enumerate(lines) if line.startswith('[ -f "$HOME/.tpu_env" ]'))
    fetch_at = next(n for n, line in enumerate(lines) if line.startswith('log "FETCH'))
    unchecked = os.path.join(root, "no-env-check.sh")
    with open(unchecked, "w") as fp:
        fp.write("\n".join(lines[:start] + lines[start + 2:fetch_at] + ["echo REACHED FETCH"])
                 + "\n")
    done = run_script(unchecked, ["google/gemma-4-26B-A4B-it", "15", out + "-control"], home)
    control("REACHED FETCH" in done.stdout,
            "the script without the check reaches the fetch with no ~/.tpu_env")


# ---------------------------------------------------------------------------------------------
# The DOWNLOAD line, on real fetches of a tiny repo.

TINY_REPO = "hf-internal-testing/tiny-random-gpt2"


def fetch(home: str) -> tuple[int, str, dict]:
    """Run fetch_weights.py on TINY_REPO with HF_HOME at `home`. Returns (exit, output, DOWNLOAD)."""
    env = dict(os.environ, HF_HOME=home, HF_HUB_DISABLE_XET="1")
    env.pop("HF_HUB_OFFLINE", None)
    done = subprocess.run([sys.executable, os.path.join(HERE, "fetch_weights.py"), TINY_REPO],
                          capture_output=True, text=True, env=env, timeout=600)
    text = done.stdout + done.stderr
    fields = {}
    for line in done.stdout.splitlines():
        if line.startswith("DOWNLOAD "):
            for word in line.split()[2:]:
                key, _, value = word.partition("=")
                if value:
                    fields[key] = value
            fields["line"] = line
    return done.returncode, text, fields


def check_download_line(root: str) -> None:
    home = os.path.join(root, "hf")
    blobs = os.path.join(home, "hub", "models--" + TINY_REPO.replace("/", "--"), "blobs")

    code, text, cold = fetch(home)
    on_disk = sum(os.path.getsize(os.path.join(blobs, n)) for n in os.listdir(blobs)) \
        if os.path.isdir(blobs) else 0
    rate = int(cold.get("fetched", 0)) / 1e6 / max(float(cold.get("secs", 1)), 0.05)
    report(code == 0 and on_disk > 0 and int(cold.get("fetched", -1)) == on_disk
           and "rate_MBps" in cold,
           f"a cold fetch counts every byte it wrote, {on_disk}: {cold.get('line', text)}")
    if code != 0:
        print(text)

    code, text, warm = fetch(home)
    report(code == 0 and warm.get("fetched") == "0" and "rate_MBps" not in warm
           and "no rate" in warm.get("line", ""),
           f"a warm rerun fetches 0 bytes and prints no rate: {warm.get('line', text)}")
    old_rate = int(warm.get("bytes", 0)) / 1e6 / max(float(warm.get("secs", 1)), 1e-9)
    control(int(warm.get("bytes", 0)) > 0,
            f"the old line, the snapshot's bytes over this run's seconds, prints {old_rate:.1f} MB/s"
            f" on this cache hit; the cold fetch read {rate:.1f}")

    # A killed fetch: the weights' blob gone, and a tagged partial of it left behind.
    shard = max(os.listdir(blobs), key=lambda n: os.path.getsize(os.path.join(blobs, n)))
    size = os.path.getsize(os.path.join(blobs, shard))
    os.remove(os.path.join(blobs, shard))
    partial = os.path.join(blobs, f"{shard}.f00dfeed.incomplete")
    with open(partial, "wb") as fp:
        fp.write(b"\0" * (size // 2))
    code, text, resumed = fetch(home)
    report(code == 0 and int(resumed.get("fetched", -1)) == size and not os.path.exists(partial)
           and f"PARTIALS removed 1 .incomplete file(s), {size // 2} bytes" in text,
           f"a rerun after a killed fetch removes the {size // 2}-byte partial and counts the"
           f" {size} bytes it fetched: {resumed.get('line', text)}")
    control(int(resumed.get("fetched", -1)) != size - size // 2,
            f"a count that took the partial as progress reads {size - size // 2}, not the"
            f" {resumed.get('fetched')} this run fetched")


root = tempfile.mkdtemp(prefix="measure-model-test-")
try:
    keep, step = step_lines(SCRIPT)
    report(len(keep) == 1 and step[0].startswith("# The steady window") and len(step) > 20,
           f"cut keep_results and the {len(step)}-line steady-window step out of measure_model.sh")

    rc, results, text = run_step(root, REAL_LOG, "real")
    report(rc == 0 and results == [REAL_RESULT],
           f"the real run's log gives its own RESULT line and leaves the status alone: RC {rc}, "
           f"{results[0]['tokens_per_s'] if results else 'no'} tokens/s over "
           f"{results[0]['window'] if results else 'no window'}")

    torn = list(REAL_LOG)
    torn[8] = torn[8] + ENGINE_LINE
    rc, results, text = run_step(root, torn, "torn")
    report(rc == 0 and results == [REAL_RESULT],
           "an engine log line on a progress line gives the same RESULT")

    cases = {
        "a run that closes its only shard after three progress lines":
            REAL_LOG[:5] + REAL_LOG[-3:],
        "a window in which the token count stands still":
            REAL_LOG[:4] + [line.replace(line.split(",")[0], '{"tokens": 21600')
                            for line in REAL_LOG[4:16]] + REAL_LOG[16:],
        "a log whose progress lines all fail to parse":
            [line.replace('"shards": 0,', '"shards": 0') for line in REAL_LOG],
    }
    for n, (name, log) in enumerate(cases.items()):
        rc, results, text = run_step(root, log, f"case{n}")
        control(rc == 1 and not results and "no steady window" in text
                and "The step's whole output" in text,
                f"{name}: RC {rc}, {len(results)} RESULT line(s), and the reason printed")

    print("\na short command line and a VM with no ~/.tpu_env")
    check_start(root)
    print("\na rerun keeps the last run")
    check_keep_last_run(root)
    print("\nthe DOWNLOAD line")
    check_download_line(root)
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
