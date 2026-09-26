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

"""Peak HBM per chip, sampled while a run is in flight.

`Device.memory_stats()` needs the calling process to hold the TPU, and the engine's scheduler
child holds it for the whole run, so the parent can't read it. `tpu-info` reads the same counters
out of `libtpu`'s metrics server from a separate process, so it works alongside. This script
calls its Python API, `tpu_info.metrics.get_chip_usage`, which returns bytes per device, and
never reads the table the `tpu-info` command prints.

    uv pip install tpu-info
    python3 scripts/peak_hbm.py --seconds 600 --out peak_hbm.json    # in a second shell
    python3 scripts/peak_hbm.py --result peak_hbm.json --stage off   # the RESULT line

Report the peak next to the chip count and the slice, and say what was running. The runtime
keeps no high-water mark of HBM in use, so this is the peak of what was sampled: sample often
enough to catch the load, and say so.

A run that reads no chip exits 1 and writes its JSON with `"chips": 0` and an `error`. That's a
host with no TPU, a `tpu-info` that isn't installed, or a metrics server that never came up
because nothing held the TPU. `--result` refuses such a file instead of printing a peak of 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

GIB = float(1 << 30)


class NoChips(Exception):
    """This host has no TPU chip for tpu-info to read, or tpu-info isn't there to read it."""


class Outage:
    """What to print about rounds that find libtpu's metrics server down.

    The first empty round of a stretch says why the rounds come back empty. The rest of the
    stretch prints nothing, and the first round that reads the chips again says how long it lasted.
    """

    def __init__(self):
        self.rounds = 0

    def down(self, code) -> str | None:
        self.rounds += 1
        if self.rounds > 1:
            return None
        return (
            f"libtpu metrics unavailable ({code}). libtpu serves them only while a process holds "
            f"the TPU, so a round before the engine loads or after it exits reads nothing. The "
            f"sampler keeps trying."
        )

    def up(self) -> str | None:
        rounds, self.rounds = self.rounds, 0
        return f"libtpu metrics came back after {rounds} empty round(s)" if rounds else None


OUTAGE = Outage()


def read_usage() -> list[dict]:
    """One reading per TPU device: HBM in use and in total, in bytes, plus the duty cycle.

    Raises NoChips on a host with no TPU. Returns an empty list while `libtpu`'s metrics server
    is down, which lasts until a process holds the TPU, and when the server answers with fewer
    metrics than chips.
    """
    try:
        import grpc
        from tpu_info import device, metrics
    except ImportError as exc:
        raise NoChips(f"tpu-info isn't importable ({exc}); uv pip install tpu-info") from exc
    chip_type, count = device.get_local_chips()
    if chip_type is None or not count:
        raise NoChips("tpu-info finds no TPU chip on this host")
    try:
        # tpu-info's own table takes the per-core reading on v7x and the per-chip one elsewhere.
        if chip_type is device.TpuChip.V7X:
            usage = metrics.get_chip_usage_new(chip_type)
        else:
            usage = metrics.get_chip_usage(chip_type)
    except grpc.RpcError as exc:
        note = OUTAGE.down(exc.code())
        if note:
            print(note, file=sys.stderr)
        return []
    except AssertionError as exc:  # get_chip_usage asserts one reading per chip
        print(f"libtpu metrics incomplete this round: {exc}", file=sys.stderr)
        return []
    note = OUTAGE.up()
    if note:
        print(note, file=sys.stderr)
    return [
        {
            "device": int(u.device_id),
            "used_bytes": int(u.memory_usage),
            "total_bytes": int(u.total_memory),
            "duty_pct": float(u.duty_cycle_pct),
        }
        for u in usage
    ]


class Reader:
    """`read_usage` on a daemon thread, so a metrics call that never returns skips rounds.

    A round that runs past `timeout` returns nothing, and no new call starts until the stuck one
    returns. An exception on the thread comes back out of the call.
    """

    def __init__(self, timeout: float = 120.0, read=read_usage):
        self.timeout = timeout
        self.read = read
        self.pending = None

    def __call__(self) -> list[dict]:
        if self.pending is not None and self.pending.is_alive():
            print("the last tpu-info call hasn't returned; round skipped", file=sys.stderr)
            return []
        box = {}

        def work():
            try:
                box["rows"] = self.read()
            except BaseException as exc:  # handed back to the sampling loop below
                box["error"] = exc

        self.pending = threading.Thread(target=work, daemon=True)
        self.pending.start()
        self.pending.join(self.timeout)
        if self.pending.is_alive():
            print(f"tpu-info ran past {self.timeout:g} s, round skipped", file=sys.stderr)
            return []
        if "error" in box:
            raise box["error"]
        return box["rows"]


def fold(peaks: dict, rows: list[dict]) -> dict:
    """Fold one round of readings into the running peak per device, and return the peaks."""
    for row in rows:
        best = peaks.setdefault(
            row["device"], {"used_bytes": 0, "total_bytes": row["total_bytes"], "duty_pct": 0.0}
        )
        best["used_bytes"] = max(best["used_bytes"], row["used_bytes"])
        best["total_bytes"] = row["total_bytes"]
        if row["duty_pct"] is not None:
            best["duty_pct"] = max(best["duty_pct"], row["duty_pct"])
    return peaks


def summarize(peaks: dict, samples: int, window_s: float, every_s: float, error=None) -> dict:
    """The JSON the sampler writes. GiB figures round to two places, as `tpu-info` prints them."""
    per_device = {
        str(k): dict(
            v,
            used_gib=round(v["used_bytes"] / GIB, 2),
            total_gib=round(v["total_bytes"] / GIB, 2),
        )
        for k, v in sorted(peaks.items())
    }
    result = {
        "chips": len(peaks),
        "samples": samples,
        "window_s": round(window_s, 1),
        "every_s": every_s,
        "peak_per_chip": per_device,
        "peak_used_gib_max": max((v["used_gib"] for v in per_device.values()), default=None),
    }
    if error is None and (not peaks or not samples):
        error = "no round read a chip"
    if error is not None:
        result["error"] = error
    return result


def result_line(path: str, stage: str) -> str:
    """The `RESULT` line for a sampler's JSON. Raises ValueError when it read no chip."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("error") or not d.get("chips") or not d.get("samples"):
        raise ValueError(
            f"{path} holds no HBM reading: {d.get('error') or 'zero chips or zero samples'}"
        )
    first = next(iter(d["peak_per_chip"].values()))
    return "RESULT " + json.dumps(
        {
            "stage": f"peak_hbm_{stage}",
            "chips": d["chips"],
            "samples": d["samples"],
            "window_s": d.get("window_s"),
            "peak_used_gib_max": d["peak_used_gib_max"],
            "total_gib": first["total_gib"],
        }
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=600.0, help="how long to sample")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--out", help="write the peaks here as JSON")
    ap.add_argument("--result", metavar="JSON", help="print the RESULT line for a sampler's JSON")
    ap.add_argument("--stage", default="run", help="the stage name the RESULT line carries")
    args = ap.parse_args(argv)

    if args.result:
        try:
            print(result_line(args.result, args.stage))
        except (OSError, ValueError) as exc:
            print(f"peak_hbm_{args.stage}: {exc}", file=sys.stderr)
            return 1
        return 0

    peaks: dict = {}
    samples = 0  # sampling rounds that read at least one device, not device rows
    error = None
    read = Reader()
    started = time.time()
    deadline = started + args.seconds
    try:
        while time.time() < deadline:
            rows = read()
            samples += bool(rows)
            fold(peaks, rows)
            time.sleep(args.every)
    except NoChips as exc:
        error = str(exc)
    except KeyboardInterrupt:
        pass

    result = summarize(peaks, samples, time.time() - started, args.every, error)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if "error" in result:
        print(f"peak_hbm: {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
