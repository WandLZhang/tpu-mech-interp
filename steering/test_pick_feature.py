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

"""CPU gate for pick_feature.py. Captures through the real patched engine, then ranks latents.

    python3 steering/test_pick_feature.py

`cpu_rig.py` serves a tiny random Gemma 4 with the real tokenizer through the patched
`sglang-jax` on CPU, and `scripts/capture_activations.py` captures slots 1 and 3 from it with the
command line the README runs on a TPU. The SAE's latents come from that capture's own statistics
and reach disk through `sae/train.py::save`, so the checkpoint records slot 3 the way a trained
one does. Every rate this file expects is computed from the shard file, found through the
manifest's own layer list, and never through `pick_feature`. The output ends on `FEATURE=<pick>`,
and bash runs the `export` line from pick_feature's docstring over it.

Needs what `cpu_rig.py` needs: the engine's runtime imports, `git`, and network access.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, "sae"))
import pick_feature  # noqa: E402
from cpu_rig import BATCH_SIZE, D_MODEL, PROMPTS, TOKEN_PADDING, Rig, engine_args  # noqa: E402
from sae import SAEConfig, SAEParams  # noqa: E402
from train import TrainResult, save  # noqa: E402

KEEP = (1, 3)
SLOT = 3  # the slot the SAE is fit on
OTHER = 1  # the other slot the capture kept
TOKENS = 4096

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


def run(argv):
    """pick_feature's command line in this process. Returns `(exit code, stdout, stderr)`."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = pick_feature.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def listed(stdout):
    """The `{"feature": ...}` lines pick_feature printed, in order."""
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def exported(stdout, root):
    """What the docstring's `export "$(tail -n 1 pick.log)"` sets FEATURE to. `(exit, value)`."""
    path = os.path.join(root, "pick.log")
    with open(path, "w") as fp:
        fp.write(stdout)
    done = subprocess.run(
        ["bash", "-c", 'export "$(tail -n 1 "$1")" && printf %s "$FEATURE"', "_", path],
        capture_output=True,
        text=True,
    )
    return done.returncode, done.stdout


def slot_rows(manifest_path, slot, tokens):
    """The first `tokens` rows of `slot` in the first shard, read with json and NumPy alone."""
    with open(manifest_path) as fp:
        manifest = json.load(fp)
    first = os.path.join(os.path.dirname(manifest_path), manifest["shards"][0]["path"])
    return np.load(first)[:tokens, manifest["layers"].index(slot), :].astype(np.float64)


def rates(x, w, b, theta):
    """Share of tokens where `w . x + b` reaches `theta`, per latent, in float64."""
    return ((x @ w + b) >= theta).mean(axis=0)


def midpoint(z, fraction):
    """A threshold halfway between two neighboring distinct values, `fraction` of the way up.

    Every prompt opens on the same BOS row, so the projections repeat. A threshold between two
    distinct values keeps every token clear of it, and float32 against float64 can't flip one.
    """
    distinct = np.unique(z)
    i = int(fraction * (len(distinct) - 1))
    return (distinct[i] + distinct[i + 1]) / 2.0


def write_sae(path, w_enc, theta, capture_layer):
    d_sae = w_enc.shape[1]
    params = SAEParams(
        w_enc=w_enc.astype(np.float32),
        b_enc=np.zeros(d_sae, np.float32),
        w_dec=np.eye(d_sae, D_MODEL, dtype=np.float32),
        b_dec=np.zeros(D_MODEL, np.float32),
    )
    cfg = SAEConfig(d_model=D_MODEL, expansion_factor=d_sae // D_MODEL, k=1)
    result = TrainResult(params, np.asarray(theta, np.float32), 1.0, [])
    return save(path, cfg, result, capture_layer)


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        rig = Rig(root)
        prompts = os.path.join(root, "prompts.txt")
        with open(prompts, "w") as fp:
            fp.write("\n".join(PROMPTS) + "\n")
        caps = os.path.join(root, "caps")
        print("capture through the patched engine")
        proc = rig.run(
            "scripts/capture_activations.py",
            "--model-path", rig.model,
            "--prompts", prompts,
            "--out", caps,
            "--layers", ",".join(str(s) for s in KEEP),
            "--dtype", "float32",
            "--tp-size", 1,
            "--batch-size", BATCH_SIZE,
            "--token-padding", TOKEN_PADDING,
            "--max-new-tokens", 3,
            *engine_args(),
        )
        manifest = os.path.join(caps, "manifest.json")
        report(
            proc.returncode == 0 and os.path.exists(manifest),
            f"the engine captured slots {list(KEEP)} of {len(PROMPTS)} prompts",
        )
        if proc.returncode != 0 or not os.path.exists(manifest):
            return 1

        x = slot_rows(manifest, SLOT, TOKENS)
        x_other = slot_rows(manifest, OTHER, TOKENS)
        n = x.shape[0]
        print(f"\nranking on slot {SLOT}, {n} tokens")

        # Latents 0 and 1 are dead. 2, 3 and 4 read the direction of slot 3's mean, with
        # thresholds 60%, 1% and 97% of the way up its distinct projections, so 3 fires on
        # nearly every token and 4 on few. Every other latent is dead.
        mean = x.mean(axis=0)
        w = np.zeros((D_MODEL, D_MODEL))
        w[:, 2] = w[:, 3] = w[:, 4] = mean / (mean @ mean)
        z = x @ w[:, 2]
        theta = np.full(D_MODEL, np.inf)
        theta[2] = midpoint(z, 0.6)
        theta[3] = midpoint(z, 0.01)
        theta[4] = midpoint(z, 0.97)
        sae = write_sae(os.path.join(root, "sae.npz"), w, theta, capture_layer=SLOT)
        # Expect from what reached disk, float32 as saved, so the reference reads the same
        # parameters pick_feature reads.
        with np.load(sae) as data:
            w = np.asarray(data["w_enc"], np.float64)
            theta = np.asarray(data["threshold"], np.float64)

        live = np.flatnonzero(np.isfinite(theta))
        want = dict(zip(live.tolist(), rates(x, w[:, live], 0.0, theta[live]).tolist()))
        want_other = dict(zip(live.tolist(), rates(x_other, w[:, live], 0.0, theta[live]).tolist()))
        in_bounds = sorted((int(f) for f in live if 0 < want[f] <= 0.5), key=lambda f: -want[f])

        code, out, err = run(["--sae", sae, "--manifest", manifest, "--layer", str(SLOT)])
        print(out + err)
        picks = listed(out)
        got = {p["feature"]: p["fire_rate"] for p in picks}
        report(
            code == 0 and f"{live.size} live of {D_MODEL} latents" in out,
            "counts live latents from the thresholds",
        )
        last = out.splitlines()[-1] if out.splitlines() else ""
        report(
            [p["feature"] for p in picks] == in_bounds and last == f"FEATURE={in_bounds[0]}",
            f"lists {in_bounds} by rate and ends on FEATURE={in_bounds[0]}, the latent that fires"
            f" most within the bound",
        )
        code_export, value = exported(out, root)
        report(
            code_export == 0 and value == str(in_bounds[0]),
            f"export \"$(tail -n 1 pick.log)\" sets FEATURE={value!r} and the ranking stays in"
            f" the log",
        )
        old_form = out.replace(f"FEATURE={in_bounds[0]}", f"PICK {in_bounds[0]}")
        code_old, _ = exported(old_form, root)
        control(
            code_old != 0,
            f"the same export over a log that ends on 'PICK {in_bounds[0]}' exits {code_old}",
        )
        report(
            all(got[f] == round(want[f], 5) for f in in_bounds),
            "every listed rate is the rate on slot 3, read through the manifest: "
            + ", ".join(f"{f}: {got.get(f)} against {round(want[f], 5)}" for f in in_bounds),
        )
        control(
            any(got.get(f) != round(want_other[f], 5) for f in in_bounds),
            "the same rates read from slot 1, the other slot on the shard, disagree: "
            + ", ".join(f"{f}: {round(want_other[f], 5)}" for f in in_bounds),
        )
        control(3 not in got, "a latent firing on almost every token is dropped at --max-rate 0.5")
        control(0 not in got and 1 not in got, "dead latents never appear")

        print("\nthe slot the checkpoint records")
        code_default, out_default, _ = run(["--sae", sae, "--manifest", manifest])
        report(
            code_default == 0 and listed(out_default) == picks,
            f"--layer defaults to slot {SLOT}, the one sae/train.py recorded",
        )
        code, out, err = run(["--sae", sae, "--manifest", manifest, "--layer", str(OTHER)])
        report(
            code == 2 and f"which is {SLOT}" in err and "FEATURE=" not in out,
            f"--layer {OTHER}, a slot the capture holds but the SAE never read, exits {code}",
        )
        unrecorded = write_sae(os.path.join(root, "unrecorded.npz"), w, theta, capture_layer=None)
        code, out, _ = run(["--sae", unrecorded, "--manifest", manifest, "--layer", str(OTHER)])
        control(
            code == 0 and "FEATURE=" in out,
            "a checkpoint that records no slot ranks --layer 1 as asked",
        )

        print("\nwithout a manifest")
        code, out, _ = run(["--sae", sae])
        by_threshold = [int(f) for f in live[np.argsort(theta[live])]]
        report(
            code == 0
            and "ranked by threshold" in out
            and [p["feature"] for p in listed(out)] == by_threshold,
            f"ranks by threshold, lowest first: {by_threshold}",
        )

        none_live = np.full(D_MODEL, np.inf)
        dead = write_sae(os.path.join(root, "dead.npz"), w, none_live, capture_layer=SLOT)
        code, _, _ = run(["--sae", dead])
        control(code == 1, "an SAE with no live latents exits 1 instead of picking one")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
