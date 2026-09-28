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

"""CPU gate for `glm5-next-probe.patch`, the GLM-5.3-Flash `SGL_PROBE_MOE=dense` switch.

    python3 upstream/models/test_glm5_next_probe.py

The switch replaces EPMoE's dispatch, gmm and unpermute in the routed experts with
`moe_probe.dense_experts`: every local expert over every token in float32 at HIGHEST, weighted by
float32 top-k weights. It reuses `test_glm5_next_model.py`'s tiny checkpoint, its reference and its
runner, on 8 simulated CPU devices in one process.

1. The probe patch applies on the model patch, a copy with one context line rewritten doesn't,
   and the files it touches compile.
2. At --tp-size 1, 4 and 8 with --ep-size equal to it, the switch leaves the batch prefill of two
   requests, the split prefill and three decode steps within the model gate's tolerance of the
   default path at tp 1, and the prefill within it of transformers. The default path at tp 4 and
   8 is reported per request beside it. Control: the dense path with its output nudged 1% moves
   past the tolerance.
3. The switch off traces the default path: with `SGL_PROBE_MOE` unset the Glm5NextEPMoE forward
   is the model patch's, and a misspelled value raises.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile

DEVICES = 8
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = f"{_flags} --xla_force_host_platform_device_count={DEVICES}".strip()

HERE = os.path.dirname(os.path.abspath(__file__))


def load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The model gate holds the tiny checkpoint, the reference and the runner this reuses.
g = load_by_path("test_glm5_next_model", os.path.join(HERE, "test_glm5_next_model.py"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

PROBE = os.path.join(HERE, "glm5-next-probe.patch")
LAYOUTS = ((1, 1), (4, 4), (8, 8))
record = g.record


@contextlib.contextmanager
def probe_env(value):
    saved = os.environ.get("SGL_PROBE_MOE")
    os.environ.pop("SGL_PROBE_MOE", None)
    if value:
        os.environ["SGL_PROBE_MOE"] = value
    try:
        yield
    finally:
        os.environ.pop("SGL_PROBE_MOE", None)
        if saved is not None:
            os.environ["SGL_PROBE_MOE"] = saved


@contextlib.contextmanager
def patched(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def build_tree(workdir):
    print("check 1: the probe patch applies on the model patch and compiles", flush=True)
    tree = g.cpu_engine.build_tree(os.path.join(workdir, "sglang-jax"),
                                   os.environ.get("SGLANG_JAX_REPO") or None)
    applied = g.git(tree, "apply", g.PATCH)
    if not record(applied.returncode == 0,
                  f"glm5-next-model.patch applies: {applied.stderr.strip()}"):
        return None
    with open(PROBE) as fp:
        bad = os.path.join(workdir, "corrupt-probe.patch")
        with open(bad, "w") as out:
            out.write(g.corrupt(fp.read()))
    record(g.git(tree, "apply", "--check", bad).returncode != 0,
           "control: the probe patch with a rewritten context line is refused")
    applied = g.git(tree, "apply", PROBE)
    if not record(applied.returncode == 0,
                  f"glm5-next-probe.patch applies on it: {applied.stderr.strip()}"):
        return None
    touched = [line[6:].strip() for line in open(PROBE) if line.startswith("+++ b/")]
    for path in touched:
        done = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(tree, path)],
                              capture_output=True, text=True)
        record(done.returncode == 0, f"{path} compiles")
    return tree


def phase_verdicts(got, want):
    """Per phase of `run_sequence`, the model gate's verdict of `got` against `want`."""
    out = {}
    for phase in ("prefill", "split", "decode"):
        errs = [g.compare(a, b) for a, b in zip(got[phase], want[phase])]
        out[phase] = g.verdict(np.concatenate([e[0] for e in errs]))
    return out


def per_request(got, want):
    """The worst per-token error of each prefill request against `want`'s."""
    return [float(g.compare(a, b)[0].max()) for a, b in zip(got["prefill"], want["prefill"])]


def text_of(verdicts):
    return ", ".join(f"{phase} {v[1]:.0%} over, worst {v[2]:.1e}" for phase, v in verdicts.items())


def dense_matches(tree):
    print(f"check 2: SGL_PROBE_MOE=dense at --tp-size 1, 4 and 8 on {DEVICES} CPU devices",
          flush=True)
    env = g.setup(tree, g.fetch_published(), quiet=True)
    from sgl_jax.srt.layers import moe_probe

    seqs = env["seqs"]
    ref_runs = g.reference_runs(env["reference"], seqs)
    with probe_env(""):
        base = g.run_sequence(env["runner"], seqs)
    for tensor, ep in LAYOUTS:
        runner = env["runner"] if tensor == 1 else g.JaxRunner(
            env["mod"], env["cfgmod"], env["dirs"]["pub"], g.make_mesh(tensor), ep_size=ep)
        experts = runner.model.model.layers[2].mlp.experts
        label = f"--tp-size {tensor} --ep-size {ep} (EPMoE {experts.ep_size} x {experts.tp_size})"
        if tensor > 1:
            with probe_env(""):
                default = g.run_sequence(runner, seqs)
            requests = per_request(default, base)
            verdicts = phase_verdicts(default, base)
            print(f"  note: {label}, default path against tp 1: {text_of(verdicts)}; prefill"
                  f" request 0 worst {requests[0]:.1e}, request 1 {requests[1]:.1e}",
                  flush=True)
        with probe_env("dense"):
            dense = g.run_sequence(runner, seqs)
        verdicts = phase_verdicts(dense, base)
        requests = per_request(dense, base)
        record(all(v[0] for v in verdicts.values()),
               f"{label}, moe=dense against the default path at tp 1: {text_of(verdicts)}; prefill "
               f"request 0 worst {requests[0]:.1e}, request 1 {requests[1]:.1e}")
        errs = [g.compare(a, r[0])[0] for a, r in zip(dense["prefill"], ref_runs)]
        against_hf = [float(e.max()) for e in errs]
        hf_ok = all(g.verdict(e)[0] for e in errs)
        record(hf_ok, f"{label}, moe=dense prefill against transformers: request 0 worst "
               f"{against_hf[0]:.1e}, request 1 {against_hf[1]:.1e}")
        if tensor == LAYOUTS[-1][0]:
            scaled = moe_probe.dense_experts
            with probe_env("dense"), patched(moe_probe, "dense_experts",
                                             lambda *a, **k: scaled(*a, **k) * 1.01):
                nudged = g.run_sequence(runner, seqs)
            moved = phase_verdicts(nudged, base)
            record(not all(v[0] for v in moved.values()),
                   f"control: {label}, the dense path nudged 1% differs: {text_of(moved)}")
    return env


def switch_off_is_default(env):
    print("check 3: the switch off is the model patch's path, and a typo raises", flush=True)
    mod = env["mod"]
    from sgl_jax.srt.layers import moe_probe

    calls = {"dense": 0}
    original = mod.Glm5NextEPMoE._probe_dense_forward

    def counting(self, *a, **k):
        calls["dense"] += 1
        return original(self, *a, **k)

    with probe_env(""), patched(mod.Glm5NextEPMoE, "_probe_dense_forward", counting):
        env["runner"].reset()
        seqs = env["seqs"]
        env["runner"].forward([(0, 0, seqs[0][:g.LENGTHS[0]])])
    record(calls["dense"] == 0, f"without SGL_PROBE_MOE the dense body runs {calls['dense']} times")
    with probe_env("dense"), patched(mod.Glm5NextEPMoE, "_probe_dense_forward", counting):
        env["runner"].reset()
        env["runner"].forward([(0, 0, seqs[0][:g.LENGTHS[0]])])
    record(calls["dense"] > 0, f"control: with it the dense body runs {calls['dense']} times")
    with probe_env("densee"):
        try:
            moe_probe.probe("MOE")
            raised = False
        except ValueError:
            raised = True
    record(raised, "SGL_PROBE_MOE=densee raises")


def main() -> int:
    if jax.device_count() < DEVICES:
        record(False, f"{jax.device_count()} CPU devices; the gate needs {DEVICES}")
        return 1
    with tempfile.TemporaryDirectory(prefix="glm5probe-") as workdir:
        try:
            tree = build_tree(workdir)
            if tree is not None:
                env = dense_matches(tree)
                switch_off_is_default(env)
        except Exception:
            import traceback

            traceback.print_exc()
            record(False, "the gate ran to the end")
    failed = [label for ok, label in g.RESULTS if not ok]
    print(f"{len(g.RESULTS)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
