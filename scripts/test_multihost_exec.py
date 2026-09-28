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

"""CPU gate for multi-host capture: one engine across two processes, through multihost_exec.sh.

    python3 scripts/test_multihost_exec.py

A multi-host slice runs one sglang-jax engine across hosts, and `multihost_exec.sh` starts the
same command on each with its rank. This gate runs two ranks on one CPU VM with
`MULTIHOST_LOCAL=1`, on `cpu_engine.py`'s tiny Qwen3:

1. `check_capture.py --reference-only` writes the float32 reference and the BF16 floor.
2. The wrapper runs `check_capture.py --tp-size 2 --reference-npz` as ranks 0 and 1 of one
   engine, on the patched tree plus `upstream/multihost-hidden-states.patch`. Every layer of both
   prompts has to pass.
3. `--wait-all` with a `jax.distributed` device count has to print one line per rank, each
   seeing both processes' devices.

Control: the same two-rank capture on the tree without the multi-host patch has to fail. There the
scheduler slices a hidden-state array whose rows sit on the other process's device.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cpu_engine  # noqa: E402

PATCH = os.path.join(REPO, "upstream", "multihost-hidden-states.patch")
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


def run(cmd, env, timeout=1200):
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return done.returncode, done.stdout.replace("\r", "\n") + done.stderr.replace("\r", "\n")


def main() -> int:
    import random

    root = tempfile.mkdtemp(prefix="multihost-test-")
    try:
        plain = cpu_engine.build_tree(os.path.join(root, "plain"), os.environ.get("SGLANG_JAX_REPO") or None)
        fixed = cpu_engine.build_tree(os.path.join(root, "fixed"), os.environ.get("SGLANG_JAX_REPO") or None)
        subprocess.run(["git", "-C", fixed, "apply", PATCH], check=True)
        model = cpu_engine.write_checkpoint(os.path.join(root, "qwen3-tiny"))
        rng = random.Random(1)
        prompts = os.path.join(root, "prompts.txt")
        with open(prompts, "w") as fp:
            fp.write("\n".join(" ".join(rng.choice(cpu_engine.WORDS) for _ in range(n))
                               for n in (12, 20, 25, 18)) + "\n")
        hosts = os.path.join(root, "hosts")
        with open(hosts, "w") as fp:
            fp.write("127.0.0.1\n127.0.0.1\n")
        ref = os.path.join(root, "ref.npz")
        common = ["--model-path", model, "--batch-size", "8", "--token-padding", "64",
                  "--prompts-file", prompts, "--num-prompts", "1", "--chunked-prefill-size", "64",
                  "--engine-arg", "device=cpu", "--engine-arg", "max_total_tokens=4096",
                  "--engine-arg", "log_level=error", "--engine-arg", "disable_overlap_schedule=true",
                  "--engine-arg", "random_seed=0"]
        check = [sys.executable, os.path.join(HERE, "check_capture.py")]

        def env_for(tree):
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(tree, "python"), HERE]),
                       JAX_PLATFORMS="cpu", MULTIHOST_HOSTS=hosts, MULTIHOST_LOCAL="1",
                       MULTIHOST_LOGS=os.path.join(root, "logs"))
            env.pop("XLA_FLAGS", None)
            return env

        print("the reference, one process", flush=True)
        rc, out = run(check + common + ["--tp-size", "1", "--reference-only", ref], env_for(fixed))
        report(rc == 0 and os.path.exists(ref), f"--reference-only exits {rc} and writes {os.path.basename(ref)}")
        if rc:
            print(out)
            return 1

        wrapper = ["bash", os.path.join(HERE, "multihost_exec.sh")]
        print("capture across two ranks, with the multi-host patch", flush=True)
        rc, out = run(wrapper + check + common + ["--tp-size", "2", "--reference-npz", ref], env_for(fixed))
        results = [line for line in out.splitlines() if line.startswith("RESULT ")]
        ok = rc == 0 and "All checks passed." in out and len(results) == 2
        report(ok, f"rank 0 exits {rc}, {len(results)} RESULT line(s), every layer inside the BF16 floor")
        if not ok:
            print(out)

        print("capture across two ranks, without it", flush=True)
        rc, out = run(wrapper + check + common + ["--tp-size", "2", "--reference-npz", ref], env_for(plain))
        control(rc != 0 and "ShardingTypeError" in out,
                f"the tree without the patch exits {rc} on a sharding error")

        print("cleanup leaves other processes alone", flush=True)
        # A long job on the same VM, such as a streamed reference, whose command line the slice
        # cleanup's engine pattern matches.
        sleeper = os.path.join(root, "check_capture.py")
        with open(sleeper, "w") as fp:
            fp.write("import time\ntime.sleep(600)\n")
        bystander = subprocess.Popen(["python3", sleeper])
        pattern = ("^[^ ]*python[0-9.]* (-u )?([^ ]*/)?"
                   "(serve_throughput|capture_activations|check_capture)\\.py")
        matched = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        rc, out = run(wrapper + [sys.executable, "-c", "print('rank done')"], env_for(fixed), timeout=300)
        alive = bystander.poll() is None
        bystander.kill()
        report(rc == 0 and alive, f"a check_capture.py outside the run survives the local cleanup (rc {rc})")
        control(str(bystander.pid) in matched,
                "the slice cleanup's engine pattern matches that process, so killing by pattern would end it")

        print("jax.distributed across the two ranks", flush=True)
        probe = ("import os, jax; jax.distributed.initialize(os.environ['SGL_DIST_INIT_ADDR'], "
                 "int(os.environ['SGL_NNODES']), int(os.environ['SGL_NODE_RANK'])); "
                 "print('DEVICES', jax.process_index(), jax.process_count(), jax.device_count(), flush=True)")
        rc, out = run(wrapper + ["--wait-all", sys.executable, "-c", probe], env_for(fixed), timeout=300)
        lines = sorted(line for line in out.splitlines() if line.startswith("DEVICES"))
        report(rc == 0 and lines == ["DEVICES 0 2 2", "DEVICES 1 2 2"], f"--wait-all exits {rc}: {lines}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{failures} failure(s)" if failures else "All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
