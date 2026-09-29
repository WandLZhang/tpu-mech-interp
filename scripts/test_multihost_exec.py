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
4. A peer that exits while rank 0 runs has to stop rank 0 after `MULTIHOST_GRACE`, so the run
   fails instead of hanging. Control: the wrapper without its watcher still runs at 40 s. When
   rank 0 then ends by itself, the wrapper's output has to close with it, well inside a 300 s
   grace. Control: a grace sleep that keeps the wrapper's stdout still holds the pipe at 60 s.
5. A peer whose first ssh login is refused, through a stand-in `ssh` on `PATH`, has to join on a
   later try. Control: the wrapper with one try starts no rank.
6. At `--tp-size 2 --dp-size 2` each process holds one data-parallel rank. Three requests pinned
   to ranks 0, 1 and 1, one of them split across 64-token prefill passes, with decode steps, have
   to come back from their own ranks, and slot 0 of every row has to be the embedding of its
   token. Control: the output processor reading one packed run of rows, as it did before the
   per-rank offsets, has to finish and misfile rank 1's rows.

Control for 2: the same two-rank capture on the tree without the multi-host patch has to fail.
There the scheduler slices a hidden-state array whose rows sit on the other process's device.

Every check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
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


def run_group(cmd, env, timeout):
    """run, in a process group of its own that goes whole when it ends. rc is None on a timeout."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        rc = None
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # a sleep the wrapper left behind
    except ProcessLookupError:
        pass
    return rc, out


# Stands in for ssh: refuses each host's first login with ssh's own status, 255, then runs the
# command in a local shell with HOME in the test folder. The cleanup's kills would reach this VM's
# own processes, so they only get logged.
FAKE_SSH = r"""#!/usr/bin/env bash
while (($#)); do case $1 in -i|-o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift
n=$(($(cat "$STATE/$host.logins" 2>/dev/null || echo 0) + 1)); echo $n >"$STATE/$host.logins"
echo "$*" >>"$STATE/$host.commands"
((n == 1)) && exit 255
case "$*" in *multihost-rank.pid*pgrep*) exit 0 ;; esac
exec env HOME="$STATE/home" bash -c "$*"
"""


def wrapper_checks(root):
    """Checks 4 and 5: the wrapper's own failure handling, with plain commands and no engine."""
    source = open(os.path.join(HERE, "multihost_exec.sh")).read()

    def mutant(name, old, new):
        if old not in source:
            raise SystemExit(f"multihost_exec.sh no longer holds {old!r}; update the {name} mutant")
        path = os.path.join(root, f"multihost_exec_{name}.sh")
        with open(path, "w") as fp:
            fp.write(source.replace(old, new))
        return ["bash", path]

    wrapper = ["bash", os.path.join(HERE, "multihost_exec.sh")]
    hosts = os.path.join(root, "hosts-local")
    with open(hosts, "w") as fp:
        fp.write("127.0.0.1\n127.0.0.1\n")
    env = dict(os.environ, MULTIHOST_HOSTS=hosts, MULTIHOST_LOCAL="1", MULTIHOST_GRACE="5",
               MULTIHOST_LOGS=os.path.join(root, "logs-wrapper"))
    fails = ["bash", "-c", 'if [ "$SGL_NODE_RANK" = 0 ]; then sleep 300; else exit 3; fi']

    print("a peer that exits early stops rank 0", flush=True)
    rc, out = run_group(wrapper + fails, env, timeout=120)
    ok = rc not in (None, 0) and "rank 1 ended with status 3" in out
    report(ok, f"rank 1 exits 3 and rank 0's sleep 300 ends: the wrapper exits {rc}")
    if not ok:
        print(out)
    rc, out = run_group(mutant("nowatch", "((WAIT_ALL)) || { watch_peers & WATCH=$!; }", ":") + fails,
                        env, timeout=40)
    control(rc is None, "the wrapper without its watcher is still waiting on rank 0 at 40 s")
    # A healthy run's peers end a moment before rank 0. The watcher is in its grace sleep then,
    # and the wrapper kills the watcher, not the sleep.
    ends = ["bash", "-c", 'if [ "$SGL_NODE_RANK" = 0 ]; then sleep 8; fi']
    env_long = dict(env, MULTIHOST_GRACE="300")
    rc, out = run_group(wrapper + ends, env_long, timeout=60)
    report(rc == 0, f"rank 1 ends first and rank 0 after 8 s: the wrapper's output closes, exit {rc}")
    if rc != 0:
        print(out)
    held = mutant("holdpipe", 'sleep "$grace" >/dev/null 2>&1', 'sleep "$grace"')
    rc, out = run_group(held + ends, env_long, timeout=60)
    control(rc is None, "a grace sleep that keeps the wrapper's stdout still holds the pipe at 60 s")

    print("a refused ssh login gets another try", flush=True)
    bindir, state = os.path.join(root, "bin"), os.path.join(root, "ssh-state")
    os.makedirs(bindir)
    os.makedirs(os.path.join(state, "home"))
    with open(os.path.join(bindir, "ssh"), "w") as fp:
        fp.write(FAKE_SSH)
    os.chmod(os.path.join(bindir, "ssh"), 0o755)
    with open(hosts, "w") as fp:
        fp.write("127.0.0.1\npeer-a\n")
    env = dict(os.environ, MULTIHOST_HOSTS=hosts, MULTIHOST_LOGS=os.path.join(root, "logs-ssh"),
               PATH=bindir + os.pathsep + os.environ["PATH"], STATE=state)
    env.pop("MULTIHOST_LOCAL", None)
    ranks = ["--wait-all", "bash", "-c", 'echo "RANK $SGL_NODE_RANK of $SGL_NNODES"']
    rc, out = run_group(wrapper + ranks, env, timeout=120)
    logins = open(os.path.join(state, "peer-a.logins")).read().strip()
    ok = (rc == 0 and "RANK 0 of 2" in out and "RANK 1 of 2" in out
          and "ssh to peer-a failed (try 1 of 20)" in out)
    report(ok, f"after one refused login both ranks run: the wrapper exits {rc}, {logins} logins to peer-a")
    if not ok:
        print(out)
    os.remove(os.path.join(state, "peer-a.logins"))
    rc, out = run_group(mutant("onetry", "for try in $(seq 1 20); do", "for try in $(seq 1 1); do") + ranks,
                        env, timeout=120)
    control(rc not in (None, 0) and "RANK 0 of 2" not in out,
            f"the wrapper with one try exits {rc} before any rank starts")


DP_RANKS = [0, 1, 1]


def dp_rows(model, out):
    """Check 6 as each rank runs it under the wrapper. Rank 0 sends the requests and writes `out`.

    Rank 1's Engine blocks in its scheduler and never returns. Slot 0 is the embedding output, so
    in float32 each row's slot 0 has to equal the embedding of its token, whatever attention
    computes: the prompt, then every generated token but the last, which no pass reads.
    """
    import numpy as np
    from safetensors.numpy import load_file

    import capture_activations as ca

    # The model's 512 ids run past the tokenizer's 400, so nothing detokenizes.
    engine = ca.open_engine(model, tp_size=2, dp_size=2, batch_size=8, token_padding=64,
                            chunked_prefill_size=64, device="cpu", dtype="float32",
                            max_total_tokens=4096, log_level="error", skip_tokenizer_init=True,
                            disable_overlap_schedule=True, random_seed=0)
    from sgl_jax.srt.managers.io_struct import GenerateReqInput

    embed = load_file(os.path.join(model, "model.safetensors"))["model.embed_tokens.weight"]
    rng = np.random.default_rng(21)
    prompts = [rng.integers(3, cpu_engine.VOCAB, size=n).tolist() for n in (12, 90, 20)]
    steps = {"temperature": 0.0, "ignore_eos": True, "max_new_tokens": 4}
    ca.log_request(call="generate_request", input_ids=prompts, sampling_params=steps,
                   return_hidden_states=True, dp_rank=DP_RANKS)
    obj = GenerateReqInput(input_ids=prompts, sampling_params=[dict(steps) for _ in prompts],
                           return_hidden_states=True, dp_rank=DP_RANKS)
    outs = engine.loop.run_until_complete(engine.tokenizer_manager.generate_request(obj, None).__anext__())
    found = []
    for prompt, reply in zip(prompts, outs):
        rows, _ = ca.hidden_states_from_output(reply)
        want = embed[list(prompt) + [int(t) for t in reply["output_ids"][:-1]]]
        got = rows[:, 0, :]
        rel = float(np.abs(got - want).max() / np.abs(want).max()) if got.shape == want.shape else float("inf")
        found.append({"rank": reply["meta_info"].get("dp_rank"),
                      "prompt_tokens": reply["meta_info"].get("prompt_tokens"),
                      "rows": int(rows.shape[0]), "slot0_rel": rel})
    print("DP_ROWS " + json.dumps(found), flush=True)
    with open(out, "w") as fp:
        json.dump(found, fp)
    engine.shutdown()
    os._exit(0)


def pack_rows(tree):
    """Check 6's control: the output processor as it read before the per-rank offsets.

    Prefill keeps one cursor across ranks, and decode reads row i for request i of every rank.
    """
    path = os.path.join(tree, "python", "sgl_jax", "srt", "managers", "scheduler_output_processor_mixin.py")
    text = open(path).read()
    for old, new in (("hidden_state_offset = dp_rank * (\n"
                      "                    logits_output.hidden_states.shape[0] // batch.dp_size\n"
                      "                )", "pass"),
                     ("row = per_dp_bs_size * dp_rank + i", "row = i")):
        if text.count(old) != 1:
            raise SystemExit(f"{path} holds {old!r} {text.count(old)} time(s); update pack_rows")
        text = text.replace(old, new)
    with open(path, "w") as fp:
        fp.write(text)


def main() -> int:
    import random

    root = tempfile.mkdtemp(prefix="multihost-test-")
    try:
        wrapper_checks(root)
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

        print("capture at --dp-size 2, one rank to a process", flush=True)
        rows_out = os.path.join(root, "dp-rows.json")
        dp = wrapper + [sys.executable, os.path.abspath(__file__), "--dp-rows", model, rows_out]

        def dp_run(tree):
            if os.path.exists(rows_out):
                os.remove(rows_out)
            rc, out = run(dp, env_for(tree), timeout=900)
            found = json.load(open(rows_out)) if os.path.exists(rows_out) else []
            return rc, out, found

        rc, out, found = dp_run(fixed)
        ranks = [r["rank"] for r in found]
        errors = [r["slot0_rel"] for r in found]
        ok = (rc == 0 and ranks == DP_RANKS and found[1]["prompt_tokens"] > 64
              and all(e < 1e-6 for e in errors))
        report(ok, f"rank 0 exits {rc}; replies from ranks {ranks}, slot-0 errors {errors}")
        if not ok:
            print(out)
        packed = cpu_engine.build_tree(os.path.join(root, "packed"), os.environ.get("SGLANG_JAX_REPO") or None)
        subprocess.run(["git", "-C", packed, "apply", PATCH], check=True)
        pack_rows(packed)
        rc, out, found = dp_run(packed)
        wrong = [r["slot0_rel"] for r in found if r["rank"] == 1]
        detected = rc == 0 and len(found) == len(DP_RANKS) and any(e >= 1e-6 for e in wrong)
        control(detected, f"one packed run of rows exits {rc} and files rank 1's rows at slot-0 errors {wrong}")
        if not detected:
            print(out)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{failures} failure(s)" if failures else "All checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--dp-rows"]:
        dp_rows(sys.argv[2], sys.argv[3])
    sys.exit(main())
