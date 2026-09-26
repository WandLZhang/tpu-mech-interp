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

"""Gate for the tree `bootstrap_tpu_vm.sh` builds and the stamp that says it's built.

    python3 scripts/test_bootstrap_tree.py
    SGLANG_JAX_REPO=/path/to/full/clone python3 scripts/test_bootstrap_tree.py

Runs the script's `--tree-only` mode against real clones of sglang-jax and a copy of the patches
in upstream/: a first build, a rerun with nothing changed, and a rerun after one patch changes.
The rerun has to keep the tree, and the changed patch has to reach a new one, with the old tree
moved aside. Then `--model`: Nemotron 3 on top, the same model again, as a second Reproduce on
one VM runs it, then gpt-oss, as a VM that moves from one model to the other runs it. The second
Nemotron 3 run has to keep the tree, and gpt-oss has to get a tree of its own, with the Nemotron 3
tree moved aside. An unknown name exits 2. Without `SGLANG_JAX_REPO` it clones GitHub, which
needs full history. The build takes `SGL_COMMIT` from the environment, as the script does,
default eb061d8.

Each check carries a control. A control that passes fails the run.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, os.pardir, "upstream")
PATCHES = ("sglang-jax-877.patch", "steering-hook.patch", "qwen3-steering-hook.patch")
MODEL_PATCHES = ("models/nemotron3-model.patch", "models/nemotron3-capture-hook.patch",
                 "models/gpt-oss-model.patch")
NEMOTRON = "python/sgl_jax/srt/models/nemotron_h.py"
GPT_OSS = "python/sgl_jax/srt/models/gpt_oss.py"
MARKER = "bootstrap gate: this line changes the patch file"
# The commit bootstrap_tpu_vm.sh builds at, read the way it reads it. test_all.sh sets it.
SGL_COMMIT = os.environ.get("SGL_COMMIT", "eb061d8")
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


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


root = tempfile.mkdtemp(prefix="bootstrap-tree-test-")
try:
    source = os.environ.get("SGLANG_JAX_REPO")
    if not source:
        source = os.path.join(root, "sglang-jax")
        print("cloning sglang-jax from GitHub (full history)", flush=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/sgl-project/sglang-jax", source],
                       check=True)
    patches = os.path.join(root, "patches")
    os.makedirs(os.path.join(patches, "models"))
    for name in PATCHES + MODEL_PATCHES:
        shutil.copyfile(os.path.join(UPSTREAM, name), os.path.join(patches, name))
    tree = os.path.join(root, "w")
    env = dict(os.environ, TREE=tree, PATCHES=patches, SGLANG_JAX_REPO=source,
               LOGS=os.path.join(root, "logs"))

    def bootstrap(*extra, quiet=False):
        run = subprocess.run(
            ["bash", os.path.join(HERE, "bootstrap_tpu_vm.sh"), "--tree-only", *extra],
            capture_output=True, text=True, env=env, timeout=1800)
        if run.returncode != 0 and not quiet:
            print("      its output:\n" + run.stdout + run.stderr)
        return run

    def applies(tree_dir, name):
        """Whether `git apply --check` takes patch `name` on `tree_dir`, and git's error."""
        done = subprocess.run(["git", "-C", tree_dir, "apply", "--check",
                               os.path.join(patches, name)], capture_output=True, text=True)
        return done.returncode == 0, done.stderr.strip()

    def stale():
        return sorted(n for n in os.listdir(root) if n.startswith("w.stale-"))

    print("first build")
    first = bootstrap()
    head = git(tree, "rev-parse", "HEAD")
    report(
        first.returncode == 0
        and git(tree, "log", "-1", "--format=%an") == "zhengkezhou1"
        and os.path.exists(os.path.join(tree, "python/sgl_jax/srt/layers/steering.py"))
        and "steering_layer" in open(os.path.join(tree, "python/sgl_jax/srt/models/qwen3.py")).read(),
        f"--tree-only exits {first.returncode} with 877 committed under its author and both "
        f"steering patches applied",
    )
    with open(os.path.join(tree, ".git", "bootstrap-done")) as fp:
        stamp = fp.read()
    hashes = {}
    for name in PATCHES:
        with open(os.path.join(patches, name), "rb") as fp:
            hashes[name] = hashlib.sha256(fp.read()).hexdigest()
    # A branch name moves in the built tree, where git am commits onto it, so the commit it named
    # comes from the clone the tree was built from.
    base = git(source, "rev-parse", f"{SGL_COMMIT}^{{commit}}")
    report(all(f"{digest}  {name}" in stamp for name, digest in hashes.items())
           and stamp.splitlines()[0] == f"commit {SGL_COMMIT}"
           and git(tree, "rev-parse", "HEAD~1") == base,
           f"the stamp names commit {SGL_COMMIT} and the SHA-256 of each patch, and 877 sits on "
           f"{base[:12]}")

    print("rerun, nothing changed")
    again = bootstrap()
    report(again.returncode == 0 and git(tree, "rev-parse", "HEAD") == head and not stale()
           and "cloning" not in again.stdout,
           "the rerun keeps the tree: same HEAD, no clone, nothing moved aside")

    print("rerun after a patch changes")
    path = os.path.join(patches, "sglang-jax-877.patch")
    with open(path) as fp:
        text = fp.read()
    head_end = text.index("\n\n") + 2  # the commit message body starts after the mail headers
    with open(path, "w") as fp:
        fp.write(text[:head_end] + MARKER + "\n\n" + text[head_end:])
    changed = bootstrap()
    moved = stale()
    report(changed.returncode == 0 and len(moved) == 1 and "other patches" in changed.stdout,
           f"the run after the change moved the old tree aside, to {moved}")
    report(MARKER in git(tree, "log", "-1", "--format=%B"),
           "the rebuilt tree carries the changed patch")
    control(bool(moved) and MARKER not in git(os.path.join(root, moved[0]), "log", "-1", "--format=%B"),
            "the tree from before the change, which a stamp keyed on the commit alone would have kept")

    print("--model nemotron3")
    before = stale()
    built = bootstrap("--model", "nemotron3")
    with open(os.path.join(tree, ".git", "bootstrap-done")) as fp:
        stamp = fp.read()
    report(built.returncode == 0 and os.path.exists(os.path.join(tree, NEMOTRON))
           and "models/nemotron3-model.patch" in stamp
           and "models/nemotron3-capture-hook.patch" in stamp
           and len(stale()) == len(before) + 1,
           f"--model nemotron3 exits {built.returncode}, builds nemotron_h.py into a new tree, and"
           f" its stamp names the model and hook patches")
    nemotron_head = git(tree, "rev-parse", "HEAD")

    print("--model nemotron3 again, a second Reproduce on one VM")
    before = stale()
    again = bootstrap("--model", "nemotron3")
    report(again.returncode == 0 and git(tree, "rev-parse", "HEAD") == nemotron_head
           and stale() == before and "cloning" not in again.stdout
           and "model nemotron3" in again.stdout,
           "the second run keeps the Nemotron 3 tree: same HEAD, no clone, nothing moved aside")
    ok, err = applies(tree, "models/nemotron3-model.patch")
    control(not ok, f"the page's old second step, git apply of the model patch on that tree:"
                    f" {err.splitlines()[0] if err else 'applies'}")

    print("--model gpt-oss on the same VM")
    before = stale()
    switched = bootstrap("--model", "gpt-oss")
    moved = [n for n in stale() if n not in before]
    report(switched.returncode == 0 and len(moved) == 1
           and os.path.exists(os.path.join(tree, GPT_OSS))
           and not os.path.exists(os.path.join(tree, NEMOTRON))
           and os.path.exists(os.path.join(root, moved[0], NEMOTRON)),
           f"--model gpt-oss exits {switched.returncode}, builds gpt_oss.py into a tree without"
           f" nemotron_h.py, and moves the Nemotron 3 tree aside to {moved}")
    nested = [n for n in stale() if os.path.exists(os.path.join(root, n, "w"))]
    report(len(stale()) == 3 and not nested,
           f"three builds moved three trees aside, each to a name of its own, in whatever second"
           f" they ran: {stale()}, none inside another")
    ok, err = applies(os.path.join(root, moved[0]) if moved else tree, "models/gpt-oss-model.patch")
    control(not ok and "moe.py" in err,
            f"a reader's second model, the gpt-oss patch on the Nemotron 3 tree:"
            f" {err.splitlines()[0] if err else 'applies'}")

    print("--model with a name upstream/models/ doesn't hold")
    before = stale()
    head = git(tree, "rev-parse", "HEAD")
    unknown = bootstrap("--model", "gpt-5", quiet=True)
    report(unknown.returncode == 2 and "nemotron3" in unknown.stdout and "gpt-oss" in unknown.stdout
           and stale() == before and git(tree, "rev-parse", "HEAD") == head,
           f"--model gpt-5 exits {unknown.returncode}, names the models it holds, and leaves the"
           f" tree alone: {unknown.stdout.strip()}")
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {failures} check(s)")
    raise SystemExit(1)
print("All checks passed.")
