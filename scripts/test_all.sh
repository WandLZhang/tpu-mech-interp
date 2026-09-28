#!/usr/bin/env bash
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

# Every CPU gate in the repo, in one command. No TPU needed. Exits nonzero if any gate fails.
#
#   bash scripts/test_all.sh
#   SGLANG_JAX_REPO=/path/to/sglang-jax bash scripts/test_all.sh   # skip the clone
#
# The gates that need no sglang-jax run first. Then the script clones sglang-jax, and
# scripts/cpu_engine.py builds one patched tree at SGL_COMMIT, the commit every patch in upstream/
# applies to. The gates that start the real engine on CPU share that tree, and the upstream suites
# apply the patches to the clone themselves. SGL_COMMIT (default eb061d8) resolves to one full
# commit id in the clone, and every gate that clones or builds gets that id, so SGL_COMMIT=main
# tests one commit throughout. On a 90-vCPU c3d-highcpu-90 on 2026-09-28 all 46 gates took 2 hours
# 3 minutes. The model tests under upstream/models/ build each model on CPU and compare it with a
# reference; the DeepSeek V4.1 one took 27 minutes of that and the GLM-5.3-Flash one 21.
#
# Every gate writes its output to its own log. A gate that fails prints its whole log, and the
# logs stay on disk, at the path the last line names, when anything failed. When the clone or the
# tree fails, each gate that needs it prints skip, and the run exits nonzero.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PYTHON:-python3}
NEEDS="jax, numpy, flax, optax, torch, transformers, tpu_info"
ENGINE_NEEDS="llguidance, pathwaysutils, PIL, pybase64, requests, setproctitle, uvicorn"
if ! "$PY" -c "import $NEEDS, $ENGINE_NEEDS" 2>/dev/null; then
  cat >&2 <<EOF
$PY can't import $NEEDS and the engine's $ENGINE_NEEDS. Run the gates in a Python 3.12 venv:
  uv venv --python 3.12 .venv && source .venv/bin/activate
  uv pip install --torch-backend=cpu -r scan/requirements.txt -r sae/requirements.txt \\
      -r upstream/models/requirements.txt -r scripts/requirements.txt
EOF
  exit 2
fi
export SGL_COMMIT=${SGL_COMMIT:-eb061d8}
WORK=$(mktemp -d)                          # the sglang-jax clone and the patched tree
LOGS=$(mktemp -d -t test_all-logs.XXXXXX)  # one log per gate
trap 'rm -rf "$WORK"' EXIT

# Gates that start the engine on the patched tree.
TREE_GATES=(
  scripts/test_capture_activations.py
  scripts/test_check_capture.py
  scripts/test_serve_throughput.py
  steering/test_pick_feature.py
  steering/test_compare.py
)
# Gates that need the clone and build their own trees from it.
CLONE_GATES=(
  scripts/test_bootstrap_tree.py
  scripts/test_multihost_exec.py
  upstream/capture-hooks/test_capture_hooks.py
  upstream/test_steering_hook.py
  upstream/test_glm5_tp_sharding.py
  upstream/test_glm5_fp8_accumulate.py
  upstream/models/test_*.py
)

ran=0
failed=0
skipped=0
run() {
  local name=$1
  shift
  ran=$((ran + 1))
  local log
  log="$LOGS/$(printf '%02d' "$ran")-${name//[\/ ]/_}.log"
  local started=$SECONDS
  if "$@" >"$log" 2>&1; then
    echo "  ok    $name ($((SECONDS - started))s)"
  else
    echo "  FAIL  $name ($((SECONDS - started))s). Its whole log, $log:"
    sed 's/^/          /' "$log"
    failed=$((failed + 1))
  fi
}
skip() {
  local gate
  for gate in "$@"; do
    skipped=$((skipped + 1))
    echo "  skip  $gate"
  done
}
needs_sglang() {
  local gate
  for gate in "${TREE_GATES[@]}" "${CLONE_GATES[@]}"; do
    [[ $gate == "$1" ]] && return 0
  done
  return 1
}

echo "repo tests that need no sglang-jax"
for t in scan/test_*.py sae/test_*.py steering/test_*.py scripts/test_*.py; do
  needs_sglang "$t" || run "$t" "$PY" "$t"
done

echo "shell scripts parse"
for s in scripts/*.sh; do
  run "bash -n $s" bash -n "$s"
done

echo "sglang-jax at $SGL_COMMIT"
SRC=$WORK/sglang-jax
before=$failed
if [[ -n "${SGLANG_JAX_REPO:-}" ]]; then
  run "clone $SGLANG_JAX_REPO" git clone -q --shared "$SGLANG_JAX_REPO" "$SRC"
else
  run "clone github.com/sgl-project/sglang-jax" \
    git clone -q https://github.com/sgl-project/sglang-jax "$SRC"
fi
if ((failed == before)); then
  run "$SGL_COMMIT in the clone" git -C "$SRC" rev-parse --verify "$SGL_COMMIT^{commit}"
fi
if ((failed > before)); then
  skip scripts/verify_patches.sh "${TREE_GATES[@]}" "${CLONE_GATES[@]}"
else
  SGL_COMMIT=$(git -C "$SRC" rev-parse --verify "$SGL_COMMIT^{commit}")
  echo "  every gate below takes sglang-jax at $SGL_COMMIT"
  export SGLANG_JAX_REPO=$SRC
  run "scripts/cpu_engine.py (the patched tree the engine gates share)" \
    "$PY" scripts/cpu_engine.py "$WORK/stacked"
  if [[ -f "$WORK/stacked/.git/cpu-engine-patches" ]]; then
    export SGLANG_JAX_TREE=$WORK/stacked
    echo "gates that start the engine on the patched tree"
    for t in "${TREE_GATES[@]}"; do
      run "$t" "$PY" "$t"
    done
  else
    skip "${TREE_GATES[@]}"
  fi
  echo "gates that apply the patches to the clone"
  run "scripts/verify_patches.sh" bash scripts/verify_patches.sh
  for t in "${CLONE_GATES[@]}"; do
    run "$t" "$PY" "$t"
  done
fi

echo
echo "$ran gates, $failed failed, $skipped skipped"
if ((failed)); then
  echo "every gate's log is in $LOGS"
else
  rm -rf "$LOGS"
fi
((failed == 0))
