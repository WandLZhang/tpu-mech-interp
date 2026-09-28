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

# Apply every patch in upstream/ to a clean sglang-jax checkout at SGL_COMMIT, in the order it's
# meant to be used, and report what no longer applies. 21 patches, 17 checks: each model patch that
# ships a separate hook shares its check with the hook, multihost-hidden-states.patch,
# glm5-tp-sharding.patch and glm5-fp8-accumulate.patch share the GLM-5.3 row's check,
# nemotron3-probe.patch gets the Nemotron 3 Ultra row's check, and glm5-next-probe.patch gets the
# GLM-5.3-Flash row's check.
#
#   bash scripts/verify_patches.sh                   # eb061d8, the tree bootstrap_tpu_vm.sh builds
#   SGL_COMMIT=main bash scripts/verify_patches.sh   # upstream main as it stands
#   SGLANG_JAX_REPO=/path/to/clone bash scripts/verify_patches.sh
#
# Upstream moves. A hunk that applies at eb061d8 can stop applying on main with no change on this
# side, so run it with SGL_COMMIT=main before you move the pin.
#
# The clone takes full history, because a `--depth 1` clone holds only the tip of main and has no
# SGL_COMMIT to check out.
#
# A patch that fails prints git's whole error under its FAIL line.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM=https://github.com/sgl-project/sglang-jax
SGL_COMMIT="${SGL_COMMIT:-eb061d8}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [[ -n "${SGLANG_JAX_REPO:-}" && -d "${SGLANG_JAX_REPO}/.git" ]]; then
  BASE="$WORK/base"
  git clone --quiet --shared "$SGLANG_JAX_REPO" "$BASE" || exit 1
else
  BASE="$WORK/base"
  echo "cloning $UPSTREAM (full history)..."
  git clone --quiet "$UPSTREAM" "$BASE" || exit 1
fi
git -C "$BASE" checkout -q "$SGL_COMMIT" || { echo "no commit $SGL_COMMIT in $BASE" >&2; exit 1; }
echo "sglang-jax at $(git -C "$BASE" rev-parse --short HEAD)"
echo

pass=0
fail=0

# fresh [SRC]: a clean copy of SRC, the upstream base by default.
fresh() {
  rm -rf "$WORK/t"
  cp -r "${1:-$BASE}" "$WORK/t"
  git -C "$WORK/t" checkout -q .
  git -C "$WORK/t" clean -qfd
}

# show FILE: print a failure's whole output, indented under its FAIL line.
show() {
  sed 's/^/          /' "$1"
}

# Model patches and capture hooks go on top of the tree bootstrap_tpu_vm.sh builds: 877 and both
# steering patches. STACKED holds that tree once the first block builds it.
STACKED=$BASE

try() {
  local label=$1
  shift
  fresh "$STACKED"
  local ok=1
  for p in "$@"; do
    if ! git -C "$WORK/t" apply "$REPO_ROOT/$p" 2>"$WORK/err"; then
      echo "  FAIL  $label  <- $(basename "$p"):"
      show "$WORK/err"
      ok=0
      break
    fi
  done
  if ((ok)); then
    echo "  ok    $label"
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
  fi
}

echo "capture and steering"
fresh
stack_ok=1
if git -C "$WORK/t" -c user.email=v@local -c user.name=verify \
     am <"$REPO_ROOT/upstream/sglang-jax-877.patch" >"$WORK/am" 2>&1; then
  echo "  ok    sglang-jax-877.patch (git am, author $(git -C "$WORK/t" log -1 --format=%an))"
  pass=$((pass + 1))
else
  echo "  FAIL  sglang-jax-877.patch (git am):"
  show "$WORK/am"
  fail=$((fail + 1))
  git -C "$WORK/t" am --abort >/dev/null 2>&1
  stack_ok=0
fi
for p in upstream/steering-hook.patch upstream/qwen3-steering-hook.patch; do
  if git -C "$WORK/t" apply "$REPO_ROOT/$p" 2>"$WORK/err"; then
    echo "  ok    $(basename "$p") (stacked)"
    pass=$((pass + 1))
  else
    echo "  FAIL  $(basename "$p"):"
    show "$WORK/err"
    fail=$((fail + 1))
    stack_ok=0
  fi
done
# `git add -A` takes the files the steering patch creates, layers/steering.py among them. A
# commit of tracked files alone leaves them untracked, and fresh() then cleans them out of every
# copy the patches below go onto, which the tree bootstrap_tpu_vm.sh builds still holds.
if ((stack_ok)) && git -C "$WORK/t" add -A &&
   git -C "$WORK/t" -c user.email=v@local -c user.name=verify \
     commit -qm "steering hooks" >"$WORK/commit" 2>&1; then
  git -C "$WORK/t" status --porcelain >"$WORK/left"
  if [[ -s "$WORK/left" ]]; then
    echo "  FAIL  the stacked tree holds files its commit left out:"
    show "$WORK/left"
    fail=$((fail + 1))
  fi
  mv "$WORK/t" "$WORK/stacked"
  STACKED="$WORK/stacked"
else
  if ((stack_ok)); then
    echo "  FAIL  committing the stacked tree:"
    show "$WORK/commit"
    fail=$((fail + 1))
  fi
  echo "  (the patches below go on a clean tree instead, because the stack above failed)"
fi

echo "capture hooks"
for p in kimi-linear qwen3_5 deepseek-v3 glm4-moe; do
  try "$p-capture-hook.patch" "upstream/capture-hooks/$p-capture-hook.patch"
done
try "glm5-capture-hook.patch" "upstream/glm5-capture-hook.patch"
# GLM-5.3's row in scripts/multihost_run.sh, in its order.
try "glm5.3 row: multihost, glm5 hook, glm5-tp-sharding.patch, then glm5-fp8-accumulate.patch" \
  "upstream/multihost-hidden-states.patch" "upstream/glm5-capture-hook.patch" \
  "upstream/glm5-tp-sharding.patch" "upstream/glm5-fp8-accumulate.patch"

# gpt-oss, kimi-k3 and nemotron3 all edit layers/moe.py, deepseek-v41 collides with kimi-k3
# and nemotron3, and glm5-next carries the same runner hooks deepseek-v41 does, so one clone
# takes one of them.
echo "model patches, one tree each"
try "deepseek-v41-model.patch" "upstream/models/deepseek-v41-model.patch"
try "glm5-next-model.patch" "upstream/models/glm5-next-model.patch"
try "gpt-oss-model.patch" "upstream/models/gpt-oss-model.patch"
try "inkling-model.patch" "upstream/models/inkling-model.patch"
try "kimi-k3 model then hook" \
  "upstream/models/kimi-k3-model.patch" "upstream/models/kimi-k3-capture-hook.patch"
try "nemotron3 model then hook" \
  "upstream/models/nemotron3-model.patch" "upstream/models/nemotron3-capture-hook.patch"
# Nemotron 3 Ultra's row in scripts/multihost_run.sh, in bootstrap_tpu_vm.sh's order.
try "nemotron3-ultra row: model, hook, multihost, then nemotron3-probe.patch" \
  "upstream/models/nemotron3-model.patch" "upstream/models/nemotron3-capture-hook.patch" \
  "upstream/multihost-hidden-states.patch" "upstream/models/nemotron3-probe.patch"
# GLM-5.3-Flash's row in scripts/multihost_run.sh.
try "glm5.3-flash row: glm5-next model, then glm5-next-probe.patch" \
  "upstream/models/glm5-next-model.patch" "upstream/models/glm5-next-probe.patch"

echo
echo "$pass checks passed, $fail failed"
((fail == 0))
