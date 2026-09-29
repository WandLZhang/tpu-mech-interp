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

# Take a bare Cloud TPU VM to a working sglang-jax with capture and steering.
#
#   bash scripts/bootstrap_tpu_vm.sh
#   bash scripts/bootstrap_tpu_vm.sh --model nemotron3   # plus a model from upstream/models/
#   PATCHES=/path/to/upstream SGL_COMMIT=eb061d8 bash scripts/bootstrap_tpu_vm.sh
#   EXTRA_PATCHES=glm5-capture-hook.patch bash scripts/bootstrap_tpu_vm.sh   # an upstream model's hook
#   bash scripts/bootstrap_tpu_vm.sh --tree-only    # build the patched tree and stop
#
# Run it on the TPU VM, not on your workstation. It's safe to run twice: every step checks for
# its own result first. The tree counts as built only for the commit and the patches it was built
# from, so after a patch in upstream/ changes, the next run moves the old tree aside and builds a
# new one.
#
# --model NAME adds upstream/models/NAME-model.patch, then NAME-capture-hook.patch where one
# exists: deepseek-v41, glm5-next, gpt-oss, inkling, kimi-k3 or nemotron3. The model patches
# count toward the tree, so a second run with the same NAME keeps the tree, and a run with
# another NAME, or none, moves the tree aside and builds the one it asks for. That's how one VM
# moves from one model to another: gpt-oss, kimi-k3 and nemotron3 all edit layers/moe.py,
# deepseek-v41 collides with kimi-k3 and nemotron3, glm5-next carries deepseek-v41's runner
# hooks, and one tree takes one of them.
#
# Each install writes its output to a log in $LOGS (default ~/bootstrap-logs). A step that fails
# prints its whole log and stops. SGLANG_JAX_REPO clones from a local sglang-jax clone instead of
# GitHub. --tree-only skips everything but the tree, which is how scripts/test_bootstrap_tree.py
# drives it on a workstation.
set -uo pipefail

USAGE="usage: bash scripts/bootstrap_tpu_vm.sh [--tree-only] [--model deepseek-v41|glm5-next|gpt-oss|inkling|kimi-k3|nemotron3]"
TREE_ONLY=0
MODEL=""
while (($#)); do
  case "$1" in
    --tree-only) TREE_ONLY=1 ;;
    --model)
      [[ $# -ge 2 && -n "$2" ]] || { echo "$USAGE" >&2; exit 2; }
      MODEL=$2
      shift ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
  shift
done

SGL_COMMIT="${SGL_COMMIT:-eb061d8}"
VENV="${VENV:-$HOME/v312}"
TREE="${TREE:-$HOME/w}"
PATCHES="${PATCHES:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../upstream" && pwd)}"
SOURCE="${SGLANG_JAX_REPO:-https://github.com/sgl-project/sglang-jax}"
LOGS="${LOGS:-$HOME/bootstrap-logs}"

log() { echo "### $(date -u +%H:%M:%S) $*"; }
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$LOGS" || exit 1

# run_logged NAME CMD...: run CMD with its output in $LOGS/NAME.log. On failure print the whole
# log and its path, and stop.
run_logged() {
  local name=$1
  shift
  if ! "$@" >"$LOGS/$name.log" 2>&1; then
    log "$name failed. Its whole log, $LOGS/$name.log:"
    cat "$LOGS/$name.log"
    exit 1
  fi
}

for p in sglang-jax-877 steering-hook qwen3-steering-hook; do
  [[ -f "$PATCHES/$p.patch" ]] ||
    { log "no $PATCHES/$p.patch; point PATCHES at this repo's upstream/"; exit 1; }
done

# The model patches, paths relative to $PATCHES, in the order they apply.
MODEL_PATCHES=()
if [[ -n "$MODEL" ]]; then
  if [[ "$MODEL" == */* || ! -f "$PATCHES/models/$MODEL-model.patch" ]]; then
    known=$(cd "$PATCHES/models" 2>/dev/null && ls -- *-model.patch 2>/dev/null |
            sed 's/-model\.patch$//' | tr '\n' ' ')
    log "no model patch for --model $MODEL in $PATCHES/models; it holds: $known"
    exit 2
  fi
  MODEL_PATCHES=("models/$MODEL-model.patch")
  [[ -f "$PATCHES/models/$MODEL-capture-hook.patch" ]] &&
    MODEL_PATCHES+=("models/$MODEL-capture-hook.patch")
fi
# EXTRA_PATCHES adds hooks for models that ship upstream, paths relative to $PATCHES, such as
# EXTRA_PATCHES=glm5-capture-hook.patch for GLM-5 and GLM-5.3. They count toward the tree too.
for p in ${EXTRA_PATCHES:-}; do
  [[ "$p" != /* && -f "$PATCHES/$p" ]] || { log "no patch $p in $PATCHES"; exit 2; }
  MODEL_PATCHES+=("$p")
done

if ((!TREE_ONLY)); then
  # Writeback throttling on the boot disk holds every read behind the flush of a big write. On a
  # v5e host the disk flushes at about 11 MB/s, and a capture's closing fsync once sat 308 s in
  # rq_qos_wait until the throttle came off. The root disk is sda on v5e and nvme0n1 on v5p.
  ROOTDEV=$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)" 2>/dev/null)
  if [[ -n "$ROOTDEV" && -e "/sys/block/$ROOTDEV/queue/wbt_lat_usec" ]]; then
    echo 0 | sudo tee "/sys/block/$ROOTDEV/queue/wbt_lat_usec" >/dev/null &&
      log "writeback throttle off on $ROOTDEV"
  fi
  # The image upgrades its packages in the first hour after boot. While a host replaced
  # google-guest-agent, sshd refused every login for minutes, and a restarted logind let RemoveIPC
  # delete everything this user kept in /dev/shm: the weights cache, HF_HOME and the compile cache
  # (v5p-64, 2026-09-28). Linger keeps the user's files. The timers stay off until the next boot;
  # an upgrade that already started runs to its end.
  sudo loginctl enable-linger "$USER" && sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer &&
    log "linger on for $USER, apt's daily timers off"
fi

# A tree counts as built only when this stamp names the commit and the SHA-256 of each patch it
# was built from, model patches included. The build writes it after the last patch. It sits in
# .git, where git status and git clean don't see it.
DONE=".git/bootstrap-done"
WANT="$(printf 'commit %s\n' "$SGL_COMMIT"
        cd "$PATCHES" && sha256sum sglang-jax-877.patch steering-hook.patch qwen3-steering-hook.patch \
          "${MODEL_PATCHES[@]}")" ||
  { log "can't hash the patches in $PATCHES"; exit 1; }
HAVE=""
[[ -f "$TREE/$DONE" ]] && HAVE="$(cat "$TREE/$DONE")"
if [[ "$HAVE" != "$WANT" ]]; then
  # TREE can be your own clone. The script moves a tree it can't vouch for aside and never
  # deletes it, so a patch applied by hand survives in the old copy. So does the last model's
  # tree when --model names another one.
  if [[ -e "$TREE" ]]; then
    stale="$TREE.stale-$(date -u +%Y%m%dT%H%M%S)"
    # Two trees moved aside in one second would share the name, and mv would put the second one
    # inside the first.
    base=$stale
    n=1
    while [[ -e "$stale" ]]; do stale="$base.$((n++))"; done
    if [[ -n "$HAVE" ]]; then
      log "$TREE was built from other patches. It holds:"
      echo "$HAVE"
      log "and this run wants:"
      echo "$WANT"
    else
      log "$TREE has no finished build of these patches at $SGL_COMMIT"
    fi
    log "moving it to $stale. A patch you applied there by hand stays in that copy; --model NAME" \
      "builds a model from upstream/models/ into the new tree"
    mv "$TREE" "$stale" || exit 1
  fi
  # Build in $TREE.partial, which the script owns, and move it to $TREE after the stamp.
  NEW="$TREE.partial"
  rm -rf "$NEW"
  log "cloning sglang-jax at $SGL_COMMIT from $SOURCE"
  git clone -q "$SOURCE" "$NEW" || exit 1
  git -C "$NEW" checkout -q "$SGL_COMMIT" || exit 1
  git -C "$NEW" -c user.email=bootstrap@localhost -c user.name=bootstrap \
    am <"$PATCHES/sglang-jax-877.patch" || { log "877 failed"; exit 1; }
  git -C "$NEW" apply "$PATCHES/steering-hook.patch" || { log "steering failed"; exit 1; }
  git -C "$NEW" apply "$PATCHES/qwen3-steering-hook.patch" || { log "qwen3 steering failed"; exit 1; }
  for p in "${MODEL_PATCHES[@]}"; do
    git -C "$NEW" apply "$PATCHES/$p" || { log "$p failed"; exit 1; }
  done
  printf '%s\n' "$WANT" >"$NEW/$DONE" || exit 1
  mv "$NEW" "$TREE" || exit 1
  log "patches applied, author $(git -C "$TREE" log -1 --format=%an)${MODEL:+, model $MODEL}"
else
  log "$TREE holds a finished build of these patches at $SGL_COMMIT${MODEL:+, model $MODEL}"
fi
((TREE_ONLY)) && exit 0

# The stock runtime is Python 3.10 with no numpy and no jax. sglang-jax needs 3.12.
if [[ ! -x "$HOME/.local/bin/uv" ]]; then
  log "installing uv"
  run_logged uv-install bash -o pipefail -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
[[ -d "$VENV" ]] || run_logged venv uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "python $(python3 -V)"

# Install through the engine's own tpu extra, which pins jax 0.11.1. jax 0.11.2 exports
# hijax.HiPrim while flax asks for hijax.HiPrimitive, so every flax import dies under it.
log "installing sglang-jax[tpu]"
run_logged sglang-jax-install uv pip install -e "$TREE/python[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# flax 0.10.7 is too old the other way round: it asks jax for mutable_array, which 0.11.1 lacks.
run_logged flax-install uv pip install "flax==0.12.9"
# transformers builds the Gemma and Qwen processors through torchvision, even for text-only work.
run_logged torch-install uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# scripts/peak_hbm.py reads HBM through tpu-info's Python API, from outside the engine's process.
run_logged tpu-info-install uv pip install tpu-info
# sglang-jax pins transformers 5.12, which doesn't know `inkling_mm_model`, `glm5_next` or
# `deepseek_v41`; an Inkling check on a v5p-64 stopped in AutoConfig on 5.12.1 (2026-09-27). The
# CPU gates run the engine, every model patch included, on 5.17.0.
case "$MODEL" in
  inkling | glm5-next | deepseek-v41)
    run_logged transformers-install uv pip install "transformers==5.17.0" ;;
esac
log "installs done; their logs are in $LOGS"

mkdir -p /dev/shm/jaxcache

cat >"$HOME/.tpu_env" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
source $VENV/bin/activate
# There's no compilation cache by default, so a killed run throws away its XLA work. It lives in
# /dev/shm for the same reason the weights do: compiled programs run to gigabytes, and the boot
# disk is slow to take them.
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jaxcache
# Xet pulls every byte of a large repo and then never finalizes: it leaves
# <sha>.<tag>.incomplete at the published size and writes a snapshot symlink to a blob it never
# materialized. Plain HTTP is slower per connection and finishes.
export HF_HUB_DISABLE_XET=1
# The boot disk flushes at about 11 MB/s, so a model download leaves a writeback backlog that
# stalls every later read and write for most of an hour. /dev/shm is RAM and has no writeback.
export HF_HOME=/dev/shm/hf
EOF
mkdir -p /dev/shm/hf
log "wrote ~/.tpu_env; source it before any run"

log "smoke test"
BOOTSTRAP_TREE="$TREE" python3 - <<'PY' || { log "smoke test failed"; exit 1; }
import os
import sys
import jax, flax, torch, torchvision
print(f"  jax {jax.__version__}  flax {flax.__version__}  torch {torch.__version__}  tv {torchvision.__version__}")
print(f"  DEVICES {len(jax.devices())} {jax.devices()[0].device_kind}")
import sgl_jax
from sgl_jax.srt.server_args import ServerArgs
where = os.path.realpath(sgl_jax.__file__)
tree = os.path.realpath(os.environ["BOOTSTRAP_TREE"])
print(f"  sgl_jax imports from {where}")
if not where.startswith(tree + os.sep):
    sys.exit(f"  sgl_jax imports from {where}, not from the tree this script built in {tree}")
fields = ("enable_return_hidden_states", "return_hidden_states_layers", "enable_steering",
          "steering_bank", "steering_layer")
for f in fields:
    print(f"  ServerArgs.{f} = {getattr(ServerArgs, f, '<MISSING>')}")
missing = [f for f in fields if not hasattr(ServerArgs, f)]
if missing:
    sys.exit(f"  the tree isn't patched: ServerArgs lacks {', '.join(missing)}")
PY

# A model's weights stay in /dev/shm, which is RAM, after the run that fetched them, and the next
# model's fetch lands beside them. Say what's there.
if compgen -G "/dev/shm/hf/hub/models--*" >/dev/null; then
  log "weights already in /dev/shm/hf/hub, which is RAM, with $(df -h --output=avail /dev/shm |
    sed -n 2p | tr -d ' ') free:"
  du -sh /dev/shm/hf/hub/models--*
  echo "  A rerun of the same model reads its weights from here, so keep those. Before another"
  echo "  model's fetch, delete a model you're done with: rm -rf /dev/shm/hf/hub/models--ORG--NAME"
fi

log "done"
cat <<'EOF'

Next:
  source ~/.tpu_env
  python3 scripts/capture_activations.py --model-path MODEL --prompts prompts.jsonl \
      --out /dev/shm/caps --layers 20 --tp-size <chips> --batch-size 8

One process holds the TPU at a time. After killing a run, wait for /dev/vfio to free and
rm /tmp/libtpu_lockfile, or the next start reports the chip already in use. Anything that builds
an Engine must sit in a real file behind `if __name__ == "__main__":`, because the engine re-runs
the main module in subprocesses.
EOF
