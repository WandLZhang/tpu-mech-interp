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

# Take one model through a READY multi-host slice: setup, capture check, measure. Run it from a
# machine that can ssh to the slice, in the repo root.
#
#   PROJECT=your-project BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh NODE ZONE MODEL STEP...
#   PROJECT=your-project BUCKET=gs://YOUR_BUCKET bash scripts/multihost_run.sh my-v5p-64 us-east5-a \
#       nemotron3-ultra setup check measure
#
# MODEL is a row of the table below. STEP is any of:
#   setup    scripts/multihost_setup.sh: bootstrap every host, mount the weights, count devices
#   check    copy the model's reference npz and prompt file to host 0, then run check_capture.py
#            across every host with --reference-npz
#   measure  scripts/measure_model.sh on host 0 with MODEL_DIR on the mount and ENGINE_WRAP set to
#            scripts/multihost_exec.sh, so both engine runs span every host
# Each step's whole output lands in $OUT/<model>-<step>.log, and the measure step pulls the
# results folder back. The runner never books or deletes a slice. The rows fit a 32-chip v5p-64:
# check runs at --tp-size 32, and measure at TP=32 MEM_FRAC=0.8 PROMPTS=1000.
#
# While measure runs, its log holds only measure_model.sh's own lines. The engine writes to host 0,
# ~/results/<model>/capture_off.out and capture_on.out, and every rank's log to ~/multihost-logs/.
# Reach host 0 with `gcloud compute tpus tpu-vm ssh NODE --zone=ZONE --project=PROJECT --worker=0`.
# A large model's routed experts load in a pass that prints nothing: Inkling's took about 2 hours
# on a v5p-64 (2026-09-29), and the fastest hosts then wait on the slowest. Every ssh the runner
# opens shares a ControlMaster socket in ~/.ssh/cm, kept open for 60 minutes.
#
# The references come from a CPU VM beforehand, with the same prompt file, and sit in
# $BUCKET/refs/ as ref-<model>.npz and prompts-<model>.jsonl:
#   python3 scripts/check_capture.py --model-path <local copy> --offload-folder /mnt/data/off \
#     --prompts-file prompts-<model>.jsonl --num-prompts 1 --reference-only ref-<model>.npz
# plus --trust-remote-code for kimi-k3 and --deepseek-inference for deepseek-v41.
#
#   BUCKET    gs:// folder that holds each checkpoint as <model>/ and the references as refs/
#             (required)
#   PROJECT   the slice's GCP project (default: gcloud's)
#   SSH_USER  login name on the hosts (default: your OS Login user)
#   OUT       local folder for logs and results (default ~/tpu-runs/<NODE>)
#   EXTRA_EARGS       engine keywords for one run, such as disable_overlap_schedule=True
#   SLICE_SSH_CONFIG  an ssh config file for every ssh the runner opens; see below
#
# The hosts read the bucket, so create the slice with the cloud-platform scope, for instance
# TPU_CREATE_FLAGS="--scopes=https://www.googleapis.com/auth/cloud-platform" for
# scripts/spray_tpu_spot.sh. The hosts also reach each other on their internal IPs, which the
# VPC has to allow; scripts/multihost_setup.sh says how.
set -uo pipefail
(($# >= 4)) || { sed -n '16,57p' "$0"; exit 2; }
NODE=$1; ZONE=$2; MODEL=$3; shift 3
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
[[ -n "$PROJECT" ]] || { echo "set PROJECT= or run: gcloud config set project YOUR_PROJECT" >&2; exit 1; }
# gcloud answers a TPU call in a project without the TPU API by offering to turn it on there. A
# stranger's run hit that offer in gcloud's default project, not the slice's (2026-09-29).
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
BUCKET=${BUCKET:-}
[[ -n "$BUCKET" ]] || { echo "set BUCKET= to the gs:// folder that holds <model>/ and refs/" >&2; exit 1; }
BUCKET=${BUCKET%/}
OUT=${OUT:-$HOME/tpu-runs/$NODE}
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/runner.log"; }

# model | bootstrap --model | EXTRA_PATCHES | capture slot for measure | extra engine args
# kimi-k3 serves one v5p-64 at --tp-size 32 --dp-size 1 with its routed experts resident in MXFP4,
# the default for its checkpoint. Its MLA layers need a page size above 1, any linear-recurrent
# model needs --disable-radix-cache, and --ep-size 32 puts 28 whole experts on each chip. Slot 46
# is the stream entering the middle of its 93 layers.
# deepseek-v41 serves at --tp-size 32 --dp-size 1 --ep-size 32 with 12 routed experts a chip at
# BF16, the Engram tables row-sharded over all 32 chips in FP8, attention replicated. It keeps its
# caches in its own per-request pool, so it needs --disable-radix-cache, and that pool is sized by
# --context-length: 4096 holds the capture's 1,323-token joined prompt at 9 slots for 0.15 GiB a
# chip. Slot 20 is the collapsed attention input of layer 20, the first decoder layer of 40. Its
# weights sit under deepseek-v4.1-flash/ in the bucket, and its reference comes from the
# checkpoint's own inference/model.py through --deepseek-inference.
# glm5.3-flash serves at --tp-size 32 --dp-size 1 --ep-size 32 with 9 routed experts a chip,
# widened from FP8 to BF16 at load, attention, KDA and mHC replicated. Its KDA state, MLA latents
# and indexer keys sit in its own per-request pool, so it needs --disable-radix-cache, and
# --context-length sizes that pool. Slot 22 is the four mHC streams entering layer 22 of 45,
# 16,384 values a token. Its reference comes from transformers' own glm5_next, streamed.
# glm5.3 serves at --tp-size 32. Its shared expert's down_proj has 16 FP8 input blocks, too few to
# tile 32 chips, so it runs with a replicated reduce axis; glm5-tp-sharding.patch reshards its
# input and places its weight to match, where the unpatched tree dies in the first precompile.
# --ep-size 32 keeps each routed expert whole on its chip. At --ep-size 1 each chip holds 64 of an
# expert's 2048 FP8 columns, half a 128-wide scale block. The same split on the tiny CPU model
# (--tp-size 4, 64 of 256) turned the stream to NaN with the interpreted GMM v1 kernel on
# 2026-09-27; a chip runs GMM v2, which that run didn't cover.
# inkling mounts with RANGE_CACHE=false. Each host reads its eighth of every expert file once, and a
# whole-file cache of reads past byte 0 pulled 31 TB into host 0 on 2026-09-27. A slice mounted with
# the cache on needs REMOUNT=1 in the setup step's environment to take the change.
# glm5-fp8-accumulate.patch keeps the block-wise FP8 matmul's accumulator, its scales and the
# row-parallel reductions in float32. The unpatched kernel sums in BF16, and the second check on
# the v5p-64 (2026-09-27 08:48Z) failed layers 10 to 13 at 2.16 to 2.75 times the BF16 floor.
# nemotron3-ultra builds nemotron3-probe.patch too. Its SGL_PROBE_* switches stay off unless set;
# each one swaps a piece of the LatentMoE path for plain JAX, so one engine start per switch shows
# whether that piece moves the check (upstream/models/README.md).
# glm5.3-flash builds glm5-next-probe.patch the same way, for SGL_PROBE_MOE=dense alone.
WEIGHTS=""
RANGE_CACHE=true
case "$MODEL" in
  nemotron3-ultra) BOOT=nemotron3; EXTRA="models/nemotron3-probe.patch"; SLOT=54; EARGS="ep_size=32" ;;
  glm5.3)          BOOT=none;      EXTRA="glm5-capture-hook.patch glm5-tp-sharding.patch glm5-fp8-accumulate.patch"
                   SLOT=39; EARGS="attention_backend=dsa_sparse ep_size=32" ;;
  inkling)         BOOT=inkling;   EXTRA="";                      SLOT=33; EARGS="attention_backend=native disable_radix_cache=True"
                   RANGE_CACHE=false ;;
  kimi-k3)         BOOT=kimi-k3;   EXTRA="";                      SLOT=46; EARGS="page_size=128 disable_radix_cache=True ep_size=32" ;;
  deepseek-v41)    BOOT=deepseek-v41; EXTRA="";                   SLOT=20; EARGS="disable_radix_cache=True ep_size=32 context_length=4096"
                   WEIGHTS=deepseek-v4.1-flash ;;
  glm5.3-flash)    BOOT=glm5-next; EXTRA="models/glm5-next-probe.patch"; SLOT=22; EARGS="disable_radix_cache=True ep_size=32 context_length=4096" ;;
  *) echo "no row for $MODEL" >&2; exit 2 ;;
esac
# EXTRA_EARGS adds engine keywords for one run, e.g. EXTRA_EARGS="disable_overlap_schedule=True".
EARGS="$EARGS ${EXTRA_EARGS:-}"
# Kimi K3 ships its tokenizer as Python code, and the reference's token ids must match the engine's.
CFLAGS=""; [ "$MODEL" = kimi-k3 ] && CFLAGS="--trust-remote-code"
# DeepSeek V4.1's reference runs the checkpoint's inference/model.py. The check reads the npz, so
# the flag only marks which reference the npz came from; the reference run needs it.
[ "$MODEL" = deepseek-v41 ] && CFLAGS="--deepseek-inference"
WEIGHTS=${WEIGHTS:-$MODEL}
# sglang-jax's scheduler watchdog kills the engine when one step runs past watchdog_timeout (300 s
# by default). Kimi K3's first real request compiled past it on the v5p-64 after 10.5 min of extend
# and 8 min of decode precompile (2026-09-27 21:44Z), so every row raises it.
EARGS="watchdog_timeout=3600${EARGS:+ $EARGS}"
# Capture across hosts needs the scheduler to gather hidden states before it slices them.
EXTRA="multihost-hidden-states.patch${EXTRA:+ $EXTRA}"
MOUNT=/mnt/weights-$MODEL
# SLICE_SSH_CONFIG names an ssh config file that replaces "-o ProxyCommand=none" for every ssh this
# runner and multihost_setup.sh open, for instance to reach hosts whose external IPs stop answering
# through host 0 with ProxyJump (seen on 2026-09-28). Unset, nothing changes.
if [ -n "${SLICE_SSH_CONFIG:-}" ]; then SSHX=(-F "$SLICE_SSH_CONFIG"); else SSHX=(-o ProxyCommand=none); fi
SSHO=(-i "$HOME/.ssh/google_compute_engine" -o IdentitiesOnly=yes "${SSHX[@]}"
      -o CanonicalizeHostname=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o LogLevel=ERROR -o ServerAliveInterval=30 -o ControlMaster=auto
      -o "ControlPath=$HOME/.ssh/cm/%C" -o ControlPersist=60m)
U=${SSH_USER:-$(gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)' 2>/dev/null)}
[[ -n "$U" ]] || { echo "set SSH_USER= to your login name on the hosts" >&2; exit 1; }
# The engine reads through a symlink on the root disk, which skips sglang-jax's whole-checkpoint
# warm-up read on every host (see multihost_setup.sh step 5).
MPATH=/home/$U/weights/weights-$MODEL
host0() {
  gcloud compute tpus tpu-vm describe "$NODE" --zone="$ZONE" --project="$PROJECT" \
    --format='value(networkEndpoints[0].accessConfig.externalIp)'
}
H0=$(host0) || { say "can't describe $NODE in $ZONE, project $PROJECT; set PROJECT= to the slice's project"; exit 1; }
on0() { ssh "${SSHO[@]}" "$U@$H0" "$@"; }

for step in "$@"; do
  say "$MODEL $step on $NODE ($H0)"
  case "$step" in
    setup)
      OUT="$OUT/setup-$MODEL" EXTRA_PATCHES="$EXTRA" MOUNT="$MOUNT" SSH_USER=$U RANGE_CACHE=$RANGE_CACHE \
        SSH_EXTRA="${SSHX[*]} -o CanonicalizeHostname=no" PROJECT=$PROJECT \
        bash scripts/multihost_setup.sh "$NODE" "$ZONE" "$BOOT" "$BUCKET/$WEIGHTS" \
        >"$OUT/$MODEL-setup.log" 2>&1
      rc=$? ;;
    check)
      on0 "mkdir -p ~/refs && gcloud storage cp $BUCKET/refs/ref-$MODEL.npz $BUCKET/refs/prompts-$MODEL.jsonl ~/refs/" \
        >"$OUT/$MODEL-refcopy.log" 2>&1 || { say "no reference in $BUCKET/refs for $MODEL"; cat "$OUT/$MODEL-refcopy.log"; exit 1; }
      eargs=""; for kv in $EARGS; do eargs+=" --engine-arg $kv"; done
      # Host 0 keeps the capture. A later reference re-gates it without the slice:
      #   check_capture.py --model-path <tokenizer dir> --prompts-file prompts-<model>.jsonl \
      #     --num-prompts 1 --reference-npz ref-<model>.npz --capture-npz <model>-check-capture.npz
      on0 "cd ~/repo && source ~/.tpu_env && export HF_HUB_OFFLINE=1 && mkdir -p ~/results &&
        bash scripts/multihost_exec.sh python3 -u scripts/check_capture.py --model-path $MPATH \
          --tp-size 32 --prompts-file ~/refs/prompts-$MODEL.jsonl --num-prompts 1 \
          --reference-npz ~/refs/ref-$MODEL.npz --save-capture ~/results/$MODEL-check-capture.npz \
          $CFLAGS --engine-arg mem_fraction_static=0.8 $eargs" \
        >"$OUT/$MODEL-check.log" 2>&1
      rc=$? ;;
    measure)
      on0 "cd ~/repo && source ~/.tpu_env && TP=32 MEM_FRAC=0.8 PROMPTS=1000 MODEL_DIR=$MPATH \
        ENGINE_WRAP='bash scripts/multihost_exec.sh' ENGINE_ARGS='$EARGS' \
        bash scripts/measure_model.sh $MPATH $SLOT ~/results/$MODEL" >"$OUT/$MODEL-measure.log" 2>&1
      rc=$?
      mkdir -p "$OUT/results-$MODEL"
      scp "${SSHO[@]}" -q -r "$U@$H0:results/$MODEL/." "$OUT/results-$MODEL/" ;;
    *) say "unknown step $step"; exit 2 ;;
  esac
  say "$MODEL $step exited $rc; whole log $OUT/$MODEL-$step.log"
  ((rc == 0)) || exit "$rc"
done
