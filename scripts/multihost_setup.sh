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

# Take a READY multi-host TPU slice to one sglang-jax engine across all its hosts.
#
#   bash scripts/multihost_setup.sh NODE ZONE MODEL GCS_WEIGHTS
#   bash scripts/multihost_setup.sh my-v5p-64 us-east5-a nemotron3 gs://my-bucket/nemotron3-ultra
#
# Run it from a workstation or CPU VM that can ssh to the slice's external IPs, from the repo root.
# MODEL is what scripts/bootstrap_tpu_vm.sh --model takes (deepseek-v41, glm5-next, gpt-oss,
# inkling, kimi-k3, nemotron3), or "none" for an upstream model. GCS_WEIGHTS is a checkpoint
# directory in a bucket in the slice's region. It prints a line per step and ends with a DEVICES
# line per host. Every host has to report the slice's whole chip count, or it exits 1.
#
# Steps, each on every host in parallel:
#   1. list the hosts in worker order and write $OUT/hosts.tsv: worker, internal IP, external IP
#   2. copy the repo's shipped directories to ~/repo
#   3. run scripts/bootstrap_tpu_vm.sh --model MODEL, each host's output in $OUT/bootstrap-wN.log
#   4. give host 0 a key every host accepts (~/.ssh/peer) and the host list (~/multihost_hosts),
#      which scripts/multihost_exec.sh reads
#   5. install gcsfuse and mount GCS_WEIGHTS read-only at $MOUNT with a file cache in /dev/shm. A checkpoint
#      this size fits neither a host's RAM nor its boot disk, and sglang-jax reads each host's
#      slice of a sharded tensor, so every host streams what it needs
#   6. from host 0, run jax.distributed.initialize() on every host through multihost_exec.sh and
#      print each host's process index and device counts
#
# The hosts talk to each other on their internal IPs. The VPC has to allow that: the default
# network's default-allow-internal rule covers 10.128.0.0/9 only if it was never edited.
#
#   PROJECT   GCP project (default: gcloud's)
#   OUT       local folder for logs (default /tmp/multihost-NODE)
#   MOUNT     where every host mounts the weights (default /mnt/weights-<last part of
#             GCS_WEIGHTS>, so a second model on the same slice gets its own mount)
#   SSH_USER  login name on the hosts (default: your OS Login user)
#   SSH_KEY   private key for the hosts (default ~/.ssh/google_compute_engine)
#   SSH_EXTRA extra ssh options, such as "-o ProxyCommand=none"
#   EXTRA_PATCHES  passed to bootstrap_tpu_vm.sh, such as glm5-capture-hook.patch with MODEL none
#   RANGE_CACHE    true (default) or false: whether a read that starts past a file's first byte
#             pulls the whole file into the cache. REMOUNT=1 applies a change to a mounted slice.
set -uo pipefail

(($# == 4)) || { echo "usage: bash scripts/multihost_setup.sh NODE ZONE MODEL GCS_WEIGHTS" >&2; exit 2; }
NODE=$1; ZONE=$2; MODEL=$3; WEIGHTS=${4%/}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
OUT=${OUT:-/tmp/multihost-$NODE}
MOUNT=${MOUNT:-/mnt/weights-${WEIGHTS##*/}}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/google_compute_engine}
SSH_USER=${SSH_USER:-$(gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)' 2>/dev/null)}
mkdir -p "$OUT" "$HOME/.ssh/cm"
log() { echo "### setup $(date -u +%H:%M:%S) $*"; }
# shellcheck disable=SC2206
SSH=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ControlMaster=auto
     -o "ControlPath=$HOME/.ssh/cm/%C" -o ControlPersist=60m ${SSH_EXTRA:-})
on() { local ip=$1; shift; "${SSH[@]}" "$SSH_USER@$ip" "$@"; }

# 1. Hosts in worker order.
gcloud compute tpus tpu-vm describe "$NODE" --zone="$ZONE" --project="$PROJECT" \
    --format='value[delimiter="\n"](networkEndpoints[].ipAddress,networkEndpoints[].accessConfig.externalIp)' \
    >"$OUT/endpoints.txt" 2>&1 || { log "can't describe $NODE:"; cat "$OUT/endpoints.txt"; exit 1; }
# gcloud joins the two lists with a tab, not the delimiter, so the last internal IP and the first
# external IP share a line (v5p-64, 2026-09-27). Split on tabs too.
mapfile -t F < <(tr '\t' '\n' < "$OUT/endpoints.txt" | sed '/^$/d')
N=$((${#F[@]} / 2))
((N >= 1)) || { log "no endpoints for $NODE:"; cat "$OUT/endpoints.txt"; exit 1; }
INT=("${F[@]:0:N}"); EXT=("${F[@]:N:N}")
: >"$OUT/hosts.tsv"
for ((w = 0; w < N; w++)); do printf 'w%d\t%s\t%s\n' "$w" "${INT[$w]}" "${EXT[$w]}" >>"$OUT/hosts.tsv"; done
log "$N hosts:"; cat "$OUT/hosts.tsv"

# Run "$@" on every host at once; each host's output goes to $OUT/$TAG-wN.log.
each() {
  local tag=$1 w rc=0; shift
  local pids=()
  for ((w = 0; w < N; w++)); do on "${EXT[$w]}" "$@" >"$OUT/$tag-w$w.log" 2>&1 & pids+=($!); done
  for ((w = 0; w < N; w++)); do
    if ! wait "${pids[$w]}"; then
      log "$tag failed on w$w. Its whole log, $OUT/$tag-w$w.log:"; cat "$OUT/$tag-w$w.log"; rc=1
    fi
  done
  return $rc
}

# 2. The repo's shipped directories.
tar czf "$OUT/repo.tgz" scripts upstream scan sae steering models || exit 1
for ((w = 0; w < N; w++)); do
  on "${EXT[$w]}" 'mkdir -p ~/repo && tar xzf - -C ~/repo' <"$OUT/repo.tgz" &
done
wait
log "repo copied"

# 3. Bootstrap every host.
args=""; [ "$MODEL" != none ] && args="--model $MODEL"
each bootstrap "cd ~/repo && EXTRA_PATCHES='${EXTRA_PATCHES:-}' bash scripts/bootstrap_tpu_vm.sh $args" || exit 1
log "bootstrap done on $N hosts"

# 4. Host 0 reaches every host.
on "${EXT[0]}" 'test -f ~/.ssh/peer || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/peer' || exit 1
PUB=$(on "${EXT[0]}" 'cat ~/.ssh/peer.pub') || exit 1
# Under OS Login a host that nobody has logged into holds no ~/.ssh yet (v5p-64 w1 to w7, 2026-09-27).
each peerkey "mkdir -p ~/.ssh && chmod 700 ~/.ssh && { grep -qxF '$PUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> ~/.ssh/authorized_keys; } && chmod 600 ~/.ssh/authorized_keys" || exit 1
printf '%s\n' "${INT[@]}" | on "${EXT[0]}" 'cat > ~/multihost_hosts'
log "host 0 holds ~/multihost_hosts and ~/.ssh/peer"

# 5. The weights, read-only, on every host, through a file cache in /dev/shm. With no cache,
# sglang-jax's MoE loader re-read each expert layer's files through gcsfuse: Nemotron 3 Ultra took
# about 2 minutes a layer at 950 MB/s per host, about 114 GB read for a 10 GB layer (v5p-64,
# 2026-09-27), which is over 3 hours for 96 layers.
# RANGE_CACHE=false suits a loader where each host reads a slice of every file once. Inkling's does:
# its experts shard over the tensor axis, so every host reads an eighth of every expert file. With
# RANGE_CACHE=true each host pulled whole files through a cache far smaller than the 1.9 TB
# checkpoint, and host 0 took in 31 TB for one load (v5p-64, 2026-09-27).
CACHE_DIR=${CACHE_DIR:-/dev/shm/gcsfuse-cache}
RANGE_CACHE=${RANGE_CACHE:-true}
CACHE_MB=${CACHE_MB:-120000}  # /dev/shm is 59 GB on a v5p host until it's remounted larger; the host has 440 GB
BUCKET=${WEIGHTS#gs://}; BUCKET=${BUCKET%%/*}
PREFIX=${WEIGHTS#gs://$BUCKET}; PREFIX=${PREFIX#/}
each gcsfuse "set -e
  if ! command -v gcsfuse >/dev/null; then
    echo \"deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-\$(lsb_release -c -s) main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null
    # An upgrade the image started before bootstrap stopped its timers holds dpkg's lock.
    sudo apt-get -qq -o DPkg::Lock::Timeout=1800 update &&
      sudo apt-get -qq -o DPkg::Lock::Timeout=1800 install -y gcsfuse
  fi
  sudo mkdir -p $MOUNT && sudo chown \$USER $MOUNT
  if [ \"${REMOUNT:-0}\" = 1 ] && mountpoint -q $MOUNT; then fusermount -u $MOUNT; fi
  case $CACHE_DIR in /dev/shm/*) sudo mount -o remount,size=$((CACHE_MB / 1024 + 20))G /dev/shm ;; esac
  mkdir -p $CACHE_DIR
  mountpoint -q $MOUNT || gcsfuse -o ro --implicit-dirs ${PREFIX:+--only-dir $PREFIX} --file-mode=444 --dir-mode=555 \\
    --cache-dir=$CACHE_DIR --file-cache-max-size-mb=$CACHE_MB --file-cache-cache-file-for-range-read=$RANGE_CACHE \\
    --file-cache-enable-parallel-downloads=true $BUCKET $MOUNT
  mkdir -p ~/weights && ln -sfn $MOUNT ~/weights/\$(basename $MOUNT)
  echo \"\$(ls $MOUNT | wc -l) entries, \$(gcsfuse --version)\"" || exit 1
# sglang-jax's loader (model_loader/loader.py _warmup_safetensors_cache) reads every safetensors
# file whole on every host when the model path sits on a fuse mount: 1.9 TB a host for Inkling, about
# 72 minutes, before any real read (v5p-64, 2026-09-27). It decides by the path's mount entry, so
# ~/weights/<mount name>, a symlink on the root disk, skips it. Engine runs take that path.
# Each host's whole install and mount output stays in $OUT/gcsfuse-w<N>.log.
log "weights mounted at $MOUNT on $N hosts; host 0: $(sed -n '$p' "$OUT/gcsfuse-w0.log")"
# Host 0 also keeps measure_model.sh's capture shards in /dev/shm. GLM-5.3-Flash's 440,000 tokens at
# 65,536 bytes a token filled the 20 GB above the cache and stopped its measure (v5p-64,
# 2026-09-28), so host 0 gets 80 GB above it. The host has 440 GB of RAM.
case $CACHE_DIR in
  /dev/shm/*) on "${EXT[0]}" "sudo mount -o remount,size=$((CACHE_MB / 1024 + 80))G /dev/shm" || exit 1 ;;
esac

# 6. Every host sees the whole slice. JAX numbers the processes itself, so host 0 needn't be
# process 0; the last field names the host.
on "${EXT[0]}" "cd ~/repo && source ~/.tpu_env && bash scripts/multihost_exec.sh --wait-all python3 -c \"import jax, socket; jax.distributed.initialize(); print('DEVICES', jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count(), socket.gethostname(), flush=True)\"" \
    >"$OUT/devices.log" 2>&1
grep -h '^DEVICES' "$OUT/devices.log" | sort -k2 -n | tee "$OUT/devices.txt"
want=$((N * $(awk 'NR==1{print $5}' "$OUT/devices.txt")))
got=$(awk -v w="$want" '$4 == w' "$OUT/devices.txt" | wc -l)
if ((got != N)); then
  log "$got of $N hosts report $want devices. The whole log, $OUT/devices.log:"; cat "$OUT/devices.log"; exit 1
fi
log "READY: $N hosts, each reports jax.device_count() = $want"
