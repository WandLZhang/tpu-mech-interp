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

# Run one command on every host of a multi-host TPU slice, as one sglang-jax engine.
#
#   bash scripts/multihost_exec.sh python3 -u scripts/check_capture.py --model-path /mnt/weights ...
#   bash scripts/multihost_exec.sh --wait-all python3 -c 'import jax; ...'
#
# Run it on host 0, from the same directory the command expects on every host.
# scripts/multihost_setup.sh writes the two files it reads: ~/multihost_hosts, one internal IP per
# line with host 0 first, and ~/.ssh/peer, a key every host accepts.
#
# Each host gets the same command with SGL_NNODES, SGL_NODE_RANK and SGL_DIST_INIT_ADDR set.
# capture_activations.py's engine_settings reads them, so every script that builds an Engine
# joins the one engine: rank 0 serves the requests, and the other ranks start a scheduler and wait
# in it. When rank 0's command ends, this kills the engine on every other host, waits for
# /dev/vfio to free and removes /tmp/libtpu_lockfile, the way measure_model.sh frees a host between
# runs, and kills the peer's command itself by the PID it left in ~/.multihost-rank.pid. It exits
# with rank 0's status.
#
# Every peer has to answer ssh before any rank starts, and a refused login gets 5 minutes of tries.
# A peer that ends while rank 0 runs gives rank 0 MULTIHOST_GRACE seconds to end by itself, and
# then rank 0 is stopped, so a run that can't finish fails instead of hanging.
#
# With MULTIHOST_COPY=1, any argument that names a file on host 0, such as a prompt file, gets
# copied to the same path on every other host first. With --wait-all it waits for every rank to
# exit by itself and prints each rank's log, for commands that end on every host, such as a device
# count; any rank's failure fails the run.
#
#   MULTIHOST_HOSTS   host list (default ~/multihost_hosts)
#   MULTIHOST_KEY     ssh key for the other hosts (default ~/.ssh/peer)
#   MULTIHOST_LOGS    where each rank's log goes (default ~/multihost-logs)
#   MULTIHOST_ENV     extra variable names to pass through (HF_HUB_OFFLINE, LOG_PAYLOADS and
#                     SGL_* always go)
#   SGL_DIST_PORT     coordinator port (default: a fresh one per run, so a lingering one can't
#                     collide)
#   MULTIHOST_GRACE   seconds rank 0 gets to end after a peer ends early (default 300)
#   MULTIHOST_LOCAL   1 runs every rank on this host with no ssh, for a CPU test of the wrapper
#                     and the engine across processes
set -uo pipefail

WAIT_ALL=0
if [ "${1:-}" = "--wait-all" ]; then WAIT_ALL=1; shift; fi
(($#)) || { echo "usage: bash scripts/multihost_exec.sh [--wait-all] COMMAND [ARGS...]" >&2; exit 2; }

HOSTS_FILE=${MULTIHOST_HOSTS:-$HOME/multihost_hosts}
KEY=${MULTIHOST_KEY:-$HOME/.ssh/peer}
LOGS=${MULTIHOST_LOGS:-$HOME/multihost-logs}
[ -s "$HOSTS_FILE" ] || { echo "no host list at $HOSTS_FILE; run scripts/multihost_setup.sh" >&2; exit 1; }
mapfile -t HOSTS < <(grep -v '^[[:space:]]*$' "$HOSTS_FILE")
N=${#HOSTS[@]}
PORT=${SGL_DIST_PORT:-$((20000 + RANDOM % 10000))}
ADDR="${HOSTS[0]}:$PORT"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes
     -o LogLevel=ERROR -o ServerAliveInterval=30)
SCP=(scp -q -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes
     -o LogLevel=ERROR)
# on HOST CMD: run CMD on a peer, over ssh or, with MULTIHOST_LOCAL=1, in a local shell.
on() { if [ "${MULTIHOST_LOCAL:-0}" = 1 ]; then bash -c "$2"; else "${SSH[@]}" "$1" "$2"; fi; }
mkdir -p "$LOGS"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
log() { echo "### multihost $(date -u +%H:%M:%S) $*"; }
# on_retry HOST CMD: on, for a short command, tried again while ssh itself fails (status 255). A
# host replacing google-guest-agent in the image's first-hour upgrade refused every login for
# minutes (v5p-64, 2026-09-28).
on_retry() {
  local try rc
  for try in $(seq 1 20); do
    on "$1" "$2"; rc=$?
    ((rc == 255)) || return "$rc"
    log "ssh to $1 failed (try $try of 20); trying again in 15 s" >&2
    sleep 15
  done
  return 255
}
# tree PID: PID and every descendant.
tree() {
  local all=$1 todo=$1 kids p
  while [ -n "$todo" ]; do
    kids=$(for p in $todo; do pgrep -P "$p"; done | tr '\n' ' ')
    all+=" $kids"; todo=$kids
  done
  echo $all
}

# The same engine pattern measure_model.sh kills, plus check_capture.py.
ENGINE='^sglang(-jax)?::'
ENGINE+='|^[^ ]*python[0-9.]* (-u )?([^ ]*/)?(serve_throughput|capture_activations|check_capture)\.py'
# A slice host runs nothing else, so every engine process on it goes. With MULTIHOST_LOCAL=1 the
# host is shared, so only the peer's own process tree goes: its PID and every descendant.
free_remote() {
  if [ "${MULTIHOST_LOCAL:-0}" = 1 ]; then
    local root; root=$(cat ~/.multihost-rank.pid 2>/dev/null) || return 0
    kill -9 $(tree "$root") 2>/dev/null
    return 0
  fi
  on_retry "$1" "[ -s ~/.multihost-rank.pid ] && kill -9 \$(cat ~/.multihost-rank.pid) 2>/dev/null;
    for p in \$(pgrep -f '$ENGINE'); do kill -9 \$p 2>/dev/null; done;
    for _ in \$(seq 1 20); do sudo fuser /dev/vfio/* >/dev/null 2>&1 || break; sleep 3; done;
    sudo rm -f /tmp/libtpu_lockfile" >/dev/null
}

# A peer that ends while rank 0 runs left the engine or never joined it, and rank 0 can wait on it
# forever: one refused ssh left seven ranks timing out and rank 0 hung (v5p-64, 2026-09-28). A
# healthy run's peers also end a moment before rank 0, when its engine shuts down, so rank 0 gets
# MULTIHOST_GRACE seconds first. The sleeps write to /dev/null: killing the watcher leaves its sleep
# running, and one that held this script's stdout kept a caller's pipe open for the whole grace.
watch_peers() {
  local k r0 grace=${MULTIHOST_GRACE:-300}
  while sleep 5 >/dev/null 2>&1; do
    for ((k = 1; k < N; k++)); do
      [ -e "$LOGS/rank$k-$STAMP.rc" ] || continue
      sleep "$grace" >/dev/null 2>&1
      r0=$(cat "$R0" 2>/dev/null) && kill -0 "$r0" 2>/dev/null || return 0
      log "rank $k ended with status $(cat "$LOGS/rank$k-$STAMP.rc") and rank 0 still runs $grace s" \
        "later; stopping rank 0. Rank $k's whole log, $LOGS/rank$k-$STAMP.log:"
      cat "$LOGS/rank$k-$STAMP.log"
      kill -9 $(tree "$r0") 2>/dev/null
      return 0
    done
  done
}

# Every variable the peers need, quoted for the remote shell.
envs="SGL_NNODES=$N SGL_DIST_INIT_ADDR=$ADDR"
for v in HF_HUB_OFFLINE LOG_PAYLOADS ${MULTIHOST_ENV:-} $(compgen -v SGL_ | grep -v -E '^SGL_(NNODES|NODE_RANK|DIST_INIT_ADDR)$'); do
  [ -n "${!v+x}" ] && envs+=" $v=$(printf %q "${!v}")"
done
cmd=""
for a in "$@"; do cmd+=" $(printf %q "$a")"; done

log "$N hosts, coordinator $ADDR, logs in $LOGS/*-$STAMP.log: $cmd"

# Files the command names travel to the same path on every peer.
for a in "$@"; do
  # check_capture.py, serve_throughput.py and capture_activations.py read their files on rank 0
  # alone. Bulk copies between hosts broke on a v5p-64 (2026-09-27: "Broken pipe" on a 20 MB pipe
  # while a plain ssh to the same host worked), so copying is opt-in with MULTIHOST_COPY=1.
  [ "${MULTIHOST_COPY:-0}" = 1 ] || continue
  [ -f "$a" ] && [ "${MULTIHOST_LOCAL:-0}" != 1 ] || continue
  f=$(readlink -f "$a")
  # A capture reference runs to 13 GB and only rank 0 reads it (check_capture.py); copying it to
  # seven peers timed out on a v5p-64 (2026-09-27). Files over 256 MiB stay on host 0.
  if (($(stat -c %s "$f") > 268435456)); then log "not copying $f ($(stat -c %s "$f") bytes) to the peers"; continue; fi
  for ((k = 1; k < N; k++)); do
    # A 2 MB copy to one peer sat 8 minutes and died with "lost connection" on a v5p-64 while the
    # next ssh to the same peer took 0.7 s (2026-09-27), so each copy gets a time limit and retries.
    ok=0
    for try in 1 2 3; do
      if timeout 120 "${SSH[@]}" -o ConnectTimeout=20 "${HOSTS[$k]}" \
          "mkdir -p $(printf %q "$(dirname "$f")") && cat > $(printf %q "$f")" <"$f"; then ok=1; break; fi
      log "copy of $f to ${HOSTS[$k]} failed (try $try)"; sleep 5
    done
    ((ok)) || { log "couldn't copy $f to ${HOSTS[$k]}"; exit 1; }
  done
done

for ((k = 1; k < N; k++)); do
  on_retry "${HOSTS[$k]}" true || { log "${HOSTS[$k]} refused ssh for 5 minutes; no rank started"; exit 1; }
done

# Each peer's status lands in rank<k>-<stamp>.rc when its command ends.
pids=()
for ((k = 1; k < N; k++)); do
  { on "${HOSTS[$k]}" "cd $(printf %q "$PWD") && { [ -f ~/.tpu_env ] && source ~/.tpu_env; true; } &&
      export $envs SGL_NODE_RANK=$k && echo \$\$ > ~/.multihost-rank.pid && exec $cmd" >"$LOGS/rank$k-$STAMP.log" 2>&1
    echo $? >"$LOGS/rank$k-$STAMP.rc"; } &
  pids+=($!)
done

R0=$LOGS/rank0-$STAMP.pid
WATCH=""
((WAIT_ALL)) || { watch_peers & WATCH=$!; }
SGL_NNODES=$N SGL_NODE_RANK=0 SGL_DIST_INIT_ADDR=$ADDR bash -c 'echo $$ >"$0" && exec "$@"' "$R0" "$@" 2>&1 |
  tee "$LOGS/rank0-$STAMP.log"
rc=${PIPESTATUS[0]}
[ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
log "rank 0 exited $rc"

if ((WAIT_ALL)); then
  for ((k = 1; k < N; k++)); do
    wait "${pids[$((k - 1))]}"; r=$(cat "$LOGS/rank$k-$STAMP.rc" 2>/dev/null || echo 1)
    log "rank $k exited $r; its log:"
    cat "$LOGS/rank$k-$STAMP.log"
    ((r == 0)) || rc=1
  done
else
  frees=()
  for ((k = 1; k < N; k++)); do
    { free_remote "${HOSTS[$k]}" || log "couldn't reach ${HOSTS[$k]} to stop rank $k; it may still hold its chips"; } &
    frees+=($!)
  done
  wait "${frees[@]}"
  # A peer whose command isn't an engine keeps its ssh open; end the clients, not just the engines.
  for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null; done
  kill "${pids[@]}" 2>/dev/null
  wait "${pids[@]}" 2>/dev/null
  for ((k = 1; k < N; k++)); do
    log "rank $k's whole log: $LOGS/rank$k-$STAMP.log ($(wc -c <"$LOGS/rank$k-$STAMP.log") bytes)"
  done
fi
exit "$rc"
