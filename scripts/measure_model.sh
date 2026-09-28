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

# Measure one model the way every model page reports it: fetch, corpus, capture off, capture on,
# with peak HBM sampled beside each engine run.
#
#   bash scripts/measure_model.sh REPO CAPTURE_SLOT OUT_DIR
#   TP=4 MEM_FRAC=0.8 bash scripts/measure_model.sh openai/gpt-oss-120b 18 ~/results/gpt-oss-120b
#
# Run it on the TPU VM after scripts/bootstrap_tpu_vm.sh, inside tmux: a large model takes tens of
# minutes to fetch and load. Every RESULT line it prints is also in OUT_DIR/results.txt. It exits 1
# when a run fails or when any of the four RESULT lines a model page reports is missing. A rerun
# with the same OUT_DIR moves the last run's files to OUT_DIR/run-<time> and its shards to
# /dev/shm/caps-<OUT_DIR's name>-<time>, and deletes neither.
#
#   TP        tensor-parallel size; the chip count (default 8)
#   MEM_FRAC  mem_fraction_static (default 0.6); 0.8 leaves a large model room for its KV pool
#   PROMPTS   prompts to build (default 400)
#   MODEL_DIR read the weights from this directory, such as a gcsfuse mount, and skip the fetch
#   ENGINE_WRAP  command that runs each engine step, such as "bash scripts/multihost_exec.sh"
#             on host 0 of a multi-host slice; peak HBM then reads host 0's chips
#   ENGINE_ARGS  extra KEY=VALUE engine keywords, space-separated, for both engine runs
set -uo pipefail
set -m   # background jobs keep SIGINT, so peak_hbm.py stops cleanly and writes its result

if (($# != 3)); then
  echo "usage: [TP=N] [MEM_FRAC=F] [PROMPTS=N] bash scripts/measure_model.sh REPO CAPTURE_SLOT OUT_DIR" >&2
  exit 2
fi
REPO=$1
SLOT=$2
OUT=$3
TP=${TP:-8}
MEM_FRAC=${MEM_FRAC:-0.6}
read -r -a WRAP <<< "${ENGINE_WRAP:-}"
EXTRA=()
for kv in ${ENGINE_ARGS:-}; do EXTRA+=(--engine-arg "$kv"); done
PROMPTS=${PROMPTS:-400}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAPS=/dev/shm/caps-$(basename "$OUT")
# Capture off reads 2 warmup batches and 30 timed batches of 8 prompts.
((PROMPTS >= 256)) || { echo "PROMPTS=$PROMPTS: capture off needs at least 256" >&2; exit 1; }
mkdir -p "$OUT"
RC=0

[ -f "$HOME/.tpu_env" ] ||
  { echo "no ~/.tpu_env; run scripts/bootstrap_tpu_vm.sh on this VM first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HOME/.tpu_env"   # HF_HOME in /dev/shm, Xet off, compile cache
log() { echo "### $(date -u +%H:%M:%S) $*"; }

# A rerun keeps the last run. Its files move to $OUT/run-<time it started>, and its shards to
# $CAPS-<the same time>, so two runs' results, logs and shard hashes sit side by side. The corpus
# stays in $OUT, so every run reads the same prompts. A kept shard stays in /dev/shm, which is
# RAM, so delete it once you're done with it.
RUN_FILES=(results.txt fetch.log capture_off.out capture_on.out capture_on_steady.txt
           hbm_off.json hbm_off.log hbm_on.json hbm_on.log manifest.json started)
keep_last_run() {
  local stamp="" f kept="" where
  [ -s "$OUT/started" ] && stamp=$(cat "$OUT/started")
  # A run from before this script wrote `started` dates from its results.
  [ -z "$stamp" ] && [ -e "$OUT/results.txt" ] &&
    stamp=$(date -u -r "$OUT/results.txt" +%Y%m%dT%H%M%SZ)
  if [ -n "$stamp" ]; then
    for f in "${RUN_FILES[@]}"; do
      [ -e "$OUT/$f" ] || continue
      mkdir -p "$OUT/run-$stamp" && mv "$OUT/$f" "$OUT/run-$stamp/" && kept=1
    done
    [ -n "$kept" ] && log "kept the last run's files in $OUT/run-$stamp"
  fi
  if [ -e "$CAPS" ]; then
    where="$CAPS-${stamp:-$(date -u -r "$CAPS" +%Y%m%dT%H%M%SZ)}"
    [ -e "$where" ] && where="$where-$$"
    mv "$CAPS" "$where" &&
      log "kept the last run's shards in $where, $(du -sh "$where" | cut -f1) of RAM"
  fi
  date -u +%Y%m%dT%H%M%SZ >"$OUT/started"
}

keep_last_run

# Kill engine processes only. The scheduler and detokenizer match on the titles sglang-jax sets.
# The two drivers match on the python command line that runs them. A bare "sglang" or script
# name also matches a tee, the launching shell or the tmux server. Killing the tee or the tmux
# server ends the run.
ENGINE='^sglang(-jax)?::'
ENGINE+='|^[^ ]*python[0-9.]* (-u )?([^ ]*/)?(serve_throughput|capture_activations)\.py'
free_tpu() {
  local p
  for p in $(pgrep -f "$ENGINE"); do
    [ "$p" = "$$" ] || [ "$p" = "$PPID" ] || kill -9 "$p" 2>/dev/null
  done
  for _ in $(seq 1 20); do sudo fuser /dev/vfio/* >/dev/null 2>&1 || break; sleep 3; done
  sudo rm -f /tmp/libtpu_lockfile
}

hbm_start() {
  rm -f "$OUT/hbm_$1.json"   # so hbm_stop can't report the last run's peak as this run's
  python3 "$HERE/peak_hbm.py" --seconds 14400 --every 2 --out "$OUT/hbm_$1.json" \
    >"$OUT/hbm_$1.log" 2>&1 &
  HBM_PID=$!
  # The sampler waits out rounds before the engine holds the TPU. It exits at once only when
  # this host has no chip for tpu-info to read, or no tpu-info, so stop before the long run.
  sleep 3
  if ! kill -0 "$HBM_PID" 2>/dev/null; then
    log "peak_hbm.py stopped before the engine started. Its whole log, $OUT/hbm_$1.log:"
    cat "$OUT/hbm_$1.log"
    exit 1
  fi
}

hbm_stop() {
  kill -INT "$HBM_PID" 2>/dev/null
  wait "$HBM_PID" 2>/dev/null
  HBM_PID=   # reaped, so the EXIT trap has nothing to stop
  # --result prints no RESULT line for a run that read no chip, so a failed sample never reads
  # as a peak of 0.
  if ! python3 "$HERE/peak_hbm.py" --result "$OUT/hbm_$1.json" --stage "$1" | keep_results; then
    log "no HBM reading for $1. The sampler's whole log, $OUT/hbm_$1.log:"
    cat "$OUT/hbm_$1.log"
    RC=1
  fi
}

# Every RESULT line also lands in $OUT/results.txt. A progress bar can share a line with it, so
# split on carriage returns before matching.
keep_results() { tr '\r' '\n' | grep '^RESULT' | tee -a "$OUT/results.txt"; }

# set -m puts the sampler in its own process group, out of reach of Ctrl-C. Left alone, it runs up
# to 4 hours and then writes over hbm_<stage>.json. Stop it when the script exits.
trap '[ -n "${HBM_PID:-}" ] && kill -INT "$HBM_PID" 2>/dev/null' EXIT

log "FETCH ${MODEL_DIR:-$REPO}"
if [ -n "${MODEL_DIR:-}" ]; then
  echo "MODEL_DIR is set, so the engine reads $MODEL_DIR and nothing is fetched"
  SNAP=$MODEL_DIR
else
  python3 -u "$HERE/fetch_weights.py" "$REPO" | tee "$OUT/fetch.log"
  SNAP=$(awk '/^PATH /{print $2}' "$OUT/fetch.log")
fi
[ -d "$SNAP" ] || { log "fetch gave no snapshot directory"; exit 1; }

log "CORPUS"
[ -s "$OUT/prompts.jsonl" ] || python3 -u "$HERE/build_corpus.py" --model "$SNAP" \
  --prompts "$PROMPTS" --tokens 440 --out "$OUT/prompts.jsonl"
[ -s "$OUT/prompts.jsonl" ] || { log "no prompts in $OUT/prompts.jsonl, stopping"; exit 1; }

export HF_HUB_OFFLINE=1   # the engine reads the local snapshot from here on

free_tpu
log "CAPTURE OFF, tp=$TP mem_frac=$MEM_FRAC"
hbm_start off
"${WRAP[@]}" python3 -u "$HERE/serve_throughput.py" --model-path "$SNAP" --prompts "$OUT/prompts.jsonl" \
  --tp-size "$TP" --engine-arg "mem_fraction_static=$MEM_FRAC" "${EXTRA[@]}" >"$OUT/capture_off.out" 2>&1
OFF_RC=$?
echo "CAPTURE_OFF_RC=$OFF_RC"
((OFF_RC == 0)) || RC=1
hbm_stop off
if ! keep_results <"$OUT/capture_off.out"; then
  log "no RESULT line from capture off. Its whole log, $OUT/capture_off.out:"
  tr '\r' '\n' <"$OUT/capture_off.out"
  RC=1
fi

free_tpu
# capture_activations.py hands --layers to the engine as --return-hidden-states-layers, so the
# engine copies the one slot to the host, and wire_bytes_per_token reads d_model x 2 at bf16.
log "CAPTURE ON, slot $SLOT"
hbm_start on
"${WRAP[@]}" python3 -u "$HERE/capture_activations.py" --model-path "$SNAP" --prompts "$OUT/prompts.jsonl" \
  --out "$CAPS" --layers "$SLOT" --tp-size "$TP" --batch-size 8 \
  --shard-bytes 4294967296 --dtype float32 --engine-arg "mem_fraction_static=$MEM_FRAC" \
  "${EXTRA[@]}" >"$OUT/capture_on.out" 2>&1
ON_RC=$?
echo "CAPTURE_ON_RC=$ON_RC"
((ON_RC == 0)) || RC=1
hbm_stop on
if ((ON_RC == 0)); then
  grep -e '^wrote ' -e '^verified ' "$OUT/capture_on.out"
else
  log "capture on failed. Its whole log, $OUT/capture_on.out:"
  tr '\r' '\n' <"$OUT/capture_on.out"
fi
cp "$CAPS/manifest.json" "$OUT/manifest.json" 2>/dev/null

# The steady window the model pages report. The progress clock starts once the engine is built and
# its rate is cumulative, so the first lines carry the first calls' compile time. Drop the first
# two lines, and end on the last line before the final shard closes, which waits on the flush.
# A run too short to hold that window, or a window in which no token moved, prints why and sets
# the exit status, since the model page's number needs that RESULT line.
if ! python3 - "$OUT/capture_on.out" <<'PY' | tee "$OUT/capture_on_steady.txt" | keep_results
import json, sys
rows = []
decode = json.JSONDecoder().raw_decode
for line in open(sys.argv[1]).read().replace("\r", "\n").splitlines():
    if line.startswith("{") and '"elapsed_s"' in line:
        # An engine log line can land on the same line as a progress line. raw_decode reads the
        # JSON object at the start and leaves the rest.
        try:
            rows.append(decode(line)[0])
        except ValueError:
            print(f"skipped a progress line that doesn't parse: {line}")
before = [r for r in rows if r["shards"] < rows[-1]["shards"]] if rows else []
if len(before) < 4:
    print(f"no steady window: {len(before)} progress line(s) before the final shard closes, need 4. "
          "Rerun with a larger PROMPTS; the default is 400.")
    raise SystemExit(1)
a, b = rows[2], before[-1]
secs = b["elapsed_s"] - a["elapsed_s"]
tokens = b["tokens"] - a["tokens"]
if secs <= 0 or tokens <= 0:
    print(f"no steady window: {tokens} token(s) in {secs:.2f} s from {a['elapsed_s']} s "
          f"to {b['elapsed_s']} s")
    raise SystemExit(1)
print("RESULT " + json.dumps({
    "stage": "capture_on_steady", "window": f"{a['elapsed_s']} s to {b['elapsed_s']} s",
    "tokens": tokens, "secs": round(secs, 2), "tokens_per_s": round(tokens / secs, 1),
    "wire_mb_per_s": round((b["wire_bytes"] - a["wire_bytes"]) / secs / 1e6, 1),
    "wire_bytes_per_token": round((b["wire_bytes"] - a["wire_bytes"]) / tokens),
    "run_tokens": rows[-1]["tokens"]}))
PY
then
  log "no capture_on_steady RESULT. The step's whole output, $OUT/capture_on_steady.txt:"
  cat "$OUT/capture_on_steady.txt"
  RC=1
fi

free_tpu
log "DONE. Shards in $CAPS, logs and RESULT lines in $OUT"
exit "$RC"
