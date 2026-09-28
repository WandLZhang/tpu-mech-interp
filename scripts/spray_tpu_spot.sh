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

# Spray a TPU Spot request across every zone that supports the requested slice.
# The first zone to reach READY wins. On any exit (win, timeout, Ctrl-C) the
# script deletes every other request it made.
#
# A queued request costs nothing until Google grants it. It uses quota in every
# state until someone deletes it. A loser that provisions before its delete
# lands bills for those minutes. Each request expires after MAX_WAIT_SECONDS.
# If the script dies before its cleanup runs, Google can still grant its
# requests until then.
#
# It prints the zones it tries as a ZONES="..." line, and on its way out it
# prints the commands that list what's left in them.
#
#   ACCEL=v6e-8 bash spray_tpu_spot.sh
#   ACCEL=v6e-256 ZONES="us-east5-a us-east5-b" bash spray_tpu_spot.sh
#   TPU_CREATE_FLAGS="--scopes=https://www.googleapis.com/auth/cloud-platform" ACCEL=v5p-64 bash spray_tpu_spot.sh
#
# TPU_CREATE_FLAGS goes to every `queued-resources create`, such as the scopes a slice needs to
# mount its weights from a bucket with gcsfuse.
set -uo pipefail

# Piped through `tee`, the script's stdout dies with tee on a Ctrl-C, and the
# next write would end the script by SIGPIPE before its cleanup deletes a thing.
# With the signal ignored, that write fails and the script goes on. Every job
# and command it starts inherits the setting.
trap '' PIPE

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[[ -n "$PROJECT" ]] || { echo "set PROJECT= or run: gcloud config set project YOUR_PROJECT" >&2; exit 1; }
ACCEL="${ACCEL:-v6e-8}"
# Runtime version follows the chip generation. Override with RUNTIME=... .
case "${ACCEL}" in
  v5litepod-*)       RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}" ;;
  v5p-*)             RUNTIME="${RUNTIME:-v2-alpha-tpuv5}" ;;
  v6e-*)             RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}" ;;
  v7x-*|tpu7x-*)     RUNTIME="${RUNTIME:-v2-alpha-tpu7x}" ;;
  *)                 RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}" ;;
esac
TAG="${TAG:-spray}"
POLL_SECONDS="${POLL_SECONDS:-20}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-1800}"
# How long each delete keeps retrying before the script gives up on it. A loser
# caught in PROVISIONING once took 12 min 38 s from its first delete to gone.
DELETE_DEADLINE_SECONDS="${DELETE_DEADLINE_SECONDS:-1200}"

# Print $1 with every line indented six spaces.
indent() { local nl=$'\n'; printf '      %s' "${1//${nl}/${nl}      }"; }

# MODE=spot (default) or MODE=ondemand.
#
# Use ondemand for anything a person logs into: Spot on v6e survived about an
# hour per slice when measured, which is fine for a throwaway benchmark and
# useless for a dev box. Spot costs about a quarter of on demand for v5p in
# us-east5, so it stays the default for experiments.
MODE="${MODE:-spot}"
case "$MODE" in
  spot)     SPOT_FLAG="--spot" ;;
  ondemand) SPOT_FLAG="" ;;
  *) echo "MODE must be spot or ondemand, got '$MODE'" >&2; exit 1 ;;
esac

# A queue can be closed to a project regardless of capacity. Submitting returns
# code 7, "User does not have permission to submit requests into this queue for
# accelerator type ... in location ...". That reads like a capacity error and
# retrying never helps, so list those zones here to skip them.
DENIED_ZONES="${DENIED_ZONES:-}"

# Print zone $1 if it offers ACCEL. Show a zone's gcloud output only when the
# call fails: gcloud can print the same warning on every call, and it would
# repeat once per zone.
offers() {
  local types
  if ! types=$(gcloud compute tpus accelerator-types list --zone="$1" \
                 --project="${PROJECT}" --filter="type=${ACCEL}" \
                 --format="value(type)" 2>&1); then
    printf "  can't list accelerator types in %s\n%s\n" "$1" "$(indent "${types}")" >&2
  elif grep -qxF "${ACCEL}" <<< "${types}"; then
    echo "$1"
  fi
}

# All zones advertising this accelerator type, discovered live and in parallel,
# one gcloud call per TPU location. Override with ZONES=... to constrain. A zone
# can advertise the accelerator type and serve nothing, so submit and find out.
if [[ -z "${ZONES:-}" ]]; then
  echo "discovering zones offering ${ACCEL}..."
  LOCATIONS=$(gcloud compute tpus locations list --project="${PROJECT}" \
                --format="value(locationId)") \
    || { echo "can't list TPU locations in ${PROJECT}. gcloud's error is above." >&2; exit 1; }
  export -f offers indent
  export PROJECT ACCEL
  # shellcheck disable=SC2086
  ZONES=$(printf '%s\n' ${LOCATIONS} | xargs -P 16 -I{} bash -c 'offers "$@"' _ {} | sort)
fi

# Drop zones whose queue is closed to us -- submitting there only produces a
# misleading permission error among the real results.
FILTERED=""
for z in ${ZONES}; do
  skip=""
  for d in ${DENIED_ZONES}; do [[ "$z" == "$d" ]] && skip=1 && break; done
  [[ -n "$skip" ]] && { echo "  skipping ${z} (queue closed)"; continue; }
  FILTERED="${FILTERED} ${z}"
done
ZONES="${FILTERED# }"

read -r -a ZONE_ARR <<< "${ZONES}"
if [[ ${#ZONE_ARR[@]} -eq 0 ]]; then
  echo "no zones offer ${ACCEL}" >&2; exit 1
fi

STAMP="$(date +%m%d-%H%M%S)"
echo "spraying ${ACCEL} ${MODE} across ${#ZONE_ARR[@]} zones: ${ZONE_ARR[*]}"
echo "  ZONES=\"${ZONE_ARR[*]}\""
echo

# This run's request in a zone. Its node gets the same name.
request_name() { echo "${TAG}-${ACCEL//./-}-${1}-${STAMP}"; }

submit() {
  local zone="$1" name out
  name="$(request_name "${zone}")"
  if out=$(gcloud compute tpus queued-resources create "${name}" \
      --node-id="${name}" --zone="${zone}" --project="${PROJECT}" \
      --accelerator-type="${ACCEL}" --runtime-version="${RUNTIME}" \
      --valid-until-duration="${MAX_WAIT_SECONDS}s" \
      ${SPOT_FLAG} ${TPU_CREATE_FLAGS:-} --quiet 2>&1); then
    echo "  submitted  ${zone}"
  # One printf per zone, so parallel workers don't interleave its lines.
  elif echo "${out}" | grep -qi "does not have permission"; then
    printf '  DENIED     %s  (queue closed, not capacity)\n%s\n' "${zone}" "$(indent "${out}")"
  else
    printf '  rejected   %s\n%s\n' "${zone}" "$(indent "${out}")"
  fi
}

# Print the state of this run's request in a zone, or nothing once it's gone.
# If gcloud can't list the zone, print its output to stderr and return 1.
request_state() {
  local want out n s
  want="$(request_name "$1")"
  if ! out=$(gcloud compute tpus queued-resources list --zone="$1" \
               --project="${PROJECT}" --filter="name~${want}" \
               --format="value(name.basename(),state.state)" 2>&1); then
    echo "${out}" >&2
    return 1
  fi
  while read -r n s; do
    if [[ "${n}" == "${want}" ]]; then echo "${s}"; break; fi
  done <<< "${out}"
  return 0
}

# printf to stdout, or to stderr once stdout's reader is gone. A Ctrl-C does
# that to the `tee` the script's output runs through.
say() { printf "$@" 2>/dev/null || printf "$@" >&2; }

# Delete this run's request in one zone and wait until it's gone.
#
# The TPU API won't delete a queued resource while it's PROVISIONING, even with
# --force. It accepts the call and fails the operation about 1.5 seconds later,
# so --async would exit 0 on a failed delete. This function deletes
# synchronously, shows gcloud's errors, and retries until the request is gone.
# It waits out the states that are still in progress.
delete_request() {
  local zone="$1" name state last="" seen="" deadline err
  name="$(request_name "${zone}")"
  deadline=$(( $(date +%s) + DELETE_DEADLINE_SECONDS ))
  while (( $(date +%s) < deadline )); do
    if ! state=$(request_state "${zone}" 2>&1); then
      say '  list failed in %s, retrying\n%s\n' "${zone}" "$(indent "${state}")"
    elif [[ -z "${state}" ]]; then
      [[ -n "${seen}" ]] && say '  deleted    %s/%s\n' "${zone}" "${name}"
      return 0
    else
      seen=1 ; last="${state}"
      case "${state}" in
        CREATING|PROVISIONING|SUSPENDING|DELETING) ;;
        *) if err=$(gcloud compute tpus queued-resources delete "${name}" --zone="${zone}" \
                      --project="${PROJECT}" --force --quiet 2>&1 >/dev/null); then
             say '  deleted    %s/%s\n' "${zone}" "${name}"
             return 0
           fi
           say '  delete failed %s/%s (%s), retrying\n%s\n' \
             "${zone}" "${name}" "${state}" "$(indent "${err}")" ;;
      esac
    fi
    sleep "${POLL_SECONDS}"
  done
  if [[ -n "${seen}" ]]; then
    say '  STILL PRESENT %s/%s (%s). Delete it by hand:\n' "${zone}" "${name}" "${last}"
  else
    say '  UNCHECKED  %s/%s. Every list failed. Check it by hand:\n' "${zone}" "${name}"
  fi
  say '    gcloud compute tpus queued-resources delete %s --zone=%s --project=%s --force\n' \
    "${name}" "${zone}" "${PROJECT}"
  return 1
}

# Turn INT, TERM and HUP into a plain exit with the usual 128+N status.
catch_signals() { trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP; }

# Start one background delete per zone, skipping the winner's. A job ignores
# INT, TERM, HUP and PIPE, so it keeps going after the script exits and after
# its stdout dies, and its messages move to stderr then. gcloud installs its own
# Ctrl-C handler, so a Ctrl-C can still kill a job's gcloud call, and the job
# retries it.
start_deletes() {
  local zone
  DELETES_STARTED=1
  trap '' INT TERM HUP
  for zone in "${ZONE_ARR[@]}"; do
    [[ "${zone}" == "${WINNER_ZONE}" ]] && continue
    ( trap '' INT TERM HUP PIPE; delete_request "${zone}" ) &
    DELETE_PIDS+=("$!")
  done
  catch_signals
}

# The commands that list what this run could have left behind, zones filled in.
teardown_check() {
  local kept=""
  [[ -n "${WINNER}" ]] && kept=", the slice this run kept among them"
  echo "Once the deletes below finish, list what's left in the zones this run tried."
  echo "Both lists come back empty when every request is gone${kept}:"
  echo "  ZONES=\"${ZONE_ARR[*]}\""
  echo "  for z in \$ZONES; do"
  echo "    gcloud compute tpus queued-resources list --zone=\"\$z\" --project=${PROJECT} --format=\"value(name)\""
  echo "    gcloud compute tpus tpu-vm list --zone=\"\$z\" --project=${PROJECT} --format=\"value(name)\""
  echo "  done"
  echo
}

# Runs on every exit once the submits start. It deletes every request but the
# winner, waits for the deletes, and exits 3 if a request might still exist.
# The deletes start before anything else prints. When stdout died with its
# reader, as `| tee` does on a Ctrl-C, what's left to say goes to stderr. The
# teardown check prints before the wait, so a second Ctrl-C still leaves it.
on_exit() {
  local status=$? p leaked=0
  [[ -n "${SUBMIT_LOG}" ]] && rm -f "${SUBMIT_LOG}"
  printf '\n' 2>/dev/null || exec 1>&2
  if [[ -z "${DELETES_STARTED}" ]]; then
    start_deletes
    echo "deleting any requests this run made..."
  fi
  teardown_check
  for p in ${DELETE_PIDS[@]+"${DELETE_PIDS[@]}"}; do wait "${p}" || leaked=1; done
  if (( leaked )); then
    echo "a request might still exist. See STILL PRESENT or UNCHECKED above." >&2
    status=3
  fi
  exit "${status}"
}

WINNER="" ; WINNER_ZONE="" ; DELETE_PIDS=() ; DELETES_STARTED="" ; SUBMIT_LOG=""
trap on_exit EXIT
catch_signals

# Print "ZONE STATE" for each zone given, all at once. A request that's gone
# prints GONE, and a zone gcloud can't list prints nothing. The poll loop stops
# reading at the first READY zone, so a worker still printing writes into a
# closed pipe; its printf error goes to /dev/null.
poll_round() {
  printf '%s\n' "$@" | xargs -P 16 -I{} bash -c \
    'state=$(request_state "$1" 2>/dev/null) && printf "%s %s\n" "$1" "${state:-GONE}" 2>/dev/null' _ {}
}

export -f submit request_name request_state indent
export PROJECT ACCEL RUNTIME TAG STAMP SPOT_FLAG MAX_WAIT_SECONDS
SUBMIT_LOG="$(mktemp)"
# Every request expires MAX_WAIT_SECONDS after its create call, and each create starts after
# this line. The poll stops at this clock plus MAX_WAIT_SECONDS by the wall clock. A round of
# list calls over many zones takes real time, so a count of sleeps alone runs past the expiry and
# polls requests that can no longer win but still hold quota.
SUBMITTED_AT=$(date +%s)
DEADLINE=$(( SUBMITTED_AT + MAX_WAIT_SECONDS ))
printf '%s\n' "${ZONE_ARR[@]}" | xargs -P 16 -I{} bash -c 'submit "$@"' _ {} | tee "${SUBMIT_LOG}"
# Poll the zones that took a request. A rejected or denied zone holds none, and
# the deletes still visit every zone.
POLL_ARR=()
while read -r word zone _; do
  [[ "${word}" == "submitted" ]] && POLL_ARR+=("${zone}")
done <"${SUBMIT_LOG}"
if [[ ${#POLL_ARR[@]} -eq 0 ]]; then
  echo "no zone accepted the request. gcloud's errors are above." >&2
  exit 1
fi

echo
echo "polling ${#POLL_ARR[@]} zone(s) for first READY (timeout ${MAX_WAIT_SECONDS}s)..."
while (( $(date +%s) < DEADLINE )); do
  while read -r zone state; do
    if [[ "${state}" == "ACTIVE" || "${state}" == "READY" ]]; then
      WINNER="$(request_name "${zone}")" ; WINNER_ZONE="${zone}" ; break 2
    fi
  done < <(poll_round "${POLL_ARR[@]}")
  left=$(( DEADLINE - $(date +%s) ))
  (( left > 0 )) || break
  sleep "$(( left < POLL_SECONDS ? left : POLL_SECONDS ))"
  # A dead stdout sends the rest of the run's output to stderr.
  printf '\r  %ss elapsed' "$(( $(date +%s) - SUBMITTED_AT ))" 2>/dev/null || exec 1>&2
done
echo

if [[ -z "${WINNER}" ]]; then
  echo "no zone granted ${ACCEL} within ${MAX_WAIT_SECONDS}s. To wait longer, raise MAX_WAIT_SECONDS."
  exit 2
fi

echo "WON: ${WINNER} in ${WINNER_ZONE}"
cat <<EOF

Connect:
  gcloud compute tpus tpu-vm ssh ${WINNER} --zone=${WINNER_ZONE} --project=${PROJECT}

Tear down when done (Spot still bills while ACTIVE):
  gcloud compute tpus queued-resources delete ${WINNER} --zone=${WINNER_ZONE} --project=${PROJECT} --force

EOF
echo "deleting losers so you aren't billed twice..."
start_deletes
exit 0
