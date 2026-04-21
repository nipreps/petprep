#!/usr/bin/env bash
# Run a command when disk use on CHECK_PATH is at or above a threshold (default 80%).
#
# Examples:
#   DISK_THRESHOLD_PERCENT=80 CHECK_PATH=/ COMMAND='docker buildx prune -a -f' ./run-if-disk-over.sh
#   ./run-if-disk-over.sh 80 -- docker buildx prune -a -f
#   ./run-if-disk-over.sh -- df -h
#
# Env:
#   DISK_THRESHOLD_PERCENT   default 80
#   CHECK_PATH               path passed to df (default /)
#   COMMAND                  if set, executed with bash -c (instead of args after --)

set -euo pipefail

CHECK_PATH="${CHECK_PATH:-/}"
THRESHOLD="${DISK_THRESHOLD_PERCENT:-80}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  THRESHOLD="$1"
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

usage() {
  echo "usage: DISK_THRESHOLD_PERCENT=80 COMMAND='cmd' $0" >&2
  echo "   or: $0 [threshold] -- command [args...]" >&2
  exit 1
}

run_cmd() {
  if [[ -n "${COMMAND:-}" ]]; then
    bash -c "$COMMAND"
  elif [[ $# -gt 0 ]]; then
    "$@"
  else
    usage
  fi
}

used="$(df -P "$CHECK_PATH" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')"
ts="$(date -Iseconds 2>/dev/null || date)"

if [[ "$used" -ge "$THRESHOLD" ]]; then
  echo "[$ts] ${CHECK_PATH} disk use ${used}% (threshold ${THRESHOLD}%) — running command"
  run_cmd "$@"
else
  echo "[$ts] ${CHECK_PATH} disk use ${used}% (below ${THRESHOLD}%) — nothing to do"
fi


