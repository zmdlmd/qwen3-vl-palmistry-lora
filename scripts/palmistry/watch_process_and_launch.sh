#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/palmistry/watch_process_and_launch.sh --pid <pid> --command "<cmd>" [options]

Options:
  --pid <pid>               Process ID to watch.
  --command "<cmd>"         Command to launch after the watched PID exits.
  --poll-seconds <seconds>  Poll interval in seconds. Default: 60.
  --grace-seconds <seconds> Wait after the process exits before launch. Default: 10.
  --log-file <path>         Optional log file path.
  -h, --help                Show this help text.
EOF
}

PID=""
COMMAND=""
POLL_SECONDS=60
GRACE_SECONDS=10
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid)
      PID="${2:-}"
      shift 2
      ;;
    --command)
      COMMAND="${2:-}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:-}"
      shift 2
      ;;
    --grace-seconds)
      GRACE_SECONDS="${2:-}"
      shift 2
      ;;
    --log-file)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PID}" || -z "${COMMAND}" ]]; then
  usage
  exit 1
fi

if ! [[ "${PID}" =~ ^[0-9]+$ ]]; then
  echo "--pid must be numeric." >&2
  exit 1
fi

log() {
  local message
  message="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${message}"
  if [[ -n "${LOG_FILE}" ]]; then
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "${message}" >> "${LOG_FILE}"
  fi
}

if kill -0 "${PID}" 2>/dev/null; then
  log "Watching PID ${PID}. Poll=${POLL_SECONDS}s Grace=${GRACE_SECONDS}s"
else
  log "PID ${PID} is not running. Launching command without waiting."
fi

while kill -0 "${PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

log "PID ${PID} has exited."

if (( GRACE_SECONDS > 0 )); then
  log "Waiting ${GRACE_SECONDS}s before launch."
  sleep "${GRACE_SECONDS}"
fi

log "Launching: ${COMMAND}"
bash -lc "${COMMAND}" >> "${LOG_FILE}" 2>&1 &
LAUNCH_PID=$!
log "Started launch PID ${LAUNCH_PID}."
