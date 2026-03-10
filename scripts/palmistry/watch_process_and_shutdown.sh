#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/palmistry/watch_process_and_shutdown.sh --pid <pid> [options]

Options:
  --pid <pid>               Process ID to watch.
  --poll-seconds <seconds>  Poll interval in seconds. Default: 60.
  --grace-seconds <seconds> Wait this many seconds after the process exits before shutdown. Default: 30.
  --log-file <path>         Optional log file path.
  --dry-run                 Do not shut down; only log what would happen.
  -h, --help                Show this help text.
EOF
}

PID=""
POLL_SECONDS=60
GRACE_SECONDS=30
LOG_FILE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid)
      PID="${2:-}"
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
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "${PID}" ]]; then
  echo "--pid is required." >&2
  usage
  exit 1
fi

if ! [[ "${PID}" =~ ^[0-9]+$ ]]; then
  echo "--pid must be a numeric process id." >&2
  exit 1
fi

if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ && "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "--poll-seconds and --grace-seconds must be integers." >&2
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
  log "Watching PID ${PID}. Poll=${POLL_SECONDS}s Grace=${GRACE_SECONDS}s DryRun=${DRY_RUN}"
else
  log "PID ${PID} is not running. Proceeding to shutdown flow immediately."
fi

while kill -0 "${PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

log "PID ${PID} has exited."

if (( GRACE_SECONDS > 0 )); then
  log "Waiting ${GRACE_SECONDS}s before shutdown."
  sleep "${GRACE_SECONDS}"
fi

if (( DRY_RUN )); then
  log "Dry run enabled. Skipping shutdown."
  exit 0
fi

log "Issuing shutdown -h now."
/sbin/shutdown -h now
