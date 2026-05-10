#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root" || exit 1

console_log="outputs/noise_9q10q/period_noise_9q10q.console.log"
exit_file="outputs/noise_9q10q/period_noise_9q10q.exit"

mkdir -p "outputs/noise_9q10q" "model/noise_9q10q" "data/period_recovery_noise_9q10q"
: > "$console_log"
rm -f "$exit_file"

{
  printf '%s repo_root=%s\n' "$(date '+%F %T')" "$repo_root"
  printf '%s launching period-noise sweep\n' "$(date '+%F %T')"
} >> "$console_log"

ALTQFT_TRAIN_DEVICE=cuda .venv/bin/python -u scripts/experiments/run_period_noise_9q10q.py "$@" >> "$console_log" 2>&1
status=$?
printf '%s' "$status" > "$exit_file"
exit "$status"
