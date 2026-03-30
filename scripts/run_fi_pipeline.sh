#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START_QUBIT="${1:-4}"
END_QUBIT="${2:-14}"

find_python() {
  local candidate
  for candidate in ".venv/Scripts/python.exe" ".venv/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "error: no Python interpreter found" >&2
  return 1
}

PYTHON_BIN="$(find_python)"

calc_cmd=(
  "${PYTHON_BIN}"
  "scripts/fi_data_cal/calculate_fi_dataset.py"
  "--nqubit-start" "${START_QUBIT}"
  "--nqubit-end" "${END_QUBIT}"
  "--replace-range"
)

if [[ -n "${ALTQFT_FI_DEVICE:-}" ]]; then
  calc_cmd+=("--device" "${ALTQFT_FI_DEVICE}")
fi

if [[ -n "${ALTQFT_FI_WORKERS:-}" ]]; then
  calc_cmd+=("--workers" "${ALTQFT_FI_WORKERS}")
fi

train_cmd=(
  "${PYTHON_BIN}"
  "scripts/train/train_ph1_min_fi.py"
  "--nqubit-start" "${START_QUBIT}"
  "--nqubit-end" "${END_QUBIT}"
)

plot_cmd=(
  "${PYTHON_BIN}"
  "scripts/plots/plot_fi_dataset.py"
  "--nqubit-start" "${START_QUBIT}"
  "--nqubit-end" "${END_QUBIT}"
)

echo "Running FI dataset build for qubits ${START_QUBIT}..${END_QUBIT}"
"${calc_cmd[@]}"

echo "Running PH1 min-FI training for qubits ${START_QUBIT}..${END_QUBIT}"
"${train_cmd[@]}"

echo "Rendering plots for qubits ${START_QUBIT}..${END_QUBIT}"
"${plot_cmd[@]}"

echo "Pipeline finished."
