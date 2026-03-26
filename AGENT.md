# AGENT.md

## Project Overview
- `altqft` contains circuit generators, state preparation utilities, FI evaluation code, and training code for PH-style circuits and related QFT experiments.
- The main implementation lives in `src/altqft/`.
- Research scripts live in `scripts/`.
- Tests live in `tests/`.

## Directory Guide
- `src/altqft/circuits/`: circuit construction utilities and PH/QFT generators.
- `src/altqft/state/`: modular-solution helpers and initial-state preparation.
- `src/altqft/nn/`: FI processing, differentiable models, and training utilities.
- `scripts/fi_data_cal/`: FI dataset generation and reporting scripts.
- `scripts/plots/`: plotting scripts for stored FI results.
- `tests/`: regression tests for circuits, FI utilities, models, and state preparation.
- `doc/`, `figs/`, `data/`, `model/`, `outputs/`: papers, figures, datasets, checkpoints, and run artifacts. Treat them as experiment assets and avoid overwriting them casually.

## Working Rules
- Use `uv` for dependency and command execution when possible.
- Typical commands:
  - `uv sync`
  - `uv run pytest`
  - `uv run mypy`
  - `uv run python scripts/train_ph1_min_fi.py`
- Keep changes focused on source code unless the task explicitly requires updating generated artifacts or research assets.
- When changing scripts that load saved results, keep backward compatibility with existing pickle/json outputs when practical.

## Refactoring Requirements
- Keep code concise and easy to read.
- Avoid deep nesting; prefer early returns and small helper functions.
- Keep functions short and focused on one task.
- Do not add defensive handling for cases that are not expected to happen.
- Add explicit type hints for every public function input and output.
- Keep the project passing `mypy`.
- Remove code smells such as duplicated logic, redundant parameters, and repeated definitions.
- Prefer function-oriented and side-effect-light designs when practical.

## Verification
- Run tests after non-trivial code changes.
- Run `mypy` after changing typed Python modules or scripts.
- If a change affects experiment scripts or saved-data loaders, verify the relevant script entry points directly.

## Commit Style
- Prefer focused commit messages such as:
  - `refactor: simplify PH circuit builders`
  - `refactor: add type hints to FI scripts`
  - `fix: keep FI dataset loader compatible`
