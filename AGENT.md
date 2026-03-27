# AGENT.md

## Project Overview
- `altqft` contains circuit generators, FI evaluation code, and training code for PH-style circuits and related QFT experiments.
- The main implementation lives in `src/altqft/`.
- Research scripts live in `scripts/`.
- Tests live in `tests/`.

## Directory Guide
- `src/altqft/circuits/`: circuit construction utilities and PH/QFT generators.
- `src/altqft/nn/`: FI processing, differentiable models, and training utilities.
- `scripts/train/`: training entry points and smoke-test runners.
- `scripts/fi_data_cal/`: FI dataset generation and reporting scripts.
- `scripts/plots/`: plotting scripts for stored FI results.
- `tests/`: regression tests for circuits, FI utilities, and models.
- `doc/`, `figs/`, `data/`, `model/`, `outputs/`: papers, figures, datasets, checkpoints, and run artifacts. Treat them as experiment assets and avoid overwriting them casually.

## Working Rules
- Use `uv` for dependency and command execution when possible.
- Typical commands:
  - `uv sync`
  - `uv run pytest`
  - `uv run mypy`
  - `uv run python scripts/train/train_ph1_min_fi.py`
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
- Prefer fixing data flow or data structures over stacking more conditionals on top.
- Eliminate special cases when possible instead of preserving them with extra branches.
- Avoid speculative abstraction; generalize only after the code has at least two real uses.
- Rewrite clever or fragile code into something boring and obvious.

## Refactor Review Style
- During refactors, use a Linus-style review lens: direct, unsentimental, and technically sharp.
- Critique the code, not the person. Be blunt about bad abstractions or unnecessary complexity, but never personal.
- Prioritize "good taste": make the common path obvious and reduce edge handling by improving invariants.
- If the logic feels tangled, question the design first instead of adding more local fixes.
- Call out pointless indirection, leaky flags, and over-engineered abstractions early.
- Do not preserve complexity for backward-looking reasons unless behavior or compatibility would actually break.
- Keep externally visible behavior stable unless the task explicitly allows a behavior change.

## Verification
- Run tests after non-trivial code changes.
- Run `mypy` after changing typed Python modules or scripts.
- If a change affects experiment scripts or saved-data loaders, verify the relevant script entry points directly.

## Commit Style
- Prefer focused commit messages such as:
  - `refactor: simplify PH circuit builders`
  - `refactor: add type hints to FI scripts`
  - `fix: keep FI dataset loader compatible`
