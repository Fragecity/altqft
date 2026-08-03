<!-- RED_QUEEN:PROTOCOL:v1 -->
# Red Queen agent protocol

This project uses `.red-queen/` as the authoritative activity ledger. If you are an AI agent, follow this protocol for every task. Prefer the `rq` CLI; do not hand-edit revisions or timestamps when the CLI is available.

## Before normal work

1. Read `.red-queen/project.toml` and the relevant task file in `.red-queen/tasks/`.
2. Reuse the task requested by the user. If none exists, create one with `rq task add`.
3. Classify the **immediate activity**, then start a session before analysis or edits:

```bash
rq agent start --project . --task <TASK_ID> --actor <AGENT_NAME> --activity auto --summary "<what you are about to do>"
```

`auto` is a fallback classifier. When the correct category is clear, pass it explicitly:

| Activity | Use when |
| --- | --- |
| `doing.solve` | designing a solution, deciding how to approach a task |
| `doing.implement` | writing, fixing, refactoring, or testing code/artifacts |
| `understanding.agent` | explaining or investigating what an AI/agent is doing |
| `understanding.background` | learning background, papers, docs, concepts, or prior context |
| `analyzing.profiling` | profiling, benchmarking, tracing, or measuring performance |
| `analyzing.locating` | debugging, diagnosing, locating a problem, or root-cause analysis |
| `waiting.external` | work cannot proceed until another person/system/dependency responds |
| `stopped` | the task is deliberately paused; ordinary time between sessions is left empty |

Classify by your immediate action, not by the user's broad goal.

## When the activity changes

Close the current bar and open a new one:

```bash
rq agent switch --project . --task <TASK_ID> --activity <ACTIVITY> --summary "<new immediate action>"
```

For example, after profiling identifies a bottleneck, switch from `analyzing.profiling` to `analyzing.locating`, then to `doing.implement` when editing the fix.

## Before your final response

Always close the active session:

```bash
rq agent finish --project . --task <TASK_ID> --status <STATUS> --outcome "<result>" --evidence "<test, file, commit, or observation>"
```

Valid task statuses are `todo`, `in_progress`, `blocked`, `paused`, `done`, and `cancelled`.

- Use `done` only when the requested outcome is complete and include evidence.
- Use `blocked` only for an external dependency; state who or what is needed.
- Use `paused` when work intentionally stops but can resume later.
- Use `in_progress` when this agent session ends but the task remains active.
- Never silently leave an open session. If interrupted later, the app or `rq doctor` may mark it stale.

## Data safety

- Do not delete timeline entries created by another actor.
- Do not rewrite another agent's task without reading its current revision.
- Do not put secrets, tokens, private prompts, or hidden chain-of-thought in Red Queen files.
- Record concise summaries, outcomes, evidence, commands, filenames, and externally observable facts only.
- The user may adjust timeline bars in the app; user edits have `source = "user"` and must be preserved.
