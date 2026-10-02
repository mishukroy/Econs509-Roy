# Agent instructions (ECONS 509, Fall 2026)

You work in one assignment folder (`psN/`) of a course repository. The student is the
principal investigator and the author of record; you implement, run, document and commit.

## Start of every session

- Read `spec.md` (the task), `tests.md` (the tests) and `log.md` (what happened so far).
- Take every definition from `spec.md`: the model, the methods, the formulas, the values.
  Where it is silent or ambiguous, list the gaps before you plan and ask; the student
  answers by revising `spec.md`. Do not fill a gap from `manual/`, from other files or
  from memory: `manual/` is the student's check on your code, and you read it only to
  implement the kernel test. Whatever you still had to interpret, name in your log entry.
- Act on prompt files: do what `prompts/NN_*.md` asks when the student points you to one.
  If the student asks in chat for something consequential (a new stage, a change of plan,
  method or interpretation), ask for it as a `prompts/` file first. Chat clarifications and
  approvals are fine; record them in your log entry.
- Plan before acting on a multi-step task; wait for approval before large changes.

## Never touch

- `spec.md`, `tests.md`, `manual/`, `prompts/`: the student's files.
- Model parameters, grids, tolerances, test criteria: never change them to make something
  pass; report the failure and stop.
- Git history (no amend, rebase or force-push, pushed or not; a mistake is fixed by a new
  commit); credentials (never commit them).

## Working rules

- Tests before code: implement `tests/` from `tests.md` before or together with the code
  they test.
- Scripts, not notebooks. `python code/run_all.py` regenerates everything in `results/`;
  never commit an output that no script produces. Write every table in `results/` as a
  complete LaTeX `tabular` environment (and as a Markdown table in `tables.md`) and every
  figure as pdf and png, so that the report can include them by reference.
- Commit at each stage boundary (2 solver runs, 3 verified, 4 experiments). The message
  states the result in numbers, for example "stage 3: VFI 379 iterations, kernel test
  passes to 0.0", never a vague "update code". Identify yourself as the author (your co-author
  trailer, or a distinct author name). Stage 5, the report, is the student's commit, not
  yours.
- A failing test: report it and stop. When the student then revises `spec.md`, `tests.md`
  or `manual/`, re-run from the first affected stage and commit that stage again, naming
  the revision in the message.
- A stop in a prompt is binding. When a prompt tells you to stop for a choice of the
  student's, report the numbers, ask and wait; do not run a later part on a provisional
  choice. Once the student has chosen a method, run only that method in the parts that
  follow, not the alternatives.
- Separate "verified by running" (with the numbers) from "claimed", everywhere.
- Do not install or upgrade packages unless asked.
- Fix random seeds in code.

## Stage 4: README.md

With the stage-4 commit write `README.md`, the file a reader opens first: what this folder
contains (one paragraph; the task itself is in `spec.md`), how to regenerate `results/` in
one command, the Python and package versions you actually ran with (read from the running
interpreter, listed in `pip freeze` format), and the runtime of each part. Rewrite it
whenever a later run changes the results or the runtime.

## The report check

When a prompt asks you to check the student's report: compare every number in the text
with `results/*.json` and every table and figure reference with `results/`; correct
numbers and typos only; leave every explanation as it is; list each correction in your
log entry; commit `report: checked against results/, N corrections`. The student reads
your diff and commits stage 5.

## End of every session

Append a dated entry to `log.md`; never edit an earlier one. The entry states: agent and
model, with version; the `prompts/` file you ran; what you did; what you verified by
running, with numbers; what is only claimed; what you interpreted beyond `spec.md`;
decisions the student gave in chat; what is open.

## Local limits (written by the student for this folder)

- Do not open files outside this folder.
- Runs longer than about three minutes: split them and save the state between calls.
- (The student adds what applies here, for example: do not read the answer key.)
