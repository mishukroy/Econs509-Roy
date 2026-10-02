# Prompt 00: Household solver and verification

## 1. Task

Build the discrete-state solver for the Question 2 household problem using `spec.md` and `tests.md`. This run covers stage 2, a running implementation, and stage 3, verification. The full comparisons and my method and grid choices come later.

## 2. Context

Start in `ps1/` and read its agent-instructions file, `spec.md`, `tests.md`, `log.md`, and this prompt. Confirm that stages 0 and 1 are committed and that this prompt was committed separately before the run. My manual derivation, `manual/manual_vfi.py`, and `manual/kernel_output.npz` must already exist.

First check whether the model, algorithms, validation inputs, test criteria, and output definitions are complete and consistent. Report any gap for my revision rather than filling it from the slides, reference code, or manual derivation. If the inputs are ready, propose a brief implementation plan and wait for my approval before large changes.

## 3. Constraints

Take the numerical definitions from `spec.md`. Implement the tests before or together with the routines they check. Work in `code/`, `tests/`, and regenerable `results/`; append dated entries to `log.md`. Leave `spec.md`, `tests.md`, `manual/`, and `prompts/` unchanged. Read manual content only for the kernel test. Missing manual inputs or an applicable test failure stop the affected work; report the measured result without weakening the criterion.

Use scripts and the fixed seeds and local limits in the specification and agent file. Split long runs with checkpoints preserving settings, state, counters, moments, and elapsed time. Do not install or upgrade packages, read outside `ps1/`, or rewrite Git history.

## 4. Implementation and outputs

1. **Implement.** Build the model and prices, grids, stacked Bellman operator, greedy policy, and joint transition in §§2.1–2.2 of `spec.md`. Implement all five solver variants, all three distribution methods in dense and sparse form, support statistics, Euler diagnostics, output writers, and `code/run_all.py` according to §§2.3–2.6 and §4. Map the automated checks to the 12 IDs in `tests.md`.

2. **Stage 2: solver runs.** Use the kernel validation case in §2.7: plain VFI from zero on 100 uniform points on `[0,20]`, with tolerance `1e-10`, followed by power iteration. Save the agent's output separately as `results/validation_kernel_vfi.npz`. Report the outer count, exit metric, value range, and mean assets. Commit the running implementation and implemented tests as stage 2, using actual measured results in the message.

3. **Stage 3: verified.** Run every check applicable at stage 3 under `tests.md`, including the manual-kernel comparison, the other validation solvers, and distribution comparisons. Report the kernel value difference, exact policy mismatch count, and each applicable test result. Full experiment checks and test 12 remain pending. Make a separate stage-3 commit only when every applicable check passes.

Keep outputs in the formats specified in §4. At each stage or session end, append the prompt path, agent/model version, work completed, measured evidence, unverified claims, interpretations, approvals, and open issues to `log.md`. Identify agent authorship in both stage commits.

## 5. Completion

Return the two commit hashes, validation and test commands, measured kernel differences, test outcomes, and files to review. Distinguish verification by running from claims. Stop after stage 3 and await `prompts/01_experiments.md`; the full experiments, report, and final sign-off are outside this run.
