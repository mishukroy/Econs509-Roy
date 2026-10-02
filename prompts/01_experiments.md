# Prompt 01: Solver comparisons, grids, and accuracy

## 1. Task

Run Question 2(a)–(h) as specified in `spec.md` and complete stage 4. Provide the comparisons and numerical evidence from which I will choose the methods and grids. I write the explanations and reasons for my choices.

## 2. Context and experiment sequence

Start in `ps1/` and read the agent-instructions file, `spec.md`, `tests.md`, `log.md`, and this prompt. Confirm separate stage-2 and stage-3 commits, passing applicable verification checks, and this prompt's separate commit before the run. If inputs changed, rerun the affected checks; report missing definitions for my revision. Present a brief execution plan and wait for my approval before large changes.

Use the formulas, stopping and counting conventions, timing rules, and output definitions in `spec.md`. Run the following blocks in order.

### (a)–(d): Compare methods

At `N=1000` on the uniform `[0,20]` grid, start every household solver from zero. Compare VFI, sparse Howard, modified Howard with `H=50`, fixed-step gradient descent with `eta=5e-4`, and Adam with learning rate `0.1`. Use the common Bellman tolerance `1e-8`; both gradient variants are capped at 20,000 outer checks. Report outer iterations, updates, inner steps where relevant, running time, exit metric, and convergence status. Save gradient traces and the contraction and conditioning diagnostics in §§2.3 and 2.5, including singular values if feasible.

For (b), use the converged Howard policy to construct dense `Q`. Compare power iteration, inverse iteration using `eigs`, and the modified-equation solve using `numpy.linalg.solve`. For (c), repeat with the same `Q` in CSR form and `spsolve` for the direct system. Save distributions, timings, correctness checks, and maximum pairwise differences within and across representations.

**STOP after (c).** Present the solver and distribution tables and ask me to choose the household solver and distribution method, including dense or sparse representation. Wait for my answer before (e). Part (d) records these choices; only converged methods passing their applicable checks are eligible.

### (e): Range of the asset grid

With my chosen methods, keep `N=1000` and compare uniform grids with `k_max=2,5,10,20,40`. For each, report top-node mass, the upper support endpoint, the share of grid points inside support, and runtime using §2.5. Retain unsuitable ranges in the comparison without altering their bounds or probabilities.

**STOP after (e).** Show the range table and wait for my upper-bound choice before (f).

### (f): Uniform and nonuniform grids

At my chosen bound and `N=1000`, compare the uniform grid with the nonuniform formula in §2.5. Keep the selected methods and other settings fixed. Save values, policies, distributions, timing, and Euler diagnostics, and report applicable checks for both grids.

**STOP after (f).** Present the comparison and wait for my grid-family choice before (g).

### (g): Number of grid points

Using the selected methods, bound, and grid formula, solve at `N=100,500,1000,2000,5000`. Save every solution and runtime. Plot the value function, asset policy, and joint probability masses for each grid size, showing both shocks clearly.

**STOP after (g).** Present the comparisons and wait for my final `N` choice before the part-(h) analysis. Preserve all five solutions.

### (h): Euler-equation accuracy

At my chosen `N`, compute the §2.6 residual at every state where the borrowing constraint is slack. Plot it against assets for both shocks and report all specified unweighted, weighted, and support summaries. Flag upper-bound choices and explain unavailable statistics. Use all five part-(g) solutions for the log-log accuracy-versus-grid-step plots and fitted slopes. Report the observed scaling; my report will assess the lecture's grid-step claim.

Run the final-grid monotonicity, consumption, and distribution checks, together with all other applicable tests. Complete the checks that were pending at stage 3.

## 3. Constraints

At each pause, save partial results and append a dated log entry. Do not proceed with a provisional choice. After my answer, record the choice and any reason I supply in `log.md` and update `code/selected_settings.json`, outside generated outputs. From (e) onward, use only my selected methods.

Preserve the student-owned inputs, parameters, numerical settings, and test criteria. Report an applicable failure with its test ID and measured value, and stop affected work. Correctly reported gradient caps and rejected range candidates remain experimental outcomes. Follow the agent file's permissions and run limits; use fixed seeds and checkpoints retaining settings, values, counters, moments, and elapsed runtime. Do not install or upgrade packages or rewrite history. Each log entry identifies the prompt and agent/model version and separates measured evidence, claims, interpretations, and open decisions.

## 4. Outputs

Follow §4 of `spec.md`: NPZ arrays, JSON summaries and test evidence, gradient traces, complete LaTeX tables with Markdown versions, and PDF/PNG figures. Tables and plotted data must agree with the numerical outputs. Run test 12 in a scratch copy: `python code/run_all.py` must regenerate all experiments, including (a)–(c), using recorded choices without asking for new decisions. Preserve the original results and choice configuration.

Write the stage-4 README with contents, the reproduction command, actual Python/package versions in `pip freeze` format, and runtime by part, including time across resumed runs.

## 5. Completion

When all experiments, applicable tests, reproduction, and the README are complete, make the stage-4 commit with measured results and identifiable agent authorship. Report its hash, my selected settings, output paths, test outcomes, capped runs, rejected ranges, runtimes, and remaining limitations.

Stop after stage 4. My independent verification and report follow; the report check requires a separately committed `prompts/02_report.md`. Stage 5 and the final submission tag remain my sign-off.
