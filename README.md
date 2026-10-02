# ECONS 509 — Question 2 computational results

This assignment folder contains the household and invariant-distribution implementations, independent verification scripts, recorded student choices, numerical experiments, Euler diagnostics and plots. The task and definitions are in `spec.md`; `tests.md` specifies acceptance criteria, `manual/` contains the student's reference, `prompts/` contains the authorized workflow and `log.md` records decisions and measured evidence. Generated outputs are in `results/`.

From this folder, regenerate the full numerical experiment scope with:

```text
python code/run_all.py
```

On this computer, the existing Anaconda executable is `C:\Users\roymi\anaconda3\python.exe`. In PowerShell:

```powershell
& 'C:\Users\roymi\anaconda3\python.exe' code/run_all.py
```

Calls are bounded to about 50 seconds. Exit code 75 means a checkpoint was saved; repeat the same command until it reports completion. Checkpoints retain values, optimizer moments, traces, counters, settings and accumulated time. The command reproduces the initial comparisons and then follows the recorded choices without asking for new decisions. Verification scripts run after each block.

The student selected `modified_howard`, `power` with sparse CSR, `nonuniform` assets in [0,5] and N=5000. Choice reasons are preserved in `code/selected_settings.json` and `log.md`.

Versions actually used: Python 3.13.5 (Anaconda).

```text
numpy==2.1.3
scipy==1.15.3
matplotlib==3.10.0
```

Measured numerical runtimes:

| Part | Seconds |
|---|---:|
| (a) | 326.676201 |
| (b) | 0.5856116 |
| (c) | 0.0225496 |
| (d) | not timed: student choices |
| (e) | 1.8406687 |
| (f) | 1.9991337 |
| (g) | 6.0503269 |
| (h) | 0.0305555 |

Timing definitions and common baseline matrix setup/conversion are recorded in `results/stage4_summary.json`. Timers exclude verification, figure rendering/export, script startup, checkpoint writes and waiting between calls. Part (h) includes input loading and the scaling JSON write, and excludes final-array and table exports. Resumed solver/utility times are accumulated. The first import affects some first-case timings.

Results include NPZ arrays and JSON settings/statistics, complete LaTeX tables and Markdown `tables.md`, PDF/PNG figures, gradient traces, and partial-block and consolidated verification reports. `h_final` preserves the selected solved arrays with recomputed Euler diagnostics. `h_scaling.json` records both fitted slopes and every observation. The capped gradient/Adam runs and unsuitable range 2 remain in the outputs.

`results/tests_final.json` is the consolidated audit; `results/reproduction.json` records clean-copy numerical comparisons. To refresh the audit of saved results, run `python tests/verify_stage4.py --refresh`. For clean-copy reproduction, use `python tests/reproduce.py --prepare`, `python tests/reproduce.py --run` (repeat on exit 75), and `python tests/reproduce.py --compare`. Preparation rejects a folder that already has results. For another independent test, add the same fresh `--scratch-name` to all three commands, preserving earlier scratch runs under `.scratch/`. Original numerical outputs and selected settings are not overwritten.

The student remains responsible for independent verification and the report; those are separate from the agent's Stage-4 computational checks.
