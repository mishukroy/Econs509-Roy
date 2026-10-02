"""Regenerate the Stage-4 reader guide and measured runtime/environment summary."""
from pathlib import Path
import json
import sys
import numpy as np
import scipy
import matplotlib

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/"results"/f"{name}.json").read_text(encoding="utf-8"))


def write_documents():
    selected = json.loads((ROOT/"code"/"selected_settings.json").read_text(encoding="utf-8"))
    baseline = read("abc_summary")
    versions = {"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}
    times = {"a":baseline["runtime_seconds"]["a_total"],"b":baseline["runtime_seconds"]["b_algorithms"],
             "c":baseline["runtime_seconds"]["c_algorithms"],"d":None,
             **{p:read(p+"_summary")["total_seconds"] for p in ("e","f","g")},"h":read("h_summary")["analysis_seconds"]}
    common = baseline["common_matrix_setup"]
    summary = dict(selected_settings=selected,python_version=sys.version.split()[0],packages=versions,runtime_seconds=times,
                   shared_baseline_matrix_setup=common,capped_runs=baseline["capped_solvers"],
                   rejected_ranges=[c["kmax"] for c in read("e_summary")["cases"] if c["candidate_status"] == "unsuitable"],
                   final_statistics=read("h_summary")["statistics"],scaling=read("h_scaling"),
                   timing_notes=["Part a sums utility setup and solver time, including all resumed batches.",
                     "Parts b/c sum distribution setup, solve and normalization; common Q construction/conversion is recorded separately.",
                     "Part d records student choices and has no numerical runtime.",
                     "Parts e/f/g include utility setup, solver, Q construction/conversion and distribution.",
                     "Part h times loading g solutions, fitting, writing the scaling JSON and recomputing Euler arrays/statistics.",
                     "Times exclude verification, figure rendering/export, script startup, checkpoint writes and user waiting; h also excludes final-array and table exports.",
                     "First SciPy imports can appear in the first case of a process; no repeated-run timing stability is claimed."])
    (ROOT/"results"/"stage4_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines = ["# ECONS 509 — Question 2 computational results", "",
      "This assignment folder contains the household and invariant-distribution implementations, independent verification scripts, recorded student choices, numerical experiments, Euler diagnostics and plots. The task and definitions are in `spec.md`; `tests.md` specifies acceptance criteria, `manual/` contains the student's reference, `prompts/` contains the authorized workflow and `log.md` records decisions and measured evidence. Generated outputs are in `results/`.","",
      "From this folder, regenerate the full numerical experiment scope with:","","```text","python code/run_all.py","```","",
      "On this computer, the existing Anaconda executable is `C:\\Users\\roymi\\anaconda3\\python.exe`. In PowerShell:","","```powershell",
      "& 'C:\\Users\\roymi\\anaconda3\\python.exe' code/run_all.py","```","",
      "Calls are bounded to about 50 seconds. Exit code 75 means a checkpoint was saved; repeat the same command until it reports completion. Checkpoints retain values, optimizer moments, traces, counters, settings and accumulated time. The command reproduces the initial comparisons and then follows the recorded choices without asking for new decisions. Verification scripts run after each block.","",
      f"The student selected `{selected['solver']}`, `{selected['distribution_method']}` with sparse CSR, `{selected['family']}` assets in [0,{selected['kmax']}] and N={selected['N']}. Choice reasons are preserved in `code/selected_settings.json` and `log.md`.","",
      f"Versions actually used: Python {summary['python_version']} (Anaconda).", "","```text",
      *[name+"=="+version for name,version in versions.items()],"```","",
      "Measured numerical runtimes:","","| Part | Seconds |","|---|---:|",
      *[f"| ({part}) | {'not timed: student choices' if seconds is None else format(seconds,'.9g')} |" for part,seconds in times.items()],"",
      "Timing definitions and common baseline matrix setup/conversion are recorded in `results/stage4_summary.json`. Timers exclude verification, figure rendering/export, script startup, checkpoint writes and waiting between calls. Part (h) includes input loading and the scaling JSON write, and excludes final-array and table exports. Resumed solver/utility times are accumulated. The first import affects some first-case timings.","",
      "Results include NPZ arrays and JSON settings/statistics, complete LaTeX tables and Markdown `tables.md`, PDF/PNG figures, gradient traces, and partial-block and consolidated verification reports. `h_final` preserves the selected solved arrays with recomputed Euler diagnostics. `h_scaling.json` records both fitted slopes and every observation. The capped gradient/Adam runs and unsuitable range 2 remain in the outputs.","",
      "`results/tests_final.json` is the consolidated audit; `results/reproduction.json` records clean-copy numerical comparisons. To refresh the audit of saved results, run `python tests/verify_stage4.py --refresh`. For clean-copy reproduction, use `python tests/reproduce.py --prepare`, `python tests/reproduce.py --run` (repeat on exit 75), and `python tests/reproduce.py --compare`. Preparation rejects a folder that already has results. For another independent test, add the same fresh `--scratch-name` to all three commands, preserving earlier scratch runs under `.scratch/`. Original numerical outputs and selected settings are not overwritten.","",
      "The student remains responsible for independent verification and the report; those are separate from the agent's Stage-4 computational checks."]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    write_documents()
