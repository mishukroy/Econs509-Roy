"""Reproducible entry point; stage 4 runs only with student-recorded choices.

python code/run_all.py --stage 2
python code/run_all.py --stage 3
python code/run_all.py                  # validation, then chosen experiments
python code/run_all.py --part abc       # later, when prompt 01 authorizes it

Exit 75 means a checkpoint was saved. Repeat the same command to resume.
No methods, bounds, or grid sizes are chosen on the student's behalf.
"""
from pathlib import Path
import argparse
import copy
import itertools
import json
import sys
import time

import numpy as np

from household import (CAPS, Problem, ResumePending, grid, invariant, parameters,
                       scaling_fit, solve, statistics, transition)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / ".checkpoints"


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_case(name):
    with np.load(RESULTS / f"{name}.npz", allow_pickle=False) as f:
        a = {key: f[key].copy() for key in f.files}
    m = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    return a, m


def write_case(name, arrays, meta):
    RESULTS.mkdir(exist_ok=True)
    meta["arrays"] = list(arrays)
    np.savez(RESULTS / f"{name}.npz", **arrays)
    json_write(RESULTS / f"{name}.json", meta)


def attach_distribution(problem, arrays, method, representation, case):
    start = time.perf_counter()
    Q = transition(arrays["G"].ravel(order="F"), problem.P, representation)
    construction_seconds = time.perf_counter()-start
    pi, metadata = invariant(Q, method, CHECKPOINTS / f"{case}_distribution.npz")
    arrays["pi"] = pi
    metadata.update(representation=representation, matrix_construction_seconds=construction_seconds,
                    matrix_conversion_seconds=0.)
    return metadata


def household_case(name, method, N=100, kmax=20., family="uniform", tolerance=1e-8,
                   distribution_method=None, representation="sparse"):
    problem = Problem(grid(N, kmax, family))
    arrays, solver, trace = solve(problem, method, tolerance, CHECKPOINTS / f"{name}_solver.npz")
    if solver["status"] != "converged" and method not in ("gradient", "adam"):
        raise RuntimeError(f"Required solver {method} failed to converge: {solver}")
    meta = dict(case=name, parameters=parameters(), grid=dict(N=N, kmax=kmax, family=family),
                solver=solver, distribution=None)
    if distribution_method is not None:
        meta["distribution"] = attach_distribution(problem, arrays, distribution_method, representation, name)
    meta["statistics"] = statistics(arrays)
    write_case(name, arrays, meta)
    if method in ("gradient", "adam"):
        np.savez(RESULTS / f"{name}_trace.npz", **trace)
    print(json.dumps({"case": name, "outer_checks": solver["outer_checks"], "exit_metric": solver["exit_metric"],
                      "status": solver["status"], "value_min": float(arrays["V"].min()),
                      "value_max": float(arrays["V"].max()), "mean_assets": meta["statistics"].get("mean_assets")}), flush=True)


def distribution_case(name, source, representation, method):
    arrays, original = load_case(source)
    meta = copy.deepcopy(original)
    meta["case"] = name
    problem = Problem(arrays["k_grid"])
    # Build CSR once as the common transition, then explicitly time conversion.
    started = time.perf_counter()
    sparse_Q = transition(arrays["G"].ravel(order="F"), problem.P)
    construction = time.perf_counter()-started
    started = time.perf_counter()
    Q = sparse_Q.toarray() if representation == "dense" else sparse_Q
    conversion = time.perf_counter()-started
    pi, dist = invariant(Q, method, CHECKPOINTS / f"{name}_distribution.npz")
    dist.update(representation=representation, matrix_construction_seconds=construction,
                matrix_conversion_seconds=conversion, policy_source=source)
    arrays["pi"] = pi
    meta.update(distribution=dist, statistics=statistics(arrays))
    write_case(name, arrays, meta)
    print(json.dumps({"case": name, "stationarity": dist["stationarity_residual"],
                      "raw_minimum": dist["raw_minimum"], "seconds": dist["distribution_seconds"]}), flush=True)


def table(stem, headers, rows):
    RESULTS.mkdir(exist_ok=True)
    def cell(value):
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)
    text_rows = [[cell(v) for v in row] for row in rows]
    def latex(value):
        return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    tex = [r"\begin{tabular}{" + "l" * len(headers) + "}", r"\hline",
           " & ".join(map(latex, headers)) + r" \\", r"\hline"]
    tex += [" & ".join(map(latex, row)) + r" \\" for row in text_rows]
    tex += [r"\hline", r"\end{tabular}"]
    (RESULTS / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    path = RESULTS / "tables.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = f"<!-- table:{stem} -->"
    end = f"<!-- end:{stem} -->"
    body = "\n".join([marker, f"## {stem}", "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(row) + " |" for row in text_rows], end])
    if marker in existing:
        i, j = existing.index(marker), existing.index(end) + len(end)
        existing = existing[:i] + body + existing[j:]
    else:
        existing += ("\n\n" if existing else "") + body + "\n"
    path.write_text(existing, encoding="utf-8")


def pairwise(names):
    groups = {"within_dense": [], "within_sparse": [], "across": []}
    for a, b in itertools.combinations(names, 2):
        pa, ma = load_case(a); pb, mb = load_case(b)
        difference = float(np.max(np.abs(pa["pi"] - pb["pi"])))
        ra, rb = ma["distribution"]["representation"], mb["distribution"]["representation"]
        key = "within_" + ra if ra == rb else "across"
        groups[key].append((difference, a, b))
    maxima = {key: dict(difference=max(rows)[0], pair=list(max(rows)[1:])) for key, rows in groups.items()}
    worst = max(row[0] for rows in groups.values() for row in rows)
    if worst > 1e-8:
        raise RuntimeError(f"Distribution agreement failure: {maxima}")
    return maxima


def baseline_tables():
    rows = []
    for method in CAPS:
        _, m = load_case(f"a_{method}")
        s = m["solver"]
        rows.append([method, s["outer_checks"], s["updates"], s["inner_steps"], s["setup_seconds"],
                     s["solver_seconds"], s["total_seconds"], s["exit_metric"], s["status"]])
    table("solvers", ["Method", "Outer checks", "Updates", "Inner steps", "Setup seconds", "Solver seconds", "Total seconds", "Exit metric", "Status"], rows)
    names = []
    for rep in ("dense", "sparse"):
        rows = []
        for method in ("power", "eigenvector", "equations"):
            name = f"bc_{rep}_{method}"; names.append(name)
            _, m = load_case(name); d = m["distribution"]
            rows.append([method, d["distribution_seconds"], d["updates"], d["matrix_construction_seconds"],
                         d["matrix_conversion_seconds"], d["stationarity_residual"], d["normalization_error"], d["raw_minimum"], d["correction_mass"]])
        table(f"distribution_{rep}", ["Method", "Solve seconds", "Power updates", "Construction seconds", "Conversion seconds", "Stationarity", "Normalization", "Raw minimum", "Correction mass"], rows)
    json_write(RESULTS / "bc_agreement.json", pairwise(names))
    json_write(RESULTS / "d_theory.json", dict(contraction_prediction=float(np.log(1e-8)/np.log(.96)),
               loss_hessian_condition_lower_bound=(1-.96)**-2,
               singular_values_status="not computed; optional feasibility calculation",
               condition_definitions={"J": "cond2(J)", "loss_Hessian": "cond2(J)^2"}))


def experiment_table(stem, names):
    rows = []
    for name in names:
        _, m = load_case(name); s = m["statistics"]
        runtime = m["solver"]["total_seconds"] + m["distribution"]["distribution_seconds"] + m["distribution"]["matrix_construction_seconds"] + m["distribution"]["matrix_conversion_seconds"]
        rows.append([name, m["grid"]["N"], m["grid"]["kmax"], s["mean_assets"], s["top_mass"],
                     s["support_endpoint"], s["support_share"], s["grid_step"], s["euler_mean"], s["euler_max"],
                     s["euler_weighted_mean"], s["maximum_weighted_contribution"], s["euler_support_max"], runtime,
                     "unsuitable" if s["top_mass"] > 1e-12 else "suitable"])
    table(stem, ["Case", "N", "Upper bound", "Mean assets", "Top mass", "Support endpoint", "Support share", "Grid step", "Euler mean", "Euler maximum", "Weighted mean", "Maximum weighted contribution", "Support maximum", "Total seconds", "Range status"], rows)


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(plt, fig, stem):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS / f"{stem}.{suffix}", dpi=160)
    plt.close(fig)


def node_figures(N):
    plt = plotting()
    a, m = load_case(f"g_N{N}")
    for stem, field, label in (("value", "V", "Value"), ("policy", "g", "Next-period assets"), ("distribution", "pi", "Joint probability mass")):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        data = a[field].reshape((N, 2), order="F")
        for s in range(2):
            ax[s].plot(a["k_grid"], data[:, s])
            ax[s].set(xlabel="Current assets", ylabel=label, title=f"Efficiency {parameters()['epsilon'][s]}, N={N}")
        save_figure(plt, fig, f"g_N{N}_{stem}")


def accuracy_outputs(settings):
    plt = plotting()
    metas = [load_case(f"g_N{N}")[1] for N in (100, 500, 1000, 2000, 5000)]
    fit = scaling_fit(metas)
    json_write(RESULTS / "h_scaling.json", fit)
    a, m = load_case(f"g_N{settings['N']}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for s in range(2):
        mask = a["slack_mask"][:, s]
        ax[s].plot(a["k_grid"][mask], a["E"][mask, s])
        ax[s].set(xlabel="Current assets", ylabel="Absolute relative Euler error", title=f"Efficiency {parameters()['epsilon'][s]}")
    save_figure(plt, fig, "h_euler")
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("euler_mean", "Mean"), ("euler_max", "Maximum")):
        pairs = np.asarray(fit[key]["observations"])
        if pairs.size:
            ax.loglog(pairs[:, 0], pairs[:, 1], "o-", label=label)
    ax.set(xlabel="Maximum grid step", ylabel="Absolute relative Euler error")
    ax.legend()
    save_figure(plt, fig, "h_scaling")
    experiment_table("accuracy", [f"g_N{N}" for N in (100, 500, 1000, 2000, 5000)])


def selected(required):
    path = ROOT / "code" / "selected_settings.json"
    if not path.exists():
        raise ValueError("Student choices required in code/selected_settings.json before this part")
    settings = json.loads(path.read_text(encoding="utf-8"))
    absent = [key for key in required if key not in settings]
    if absent:
        raise ValueError("Student choices missing: " + ", ".join(absent))
    if settings.get("solver") not in CAPS or settings.get("distribution_method") not in ("power", "eigenvector", "equations") or settings.get("representation") not in ("dense", "sparse"):
        raise ValueError("Invalid recorded method choices")
    if "kmax" in settings and settings["kmax"] not in (2, 5, 10, 20, 40):
        raise ValueError("Recorded bound must be a specified trial range")
    if "family" in settings and settings["family"] not in ("uniform", "nonuniform"):
        raise ValueError("Invalid recorded grid formula")
    if "N" in settings and settings["N"] not in (100, 500, 1000, 2000, 5000):
        raise ValueError("Invalid recorded final node count")
    # Choices must have converged and passed baseline checks before use.
    _, solver = load_case(f"a_{settings['solver']}")
    _, dist = load_case(f"bc_{settings['representation']}_{settings['distribution_method']}")
    if solver["solver"]["status"] != "converged" or dist["distribution"]["status"] != "converged":
        raise ValueError("Selected methods must have converged")
    return settings


def validation_tasks(stage):
    tasks = [("kernel", lambda: household_case("validation_kernel_vfi", "vfi", tolerance=1e-10,
                                              distribution_method="power", representation="dense"))]
    if stage >= 3:
        for method in ("vfi", "howard", "modified_howard"):
            tasks.append((f"validation_{method}", lambda method=method: household_case(f"validation_{method}", method,
                          distribution_method="power" if method == "howard" else None)))
        for rep in ("dense", "sparse"):
            for method in ("power", "eigenvector", "equations"):
                name = f"validation_{rep}_{method}"
                tasks.append((name, lambda name=name, rep=rep, method=method: distribution_case(name, "validation_howard", rep, method)))
        def verify_stage3():
            sys.path.insert(0, str(ROOT / "tests"))
            from verify import verify
            verify(3)
        tasks.append(("verify_stage3", verify_stage3))
    return tasks


def part_tasks(part):
    tasks = []
    if part == "abc":
        for method in CAPS:
            name = f"a_{method}"
            tasks.append((name, lambda name=name, method=method: household_case(name, method, N=1000)))
        for rep in ("dense", "sparse"):
            for method in ("power", "eigenvector", "equations"):
                name = f"bc_{rep}_{method}"
                tasks.append((name, lambda name=name, rep=rep, method=method: distribution_case(name, "a_howard", rep, method)))
        tasks.append(("baseline_tables", baseline_tables))
        return tasks
    requirements = ["solver", "distribution_method", "representation"]
    if part in ("f", "g", "h"):
        requirements.append("kmax")
    if part in ("g", "h"):
        requirements.append("family")
    if part == "h":
        requirements.append("N")
    s = selected(requirements)
    def case(name, N, bound, family):
        household_case(name, s["solver"], N, bound, family, distribution_method=s["distribution_method"], representation=s["representation"])
    if part == "e":
        for bound in (2, 5, 10, 20, 40):
            name = f"e_kmax{bound}"
            tasks.append((name, lambda name=name, bound=bound: case(name, 1000, bound, "uniform")))
        tasks.append(("ranges_table", lambda: experiment_table("ranges", [f"e_kmax{b}" for b in (2, 5, 10, 20, 40)])))
    elif part == "f":
        for family in ("uniform", "nonuniform"):
            name = f"f_{family}"
            tasks.append((name, lambda name=name, family=family: case(name, 1000, s["kmax"], family)))
        tasks.append(("grids_table", lambda: experiment_table("grids", ["f_uniform", "f_nonuniform"])))
    elif part == "g":
        for N in (100, 500, 1000, 2000, 5000):
            name = f"g_N{N}"
            tasks.append((name, lambda name=name, N=N: case(name, N, s["kmax"], s["family"])))
            tasks.append((name+"_figures", lambda N=N: node_figures(N)))
    elif part == "h":
        tasks.append(("accuracy_outputs", lambda: accuracy_outputs(s)))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(2, 3))
    parser.add_argument("--part", choices=("abc", "e", "f", "g", "h"))
    args = parser.parse_args()
    if args.stage and args.part:
        parser.error("Choose --stage or --part")
    RESULTS.mkdir(exist_ok=True)
    if args.stage:
        tasks = validation_tasks(args.stage)
        scope = f"stage{args.stage}"
    elif args.part:
        tasks = part_tasks(args.part)
        scope = f"part_{args.part}"
    else:
        # Until student choices exist, the default command reproduces stages 2–3.
        tasks = validation_tasks(3)
        scope = "all"
        if (ROOT / "code" / "selected_settings.json").exists():
            for part in ("abc", "e", "f", "g", "h"):
                tasks += part_tasks(part)
        else:
            print("No recorded stage-4 choices: reproducing validation only.", flush=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ledger_path = CHECKPOINTS / f"{scope}_completed.json"
    completed = set(json.loads(ledger_path.read_text()) if ledger_path.exists() else [])
    started = time.perf_counter()
    try:
        for key, action in tasks:
            if key in completed:
                continue
            if time.perf_counter()-started >= 120:
                raise ResumePending("Completed cases saved; rerun the same command")
            action()
            completed.add(key)
            json_write(ledger_path, sorted(completed))
    except ResumePending as exc:
        print(str(exc), flush=True)
        return 75
    if ledger_path.exists():
        ledger_path.unlink()
    if args.stage == 3 or not args.stage and not args.part:
        print("Stage 3 finished. Await prompts/01_experiments.md before running experiments.", flush=True)
    if args.part:
        print(f"Part {args.part} finished. Stop for the student's next choice/prompt.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
