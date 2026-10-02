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
import hashlib
import itertools
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

from household import (CAPS, Problem, ResumePending, grid, invariant, parameters,
                       scaling_fit, solve, statistics, transition, euler)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / ".checkpoints"
CALL_DEADLINE = None
MATRIX_CACHE = {}


def remaining_seconds():
    return 50. if CALL_DEADLINE is None else max(.001, CALL_DEADLINE-time.perf_counter())


def check_experiment_case(name):
    if name.startswith(("a_", "bc_", "e_", "f_", "g_")):
        sys.path.insert(0, str(ROOT / "tests"))
        from verify_experiments import verify_case
        verify_case(name)


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
    pi, metadata = invariant(Q, method, CHECKPOINTS / f"{case}_distribution.npz", max_seconds=remaining_seconds())
    arrays["pi"] = pi
    metadata.update(representation=representation, matrix_construction_seconds=construction_seconds,
                    matrix_conversion_seconds=0.)
    return metadata


def household_case(name, method, N=100, kmax=20., family="uniform", tolerance=1e-8,
                   distribution_method=None, representation="sparse"):
    problem = Problem(grid(N, kmax, family))
    arrays, solver, trace = solve(problem, method, tolerance, CHECKPOINTS / f"{name}_solver.npz", max_seconds=remaining_seconds())
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
    check_experiment_case(name)
    print(json.dumps({"case": name, "outer_checks": solver["outer_checks"], "exit_metric": solver["exit_metric"],
                      "status": solver["status"], "value_min": float(arrays["V"].min()),
                      "value_max": float(arrays["V"].max()), "mean_assets": meta["statistics"].get("mean_assets")}), flush=True)


def distribution_case(name, source, representation, method):
    arrays, original = load_case(source)
    meta = copy.deepcopy(original)
    meta["case"] = name
    P = np.asarray(parameters()["P"], dtype=np.float64)
    fingerprint = hashlib.sha256(arrays["G"].tobytes()+P.tobytes()).hexdigest()
    sparse_path = CHECKPOINTS/f"matrix_{fingerprint}.npz"
    dense_path = CHECKPOINTS/f"matrix_{fingerprint}.npy"
    timing_path = CHECKPOINTS/f"matrix_{fingerprint}.json"
    if fingerprint not in MATRIX_CACHE:
        from scipy.sparse import load_npz, save_npz
        if sparse_path.exists() and timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            started = time.perf_counter()
            sparse_Q = load_npz(sparse_path)
            MATRIX_CACHE[fingerprint] = dict(sparse=sparse_Q, construction=timing["construction"],
                                             conversion=timing["conversion"], load_seconds=time.perf_counter()-started)
        else:
            started = time.perf_counter()
            sparse_Q = transition(arrays["G"].ravel(order="F"), P)
            MATRIX_CACHE[fingerprint] = dict(sparse=sparse_Q, construction=time.perf_counter()-started, conversion=0., load_seconds=0.)
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            save_npz(sparse_path, sparse_Q)
            json_write(timing_path, dict(construction=MATRIX_CACHE[fingerprint]["construction"], conversion=0.))
    common = MATRIX_CACHE[fingerprint]
    if representation == "dense" and "dense" not in common:
        started = time.perf_counter()
        if dense_path.exists():
            common["dense"] = np.load(dense_path, allow_pickle=False)
            common["load_seconds"] += time.perf_counter()-started
        else:
            common["dense"] = common["sparse"].toarray()
            common["conversion"] = time.perf_counter()-started
            np.save(dense_path, common["dense"])
            json_write(timing_path, dict(construction=common["construction"], conversion=common["conversion"]))
    Q = common[representation]
    construction, conversion = common["construction"], common["conversion"] if representation == "dense" else 0.
    if name.startswith("bc_"):
        json_write(RESULTS/"bc_matrix.json", dict(policy_source=source, fingerprint=fingerprint,
                   shape=list(Q.shape), common_construction_seconds=common["construction"],
                   dense_conversion_seconds=common["conversion"],
                   matrix_load_seconds=common["load_seconds"],
                   timing_note="Shared matrix setup, excluded from each distribution algorithm's solve time"))
    pi, dist = invariant(Q, method, CHECKPOINTS / f"{name}_distribution.npz", max_seconds=remaining_seconds())
    dist.update(representation=representation, matrix_construction_seconds=construction,
                matrix_conversion_seconds=conversion, matrix_load_seconds=common["load_seconds"], policy_source=source)
    arrays["pi"] = pi
    meta.update(distribution=dist, statistics=statistics(arrays))
    write_case(name, arrays, meta)
    check_experiment_case(name)
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


def conditioning_case(label):
    from scipy.sparse import eye
    from scipy.sparse.linalg import eigsh
    started = time.perf_counter()
    a, _ = load_case("a_howard")
    problem = Problem(a["k_grid"])
    G = (problem.bellman(np.zeros(problem.M))[1] if label == "zero"
         else a["G"].ravel(order="F"))
    J = eye(problem.M, format="csr")-.96*transition(G, problem.P)
    H = J.T @ J
    v0 = np.ones(problem.M)/np.sqrt(problem.M)
    try:
        small, vs = eigsh(H, k=1, sigma=0., which="LM", v0=v0, tol=1e-10, maxiter=100000)
        large, vl = eigsh(H, k=1, which="LA", v0=v0, tol=1e-10, maxiter=100000)
        if not (np.isfinite(small[0]) and np.isfinite(large[0]) and small[0] > 0):
            raise ValueError("Nonpositive or nonfinite squared singular values")
        minimum, maximum = float(np.sqrt(small[0])), float(np.sqrt(large[0]))
        result = dict(status="computed", singular_min=minimum, singular_max=maximum,
                      cond2_J=maximum/minimum, cond2_loss_Hessian=(maximum/minimum)**2,
                      small_eigenpair_residual=float(np.max(np.abs(H@vs[:,0]-small[0]*vs[:,0]))),
                      large_eigenpair_residual=float(np.max(np.abs(H@vl[:,0]-large[0]*vl[:,0]))),
                      method="Eigenvalues of J.T@J using deterministic eigsh, shift-invert for the minimum")
    except Exception as exc:
        result = dict(status="unavailable", reason=str(exc))
    result["seconds"] = time.perf_counter()-started
    path = RESULTS/"d_theory.json"
    theory = json.loads(path.read_text(encoding="utf-8"))
    theory.setdefault("conditioning", {})[label] = result
    theory["singular_values_status"] = "See conditioning entries for computed values or feasibility limitations"
    json_write(path, theory)
    print(json.dumps({"conditioning_case": label, **result}), flush=True)


def verify_baseline():
    sys.path.insert(0, str(ROOT / "tests"))
    from verify_experiments import verify_baseline as check
    check()


def baseline_summary():
    report = json.loads((RESULTS/"tests_abc.json").read_text(encoding="utf-8"))
    solvers = {method: load_case(f"a_{method}")[1]["solver"] for method in CAPS}
    distribution_names = [f"bc_{rep}_{method}" for rep in ("dense","sparse") for method in ("power","eigenvector","equations")]
    dists = {name: load_case(name)[1]["distribution"] for name in distribution_names}
    theory = json.loads((RESULTS/"d_theory.json").read_text(encoding="utf-8"))
    upper_bound = (1+.96*np.sqrt(2000))**2
    theory.update(joint_state_count=2000, local_loss_hessian_norm_upper_bound=float(upper_bound),
                  fixed_gradient_step=.0005, local_fixed_step_upper_bound=float(2/upper_bound),
                  fixed_step_below_local_bound=bool(.0005 < 2/upper_bound))
    json_write(RESULTS/"d_theory.json", theory)
    if report["failed"]:
        raise RuntimeError("Baseline checks failed; no methods may be selected")
    json_write(RESULTS/"abc_summary.json", dict(scope="parts_a_b_c", stage4_complete=False,
               eligible_solvers=[method for method,s in solvers.items() if s["status"] == "converged"],
               capped_solvers=[method for method,s in solvers.items() if s["status"] == "unconverged"],
               eligible_distribution_cases=[name for name,d in dists.items() if d["status"] == "converged"],
               test_passes=report["passed"], test_failures=report["failed"], later_pending_groups=report["pending"],
               runtime_seconds=dict(a_solver=sum(s["solver_seconds"] for s in solvers.values()),
                                    a_utility_setup=sum(s["setup_seconds"] for s in solvers.values()),
                                    a_total=sum(s["total_seconds"] for s in solvers.values()),
                                    b_algorithms=sum(d["distribution_seconds"] for name,d in dists.items() if "dense" in name),
                                    c_algorithms=sum(d["distribution_seconds"] for name,d in dists.items() if "sparse" in name)),
               common_matrix_setup=json.loads((RESULTS/"bc_matrix.json").read_text(encoding="utf-8")),
               timing_notes=["Solver runtimes accumulate active computation across resumed calls; utility setup is separate",
                             "Howard time includes the first SciPy import in this baseline run",
                             "Distribution timings include equation setup, solve/iteration and normalization; matrix setup is shared and separate",
                             "Checkpoint serialization, script startup, verification and time between calls are outside the solver timers"],
               awaiting_choices=["household_solver", "distribution_method", "matrix_representation"]))


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
        fig.savefig(RESULTS / f"{stem}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def node_figures(N):
    plt = plotting()
    a, m = load_case(f"g_N{N}")
    sys.path.insert(0,str(ROOT/"tests"))
    from verify_experiments import Checks, node_plot_check, node_figure_files
    checks = Checks(4,"tests_g_figures.json")
    report = RESULTS/"tests_g_figures.json"
    if report.exists():
        checks.rows = [row for row in json.loads(report.read_text(encoding="utf-8"))["checks"]
                       if not row["input"].startswith(f"g_N{N}:")]
    for stem, field, label in (("value", "V", "Value"), ("policy", "g", "Next-period assets"), ("distribution", "pi", "Joint probability mass")):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        data = a[field].reshape((N, 2), order="F")
        for s in range(2):
            ax[s].plot(a["k_grid"], data[:, s])
            ax[s].set(xlabel="Current assets", ylabel=label, title=f"Efficiency {parameters()['epsilon'][s]}, N={N}")
        node_plot_check(checks,f"g_N{N}",stem,fig)
        save_figure(plt, fig, f"g_N{N}_{stem}")
    node_figure_files(checks,f"g_N{N}")
    checks.write()


def accuracy_outputs(settings):
    plt = plotting()
    started = time.perf_counter()
    metas = [load_case(f"g_N{N}")[1] for N in (100, 500, 1000, 2000, 5000)]
    fit = scaling_fit(metas)
    json_write(RESULTS / "h_scaling.json", fit)
    a, m = load_case(f"g_N{settings['N']}")
    p = parameters()
    problem = SimpleNamespace(N=len(a["k_grid"]),k=a["k_grid"],p=p,P=np.asarray(p["P"]),beta=p["beta"])
    a.update(euler(problem,a["G"],a["c"]))
    m = copy.deepcopy(m)
    m.update(case="h_final",source_case=f"g_N{settings['N']}",selected_settings=settings,statistics=statistics(a))
    analysis_seconds = time.perf_counter()-started
    write_case("h_final",a,m)
    json_write(RESULTS/"h_summary.json",dict(selected_settings=settings,source_case=m["source_case"],
               statistics=m["statistics"],scaling=fit,analysis_seconds=analysis_seconds,
               timing_definition="Loading saved part-g solutions, fits, scaling JSON write and Euler/statistic recomputation; excludes final-array/table exports and plot rendering/export."))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for s in range(2):
        mask = a["slack_mask"][:, s]
        ax[s].plot(a["k_grid"][mask], a["E"][mask, s])
        ax[s].set(xlabel="Current assets", ylabel="Absolute relative Euler error", title=f"Efficiency {parameters()['epsilon'][s]}, N={settings['N']}")
    euler_fig = fig
    scaling_fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("euler_mean", "Mean"), ("euler_max", "Maximum")):
        pairs = np.asarray(fit[key]["observations"])
        if pairs.size:
            ax.loglog(pairs[:, 0], pairs[:, 1], "o-", label=label)
    ax.set(xlabel="Maximum grid step", ylabel="Absolute relative Euler error")
    ax.legend()
    sys.path.insert(0,str(ROOT/"tests"))
    from verify_accuracy import Checks, accuracy_plot_checks
    checks = Checks(4,"tests_h_figures.json")
    accuracy_plot_checks(checks,euler_fig,scaling_fig)
    checks.write()
    save_figure(plt,euler_fig,"h_euler")
    save_figure(plt,scaling_fig,"h_scaling")
    experiment_table("accuracy", [f"g_N{N}" for N in (100, 500, 1000, 2000, 5000)])


def selected(required, verify_methods=True):
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
    if verify_methods:
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


def part_tasks(part, reproducing=False):
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
        tasks.append(("conditioning_zero", lambda: conditioning_case("zero")))
        tasks.append(("conditioning_howard", lambda: conditioning_case("howard")))
        tasks.append(("baseline_verification", verify_baseline))
        tasks.append(("baseline_summary", baseline_summary))
        return tasks
    requirements = ["solver", "distribution_method", "representation"]
    if part in ("f", "g", "h"):
        requirements.append("kmax")
    if part in ("g", "h"):
        requirements.append("family")
    if part == "h":
        requirements.append("N")
    # In clean-copy reproduction the baseline will be generated by earlier tasks.
    s = selected(requirements, verify_methods=not reproducing)
    def case(name, N, bound, family):
        selected(requirements)
        household_case(name, s["solver"], N, bound, family, distribution_method=s["distribution_method"], representation=s["representation"])
    if part == "e":
        for bound in (2, 5, 10, 20, 40):
            name = f"e_kmax{bound}"
            tasks.append((name, lambda name=name, bound=bound: case(name, 1000, bound, "uniform")))
        tasks.append(("ranges_table", lambda: experiment_table("ranges", [f"e_kmax{b}" for b in (2, 5, 10, 20, 40)])))
        def verify_part_e():
            sys.path.insert(0, str(ROOT / "tests"))
            from verify_experiments import verify_ranges
            verify_ranges()
        tasks.append(("ranges_verification", verify_part_e))
    elif part == "f":
        for family in ("uniform", "nonuniform"):
            name = f"f_{family}"
            tasks.append((name, lambda name=name, family=family: case(name, 1000, s["kmax"], family)))
        tasks.append(("grids_table", lambda: experiment_table("grids", ["f_uniform", "f_nonuniform"])))
        def verify_part_f():
            sys.path.insert(0, str(ROOT / "tests"))
            from verify_experiments import verify_grids
            verify_grids()
        tasks.append(("grids_verification", verify_part_f))
    elif part == "g":
        for N in (100, 500, 1000, 2000, 5000):
            name = f"g_N{N}"
            tasks.append((name, lambda name=name, N=N: case(name, N, s["kmax"], s["family"])))
            tasks.append((name+"_figures", lambda N=N: node_figures(N)))
        tasks.append(("nodes_table",lambda:experiment_table("nodes",[f"g_N{N}" for N in (100,500,1000,2000,5000)])))
        def verify_part_g():
            sys.path.insert(0,str(ROOT/"tests"))
            from verify_experiments import verify_nodes
            verify_nodes()
        tasks.append(("nodes_verification",verify_part_g))
    elif part == "h":
        tasks.append(("accuracy_outputs", lambda: accuracy_outputs(s)))
        def verify_part_h():
            sys.path.insert(0,str(ROOT/"tests"))
            from verify_accuracy import verify_accuracy
            verify_accuracy()
        tasks.append(("accuracy_verification",verify_part_h))
    return tasks


def main():
    global CALL_DEADLINE
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
        # Reproduce completed scope without supplying any provisional choices.
        tasks = validation_tasks(3)
        scope = "all"
        choice_path = ROOT/"code"/"selected_settings.json"
        if (ROOT/"prompts"/"01_experiments.md").exists() or choice_path.exists():
            tasks += part_tasks("abc", reproducing=True)
        if choice_path.exists():
            choices = json.loads(choice_path.read_text(encoding="utf-8"))
            method_keys = {"solver", "distribution_method", "representation"}
            for part, keys in [("e",method_keys), ("f",method_keys|{"kmax"}),
                               ("g",method_keys|{"kmax","family"}), ("h",method_keys|{"kmax","family","N"})]:
                if keys <= choices.keys():
                    tasks += part_tasks(part, reproducing=True)
            if {"solver","distribution_method","representation","kmax","family","N"} <= choices.keys():
                from stage4_documents import write_documents
                def final_verification():
                    sys.path.insert(0,str(ROOT/"tests"))
                    from verify_stage4 import verify_final
                    verify_final()
                tasks += [("stage4_documents",write_documents),("stage4_verification",final_verification)]
        else:
            print("No recorded choices: reproduction stops after the authorized baseline comparisons.", flush=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ledger_path = CHECKPOINTS / f"{scope}_completed.json"
    completed = set(json.loads(ledger_path.read_text()) if ledger_path.exists() else [])
    started = time.perf_counter()
    CALL_DEADLINE = started+50.
    try:
        for key, action in tasks:
            if key in completed:
                continue
            if time.perf_counter() >= CALL_DEADLINE:
                raise ResumePending("Completed cases saved; rerun the same command")
            action()
            completed.add(key)
            json_write(ledger_path, sorted(completed))
    except ResumePending as exc:
        print(str(exc), flush=True)
        return 75
    if ledger_path.exists():
        ledger_path.unlink()
    if args.stage == 3:
        print("Stage 3 finished. Await prompts/01_experiments.md before running experiments.", flush=True)
    if not args.stage and not args.part:
        print("Reproduction finished for the available recorded choices. Stop for the next student choice/prompt.", flush=True)
    if args.part:
        print(f"Part {args.part} finished. Stop for the student's next choice/prompt.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
