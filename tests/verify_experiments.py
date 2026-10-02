"""Stage-4 checks, recorded immediately after each completed experiment case."""
import argparse
import itertools
import json

import numpy as np

from verify import ROOT, Checks, bellman_tests, mathematical_checks, prices, read_case, solver_check


def output_settings(checks, name):
    a, m = read_case(name)
    k = a["k_grid"]
    N = 1000
    expected_grid = 20*np.arange(N)/(N-1)
    error = float(np.max(np.abs(k-expected_grid))) if k.shape == (N,) else None
    r, w = prices()
    expected = dict(beta=.96, sigma=1.5, labor=1., k_min=0., z=1., alpha=.36, delta=.1, K0=5., L=1., r=r, w=w)
    param_error = max(abs(m["parameters"][key]-value) for key, value in expected.items())
    ok = (m["grid"] == dict(N=1000, kmax=20., family="uniform")
          and error is not None and error <= 1e-12 and param_error <= 1e-12
          and a["V"].shape == (N, 2) and a["G"].shape == (N, 2)
          and k[0] == 0 and k[-1] == 20
          and m["solver"]["tolerance"] == 1e-8 and m["solver"]["seed"] == 0
          and np.array_equal(m["parameters"]["epsilon"], [.8, 1.2])
          and np.array_equal(m["parameters"]["P"], [[.5, .5], [.5, .5]])
          and set(m["arrays"]) == set(a))
    mismatches = None
    if name.startswith("bc_"):
        source, _ = read_case("a_howard")
        mismatches = int(np.count_nonzero(a["G"] != source["G"]))
        representation, method = name.split("_", 2)[1:]
        d = m["distribution"]
        ok = ok and mismatches == 0 and np.array_equal(a["V"], source["V"])
        ok = ok and d["policy_source"] == "a_howard" and d["representation"] == representation and d["method"] == method
        ok = ok and d["power_tolerance"] == 1e-12 and d["power_cap"] == 100000
        if method == "power":
            ok = ok and 1 <= d["updates"] <= 100000
        if method == "eigenvector":
            ok = ok and abs(d["eigenvalue_real"]-1) <= 1e-10 and d["eigenvector_imaginary_max"] <= 1e-10
            ok = ok and d["eigen_settings"] == dict(k=1, sigma=1-1e-10, which="LM", tol=1e-12, maxiter=100000, v0="ones(M)/sqrt(M)")
    else:
        ok = ok and m["solver"]["method"] == name[2:]
    checks.add(9, name+":baseline settings", {"grid_error": error, "parameter_error": float(param_error), "Howard_policy_mismatches": mismatches},
               "specified baseline N/grid/parameters/tolerance/seed; recorded arrays; common Howard policy and specified distribution settings", ok)


def selected_grid_settings(checks, name):
    a, m = read_case(name)
    chosen = json.loads((ROOT/"code"/"selected_settings.json").read_text(encoding="utf-8"))
    is_range = name.startswith("e_")
    is_nodes = name.startswith("g_")
    bound = int(name.removeprefix("e_kmax")) if is_range else chosen["kmax"]
    family = "uniform" if is_range else chosen["family"] if is_nodes else name.removeprefix("f_")
    N = int(name.removeprefix("g_N")) if is_nodes else 1000
    k = a["k_grid"]
    expected_grid = (bound*np.arange(N)/(N-1) if family == "uniform"
                     else np.exp(np.arange(N)/(N-1)*np.log(1+bound))-1)
    expected_grid[0], expected_grid[-1] = 0., float(bound)
    error = float(np.max(np.abs(k-expected_grid))) if k.shape == (N,) else None
    r, w = prices()
    expected = dict(beta=.96, sigma=1.5, labor=1., k_min=0., z=1., alpha=.36, delta=.1, K0=5., L=1., r=r, w=w)
    param_error = max(abs(m["parameters"][key]-value) for key, value in expected.items())
    d = m["distribution"]
    baseline = json.loads((ROOT/"results"/"tests_abc.json").read_text(encoding="utf-8"))
    selected_names = ["a_"+chosen["solver"], "bc_"+chosen["representation"]+"_"+chosen["distribution_method"]]
    eligibility = all(any(row["status"] == "pass" and row["input"].startswith(case+":") for row in baseline["checks"])
                      and not any(row["status"] != "pass" and (row["input"].startswith(case+":") or row["input"] == case) for row in baseline["checks"])
                      for case in selected_names)
    accepted_bound = True
    accepted_family = True
    if not is_range:
        _, trial = read_case(f"e_kmax{bound}")
        report = json.loads((ROOT/"results"/"tests_e.json").read_text(encoding="utf-8"))
        accepted_bound = (trial["statistics"]["top_mass"] <= 1e-12
                          and any(row["input"] == f"e_kmax{bound}" and row["id"] == 6 and row["status"] == "pass" for row in report["checks"]))
    if is_nodes:
        _, prior_grid = read_case("f_"+family)
        report = json.loads((ROOT/"results"/"tests_f.json").read_text(encoding="utf-8"))
        accepted_family = (prior_grid["statistics"]["top_mass"] <= 1e-12
                           and any(row["input"] == "f_"+family and row["id"] == 6 and row["status"] == "pass" for row in report["checks"]))
    ok = (bound in (2,5,10,20,40) and family in ("uniform","nonuniform")
          and (N in (100,500,1000,2000,5000) if is_nodes else N == 1000)
          and m["grid"] == dict(N=N, kmax=bound, family=family)
          and error is not None and error <= 1e-12 and param_error <= 1e-12
          and k[0] == 0 and k[-1] == bound and a["V"].shape == (N,2) and a["G"].shape == (N,2)
          and set(a) == {"k_grid","V","G","g","c","pi","E","slack_mask","upper_bound_mask"}
          and set(m["arrays"]) == set(a) and a["pi"].shape == (2*N,)
          and m["solver"]["method"] == chosen["solver"] and m["solver"]["tolerance"] == 1e-8
          and m["solver"]["seed"] == 0 and m["solver"]["status"] == "converged"
          and d["method"] == chosen["distribution_method"] and d["representation"] == chosen["representation"]
          and d["status"] == "converged" and d["power_tolerance"] == 1e-12 and d["power_cap"] == 100000
          and (d["method"] != "power" or 1 <= d["updates"] <= 100000)
          and np.array_equal(m["parameters"]["epsilon"], [.8,1.2])
          and np.array_equal(m["parameters"]["P"], [[.5,.5],[.5,.5]]) and eligibility and accepted_bound and accepted_family)
    checks.add(9, name+":selected grid settings", {"grid_error":error, "parameter_error":float(param_error),
               "solver":m["solver"]["method"], "distribution_method":d["method"], "representation":d["representation"],
               "baseline_eligibility":eligibility, "accepted_bound":accepted_bound, "accepted_family":accepted_family, "family":family, "N":N},
               "prescribed node count and grid formula; selected bound for f/g and selected family for g; unchanged model/seed/tolerances; eligible selected methods; complete arrays", ok)


def unweighted_euler(checks, name):
    a, m = read_case(name)
    k, G, E = a["k_grid"], a["G"], a["E"]
    r, w = prices()
    for n, s in [(0, 0), (len(k)//2, 1), (len(k)-1, 0)]:
        j = int(G[n, s])
        c = (1+r)*k[n]+w*[.8, 1.2][s]-k[j]
        cs = [(1+r)*k[j]+w*eps-k[int(G[j, sp])] for sp, eps in enumerate([.8, 1.2])]
        residual = abs(1-(.96*(1+r)*sum(.5*nc**(-1.5) for nc in cs))**(-1/1.5)/c)
        error = float(abs(residual-E[n, s]))
        checks.add(11, f"{name}:state({n},{s})", {"scalar_E": float(residual), "saved_E": float(E[n, s]), "difference": error},
                   "scalar/saved discrepancy <= 1e-12*max(1,abs(E))", error <= 1e-12*max(1, abs(residual)))
    mask = k[G] > 0
    upper = G == len(k)-1
    expected = dict(grid_step=float(np.max(np.diff(k))), slack_count=int(mask.sum()), upper_bound_count=int(upper.sum()),
                    euler_max=float(E[mask].max()) if mask.any() else None,
                    euler_mean=float(E[mask].mean()) if mask.any() else None)
    ok = np.array_equal(mask, a["slack_mask"]) and np.array_equal(upper, a["upper_bound_mask"]) and np.array_equal(k[G], a["g"])
    differences = {}
    for key, value in expected.items():
        saved = m["statistics"][key]
        differences[key] = None if saved is None or value is None else float(abs(saved-value))
        ok = ok and (saved == value if isinstance(value, int) or value is None else abs(saved-value) <= 1e-12*max(1,abs(value)))
    checks.add(11, name+":unweighted statistics", differences, "exact masks/counts; summary discrepancy <= 1e-12*max(1,abs(reference))", ok)


def gradient_trace(checks, name):
    import household as impl
    a, m = read_case(name)
    count = m["solver"]["outer_checks"]
    with np.load(ROOT / "results" / f"{name}_trace.npz", allow_pickle=False) as f:
        d, loss, indices = f["bellman_metric"], f["loss"], f["outer_check"]
    problem = impl.Problem(a["k_grid"])
    initial, _ = problem.bellman(np.zeros(2000))
    final, _ = problem.bellman(a["V"].ravel(order="F"))
    F = a["V"].ravel(order="F")-final
    expected_final_loss = float(.5*np.dot(F,F))
    expected_initial_loss = float(.5*np.dot(initial,initial))
    ok = (len(d) == count and len(loss) == count and np.array_equal(indices, np.arange(1,count+1))
          and np.all(np.isfinite(d)) and np.all(np.isfinite(loss)) and np.all(loss >= 0)
          and abs(d[0]-m["solver"]["initial_exit_metric"]) <= 1e-12
          and abs(d[-1]-m["solver"]["exit_metric"]) <= 1e-12
          and abs(loss[0]-expected_initial_loss) <= 1e-12*max(1,abs(expected_initial_loss))
          and abs(loss[-1]-expected_final_loss) <= 1e-12*max(1,abs(expected_final_loss)))
    checks.add(10, name+":trace", {"checks": count, "trace_length": len(d), "initial_loss": float(loss[0]),
                                  "final_loss": float(loss[-1]), "recomputed_final_loss": expected_final_loss},
               "one finite loss/metric per outer check, exact check indices; recomputed endpoint metrics/losses agree", ok)


def verify_case(name, checks=None):
    own = checks is None
    if own:
        report = f"tests_{name[0]}.json" if name.startswith(("e_","f_","g_")) else "tests_abc.json"
        checks = Checks(4, report)
        path = ROOT / "results" / report
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            checks.rows = [r for r in prior["checks"] if not r["input"].startswith(name+":") and r["input"] != name]
    selected_grid_settings(checks, name) if name.startswith(("e_","f_","g_")) else output_settings(checks, name)
    if name.startswith(("e_","f_","g_")):
        solver_check(checks, name)
    if name.startswith("a_"):
        solver_check(checks, name)
        _, m = read_case(name)
        if name in ("a_gradient", "a_adam"):
            gradient_trace(checks, name)
        if m["solver"]["status"] == "converged":
            mathematical_checks(checks, name)
            unweighted_euler(checks, name)
    else:
        mathematical_checks(checks, name)
    if own:
        checks.write()
    return checks


def verify_grid_comparison(part):
    checks = Checks(4, f"tests_{part}.json")
    names = ([f"e_kmax{b}" for b in (2,5,10,20,40)] if part == "e" else ["f_uniform","f_nonuniform"]
             if part == "f" else [f"g_N{N}" for N in (100,500,1000,2000,5000)])
    stem = {"e":"ranges","f":"grids","g":"nodes"}[part]
    for name in names:
        verify_case(name, checks)
    markdown = (ROOT/"results"/"tables.md").read_text(encoding="utf-8")
    tex = (ROOT/"results"/f"{stem}.tex").read_text(encoding="utf-8")
    missing, summary = [], []
    for name in names:
        _, m = read_case(name)
        s, d = m["statistics"], m["distribution"]
        runtime = m["solver"]["total_seconds"]+d["distribution_seconds"]+d["matrix_construction_seconds"]+d["matrix_conversion_seconds"]
        status = "unsuitable" if s["top_mass"] > 1e-12 else "suitable"
        values = [name, m["grid"]["N"], m["grid"]["kmax"]]+[s[key] for key in
                  ("mean_assets","top_mass","support_endpoint","support_share","grid_step","euler_mean","euler_max",
                   "euler_weighted_mean","maximum_weighted_contribution","euler_support_max")]+[runtime,status]
        formatted = ["unavailable" if v is None else f"{v:.12g}" if isinstance(v,float) else str(v) for v in values]
        if ("| "+" | ".join(formatted)+" |" not in markdown
                or " & ".join(v.replace("_",r"\_") for v in formatted)+r" \\" not in tex):
            missing.append(name)
        summary.append(dict(case=name, N=m["grid"]["N"], family=m["grid"]["family"], kmax=m["grid"]["kmax"], top_mass=s["top_mass"],
                            support_endpoint=s["support_endpoint"], support_share=s["support_share"],
                            total_seconds=runtime, candidate_status=status, outer_checks=m["solver"]["outer_checks"],
                            exit_metric=m["solver"]["exit_metric"], stationarity_residual=d["stationarity_residual"],
                            mean_assets=s["mean_assets"], grid_step=s["grid_step"], euler_mean=s["euler_mean"],
                            euler_max=s["euler_max"], euler_weighted_mean=s["euler_weighted_mean"],
                            maximum_weighted_contribution=s["maximum_weighted_contribution"], euler_support_max=s["euler_support_max"]))
    checks.add(12, stem+":partial table contents", {"mismatched_rows":missing},
               "all prescribed JSON-derived rows and unchanged top masses in complete LaTeX tabular and Markdown", not missing
               and r"\begin{tabular}" in tex and r"\end{tabular}" in tex)
    if part == "g":
        figures = json.loads((ROOT/"results"/"tests_g_figures.json").read_text(encoding="utf-8"))
        checks.rows.extend(figures["checks"])
        actual = {row["input"] for row in figures["checks"] if row["status"] == "pass"}
        expected = {name+":"+stem+":plotted data" for name in names for stem in ("value","policy","distribution")}
        expected |= {name+":figure files" for name in names}
        checks.add(12,"part g:complete figure checks", {"missing_checks":sorted(expected-actual),"expected_count":len(expected)},
                   "all 15 plots' data and all five PDF/PNG file groups verified", expected <= actual and figures["failed"] == 0)
    checks.add(9, "remaining parts and student choices", None,
               "Pending student's upper-bound, grid-family and final-N choices" if part == "e" else
               "Pending parts g-h and student's grid-family/final-N choices" if part == "f" else "Pending part h and student's final-N choice")
    checks.add(11, "final Euler/scaling observations and slopes", None, "Pending part h")
    checks.add(12, "complete stage-4 outputs, README and scratch reproduction", None, "Pending remaining experiments and recorded choices")
    checks.write()
    result = dict(cases=summary, passed=sum(r["status"] == "pass" for r in checks.rows),
                  failed=sum(r["status"] == "fail" for r in checks.rows), pending=sum(r["status"] == "pending" for r in checks.rows),
                  total_seconds=sum(row["total_seconds"] for row in summary),
                  timing_definition="Household utility setup plus solver plus transition construction/conversion plus invariant distribution; excludes verification and script startup.")
    (ROOT/"results"/f"{part}_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return checks


def verify_ranges():
    return verify_grid_comparison("e")


def verify_grids():
    return verify_grid_comparison("f")


def verify_nodes():
    return verify_grid_comparison("g")


def node_plot_check(checks, name, stem, figure):
    a, m = read_case(name)
    field = {"value":"V","policy":"g","distribution":"pi"}[stem]
    data = a[field].reshape((len(a["k_grid"]),2),order="F")
    ok = len(figure.axes) == 2
    differences = []
    for s, ax in enumerate(figure.axes):
        valid = s < 2 and len(ax.lines) == 1
        if valid:
            x, y = ax.lines[0].get_data()
            valid = np.array_equal(x,a["k_grid"]) and np.array_equal(y,data[:,s])
            differences.append(dict(shock=s, x_error=float(np.max(np.abs(x-a["k_grid"]))),
                                    y_error=float(np.max(np.abs(y-data[:,s])))))
        ok = ok and valid and ax.get_title() == f"Efficiency {m['parameters']['epsilon'][s]}, N={len(a['k_grid'])}"
        if stem == "distribution":
            ok = ok and ax.get_ylabel() == "Joint probability mass"
    checks.add(12,name+":"+stem+":plotted data",differences,
               "exact assets and saved values/policies/joint masses for both labeled shocks; nonuniform plots are masses",ok)


def node_figure_files(checks, name):
    missing, invalid = [], []
    for stem in ("value","policy","distribution"):
        for suffix, header in (("pdf",b"%PDF"),("png",b"\x89PNG\r\n\x1a\n")):
            path = ROOT/"results"/f"{name}_{stem}.{suffix}"
            if not path.is_file():
                missing.append(path.name)
            elif path.stat().st_size <= len(header) or not path.read_bytes().startswith(header):
                invalid.append(path.name)
    checks.add(12,name+":figure files",dict(missing=missing,invalid=invalid),
               "value/policy/joint-mass figures exist as nonempty PDF and PNG",not missing and not invalid)


def verify_baseline():
    import household as impl
    checks = Checks(4, "tests_abc.json")
    bellman_tests(checks)
    for method in impl.CAPS:
        verify_case(f"a_{method}", checks)
    names = [f"bc_{rep}_{method}" for rep in ("dense", "sparse") for method in ("power", "eigenvector", "equations")]
    for name in names:
        verify_case(name, checks)
    pairs = [(float(np.max(np.abs(read_case(a)[0]["pi"]-read_case(b)[0]["pi"]))), a, b) for a,b in itertools.combinations(names,2)]
    worst = max(pairs)
    checks.add(9, "baseline distribution agreement", {"maximum_difference": worst[0], "pair": list(worst[1:])}, "maximum pairwise difference <= 1e-8", worst[0] <= 1e-8)
    a, _ = read_case("a_howard")
    dense = impl.transition(a["G"].ravel(order="F"), np.full((2,2), .5), "dense")
    sparse = impl.transition(a["G"].ravel(order="F"), np.full((2,2), .5))
    error = float(np.max(np.abs(dense-sparse.toarray())))
    checks.add(8, "baseline dense/CSR transition agreement", {"maximum_entry_difference": error}, "entry difference <= 1e-12", error <= 1e-12)
    # Check every printed entry against the saved numbers in the writer's format.
    tables = {
        "solvers": [(method, read_case(f"a_{method}")[1]["solver"]) for method in impl.CAPS],
        "distribution_dense": [(method, read_case(f"bc_dense_{method}")[1]["distribution"]) for method in ("power","eigenvector","equations")],
        "distribution_sparse": [(method, read_case(f"bc_sparse_{method}")[1]["distribution"]) for method in ("power","eigenvector","equations")]}
    markdown = (ROOT/"results"/"tables.md").read_text(encoding="utf-8")
    for stem, entries in tables.items():
        tex = (ROOT/"results"/f"{stem}.tex").read_text(encoding="utf-8")
        def format_value(v):
            return "unavailable" if v is None else f"{v:.12g}" if isinstance(v,float) else str(v)
        missing = []
        for method, row in entries:
            keys = (["outer_checks","updates","inner_steps","setup_seconds","solver_seconds","total_seconds","exit_metric","status"]
                    if stem == "solvers" else ["distribution_seconds","updates","matrix_construction_seconds","matrix_conversion_seconds","stationarity_residual","normalization_error","raw_minimum","correction_mass"])
            values = [method]+[format_value(row[key]) for key in keys]
            md_row = "| "+" | ".join(values)+" |"
            tex_row = " & ".join(v.replace("_",r"\_") for v in values)+r" \\"
            if md_row not in markdown or tex_row not in tex:
                missing.append(method)
        checks.add(12, stem+":partial table contents", {"mismatched_rows": missing}, "complete tabular and exact formatted rows from numerical JSON in LaTeX/Markdown",
                   not missing and r"\begin{tabular}" in tex and r"\end{tabular}" in tex)
    checks.add(9, "parts e-h and recorded student choices", None, "Pending the student's binding choices")
    checks.add(11, "final Euler/scaling observations and slopes", None, "Pending parts g-h")
    checks.add(12, "complete stage-4 outputs, README and scratch reproduction", None, "Pending remaining experiments and recorded choices")
    checks.write()
    print(json.dumps({"baseline_passed":sum(r["status"]=="pass" for r in checks.rows), "pending":sum(r["status"]=="pending" for r in checks.rows),
                      "failed":sum(r["status"]=="fail" for r in checks.rows), "maximum_distribution_difference":worst[0]},indent=2), flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--ranges", action="store_true")
    parser.add_argument("--grids", action="store_true")
    parser.add_argument("--nodes", action="store_true")
    args = parser.parse_args()
    verify_case(args.case) if args.case else verify_ranges() if args.ranges else verify_grids() if args.grids else verify_nodes() if args.nodes else verify_baseline()
