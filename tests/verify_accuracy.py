"""Independent part-(h) numerical, fitted-slope and plot checks."""
import argparse
import json
import numpy as np
from verify import ROOT, Checks, mathematical_checks, read_case, solver_check


def accuracy_plot_checks(checks, euler_figure, scaling_figure):
    chosen = json.loads((ROOT/"code"/"selected_settings.json").read_text(encoding="utf-8"))
    a, _ = read_case("h_final")
    ok = len(euler_figure.axes) == 2
    for s, ax in enumerate(euler_figure.axes):
        mask = a["g"][:,s] > 0
        ok = ok and len(ax.lines) == 1
        if ok:
            x, y = ax.lines[0].get_data()
            ok = (np.array_equal(x,a["k_grid"][mask]) and np.array_equal(y,a["E"][mask,s])
                  and ax.get_title() == f"Efficiency {[.8,1.2][s]}, N={chosen['N']}")
    checks.add(12,"h_euler:plotted data",{"slack_states":int((a["g"]>0).sum())},
               "exact saved assets/residuals on each shock's positive-saving mask; both shocks/N labeled",ok)
    fit = json.loads((ROOT/"results"/"h_scaling.json").read_text(encoding="utf-8"))
    ax = scaling_figure.axes[0]
    ok = len(scaling_figure.axes) == 1 and len(ax.lines) == 2 and ax.get_xscale() == ax.get_yscale() == "log"
    for line, key in zip(ax.lines,("euler_mean","euler_max")):
        pairs = np.asarray(fit[key]["observations"])
        x, y = line.get_data()
        ok = ok and np.array_equal(x,pairs[:,0]) and np.array_equal(y,pairs[:,1])
    checks.add(12,"h_scaling:plotted data",{"observations_per_curve":[len(fit[k]["observations"]) for k in ("euler_mean","euler_max")]},
               "both log-log curves equal saved positive finite step/error observations",ok)


def verify_accuracy():
    checks = Checks(4,"tests_h.json")
    chosen = json.loads((ROOT/"code"/"selected_settings.json").read_text(encoding="utf-8"))
    a, m = read_case("h_final")
    original, source = read_case(f"g_N{chosen['N']}")
    identical = all(np.array_equal(a[k],original[k]) for k in original if k not in ("E","slack_mask","upper_bound_mask"))
    settings_ok = (m["source_case"] == f"g_N{chosen['N']}" and m["selected_settings"] == chosen
                   and m["grid"] == dict(N=chosen["N"],kmax=chosen["kmax"],family=chosen["family"])
                   and m["solver"]["method"] == chosen["solver"]
                   and m["distribution"]["method"] == chosen["distribution_method"]
                   and m["distribution"]["representation"] == chosen["representation"] and identical)
    checks.add(9,"h_final:selected settings/source",{"unchanged_solution_arrays":identical,"settings":chosen},
               "final Euler analysis uses only the chosen methods/grid/N and retains its solved policy/value/distribution",settings_ok)
    solver_check(checks,"h_final")
    mathematical_checks(checks,"h_final")
    fit = json.loads((ROOT/"results"/"h_scaling.json").read_text(encoding="utf-8"))
    for key in ("euler_mean","euler_max"):
        pairs = []
        for N in (100,500,1000,2000,5000):
            arrays, meta = read_case(f"g_N{N}")
            h = float(np.max(np.diff(arrays["k_grid"])))
            errors = arrays["E"][arrays["g"]>0]
            err = float(np.mean(errors) if key == "euler_mean" else np.max(errors))
            if h > 0 and err > 0 and np.isfinite(h) and np.isfinite(err):
                pairs.append((h,err))
        x, y = np.log(np.asarray(pairs)).T
        # Independent least-squares system, rather than the implementation's covariance formula.
        slope, intercept = np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
        error = max(abs(slope-fit[key]["slope"]),abs(intercept-fit[key]["intercept"]))
        ok = np.allclose(pairs,fit[key]["observations"],rtol=0,atol=1e-12) and error <= 1e-12*max(1,abs(slope),abs(intercept))
        checks.add(11,"h_scaling:"+key,{"observations":pairs,"slope":float(slope),"intercept":float(intercept),"fit_discrepancy":float(error)},
                   "five positive finite observations recomputed from arrays; independent OLS agrees to scaled 1e-12; no required slope",ok)
    plot_report = json.loads((ROOT/"results"/"tests_h_figures.json").read_text(encoding="utf-8"))
    checks.rows.extend(plot_report["checks"])
    missing, invalid = [], []
    for stem in ("h_euler","h_scaling"):
        for ext, header in (("pdf",b"%PDF"),("png",b"\x89PNG\r\n\x1a\n")):
            path = ROOT/"results"/f"{stem}.{ext}"
            if not path.is_file(): missing.append(path.name)
            elif not path.read_bytes().startswith(header): invalid.append(path.name)
    checks.add(12,"part h:figure files",{"missing":missing,"invalid":invalid},"both h figures exist in PDF/PNG with passing data checks",
               not missing and not invalid and plot_report["passed"] == 2 and plot_report["failed"] == 0)
    # Reuse the already independently checked five-size rows for the required accuracy table.
    md = (ROOT/"results"/"tables.md").read_text(encoding="utf-8")
    tex = (ROOT/"results"/"accuracy.tex").read_text(encoding="utf-8")
    node_tex = (ROOT/"results"/"nodes.tex").read_text(encoding="utf-8")
    node_md = md.split("<!-- table:nodes -->",1)[1].split("<!-- end:nodes -->",1)[0]
    accuracy_md = md.split("<!-- table:accuracy -->",1)[1].split("<!-- end:accuracy -->",1)[0]
    checks.add(12,"accuracy:table contents",{"latex_matches_nodes":tex == node_tex},
               "complete accuracy table repeats the verified five-size JSON-derived rows in LaTeX/Markdown",
               tex == node_tex and node_md.replace("## nodes","## accuracy").strip() == accuracy_md.strip())
    summary = json.loads((ROOT/"results"/"h_summary.json").read_text(encoding="utf-8"))
    checks.add(11,"h_summary:statistics/slopes",None,"summary repeats final statistics, scaling and chosen settings exactly",
               summary["statistics"] == m["statistics"] and summary["scaling"] == fit and summary["selected_settings"] == chosen)
    checks.add(12,"Stage-4 full completeness/README/scratch reproduction",None,"Pending complete Stage-4 audit")
    checks.write()
    print(json.dumps({"passed":sum(r['status']=='pass' for r in checks.rows),"failed":sum(r['status']=='fail' for r in checks.rows),
                      "pending":sum(r['status']=='pending' for r in checks.rows),"statistics":m["statistics"],"scaling":fit},indent=2))
    return checks


if __name__ == "__main__":
    verify_accuracy()
