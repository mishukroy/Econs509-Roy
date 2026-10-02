"""Consolidate the independently run portions of all twelve test IDs."""
import argparse
import json
import numpy as np
from verify import ROOT, Checks, verify


def cases():
    return (["validation_kernel_vfi","validation_vfi","validation_howard","validation_modified_howard"]
            +[f"validation_{rep}_{method}" for rep in ("dense","sparse") for method in ("power","eigenvector","equations")]
            +["a_"+m for m in ("vfi","howard","modified_howard","gradient","adam")]
            +[f"bc_{rep}_{method}" for rep in ("dense","sparse") for method in ("power","eigenvector","equations")]
            +[f"e_kmax{b}" for b in (2,5,10,20,40)]+["f_uniform","f_nonuniform"]
            +[f"g_N{n}" for n in (100,500,1000,2000,5000)]+["h_final"])


def required_files():
    files = [name+ext for name in cases() for ext in (".npz",".json")]
    files += ["a_gradient_trace.npz","a_adam_trace.npz","tables.md"]
    files += [stem+".tex" for stem in ("solvers","distribution_dense","distribution_sparse","ranges","grids","nodes","accuracy")]
    files += [f"g_N{n}_{stem}.{ext}" for n in (100,500,1000,2000,5000) for stem in ("value","policy","distribution") for ext in ("pdf","png")]
    files += [stem+"."+ext for stem in ("h_euler","h_scaling") for ext in ("pdf","png")]
    files += [stem+".json" for stem in ("tests_stage3","tests_abc","tests_e","tests_f","tests_g","tests_h",
              "tests_g_figures","tests_h_figures","abc_summary","e_summary","f_summary","g_summary","h_summary",
              "bc_agreement","bc_matrix","d_theory","h_scaling","stage4_summary")]
    return files


def verify_final(refresh=False):
    if refresh:
        from verify_experiments import verify_baseline, verify_ranges, verify_grids, verify_nodes
        from verify_accuracy import verify_accuracy
        verify(3)
        verify_baseline(); verify_ranges(); verify_grids(); verify_nodes(); verify_accuracy()
    checks = Checks(4,"tests_final.json")
    # Earlier pending placeholders are superseded by the freshly run corresponding stage-4 portions.
    for stem in ("tests_stage3","tests_abc","tests_e","tests_f","tests_g","tests_h"):
        report = json.loads((ROOT/"results"/f"{stem}.json").read_text(encoding="utf-8"))
        checks.rows.extend(row for row in report["checks"] if row["status"] != "pending")
    chosen = json.loads((ROOT/"code"/"selected_settings.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT/"results"/"h_final.json").read_text(encoding="utf-8"))
    ok = (all(key in chosen for key in ("solver","distribution_method","representation","kmax","family","N"))
          and all(chosen["selection_reasons"].get(key) for key in ("household_solver","invariant_distribution","asset_range","grid_family","final_N"))
          and final["selected_settings"] == chosen)
    checks.add(9,"all recorded student choices/reasons",chosen,"all choices/reasons recorded and used by final analysis",ok)
    theory = json.loads((ROOT/"results"/"d_theory.json").read_text(encoding="utf-8"))
    prediction = float(np.log(1e-8)/np.log(.96))
    checks.add(9,"theory prediction",{"saved":theory["contraction_prediction"],"recomputed":prediction},
               "recorded contraction prediction agrees to absolute 1e-12",abs(theory["contraction_prediction"]-prediction)<=1e-12)
    missing = [name for name in required_files() if not (ROOT/"results"/name).is_file()]
    checks.add(12,"Stage-4 output completeness",{"required_file_count":len(required_files()),"missing":missing},
               "all required numerical files, traces, tables, figures and block verification reports exist",not missing)
    summary = json.loads((ROOT/"results"/"stage4_summary.json").read_text(encoding="utf-8"))
    readme = (ROOT/"README.md").read_text(encoding="utf-8")
    expected = ["python code/run_all.py",summary["python_version"],"spec.md"]
    expected += [k+"=="+v for k,v in summary["packages"].items()]
    expected += [format(v,".9g") for v in summary["runtime_seconds"].values() if v is not None]
    missing = [text for text in expected if text not in readme]
    checks.add(12,"README/environment/runtime",{"missing_items":missing,"environment":summary["packages"]},
               "reader guide states contents, regeneration command, queried versions and each measured part runtime",not missing)
    path = ROOT/"results"/"reproduction.json"
    if path.exists():
        reproduction = json.loads(path.read_text(encoding="utf-8"))
        checks.rows.extend(reproduction["checks"])
        checks.add(12,"scratch reproduction completion",{"passed":reproduction["passed"],"failed":reproduction["failed"]},
                   "clean-copy run finished and every numerical comparison passed",reproduction["failed"] == 0 and reproduction["complete"])
    else:
        checks.add(12,"clean-copy reproduction",None,"Pending regeneration in an empty-results scratch copy and elementwise comparison")
    checks.write()
    result = dict(passed=sum(r["status"]=="pass" for r in checks.rows),failed=sum(r["status"]=="fail" for r in checks.rows),
                  pending=sum(r["status"]=="pending" for r in checks.rows))
    print(json.dumps(result,indent=2),flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",action="store_true")
    verify_final(parser.parse_args().refresh)
