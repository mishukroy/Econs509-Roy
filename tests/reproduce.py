"""Bounded clean-copy reproduction under the assignment root; preserve originals."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import numpy as np
from verify import ROOT, Checks
from verify_stage4 import cases, required_files

SCRATCH = ROOT/".scratch"/"stage4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (SCRATCH/"results").exists():
        raise RuntimeError("Scratch already has results; use --run to resume. Preserve it rather than deleting it.")
    SCRATCH.mkdir(parents=True,exist_ok=True)
    for name in ("code","tests","prompts"):
        shutil.copytree(ROOT/name,SCRATCH/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name in ("AGENTS.md","spec.md","tests.md","log.md"):
        shutil.copy2(ROOT/name,SCRATCH/name)
    # The reference is copied solely for the required manual-kernel reproduction check.
    (SCRATCH/"manual").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"manual"/"kernel_output.npz",SCRATCH/"manual"/"kernel_output.npz")
    protected = ["spec.md","tests.md","code/selected_settings.json","manual/kernel_output.npz"]
    result_hashes = {p.name:digest(p) for p in (ROOT/"results").iterdir() if p.is_file()}
    state = dict(protected_hashes={name:digest(ROOT/name) for name in protected},result_hashes=result_hashes,
                 clean_start=not (SCRATCH/"results").exists(),complete=False)
    (SCRATCH/"audit.json").write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    print("Prepared scratch copy with no results and the recorded choices.",flush=True)


def run():
    state = json.loads((SCRATCH/"audit.json").read_text(encoding="utf-8"))
    process = subprocess.run([sys.executable,"code/run_all.py"],cwd=SCRATCH)
    if process.returncode == 0:
        state["complete"] = True
        (SCRATCH/"audit.json").write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    return process.returncode


def compare():
    state = json.loads((SCRATCH/"audit.json").read_text(encoding="utf-8"))
    checks = Checks(4,"reproduction.json")
    checks.add(12,"scratch clean start/completion",{"clean_start":state["clean_start"],"complete":state["complete"]},
               "no initial results; python code/run_all.py completed without new choices",state["clean_start"] and state["complete"])
    changed = [name for name,h in state["protected_hashes"].items() if digest(ROOT/name)!=h]
    changed += ["results/"+name for name,h in state["result_hashes"].items() if digest(ROOT/"results"/name)!=h]
    checks.add(12,"original files preserved",{"changed":changed},"original numerical outputs and protected settings/reference unchanged",not changed)
    missing = [name for name in required_files() if not (SCRATCH/"results"/name).is_file()]
    checks.add(12,"scratch output completeness",{"missing":missing},"all required outputs regenerate in scratch",not missing)
    checks.add(12,"scratch choice configuration",None,"recorded student choices reproduce exactly",
               digest(ROOT/"code"/"selected_settings.json")==digest(SCRATCH/"code"/"selected_settings.json"))
    paths = [name+".npz" for name in cases()]+["a_gradient_trace.npz","a_adam_trace.npz"]
    for name in paths:
        with np.load(ROOT/"results"/name,allow_pickle=False) as f: a={k:f[k] for k in f.files}
        with np.load(SCRATCH/"results"/name,allow_pickle=False) as f: b={k:f[k] for k in f.files}
        differences, ok = {}, set(a)==set(b)
        for key,x in a.items():
            y=b.get(key)
            if y is None or x.shape!=y.shape:
                ok=False; differences[key]="missing or shape mismatch"; continue
            if key in ("G","outer_check","slack_mask","upper_bound_mask"):
                good=np.array_equal(x,y); differences[key]=int(np.count_nonzero(x!=y))
            elif key == "k_grid":
                err=float(np.max(np.abs(x-y))); good=err<=1e-12; differences[key]=err
            else:
                err=float(np.max(np.abs(x-y)/np.maximum(1,np.abs(x)))); good=bool(np.all(np.isfinite(y))) and err<=1e-8
                differences[key]=err
            ok=ok and good
        checks.add(12,name,differences,"same shapes; exact policy/mask/check indices; assets <= 1e-12; other arrays elementwise scaled error <= 1e-8",ok)
    for name in cases():
        a=json.loads((ROOT/"results"/(name+".json")).read_text(encoding="utf-8"))
        b=json.loads((SCRATCH/"results"/(name+".json")).read_text(encoding="utf-8"))
        mismatches=[]
        for key in ("parameters","grid","selected_settings","source_case"):
            if a.get(key)!=b.get(key): mismatches.append(key)
        for key,value in a["statistics"].items():
            other=b["statistics"].get(key)
            good=(other==value if not isinstance(value,(float,int)) else other is not None and abs(value-other)<=1e-8*max(1,abs(value)))
            if not good: mismatches.append("statistics."+key)
        checks.add(12,name+":settings/statistics",{"mismatches":mismatches},"exact chosen settings/model/grid; statistics agree to scaled 1e-8; counts/runtimes may differ",not mismatches)
    fit_a=json.loads((ROOT/"results"/"h_scaling.json").read_text())
    fit_b=json.loads((SCRATCH/"results"/"h_scaling.json").read_text())
    error=max(abs(fit_a[key][field]-fit_b[key][field]) for key in fit_a for field in ("slope","intercept"))
    checks.add(12,"scratch fitted slopes",{"maximum_difference":error},"slope/intercept agreement <= 1e-8",error<=1e-8)
    report=json.loads((SCRATCH/"results"/"tests_final.json").read_text())
    checks.add(12,"scratch stricter applicable tests",{"passed":report["passed"],"failed":report["failed"],"pending":report["pending"]},
               "all applicable numerical/output checks pass; only recursive scratch reproduction can be pending",report["failed"]==0 and report["pending"]==1)
    checks.write()
    path=ROOT/"results"/"reproduction.json"
    report=json.loads(path.read_text()); report.update(complete=True,scratch_directory=str(SCRATCH))
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"reproduction_passed":report["passed"],"failed":report["failed"],"maximum_fit_difference":error},indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scratch-name",default="stage4",help="New or existing scratch folder name under .scratch")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true"); mode.add_argument("--run",action="store_true"); mode.add_argument("--compare",action="store_true")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.scratch_name):
        parser.error("scratch-name must be a single alphanumeric/hyphen/underscore folder name")
    SCRATCH=ROOT/".scratch"/args.scratch_name
    if args.prepare: prepare()
    elif args.run: sys.exit(run())
    else: compare()
