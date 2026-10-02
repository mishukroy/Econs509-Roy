"""Independent checks keyed to the student's twelve test IDs.

Run with python tests/verify.py --stage 3 after code/run_all.py --stage 3.
The scalar model, Bellman, transition, and Euler calculations below do not use
the implementation's calculation routines.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


class Checks:
    def __init__(self, stage):
        self.stage = stage
        self.rows = []

    def write(self):
        target = ROOT / "results" / ("tests_stage3.json" if self.stage == 3 else "tests_final.json")
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({"stage": self.stage, "checks": self.rows,
                                     "passed": sum(r["status"] == "pass" for r in self.rows),
                                     "pending": sum(r["status"] == "pending" for r in self.rows),
                                     "failed": sum(r["status"] == "fail" for r in self.rows)},
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def add(self, test_id, case, measured, criterion, ok=None):
        self.rows.append({"id": test_id, "input": case, "measured": measured,
                          "acceptance_criterion": criterion,
                          "status": "pending" if ok is None else "pass" if bool(ok) else "fail"})
        if ok is not None and not bool(ok):
            self.write()
            raise RuntimeError(f"Test {test_id} failed for {case}: {measured}")


def read_case(name):
    with np.load(ROOT / "results" / f"{name}.npz", allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in f.files}
    meta = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
    return arrays, meta


def reference_interface():
    path = ROOT / "manual" / "kernel_output.npz"
    if not path.is_file():
        raise RuntimeError("Student reference manual/kernel_output.npz is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as f:
        required = {"V": (100, 2), "G": (100, 2), "pi": (200,), "iterations": ()}
        problems = [f"missing {key}" for key in required if key not in f.files]
        problems += [f"{key}: shape {f[key].shape}, expected {shape}"
                     for key, shape in required.items() if key in f.files and f[key].shape != shape]
        if "G" in f.files and (not np.issubdtype(f["G"].dtype, np.integer)
                               or np.any(f["G"] < 0) or np.any(f["G"] >= 100)):
            problems.append("G must contain integer zero-based indices in [0,99]")
        if problems:
            raise RuntimeError("Manual interface needs student revision: " + "; ".join(problems))
        return {key: f[key].copy() for key in required}, digest


def prices():
    return .36 * 5.0 ** (-.64) - .1, .64 * 5.0 ** .36


def scalar_bellman(k, V, P):
    r, w = prices()
    values = np.empty_like(V)
    policy = np.empty(V.shape, dtype=np.int64)
    for n, asset in enumerate(k):
        for s, eps in enumerate((.8, 1.2)):
            best, index = -np.inf, -1
            for j, saving in enumerate(k):
                c = (1 + r) * asset + w * eps - saving
                if c <= 0:
                    continue
                ev = sum(P[s, sp] * V[j, sp] for sp in range(2))
                objective = c ** (-.5) / (-.5) + .96 * ev
                if objective > best:
                    best, index = objective, j
            values[n, s], policy[n, s] = best, index
    return values, policy


def bellman_tests(checks):
    import household as impl
    expected = {"beta": .96, "sigma": 1.5, "labor": 1., "k_min": 0.,
                "z": 1., "alpha": .36, "delta": .1, "K0": 5.}
    r, w = prices()
    p = impl.parameters()
    discrepancies = {key: abs(p[key] - val) for key, val in expected.items()}
    discrepancies.update(r=abs(p["r"] - r), w=abs(p["w"] - w),
                         L=abs(p["L"] - 1.),
                         shock_order=float(np.max(np.abs(np.asarray(p["epsilon"]) - [.8, 1.2]))),
                         P=float(np.max(np.abs(np.asarray(p["P"]) - .5))))
    grid_errors = {}
    for family in ("uniform", "nonuniform"):
        for N in (100, 1000):
            t = np.arange(N) / (N - 1)
            expected_grid = 20 * t if family == "uniform" else np.exp(t * np.log(21)) - 1
            actual = impl.grid(N, 20, family)
            grid_errors[f"{family}_{N}"] = float(np.max(np.abs(actual - expected_grid)))
            assert actual[0] == 0 and actual[-1] == 20
    checks.add(7, "parameters, prices, shock order, both grid formulas", {**discrepancies, **grid_errors},
               "absolute discrepancies <= 1e-12; exact endpoints", max([*discrepancies.values(), *grid_errors.values()]) <= 1e-12)
    for label, k, P in [("validation", np.arange(100) * 20 / 99, np.full((2, 2), .5)),
                         ("asymmetric_shock_test", np.array([0., .5, 1., 2.]), np.array([[.8, .2], [.3, .7]]))]:
        V = (np.arange(len(k)) / (len(k) - 1))[:, None] ** 2 + .3 * np.arange(2)[None, :]
        problem = impl.Problem(k, P=P)
        actual, G = problem.bellman(V.ravel(order="F"))
        expected_values, expected_G = scalar_bellman(k, V, P)
        error = float(np.max(np.abs(actual - expected_values.ravel(order="F"))))
        mismatches = int(np.sum(G != expected_G.ravel(order="F")))
        bound = 1e-12 * max(1., float(np.max(np.abs(expected_values))))
        checks.add(7, label, {"max_value_difference": error, "policy_mismatches": mismatches},
                   f"value difference <= {bound}; zero policy mismatches", error <= bound and mismatches == 0)
    # An exact tie in the Bellman objective must pick its first column.
    tie_problem = impl.Problem(np.array([0., .5, 1., 2.]))
    tie_x = np.zeros(8)
    # Synthetic equal objectives isolate the tie rule from floating subtraction.
    tie_problem.R[0, :] = [-1., -1., -np.inf, -np.inf]
    _, tie_G = tie_problem.bellman(tie_x)
    checks.add(7, "exact tie at joint state 0", {"chosen_index": int(tie_G[0])},
               "smallest savings index, 0", tie_G[0] == 0)


def solver_check(checks, name):
    import household as impl
    a, m = read_case(name)
    problem = impl.Problem(a["k_grid"])
    x = a["V"].ravel(order="F")
    Tv, G = problem.bellman(x)
    d = float(np.max(np.abs(Tv - x) / (np.abs(x) + 1)))
    counts = m["solver"]
    method = counts["method"]
    caps = {"vfi": 10000, "howard": 1000, "modified_howard": 10000, "gradient": 20000, "adam": 20000}
    ok = (np.all(np.isfinite(x)) and abs(d - counts["exit_metric"]) <= 1e-12
          and np.array_equal(G, a["G"].ravel(order="F"))
          and counts["updates"] == counts["outer_checks"] - 1
          and counts["cap"] == caps[method]
          and 1 <= counts["outer_checks"] <= caps[method]
          and counts["initialization"] == "zeros" and counts["dtype"] == "float64")
    if counts["status"] == "converged":
        ok = ok and d <= counts["tolerance"]
    else:
        ok = ok and method in ("gradient", "adam") and counts["outer_checks"] == 20000 and d > counts["tolerance"]
    ok = ok and counts["modified_howard_H"] == 50 and counts["inner_steps"] == (50 * counts["updates"] if method == "modified_howard" else 0)
    ok = ok and counts["gradient_step"] == .0005 and counts["adam_learning_rate"] == .1
    ok = ok and counts["adam_beta1"] == .9 and counts["adam_beta2"] == .999 and counts["adam_epsilon"] == 1e-8
    initial, _ = problem.bellman(np.zeros_like(x))
    ok = ok and abs(counts["initial_exit_metric"] - np.max(np.abs(initial))) <= 1e-12
    checks.add(10, name, {"recomputed_metric": d, "reported_metric": counts["exit_metric"],
                         "outer_checks": counts["outer_checks"], "updates": counts["updates"],
                         "inner_steps": counts["inner_steps"], "status": counts["status"]},
               "saved metric agrees <= 1e-12; specified settings/counts, finite values and valid exit", ok)


def mathematical_checks(checks, name):
    a, m = read_case(name)
    k, G, c = a["k_grid"], a["G"], a["c"]
    N = len(k)
    violations = np.argwhere(np.diff(G, axis=0) < 0).tolist()
    checks.add(2, name, {"violation_count": len(violations), "locations": violations}, "no decreasing adjacent policy indices", not violations)
    valid = np.issubdtype(G.dtype, np.integer) and np.all((G >= 0) & (G < N))
    checks.add(3, name + ":indices", {"valid_integer_indices": bool(valid)}, "integer indices in [0,N-1]", valid)
    r, w = prices()
    expected_c = (1 + r) * k[:, None] + w * np.array([.8, 1.2])[None, :] - k[G]
    error = float(np.max(np.abs(expected_c - c)))
    state = np.unravel_index(np.argmin(expected_c), c.shape)
    checks.add(3, name, {"minimum_consumption": float(expected_c[state]), "minimum_state": list(map(int, state)), "budget_discrepancy": error},
               "finite positive consumption; saved budget agreement <= 1e-12", np.all(np.isfinite(c)) and np.min(c) > 0 and error <= 1e-12)
    if "pi" not in a:
        return
    pi = a["pi"]
    checks.add(4, name, {"sum": float(pi.sum()), "normalization_error": float(abs(pi.sum() - 1))},
               "finite entries; normalization error <= 1e-10", np.all(np.isfinite(pi)) and abs(pi.sum() - 1) <= 1e-10)
    dist = m["distribution"]
    checks.add(5, name, {"raw_minimum": dist["raw_minimum"], "correction_mass": dist["correction_mass"], "saved_minimum": float(pi.min())},
               "raw minimum >= -1e-14; saved minimum >= 0; correction mass nonnegative", dist["raw_minimum"] >= -1e-14 and pi.min() >= 0 and dist["correction_mass"] >= 0)
    top = float(pi.reshape((N, 2), order="F")[-1].sum())
    is_trial = name.startswith("e_")
    checks.add(6, name, {"N": N, "kmax": float(k[-1]), "top_mass": top,
                         "candidate_status": "unsuitable" if top > 1e-12 else "suitable"},
               "top mass <= 1e-12 on accepted grids; retain unsuitable range trials", is_trial or top <= 1e-12)
    import household as impl
    from scipy.sparse import csr_matrix
    rows, cols, data = [], [], []
    for s in range(2):
        for n in range(N):
            for sp in range(2):
                rows.append(s * N + n); cols.append(sp * N + int(G[n, s])); data.append(.5)
    expected_Q = csr_matrix((data, (rows, cols)), shape=(2 * N, 2 * N))
    actual_Q = impl.transition(G.ravel(order="F"), np.full((2, 2), .5))
    delta = actual_Q - expected_Q
    transition_error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.
    row_error = float(np.max(np.abs(np.asarray(actual_Q.sum(axis=1)).ravel() - 1)))
    stationarity = float(np.max(np.abs(expected_Q.T @ pi - pi)))
    marginals = pi.reshape((N, 2), order="F").sum(axis=0)
    checks.add(8, name, {"shape": list(actual_Q.shape), "transition_error": transition_error,
                         "row_sum_error": row_error, "stationarity_residual": stationarity, "shock_marginals": marginals.tolist()},
               "correct nonnegative transitions/shape; row/entry error <= 1e-12; stationarity/shock errors <= 1e-10",
               actual_Q.shape == (2*N, 2*N) and np.min(actual_Q.data) >= 0 and transition_error <= 1e-12 and row_error <= 1e-12
               and stationarity <= 1e-10 and np.max(np.abs(marginals - .5)) <= 1e-10)
    for n, s in [(0, 0), (N // 2, 1), (N - 1, 0)]:
        saving = int(G[n, s])
        next_cs = [(1+r)*k[saving] + w*eps - k[int(G[saving, sp])] for sp, eps in enumerate((.8, 1.2))]
        cee = (.96 * (1+r) * sum(.5 * nc**(-1.5) for nc in next_cs))**(-1/1.5)
        expected_E = abs(1 - cee / expected_c[n, s])
        error = abs(expected_E - a["E"][n, s])
        checks.add(11, f"{name}:state({n},{s})", {"scalar_E": float(expected_E), "saved_E": float(a["E"][n, s]), "difference": float(error)},
                   "scalar/saved discrepancy <= 1e-12*max(1,abs(E))", error <= 1e-12*max(1., abs(expected_E)))
    mask = k[G] > 0
    upper = G == N-1
    mass = pi.reshape((N, 2), order="F")
    p = mass.sum(axis=1)
    support = p > 1e-12
    E = a["E"]
    denominator = float(mass[mask].sum())
    q = mass[mask] / denominator if denominator > 0 else None
    supported = mask & (mass > 1e-12)
    stats = {"mean_assets": float(sum(float(k[n]*p[n]) for n in range(N))), "top_mass": top,
             "support_endpoint": float(k[support].max()) if np.any(support) else None,
             "support_share": float(np.count_nonzero(support)/N), "support_count": int(np.count_nonzero(support)),
             "grid_step": float(np.max(np.diff(k))), "slack_count": int(np.count_nonzero(mask)),
             "upper_bound_count": int(np.count_nonzero(upper)), "slack_probability": denominator,
             "euler_max": float(np.max(E[mask])) if np.any(mask) else None,
             "euler_mean": float(np.mean(E[mask])) if np.any(mask) else None,
             "euler_weighted_mean": float(sum(q*E[mask])) if q is not None else None,
             "maximum_weighted_contribution": float(np.max(q*E[mask])) if q is not None else None,
             "euler_support_max": float(np.max(E[supported])) if np.any(supported) else None}
    differences = {}
    ok = np.array_equal(mask, a["slack_mask"]) and np.array_equal(upper, a["upper_bound_mask"]) and np.array_equal(a["g"], k[G])
    for key, expected in stats.items():
        saved = m["statistics"][key]
        differences[key] = None if expected is None or saved is None else float(abs(saved - expected))
        if expected is None:
            ok = ok and saved is None and bool(m["statistics"]["unavailable_reasons"])
        elif isinstance(expected, int):
            ok = ok and saved == expected
        else:
            ok = ok and saved is not None and abs(saved - expected) <= 1e-12 * max(1., abs(expected))
    checks.add(11, name + ":statistics/masks", differences, "exact masks/counts; statistics agree within 1e-12*max(1,abs(reference))", ok)


def kernel_check(checks):
    reference, digest = reference_interface()
    a, m = read_case("validation_kernel_vfi")
    difference = float(np.max(np.abs(a["V"] - reference["V"])))
    mismatches = int(np.count_nonzero(a["G"] != reference["G"]))
    checks.add(1, "validation_kernel_vfi vs manual/kernel_output.npz", {"max_value_difference": difference, "policy_mismatches": mismatches,
               "agent_outer_checks": m["solver"]["outer_checks"], "manual_outer_checks": int(reference["iterations"]), "reference_sha256": digest,
               "pi_difference_diagnostic": float(np.max(np.abs(a["pi"] - reference["pi"])))},
               "max value difference <= 1e-8; zero policy mismatches", difference <= 1e-8 and mismatches == 0)
    return difference, mismatches


def validation_settings_check(checks, names):
    expected_prices = prices()
    expected_parameters = {"beta": .96, "sigma": 1.5, "labor": 1., "k_min": 0.,
                           "z": 1., "alpha": .36, "delta": .1, "K0": 5.,
                           "L": 1., "r": expected_prices[0], "w": expected_prices[1]}
    howard, _ = read_case("validation_howard")
    measured = {}
    ok = True
    for name in names:
        a, m = read_case(name)
        expected_method = ("howard" if "dense" in name or "sparse" in name or name == "validation_howard"
                           else "modified_howard" if name == "validation_modified_howard" else "vfi")
        expected_tolerance = 1e-10 if name == "validation_kernel_vfi" else 1e-8
        grid_error = float(np.max(np.abs(a["k_grid"] - 20*np.arange(100)/99)))
        parameter_error = max(abs(m["parameters"][key]-value) for key, value in expected_parameters.items())
        policy_mismatches = int(np.count_nonzero(a["G"] != howard["G"])) if "dense" in name or "sparse" in name else 0
        setting_ok = (m["grid"] == {"N": 100, "kmax": 20., "family": "uniform"}
                      and a["V"].shape == (100, 2) and a["G"].shape == (100, 2)
                      and a["k_grid"][0] == 0 and a["k_grid"][-1] == 20
                      and m["solver"]["method"] == expected_method
                      and m["solver"]["tolerance"] == expected_tolerance
                      and m["solver"]["seed"] == 0
                      and np.array_equal(m["parameters"]["epsilon"], [.8, 1.2])
                      and np.array_equal(m["parameters"]["P"], [[.5, .5], [.5, .5]])
                      and grid_error <= 1e-12 and parameter_error <= 1e-12 and policy_mismatches == 0)
        if "dense" in name or "sparse" in name:
            representation = "dense" if "dense" in name else "sparse"
            distribution_method = name.split(representation + "_", 1)[1]
            setting_ok = setting_ok and m["distribution"]["representation"] == representation
            setting_ok = setting_ok and m["distribution"]["method"] == distribution_method
            setting_ok = setting_ok and m["distribution"]["policy_source"] == "validation_howard"
        measured[name] = {"grid_error": grid_error, "parameter_error": float(parameter_error),
                          "Howard_policy_mismatches": policy_mismatches, "settings_match": bool(setting_ok)}
        ok = ok and setting_ok
    checks.add(9, "saved validation settings and common Howard policy", measured,
               "N=100 uniform [0,20]; specified model/tolerances/methods, exact discrete settings; numeric error <= 1e-12; common Howard policy", ok)


def verify(stage=3, kernel_only=False):
    checks = Checks(stage)
    difference, mismatches = kernel_check(checks)
    bellman_tests(checks)
    if kernel_only:
        solver_check(checks, "validation_kernel_vfi")
        for test_id in (2, 3, 4, 5, 6, 8, 9, 10, 11):
            checks.add(test_id, "remaining stage-3 validation", None,
                       "Requires Howard validation and six distribution runs; SciPy environment pending")
        checks.add(12, "stage-4 completeness and reproduction", None, "Full experiments and student choices required")
        checks.write()
        print(json.dumps({"passed": sum(r["status"] == "pass" for r in checks.rows),
                          "pending": sum(r["status"] == "pending" for r in checks.rows),
                          "kernel_value_difference": difference, "kernel_policy_mismatches": mismatches}, indent=2))
        return checks
    for name in ("validation_kernel_vfi", "validation_vfi", "validation_howard", "validation_modified_howard"):
        solver_check(checks, name)
    mathematical_checks(checks, "validation_howard")
    distribution_names = [f"validation_{rep}_{method}" for rep in ("dense", "sparse") for method in ("power", "eigenvector", "equations")]
    validation_settings_check(checks, ["validation_kernel_vfi", "validation_vfi", "validation_howard", "validation_modified_howard", *distribution_names])
    for name in distribution_names:
        mathematical_checks(checks, name)
    pairs = [(float(np.max(np.abs(read_case(a)[0]["pi"] - read_case(b)[0]["pi"]))), a, b)
             for a, b in itertools.combinations(distribution_names, 2)]
    worst = max(pairs)
    groups = {}
    for label, subset in [("within_dense", [p for p in pairs if "dense" in p[1] and "dense" in p[2]]),
                          ("within_sparse", [p for p in pairs if "sparse" in p[1] and "sparse" in p[2]]),
                          ("across_representations", [p for p in pairs if ("dense" in p[1]) != ("dense" in p[2])])]:
        maximum = max(subset)
        groups[label] = {"maximum_pairwise_difference": maximum[0], "pair": list(maximum[1:])}
    checks.add(9, "six Howard-policy validation distributions", {"maximum_pairwise_difference": worst[0], "pair": list(worst[1:]), **groups},
               "maximum pairwise sup-norm difference <= 1e-8", worst[0] <= 1e-8)
    import household as impl
    howard, _ = read_case("validation_howard")
    Q = impl.transition(howard["G"].ravel(order="F"), np.full((2, 2), .5))
    expected_dense = np.zeros((200, 200))
    for s in range(2):
        for n in range(100):
            for sp in range(2):
                expected_dense[s*100+n, sp*100+int(howard["G"][n, s])] = .5
    actual_dense = impl.transition(howard["G"].ravel(order="F"), np.full((2, 2), .5), "dense")
    dense_error = float(max(np.max(np.abs(Q.toarray()-actual_dense)), np.max(np.abs(expected_dense-actual_dense))))
    checks.add(8, "dense vs CSR validation transition", {"maximum_entry_difference": dense_error}, "entry difference <= 1e-12", dense_error <= 1e-12)
    checks.add(9, "full stage-4 experiments/settings", None, "specified baseline/range/grid/N runs and recorded choices")
    checks.add(10, "baseline gradient and Adam", None, "full specified settings/traces and valid capped/converged exit")
    checks.add(11, "final Euler plots and scaling slopes", None, "scalar/statistic agreement and independent OLS fit")
    checks.add(12, "stage-4 complete outputs and clean-copy reproduction", None, "all outputs and clean-copy regeneration")
    checks.write()
    print(json.dumps({"passed": sum(r["status"] == "pass" for r in checks.rows), "pending": sum(r["status"] == "pending" for r in checks.rows),
                      "kernel_value_difference": difference, "kernel_policy_mismatches": mismatches, "worst_distribution_difference": worst[0]}, indent=2))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[3], default=3)
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--kernel-only", action="store_true", help="Run the available NumPy checks while the SciPy environment is pending")
    args = parser.parse_args()
    if args.interface_only:
        ref, digest = reference_interface()
        print(json.dumps({"keys": list(ref), "shapes": {k: list(v.shape) for k, v in ref.items()}, "sha256": digest}))
    else:
        verify(args.stage, kernel_only=args.kernel_only)
