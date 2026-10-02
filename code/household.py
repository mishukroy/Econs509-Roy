"""Discrete household model, using only definitions in spec.md."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

np.random.seed(0)
CAPS = {"vfi": 10000, "howard": 1000, "modified_howard": 10000,
        "gradient": 20000, "adam": 20000}


class ResumePending(RuntimeError):
    """The bounded call saved its state; rerun the same command to continue."""


def parameters():
    p = dict(beta=.96, sigma=1.5, labor=1., k_min=0., z=1., alpha=.36,
             delta=.1, K0=5., epsilon=[.8, 1.2], P=[[.5, .5], [.5, .5]])
    p["L"] = sum(.5 * eps * p["labor"] for eps in p["epsilon"])
    p["w"] = (1-p["alpha"]) * p["z"] * (p["K0"]/p["L"])**p["alpha"]
    p["r"] = p["alpha"] * p["z"] * (p["K0"]/p["L"])**(p["alpha"]-1) - p["delta"]
    return p


def grid(N, kmax, family="uniform"):
    if N < 2 or kmax <= 0:
        raise ValueError("N >= 2 and positive kmax required")
    t = np.arange(N, dtype=np.float64) / (N-1)
    if family == "uniform":
        k = kmax * t
    elif family == "nonuniform":
        k = np.exp(t*np.log(1+kmax)) - 1
    else:
        raise ValueError(f"Unknown grid family {family}")
    k[0], k[-1] = 0., float(kmax)
    return k


class Problem:
    def __init__(self, k, P=None):
        self.k = np.asarray(k, dtype=np.float64)
        self.N = len(self.k)
        self.M = 2*self.N
        self.p = parameters()
        self.P = np.asarray(self.p["P"] if P is None else P, dtype=np.float64)
        self.beta = self.p["beta"]
        started = time.perf_counter()
        resources = ((1+self.p["r"])*self.k[:, None]
                     + self.p["w"]*np.array(self.p["epsilon"])[None, :]*self.p["labor"])
        self.resources = resources
        c = resources.ravel(order="F")[:, None] - self.k[None, :]
        self.R = np.full(c.shape, -np.inf, dtype=np.float64)
        feasible = c > 0
        self.R[feasible] = c[feasible]**(1-self.p["sigma"])/(1-self.p["sigma"])
        self.setup_seconds = time.perf_counter()-started

    def bellman(self, x):
        V = np.asarray(x).reshape((self.N, 2), order="F")
        continuation = self.beta * (self.P @ V.T)
        G = np.empty(self.M, dtype=np.int64)
        Tx = np.empty(self.M, dtype=np.float64)
        # Broadcasting one shock block at a time avoids a full continuation matrix.
        for s in range(2):
            sl = slice(s*self.N, (s+1)*self.N)
            objectives = self.R[sl] + continuation[s][None, :]
            chosen = np.argmax(objectives, axis=1)  # First index wins exact ties.
            G[sl] = chosen
            Tx[sl] = objectives[np.arange(self.N), chosen]
        return Tx, G

    def policy_apply(self, G, x):
        V = np.asarray(x).reshape((self.N, 2), order="F")
        return np.concatenate([V[G[s*self.N:(s+1)*self.N]] @ self.P[s] for s in range(2)])

    def policy_transpose_apply(self, G, x):
        out = np.zeros(self.M, dtype=np.float64)
        for s in range(2):
            sl = slice(s*self.N, (s+1)*self.N)
            for sp in range(2):
                np.add.at(out, sp*self.N + G[sl], self.P[s, sp] * x[sl])
        return out


def transition(G, P, representation="sparse"):
    G = np.asarray(G, dtype=np.int64).ravel()
    N, M = len(G)//2, len(G)
    rows = np.repeat(np.arange(M), 2)
    cols = np.column_stack((G, N+G)).ravel()
    data = np.repeat(np.asarray(P), N, axis=0).ravel()
    if representation == "dense":
        Q = np.zeros((M, M), dtype=np.float64)
        Q[rows, cols] = data
        return Q
    if representation != "sparse":
        raise ValueError(representation)
    from scipy.sparse import csr_matrix
    return csr_matrix((data, (rows, cols)), shape=(M, M))


def _fingerprint(problem, settings):
    h = hashlib.sha256(problem.k.tobytes() + problem.P.tobytes())
    h.update(json.dumps({"parameters": problem.p, **settings}, sort_keys=True).encode())
    return h.hexdigest()


def solve(problem, method="vfi", tolerance=1e-8, checkpoint=None, max_seconds=120):
    """Return current values on the stopping/capped check, with no extra update.

    Checkpoints are written after an update and resume at the next check. All
    optimizer moments, traces, counts, settings and accumulated runtime survive.
    """
    if method not in CAPS:
        raise ValueError(method)
    cap = CAPS[method]
    settings = dict(method=method, tolerance=tolerance, cap=cap,
                    modified_howard_H=50, gradient_step=.0005, adam_learning_rate=.1,
                    adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
                    initialization="zeros", dtype="float64", seed=0)
    signature = _fingerprint(problem, settings)
    x = np.zeros(problem.M, dtype=np.float64)
    a, b = np.zeros_like(x), np.zeros_like(x)
    check, elapsed, setup_elapsed = 1, 0., problem.setup_seconds
    trace_d, trace_loss = [], []
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved["signature"]) != signature:
                    raise ValueError(f"Checkpoint settings mismatch: {checkpoint}")
                x, a, b = saved["x"].copy(), saved["a"].copy(), saved["b"].copy()
                check, elapsed = int(saved["next_check"]), float(saved["elapsed_seconds"])
                setup_elapsed += float(saved["setup_seconds"])
                trace_d, trace_loss = saved["trace_d"].tolist(), saved["trace_loss"].tolist()
    started = time.perf_counter()
    initial, _ = problem.bellman(np.zeros(problem.M))
    initial_metric = float(np.max(np.abs(initial)))
    while True:
        Tx, G = problem.bellman(x)
        F = x-Tx
        d = float(np.max(np.abs(F)/(np.abs(x)+1)))
        if not np.all(np.isfinite(x)) or not np.isfinite(d):
            raise FloatingPointError(f"{method}: numerical breakdown at check {check}")
        if method in ("gradient", "adam"):
            loss = float(.5*np.dot(F, F))
            if not np.isfinite(loss):
                raise FloatingPointError(f"{method}: nonfinite loss at check {check}")
            trace_d.append(d); trace_loss.append(loss)
        if d <= tolerance or check == cap:
            break
        chosen_R = problem.R[np.arange(problem.M), G]
        if method == "vfi":
            x = Tx
        elif method == "howard":
            from scipy.sparse import eye
            from scipy.sparse.linalg import spsolve
            Q = transition(G, problem.P)
            x = spsolve(eye(problem.M, format="csr") - problem.beta*Q, chosen_R)
        elif method == "modified_howard":
            x = Tx
            for _ in range(50):
                x = chosen_R + problem.beta*problem.policy_apply(G, x)
        else:
            h = F - problem.beta*problem.policy_transpose_apply(G, F)
            if method == "gradient":
                x = x - .0005*h
            else:
                j = check
                a = .9*a + .1*h
                b = .999*b + .001*h*h
                x = x - .1*(a/(1-.9**j))/(np.sqrt(b/(1-.999**j))+1e-8)
        check += 1
        if time.perf_counter()-started >= max_seconds:
            if checkpoint is None:
                raise RuntimeError("Long run needs a checkpoint path")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez(checkpoint, x=x, a=a, b=b, next_check=check,
                     elapsed_seconds=elapsed+time.perf_counter()-started,
                     setup_seconds=setup_elapsed, signature=signature,
                     settings=json.dumps(settings, sort_keys=True),
                     trace_d=np.array(trace_d), trace_loss=np.array(trace_loss))
            raise ResumePending(f"{method} saved after {check-1} updates; rerun the same command")
    # Re-evaluate exactly the saved x, including its greedy policy.
    final_T, final_G = problem.bellman(x)
    saved_d = float(np.max(np.abs(final_T-x)/(np.abs(x)+1)))
    elapsed += time.perf_counter()-started
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    metadata = {**settings, "outer_checks": check, "updates": check-1,
                "inner_steps": 50*(check-1) if method == "modified_howard" else 0,
                "exit_metric": saved_d, "initial_exit_metric": initial_metric,
                "status": "converged" if saved_d <= tolerance else "unconverged",
                "setup_seconds": setup_elapsed, "solver_seconds": elapsed,
                "total_seconds": setup_elapsed+elapsed}
    arrays = dict(k_grid=problem.k.copy(), V=x.reshape((problem.N, 2), order="F"),
                  G=final_G.reshape((problem.N, 2), order="F"))
    arrays["g"] = problem.k[arrays["G"]]
    arrays["c"] = problem.resources-arrays["g"]
    arrays.update(euler(problem, arrays["G"], arrays["c"]))
    trace = dict(bellman_metric=np.array(trace_d), loss=np.array(trace_loss),
                 outer_check=np.arange(1, len(trace_d)+1))
    return arrays, metadata, trace


def normalize_distribution(raw, apply_transpose):
    raw = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw)) or not np.isfinite(raw.sum()) or raw.sum() == 0:
        raise FloatingPointError("Nonfinite distribution or zero sum")
    pi = raw/raw.sum()
    minimum = float(pi.min())
    if minimum < -1e-14:
        raise FloatingPointError(f"Material negative probability: {minimum}")
    correction = float(-pi[pi < 0].sum())
    pi[pi < 0] = 0.
    pi /= pi.sum()
    normalization = float(abs(pi.sum()-1))
    residual = float(np.max(np.abs(apply_transpose(pi)-pi)))
    if normalization > 1e-10 or residual > 1e-10:
        raise RuntimeError(f"Invalid distribution: normalization={normalization}, stationarity={residual}")
    return pi, dict(raw_minimum=minimum, correction_mass=correction,
                   normalization_error=normalization, stationarity_residual=residual)


def invariant(Q, method="power", checkpoint=None, max_seconds=120):
    started = time.perf_counter()
    M = Q.shape[0]
    updates, elapsed = 0, 0.
    eigenvalue, imaginary = None, None
    signature = None
    if method == "power":
        raw = np.ones(M)/M
        # Matrix fingerprints protect resumed states against policy changes.
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if isinstance(Q, np.ndarray):
                signature = hashlib.sha256(Q.tobytes()).hexdigest()
            else:
                signature = hashlib.sha256(Q.data.tobytes()+Q.indices.tobytes()+Q.indptr.tobytes()).hexdigest()
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if str(saved["signature"]) != signature:
                        raise ValueError("Distribution checkpoint matrix mismatch")
                    raw, updates, elapsed = saved["pi"].copy(), int(saved["updates"]), float(saved["elapsed_seconds"])
        while updates < 100000:
            new = Q.T @ raw
            difference = float(np.max(np.abs(new-raw)))
            raw = new
            updates += 1
            if not np.all(np.isfinite(raw)):
                raise FloatingPointError("Power iteration produced nonfinite entries")
            if difference <= 1e-12:
                break
            if time.perf_counter()-started >= max_seconds:
                if checkpoint is None:
                    raise RuntimeError("Long power run needs a checkpoint path")
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                np.savez(checkpoint, pi=raw, updates=updates, signature=signature,
                         elapsed_seconds=elapsed+time.perf_counter()-started,
                         tolerance=1e-12, cap=100000)
                raise ResumePending(f"Power iteration saved after {updates} updates")
        else:
            raise RuntimeError("Power iteration did not converge within 100000 updates")
    elif method == "eigenvector":
        from scipy.sparse.linalg import eigs
        values, vectors = eigs(Q.T, k=1, sigma=1-1e-10, which="LM",
                              v0=np.ones(M)/np.sqrt(M), tol=1e-12, maxiter=100000)
        eigenvalue = float(values[0].real)
        imaginary = float(np.max(np.abs(vectors[:, 0].imag)))
        if abs(values[0]-1) > 1e-10 or imaginary > 1e-10:
            raise RuntimeError(f"Eigenpair invalid: eigenvalue={values[0]}, imaginary={imaginary}")
        raw = vectors[:, 0].real
    elif method == "equations":
        rhs = np.zeros(M); rhs[-1] = 1.
        if isinstance(Q, np.ndarray):
            A = np.eye(M)-Q.T
            A[-1, :] = 1.
            raw = np.linalg.solve(A, rhs)
        else:
            from scipy.sparse import eye
            from scipy.sparse.linalg import spsolve
            A = (eye(M, format="csr")-Q.T).tolil()
            A[-1, :] = np.ones(M)
            raw = spsolve(A.tocsr(), rhs)
    else:
        raise ValueError(method)
    pi, metrics = normalize_distribution(raw, lambda v: Q.T @ v)
    elapsed += time.perf_counter()-started
    if checkpoint is not None and checkpoint.exists():
        checkpoint.unlink()
    return pi, {"method": method, "updates": updates if method == "power" else None,
                "power_tolerance": 1e-12, "power_cap": 100000,
                "eigenvalue_real": eigenvalue, "eigenvector_imaginary_max": imaginary,
                "eigen_settings": dict(k=1, sigma=1-1e-10, which="LM", tol=1e-12, maxiter=100000,
                                       v0="ones(M)/sqrt(M)"),
                "distribution_seconds": elapsed, "status": "converged", **metrics}


def euler(problem, G, c):
    # next_c[n,s,sp] requires a separate index for the current shock.
    next_c = np.stack([c[G, sp] for sp in range(2)], axis=2)
    expectation = np.sum(next_c**(-problem.p["sigma"])*problem.P[None, :, :], axis=2)
    cee = (problem.beta*(1+problem.p["r"])*expectation)**(-1/problem.p["sigma"])
    E = np.abs(1-cee/c)
    return dict(E=E, slack_mask=problem.k[G] > 0, upper_bound_mask=G == problem.N-1)


def statistics(arrays):
    k, E, mask = arrays["k_grid"], arrays["E"], arrays["slack_mask"]
    stats = dict(grid_step=float(np.max(np.diff(k))), slack_count=int(mask.sum()),
                 upper_bound_count=int(arrays["upper_bound_mask"].sum()),
                 euler_max=float(E[mask].max()) if mask.any() else None,
                 euler_mean=float(E[mask].mean()) if mask.any() else None,
                 unavailable_reasons={})
    if not mask.any():
        stats["unavailable_reasons"]["unweighted_euler"] = "No states with positive saving"
    if "pi" not in arrays:
        stats["unavailable_reasons"]["distribution_statistics"] = "This solver run has no invariant distribution"
        return stats
    mass = arrays["pi"].reshape((len(k), 2), order="F")
    p = mass.sum(axis=1)
    support = p > 1e-12
    denominator = float(mass[mask].sum())
    q = mass[mask]/denominator if denominator > 0 else None
    supported = mask & (mass > 1e-12)
    stats.update(mean_assets=float(k @ p), top_mass=float(p[-1]),
                 support_endpoint=float(k[support].max()) if support.any() else None,
                 support_count=int(support.sum()), support_share=float(support.sum()/len(k)),
                 slack_probability=denominator,
                 euler_weighted_mean=float(q @ E[mask]) if q is not None else None,
                 maximum_weighted_contribution=float(np.max(q*E[mask])) if q is not None else None,
                 euler_support_max=float(E[supported].max()) if supported.any() else None)
    if q is None:
        stats["unavailable_reasons"]["weighted_euler"] = "Slack-state probability is zero"
    if not supported.any():
        stats["unavailable_reasons"]["euler_support_max"] = "No slack state has probability above 1e-12"
    if not support.any():
        stats["unavailable_reasons"]["support_endpoint"] = "No marginal asset mass exceeds 1e-12"
    return stats


def scaling_fit(cases):
    out = {}
    for metric in ("euler_mean", "euler_max"):
        observations = [(m["statistics"]["grid_step"], m["statistics"][metric]) for m in cases]
        pairs = [(h, error) for h, error in observations if error is not None and h > 0 and error > 0
                 and np.isfinite(h) and np.isfinite(error)]
        if len(pairs) >= 2 and len({h for h, _ in pairs}) >= 2:
            x, y = np.log(np.array(pairs)).T
            slope = float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
            intercept = float(y.mean()-slope*x.mean())
            out[metric] = dict(slope=slope, intercept=intercept, observations=pairs)
        else:
            out[metric] = dict(slope=None, intercept=None, observations=pairs,
                               unavailable_reason="Need two distinct positive finite grid steps and errors")
    return out
