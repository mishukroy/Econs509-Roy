# PS1 specification: Discrete State Dynamic Programming

## 1. Task

I solve the Question 2 household problem and compute its joint invariant distribution. The comparisons cover the four discrete-state solvers, including fixed-step gradient descent and Adam, and three distribution algorithms with dense and sparse matrices. I then examine the asset range, nonuniform spacing, number of grid points, and Euler-equation accuracy. Prices are fixed at the given off-equilibrium capital. I choose the methods and grids from these results and explain my choices.

## 2. Context

### 2.1 Model, parameters, and prices

Consider a household with current assets $k$ and labor-efficiency state $s$. Its choice of next-period assets $k'$ solves

$$
v(k,s)=\max_{k'\in\mathcal K,\ k'\geq0,\ c>0}
\left\{u(c)+\beta\sum_{s'=1}^{2}\mathcal P_{ss'}v(k',s')\right\},
\qquad c=(1+r^0)k+w^0\bar\epsilon_s\bar l-k',
\qquad u(c)=\frac{c^{1-\sigma}}{1-\sigma}.
$$

The state space and the choice of savings use the same asset grid. Choices giving consumption at or below zero are infeasible. The parameters are

| Parameter | Value | Parameter | Value |
|---|---:|---|---:|
| $\beta$ | 0.96 | $\sigma$ | 1.5 |
| $\bar l$ | 1 | $\underline k$ | 0 |
| $z$ | 1 | $\alpha$ | 0.36 |
| $\delta$ | 0.1 | $K^0$ | 5 |

The two labor-efficiency states are ordered as $\bar\epsilon_1=0.8$, $\bar\epsilon_2=1.2$. Their transition matrix and stationary probabilities are

$$
\mathcal P=\begin{pmatrix}0.5&0.5\\0.5&0.5\end{pmatrix},
\qquad \mathcal P_{ss'}=\Pr(s_{t+1}=s'\mid s_t=s),
\qquad \pi^{shock}=(0.5,0.5).
$$

The firm has Cobb–Douglas production, with aggregate labor evaluated at the stationary shock distribution:

$$
Y=zK^\alpha L^{1-\alpha},\qquad
L^*=\sum_s\pi_s^{shock}\bar\epsilon_s\bar l=1,
$$

$$
w^0=(1-\alpha)z(K^0/L^*)^\alpha,
\qquad r^0=\alpha z(K^0/L^*)^{\alpha-1}-\delta.
$$

Use the price formulas at full precision; approximately $w^0=1.1423762763050482$ and $r^0=0.0285173310843179$. Since this is an off-equilibrium exercise, the computed mean of household assets need not equal $K^0$.

### 2.2 Discrete-state objects

Let $S=2$, $M=NS$, and index the joint state by $m(n,s)=(s-1)N+n$ using one-based mathematical indices. The Python implementation uses zero-based indices, `m = s*N+n`, and column stacking with `order="F"`. The discrete-state objects are

| Object | Meaning | Shape |
|---|---|---|
| $\mathcal V$, $x=\mathcal V(:)$ | Values and stacked values | $N\times2$, $M$ |
| $\mathcal R$ | Utility for each current state and savings choice | $M\times N$ |
| $\mathcal G$, $g$ | Policy indices and assets $g_{n,s}=k_{\mathcal G(n,s)}$ | $N\times2$ each |
| $\mathcal Q_G$ | Joint transition under policy $G$ | $M\times M$ |
| $\mathcal R_G^*$, $\pi$ | Chosen utility and joint probability mass | $M$ each |

The flow-utility matrix has entries $\mathcal R[m(n,s),n']=u((1+r^0)k_n+w^0\bar\epsilon_s\bar l-k_{n'})$ when consumption is positive, and $-\infty$ otherwise. The stacked Bellman operator is

$$
\mathcal T(x)=\max_{\text{columns}}
\left[\mathcal R+\beta(\mathcal P\mathcal V^T)\otimes\mathbf1_N\right].
$$

Taking the argmax across columns gives $\mathcal G(:)$. The maximization searches all feasible grid choices; an exact tie is resolved by choosing the smallest index. The policy-induced transition matrix and chosen utility vector satisfy

$$
\mathcal Q_G[m(n,s),m(n',s')]=\mathcal P_{ss'}\mathbf1\{n'=\mathcal G(n,s)\},
\qquad \mathcal R_G^*[m(n,s)]=\mathcal R[m(n,s),\mathcal G(n,s)].
$$

### 2.3 The four solvers

All household runs use float64 and start from a zero value function. Following the lecture's optimization, check, and update steps, first compute $\widehat x=\mathcal T(x)$ and the greedy policy. The percentage-change criterion is

$$
d(x)=\max_m\frac{|\widehat x_m-x_m|}{|x_m|+1}\leq\tau,
\qquad \tau=10^{-8}.
$$

When the criterion holds, return the current $x$ and its greedy policy. Count every outer check, including the stopping check. A failed check at the cap returns current values as unconverged, without another update. Thus updates equal outer checks minus one at exit. Recompute the saved-value criterion; gradient methods use this Bellman check even if their steps become small.

| Method | Update after a failed check, before the cap | Cap |
|---|---|---:|
| VFI | $x\leftarrow\widehat x$ | 10,000 |
| Howard | Solve $(I-\beta\mathcal Q_G)x_{new}=\mathcal R_G^*$ using sparse `spsolve` | 1,000 |
| Modified Howard | $B_0=\widehat x$; $B_h=\mathcal R_G^*+\beta\mathcal Q_G B_{h-1}$ for $h=1,\ldots,50$; $x\leftarrow B_{50}$ | 10,000 |
| Fixed-step gradient descent | $x\leftarrow x-\eta h$, $\eta=5\times10^{-4}$ | 20,000 |
| Adam | Moment update below, learning rate 0.1 | 20,000 |

Howard evaluates the policy by solving the linear system directly. I choose $H=50$ for modified Howard and hold the policy fixed during these inner steps; their total is reported separately. For the two gradient runs, define

$$
F(x)=x-\mathcal T(x),\qquad L(x)=\tfrac12\|F(x)\|_2^2,
\qquad h=(I-\beta\mathcal Q_G)^TF(x),
$$

The greedy policy is recomputed at the current values. My fixed step is conservative: at baseline $M=2000$, $(1+\beta\sqrt M)^2$ bounds the local Hessian norm, and $\eta$ is below $2/(1+\beta\sqrt M)^2$. A step of this size may still give slow convergence.

For Adam, initialize the moments at $a_0=b_0=0$. At update $j$,

$$
a_j=0.9a_{j-1}+0.1h_j,\quad b_j=0.999b_{j-1}+0.001h_j^2,
\quad \hat a_j=\frac{a_j}{1-0.9^j},\quad \hat b_j=\frac{b_j}{1-0.999^j},
$$

$$
x_j=x_{j-1}-0.1\frac{\hat a_j}{\sqrt{\hat b_j}+10^{-8}},
$$

The squares and division are elementwise. Record the gradient loss and Bellman criterion at every check. A gradient run that reaches the cap with finite values is reported as unconverged; numerical breakdown is a failure.

Solver time includes optimization, policy evaluation, and convergence checks. Report initial $\mathcal R$ construction separately as setup time, along with total time, outer iterations, updates, and the exit criterion.

### 2.4 Invariant-distribution algorithms

The joint invariant distribution is the column vector satisfying $\mathcal Q^T\pi=\pi$, $\mathbf1^T\pi=1$, $\pi\geq0$. To compare the algorithms on the same problem, use the converged Howard policy in all six baseline runs.

| Method | Numerical definition |
|---|---|
| Power iteration | Start at $\pi_0=\mathbf1/M$. Update $\pi_{j+1}=\mathcal Q^T\pi_j$; return the updated vector when $\|\pi_{j+1}-\pi_j\|_\infty\leq10^{-12}$. Cap: 100,000 updates. |
| Inverse iteration / eigenvector | Use `eigs(Q.T, k=1, sigma=1-1e-10, which="LM", v0=ones(M)/sqrt(M), tol=1e-12, maxiter=100000)`. Require $|\lambda-1|\leq10^{-10}$ and eigenvector imaginary parts at most $10^{-10}$; take the real part and divide by its nonzero sum. |
| Modified equation system | Replace the last row of $I-\mathcal Q^T$ with ones; use a right-hand side of zeros except its last entry, one. Solve with `numpy.linalg.solve` when dense and `spsolve` when sparse. |

Use dense $\mathcal Q$ in part (b) and CSR in part (c). Time equation setup, iteration or solution, and normalization; record common matrix construction and conversion separately. Report maximum pairwise sup-norm differences within and across representations.

Every computed distribution must have normalization and stationarity errors at most $10^{-10}$. Allow only normalized entries in $[-10^{-14},0)$ to be rounded to zero. Record the raw minimum and correction mass, then renormalize and repeat the checks. Materially negative probabilities, nonfinite entries, and unsuccessful solves are failures. Taking absolute values of an eigenvector is not an acceptable correction.

### 2.5 Grids, statistics, and experiments

For the uniform grid, $k_n=k_{max}(n-1)/(N-1)$. I compare it with the following nonuniform grid, which places more points near the left boundary:

$$
k_n=\exp\left[\frac{n-1}{N-1}\log(1+k_{max})\right]-1.
$$

Both grids include the exact endpoints. For parts (a)–(d), use $N=1000$ on $[0,20]$. The experiments proceed in the order below.

| Part | Experiment and choice |
|---|---|
| (a) | Compare the five solver runs: iterations, runtime, exit metric, and status. |
| (b)–(d) | Compare three dense and three sparse distribution runs. Stop after (c) for my household solver and distribution-method/representation choices. |
| (e) | With selected methods, keep $N=1000$ and test uniform ranges with $k_{max}\in\{2,5,10,20,40\}$. Report range diagnostics and runtime; stop for my upper-bound choice. |
| (f) | At the selected bound and $N=1000$, compare uniform and nonuniform grids using values, policies, distributions, runtime, and Euler diagnostics; stop for my grid choice. |
| (g) | Hold selected methods, bound, and grid formula fixed; solve at $N\in\{100,500,1000,2000,5000\}$. Plot values, asset policies, and joint distributions for both shocks at every $N$; stop for my final $N$ choice. |
| (h) | At my selected $N$, report Euler accuracy and final tests; use all five solutions for accuracy scaling. |

After parts (a)–(c), I choose from methods that converged and passed the checks. Use only my chosen methods from (e) onward, and record them in `code/selected_settings.json`.

Let $p_n=\sum_s\pi_{n,s}$ be the marginal asset distribution. Mean assets are $\sum_n k_np_n$, and mass on the top node is $p_N$. Define numerical support by $I=\{n:p_n>10^{-12}\}$. Its upper endpoint is $\max_{n\in I}k_n$, and the share of grid points inside it is $|I|/N$. A trial range with top mass above $10^{-12}$ is unsuitable; its mass must remain in the reported results. Distribution plots show joint probability masses for each shock. On a nonuniform grid, these masses are not densities.

To relate the comparison to the lecture's theory, record the contraction-based prediction $\log(10^{-8})/\log(\beta)\approx451$, the observed counts, modified-Howard inner counts, and gradient traces. This predicts the scale of an eight-order error reduction, rather than the exact count under the percentage-change criterion. Report the local loss-Hessian bound $\kappa\geq(1-\beta)^{-2}=625$. If feasible, compute the extreme singular values of $J=I-\beta\mathcal Q_G$ at zero values and at the Howard solution, keeping $\operatorname{cond}_2(J)$ distinct from $\operatorname{cond}_2(J^TJ)=\operatorname{cond}_2(J)^2$.

### 2.6 Euler residual

For the continuous-choice problem, when the artificial upper grid bound is inactive, the Euler condition is

$$
c^{-\sigma}\geq\beta(1+r^0)\sum_{s'}\mathcal P_{ss'}(c^{next}_{s'})^{-\sigma},
$$

Equality holds when the borrowing constraint is slack. To evaluate the discrete policy, define current and next-period consumption by

$$
c_{n,s}=(1+r^0)k_n+w^0\bar\epsilon_s\bar l-k_{\mathcal G(n,s)},
$$

$$
c^{next}_{n,s,s'}=(1+r^0)k_{\mathcal G(n,s)}+w^0\bar\epsilon_{s'}\bar l
-k_{\mathcal G(\mathcal G(n,s),s')},
$$

$$
c^{EE}_{n,s}=\left[\beta(1+r^0)\sum_{s'}\mathcal P_{ss'}(c^{next}_{n,s,s'})^{-\sigma}\right]^{-1/\sigma},
\qquad E_{n,s}=\left|1-\frac{c^{EE}_{n,s}}{c_{n,s}}\right|.
$$

Compute the residual at every state in $A=\{(n,s):g_{n,s}>0\}$, and flag any choice at the upper grid bound. Report the maximum $\max_A E$, the mean $|A|^{-1}\sum_A E$, and the probability-weighted mean $\sum_A qE$, using conditional weights $q=\pi/\sum_A\pi$. I define the weighted maximum as $\max_A(qE)$ and label it the **maximum weighted contribution**. Also report the maximum on stationary support, $\max\{E_{n,s}:(n,s)\in A,\pi_{n,s}>10^{-12}\}$. If a set is empty or the probability denominator is zero, report the statistic as unavailable and explain why.

At my chosen grid size, plot residuals against assets separately for the two shocks. Using all five node counts, plot the mean and maximum errors against $h=\max_n(k_{n+1}-k_n)$ on log-log axes; for a uniform grid this is $k_{max}/(N-1)$. Estimate separate OLS slopes of $\log(error)$ on $\log(h)$ using positive finite pairs and at least two distinct steps. The observed scaling is a result to interpret, so no particular slope or improvement at every $N$ is required.

### 2.7 Validation inputs

Before full experiments, check the same model on $N=100$ uniform points on $[0,20]$. The manual comparison uses plain VFI from zero, $\tau=10^{-10}$, and the return/counting conventions above. My reference is `manual/kernel_output.npz`, with keys `V`, `G`, `pi`, `iterations`: shapes `(100,2)`, `(100,2)`, `(200,)`, and a scalar outer count. Indices `G` are zero-based; `pi` is column-stacked. Manual power iteration follows §2.4. The agent must not create or overwrite the reference. Report any difference in my existing kernel's interface for my documented revision before comparison.

Also check VFI, Howard, and modified Howard at $\tau=10^{-8}$. Use the Howard solution for the mathematical checks and the six distribution comparisons. The deterministic Bellman inputs are given in `tests.md`. The full baseline gradient runs and the experiment and reproduction checks belong to stage 4.

## 3. Constraints

Use Python scripts with NumPy, SciPy, and Matplotlib, and set random seeds to zero. Follow the assignment agent file. The agent implements `code/` and `tests/`, generates reproducible `results/`, and appends dated entries to `log.md`. I am responsible for the inputs, manual work, choices, and report. The agent reads `manual/` only for the kernel comparison. If a definition is missing or an applicable test fails, stop the affected work and report the issue. Any revision requires a new commit and log entry; parameters, criteria, and Git history must be preserved. Runs longer than about three minutes should be split using checkpoints that retain values, counters, optimizer moments, settings, and elapsed time.

## 4. Output format

The command `python code/run_all.py` must reproduce all experiments, including the initial comparisons, using my recorded choices. Store numerical arrays in `results/{case}.npz` with keys `k_grid`, `V`, `G`, `g`, `c`, `pi`, `E`, and the settings, counts, timings, metrics, statistics, and convergence status in the corresponding JSON file. If a solver run has no distribution, save only the available arrays and list them in JSON. Use case names `validation_{method}`, `a_{method}`, `bc_{dense|sparse}_{method}`, `e_kmax{bound}`, `f_{uniform|nonuniform}`, and `g_N{N}`. Gradient traces are saved as `a_{method}_trace.npz`.

Write the tables `solvers`, `distribution_dense`, `distribution_sparse`, `ranges`, `grids`, and `accuracy` as complete LaTeX `tabular` files, with Markdown versions in `results/tables.md`. Save figures under the stems `g_N{N}_{value|policy|distribution}`, `h_euler`, and `h_scaling`, each as PDF and PNG. Report test results in `results/tests_stage3.json` and `results/tests_final.json`. At stage 4, the README must describe the contents, reproduction command, actual Python/package versions in `pip freeze` format, and runtime of each part.

## 5. Success criteria

The computational work is complete when the applicable tests pass, the comparisons and figures are available, my four choices are recorded, and the results reproduce from a clean copy. The results must also retain unconverged gradient runs and unsuitable trial ranges.

Follow the six-stage Git workflow: I commit the inputs at stage 0 and the manual work at stage 1 before solver code. Each consequential prompt is committed before its run. The agent makes separate stage-2, stage-3, and stage-4 commits, stating measured results and identifying its authorship. Before writing the report and making the stage-5 sign-off, I independently verify the manual-kernel comparison and at least one other result, recording the numbers and dates in `log.md`.
