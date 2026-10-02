# Worked reference: PS1 M.2; NumPy only, column-stacked states.
import numpy as np

beta, sigma, labor = 0.96, 1.5, 1.0
alpha, delta, z, K0 = 0.36, 0.1, 1.0, 5.0
N, epsilon = 100, np.array([0.8, 1.2])
P = np.full((2, 2), 0.5)
L = np.array([0.5, 0.5]) @ epsilon * labor
w = (1 - alpha) * z * (K0 / L)**alpha
r = alpha * z * (K0 / L)**(alpha - 1) - delta
k = np.linspace(0.0, 20.0, N)
resources = ((1 + r) * k[:, None] + w * epsilon * labor).ravel(order="F")
c = resources[:, None] - k[None, :]
R = np.full(c.shape, -np.inf)
R[c > 0] = c[c > 0]**(1 - sigma) / (1 - sigma)
V = np.zeros((N, 2))
for iterations in range(1, 10001):
    B = R + beta * np.kron(P @ V.T, np.ones((N, 1)))
    V_hat = B.max(axis=1).reshape((N, 2), order="F")
    G = B.argmax(axis=1).reshape((N, 2), order="F")
    metric = np.max(np.abs(V_hat - V) / (np.abs(V) + 1))
    if metric <= 1e-10:
        break
    if iterations == 10000:
        raise RuntimeError("VFI did not converge")
    V = V_hat
Q = np.zeros((2 * N, 2 * N))
rows, policy = np.arange(2 * N), G.ravel(order="F")
for s_next in range(2):
    Q[rows, policy + s_next * N] = P[rows // N, s_next]
pi = np.full(2 * N, 1.0 / (2 * N))
for dist_iterations in range(1, 100001):
    pi_new = Q.T @ pi
    dist_metric = np.max(np.abs(pi_new - pi))
    pi = pi_new
    if dist_metric <= 1e-12:
        break
else:
    raise RuntimeError("Power iteration did not converge")
mean_assets = np.tile(k, 2) @ pi
folder = __file__.replace("\\", "/").rpartition("/")[0]
output = (folder + "/" if folder else "") + "kernel_output.npz"
np.savez(output, V=V, G=G, pi=pi, iterations=iterations, k_grid=k,
         metric=metric, dist_iterations=dist_iterations, r0=r, w0=w)
print(f"VFI iterations (outer checks): {iterations}")
print(f"Vmin = {V.min():.12f}; Vmax = {V.max():.12f}")
print(f"Mean assets = {mean_assets:.12f}")
print(f"Bellman metric = {metric:.3e}; distribution updates = {dist_iterations}")
print(f"Saved output: {output}")
