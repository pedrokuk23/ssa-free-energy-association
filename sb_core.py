"""
Núcleo: dinâmica Clohessy-Wiltshire planar, ponte de Schrödinger gaussiana
e decomposição do suavizador de Kalman (MAP / Onsager-Machlup linear).

Estado x = [x_radial, y_along, vx, vy]  (m, m/s)
dx = A x dt + B (u dt + sqrt(q) dW)
"""
import numpy as np
from scipy.linalg import expm, sqrtm

MU = 3.986004418e14


def cw_matrices(alt_km=500.0):
    a = 6378.137e3 + alt_km * 1e3
    n = np.sqrt(MU / a**3)
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [3 * n**2, 0, 0, 2 * n],
                  [0, 0, -2 * n, 0]], float)
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
    return A, B, n


def phi_gramian(A, B, T):
    """Phi(T) e gramiano W = ∫0^T Phi(T,s) B B^T Phi(T,s)^T ds (Van Loan)."""
    nx = A.shape[0]
    M = np.zeros((2 * nx, 2 * nx))
    M[:nx, :nx] = -A
    M[:nx, nx:] = B @ B.T
    M[nx:, nx:] = A.T
    E = expm(M * T)
    Phi = E[nx:, nx:].T
    W = Phi @ E[:nx, nx:]
    W = 0.5 * (W + W.T)
    return Phi, W


def influence_integral(A, B, T, ngrid=400):
    """∫0^T B^T Phi(T,t)^T dt  (2 x nx) — para reconstruir Δv integrado."""
    ts = np.linspace(0, T, ngrid)
    vals = np.array([B.T @ expm(A * (T - t)).T for t in ts])
    return np.trapezoid(vals, ts, axis=0), ts, vals


def _symsqrt(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 1e-300, None)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def sb_kl(m0, S0, m1, S1, Phi, G):
    """KL fechada da ponte de Schrödinger gaussiana entre N(m0,S0) e N(m1,S1)
    sob a referência x1 = Phi x0 + w, w ~ N(0,G).
    Retorna (KL total, termo de média, termo de covariância)."""
    n = len(m0)
    _, Gi12 = _symsqrt(G)
    Mw = Gi12 @ Phi
    Aw = Mw @ S0 @ Mw.T
    Bw = Gi12 @ S1 @ Gi12
    dlt = Gi12 @ (m1 - Phi @ m0)
    Ah, Aih = _symsqrt(Aw)
    D = Ah @ Bw @ Ah
    Dh, _ = _symsqrt(D + 0.25 * np.eye(n))
    Y = Dh - 0.5 * np.eye(n)
    K = Ah @ Y @ Aih
    S = Aih @ Y @ Aih
    sign, logdetS = np.linalg.slogdet(0.5 * (S + S.T))
    mean_term = 0.5 * dlt @ dlt
    cov_term = 0.5 * np.trace(Aw + Bw - 2 * K) - 0.5 * logdetS - 0.5 * n
    return mean_term + cov_term, mean_term, cov_term


def sb_kl_numeric(m0, S0, m1, S1, Phi, G, seed=0):
    """Verificação: minimiza a KL sobre a covariância cruzada K diretamente."""
    from scipy.optimize import minimize
    n = len(m0)
    _, Gi12 = _symsqrt(G)
    Mw = Gi12 @ Phi
    Aw = Mw @ S0 @ Mw.T
    Bw = Gi12 @ S1 @ Gi12
    dlt = Gi12 @ (m1 - Phi @ m0)
    Ai = np.linalg.inv(Aw)

    def f(k):
        K = k.reshape(n, n)
        S = 0.5 * ((Bw - K.T @ Ai @ K) + (Bw - K.T @ Ai @ K).T)
        ev = np.linalg.eigvalsh(S)
        if ev.min() <= 0:
            return 1e30
        ld = np.log(ev).sum()
        return 0.5 * dlt @ dlt + 0.5 * np.trace(Aw + Bw - 2 * K) - 0.5 * ld - 0.5 * n

    best = None
    rng = np.random.default_rng(seed)
    for _ in range(5):
        k0 = 1e-3 * rng.standard_normal(n * n)
        r = minimize(f, k0, method="Nelder-Mead",
                     options=dict(maxiter=200000, maxfev=200000, xatol=1e-12, fatol=1e-14))
        r = minimize(f, r.x, method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    return best.fun


def smoother_decomp(m0, S0, m1, S1, Phi, G):
    """MAP conjunto (x0, u(.), x1) do suavizador linear.
    Retorna custo total (=1/2 Mahalanobis da inovação) e partes:
    inicial, controle/processo, terminal; e lambda."""
    d = m1 - Phi @ m0
    PS = Phi @ S0 @ Phi.T
    Sig = PS + G + S1
    lam = np.linalg.solve(Sig, d)
    c0 = 0.5 * lam @ PS @ lam
    cu = 0.5 * lam @ G @ lam
    c1 = 0.5 * lam @ S1 @ lam
    return dict(total=0.5 * d @ lam, c0=c0, cu=cu, c1=c1, lam=lam, d=d, Sig=Sig)


def logN0(d, Sig):
    s, ld = np.linalg.slogdet(Sig)
    return -0.5 * (d @ np.linalg.solve(Sig, d) + ld + len(d) * np.log(2 * np.pi))
