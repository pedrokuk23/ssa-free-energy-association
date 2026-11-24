"""
E6 — Associação após lacuna longa (1 dia ≈ 15 órbitas), dinâmica não linear (2 corpos + J2).
Prior inicial gaussiano (UCT recém-formado) com incerteza na velocidade along-track σ_vI
→ a densidade prevista vira uma "banana" ao longo da órbita.

Energia livre do tracklet:  F = −log p(z | trilha) = min_Q {KL(Q‖P) − E_Q log L(z|x_T)}.
  UKF     : P aproximada por uma única gaussiana no instante T
  GM-K    : P como mistura de K gaussianas (divisão ao longo de v_I) + UKF por componente
  MC      : partículas (1e6) — referência independente para validação
Medida: posição (fix de radar) com σ_m isotrópico.
"""
import numpy as np, json, sys, time
from scipy.special import logsumexp
from orbit import propagate, ref_state, ric_frame

T = 86400.0
SIG_M = 50.0
SIG0 = dict(R=100.0, I=300.0, C=100.0, vR=0.2, vC=0.2)


def prior_cov(X, s_vI):
    Rm = ric_frame(X)
    Q = np.zeros((6, 6)); Q[:3, :3] = Rm.T; Q[3:, 3:] = Rm.T
    D = np.diag([SIG0["R"], SIG0["I"], SIG0["C"], SIG0["vR"], s_vI, SIG0["vC"]]) ** 2
    return Q @ D @ Q.T, Q[:, 4]          # covariância ECI e direção de v_I


def split(m, P, e_dir, K):
    """Divide N(m,P) em K gaussianas ao longo da direção que maximiza var(e·x)."""
    if K == 1:
        return np.ones(1), m[None], P[None]
    L = np.linalg.cholesky(P)
    u = L.T @ e_dir; u /= np.linalg.norm(u)
    Lb = 4.0 if K > 9 else 2.5
    mu = np.linspace(-Lb, Lb, K); h = mu[1] - mu[0]
    st = min(0.85, 0.75 * h)
    w = np.exp(-0.5 * mu**2 / (1 - st**2)); w /= w.sum()
    Ck = np.eye(6) - (1 - st**2) * np.outer(u, u)
    M = m[None] + (mu[:, None] * u[None]) @ L.T
    Pk = L @ Ck @ L.T
    return w, M, np.repeat(Pk[None], K, 0)


def ut_points(M, P):
    n = 6; K = len(M)
    S = np.linalg.cholesky(n * P)                           # κ = 0
    pts = np.concatenate([M[:, None], M[:, None] + np.transpose(S, (0, 2, 1)),
                          M[:, None] - np.transpose(S, (0, 2, 1))], 1)  # (K, 13, 6)
    W = np.r_[0.0, np.full(2 * n, 1 / (2 * n))]
    return pts, W


def gm_ukf_logev(z, m0, P0, e_dir, K):
    w, M, P = split(m0, P0, e_dir, K)
    pts, W = ut_points(M, P)
    Y = propagate(pts.reshape(-1, 6), T).reshape(K, 13, 6)[..., :3]
    my = np.einsum("j,kjd->kd", W, Y)
    D = Y - my[:, None]
    Py = np.einsum("j,kjd,kje->kde", W, D, D) + SIG_M**2 * np.eye(3)
    r = z[None] - my
    sol = np.linalg.solve(Py, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(Py)
    lp = -0.5 * (np.sum(r * sol, -1) + ld + 3 * np.log(2 * np.pi))
    return logsumexp(lp + np.log(w)), my, Py


def mc_logev(z, m0, P0, N=1_000_000, chunk=250_000, rng=None):
    L = np.linalg.cholesky(P0); acc = []
    for _ in range(N // chunk):
        X = m0 + rng.standard_normal((chunk, 6)) @ L.T
        Y = propagate(X, T)[:, :3]
        d2 = np.sum((Y - z) ** 2, 1) / SIG_M**2
        acc.append(-0.5 * d2 - 1.5 * np.log(2 * np.pi * SIG_M**2))
    a = np.concatenate(acc)
    return logsumexp(a) - np.log(len(a)), np.exp(2 * logsumexp(a) - logsumexp(2 * a))


def make_case(rng, s_vI):
    Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
    P0, e_dir = prior_cov(Xref, s_vI)
    L = np.linalg.cholesky(P0)
    x_true = Xref + L @ rng.standard_normal(6)          # verdade ~ prior centrado em Xref
    yT = propagate(x_true, T)
    return Xref, P0, e_dir, yT
