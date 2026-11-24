"""
E5 — Associação com tracklet angles-only curto (marginal terminal não-gaussiana).

Sensor co-orbital na origem do referencial CW mede, no instante T,
  θ = atan2(y, x)  e  θ̇ = (x vy − y vx)/ρ²   (ρ e ρ̇ não observados).
Pergunta: o tracklet pertence ao objeto de catálogo (prior gaussiano em t0)?

Métodos comparados (menor = "mesmo objeto"):
  maha_AR  : Mahalanobis com a admissible region (AR) gaussianizada por momentos
  hyb_AR   : evidência gaussiana H0∪H1 com AR gaussianizada
  sb_AR    : KL da ponte de Schrödinger gaussiana entre prior e AR gaussianizada
  ctrl_AR  : distância de controle mínima sobre a AR (estilo Holzinger)
  F0       : energia livre de caminhos  −log E_P0[L(z|x1)]       (sem manobra)
  Fmix     : energia livre com referência P = (1−p)P0 + p P1     (com manobra)
F = min_Q { KL(Q‖P) − E_Q[log L] } = −log evidência (Donsker–Varadhan / Gibbs),
calculada por amostragem de caminhos (Monte Carlo, vetorizado).
"""
import numpy as np, json, sys
from scipy.linalg import expm
from scipy.special import logsumexp
from sb_core import cw_matrices, phi_gramian, sb_kl, logN0

rng = np.random.default_rng(7)
A, B, n = cw_matrices(500)
T = 4000.0
q0 = 1e-10
Phi, W = phi_gramian(A, B, T)
G0 = q0 * W
qm = 0.1**2 / T
G1 = (q0 + qm) * W
Wi = np.linalg.inv(W)
L0 = np.linalg.cholesky(G0 + 1e-18 * np.eye(4))

# matrizes de impulso para tempos de manobra em grade
TM = np.linspace(0.05 * T, 0.95 * T, 64)
IMP = np.array([expm(A * (T - t)) @ B for t in TM])   # (64,4,2)

sth, sthd = 2e-4, 2e-6             # ruído do tracklet (rad, rad/s)
RHO = (300.0, 6000.0)              # AR: limites de alcance [m]
RHOD = 3.0                         # AR: |ρ̇| ≤ 3 m/s
sp, sv = 30.0, 0.03
OFF = np.diag([150, 150, 0.15, 0.15])
NP = 20000                          # partículas por avaliação
P1 = 0.2                            # prob. a priori de manobra
DV_SCALE = 0.15                     # prior de |Δv| ~ semi-normal (m/s)


def rand_cov():
    Lr = np.eye(4) + 0.3 * rng.standard_normal((4, 4))
    D = np.diag([sp, sp, sv, sv]) * rng.uniform(0.6, 1.4)
    S = D @ (Lr @ Lr.T) @ D / np.trace(Lr @ Lr.T) * 4
    return 0.5 * (S + S.T)


def sample(S, k=None):
    Lc = np.linalg.cholesky(S)
    if k is None:
        return Lc @ rng.standard_normal(4)
    return rng.standard_normal((k, 4)) @ Lc.T


def h(X):
    X = np.atleast_2d(X)
    x, y, vx, vy = X.T
    r2 = x * x + y * y
    return np.c_[np.arctan2(y, x), (x * vy - y * vx) / r2]


def loglik(X, z):
    H = h(X)
    dth = np.angle(np.exp(1j * (H[:, 0] - z[0])))
    return -0.5 * ((dth / sth) ** 2 + ((H[:, 1] - z[1]) / sthd) ** 2)


def maneuver_kicks(k):
    """Δv aleatório (impulsivo) com tempo uniforme e |Δv| semi-normal."""
    idx = rng.integers(0, len(TM), k)
    mag = np.abs(rng.standard_normal(k)) * DV_SCALE
    ang = rng.uniform(0, 2 * np.pi, k)
    dv = np.c_[mag * np.cos(ang), mag * np.sin(ang)]
    return np.einsum("kij,kj->ki", IMP[idx], dv)


def free_energies(m0, S0, z):
    X0 = m0 + sample(S0, NP)
    X1 = X0 @ Phi.T + rng.standard_normal((NP, 4)) @ L0.T
    l0 = loglik(X1, z)
    X1m = X1 + maneuver_kicks(NP)
    l1 = loglik(X1m, z)
    lE0 = logsumexp(l0) - np.log(NP)
    lE1 = logsumexp(l1) - np.log(NP)
    lmix = np.logaddexp(np.log(1 - P1) + lE0, np.log(P1) + lE1)
    # reconstrução sob H1: média ponderada do Δv equivalente (pós-manobra)
    return -lE0, -lmix


def ar_samples(z, k=4000):
    rho = rng.uniform(*RHO, k)
    rhod = rng.uniform(-RHOD, RHOD, k)
    th = z[0] + sth * rng.standard_normal(k)
    thd = z[1] + sthd * rng.standard_normal(k)
    c, s = np.cos(th), np.sin(th)
    return np.c_[rho * c, rho * s, rhod * c - rho * thd * s, rhod * s + rho * thd * c]


def gaussian_ar(z):
    Y = ar_samples(z)
    return Y.mean(0), np.cov(Y.T) + 1e-9 * np.eye(4), Y


def classical(m0, S0, z):
    m1, S1, Y = gaussian_ar(z)
    d = m1 - Phi @ m0
    Sg0 = Phi @ S0 @ Phi.T + G0 + S1
    Sg1 = Phi @ S0 @ Phi.T + G1 + S1
    maha = d @ np.linalg.solve(Sg0, d)
    l0, l1 = logN0(d, Sg0), logN0(d, Sg1)
    hyb = -np.logaddexp(np.log(1 - P1) + l0, np.log(P1) + l1)
    sb = sb_kl(m0, S0, m1, S1, Phi, G0)[0]
    D = Y - Phi @ m0
    ctrl = np.min(np.einsum("ki,ij,kj->k", D, Wi, D))
    return maha, hyb, sb, ctrl


def make_case(same, dv):
    # alvo a 1–4 km do sensor, com velocidade relativa compatível com CW
    rho = rng.uniform(1000, 4000); ang = rng.uniform(0, 2 * np.pi)
    x0 = np.r_[rho * np.cos(ang), rho * np.sin(ang), 0.3 * rng.standard_normal(2)]
    S0 = rand_cov()
    m0 = x0 + sample(S0)
    xa = x0 if same else x0 + sample(OFF @ OFF)
    x1 = Phi @ xa + sample(G0)
    if dv > 0:
        t = rng.uniform(0.2 * T, 0.8 * T); a = rng.uniform(0, 2 * np.pi)
        x1 = x1 + expm(A * (T - t)) @ B @ (dv * np.r_[np.cos(a), np.sin(a)])
    z = h(x1)[0] + np.r_[sth, sthd] * rng.standard_normal(2)
    return m0, S0, z


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


NAMES = ["maha_AR", "hyb_AR", "sb_AR", "ctrl_AR", "F0", "Fmix"]


def run(same, dv, N):
    out = []
    for _ in range(N):
        m0, S0, z = make_case(same, dv)
        out.append([*classical(m0, S0, z), *free_energies(m0, S0, z)])
    return np.array(out)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    neg = run(False, 0.0, N)
    res = {}
    allv = {"neg": neg}
    for dv in [0.0, 0.05, 0.1, 0.2]:
        pos = run(True, dv, N)
        allv[f"pos_{dv}"] = pos
        res[dv] = {k: round(float(auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(NAMES)}
        print(dv, res[dv], flush=True)
    json.dump(res, open("e5_summary.json", "w"), indent=1)
    np.savez("e5_results.npz", **allv)


# ---------------- versão Rao-Blackwellizada + linha de base EKF ----------------
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm as _norm
GHx, GHw = hermegauss(3); GHw = GHw / GHw.sum()
MM = np.array([M @ M.T for M in IMP])          # (64,4,4)
SDV = DV_SCALE                                  # σ por eixo do Δv sob H1


def _log_g(theta, thd, m, S):
    """log ∫ N(J r; m, S) ρ² dr, com x = J [ρ, ρ̇] (linear dados θ, θ̇)."""
    c, s = np.cos(theta), np.sin(theta)
    J = np.array([[c, 0], [s, 0], [-thd * s, c], [thd * c, s]])
    Si = np.linalg.inv(S)
    P = J.T @ Si @ J; b = J.T @ Si @ m
    Pi = np.linalg.inv(P); r = Pi @ b
    cq = m @ Si @ m - b @ r
    _, ldS = np.linalg.slogdet(S); _, ldP = np.linalg.slogdet(P)
    sr = np.sqrt(Pi[0, 0])
    trunc = _norm.cdf(r[0] / sr)
    return (-0.5 * cq - 0.5 * ldS - 0.5 * ldP - np.log(2 * np.pi)
            + np.log(r[0] ** 2 + Pi[0, 0]) + np.log(trunc + 1e-300))


def log_evidence(z, m, S):
    vals = [np.log(wi * wj) + _log_g(z[0] + sth * xi, z[1] + sthd * xj, m, S)
            for xi, wi in zip(GHx, GHw) for xj, wj in zip(GHx, GHw)]
    return logsumexp(vals)


def rb_free_energies(m0, S0, z):
    mp = Phi @ m0; Pp = Phi @ S0 @ Phi.T + G0
    lE0 = log_evidence(z, mp, Pp)
    lE1 = logsumexp([log_evidence(z, mp, Pp + SDV**2 * Mk) for Mk in MM]) - np.log(len(MM))
    return -lE0, -np.logaddexp(np.log(1 - P1) + lE0, np.log(P1) + lE1)


def hjac(x):
    x_, y, vx, vy = x; r2 = x_**2 + y**2
    th = [-y / r2, x_ / r2, 0, 0]
    num = x_ * vy - y * vx
    thd = [vy / r2 - 2 * x_ * num / r2**2, -vx / r2 - 2 * y * num / r2**2, -y / r2, x_ / r2]
    return np.array([th, thd])


def ekf_scores(m0, S0, z):
    mp = Phi @ m0; Hj = hjac(mp); R = np.diag([sth**2, sthd**2])
    nu = z - h(mp)[0]; nu[0] = np.angle(np.exp(1j * nu[0]))
    Sz0 = Hj @ (Phi @ S0 @ Phi.T + G0) @ Hj.T + R
    Sz1 = Hj @ (Phi @ S0 @ Phi.T + G1) @ Hj.T + R
    maha = nu @ np.linalg.solve(Sz0, nu)
    hyb = -np.logaddexp(np.log(1 - P1) + logN0(nu, Sz0), np.log(P1) + logN0(nu, Sz1))
    return maha, hyb


NAMES2 = ["SB_ARgauss", "EKF_maha", "EKF_hib", "F0", "Fmix"]


def run2(same, dv, N, check_mc=False):
    out, chk = [], []
    for _ in range(N):
        m0, S0, z = make_case(same, dv)
        _, _, sb, _ = classical(m0, S0, z)
        f0, fm = rb_free_energies(m0, S0, z)
        out.append([sb, *ekf_scores(m0, S0, z), f0, fm])
        if check_mc:
            chk.append((f0, free_energies(m0, S0, z)[0]))
    return np.array(out), np.array(chk)


GH7x, GH7w = hermegauss(7); GH7w = GH7w / GH7w.sum()


def log_evidence(z, m, S):          # substitui a versão de 3 pontos
    """p(z) = ∫ N(z; h(x), R) N(x; m, S) dx, com (ρ, ρ̇) integrados em forma fechada e
    (θ, θ̇) por Gauss-Hermite 7×7 sob proposta = posterior linearizada em θ, θ̇."""
    Hj = hjac(m); R = np.diag([sth**2, sthd**2])
    Pz = Hj @ S @ Hj.T + 1e-30 * np.eye(2)
    hz = h(m)[0]; dz = z - hz; dz[0] = np.angle(np.exp(1j * dz[0]))
    Ci = np.linalg.inv(R) + np.linalg.inv(Pz)
    C = np.linalg.inv(Ci); C = 0.5 * (C + C.T) * 1.5**2   # proposta alargada
    c = hz + C / 1.5**2 @ np.linalg.solve(Pz, np.zeros(2)) + (C / 1.5**2) @ np.linalg.solve(R, dz)
    Lc = np.linalg.cholesky(C)
    _, ldC = np.linalg.slogdet(C)
    vals = []
    for xi, wi in zip(GH7x, GH7w):
        for xj, wj in zip(GH7x, GH7w):
            u = np.r_[xi, xj]; p = c + Lc @ u
            e = p - hz - dz            # p − z
            logNR = -0.5 * (e[0]**2 / sth**2 + e[1]**2 / sthd**2) - np.log(2 * np.pi * sth * sthd)
            logq = -0.5 * u @ u - np.log(2 * np.pi) - 0.5 * ldC
            vals.append(np.log(wi * wj) + logNR + _log_g(p[0], p[1], m, S) - logq)
    return logsumexp(vals)
