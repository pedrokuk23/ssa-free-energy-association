"""
E7 — Lacuna longa não linear (2 corpos + J2, 1 dia) COM manobra impulsiva em parte dos casos.

Estatísticas (menor = "mesmo objeto"):
  UKF        : −log p(z) com gaussiana única, sem hipótese de manobra
  UKF+M      : idem, referência em mistura (natural + manobra)              [ablação]
  GM         : −log p(z) com mistura gaussiana K=41, sem manobra          [estilo Montilla et al. 2025]
  CTRL       : Δv mínimo de impulso único que leva alguma componente ao fix [estilo Holzinger/Serra]
  GM+M       : energia livre com referência P = (1−p)P0 + p·P1(t_j, Δv~N(0,σ_m² I))   [proposto]
"""
import numpy as np, json, sys, time
from scipy.special import logsumexp
from orbit import propagate
from e6_nonlinear import make_case, split, ut_points, T, SIG_M

K = 41
TJ = np.linspace(0.05, 0.95, 12) * T     # grade de tempos de manobra
SIGMA_MAN = 0.3                           # σ por eixo do Δv sob H1 [m/s]
P_MAN = 0.2
HFD = 0.01                                # passo de diferenças finitas [m/s]


def gm_predict(m0, P0, e, Kc):
    w, M, P = split(m0, P0, e, Kc)
    pts, W = ut_points(M, P)
    Y = propagate(pts.reshape(-1, 6), T).reshape(Kc, 13, 6)[..., :3]
    my = np.einsum("j,kjd->kd", W, Y)
    D = Y - my[:, None]
    Pp = np.einsum("j,kjd,kje->kde", W, D, D)
    return w, M, my, Pp


def impulse_maps(M):
    """Φ_rv(T, t_j) ao longo da média de cada componente: (Kc, J, 3, 3)."""
    Kc = len(M)
    Xj = propagate(np.repeat(M[:, None], len(TJ), 1), TJ[None, :])          # (Kc,J,6)
    dX = np.zeros((7, 6)); dX[1:4, 3:] = HFD * np.eye(3); dX[4:7, 3:] = -HFD * np.eye(3)
    Xp = Xj[:, :, None] + dX[None, None]                                     # (Kc,J,7,6)
    Y = propagate(Xp, (T - TJ)[None, :, None])[..., :3]
    return np.transpose((Y[:, :, 1:4] - Y[:, :, 4:7]) / (2 * HFD), (0, 1, 3, 2))


def loggauss(r, S):
    sol = np.linalg.solve(S, r[..., None])[..., 0]
    _, ld = np.linalg.slogdet(S)
    return -0.5 * (np.sum(r * sol, -1) + ld + 3 * np.log(2 * np.pi))


def all_scores(z, m0, P0, e):
    R = SIG_M**2 * np.eye(3)
    out = {}
    for name, Kc in [("UKF", 1), ("GM", K)]:
        w, M, my, Pp = gm_predict(m0, P0, e, Kc)
        r = z[None] - my
        l0 = logsumexp(loggauss(r, Pp + R) + np.log(w))
        Phi = impulse_maps(M)                                                # (Kc,J,3,3)
        S1 = Pp[:, None] + R + SIGMA_MAN**2 * np.einsum("kjab,kjcb->kjac", Phi, Phi)
        l1 = logsumexp(loggauss(r[:, None], S1) + np.log(w)[:, None]) - np.log(len(TJ))
        out[name] = -l0
        out[name + "+M"] = -np.logaddexp(np.log(1 - P_MAN) + l0, np.log(P_MAN) + l1)
        if name == "GM":
            dv = np.linalg.solve(Phi, np.broadcast_to(r[:, None, :, None], Phi.shape[:2] + (3, 1)))[..., 0]
            out["CTRL"] = np.linalg.norm(dv, axis=-1).min()
    return out, my, Pp


NAMES = ["UKF", "UKF+M", "GM", "CTRL", "GM+M"]


def truth_with_maneuver(rng, s_vI, dv):
    m0, P0, e, yT = make_case(rng, s_vI)
    if dv > 0:
        # refaz a verdade: mesma condição inicial amostrada, com impulso em t_m
        L = np.linalg.cholesky(P0)
        x0 = m0 + L @ rng.standard_normal(6)
        tm = rng.uniform(0.1, 0.9) * T
        xm = propagate(x0, tm)
        u = rng.standard_normal(3); u /= np.linalg.norm(u)
        xm = xm.copy(); xm[3:] += dv * u
        yT = propagate(xm, T - tm)
    return m0, P0, e, yT


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tpr_at(pos, neg, fpr=0.05):
    return float(np.mean(np.asarray(pos) < np.quantile(neg, fpr)))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    s_vI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    rng = np.random.default_rng(21)
    t = time.time()
    negA, negR = [], []
    for _ in range(N):                       # vizinhos (sem manobra)
        m0, P0, e, yT = make_case(rng, s_vI)
        _, my, Pp = all_scores(yT[:3], m0, P0, e)
        Pu = Pp[0] if len(Pp) == 1 else None
        w, M, myu, Ppu = gm_predict(m0, P0, e, 1)
        Lp = np.linalg.cholesky(Ppu[0]); d = rng.standard_normal(3); d /= np.linalg.norm(d)
        za = yT[:3] + rng.uniform(1.5, 3.0) * (Lp @ d) + SIG_M * rng.standard_normal(3)
        negA.append([all_scores(za, m0, P0, e)[0][k] for k in NAMES])
        uR = yT[:3] / np.linalg.norm(yT[:3]); sR = np.sqrt(uR @ Ppu[0] @ uR)
        zr = yT[:3] + rng.choice([-1, 1]) * rng.uniform(1.5, 3.0) * sR * uR + SIG_M * rng.standard_normal(3)
        negR.append([all_scores(zr, m0, P0, e)[0][k] for k in NAMES])
    negA, negR = np.array(negA), np.array(negR)
    print(f"negativos prontos ({time.time()-t:.0f}s)", flush=True)
    res = {}
    for dv in [0.0, 0.1, 0.3, 1.0]:
        pos = []
        for _ in range(N):
            m0, P0, e, yT = truth_with_maneuver(rng, s_vI, dv)
            z = yT[:3] + SIG_M * rng.standard_normal(3)
            pos.append([all_scores(z, m0, P0, e)[0][k] for k in NAMES])
        pos = np.array(pos)
        res[dv] = dict(auc_agn={k: round(float(auc(pos[:, i], negA[:, i])), 3) for i, k in enumerate(NAMES)},
                       auc_shell={k: round(float(auc(pos[:, i], negR[:, i])), 3) for i, k in enumerate(NAMES)},
                       tpr5_agn={k: round(tpr_at(pos[:, i], negA[:, i]), 3) for i, k in enumerate(NAMES)})
        np.savez(f"e7_s{s_vI}_dv{dv}.npz", pos=pos, negA=negA, negR=negR)
        print(dv, json.dumps(res[dv]), f"({time.time()-t:.0f}s)", flush=True)
    json.dump(res, open(f"e7_summary_s{s_vI}.json", "w"), indent=1)
