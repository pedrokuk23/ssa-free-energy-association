"""
Monte Carlo: associação track-to-track e detecção de manobra em CW planar (LEO 500 km).
Compara: Mahalanobis da inovação, distância de controle L2 (Holzinger, entre médias),
KL da ponte de Schrödinger gaussiana, e GLR com decomposição do suavizador.
"""
import numpy as np, json
from scipy.linalg import expm
from sb_core import *

rng = np.random.default_rng(42)
A, B, n = cw_matrices(500)
T = 4000.0
q0 = 1e-10            # PSD da aceleração não modelada natural [m^2/s^3]
dv_typ = 0.1          # Δv típico da hipótese de manobra [m/s]
qm = dv_typ**2 / T    # PSD equivalente da hipótese H1
Phi, W = phi_gramian(A, B, T)
G0, G1 = q0 * W, (q0 + qm) * W
Iint, ts, vals = influence_integral(A, B, T)
Wi = np.linalg.inv(W)

sp, sv = 30.0, 0.03                     # erro de rastreio (1σ) pos/vel
OFF = np.diag([150, 150, 0.15, 0.15])   # dispersão relativa entre objetos vizinhos
N = 3000

def rand_cov():
    L = np.eye(4) + 0.3 * rng.standard_normal((4, 4))
    D = np.diag([sp, sp, sv, sv]) * rng.uniform(0.6, 1.4)
    S = D @ (L @ L.T) @ D / np.trace(L @ L.T) * 4
    return 0.5 * (S + S.T)

def sample(S): return np.linalg.cholesky(S) @ rng.standard_normal(4)

def propagate(x0, dv=0.0):
    x1 = Phi @ x0 + sample(G0)
    dvec = None
    if dv > 0:
        tm = rng.uniform(0.2 * T, 0.8 * T)
        ang = rng.uniform(0, 2 * np.pi)
        dvec = dv * np.array([np.cos(ang), np.sin(ang)])
        x1 = x1 + expm(A * (T - tm)) @ B @ dvec
    return x1, dvec

def stats(m0, S0, m1, S1):
    d = m1 - Phi @ m0
    r0 = smoother_decomp(m0, S0, m1, S1, Phi, G0)
    r1 = smoother_decomp(m0, S0, m1, S1, Phi, G1)
    kl, klm, klc = sb_kl(m0, S0, m1, S1, Phi, G0)
    u_int = (q0 + qm) * (Iint @ r1["lam"])
    u_t = (q0 + qm) * np.einsum("tij,j->ti", vals, r1["lam"])
    l1 = np.trapezoid(np.linalg.norm(u_t, axis=1), ts)
    return dict(maha=2 * r0["total"], ctrl=d @ Wi @ d, sb=kl, sb_mean=klm, sb_cov=klc,
                glr=logN0(d, r1["Sig"]) - logN0(d, r0["Sig"]),
                share=r1["cu"] / r1["total"], dv_int=np.linalg.norm(u_int), dv_l1=l1)

def trial(same=True, dv=0.0):
    x0 = rng.standard_normal(4) * [500, 500, .5, .5]
    S0, S1 = rand_cov(), rand_cov()
    m0 = x0 + sample(S0)
    xa = x0 if same else x0 + sample(OFF @ OFF)
    x1, dvec = propagate(xa, dv)
    m1 = x1 + sample(S1)
    s = stats(m0, S0, m1, S1)
    s["dv_true"] = dv
    return s

def auc(pos, neg):
    """AUC para estatística onde MENOR = classe 'positiva'."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    rp = r[:len(pos)].sum()
    return 1 - (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def roc(pos, neg, lower_is_pos=True):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s if lower_is_pos else -s); y = y[o]
    return np.r_[0, np.cumsum(1 - y) / len(neg)], np.r_[0, np.cumsum(y) / len(pos)]

keys = ["maha", "ctrl", "sb", "glr"]
res = {}

# E1: associação sem manobra — mesmo objeto vs vizinho
E1p = [trial(True) for _ in range(N)]; E1n = [trial(False) for _ in range(N)]
# E2: associação com o objeto verdadeiro manobrando (Δv = 0.1 m/s)
E2p = [trial(True, 0.1) for _ in range(N)]; E2n = E1n
res["E1"] = {k: auc([s[k] for s in E1p], [s[k] for s in E1n]) for k in ["maha", "ctrl", "sb"]}
res["E2"] = {k: auc([s[k] for s in E2p], [s[k] for s in E2n]) for k in ["maha", "ctrl", "sb"]}

# E3: detecção de manobra — mesmo objeto, Δv vs sem manobra (maior = manobra)
dvs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
res["E3"] = {}
E3 = {}
for dv in dvs:
    P = [trial(True, dv) for _ in range(1500)]
    E3[dv] = P
    res["E3"][dv] = {k: 1 - auc([s[k] for s in P], [s[k] for s in E1p[:1500]])
                     for k in ["maha", "ctrl", "sb", "glr", "share"]}

# E4: reconstrução de Δv
rec = [(s["dv_true"], s["dv_int"], s["dv_l1"]) for dv in dvs for s in E3[dv]]
res["E4"] = {dv: dict(med_int=float(np.median([s["dv_int"] for s in E3[dv]])),
                      med_l1=float(np.median([s["dv_l1"] for s in E3[dv]])))
             for dv in dvs}

# decomposição da KL da ponte: quanto vem da média vs covariância
res["sb_split"] = dict(
    same=float(np.median([s["sb_cov"] / s["sb"] for s in E1p])),
    other=float(np.median([s["sb_cov"] / s["sb"] for s in E1n])))

print(json.dumps(res, indent=1, default=float))
np.savez("mc_results.npz",
         E1p=np.array([[s[k] for k in keys] for s in E1p]),
         E1n=np.array([[s[k] for k in keys] for s in E1n]),
         E2p=np.array([[s[k] for k in keys] for s in E2p]),
         rec=np.array(rec),
         share0=np.array([s["share"] for s in E1p]),
         share_m=np.array([s["share"] for s in E3[0.05]]))
json.dump(res, open("mc_summary.json", "w"), indent=1, default=float)
