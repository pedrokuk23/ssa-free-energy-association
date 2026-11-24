import numpy as np, json, sys, time
from e6_nonlinear import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SIGMAS = [0.05, 0.2, 0.5, 1.0, 2.0]
rng = np.random.default_rng(11)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tpr_at(pos, neg, fpr=0.05):
    thr = np.quantile(neg, fpr)          # aceita se estatística < thr
    return float(np.mean(pos < thr))


def scores(z, m0, P0, e):
    l1, my, Py = gm_ukf_logev(z, m0, P0, e, 1)
    r = z - my[0]
    maha = r @ np.linalg.solve(Py[0], r)
    l15 = gm_ukf_logev(z, m0, P0, e, 15)[0]
    l41 = gm_ukf_logev(z, m0, P0, e, 41)[0]
    return [maha, -l1, -l15, -l41], my[0], Py[0] - SIG_M**2 * np.eye(3)


NAMES = ["UKF_maha", "UKF_F", "GM15_F", "GM41_F"]
res = {}
for s in SIGMAS:
    t = time.time(); pos, neg, neg2, calib = [], [], [], []
    for _ in range(N):
        m0, P0, e, yT = make_case(rng, s)
        z = yT[:3] + SIG_M * rng.standard_normal(3)
        sc, my, Pp = scores(z, m0, P0, e)
        pos.append(sc)
        l121 = gm_ukf_logev(z, m0, P0, e, 121)[0]
        calib.append([-sc[1] - l121, -sc[2] - l121, -sc[3] - l121])
        # vizinho em camada próxima: deslocado radialmente s·σ_R(UKF) a partir da verdade
        uR = yT[:3] / np.linalg.norm(yT[:3])
        sR = np.sqrt(uR @ Pp @ uR)
        zn = yT[:3] + rng.choice([-1, 1]) * rng.uniform(1.5, 3.0) * sR * uR + SIG_M * rng.standard_normal(3)
        neg.append(scores(zn, m0, P0, e)[0])
        # negativo agnóstico: direção aleatória na métrica do UKF, distância 1,5–3σ
        Lp = np.linalg.cholesky(Pp); dvec = rng.standard_normal(3); dvec /= np.linalg.norm(dvec)
        za = yT[:3] + rng.uniform(1.5, 3.0) * (Lp @ dvec) + SIG_M * rng.standard_normal(3)
        neg2.append(scores(za, m0, P0, e)[0])
    pos, neg, neg2, calib = map(np.array, (pos, neg, neg2, calib))
    res[s] = dict(
        auc={k: round(float(auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(NAMES)},
        auc_agn={k: round(float(auc(pos[:, i], neg2[:, i])), 3) for i, k in enumerate(NAMES)},
        tpr5_agn={k: round(tpr_at(pos[:, i], neg2[:, i]), 3) for i, k in enumerate(NAMES)},
        tpr5={k: round(tpr_at(pos[:, i], neg[:, i]), 3) for i, k in enumerate(NAMES)},
        ukf_gate99_miss=round(float(np.mean(pos[:, 0] > 11.345)), 3),
        logev_err_median={k: round(float(np.median(calib[:, j])), 3) for j, k in enumerate(["UKF", "GM15", "GM41"])},
        logev_err_p05={k: round(float(np.quantile(calib[:, j], 0.05)), 3) for j, k in enumerate(["UKF", "GM15", "GM41"])},
    )
    np.savez(f"e6_s{s}.npz", pos=pos, neg=neg, neg2=neg2, calib=calib)
    print(s, json.dumps(res[s]), f"({time.time()-t:.0f}s)", flush=True)
json.dump(res, open("e6_summary.json", "w"), indent=1)
