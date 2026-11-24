"""E4 refeito com propagação numérica (J2–J4 + arrasto, densidade incerta): evidência UKF vs GM
após 1 dia, fix de posição σ = 50 m; validação da referência GM(K=81) por partículas."""
import numpy as np, json, sys, time
from scipy.special import logsumexp
from e9_numeric import split7, prop7, SIG_BETA, T
from orbit import ref_state
from e6_nonlinear import prior_cov
SIG_M = 50.0

def gm_logev(z, m, P, e6, K):
    w, M, Pk = split7(m, P, e6, K)
    S = np.linalg.cholesky(7 * Pk)
    pts = np.concatenate([M[:, None], M[:, None] + S.T[None], M[:, None] - S.T[None]], 1)
    Wt = np.r_[0.0, np.full(14, 1 / 14)]
    Y = prop7(pts, 0.0, T)[..., :3]
    my = np.einsum("j,kjd->kd", Wt, Y); D = Y - my[:, None]
    Py = np.einsum("j,kjd,kje->kde", Wt, D, D) + SIG_M**2 * np.eye(3)
    r = z - my; sol = np.linalg.solve(Py, r[..., None])[..., 0]; _, ld = np.linalg.slogdet(Py)
    lp = -0.5 * (np.sum(r * sol, -1) + ld + 3 * np.log(2 * np.pi))
    return logsumexp(lp + np.log(w)), my, Py

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

if __name__ == "__main__":
    N = int(sys.argv[1]); rng = np.random.default_rng(5); res = {}
    for s in [0.05, 0.2, 0.5, 1.0]:
        pos, neg, cal = [], [], []; t0 = time.time()
        for _ in range(N):
            Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
            P6, e = prior_cov(Xref, s); m = np.r_[Xref, 0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
            x = m + np.linalg.cholesky(P) @ rng.standard_normal(7); yT = prop7(x, 0.0, T)[:3]
            z = yT + SIG_M * rng.standard_normal(3)
            l1, my, Py = gm_logev(z, m, P, e, 1); l15 = gm_logev(z, m, P, e, 15)[0]; l41 = gm_logev(z, m, P, e, 41)[0]
            l81 = gm_logev(z, m, P, e, 81)[0]
            pos.append([-l1, -l15, -l41]); cal.append([l81 - l1, l81 - l15, l81 - l41])
            Pp = Py[0] - SIG_M**2 * np.eye(3); Lp = np.linalg.cholesky(Pp)
            d = rng.standard_normal(3); d /= np.linalg.norm(d)
            za = yT + rng.uniform(1.5, 3.0) * Lp @ d + SIG_M * rng.standard_normal(3)
            neg.append([-gm_logev(za, m, P, e, K)[0] for K in (1, 15, 41)])
        pos, neg, cal = map(np.array, (pos, neg, cal))
        res[s] = dict(auc={k: round(float(auc(pos[:, i], neg[:, i])), 3) for i, k in enumerate(["UKF", "GM15", "GM41"])},
                      logev_err_median={k: round(float(np.median(cal[:, i])), 3) for i, k in enumerate(["UKF", "GM15", "GM41"])})
        print(s, json.dumps(res[s]), f"({time.time()-t0:.0f}s)", flush=True)
    # validação da referência por partículas
    chk = []
    for s in [0.5, 1.0]:
        for _ in range(3):
            Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
            P6, e = prior_cov(Xref, s); m = np.r_[Xref, 0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
            Lc = np.linalg.cholesky(P); x = m + Lc @ rng.standard_normal(7); z = prop7(x, 0.0, T)[:3] + SIG_M * rng.standard_normal(3)
            l81 = gm_logev(z, m, P, e, 81)[0]
            Xp = m + rng.standard_normal((200_000, 7)) @ Lc.T; Y = prop7(Xp, 0.0, T)[:, :3]
            a = -0.5 * np.sum((Y - z)**2, 1) / SIG_M**2 - 1.5 * np.log(2 * np.pi * SIG_M**2)
            lmc = logsumexp(a) - np.log(len(a)); ess = np.exp(2 * logsumexp(a) - logsumexp(2 * a))
            chk.append(dict(s=s, gm81=float(l81), mc=float(lmc), ess=float(ess))); print(chk[-1], flush=True)
    res["particle_check"] = chk
    json.dump(res, open("e6n_summary.json", "w"), indent=1)
