"""E4: erro ABSOLUTO da evidência (UKF, GM15, GM41, GM81) contra referência por amostragem por importância
(proposta = posteriores UT por componente, alargadas, + mistura defensiva). Também AUC recalculada com N casos."""
import sys, json, time, numpy as np
from scipy.special import logsumexp
from e9_numeric import split7, prop7, SIG_BETA, T
from e6n_numeric import gm_logev, SIG_M
from orbit import ref_state; from e6_nonlinear import prior_cov

def mvn_logpdf(X, m, S):
    L = np.linalg.cholesky(S); d = np.linalg.solve(L, np.atleast_2d(X - m).T)
    return -0.5*np.sum(d*d, 0) - np.log(np.diag(L)).sum() - 0.5*len(m)*np.log(2*np.pi)

def is_logev(z, m, P, e, rng, K=41, N=20000, infl=1.5):
    w, M, Pk = split7(m, P, e, K); S = np.linalg.cholesky(7*Pk)
    pts = np.concatenate([M[:, None], M[:, None]+S.T[None], M[:, None]-S.T[None]], 1); Wt = np.r_[0, np.full(14, 1/14)]
    Y = prop7(pts, 0., T)[..., :3]; my = np.einsum("j,kjd->kd", Wt, Y); DY = Y-my[:, None]; DX = pts-M[:, None]
    Py = np.einsum("j,kjd,kje->kde", Wt, DY, DY)+SIG_M**2*np.eye(3); Cxy = np.einsum("j,kjd,kje->kde", Wt, DX, DY)
    lw = np.log(w)+np.array([mvn_logpdf(z, my[k], Py[k])[0] for k in range(K)]); a = np.exp(lw-logsumexp(lw))
    Kg = np.einsum("kde,kef->kdf", Cxy, np.linalg.inv(Py)); mpost = M+np.einsum("kde,ke->kd", Kg, z-my)
    Pp = Pk[None]-np.einsum("kde,kfe->kdf", Kg, Cxy); Pp = 0.5*(Pp+np.transpose(Pp, (0, 2, 1)))*infl**2
    D = np.sqrt(np.diag(P)); Wn = Pp/np.outer(D, D); lam, V = np.linalg.eigh(Wn); lam = np.clip(lam, 1e-5, None)
    Pp = np.einsum('kij,kj,klj->kil', V, lam, V)*np.outer(D, D)
    a = 0.9*a+0.1*w
    idx = rng.choice(K, N, p=a); X = np.empty((N, 7))
    for k in np.unique(idx):
        n = (idx == k).sum(); X[idx == k] = rng.multivariate_normal(mpost[k], Pp[k], n)
    lq = logsumexp(np.stack([np.log(a[k])+mvn_logpdf(X, mpost[k], Pp[k]) for k in range(K)]), 0)
    lp = mvn_logpdf(X, m, P)
    Yx = prop7(X, 0., T)[:, :3]; ll = -0.5*np.sum((Yx-z)**2, 1)/SIG_M**2-1.5*np.log(2*np.pi*SIG_M**2)
    lwt = lp+ll-lq; est = logsumexp(lwt)-np.log(N); ess = np.exp(2*logsumexp(lwt)-logsumexp(2*lwt))
    # erro-padrão MC do log (método delta)
    wt = np.exp(lwt-lwt.max()); se = np.std(wt)/np.sqrt(N)/np.mean(wt)
    return est, ess, se

if __name__ == "__main__":
    N = int(sys.argv[1]); rng = np.random.default_rng(77); out = {}
    for s in [0.2, 0.5, 1.0]:
        rows = []; t0 = time.time()
        for i in range(N):
            Xref = ref_state(M=rng.uniform(0, 2*np.pi), raan=rng.uniform(0, 2*np.pi)); P6, e = prior_cov(Xref, s)
            m = np.r_[Xref, 0]; P = np.zeros((7, 7)); P[:6, :6] = P6; P[6, 6] = SIG_BETA**2
            x = m+np.linalg.cholesky(P)@rng.standard_normal(7); z = prop7(x, 0., T)[:3]+SIG_M*rng.standard_normal(3)
            g = [gm_logev(z, m, P, e, K)[0] for K in (1, 15, 41, 81)]
            est, ess, se = is_logev(z, m, P, e, rng)
            rows.append(g+[est, ess, se])
        r = np.array(rows); err = r[:, 4:5]-r[:, :4]   # IS − GM  (>0: GM subestima evidência)
        out[s] = dict(n=N, abs_err_median={k: round(float(np.median(np.abs(err[:, i]))), 3) for i, k in enumerate(["UKF", "GM15", "GM41", "GM81"])},
                      err_median={k: round(float(np.median(err[:, i])), 3) for i, k in enumerate(["UKF", "GM15", "GM41", "GM81"])},
                      err_p90={k: round(float(np.percentile(np.abs(err[:, i]), 90)), 3) for i, k in enumerate(["UKF", "GM15", "GM41", "GM81"])},
                      ess_min=float(r[:, 5].min()), ess_median=float(np.median(r[:, 5])), se_max=float(r[:, 6].max()))
        print(s, json.dumps(out[s]), f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open("e4_is_summary.json", "w"), indent=1)
        np.save(f"e4_is_rows_s{s}.npy", r)
