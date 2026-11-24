"""Revisão (exploratório): estado de catálogo por DETERMINAÇÃO DE ÓRBITA em vez de verdade + erro sintético.
Cenário do E5, estudo B (LEO 550 km, 53°, maneuver labels DORIS, a priori aprendida LOSO). Para o alvo e para o vizinho,
um dia de rastreio ANTES de t0: NOBS atribuíveis ópticos (mesmo modelo de medida) gerados pela dinâmica verdadeira
(com variação de densidade se MISMATCH=1); ajuste em lote (Gauss-Newton, 7 parâmetros: estado + fator de densidade β)
com o modelo do estimador; o estado e a covariância do ajuste propagados até t0 formam a trilha de catálogo (covariância
cheia e correlacionada, sem inflação). Depois, tudo como no e11_numeric (UCT após 1 dia; M, Mn, Z, D, Dn, F).
Env: MEAS, MISMATCH, NOBS (4), COV_INFL (1 = covariância do ajuste como sai), EXTRA, EXTRA2, TAG.
Uso: python e22_od.py N SIG_VI SEED   Saída: e11_rows_<TAG>_..."""
import os, sys, json, time
import numpy as np
import e11_numeric as X
import e10_numeric as E
from orbit import ref_state
from e6_nonlinear import prior_cov
NOBS = int(os.environ.get("NOBS", "4")); COV_INFL = float(os.environ.get("COV_INFL", "1")); DAY = 86400.0

def obs_tracklet(y, rng):
    st = X.station_for(y, rng); z = X.h_tracklet(y, st)
    sa = X.SIG_ANG / max(np.cos(z[1]), 0.2); sr = X.SIG_RATE / max(np.cos(z[1]), 0.2)
    Rm = np.diag([sa ** 2, X.SIG_ANG ** 2, sr ** 2, X.SIG_RATE ** 2]); zz = z + np.sqrt(np.diag(Rm)) * rng.standard_normal(4)
    zz[0] = (zz[0] + np.pi) % (2 * np.pi) - np.pi
    return zz, Rm, st

def od(x_start, betas, rng, P_guess):
    """um dia de rastreio a partir de x_start (verdade, t=-1 dia); devolve verdade em t0, média e cov. 6x6 do ajuste em t0"""
    tt = np.sort(np.round(rng.uniform(0.0, 0.85 * DAY, NOBS) / E.DT) * E.DT)
    ys = []; t = 0.0; x = x_start.copy()
    for ti in tt:
        x = E._seg_prop(x, betas, t, ti) if ti > t else x; t = ti; ys.append(x.copy())
    x_end = E._seg_prop(x, betas, t, DAY)
    Z = [obs_tracklet(y, rng) for y in ys]
    W = np.zeros((4 * NOBS, 4 * NOBS)); zv = np.concatenate([z for z, _, _ in Z])
    for i, (_, Rm, _) in enumerate(Z): W[4 * i:4 * i + 4, 4 * i:4 * i + 4] = np.linalg.inv(Rm)
    p_prior = np.r_[x_start + np.linalg.cholesky(P_guess) @ rng.standard_normal(6), 0.0]
    Pp = np.zeros((7, 7)); Pp[:6, :6] = 25.0 * P_guess; Pp[6, 6] = E.SIG_BETA ** 2; Ppi = np.linalg.inv(Pp)
    step = np.r_[1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-2]

    def hfun(P7):  # P7 (n,7) -> medidas (n, 4*NOBS)
        Y = E.prop7(np.repeat(P7[:, None], NOBS, 1), 0.0, tt[None, :])[..., :6]
        return np.concatenate([X.h_tracklet(Y[:, i], Z[i][2]) for i in range(NOBS)], -1)

    def resid(Pset):
        Hs = hfun(Pset); return X.wrap((zv[None] - Hs).reshape(len(Pset), NOBS, 4)).reshape(len(Pset), -1)
    def cost(p_, r_): return 0.5 * (r_ @ W @ r_ + (p_ - p_prior) @ Ppi @ (p_ - p_prior))
    p = p_prior.copy(); Rs = resid(np.vstack([p, p + np.diag(step)])); Jc = cost(p, Rs[0]); lam = 1e-2
    for _ in range(60):
        Jm = ((Rs[0] - Rs[1:]) / step[:, None]).T                                # dh/dp
        A = Jm.T @ W @ Jm + Ppi; b = Jm.T @ W @ Rs[0] - Ppi @ (p - p_prior)
        dp = np.linalg.solve(A + lam * np.diag(np.diag(A)), b); pn = p + dp
        try:
            Rn = resid(np.vstack([pn, pn + np.diag(step)])); Jn = cost(pn, Rn[0])
        except Exception:
            Jn = np.inf
        if np.isfinite(Jn) and Jn < Jc:
            p, Rs, Jc = pn, Rn, Jn; lam = max(lam / 5, 1e-7)
            if np.all(np.abs(dp) < 1e-3 * step): break
        else:
            lam *= 10
            if lam > 1e9: break
    Jm = ((Rs[0] - Rs[1:]) / step[:, None]).T; A = Jm.T @ W @ Jm + Ppi
    Cp = np.linalg.inv(A)
    Ps = np.vstack([p, p + np.diag(step)]); Ye = E.prop7(Ps, 0.0, DAY)
    Phi = ((Ye[1:] - Ye[0]) / step[:, None]).T                                  # (7,7)
    Ce = Phi @ Cp @ Phi.T; Ce = 0.5 * (Ce + Ce.T)
    return x_end, Ye[0, :6], COV_INFL ** 2 * Ce[:6, :6], float(p[6]), COV_INFL ** 2 * Ce

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2]); seed = int(sys.argv[3]); tag = os.environ.get("TAG", "od2")
    MEAS = X.MEAS; MIS = X.MISMATCH
    L = X.load_labels()
    lab = L[(L.orbit_class == "LEO") & L.delta_v.notna() & (L.delta_v > 1e-4) & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    sats = sorted(lab.norad_id.unique()); priors = {s: X.make_priors(X.fold_prior(L, s)) for s in sats}
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(1000 * s_vI)), int(MIS), MEAS == "tracklet", 22]))
    fn = f"e11_rows_{tag}_{MEAS}_s{s_vI}_seed{seed}_K{X.K2}{'_mis' if MIS else ''}{'_ci%g' % COV_INFL if COV_INFL != 1 else ''}.json"
    rows = []; t0 = time.time(); nees = []
    for it in range(N):
        xs = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi))
        P0p, e = prior_cov(xs, s_vI); Lp0 = np.linalg.cholesky(P0p)
        beta = X.SIG_BETA_TRUE * rng.standard_normal()
        if MIS:
            b = [0.4 * rng.standard_normal()]
            for _ in range(15):
                b.append(0.7 * b[-1] + 0.4 * np.sqrt(1 - 0.49) * rng.standard_normal())
            bod, bgap = np.array(b[:8] + [b[8]]), np.array(b[8:] + [b[-1]])
        else:
            bod = bgap = np.full(10, beta)
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6); xsN = xs + rng.uniform(1.5, 3.0) * Lp0 @ d6
        x0, xh, P6, bh, C7 = od(xs, bod, rng, P0p); xN, xNh, PN6, bhN, C7N = od(xsN, bod, rng, P0p)
        ex = x0 - xh; nees.append(float(ex @ np.linalg.solve(P6, ex)))
        held = sats[rng.integers(len(sats))]
        g = lab[lab.norad_id == held].iloc[rng.integers((lab.norad_id == held).sum())]
        man = (g.maneuver_type, float(g.delta_v)); tm = np.round(rng.uniform(0.05, 0.95) * X.T / X.DT) * X.DT; sgn = rng.choice([-1, 1])
        y0 = X.truth(x0, beta, None, rng, betas=bgap); yM = X.truth(x0, beta, man, rng, tm, sgn, bgap); yD = X.truth(xN, beta, None, rng, betas=bgap)
        e6 = prior_cov(xh, s_vI)[1]
        E.BETA0, E.P7X = bh, 0.5 * (C7 + C7.T); trk = X.Track3(xh, P6, e6)
        E.BETA0, E.P7X = bhN, 0.5 * (C7N + C7N.T); trkN = X.Track3(xNh, PN6, prior_cov(xNh, s_vI)[1]); E.BETA0, E.P7X = 0.0, None
        z0, R0, m0 = X.make_uct(rng, y0); zM, RM, mM = X.make_uct(rng, yM)
        zD, RD, mD = X.make_uct(rng, yD); zF, RF, mF = X.make_uct(rng, y0 + X.field_offset(rng, y0))
        pr = priors[held]
        row = dict(sat=int(held), tau=man[0], s=man[1], tm=float(tm), sgn=int(sgn), p1=pr["learn"][2], nees=nees[-1],
                   od_sig_pos=float(np.sqrt(np.trace(P6[:3, :3]))), od_err_pos=float(np.linalg.norm(ex[:3])), beta_hat=bh, beta=float(beta),
                   Z=trk.scores(z0, R0, m0, pr), M=trk.scores(zM, RM, mM, pr), D=trk.scores(zD, RD, mD, pr), F=trk.scores(zF, RF, mF, pr, refine=False))
        row["Mn"] = trkN.scores(zM, RM, mM, pr, refine=False); row["Dn"] = trkN.scores(zD, RD, mD, pr, refine=False)
        for c in ("Z", "M", "D", "F", "Mn", "Dn"):
            row[c] = {k: (float("%.7g" % v) if isinstance(v, float) else v) for k, v in row[c].items()}
        rows.append(row)
        if (it + 1) % 25 == 0:
            print(f"{it+1}/{N} ({time.time()-t0:.0f}s) nees_med {np.median(nees):.1f}", flush=True)
            json.dump(dict(meta=dict(complete=False, n_done=len(rows)), rows=rows), open(fn + ".partial", "w"))
    meta = dict(N=N, s_vI=s_vI, seed=seed, meas=MEAS, mismatch=MIS, nobs=NOBS, cov_infl=COV_INFL, tag=tag, complete=True,
                nees_median=float(np.median(nees)), nees_p90=float(np.quantile(nees, 0.9)), seconds=time.time() - t0)
    json.dump(dict(meta=meta, rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, meta)
