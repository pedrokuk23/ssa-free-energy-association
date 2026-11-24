"""Revisão (exploratório): (i) decomposição F = I + termo de normalização (Laplace) + resto, com I o mínimo do
funcional de energia (estado inicial, fator de densidade, medida; Σw = 0) obtido por otimização direta; (ii) referência
de evidência INDEPENDENTE da aproximação do estimador: amostragem por importância centrada no MAP (t de Student,
ν = 5, escala 2H⁻¹, 10 % defensivo na priori), estratificada em positivos, vizinhos (1,5-3σ) e objetos de campo.
Hipótese natural apenas (sem manobra), cenário do E4/E5 (LEO 550 km, 1 dia). Env: MEAS, NS (20000).
Uso: python e24_decomp.py N SIG_VI SEED   Saída: e24_<MEAS>_s<SIG>.json"""
import os, sys, json, time
import numpy as np
from scipy.special import logsumexp
from scipy import stats
import e11_numeric as X
import e10_numeric as E
from orbit import ref_state
from e6_nonlinear import prior_cov
NS = int(os.environ.get("NS", "20000")); STEP = np.r_[1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-2]

def hz(meas, P7):
    return meas.h(E.prop7(P7, 0.0, E.T)[..., :6])

def res(meas, z, H):
    return z - H if meas.kind == "full" else X.wrap(z - H)

def map_part(meas, z, R, m7, P7, w, M, m0, P0):
    dz = len(z); Ri = np.linalg.inv(R); Ppi = np.linalg.inv(P7)
    lk = X.ev(meas, z, R, m0, P0, 2)[0]; F_gm = -float(logsumexp(lk + np.log(w)))
    J = lambda p_, r_: 0.5 * (r_ @ Ri @ r_ + (p_ - m7) @ Ppi @ (p_ - m7))
    best = None
    for kk in np.argsort(-(lk + np.log(w)))[:2]:
        o_ = lm(meas, z, M[kk].copy(), Ri, Ppi, m7, J)
        if best is None or o_[2] < best[2]: best = o_
    p, r, Jc, Hs = best
    Jm = (res(meas, z, Hs[0]) - res(meas, z, Hs[1:])).T / STEP; A = Jm.T @ Ri @ Jm + Ppi
    F_lap = Jc + 0.5 * dz * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(R)[1] + 0.5 * np.linalg.slogdet(P7)[1] + 0.5 * np.linalg.slogdet(A)[1]
    return F_gm, p, Jc, A, F_lap

def lm(meas, z, p, Ri, Ppi, m7, J):
    lam = 1e-2; Hs = hz(meas, np.vstack([p, p + np.diag(STEP)])); r = res(meas, z, Hs[0]); Jc = J(p, r)
    for _ in range(40):
        Jm = (res(meas, z, Hs[0]) - res(meas, z, Hs[1:])).T / STEP              # dr/dp com sinal de h
        A = Jm.T @ Ri @ Jm + Ppi; g = Jm.T @ Ri @ r - Ppi @ (p - m7)
        dp = np.linalg.solve(A + lam * np.diag(np.diag(A)), g); pn = p + dp
        Hn = hz(meas, np.vstack([pn, pn + np.diag(STEP)])); rn = res(meas, z, Hn[0]); Jn = J(pn, rn)
        if Jn < Jc:
            p, r, Jc, Hs = pn, rn, Jn, Hn; lam = max(lam / 5, 1e-7)
            if np.all(np.abs(dp) < 1e-3 * STEP): break
        else:
            lam *= 10
            if lam > 1e8: break
    return p, r, Jc, Hs

def is_ref(meas, z, R, m7, P7, rng, F_gm, p, Jc, A, F_lap):
    dz = len(z); Ri = np.linalg.inv(R)
    # amostragem por importância centrada no MAP (densidades calculadas à mão, em coordenadas branqueadas)
    from scipy.special import gammaln
    d = len(p); nu = 5.0; nd = int(0.1 * NS)
    C = 2.0 * np.linalg.inv(A); Lc = np.linalg.cholesky(0.5 * (C + C.T)); Lp7 = np.linalg.cholesky(0.5 * (P7 + P7.T))
    def lt(Sx):
        u = np.linalg.solve(Lc, (Sx - p).T).T; q = np.sum(u * u, 1)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi) - np.sum(np.log(np.diag(Lc)))
                - 0.5 * (nu + d) * np.log1p(q / nu))
    def lg(Sx):
        u = np.linalg.solve(Lp7, (Sx - m7).T).T
        return -0.5 * np.sum(u * u, 1) - 0.5 * d * np.log(2 * np.pi) - np.sum(np.log(np.diag(Lp7)))
    g1 = rng.standard_normal((NS - nd, d)); ch = rng.chisquare(nu, NS - nd)
    S = np.vstack([p + (g1 / np.sqrt(ch / nu)[:, None]) @ Lc.T, m7 + rng.standard_normal((nd, d)) @ Lp7.T])
    lq = np.logaddexp(np.log(0.9) + lt(S), np.log(0.1) + lg(S))
    H = hz(meas, S); rr = res(meas, z, H); ll = -0.5 * np.einsum("ni,ij,nj->n", rr, Ri, rr) - 0.5 * dz * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(R)[1]
    lw = ll + lg(S) - lq; F_is = -float(logsumexp(lw) - np.log(len(S))); ess = float(np.exp(2 * logsumexp(lw) - logsumexp(2 * lw)))
    return dict(GM=F_gm, I=float(Jc), LAP=float(F_lap), IS=F_is, ess=ess)

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2]); seed = int(sys.argv[3]); MEAS = X.MEAS
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(1000 * s_vI), MEAS == "tracklet", 24])); rows = []; t0 = time.time()
    for it in range(N):
        Xref = ref_state(M=rng.uniform(0, 2 * np.pi), raan=rng.uniform(0, 2 * np.pi)); P6, e = prior_cov(Xref, s_vI); Lp = np.linalg.cholesky(P6)
        x0 = Xref + Lp @ rng.standard_normal(6); beta = X.SIG_BETA_TRUE * rng.standard_normal()
        m7 = np.r_[Xref, 0.0]; P7 = np.zeros((7, 7)); P7[:6, :6] = P6; P7[6, 6] = E.SIG_BETA ** 2
        w, M, Pk = E.split7(m7, P7, e, E.K0); Sg = np.linalg.cholesky(7 * Pk)
        pts = np.concatenate([M[:, None], M[:, None] + Sg.T[None], M[:, None] - Sg.T[None]], 1)
        Wt = np.r_[0.0, np.full(14, 1 / 14)]; Y = E.prop7(pts, 0.0, E.T)[..., :6]
        m0 = np.einsum("j,kjd->kd", Wt, Y); D = Y - m0[:, None]; P0 = np.einsum("j,kjd,kje->kde", Wt, D, D)
        y0 = E._seg_prop(x0, np.full(10, beta), 0.0, E.T)
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6); yD = E._seg_prop(x0 + rng.uniform(1.5, 3.0) * Lp @ d6, np.full(10, beta), 0.0, E.T)
        row = {}
        for nm, y in (("pos", y0), ("nb", yD)):
            z, R, meas = X.make_uct(rng, y)
            try:
                row[nm] = is_ref(meas, z, R, m7, P7, rng, *map_part(meas, z, R, m7, P7, w, M, m0, P0))
            except Exception as ex:
                row[nm] = dict(err=repr(ex)[:80])
        rows.append(row)
        if (it + 1) % 10 == 0:
            print(f"{it+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(meta=dict(N=N, s_vI=s_vI, meas=MEAS, NS=NS, seed=seed, seconds=time.time() - t0), rows=rows, summary={})
    ok = lambda r, k: k in r and "GM" in r[k] and r[k]["ess"] > 100
    for k in ("pos", "nb", "field"):
        v = [r[k] for r in rows if ok(r, k)]
        if v:
            out["summary"][k] = dict(n=len(v), ess_med=round(float(np.median([x["ess"] for x in v])), 0),
                                     err_GM_med=round(float(np.median([abs(x["GM"] - x["IS"]) for x in v])), 3), err_GM_p90=round(float(np.quantile([abs(x["GM"] - x["IS"]) for x in v], 0.9)), 3),
                                     err_LAP_med=round(float(np.median([abs(x["LAP"] - x["IS"]) for x in v])), 3),
                                     norm_term_med=round(float(np.median([x["IS"] - x["I"] for x in v])), 2), norm_term_iqr=round(float(np.subtract(*np.quantile([x["IS"] - x["I"] for x in v], [0.75, 0.25]))), 2))
    pos = [r["pos"] for r in rows if ok(r, "pos")]
    for neg in ("nb", "field"):
        ng = [r[neg] for r in rows if ok(r, neg)]
        for s in ("GM", "I", "LAP", "IS"):
            a = np.array([x[s] for x in pos]); b = np.array([x[s] for x in ng])
            if len(a) and len(b):
                out["summary"][f"auc_{neg}_{s}"] = round(float(np.mean(a[:, None] < b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :])), 4)
    fn = f"e24_{MEAS}_s{s_vI}.json"; json.dump(out, open(fn, "w")); print("ok", fn, json.dumps(out["summary"]))
