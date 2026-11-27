"""E6c covariance check with the production settings (audit, revision 3, items R01-R03).
Re-runs e17_scoreK3.py scoring (same events, seeds, refine=False, UCTs from make_uct) for selected real events and
draws, with (a) the production loggauss, instrumented (non-finite and non-positive-definite matrices, batch fallback
count), and (b) a per-matrix safe loggauss (finiteness, Cholesky in correlation scale, individual eigenvalue repair).
Compares scores with the archived e17K3 rows and the joint decisions (target vs neighbour) between (a) and (b).
Usage: MEAS=tracklet|full python e33_e6c_covariance_check.py <events.json> <eid,eid,...> <ndraws> [pkgdir]"""
import os, sys, json, time
P = sys.argv[4] if len(sys.argv) > 4 else "."
sys.path.insert(0, P); os.chdir(P)

import numpy as np
import e11_numeric as X, e10_numeric as E, numprop as NP
from e6_nonlinear import prior_cov
from e17_scoreK3 import slim, CRAM, KEEP
ST = dict(calls=0, mats=0, nonfinite=0, nonpd=0, fallback=0, repaired=0, corr_cond_max=0.0)
try:   # after the R01-R02 fix the modules import the validated routine; compare against the original arithmetic
    import gauss_logpdf as GL
    prod_X = GL.loggauss_legacy; NEW = GL.loggauss
    def prod_E(r, S):
        sol = np.linalg.solve(S, r[..., None])[..., 0]; _, ld = np.linalg.slogdet(S)
        return -0.5 * (np.sum(r * sol, -1) + ld + r.shape[-1] * np.log(2 * np.pi))
except ImportError:
    prod_X, prod_E = X.loggauss, E.loggauss; NEW = safe_loggauss
def inspect(r, S):
    S = np.asarray(S, float); ST["calls"] += 1; ST["mats"] += int(np.prod(S.shape[:-2])) if S.ndim > 2 else 1
    fin = np.isfinite(S).all((-1, -2)); ST["nonfinite"] += int(np.sum(~fin))
    Sf = np.where(fin[..., None, None], S, np.eye(S.shape[-1]))
    d = np.sqrt(np.abs(np.einsum("...ii->...i", Sf))); d = np.where(d > 0, d, 1.0)
    C = Sf / (d[..., :, None] * d[..., None, :]); lam = np.linalg.eigvalsh(0.5 * (C + np.swapaxes(C, -1, -2)))
    ST["nonpd"] += int(np.sum((lam[..., 0] <= 0) & fin)); pos = lam[..., 0] > 0
    if np.any(pos): ST["corr_cond_max"] = max(ST["corr_cond_max"], float(np.max(lam[..., -1][pos] / lam[..., 0][pos])))
    try:
        np.linalg.solve(S, np.asarray(r)[..., None])
    except np.linalg.LinAlgError:
        ST["fallback"] += 1
def make_prod(f):
    def g(r, S):
        inspect(r, S); return f(r, S)
    return g
def safe_loggauss(r, S, tol=1e-12):
    """Per-matrix: non-finite -> -inf; symmetrize; Cholesky in correlation scale; non-PD matrices repaired
    individually by clipping correlation eigenvalues at tol*max (counted)."""
    S = np.asarray(S, float); r = np.asarray(r, float); sh = np.broadcast_shapes(S.shape[:-2], r.shape[:-1])
    S = np.broadcast_to(S, sh + S.shape[-2:]).reshape(-1, *S.shape[-2:]); r = np.broadcast_to(r, sh + r.shape[-1:]).reshape(-1, r.shape[-1])
    out = np.empty(len(S)); n = S.shape[-1]
    for i in range(len(S)):
        Si = 0.5 * (S[i] + S[i].T)
        if not (np.isfinite(Si).all() and np.isfinite(r[i]).all()):
            out[i] = -np.inf; continue
        d = np.sqrt(np.abs(np.diag(Si))); d[d == 0] = 1.0; C = Si / np.outer(d, d)
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            lam, V = np.linalg.eigh(C); lam = np.maximum(lam, tol * lam.max()); C = (V * lam) @ V.T
            Lc = np.linalg.cholesky(C); ST["repaired"] += 1
        u = np.linalg.solve(Lc, r[i] / d)
        out[i] = -0.5 * (u @ u + 2 * np.sum(np.log(np.diag(Lc))) + 2 * np.sum(np.log(d)) + n * np.log(2 * np.pi))
    return out.reshape(sh)
def score_all(trk, trkN, z, R, m, pr):
    return dict(M=slim(trk.scores(z, R, m, pr, refine=False)), Mn=slim(trkN.scores(z, R, m, pr, refine=False)))
if __name__ == "__main__":
    EVF, eids, KD = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]); MEAS = X.MEAS; s_vI, seed = 0.05, 17
    allev = json.load(open(EVF))["events"]; L = X.load_labels()
    arch = {}
    import glob
    for f in glob.glob(f"e17K3_{os.path.splitext(os.path.basename(EVF))[0]}_{MEAS}_s0.05_seed17_c*.json"):
        for row in json.load(open(f))["rows"]: arch[row["eid"]] = row
    for eid in eids:
        e = dict(allev[int(eid)]); e["eid"] = eid; pr = X.make_priors(X.fold_prior(L, e["sat"]))
        if e.get("regime", "LEO") != "LEO": NP.set_forces(1, e["epoch_j2000"], CRAM.get(e.get("prn", "G")[0], 0.02))
        else: NP.set_forces(0)
        x0 = np.array(e["x0"]); yT = np.array(e["yT"]); P0, en = prior_cov(x0, s_vI); Lp = np.linalg.cholesky(P0)
        for k in range(KD):
            t = time.time()
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(eid), k, MEAS == "tracklet", 17]))
            Xref = x0 - Lp @ rng.standard_normal(6); P0r, er = prior_cov(Xref, s_vI); trk = X.Track3(Xref, P0r, er)
            y0m = X.truth(x0, 0.0, None, rng); d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
            xN = x0 + rng.uniform(1.5, 3.0) * Lp @ d6; yD = X.truth(xN, 0.0, None, rng); xN_hat = xN + Lp @ rng.standard_normal(6)
            P0n, enn = prior_cov(xN_hat, s_vI); trkN = X.Track3(xN_hat, P0n, enn)
            z, R, m = X.make_uct(rng, yT)
            X.loggauss, E.loggauss = make_prod(prod_X), make_prod(prod_E); a = score_all(trk, trkN, z, R, m, pr)
            b = a if os.environ.get("ONLY_PROD") else None
            if b is None: X.loggauss = E.loggauss = NEW; b = score_all(trk, trkN, z, R, m, pr)
            X.loggauss, E.loggauss = prod_X, prod_E
            ar = arch.get(eid, {}).get("draws", [None] * (k + 1))[k] if eid in arch and k < len(arch[eid]["draws"]) else None
            rep = max(abs(a["M"][q] - ar["M"][q]) / max(1, abs(ar["M"][q])) for q in a["M"] if ar and q in ar["M"]) if ar else float("nan")
            dmax = max(abs(a[s][q] - b[s][q]) for s in ("M", "Mn") for q in a[s])
            flips = [q for q in a["M"] if (a["M"][q] < a["Mn"][q]) != (b["M"][q] < b["Mn"][q])]
            marg = min(abs(a["M"][q] - a["Mn"][q]) for q in ("learn", "CTRLu"))
            print(json.dumps(dict(ev=EVF, eid=eid, regime=e.get("regime", "LEO"), meas=MEAS, k=k, rel_diff_archive=rep, max_abs_diff_prod_vs_safe=dmax,
                                  decision_flips=flips, min_margin_learn_ctrlu=marg, **({"per_method": {q: [a["M"][q], ar["M"].get(q)] for q in a["M"]}} if os.environ.get("ONLY_PROD") and ar else {}), sec=round(time.time() - t))), flush=True)
    print("STAT", json.dumps(ST)); print("STAT_NEW", json.dumps(getattr(sys.modules.get("gauss_logpdf"), "STATS", {})))
