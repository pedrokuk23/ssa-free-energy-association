"""E6c covariance check with the production settings (audit, revision 3, items R01-R03).
Re-runs e17_scoreK3.py scoring (same events, seeds, refine=False, UCTs from make_uct) for selected real events and
draws, with (a) the production loggauss, instrumented (non-finite and non-positive-definite matrices, batch fallback
count), and (b) a per-matrix safe loggauss (finiteness, Cholesky in correlation scale, individual eigenvalue repair).
Compares scores with the archived e17K3 rows and the joint decisions (target vs neighbour) between (a) and (b).
Revision 4 (R4-01, R4-03): unrounded target (M) and neighbour (Mn) scores of both routines and of the archive, per
method; decisions with the analysis rule (1 / 0.5 tie / 0) compared new vs legacy and new vs archive; asymmetry
counters and their source lines; fallback counters of _inv/_solve (gauss_logpdf.STATS).
PERTURB_EPS=e1,e2,... also rescores each draw (original routine) with the predicted covariance scaled by (1+eps),
to measure the sensitivity of the scores and decisions to rounding-level differences between platforms.
Usage: MEAS=tracklet|full [K0DRAW=k0] [PERTURB_EPS=...] python e33_e6c_covariance_check.py <events.json> <eid,eid,...> <ndraws> [pkgdir]"""
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
    return dict(M=trk.scores(z, R, m, pr, refine=False), Mn=trkN.scores(z, R, m, pr, refine=False))
def dec(Ft, Fn):   # decision rule of e27_circ_analyze / e12K_analyze: 1 if target better, 0.5 tie, 0 otherwise
    return 1.0 if Ft < Fn else (0.5 if Ft == Fn else 0.0)
if __name__ == "__main__":
    EVF, eids, KD = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]); MEAS = X.MEAS; s_vI, seed = 0.05, 17
    K0 = int(os.environ.get("K0DRAW", "0"))
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
        for k in range(K0, K0 + KD):
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
            pert = {}
            for ep in [float(v) for v in os.environ.get("PERTURB_EPS", "").split(",") if v]:
                ev0 = X.ev                                   # rounding-level perturbation of the predicted covariance
                X.ev = lambda meas, z_, R_, mT, PT, it=2, ep=ep, ev0=ev0: ev0(meas, z_, R_, mT, PT * (1 + ep), it)
                c = score_all(trk, trkN, z, R, m, pr); X.ev = ev0
                pert[repr(ep)] = dict(M=slim(c["M"]), Mn=slim(c["Mn"]))
            ar = arch[eid]["draws"][k] if eid in arch and k < len(arch[eid]["draws"]) else None
            meth = [q for q in KEEP if q in a["M"] and not q.startswith("pH1_")]
            per = {}
            for q in meth:
                per[q] = dict(legacy=[a["M"][q], a["Mn"][q]], new=[b["M"][q], b["Mn"][q]],
                              archive=[ar["M"].get(q), ar["Mn"].get(q)] if ar else None)
            dmax = max(abs(a[s][q] - b[s][q]) for s in ("M", "Mn") for q in meth)
            sa, sb = dict(M=slim(a["M"]), Mn=slim(a["Mn"])), dict(M=slim(b["M"]), Mn=slim(b["Mn"]))
            rel = {s: max((abs(sa[s][q] - ar[s][q]) / max(1, abs(ar[s][q])) for q in meth if q in ar[s]), default=float("nan")) for s in ("M", "Mn")} if ar else None
            flips_new_legacy = [q for q in meth if dec(sa["M"][q], sa["Mn"][q]) != dec(sb["M"][q], sb["Mn"][q])]
            flips_arch = ({q: [dec(ar["M"][q], ar["Mn"][q]), dec(sa["M"][q], sa["Mn"][q]), dec(sb["M"][q], sb["Mn"][q])]
                           for q in meth if q in ar["M"] and dec(ar["M"][q], ar["Mn"][q]) != dec(sb["M"][q], sb["Mn"][q])}) if ar else None
            print(json.dumps(dict(ev=EVF, eid=eid, regime=e.get("regime", "LEO"), meas=MEAS, k=k,
                                  rel_diff_archive_M=rel and rel["M"], rel_diff_archive_Mn=rel and rel["Mn"],
                                  max_abs_diff_legacy_vs_new_unrounded=dmax, decision_changes_new_vs_legacy=flips_new_legacy,
                                  decision_changes_vs_archive=flips_arch, per_method=per,
                                  perturbed={ep: dict(scores=v, decisions={q: dec(v["M"][q], v["Mn"][q]) for q in meth}) for ep, v in pert.items()},
                                  decisions_unperturbed={q: dec(sa["M"][q], sa["Mn"][q]) for q in meth}, sec=round(time.time() - t))), flush=True)
    print("STAT", json.dumps(ST)); print("STAT_NEW", json.dumps(getattr(sys.modules.get("gauss_logpdf"), "STATS", {})))
    print("ASYM_SOURCES", json.dumps(getattr(sys.modules.get("gauss_logpdf"), "ASYM_SOURCES", {})))
