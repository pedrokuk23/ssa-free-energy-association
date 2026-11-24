"""Ablação da priori (propagação exata idêntica em todas as variantes):
  learn      : tipo aprendido (I/C) × |Δv| aprendido por tipo          [proposto]
  dir_only   : tipo aprendido, |Δv| uniforme nos bins log
  mag_only   : direção uniforme R/I/C, |Δv| aprendido (agregado)
  flat       : direção uniforme R/I/C, |Δv| uniforme nos bins log
"""
import numpy as np, json, sys, time
from scipy.special import logsumexp
import e8_uct_learned as E
from orbit import propagate

def variant_blocks(trk_PR, name):
    cen = trk_PR["centers"]; B = len(cen); q = trk_PR["q"]; pt = trk_PR["ptype"]
    uni = np.full(B, 1 / B)
    pooled = pt["in-track"] * q["in-track"] + pt["cross-track"] * q["cross-track"]
    if name == "learn":    return [(1, pt["in-track"], q["in-track"]), (2, pt["cross-track"], q["cross-track"])]
    if name == "dir_only": return [(1, pt["in-track"], uni), (2, pt["cross-track"], uni)]
    if name == "mag_only": return [(0, 1/3, pooled), (1, 1/3, pooled), (2, 1/3, pooled)]
    if name == "flat":     return [(0, 1/3, uni), (1, 1/3, uni), (2, 1/3, uni)]
    if name.startswith("maxwell"):   # |Δv| de gaussiana isotrópica N(0,σ²I): Maxwell, escala única
        sig = float(name.split("_")[1]); ed = np.logspace(-3, 1.1, B + 1)
        from scipy.stats import maxwell
        qb = np.diff(maxwell.cdf(ed, scale=sig)); qb = (qb + 1e-12) / (qb + 1e-12).sum()
        return [(0, 1/3, qb), (1, 1/3, qb), (2, 1/3, qb)]

class TrackV(E.Track):
    def __init__(self, m0, P0, e, PR):
        super().__init__(m0, P0, e, PR)
        # recomputa blocos por direção (R, I, C) uma vez; combina pesos por variante
        w2, M2, P2 = E.split(m0, P0, e, E.K2); pts2, _ = E.ut_points(M2, P2)
        Xj = propagate(pts2[:, None], E.TJ[None, :, None]); mj = np.einsum("j,kajd->kad", E.W_UT, Xj)
        R, I, C = E.ric_dirs(mj); dirs = [R, I, C]; cen = PR["centers"]; B = len(cen)
        self.blk = {}
        for di in range(3):
            for sg in (+1, -1):
                dv = sg * cen[None, None, :, None] * dirs[di][:, :, None, :]
                X = Xj[:, :, None].repeat(B, 2).copy(); X[..., 3:] += dv[:, :, :, None, :]
                m, P = E.ut_stats(propagate(X, (E.T - E.TJ)[None, :, None, None]))
                Xs = mj[:, :, None].repeat(B, 2).copy(); Xs[..., 3:] += 1.1 * dv
                Xc = mj[:, :, None].repeat(B, 2).copy(); Xc[..., 3:] += dv
                g = (propagate(Xs, (E.T - E.TJ)[None, :, None]) - propagate(Xc, (E.T - E.TJ)[None, :, None])) / 0.1
                P = P + 0.35**2 * np.einsum("...d,...e->...de", g, g)
                self.blk[(di, sg)] = (m.reshape(-1, 6), P.reshape(-1, 6, 6))
        self.lw_base = np.broadcast_to((np.log(w2)[:, None, None] - np.log(len(E.TJ))), (E.K2, len(E.TJ), B)).reshape(-1)
        self.B = B

    def variants(self, z, S):
        p1 = np.log(self.PR["p1"]); p0 = np.log1p(-self.PR["p1"])
        l0 = logsumexp(E.loggauss(z - self.m0, self.P0 + S) + np.log(self.w0))
        cache = {k: E.loggauss(z - m, P + S) for k, (m, P) in self.blk.items()}
        out = {"GM": -l0}
        for name in ["learn", "dir_only", "mag_only", "flat", "maxwell_0.3", "maxwell_0.03", "maxwell_3"]:
            terms = []
            for di, pdir, qb in variant_blocks(self.PR, name):
                lq = np.tile(np.log(qb), E.K2 * len(E.TJ))
                for sg in (+1, -1):
                    terms.append(cache[(di, sg)] + self.lw_base + lq + np.log(pdir) + np.log(0.5))
            l1 = logsumexp(np.concatenate(terms))
            out[name] = -np.logaddexp(p0 + l0, p1 + l1)
        return out

if __name__ == "__main__":
    N = int(sys.argv[1]); s_vI = float(sys.argv[2])
    PR = E.learn_prior(); PR["p1"] = float(min(0.5, PR["rate"]))
    rng = np.random.default_rng(777); test = PR["test"]; rows = []; t0 = time.time()
    for it in range(N):
        Xref, P0, e, x0 = E.make_object(rng, s_vI); trk = TrackV(Xref, P0, e, PR)
        tau, s, _ = test[rng.integers(len(test))]
        yM = E.truth_T(x0, (tau, float(s)), rng); zM, SM = E.uct(rng, yM)
        y0 = E.truth_T(x0)
        R, I, C = E.ric_dirs(y0); Qm = np.stack([R, I, C], -1)
        off = np.r_[Qm @ (rng.uniform(-1, 1, 3) * [5e3, 50e3, 15e3]), Qm @ (rng.uniform(-1, 1, 3) * [1, 1, 10])]
        zF, SF = E.uct(rng, y0 + off)
        z0, S0 = E.uct(rng, y0)
        mu = trk.w0 @ trk.m0
        Pu = np.einsum("k,kde->de", trk.w0, trk.P0 + np.einsum("kd,ke->kde", trk.m0 - mu, trk.m0 - mu))
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
        zN, SN = E.uct(rng, y0 + rng.uniform(1.5, 3.0) * np.linalg.cholesky(Pu) @ d6)
        rows.append(dict(tau=tau, s=float(s), M=trk.variants(zM, SM), F=trk.variants(zF, SF),
                         Z=trk.variants(z0, S0), N=trk.variants(zN, SN)))
        if (it + 1) % 100 == 0: print(f"{it+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    V = ["GM", "learn", "dir_only", "mag_only", "flat", "maxwell_0.3", "maxwell_0.03", "maxwell_3"]
    def tpr(sel):
        P = [r for r in rows if sel(r)]
        return {v: round(float(np.mean(np.array([r["M"][v] for r in P]) < np.quantile([r["F"][v] for r in rows], 0.01))), 3) for v in V}
    res = {"dv>=0.05": tpr(lambda r: r["s"] >= 0.05), "cross-track": tpr(lambda r: r["tau"] == "cross-track"),
           "in-track>=0.05": tpr(lambda r: r["tau"] == "in-track" and r["s"] >= 0.05), "todas": tpr(lambda r: True)}
    json.dump(dict(p1=PR["p1"], rows=rows), open(f"e8_ablrows_s{s_vI}.json", "w"))
    print(json.dumps(res, indent=1)); json.dump(res, open(f"e8_ablation2_s{s_vI}.json", "w"), indent=1)
