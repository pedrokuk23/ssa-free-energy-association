"""E6 (dados reais) — passo 2: pontua os eventos reais com o estimador do E5 v3 (e11_numeric).
Verdade = órbitas DORIS precisas (manobra real, forças reais); sintéticos: erro de catálogo, ruído da UCT,
vizinho catalogado e objetos de campo (propagados pelo modelo a partir do estado real em t0).
Env: MEAS=full|tracklet, K1, USE_GPU. Uso: python e12_score.py SIG_VI SEED
"""
import os, sys, json, time
import numpy as np
import e11_numeric as X
from e6_nonlinear import prior_cov

if __name__ == "__main__":
    s_vI = float(sys.argv[1]); seed = int(sys.argv[2]); MEAS = X.MEAS
    ev = json.load(open(os.environ.get("EVENTS", "e12_events.json")))["events"]
    L = X.load_labels(); sats = sorted({e["sat"] for e in ev})
    priors = {s: X.make_priors(X.fold_prior(L, s)) for s in sats}
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(1000 * s_vI)), MEAS == "tracklet"]))
    fn = f"e12_rows_{MEAS}_s{s_vI}_seed{seed}.json"; rows = []; t0c = time.time()
    for i, e in enumerate(ev):
        x0 = np.array(e["x0"]); yT = np.array(e["yT"])
        P0, en = prior_cov(x0, s_vI); Lp = np.linalg.cholesky(P0)
        Xref = x0 - Lp @ rng.standard_normal(6)                      # média do catálogo (erro de catálogo sintético)
        P0r, er = prior_cov(Xref, s_vI)
        trk = X.Track3(Xref, P0r, er)
        y0m = X.truth(x0, 0.0, None, rng)                             # propagação do MODELO a partir do estado real
        d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
        xN = x0 + rng.uniform(1.5, 3.0) * Lp @ d6
        yD = X.truth(xN, 0.0, None, rng); xN_hat = xN + Lp @ rng.standard_normal(6)
        P0n, enn = prior_cov(xN_hat, s_vI); trkN = X.Track3(xN_hat, P0n, enn)
        pr = priors[e["sat"]]
        z, R, m = X.make_uct(rng, yT); zD, RD, mD = X.make_uct(rng, yD)
        zF, RF, mF = X.make_uct(rng, y0m + X.field_offset(rng, y0m))
        c = "M" if e["man"] else "Z"
        row = dict(sat=e["sat"], man=e["man"], tau=e["tau"], s=e["s"], tm=e["tm"], t0=e["t0"],
                   model_err_pos=float(np.linalg.norm(y0m[:3] - yT[:3])), model_err_vel=float(np.linalg.norm(y0m[3:] - yT[3:])))
        row[c] = trk.scores(z, R, m, pr, refine=e["man"]); row[c + "n"] = trkN.scores(z, R, m, pr, refine=False)
        row["D"] = trk.scores(zD, RD, mD, pr, refine=False); row["Dn"] = trkN.scores(zD, RD, mD, pr, refine=False)
        row["F"] = trk.scores(zF, RF, mF, pr, refine=False)
        for k in (c, c + "n", "D", "Dn", "F"):
            row[k] = {a: (float("%.7g" % b) if isinstance(b, float) else b) for a, b in row[k].items()}
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(ev)} ({time.time()-t0c:.0f}s)", flush=True)
            json.dump(dict(meta=dict(complete=False, n=len(rows)), rows=rows), open(fn + ".partial", "w"))
    json.dump(dict(meta=dict(complete=True, n=len(rows), s_vI=s_vI, seed=seed, meas=MEAS, K1=X.K2), rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, time.time() - t0c)
