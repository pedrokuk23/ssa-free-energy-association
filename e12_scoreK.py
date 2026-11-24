"""E6s (análise de sensibilidade pós hoc declarada, PREREG_E6s.md): K sorteios dos componentes sintéticos por evento real.
Eventos: manobras com |dv| >= 5 cm/s de E6a (e12_events_seed7.json) e E6b (e12_events_bq.json).
Por sorteio k: erro de catálogo, vizinho catalogado (1.5-3 sigma, track próprio), ruído/estação da UCT, objeto de campo.
Env: MEAS=full|tracklet, K1, USE_GPU, KDRAW (20), CHUNK=i/n. Uso: python e12_scoreK.py SIG_VI SEED"""
import os, sys, json, time
import numpy as np
import e11_numeric as X
from e6_nonlinear import prior_cov

KEEP = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "pH1_learn", "pH1_lin_learn", "pH1_iso_lin"]

def slim(d):
    return {k: float("%.7g" % d[k]) for k in KEEP if k in d}

if __name__ == "__main__":
    s_vI = float(sys.argv[1]); seed = int(sys.argv[2]); MEAS = X.MEAS; KD = int(os.environ.get("KDRAW", "20"))
    ci, cn = [int(x) for x in os.environ.get("CHUNK", "0/1").split("/")]
    ev = []
    for f, tag in (("e12_events_seed7.json", "a"), ("e12_events_bq.json", "b")):
        for j, e in enumerate(json.load(open(f))["events"]):
            if e["man"] and e["s"] >= 0.05:
                e = dict(e); e["study"] = tag; e["eid"] = f"{tag}{j}"; ev.append(e)
    ev = ev[ci::cn]
    L = X.load_labels(); sats = sorted({e["sat"] for e in ev})
    priors = {s: X.make_priors(X.fold_prior(L, s)) for s in sats}
    fn = f"e12K_{MEAS}_s{s_vI}_seed{seed}_c{ci}of{cn}.json"; rows = []; t0c = time.time()
    for i, e in enumerate(ev):
        x0 = np.array(e["x0"]); yT = np.array(e["yT"]); pr = priors[e["sat"]]
        P0, en = prior_cov(x0, s_vI); Lp = np.linalg.cholesky(P0)
        draws = []
        for k in range(KD):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(e["eid"][1:]), ord(e["study"]), k, MEAS == "tracklet"]))
            Xref = x0 - Lp @ rng.standard_normal(6)
            P0r, er = prior_cov(Xref, s_vI); trk = X.Track3(Xref, P0r, er)
            y0m = X.truth(x0, 0.0, None, rng)
            d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
            xN = x0 + rng.uniform(1.5, 3.0) * Lp @ d6
            yD = X.truth(xN, 0.0, None, rng); xN_hat = xN + Lp @ rng.standard_normal(6)
            P0n, enn = prior_cov(xN_hat, s_vI); trkN = X.Track3(xN_hat, P0n, enn)
            z, R, m = X.make_uct(rng, yT); zD, RD, mD = X.make_uct(rng, yD)
            zF, RF, mF = X.make_uct(rng, y0m + X.field_offset(rng, y0m))
            draws.append(dict(M=slim(trk.scores(z, R, m, pr, refine=False)), Mn=slim(trkN.scores(z, R, m, pr, refine=False)),
                              D=slim(trk.scores(zD, RD, mD, pr, refine=False)), Dn=slim(trkN.scores(zD, RD, mD, pr, refine=False)),
                              F=slim(trk.scores(zF, RF, mF, pr, refine=False))))
        rows.append(dict(eid=e["eid"], study=e["study"], sat=e["sat"], tau=e["tau"], s=e["s"], draws=draws))
        print(f"{i+1}/{len(ev)} ({time.time()-t0c:.0f}s)", flush=True)
        json.dump(dict(meta=dict(complete=False, n=len(rows)), rows=rows), open(fn + ".partial", "w"))
    json.dump(dict(meta=dict(complete=True, n=len(rows), K=KD, s_vI=s_vI, seed=seed, meas=MEAS, chunk=[ci, cn]), rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, time.time() - t0c)
