"""E6c (confirmatório) — K sorteios dos componentes sintéticos por evento real, para eventos GNSS/QZSS (intervalo de
2 dias, forças J2-J4 + Sol + Lua + SRP) e LEO novos. Mesmo desenho do E6s (e12_scoreK.py): por sorteio, erro de
catálogo, vizinho catalogado (1,5-3 sigma, trilha própria), ruído/estação da UCT, objeto de campo.
Env: EVENTS, MEAS=full|tracklet, T_DAYS, A_REF, LABELS, PRIOR_CLASS, KDRAW (20), CHUNK=i/n, K1, USE_GPU.
Uso: python e17_scoreK.py SIG_VI SEED
e17_scoreK3 (revisão, exploratório): os mesmos sorteios (M, Mn, F reproduzem e17K/e17K2), mais IMM com a priori
aprendida (IMMl, IMMd; EXTRA2=1) e um confusor REAL F3: o satélite GNSS/BeiDou mais próximo em T segundo as órbitas
precisas (CONF=e21_conf_<g>.json, de e21_conf.py), observado com o mesmo modelo de UCT (gerador próprio)."""
import os, sys, json, time
os.environ["EXTRA"] = "1"; os.environ["EXTRA2"] = "1"
import numpy as np
import e11_numeric as X
import numprop as NP
from e6_nonlinear import prior_cov

KEEP = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "pH1_learn", "pH1_lin_learn", "pH1_iso_lin", "IMM", "CTRLp", "IMMl", "IMMd"]
CRAM = {"G": 0.02, "E": 0.015, "J": 0.012}

def slim(d):
    return {k: float("%.7g" % d[k]) for k in KEEP if k in d}

if __name__ == "__main__":
    s_vI = float(sys.argv[1]); seed = int(sys.argv[2]); MEAS = X.MEAS; KD = int(os.environ.get("KDRAW", "20"))
    ci, cn = [int(x) for x in os.environ.get("CHUNK", "0/1").split("/")]
    EVF = os.environ["EVENTS"]; ev = []
    for j, e in enumerate(json.load(open(EVF))["events"]):
        if e["man"] and e["s"] >= 0.05:
            e = dict(e); e["eid"] = f"{j}"; ev.append(e)
    ev = ev[ci::cn]
    L = X.load_labels(); sats = sorted({e["sat"] for e in ev})
    priors = {s: X.make_priors(X.fold_prior(L, s)) for s in sats}
    tag = os.path.splitext(os.path.basename(EVF))[0]
    CONF = json.load(open(os.environ["CONF"])) if os.environ.get("CONF") else {}
    fn = f"e17K3_{tag}_{MEAS}_s{s_vI}_seed{seed}_c{ci}of{cn}.json"; rows = []; t0c = time.time()
    if os.path.exists(fn + ".partial"):  # retomada determinística (sementes por evento e sorteio)
        rows = json.load(open(fn + ".partial"))["rows"]; print("retomando", len(rows), flush=True)
    done = {r["eid"] for r in rows}
    for i, e in enumerate(ev):
        if e["eid"] in done:
            continue
        if e.get("regime", "LEO") != "LEO":
            NP.set_forces(1, e["epoch_j2000"], CRAM.get(e.get("prn", "G")[0], 0.02))
        else:
            NP.set_forces(0)
        x0 = np.array(e["x0"]); yT = np.array(e["yT"]); pr = priors[e["sat"]]
        P0, en = prior_cov(x0, s_vI); Lp = np.linalg.cholesky(P0)
        draws = []; failed = []
        for k in range(KD):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(e["eid"]), k, MEAS == "tracklet", 17]))
            Xref = x0 - Lp @ rng.standard_normal(6)
            P0r, er = prior_cov(Xref, s_vI); trk = X.Track3(Xref, P0r, er)
            y0m = X.truth(x0, 0.0, None, rng)
            d6 = rng.standard_normal(6); d6 /= np.linalg.norm(d6)
            xN = x0 + rng.uniform(1.5, 3.0) * Lp @ d6
            yD = X.truth(xN, 0.0, None, rng); xN_hat = xN + Lp @ rng.standard_normal(6)
            P0n, enn = prior_cov(xN_hat, s_vI); trkN = X.Track3(xN_hat, P0n, enn)
            z, R, m = X.make_uct(rng, yT); zD, RD, mD = X.make_uct(rng, yD)
            zF, RF, mF = X.make_uct(rng, y0m + X.field_offset(rng, y0m))
            d6f = rng.standard_normal(6); d6f /= np.linalg.norm(d6f)
            yF2 = X.truth(x0 + rng.uniform(3.0, 6.0) * Lp @ d6f, 0.0, None, rng)
            zF2, RF2, mF2 = X.make_uct(rng, yF2)
            try:
                dd = dict(M=slim(trk.scores(z, R, m, pr, refine=False)), Mn=slim(trkN.scores(z, R, m, pr, refine=False)),
                          F=slim(trk.scores(zF, RF, mF, pr, refine=False)))
                cf = CONF.get(e["eid"])
                if cf:
                    rng3 = np.random.default_rng(np.random.SeedSequence([seed, int(e["eid"]), k, MEAS == "tracklet", 99]))
                    z3, R3, m3 = X.make_uct(rng3, np.array(cf["y"])); dd["F3"] = slim(trk.scores(z3, R3, m3, pr, refine=False))
                draws.append(dd)
            except Exception as ex:  # emenda 3: sorteio descartado e registrado (não derruba o processo)
                failed.append(dict(k=k, err=repr(ex)[:120])); print("sorteio descartado", e["eid"], k, repr(ex)[:80], flush=True)
        rows.append(dict(eid=e["eid"], sat=e["sat"], prn=e.get("prn"), regime=e.get("regime", "LEO"), src=e.get("src"),
                         tau=e["tau"], s=e["s"], model_err_pos=float(np.linalg.norm(X.truth(x0, 0.0, None, np.random.default_rng(0))[:3] - yT[:3])),
                         draws=draws, failed=failed))
        print(f"{i+1}/{len(ev)} ({time.time()-t0c:.0f}s)", flush=True)
        json.dump(dict(meta=dict(complete=False, n=len(rows)), rows=rows), open(fn + ".partial", "w"))
    json.dump(dict(meta=dict(complete=True, n=len(rows), K=KD, s_vI=s_vI, seed=seed, meas=MEAS, chunk=[ci, cn], events=EVF), rows=rows), open(fn, "w"))
    if os.path.exists(fn + ".partial"):
        os.remove(fn + ".partial")
    print("ok", fn, time.time() - t0c)
