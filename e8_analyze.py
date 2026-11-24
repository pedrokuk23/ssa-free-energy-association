import json, sys, numpy as np

def auc_lowpos(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return 1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def tpr(pos, neg, f):
    return float(np.mean(np.asarray(pos) < np.quantile(neg, f)))

M = ["UKF", "GM", "CTRL", "GM+iso", "GM+learn"]
out = {}
for s in sys.argv[1:]:
    rows = json.load(open(f"e8_rows_s{s}.json"))["rows"]
    g = lambda key, m, sel=None: np.array([r[key][m] for r in rows if sel is None or sel(r)])
    big = lambda r: r["s"] >= 0.05          # manobras "relevantes" (≥ 5 cm/s)
    res = {}
    for nm, sel in [("todas", None), ("dv>=0.05", big), ("in-track", lambda r: r["tau"] == "in-track"),
                    ("cross-track", lambda r: r["tau"] == "cross-track")]:
        res[nm] = {
            "n": int(len(g("sM", "GM", sel))),
            "AUC_vizinho (manobrado)": {m: round(auc_lowpos(g("sM", m, sel), g("sN", m)), 3) for m in M},
            "TPR@1%_campo (manobrado)": {m: round(tpr(g("sM", m, sel), g("sF", m), 0.01), 3) for m in M},
            "TPR@5%_vizinho (manobrado)": {m: round(tpr(g("sM", m, sel), g("sN", m), 0.05), 3) for m in M},
        }
        det = {"GM: −log p0": ("GM", 1), "CTRL": ("CTRL", 1), "P(H1) iso": ("pH1_iso", 1), "P(H1) learn": ("pH1_learn", 1)}
        res[nm]["AUC_deteccao"] = {k: round(auc_lowpos(g("s0", m), g("sM", m, sel)), 3) for k, (m, _) in det.items()}
    res["sem manobra"] = {
        "AUC_vizinho": {m: round(auc_lowpos(g("s0", m), g("sN", m)), 3) for m in M},
        "TPR@1%_campo": {m: round(tpr(g("s0", m), g("sF", m), 0.01), 3) for m in M}}
    # caracterização (priori aprendida), manobras ≥ 0,05 m/s detectadas
    sel = [r for r in rows if r["s"] >= 0.05 and r["sM"]["pH1_learn"] > 0.5]
    if sel:
        acc = np.mean([(r["sM"]["p_cross"] > 0.5) == (r["tau"] == "cross-track") for r in sel])
        err = np.array([np.log10(r["sM"]["dv_hat"] / r["s"]) for r in sel])
        res["caracterizacao"] = dict(n=len(sel), acerto_tipo=round(float(acc), 3),
                                     erro_log10_dv_mediana=round(float(np.median(np.abs(err))), 3),
                                     fator_mediano=round(float(10 ** np.median(np.abs(err))), 2))
    # calibração de P(H1|z) numa população com a taxa real de manobra (p1), ponderando as classes
    p1 = json.load(open(f"e8_rows_s{s}.json"))["prior"]["p1"]
    cal = {}
    for k in ["pH1_iso", "pH1_learn"]:
        a0, a1 = np.clip(g("s0", k), 1e-9, 1 - 1e-9), np.clip(g("sM", k), 1e-9, 1 - 1e-9)
        brier = (1 - p1) * np.mean(a0**2) + p1 * np.mean((1 - a1)**2)
        ll = -((1 - p1) * np.mean(np.log(1 - a0)) + p1 * np.mean(np.log(a1)))
        cal[k] = dict(brier=round(float(brier), 4), logloss=round(float(ll), 4),
                      media_P_sem=round(float(a0.mean()), 3), media_P_com=round(float(a1.mean()), 3))
    cal["referencia_constante_p1"] = dict(brier=round(p1 * (1 - p1), 4),
                                          logloss=round(float(-(p1*np.log(p1)+(1-p1)*np.log(1-p1))), 4))
    res["calibracao"] = cal
    out[s] = res
    print(f"===== σ_vI = {s} m/s"); print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(out, open("e8_summary.json", "w"), indent=1, ensure_ascii=False)
