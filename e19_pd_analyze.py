"""Fase D (exploratório): sensibilidades do E5 (estudo B) — J=48, intervalo de 2 dias, escala da covariância x2,
priori com sinal in-track, dois atribuíveis simultâneos. Para cada execução: métricas por método e diferenças
Prop-CTRLu e Prop-Lin com bootstrap por satélite; contagem da regra 7σ. Saída: e19_pd_summary.json"""
import json, glob, numpy as np
import e11_analyze as A
EXTRA_M = ["learn_sgn"]
out = {}
for f in sorted(glob.glob("e11_rows_pd_*_seed31_*.json")):
    D = json.load(open(f))
    if not D["meta"]["complete"]:
        continue
    rows = D["rows"]; key = f.replace("e11_rows_pd_", "").replace(".json", "")
    has_sgn = "learn_sgn" in rows[0]["M"]
    A.METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn"] + (EXTRA_M if has_sgn else [])
    A.PROB = ["iso_lin", "lin_learn", "learn"] + (EXTRA_M if has_sgn else [])
    X = A.Arr(rows); idx = np.arange(len(rows))
    keys = ["acc_big", "acc_cross", "aucJ_big", "auc_D_big", "det_big"]
    o = {"n": len(rows), "n_big": int((X.s >= 0.05).sum()), "meta": D["meta"],
         "metrics": {m: {k: round(float(X.metric(m, k, idx)), 3) for k in keys + (["logloss"] if m in A.PROB else [])}
                     for m in A.METH}}
    o["diff"] = {}
    for k in ("acc_big", "aucJ_big"):
        for b in ("CTRLu", "lin_learn"):
            d = X.metric("learn", k, idx) - X.metric(b, k, idx)
            o["diff"][f"{k}:learn-{b}"] = [round(float(d), 3)] + A.cboot(X, A.dfn(X, "learn", b, k), B=1000)
        if has_sgn:
            d = X.metric("learn_sgn", k, idx) - X.metric("learn", k, idx)
            o["diff"][f"{k}:learn_sgn-learn"] = [round(float(d), 3)] + A.cboot(X, A.dfn(X, "learn_sgn", "learn", k), B=1000)
    c = D["meta"].get("cnt7", {})
    if c.get("active"):
        o["rule7"] = dict(frac_all=round(c["refused"] / c["active"], 4),
                          frac_relevant=round(c["refused_rel"] / max(c["active_rel"], 1), 4), **c)
    out[key] = o; print(key, o["metrics"]["learn"], o["diff"], o.get("rule7"), flush=True)
json.dump(out, open("e19_pd_summary.json", "w"), indent=1)
