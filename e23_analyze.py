"""Revisão (exploratório): métricas do estudo B para arquivos e11_rows_* (padrão em argv[1]), com IMM, IMMl, IMMd,
CTRLp quando presentes; diferenças Prop-X com bootstrap por satélite; NEES do catálogo por OD quando houver.
Uso: python e23_analyze.py 'e11_rows_od*_seed41_*.json' saida.json"""
import sys, json, glob, numpy as np
import e11_analyze as A
out = {}
for f in sorted(glob.glob(sys.argv[1])):
    D = json.load(open(f))
    if not D["meta"].get("complete"):
        continue
    rows = D["rows"]; ex = [m for m in ["CTRLp", "IMM", "IMMl", "IMMd"] if m in rows[0]["M"]]
    A.METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn"] + ex; A.PROB = ["iso_lin", "lin_learn", "learn"]
    X = A.Arr(rows); idx = np.arange(len(rows)); keys = ["acc_big", "acc_cross", "aucJ_big", "auc_D_big", "det_big"]
    o = {"n": len(rows), "meta": D["meta"], "metrics": {m: {k: round(float(X.metric(m, k, idx)), 3) for k in keys + (["logloss"] if m in A.PROB else [])}
                                                        for m in A.METH}, "diff": {}}
    for k in ("acc_big", "aucJ_big"):
        for b in ["CTRLu", "lin_learn"] + ex:
            d = X.metric("learn", k, idx) - X.metric(b, k, idx)
            o["diff"][f"{k}:learn-{b}"] = [round(float(d), 3)] + A.cboot(X, A.dfn(X, "learn", b, k), B=1000)
    if "nees" in rows[0]:
        nz = np.array([r["nees"] for r in rows]); o["od"] = dict(nees_median=round(float(np.median(nz)), 1), nees_p90=round(float(np.quantile(nz, 0.9)), 1),
                                                                 frac_nees_gt_chi2_99=round(float(np.mean(nz > 16.81)), 3),
                                                                 sig_pos_median_m=round(float(np.median([r["od_sig_pos"] for r in rows])), 1),
                                                                 err_pos_median_m=round(float(np.median([r["od_err_pos"] for r in rows])), 1))
    c = D["meta"].get("cnt7", {})
    if c.get("active"):
        o["rule7"] = dict(frac_all=round(c["refused"] / c["active"], 4), frac_relevant=round(c["refused_rel"] / max(c["active_rel"], 1), 4))
    key = f.replace("e11_rows_", "").replace(".json", ""); out[key] = o
    print(key, {m: o["metrics"][m] for m in ["learn", "CTRLu"] + ex}, o["diff"], o.get("od"), o.get("rule7"), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
