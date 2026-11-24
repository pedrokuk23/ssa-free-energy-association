"""Diagnóstico do E6: AUC conjunta (Prop, CTRLu, GM, UKF) por tipo, magnitude e erro de modelo; calibração em controles."""
import json, numpy as np
from e12_analyze import Arr
from e11_analyze import auc
out = {}
for meas in ["full", "tracklet"]:
    D = json.load(open(f"e12_rows_{meas}_s0.05_seedpool.json")); A = Arr(D["rows"]); n = len(A.r)
    big = A.man & (A.s >= 0.05); me = np.array([r["model_err_pos"] for r in A.r])
    def aj(m, sel):
        jm = A.jsc(A.obs[m], A.obsn[m], m)[sel]; jd = A.jsc(A.D[m], A.Dn[m], m)
        return round(float(auc(jm, jd)), 3), int(sel.sum())
    S = {"big": big, "cross": big & (A.tau == "cross-track"), "intrack": big & (A.tau == "in-track"),
         "s5-10": big & (A.s < 0.1), "s10-30": big & (A.s >= 0.1) & (A.s < 0.3), "s>30": big & (A.s >= 0.3),
         "lowME": big & (me < np.median(me[~A.man])), "highME": big & (me >= np.median(me[~A.man])),
         "ctrl": ~A.man, "ctrl_lowME": ~A.man & (me < np.median(me[~A.man])), "ctrl_highME": ~A.man & (me >= np.median(me[~A.man]))}
    o = {k: {m: aj(m, v) for m in ["learn", "CTRLu", "GM", "UKF"]} for k, v in S.items()}
    # calibração: P(mnv) média nos controles e nas manobras grandes
    o["pH1_learn"] = {"ctrl": round(float(np.mean(1 / (1 + np.exp(-A.lg["learn"][~A.man])))), 3),
                      "big": round(float(np.mean(1 / (1 + np.exp(-A.lg["learn"][big])))), 3)}
    # escala da evidência: mediana do score do objeto real (controle) vs vizinho no próprio track
    o["median_score_ctrl_vs_D_n"] = {m: [round(float(np.median(A.obs[m][~A.man])), 2), round(float(np.median(A.Dn[m])), 2),
                                         round(float(np.median(A.F[m])), 2)] for m in ["learn", "UKF"]}
    o["model_err_km_median_ctrl"] = round(float(np.median(me[~A.man])) / 1e3, 2)
    out[meas] = o
print(json.dumps(out))
