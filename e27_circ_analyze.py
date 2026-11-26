"""[Auditoria 2026-11-26: cópia de e17K_analyze.py aplicada às linhas e17K3 com a correção de ângulos (run22_circ.sh);
análise de sensibilidade declarada, não confirmatória; inclui CTRLp, IMM, IMMl, IMMd.]
Análise confirmatória do E6c (PREREG_E6c.md, emenda 1). Primário: por evento, taxa de associação conjunta correta
(F_t(z) < F_n(z)) sobre K sorteios, média das UCTs de estado completo e tracklet; diferença Prop('learn') - CTRLu;
HKSJ no nível de satélite (variância dentro do satélite agrupada) + teste exato de troca de sinal; unilateral 0,05.
Saída: e17K_summary.json"""
import json, glob, numpy as np
import e12K_analyze as _K
_K.METH[:] = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "IMM", "CTRLp", "IMMl", "IMMd"]
from e12K_analyze import hksj, signflip, by_sat, METH

def per_event(rows, thr):  # como em e12K_analyze, sem a UCT do vizinho (ausente nas linhas e17K3)
    out = {m: dict(corr=[], acc=[]) for m in METH}
    for r in rows:
        for m in METH:
            Ft = np.array([d["M"][m] for d in r["draws"]]); Fn = np.array([d["Mn"][m] for d in r["draws"]])
            out[m]["corr"].append(np.mean(Ft < Fn) + 0.5 * np.mean(Ft == Fn)); out[m]["acc"].append(np.mean(Ft <= thr[m]))
    return {m: {k: np.array(v) for k, v in o.items()} for m, o in out.items()}
GROUPS = ["meo", "qzss", "leo"]

def load(g, meas):
    rows = []
    for f in sorted(glob.glob(f"e17K3_e17sel_{g}_{meas}_s0.05_seed17_c*of*.json")):
        D = json.load(open(f))
        if not D["meta"]["complete"]:
            raise SystemExit(f"incompleto: {f}")
        rows += [r for r in D["rows"] if len(r["draws"]) > 0]
    return rows

E, info = {}, {}
for g in GROUPS:
    for meas in ["full", "tracklet"]:
        rows = load(g, meas)
        if not rows:
            continue
        thr = {m: np.quantile([d["F"][m] for r in rows for d in r["draws"]], 0.01) for m in METH}
        pe = per_event(rows, thr)
        for i, r in enumerate(rows):
            k = (g, r["eid"]); E.setdefault(k, dict(sat=r["sat"], s=r["s"], g=g, meas={}))
            E[k]["meas"][meas] = {m: {q: float(pe[m][q][i]) for q in ("corr", "acc")} for m in METH}
keys = [k for k in E if len(E[k]["meas"]) == 2]
sats = np.array([E[k]["sat"] for k in keys]); grp = np.array([E[k]["g"] for k in keys]); s = np.array([E[k]["s"] for k in keys])

def val(m, q, meas=None):
    if meas: return np.array([E[k]["meas"][meas][m][q] for k in keys])
    return np.array([np.mean([E[k]["meas"][x][m][q] for x in ("full", "tracklet")]) for k in keys])

def test(a, b, q="corr", sel=None, meas=None):
    d = val(a, q, meas) - val(b, q, meas); sl = np.ones(len(d), bool) if sel is None else sel
    y, v, n, us = by_sat(d[sl], sats[sl]); o = hksj(y, v) if len(y) > 2 else dict(k=len(y))
    o["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-12)) if len(y) > 1 else None
    o["mean_event"] = round(float(np.mean(d[sl])), 4); o["n"] = int(sl.sum())
    from scipy import stats
    if len(y) > 2:  # limite inferior unilateral 95 % (critério de sucesso)
        o["lower_one_sided_95"] = round(float(o["mu"] - stats.t.ppf(0.95, o["k"] - 1) * (o["ci"][1] - o["mu"]) / stats.t.ppf(0.975, o["k"] - 1)), 4)
    o["forest"] = {str(u): [round(float(yy), 4), int(nn)] for u, yy, nn in zip(us, y, n)}
    return o

res = {"n_events": len(keys), "n_sat": len(set(sats)), "by_group": {g: dict(events=int((grp == g).sum()), sats=len(set(sats[grp == g]))) for g in GROUPS},
       "rates": {m: round(float(np.mean(val(m, "corr"))), 4) for m in METH}}
res["primary"] = test("learn", "CTRLu")
res["primary"]["success"] = bool(res["primary"].get("lower_one_sided_95", -1) > 0)
res["secondary"] = {"by_regime": {g: test("learn", "CTRLu", sel=grp == g) for g in GROUPS if (grp == g).sum() > 0},
                    "by_meas": {mm: test("learn", "CTRLu", meas=mm) for mm in ("full", "tracklet")},
                    "vs": {b: test("learn", b) for b in ["GM", "UKF", "lin_learn", "CTRL", "iso_lin", "CTRLp", "IMM", "IMMl", "IMMd"]},
                    "vs_meas": {mm: {b: test("learn", b, meas=mm) for b in ["GM", "UKF", "lin_learn", "CTRLp", "IMM", "IMMl", "IMMd", "iso_lin"]} for mm in ("full", "tracklet")},
                    "ge30": test("learn", "CTRLu", sel=s >= 0.30),
                    "acc_1pct": {b: test("learn", b, q="acc") for b in ["CTRLu", "GM", "UKF"]}}
for d in [res["secondary"]["by_regime"], res["secondary"]["by_meas"], res["secondary"]["vs"], res["secondary"]["acc_1pct"], *res["secondary"]["vs_meas"].values()]:
    for o in d.values(): o.pop("forest", None)
res["secondary"]["ge30"].pop("forest", None)
json.dump(res, open("e27_circ_summary.json", "w"), indent=1)
p = res["primary"]; print("E6c primário:", {k: p.get(k) for k in ("k", "mu", "ci", "lower_one_sided_95", "p_one", "signflip_p", "tau", "success")})
print("por regime:", {g: (o.get("mu"), o.get("ci"), o.get("k")) for g, o in res["secondary"]["by_regime"].items()})
