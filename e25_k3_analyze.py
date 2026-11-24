"""Revisão (exploratório) — E6c reavaliado (e17K3): (i) reprodução de M/Mn/F contra e17K2; (ii) associação conjunta
Prop vs IMMl e IMMd (IMM com a priori aprendida), total e por tipo de UCT; (iii) aceitação a 1% contra o confusor REAL
F3 (satélite mais próximo pelas órbitas precisas), por regime e tipo, e Prop-X; (iv) primário com todos os eventos
elegíveis (selecionados + os que o teto de 8 por satélite excluiu). Saída: e25_summary.json e c125.txt"""
import json, glob, numpy as np
from e12K_analyze import hksj, signflip, by_sat
G = ["meo", "qzss", "leo"]; MS = ["full", "tracklet"]
METH = ["UKF", "GM", "CTRL", "CTRLu", "iso_lin", "lin_learn", "learn", "IMM", "CTRLp", "IMMl", "IMMd"]

def load(pfx, tag, m):
    rows = []
    for f in sorted(glob.glob(f"{pfx}_{tag}_{m}_s0.05_seed17_c*of*.json")):
        D = json.load(open(f))
        if D["meta"].get("complete"):
            rows += [r for r in D["rows"] if r["draws"]]
    return rows

def corr(r, m):
    return np.mean([float(d["M"][m] < d["Mn"][m]) + 0.5 * float(d["M"][m] == d["Mn"][m]) for d in r["draws"]])

res = {}; L = []
P = lambda *a: L.append(" ".join(str(x) for x in a))
# (i) reprodução
rep = []
for g in G:
    for m in MS:
        A = {r["eid"]: r for r in load("e17K2", f"e17sel_{g}", m)}
        for r in load("e17K3", f"e17sel_{g}", m):
            if r["eid"] in A:
                for d3, d2 in zip(r["draws"], A[r["eid"]]["draws"]):
                    for c in ("M", "Mn", "F"):
                        for k in ("learn", "CTRLu", "CTRLp", "IMM"):
                            if k in d2[c]:
                                rep.append(abs(d3[c][k] - d2[c][k]) / max(abs(d2[c][k]), 1e-9))
res["repro_max_rel"] = float(max(rep)) if rep else None; res["repro_n"] = len(rep); P("repro", res["repro_max_rel"], len(rep))
# eventos (selecionados) e extras
E = {}
for src, tag in (("sel", "e17sel"), ("extra", "e17extra")):
    for g in G:
        for m in MS:
            for r in load("e17K3", f"{tag}_{g}", m):
                k = (src, g, r["eid"]); E.setdefault(k, dict(sat=r["sat"], g=g, src=src, meas={}, rows={}))
                E[k]["meas"][m] = {x: corr(r, x) for x in METH if x in r["draws"][0]["M"]}; E[k]["rows"][m] = r
keys = [k for k in E if len(E[k]["meas"]) == 2]

def test(a, b, sel, meas=None):
    ks = [k for k in keys if sel(E[k])]
    d = np.array([(E[k]["meas"][meas][a] - E[k]["meas"][meas][b]) if meas else
                  np.mean([E[k]["meas"][x][a] - E[k]["meas"][x][b] for x in MS]) for k in ks])
    sats = np.array([E[k]["sat"] for k in ks]); y, v, n, us = by_sat(d, sats); o = hksj(y, v)
    o["signflip_p"] = signflip(y, 1 / np.maximum(v, 1e-12)); o["n_ev"] = len(ks); return o

sel = lambda e: e["src"] == "sel"; allv = lambda e: True
res["rates_sel"] = {x: round(float(np.mean([np.mean([E[k]["meas"][m][x] for m in MS]) for k in keys if sel(E[k])])), 4) for x in METH}
P("rates_sel", res["rates_sel"])
res["joint"] = {}
for b in ["IMMl", "IMMd", "IMM", "CTRLp", "CTRLu"]:
    for mm in [None, "full", "tracklet"]:
        o = test("learn", b, sel, mm); res["joint"][f"learn-{b}:{mm or 'mean'}"] = o
        P("joint", b, mm or "mean", o["mu"], o["ci"], o["p_one"], "sf", o["signflip_p"], "k", o["k"])
# (iv) todos os elegíveis
for b in ["CTRLu", "CTRLp", "IMMl"]:
    o = test("learn", b, allv); res["joint"][f"ALL_learn-{b}"] = o; P("ALL eligible", b, o["mu"], o["ci"], o["p_one"], "k", o["k"], "n_ev", o["n_ev"])
    for g in G:
        o = test("learn", b, lambda e, g=g: e["g"] == g); P("  ALL", g, b, o["mu"], o["ci"], o["p_one"], "k", o["k"], "n_ev", o["n_ev"])
# (iii) aceitação contra o confusor real F3
res["acc_F3"] = {}
for g in ["meo", "qzss"]:
    for m in MS:
        rows = [E[k]["rows"][m] for k in keys if sel(E[k]) and E[k]["g"] == g and "F3" in E[k]["rows"][m]["draws"][0]]
        if not rows: continue
        acc = {}
        for x in METH:
            if x not in rows[0]["draws"][0]["F3"]: continue
            thr = np.quantile([d["F3"][x] for r in rows for d in r["draws"]], 0.01)
            thrF = np.quantile([d["F"][x] for r in rows for d in r["draws"]], 0.01)
            acc[x] = dict(F3=round(float(np.mean([d["M"][x] <= thr for r in rows for d in r["draws"]])), 3),
                          F=round(float(np.mean([d["M"][x] <= thrF for r in rows for d in r["draws"]])), 3),
                          sep=round(float(np.mean([d["M"][x] < d["F3"][x] for r in rows for d in r["draws"]])), 3))
        res["acc_F3"][f"{g}_{m}"] = acc; P("accF3", g, m, {x: (v["F3"], v["F"], v["sep"]) for x, v in acc.items()})
try:
    for g in ["meo", "qzss"]:
        C = json.load(open(f"e21_conf_{g}.json")); dk = [c["dist_km"] for c in C.values()]
        res[f"conf_{g}"] = dict(n=len(C), dist_km_median=float(np.median(dk)), dist_km_min=float(np.min(dk)), dist_km_max=float(np.max(dk)),
                                kinds={p: sum(1 for c in C.values() if c["prn"][0] == p) for p in "GREJC"})
        P("conf", g, res[f"conf_{g}"])
except FileNotFoundError:
    pass
json.dump(res, open("e25_summary.json", "w"), indent=1); open("c125.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
