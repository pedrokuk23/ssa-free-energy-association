"""[Auditoria: versão com a correção de ângulos, sobre e27_circ_analyze.py]
Tabela descritiva do E6c: taxas por método x UCT x regime (corr e acc) e floresta por satélite com regime."""
import json, numpy as np
import e27_circ_analyze as A
out = {"rates": {}, "forest": {}}
for g in ["all"] + A.GROUPS:
    sl = np.ones(len(A.keys), bool) if g == "all" else A.grp == g
    for mm in ("full", "tracklet"):
        for q in ("corr", "acc"):
            out["rates"][f"{g}|{mm}|{q}"] = {m: round(float(np.mean(A.val(m, q, mm)[sl])), 3) for m in A.METH}
d = A.val("learn", "corr") - A.val("CTRLu", "corr")
for u in sorted(set(A.sats)):
    s = A.sats == u
    out["forest"][str(u)] = [A.grp[s][0], round(float(d[s].mean()), 4), int(s.sum())]
json.dump(out, open("e27_circ_table.json", "w"), indent=0)
print("TABJSON" + json.dumps(out, separators=(",", ":")))
