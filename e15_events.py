"""E6c — eventos do detector D nas varreduras: z normalizado pelo satélite (mediana de z_it e de z_ct sobre todas as
janelas do ano), limiar ZN=4 (fixado na validação DORIS, v2: recall 0.986, precisão 1.00, fator de magnitude 1.02).
Janelas consecutivas que acham a mesma manobra (|dt| < 12 h) são fundidas (maior z). Isolado: nenhum outro evento
em +-2.5 dias. Saída: e15_events.json + resumo."""
import json, glob, sys, numpy as np
from datetime import datetime
ZN = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"  # "it": só no plano (z_it normalizado; dv = dv_it)
OUTF = "e15_events.json" if MODE == "both" else f"e15_events_{MODE}.json"
out, rows = {}, []; nglitch = {}
for f in sorted(glob.glob("e15_scan_*_????-????.json")):
    code = f.split("_")[2]; W = json.load(open(f))
    if len(W) < 20:
        continue
    mi = np.median([w["det"]["z_it"] for w in W]); mc = np.median([w["det"]["z_ct"] for w in W])
    ev = []
    for w in W:
        d = dict(w["det"]); zn = max(d["z_it"] / mi, d["z_ct"] / mc)
        if MODE == "it":
            zn = d["z_it"] / mi; d["dv"] = d["dv_it"]; d["typ"] = "in-track"; d["ct_flag"] = bool(d["z_ct"] / mc > ZN)
        if zn > ZN and d["dv"] > 10.0:
            nglitch[code] = nglitch.get(code, 0) + 1; continue
        if zn > ZN:
            ev.append(dict(t=d["t_abs"], zn=float(zn), dv=d["dv"], typ=d["typ"], ct_flag=d.get("ct_flag", False)))
    ev.sort(key=lambda e: e["t"]); m = []
    for e in ev:
        if m and abs((datetime.fromisoformat(e["t"]) - datetime.fromisoformat(m[-1]["t"])).total_seconds()) < 12 * 3600:
            if e["zn"] > m[-1]["zn"]: m[-1] = e
        else:
            m.append(e)
    for e in m:
        e["iso"] = all(abs((datetime.fromisoformat(e["t"]) - datetime.fromisoformat(o["t"])).total_seconds()) > 2.5 * 86400 for o in m if o is not e)
    big = [e for e in m if e["iso"] and e["dv"] >= 0.05]
    out[code] = m
    rows.append((code, len(W), round(mi, 2), round(mc, 1), len(m), len(big), sorted(round(e["dv"], 3) for e in big)[-6:], {t: sum(e["typ"] == t for e in big) for t in ("in-track", "cross-track")}))
json.dump(out, open(OUTF, "w"), indent=0)
for r in rows: print(*r)
print("descartados dv>10 m/s (erro nos dados):", nglitch)
print("satélites com >=3 eventos isolados >=5 cm/s:", [r[0] for r in rows if r[5] >= 3])
