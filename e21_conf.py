"""Revisão (exploratório): confusor REAL para cada evento do E6c em MEO/QZSS = o satélite (GPS, GLONASS, Galileo,
BeiDou, QZSS) mais próximo, em posição, do alvo verdadeiro em T, segundo as órbitas precisas da GFZ (mesma fonte dos
eventos). Também grava os eventos elegíveis que o teto de 8 por satélite deixou de fora (e17extra_<g>.json).
Uso: python e21_conf.py meo|qzss   Saída: e21_conf_<g>.json"""
import sys, json, random
import numpy as np
from datetime import datetime, timedelta
import e15_src as S
g = sys.argv[1]; src = "gnss" if g == "meo" else "mgex"
DO_CONF = g != "leo"
cand = [f"G{i:02d}" for i in range(1, 33)] + [f"R{i:02d}" for i in range(1, 28)] + [f"E{i:02d}" for i in range(1, 37)] \
       + [f"C{i:02d}" for i in range(1, 64)] + [f"J{i:02d}" for i in range(1, 8)]
for p in cand: S.add_gnss(p, src) if DO_CONF else None
ev = json.load(open(f"e17sel_{g}.json"))["events"]
out = {}; files = None; cache = {}
for j, e in enumerate(ev if DO_CONF else []):
    if not (e["man"] and e["s"] >= 0.05):
        continue
    if files is None:
        files = S.listing(e["prn"]); print("files", len(files), flush=True)
    T = datetime.fromisoformat(e["t0"]) + timedelta(seconds=e["Tgap"]); yT = np.array(e["yT"]); best = []
    for p in cand:
        if p == e["prn"]:
            continue
        try:
            x = S.state(p, T, files, cache)
        except Exception:
            x = None
        if x is not None and np.isfinite(x).all():
            best.append((float(np.linalg.norm(x[:3] - yT[:3])), p, x))
    best.sort(key=lambda b: b[0])
    if best:
        d, p, x = best[0]
        out[str(j)] = dict(prn=p, dist_km=round(d / 1e3, 1), dv_kms=round(float(np.linalg.norm(x[3:] - yT[3:])) / 1e3, 3),
                           next_km=[round(b[0] / 1e3, 1) for b in best[1:4]], n_cand=len(best), y=[float(v) for v in x])
    print(j, e["prn"], out.get(str(j), {}).get("prn"), out.get(str(j), {}).get("dist_km"), flush=True)
    S.evict(cache, files, T - timedelta(days=3))
if DO_CONF: json.dump(out, open(f"e21_conf_{g}.json", "w"))
# eventos elegíveis fora do teto de 8 por satélite (mesma regra de e17_select.py)
fs = {"meo": ["e17_events_gnss.json"], "qzss": ["e17_events_qzss.json", "e17_events_qzss2.json"], "leo": ["e17_events_leo.json"]}[g]; allev = []
for f in fs:
    try:
        E = json.load(open(f))["events"]
    except FileNotFoundError:
        continue
    if f == "e17_events_qzss.json":
        E = [x for x in E if x["sat"] not in {49336, 62876}]
    allev += E
man = [x for x in allev if x["man"] and x["s"] >= 0.05]; kf = lambda x: (x["sat"], round(float(x["x0"][0]), 1))
selk = {kf(x) for x in ev}; extra = [x for x in man if kf(x) not in selk]
json.dump(dict(events=extra), open(f"e17extra_{g}.json", "w"))
print("conf", len(out), "extra", len(extra), "dist_km median", np.median([o["dist_km"] for o in out.values()]) if out else None)
