"""Agrupa E6 (seed 7) e E6b (seed 11) por MEAS (análise secundária). Saída: e12_rows_{MEAS}_s0.05_seedpool.json"""
import json
for m in ["full", "tracklet"]:
    R = []; meta = None
    for s in (7, 11):
        D = json.load(open(f"e12_rows_{m}_s0.05_seed{s}.json")); R += D["rows"]; meta = dict(D["meta"])
    meta.update(seed="pool", n=len(R)); json.dump(dict(meta=meta, rows=R), open(f"e12_rows_{m}_s0.05_seedpool.json", "w"))
    print(m, len(R))
