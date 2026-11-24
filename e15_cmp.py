"""Validação do modo varredura em satélites DORIS rotulados: eventos (e15_events.json) x rótulos IDS (todos os tamanhos).
Casamento: |t_evento - t_rótulo| <= 6 h."""
import json, sys, numpy as np, pandas as pd
from datetime import datetime, timedelta
E = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "e15_events.json")); L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json")))
L["t"] = pd.to_datetime(L.epoch, utc=True, format="ISO8601")
for code in [c for c in E if c.isdigit()]:
    W = json.load(open([f for f in __import__("glob").glob(f"e15_scan_{code}_*.json") if "partial" not in f][0]))
    t0 = pd.Timestamp(W[0]["tm"]) - pd.Timedelta(days=1); t1 = pd.Timestamp(W[-1]["tm"]) + pd.Timedelta(days=1)
    lab = L[(L.norad_id == int(code)) & (L.t >= t0) & (L.t <= t1)]
    ev = E[code]; te = [pd.Timestamp(e["t"]) for e in ev]
    match = [any(abs((x - y).total_seconds()) <= 6 * 3600 for y in lab.t) for x in te]
    big = lab[lab.delta_v >= 0.05]
    rec = [any(abs((x - y).total_seconds()) <= 6 * 3600 for x in te) for y in big.t]
    fp = [e for e, m in zip(ev, match) if not m]
    print(code, f"janelas {len(W)} rótulos {len(lab)} (>=5cm/s {len(big)}) eventos {len(ev)} casados {sum(match)} recall>=5cm/s {sum(rec)}/{len(big)}",
          "FP:", len(fp), "FP>=5cm/s isolados:", sum(1 for e in fp if e["dv"] >= 0.05 and e["iso"]),
          "tipos FP", {t: sum(e["typ"] == t for e in fp) for t in ("in-track", "cross-track")},
          "dv FP", sorted(round(e["dv"], 3) for e in fp)[-6:], "zn FP", sorted(round(e["zn"], 1) for e in fp)[-6:])
