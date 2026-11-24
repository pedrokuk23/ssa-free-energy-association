"""Rótulos para as prioris GNSS (E6c): os rótulos do operador (NANU/NAGU/OHI) com Δv e tipo preenchidos pelos eventos
quando o operador não informa Δv (GPS/Galileo). Saída: data/labels_gnss.json (mesmo formato do maneuver-detect)."""
import json, glob, pandas as pd, numpy as np
from datetime import datetime, timedelta
L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json"))); L["t"] = pd.to_datetime(L.epoch, utc=True, format="ISO8601")
ev = []
for f in glob.glob("e17_events_*.json"):
    ev += [e for e in json.load(open(f))["events"] if e["man"]]
n = 0
for e in ev:
    tm = pd.Timestamp(e["t0"]) + pd.Timedelta(seconds=e["tm"])
    m = (L.norad_id == e["sat"]) & ((L.t - tm).abs() < pd.Timedelta(hours=1))
    if m.any():
        i = L[m].index[0]
        if pd.isna(L.at[i, "delta_v"]): L.at[i, "delta_v"] = e["s"]
        if pd.isna(L.at[i, "maneuver_type"]): L.at[i, "maneuver_type"] = e["tau"]
        n += 1
L = L.drop(columns=["t"]); json.dump(L.to_dict(orient="records"), open("data/labels_gnss.json", "w"), default=str)
print("eventos casados", n, "de", len(ev))
