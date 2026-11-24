"""Parser dos arquivos de manobra do IDS (man.readme). Uso: python ids_parse.py arquivos... -> resumo JSON e ids_labels.json"""
import sys, json, numpy as np
from datetime import datetime, timedelta
def dt(y, d, h, m, s=0.0):
    return datetime(int(y), 1, 1) + timedelta(days=int(d) - 1, hours=int(h), minutes=int(m), seconds=float(s))
out = {}; labels = []
for fn in sys.argv[1:]:
    st = dict(n=0, big=0, types={}, ptypes={}, first=None, last=None, bad=0)
    for ln in open(fn, encoding="ascii", errors="ignore"):
        ln = ln.rstrip("\r\n")
        if len(ln) < 45 or not ln[6:10].strip().isdigit():
            continue
        try:
            sat = ln[0:5].strip(); t0 = dt(ln[6:10], ln[11:14], ln[15:17], ln[18:20])
            ptype = ln[40:43].strip(); nb = int(ln[44:46].strip() or ln[44])
            dv = np.zeros(3); tm = []
            for i in range(nb):
                k = i * 232
                tm.append(dt(ln[46+k:50+k], ln[51+k:54+k], ln[55+k:57+k], ln[58+k:60+k], ln[61+k:67+k]))
                dv += [float(ln[89+k:109+k]), float(ln[110+k:130+k]), float(ln[131+k:151+k])]
        except Exception:
            st["bad"] += 1; continue
        if not tm:
            st["bad"] += 1; continue
        if ptype == "005":    # SPOT: T,R,L
            rtn = np.array([dv[1], dv[0], dv[2]])
        elif ptype == "007":  # Jason: Q,S,W  (radial, along, cross)
            rtn = dv.copy()
        else:                 # 006 e outros: radial, along, cross
            rtn = dv.copy()
        mag = float(np.linalg.norm(rtn)); typ = ["radial", "in-track", "cross-track"][int(np.argmax(np.abs(rtn)))]
        st["n"] += 1; st["big"] += mag >= 0.05; st["types"][typ] = st["types"].get(typ, 0) + 1
        st["ptypes"][ptype] = st["ptypes"].get(ptype, 0) + 1
        st["first"] = st["first"] or t0.date().isoformat(); st["last"] = t0.date().isoformat()
        labels.append(dict(file=fn.split("/")[-1], sat=sat, epoch=tm[len(tm) // 2].isoformat(), t_begin=t0.isoformat(), nburn=nb,
                           ptype=ptype, dv_rtn=[round(x, 6) for x in rtn], delta_v=round(mag, 6), maneuver_type=typ))
    st["big_types"] = {}
    for l in labels:
        if l["file"] == fn.split("/")[-1] and l["delta_v"] >= 0.05:
            st["big_types"][l["maneuver_type"]] = st["big_types"].get(l["maneuver_type"], 0) + 1
    out[fn.split("/")[-1]] = st
json.dump(labels, open("ids_labels.json", "w"))
print(json.dumps(out))
