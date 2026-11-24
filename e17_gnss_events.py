"""E6c — eventos GNSS/QZSS (GPS NANU, Galileo NAGU, QZSS OHI) com órbitas finais multi-GNSS da GFZ (15 min).
Os centros removem o satélite no dia da manobra; o evento usa o estado x0 em t0 (antes da janela do rótulo) e a
observação em T = t0 + 2 dias (depois dela). Dv e tipo: elementos médios (1 período) da verdade em [T, T+P] menos
os da propagação do modelo (J2-J4 + Sol + Lua + SRP) a partir de x0 — o modelo remove a deriva natural.
Controles: janelas de 2 dias sem rótulo em +-3 dias, com dados em t0 e T. Saída: e17_events_gnss.json.
Uso: YEARS=2020,2026 python e17_gnss_events.py NCTRL SEED"""
import os, re, sys, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import e15_src as S, numprop as N
MU = 3.986004418e14; TG = 2 * 86400.0
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
QZ = {"QZS-2": "J02", "QZS-3": "J07", "QZS-4": "J03"}   # QZS-1R e QZS-6: PRN identificado pelos dias sem órbita
CRAM = {"G": 0.02, "E": 0.015, "J": 0.012}

def prn_of(r):
    m = re.search(r"PRN(\d+)", r.source_ref)
    if m: return "G%02d" % int(m[1])
    m = re.search(r"SVID(\d+)", r.source_ref)
    if m: return "E%02d" % int(m[1])
    m = re.search(r"(QZS-[0-9R]+)", r.source_ref)
    return QZ.get(m[1]) if m else None

def elems(X):
    r, v = X[:, :3], X[:, 3:]; rn = np.linalg.norm(r, axis=1); vn = np.linalg.norm(v, axis=1)
    a = 1 / (2 / rn - vn ** 2 / MU); h = np.cross(r, v); hn = np.linalg.norm(h, axis=1)
    return a, np.arccos(h[:, 2] / hn), np.unwrap(np.arctan2(h[:, 0], -h[:, 1])), vn

def series(prn, ta, tb, files, cache, step=900):
    T, X = [], []; t = ta
    while t <= tb:
        x = S.state(prn, t, files, cache)
        if x is not None: T.append((t - ta).total_seconds()); X.append(x)
        t += timedelta(seconds=step)
    return np.array(T), np.array(X)

def maneuver_est(prn, t0, x0, files, cache):
    a0 = 1 / (2 / np.linalg.norm(x0[:3]) - np.dot(x0[3:], x0[3:]) / MU); P = 2 * np.pi * np.sqrt(a0 ** 3 / MU)
    Tt, Xt = series(prn, t0 + timedelta(seconds=TG), t0 + timedelta(seconds=TG + P), files, cache)
    if len(Tt) < 0.8 * P / 900:
        return None
    N.set_forces(1, (t0 - J2000).total_seconds(), CRAM[prn[0]])
    Xm = N.integrate(np.repeat(x0[None], len(Tt), 0), 0.0, 0.0, TG + Tt)
    at, it, rt, vt = elems(Xt); am, im, rm, _ = elems(Xm)
    da = at.mean() - am.mean(); di = it.mean() - im.mean(); dr = np.angle(np.exp(1j * (rt.mean() - rm.mean())))
    v = vt.mean(); dv_it = v * abs(da) / (2 * a0); dv_ct = v * np.hypot(di, np.sin(it.mean()) * dr)
    return dict(dv_it=float(dv_it), dv_ct=float(dv_ct), da=float(da), period=float(P))

if __name__ == "__main__":
    NCTRL, seed = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.default_rng(seed)
    L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json"))); L["t"] = pd.to_datetime(L.epoch, utc=True, format="ISO8601")
    y0, y1 = [int(x) for x in os.environ.get("YEARS", "2020,2026").split(",")]
    T0MIN = datetime(2020, 9, 1, tzinfo=timezone.utc) if os.environ.get("GSRC", "gnss") == "gnss" else datetime(y0, 1, 1, tzinfo=timezone.utc)
    G = L[L.source.isin(os.environ.get("SOURCES", "GPS-NANU,GALILEO-NAGU").split(",")) & (L.t >= T0MIN)].copy()
    G["prn"] = [prn_of(r) for r in G.itertuples()]
    if os.environ.get("NORADS"):
        G = G[G.norad_id.isin([int(x) for x in os.environ["NORADS"].split(",")])]
    SRC = os.environ.get("GSRC", "gnss")
    for p in G.prn.dropna().unique(): S.add_gnss(p, SRC)
    for k in range(1, 8): S.add_gnss("J%02d" % k, SRC)
    files = S.listing("J01" if SRC == "mgex" else G.prn.dropna().iloc[0]); print("files", len(files), flush=True)
    byday = {f[2].date(): f for f in files}; cache = {}
    def has(prn, t): return S.state(prn, t, files, cache) is not None
    # PRN dos QZSS sem mapeamento fixo: o J-PRN ausente nos dias dos rótulos
    for nid in G[G.prn.isna() & (G.source == "QZSS-OHI")].norad_id.unique():
        days = G[G.norad_id == nid].t.dt.floor("D").unique()[:6]; cnt = {}
        for k in range(1, 8):
            p = "J%02d" % k
            cnt[p] = sum(1 for d in days if not has(p, d.to_pydatetime() + timedelta(hours=12))
                         and has(p, d.to_pydatetime() - timedelta(days=3)) and has(p, d.to_pydatetime() + timedelta(days=3)))
        best = max(cnt, key=cnt.get); print("QZSS", nid, "->", best, cnt, flush=True)
        G.loc[G.norad_id == nid, "prn"] = best
    events, log = [], {}
    for nid, g in G.groupby("norad_id"):
        prn_list = g.prn.dropna().unique()
        allm = g.t.sort_values().tolist(); nm = 0
        for r in g.itertuples():
            prn = r.prn
            if prn is None: continue
            ws = pd.Timestamp(r.window_start).to_pydatetime() if isinstance(r.window_start, str) else r.t.to_pydatetime()
            we = pd.Timestamp(r.window_end).to_pydatetime() if isinstance(r.window_end, str) else r.t.to_pydatetime()
            # último instante com órbita antes da janela (procura até 30 h antes)
            t0 = None; t = ws.replace(minute=15 * (ws.minute // 15), second=0, microsecond=0)
            for k in range(0, 121):
                tt = t - timedelta(minutes=15 * k)
                if has(prn, tt): t0 = tt; break
            if t0 is None: continue
            t0 = t0 - timedelta(minutes=15 * int(rng.integers(0, 24)))
            T = t0 + timedelta(seconds=TG)
            if T <= we + timedelta(hours=1) or not has(prn, T) or not has(prn, t0): continue
            if any(x != r.t and t0 - timedelta(days=1) <= x <= T + timedelta(days=1) for x in allm): continue
            x0 = S.state(prn, t0, files, cache); yT = S.state(prn, T, files, cache)
            me = maneuver_est(prn, t0, x0, files, cache)
            if me is None: continue
            dv = float(r.delta_v) if pd.notna(r.delta_v) else float(np.hypot(me["dv_it"], me["dv_ct"]))
            tau = "in-track" if me["dv_it"] >= me["dv_ct"] else "cross-track"
            events.append(dict(sat=int(nid), prn=prn, src=r.source, regime=r.orbit_class, man=True, tau=tau, s=dv,
                               s_label=(float(r.delta_v) if pd.notna(r.delta_v) else None), est=me,
                               t0=t0.isoformat(), tm=(r.t.to_pydatetime() - t0).total_seconds(), Tgap=TG,
                               epoch_j2000=(t0 - J2000).total_seconds(), x0=x0.tolist(), yT=yT.tolist()))
            nm += 1
        # controles
        prn = prn_list[0] if len(prn_list) else None; nc = 0; tries = 0
        while prn and nc < NCTRL and tries < 30 * NCTRL:
            tries += 1
            f = files[int(rng.integers(len(files)))]; t0 = f[2] + timedelta(minutes=15 * int(rng.integers(0, 96)))
            T = t0 + timedelta(seconds=TG)
            if any(t0 - timedelta(days=3) <= x <= T + timedelta(days=3) for x in allm): continue
            if not (has(prn, t0) and has(prn, T) and has(prn, t0 + timedelta(days=1))): continue
            x0 = S.state(prn, t0, files, cache); yT = S.state(prn, T, files, cache)
            me = maneuver_est(prn, t0, x0, files, cache)
            if me is None: continue
            events.append(dict(sat=int(nid), prn=prn, src=g.source.iloc[0], regime=g.orbit_class.iloc[0], man=False, tau="none", s=0.0,
                               est=me, t0=t0.isoformat(), tm=-1.0, Tgap=TG, epoch_j2000=(t0 - J2000).total_seconds(),
                               x0=x0.tolist(), yT=yT.tolist())); nc += 1
        log[int(nid)] = dict(prn=list(map(str, prn_list)), labels=len(g), maneuvers=nm, controls=nc)
        print(nid, log[int(nid)], flush=True)
        for k in list(cache):
            del cache[k]
        json.dump(dict(log=log, events=events), open(os.environ.get("OUT", "e17_events_gnss.json"), "w"))
    print("ok", len(events), sum(e["man"] for e in events))
