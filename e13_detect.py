"""Detector D de manobras a partir de órbitas precisas (E6c, PREREG_E6c.md) e sua validação nos satélites DORIS
com rótulos de operador (maneuver-detect v0.3).
Método: elementos osculadores (a, i, RAAN) a cada 120 s -> média móvel de um período orbital -> para cada época
candidata t_c (passo 10 min), ajuste por mínimos quadrados de tendência quadrática comum + degrau em t_c, excluindo
|t - t_c| < P (o degrau fica espalhado pela média móvel). dv no plano = v|da|/(2a); fora do plano
= v*sqrt(di^2 + (sin i dRAAN)^2). Estatística z = max(|da|/se, |dcross|/se_cross); época = argmax.
Uso:  ACS=ssa,gsc,grg,lca,ign python e13_detect.py validate NCTRL SEED [sats] > out.json
      (por janela de +-2 dias em torno de cada manobra rotulada >= 5 cm/s e em janelas de controle)"""
import os, sys, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import e12_fetch as F

MU = 3.986004418e14

def series(sat, t0, t1, files, cache, step=120):
    T, X = [], []
    t = t0
    while t <= t1:
        x = F.state(sat, t, files, cache)
        if x is not None:
            T.append((t - t0).total_seconds()); X.append(x)
        t += timedelta(seconds=step)
    return np.array(T), np.array(X)

def elements(X):
    r, v = X[:, :3], X[:, 3:]; rn = np.linalg.norm(r, axis=1); vn = np.linalg.norm(v, axis=1)
    a = 1.0 / (2.0 / rn - vn ** 2 / MU)
    h = np.cross(r, v); hn = np.linalg.norm(h, axis=1)
    inc = np.arccos(np.clip(h[:, 2] / hn, -1, 1)); raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
    return a, inc, raan, vn

def movmean(t, y, P):
    out = np.full_like(y, np.nan); c = np.r_[0, np.cumsum(y)]
    lo = np.searchsorted(t, t - P / 2); hi = np.searchsorted(t, t + P / 2, side="right")
    ok = (t - P / 2 >= t[0]) & (t + P / 2 <= t[-1]) & (hi - lo >= 0.8 * P / max(np.median(np.diff(t)), 1))
    out[ok] = (c[hi] - c[lo])[ok] / (hi - lo)[ok]
    return out

def step_fit(t, y, tc, P, kink=False):
    m = np.isfinite(y) & (np.abs(t - tc) >= P)
    if m.sum() < 30 or (t[m] < tc).sum() < 10 or (t[m] > tc).sum() < 10:
        return np.nan, np.nan
    tt = (t[m] - tc) / 86400.0
    A = np.c_[np.ones_like(tt), tt, tt ** 2, (tt > 0).astype(float)]
    if kink:  # mudanca de taxa (ex.: deriva do RAAN apos mudanca de a)
        A = np.c_[A, np.where(tt > 0, tt, 0.0)]
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None); res = y[m] - A @ coef
    s2 = res @ res / max(len(res) - 4, 1); cov = s2 * np.linalg.inv(A.T @ A)
    return coef[3], np.sqrt(cov[3, 3])

TAG = os.environ.get("TAG", "")
KINK = os.environ.get("KINK", "1") == "1"; ZSIG = float(os.environ.get("ZSIG", "5"))

def detect(T, X, tmin, tmax, dt=600):
    a, inc, raan, vn = elements(X); P = 2 * np.pi * np.sqrt(np.median(a) ** 3 / MU)
    am, im, rm = movmean(T, a, P), movmean(T, inc, P), movmean(T, raan, P)
    v = np.median(vn); A0 = np.median(a); si = np.sin(np.median(inc))
    best = None
    for tc in np.arange(tmin, tmax + 1, dt):
        da, sa = step_fit(T, am, tc, P); di, sdi = step_fit(T, im, tc, P, KINK); dr, sdr = step_fit(T, rm, tc, P, KINK)
        if not np.isfinite(da + di + dr):
            continue
        dv_it = v * abs(da) / (2 * A0); dv_ct = v * np.hypot(di, si * dr)
        z_it = abs(da) / sa; z_ct = np.hypot(di / sdi, dr / sdr) / np.sqrt(2)
        z = max(z_it, z_ct)
        dv_it_s = dv_it if z_it > ZSIG else 0.0; dv_ct_s = dv_ct if z_ct > ZSIG else 0.0
        if best is None or z > best["z"]:
            best = dict(z=float(z), t=float(tc), dv=float(np.hypot(dv_it_s, dv_ct_s)), dv_raw=float(np.hypot(dv_it, dv_ct)), dv_it=float(dv_it), dv_ct=float(dv_ct),
                        typ="in-track" if dv_it_s >= dv_ct_s else "cross-track", z_it=float(z_it), z_ct=float(z_ct))
    return best

def window(sat, tm, files, cache, half=2.0):
    t0 = (tm - timedelta(days=half)).replace(second=0, microsecond=0); t0 = t0.replace(minute=2 * (t0.minute // 2))
    T, X = series(sat, t0, tm + timedelta(days=half), files, cache)
    if len(T) < 0.8 * (2 * half * 86400 / 120):
        return None
    return detect(T, X, (half - 1.0) * 86400, (half + 1.0) * 86400), t0

if __name__ == "__main__":
    mode = sys.argv[1]; nctrl = int(sys.argv[2]); seed = int(sys.argv[3])
    only = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    L = pd.DataFrame(json.load(open("data/maneuver_labels_v0.3.json")))
    L = L[(L.orbit_class == "LEO") & L.norad_id.isin(F.CODE)].copy(); L["t"] = pd.to_datetime(L.epoch, utc=True, format="ISO8601")
    out = []; rng = np.random.default_rng(seed)
    for sat in (only or sorted(F.CODE)):
        files = F.listing(F.CODE[sat]); cache = {}
        if not files:
            continue
        allm = L[L.norad_id == sat].t
        big = L[(L.norad_id == sat) & L.delta_v.notna() & (L.delta_v >= 0.05)]
        for _, g in big.iterrows():
            tm = g.t.to_pydatetime()
            near = [x for x in allm if x != g.t and abs((x - g.t).total_seconds()) < 2.5 * 86400]
            r = window(sat, tm, files, cache)
            if r is None or r[0] is None:
                continue
            d, t0 = r
            out.append(dict(sat=sat, kind="man", tm=tm.isoformat(), dv=float(g.delta_v), typ=g.maneuver_type, n_near=len(near),
                            det=d, dt_hours=(t0 + timedelta(seconds=d["t"]) - tm).total_seconds() / 3600))
            print(sat, "man", len(out), flush=True)
        # controles: janelas sem manobra rotulada em +-3 dias, dentro da cobertura
        b0, b1 = files[0][2], files[-1][3]; nc = 0; tries = 0
        while nc < nctrl and tries < 50 * nctrl:
            tries += 1
            tm = b0 + timedelta(seconds=float(rng.uniform(3 * 86400, (b1 - b0).total_seconds() - 3 * 86400)))
            tm = tm.replace(second=0, microsecond=0)
            if any(abs((x.to_pydatetime() - tm).total_seconds()) < 3 * 86400 for x in allm):
                continue
            r = window(sat, tm, files, cache)
            if r is None or r[0] is None:
                continue
            nc += 1; out.append(dict(sat=sat, kind="ctrl", tm=tm.isoformat(), det=r[0]))
        print(sat, "done", flush=True)
        json.dump(out, open(f"e13_val_{seed}{TAG}.partial.json", "w"))
    json.dump(out, open(f"e13_val_{seed}{TAG}.json", "w")); print("ok", len(out))
