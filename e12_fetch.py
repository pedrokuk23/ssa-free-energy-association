"""E6 (dados reais) — passo 1: eventos a partir de órbitas precisas DORIS (IGN, FTP anônimo) e rótulos de manobra.
Para cada manobra rotulada (in-track/cross-track/radial, |Δv|>0.1 mm/s) escolhe t0 ∈ [t_m − 22 h, t_m − 2 h] com
T = t0 + 1 dia dentro da cobertura SP3 e sem outra manobra em [t0 − 1 h, T + 1 h]; controles: janelas de 1 dia sem
manobra em ±6 h. Estados ITRF → pseudo-inercial por rotação de GMST (constantes de tempo comuns às duas épocas não
afetam a dinâmica zonal axissimétrica). Saída: e12_events.json.
Uso: python e12_fetch.py NMAX_POR_SAT NCTRL_POR_SAT SEED
"""
import os, re, sys, json, gzip, subprocess, io, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone

BASE = "ftp://doris.ign.fr/pub/doris/products/orbits"
ACS = os.environ.get("ACS", "ssa,gsc,grg").split(",")
CODE = {90001: "swo", 90002: "s3c", 90003: "s6b", 20436: "sp2", 25260: "sp4", 26997: "ja1", 27386: "en1", 27421: "sp5", 33105: "ja2", 36508: "cs2",
        37781: "h2a", 39086: "srl", 41240: "ja3", 41335: "s3a", 43437: "s3b", 46984: "s6a"}
CACHE = "doris_cache"; os.makedirs(CACHE, exist_ok=True)
OMEGA = 7.292115e-5

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600).stdout

def yd2dt(yyddd):
    y = int(yyddd[:2]); y += 1900 if y > 80 else 2000
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(yyddd[2:]) - 1)

def listing(code):
    """[(ac, fname, begin, end)] de TODAS as ACs (ordem de prioridade ACS); maior versão por (ac, b, e)."""
    out = []
    for ac in ACS:
        txt = ""
        for k in range(5):
            txt = sh(f"curl -s -m 60 --list-only {BASE}/{ac}/{code}/")
            if txt.strip() or k == 4:
                break
            time.sleep(10 * (k + 1))
        best = {}
        for f in txt.split():
            m = re.search(r"\.b(\d{5})\.e(\d{5})\..*sp3\.(\d{3})\.Z$", f)
            if m:
                k = (m.group(1), m.group(2))
                if k not in best or f > best[k]:
                    best[k] = f
        out += [(ac, f, yd2dt(b), yd2dt(e) + timedelta(days=1)) for (b, e), f in sorted(best.items())]
    return out

def load_sp3(ac, code, fname):
    loc = os.path.join(CACHE, fname)
    if not os.path.exists(loc):
        sh(f"curl -s -m 300 -o {loc} {BASE}/{ac}/{code}/{fname}")
    raw = subprocess.run(f"gzip -dc {loc}", shell=True, capture_output=True).stdout.decode("ascii", "ignore")
    ep, P, V = [], {}, {}
    t = None
    for ln in raw.splitlines():
        if ln.startswith("* "):
            a = ln[1:].split(); t = datetime(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]),
                                             int(float(a[5])), tzinfo=timezone.utc); ep.append(t)
        elif ln.startswith("P") and t is not None:
            P[t] = np.array([float(ln[4:18]), float(ln[18:32]), float(ln[32:46])]) * 1e3
        elif ln.startswith("V") and t is not None:
            V[t] = np.array([float(ln[4:18]), float(ln[18:32]), float(ln[32:46])]) * 0.1
    # velocidades por derivada de Lagrange (9 pontos) das posições; substituem as do arquivo se ausentes ou
    # inconsistentes (mediana |V - dP/dt| > 1 m/s, p. ex. unidades fora do padrão em séries antigas)
    Vd = {}
    if len(P) > 9:
        ts = sorted(P); tt = np.array([(x - ts[0]).total_seconds() for x in ts]); XX = np.array([P[x] for x in ts])
        for i in range(4, len(ts) - 4):
            w = slice(i - 4, i + 5); dt = tt[w] - tt[i]
            if np.ptp(dt) > 8 * 300:
                continue
            Vd[ts[i]] = np.array([np.polyder(np.polyfit(dt / 60.0, XX[w, j], 8))[-1] / 60.0 for j in range(3)])
    common = [x for x in Vd if x in V]
    if len(V) < len(P) // 2 or (common and np.median([np.linalg.norm(V[x] - Vd[x]) for x in common]) > 1.0):
        V = Vd
    return P, V

def gmst(t):
    jd = (t - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    return np.mod(np.deg2rad(280.46061837 + 360.98564736629 * jd), 2 * np.pi)

def to_inertial(t, r, v):
    th = gmst(t); c, s = np.cos(th), np.sin(th); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    w = np.array([0, 0, OMEGA])
    return np.r_[R @ r, R @ (v + np.cross(w, r))]

def state(sat, t, files, cache):
    for ac, f, b, e in files:
        if b <= t < e:
            key = (sat, f)
            if key not in cache:
                cache[key] = load_sp3(ac, CODE[sat], f)
            P, V = cache[key]
            if t in P and t in V:
                return to_inertial(t, P[t], V[t])
    return None

if __name__ == "__main__":
    NMAX, NCTRL, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ONLY = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else sorted(CODE)
    OUT = sys.argv[5] if len(sys.argv) > 5 else "e12_events.json"
    rng = np.random.default_rng([seed] + ONLY)
    L = pd.DataFrame(json.load(open(os.environ.get("LABELS", "data/maneuver_labels_v0.3.json"))))
    L = L[(L.orbit_class == "LEO") & L.norad_id.isin(CODE)].copy()
    L["t"] = pd.to_datetime(L.epoch, utc=True, format="ISO8601")
    lab = L[L.delta_v.notna() & (L.delta_v > float(os.environ.get("MINDV", "1e-4"))) & L.maneuver_type.isin(["in-track", "cross-track", "radial"])]
    EXCL = set()
    if os.environ.get("EXCL"):
        for e in json.load(open(os.environ["EXCL"]))["events"]:
            if e["man"]:
                EXCL.add((e["sat"], round(datetime.fromisoformat(e["t0"]).timestamp() + e["tm"])))
    events = []; cache = {}; log = {}
    for sat in ONLY:
        files = listing(CODE[sat])
        if not files:
            log[sat] = "no orbits"; print(sat, "no orbits", flush=True); continue
        allm = L[L.norad_id == sat].t.sort_values().tolist()
        cand = lab[lab.norad_id == sat].sample(frac=1, random_state=int(rng.integers(1e9)))
        nm = 0; tried = 0
        for _, g in cand.iterrows():
            if nm >= NMAX or tried > 4 * NMAX:
                break
            tried += 1
            tm = g.t.to_pydatetime()
            if (int(sat), round(tm.timestamp())) in EXCL:
                continue
            for _ in range(int(os.environ.get("NDRAW", "4"))):
                t0 = tm - timedelta(seconds=float(rng.uniform(2, 22)) * 3600)
                t0 = t0.replace(minute=5 * (t0.minute // 5), second=0, microsecond=0)
                T = t0 + timedelta(days=1)
                if any((x.to_pydatetime() != tm) and (t0 - timedelta(hours=1) <= x.to_pydatetime() <= T + timedelta(hours=1)) for x in allm):
                    if os.environ.get("NDRAW"):
                        continue
                    break
                x0 = state(sat, t0, files, cache); yT = state(sat, T, files, cache)
                if x0 is not None and yT is not None:
                    events.append(dict(sat=int(sat), man=True, tau=g.maneuver_type, s=float(g.delta_v),
                                       t0=t0.isoformat(), tm=(tm - t0).total_seconds(), x0=x0.tolist(), yT=yT.tolist()))
                    nm += 1; break
        # controles nos arcos já baixados
        nc = 0; keys = [k for k in cache if k[0] == sat]
        for _ in range(20 * NCTRL):
            if nc >= NCTRL or not keys:
                break
            k = keys[rng.integers(len(keys))]; P, V = cache[k]
            ts = sorted(P)
            if len(ts) < 10:
                continue
            t0 = ts[rng.integers(len(ts))]; t0 = t0.replace(minute=5 * (t0.minute // 5), second=0, microsecond=0)
            T = t0 + timedelta(days=1)
            if any(t0 - timedelta(hours=6) <= x.to_pydatetime() <= T + timedelta(hours=6) for x in allm):
                continue
            x0 = state(sat, t0, files, cache); yT = state(sat, T, files, cache)
            if x0 is not None and yT is not None:
                events.append(dict(sat=int(sat), man=False, tau="none", s=0.0, t0=t0.isoformat(), tm=-1.0,
                                   x0=x0.tolist(), yT=yT.tolist())); nc += 1
        log[sat] = dict(files=len(files), ac=sorted({f[0] for f in files}), maneuvers=nm, controls=nc)
        print(sat, CODE[sat], log[sat], flush=True)
        json.dump(dict(log={str(k): v for k, v in log.items()}, events=events), open(OUT, "w"))
    print("ok", len(events))
