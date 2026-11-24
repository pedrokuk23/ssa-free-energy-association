"""E6c — fontes de órbita precisa dos satélites novos (todas públicas, sem conta; verificadas em 2026-11-07).
Mesma interface de e12_fetch: CODE, listing(code) -> [(src, ref, b, e)], state(sat, t, files, cache) -> estado
pseudo-inercial [m, m/s] (rotação GMST, como em E6). Estados interpolados (Lagrange, 9 pontos) quando t não cai na
grade do arquivo; arcos sobrepostos: usa o arco em que t está mais longe das bordas.
Fontes: GFZ ISDC RSO SP3 (GRACE-FO C/D, TerraSAR-X, TanDEM-X, CHAMP); ESA swarm-diss SP3ACOM (Swarm A/B/C);
UCAR CDAAC leoOrb NRT SP3 (COSMIC-2 FM1-6); ESA STEP POEORB EOF (Sentinel-1A/B/C).
"""
import os, re, io, json, gzip, zipfile, tarfile, time, urllib.request as U, urllib.parse as UP
import numpy as np
from datetime import datetime, timedelta, timezone
from e12_fetch import to_inertial

GFZ = "https://isdc-data.gfz.de"
SWARM = "https://swarm-diss.eo.esa.int/"
C2 = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level1b"
STEP = "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
GNSS = "https://isdc-data.gfz.de/gnss/products/final"
MGEX = "ftp://ftp.gfz-potsdam.de/pub/GNSS/products/mgex"
MAXSTEP = {"gnss": 900.0, "mgex": 300.0}  # espaçamento máximo entre épocas (s) por fonte; padrão 300 s
# código -> (fonte, parâmetro)
CODE = {"gfo1": ("gfz", "grace-fo/ORBIT/L64/RSO"), "gfo2": ("gfz", "grace-fo/ORBIT/L65/RSO"),
        "tsx": ("gfz", "tsxtdx/ORBIT/L13/RSO"), "tdx": ("gfz", "tsxtdx/ORBIT/L20/RSO"),
        "champ": ("gfz", "champ/ORBIT/L06/RSO"),
        "swa": ("swarm", "A"), "swb": ("swarm", "B"), "swc": ("swarm", "C"),
        **{f"c2fm{k}": ("cosmic2", k) for k in range(1, 7)},
        "s1a": ("s1", "S1A"), "s1b": ("s1", "S1B"), "s1c": ("s1", "S1C")}
CACHE = "new_cache"; os.makedirs(CACHE, exist_ok=True)
T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)
YEARS = None  # [ano0, ano1] — definido pelo chamador (os.environ YEARS="2023,2023")

def add_gnss(prn, src="gnss"):
    """registra um satélite GNSS pelo PRN do SP3 (G16, E24, J07...); src = gnss (GFZ final) ou mgex (GFZ MGEX rápido)."""
    CODE[prn] = (src, prn)

def _ftpls(url):
    import subprocess
    for k in range(5):
        o = subprocess.run(f"curl -s -m 90 --list-only {url}", shell=True, capture_output=True, text=True).stdout
        if o.strip():
            return o.split()
        time.sleep(5 * (k + 1))
    return []

def _get(url, tries=5, timeout=120):
    for k in range(tries):
        try:
            return U.urlopen(url, timeout=timeout).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))

def _href(url):
    h = _get(url).decode("latin1")
    return [x for x in dict.fromkeys(re.findall(r'href="([^"?/][^"?]*)"', h)) if not x.startswith(("http", ".."))]

def _years():
    y = os.environ.get("YEARS", "2023,2023").split(",")
    return range(int(y[0]), int(y[1]) + 1)

def _dt(s):
    return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

def listing(code):
    src, par = CODE[code]
    lc = os.path.join(CACHE, f"list_{code}_{os.environ.get('YEARS', '2023,2023').replace(',', '-')}.json")
    if os.path.exists(lc):
        return [(a, b, datetime.fromisoformat(c), datetime.fromisoformat(d)) for a, b, c, d in json.load(open(lc))]
    out = []
    if src == "gfz":
        for y in _years():
            try:
                days = [d for d in _href(f"{GFZ}/{par}/{y}/") if d.endswith("/")]
            except Exception:
                continue
            for d in days:
                for f in _href(f"{GFZ}/{par}/{y}/{d}"):
                    m = re.search(r"_(\d{8})_(\d{6})_(\d{8})_(\d{6})_v(\d+)\.sp3\.gz$", f)
                    if m:
                        out.append(("gfz", f"{par}/{y}/{d}{f}", _dt(m[1] + m[2]), _dt(m[3] + m[4])))
    elif src == "swarm":
        path = f"swarm/Level2daily/Latest_baselines/POD/RD/Sat_{par}"; pos = 0
        while True:
            q = UP.urlencode({"do": "list", "maxfiles": 2000, "pos": pos, "file": path})
            r = json.loads(_get(SWARM + "?" + q)); res = r.get("results", [])
            for x in res:
                m = re.search(r"__(\d{8})T(\d{6})_(\d{8})T(\d{6})_(\d{4})\.ZIP$", x["name"])
                if m and int(m[1][:4]) in _years():
                    out.append(("swarm", x["path"], _dt(m[1] + m[2]), _dt(m[3] + m[4])))
            if not res or pos > 40000:
                break
            pos += len(res)
        best = {}
        for o in out:  # maior versão de baseline por intervalo
            k = (o[2], o[3])
            if k not in best or o[1] > best[k][1]:
                best[k] = o
        out = sorted(best.values(), key=lambda o: o[2])
    elif src == "cosmic2":
        for y in _years():
            for doy in range(1, 367):
                d0 = datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                if d0.year != y:
                    break
                out.append(("cosmic2", f"{y}/{doy:03d}/leoOrb_nrt_{y}_{doy:03d}.tar.gz", d0, d0 + timedelta(days=1)))
    elif src == "gnss":
        lg = os.path.join(CACHE, f"list_gnssfin_{os.environ.get('YEARS', '2023,2023').replace(',', '-')}.json")
        if os.path.exists(lg):
            out = [(a, b, datetime.fromisoformat(c), datetime.fromisoformat(d)) for a, b, c, d in json.load(open(lg))]
            return out
        for w in sorted(x.strip("/") for x in _href(GNSS + "/") if x.startswith("w")):
            wk = int(w[1:]); d0 = datetime(1980, 1, 6, tzinfo=timezone.utc) + timedelta(weeks=wk)
            if d0.year + 1 < min(_years()) or d0.year > max(_years()):
                continue
            for f in _href(f"{GNSS}/{w}/"):
                m = re.search(r"GFZ0OPSFIN_(\d{4})(\d{3})0000_01D_\d+M_ORB\.SP3\.gz$", f)
                if m and int(m[1]) in _years():
                    b = datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1)
                    out.append(("gnss", f"{w}/{f}", b, b + timedelta(days=1)))
        out.sort(key=lambda o: o[2]); json.dump([(a, b, c.isoformat(), d.isoformat()) for a, b, c, d in out], open(lg, "w"))
        return out
    elif src == "mgex":
        lg = os.path.join(CACHE, f"list_mgex_{os.environ.get('YEARS', '2023,2023').replace(',', '-')}.json")
        if os.path.exists(lg):
            return [(a, b, datetime.fromisoformat(c), datetime.fromisoformat(d)) for a, b, c, d in json.load(open(lg))]
        dirs = {}
        for d in _ftpls(MGEX + "/"):
            m = re.match(r"^(\d{4})(_IGS20)?$", d)
            if m and (int(m[1]) not in dirs or m[2]):
                dirs[int(m[1])] = d
        for wk, d in sorted(dirs.items()):
            d0 = datetime(1980, 1, 6, tzinfo=timezone.utc) + timedelta(weeks=wk)
            if d0.year + 1 < min(_years()) or d0.year > max(_years()):
                continue
            best = {}
            for f in _ftpls(f"{MGEX}/{d}/"):
                m = re.search(r"GBM0MGX(RAP|FIN)_(\d{4})(\d{3})0000_01D_\d+M_ORB\.SP3\.gz$", f)
                if m and int(m[2]) in _years():
                    k = (m[2], m[3])
                    if k not in best or m[1] == "FIN":
                        best[k] = f
            for (yy, dd), f in best.items():
                b = datetime(int(yy), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(dd) - 1)
                out.append(("mgex", f"{d}/{f}", b, b + timedelta(days=1)))
        out.sort(key=lambda o: o[2]); json.dump([(a, b, c.isoformat(), d.isoformat()) for a, b, c, d in out], open(lg, "w"))
        return out
    elif src == "s1":
        for y in _years():
            for mm in range(1, 13):
                try:
                    fs = _href(f"{STEP}/{par}/{y}/{mm:02d}/")
                except Exception:
                    continue
                for f in fs:
                    m = re.search(r"_V(\d{8})T(\d{6})_(\d{8})T(\d{6})\.EOF", f)
                    if m:
                        out.append(("s1", f"{par}/{y}/{mm:02d}/{f}", _dt(m[1] + m[2]), _dt(m[3] + m[4])))
    out.sort(key=lambda o: o[2])
    json.dump([(a, b, c.isoformat(), d.isoformat()) for a, b, c, d in out], open(lc, "w"))
    return out

def _sp3(txt, sid=None):
    T, P, V = [], [], []; t = None; p = v = None
    def push():
        if t is not None and p is not None:
            T.append(t); P.append(p); V.append(v if v is not None else [np.nan] * 3)
    for ln in txt.splitlines():
        if ln.startswith("* "):
            push(); a = ln[1:].split()
            sec = float(a[5]); t = (datetime(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), tzinfo=timezone.utc) - T0).total_seconds() + sec
            p = v = None
        elif ln.startswith("P") and t is not None and p is None and (sid is None or ln[1:4] == sid):
            p = [float(ln[4:18]) * 1e3, float(ln[18:32]) * 1e3, float(ln[32:46]) * 1e3]
        elif ln.startswith("V") and t is not None and v is None and (sid is None or ln[1:4] == sid):
            v = [float(ln[4:18]) * 0.1, float(ln[18:32]) * 0.1, float(ln[32:46]) * 0.1]
    push()
    return np.array(T), np.array(P), np.array(V)

def _eof(txt):
    T = [(datetime.fromisoformat(s).replace(tzinfo=timezone.utc) - T0).total_seconds() for s in re.findall(r"<UTC>UTC=([^<]+)</UTC>", txt)]
    g = lambda k: np.array([float(x) for x in re.findall(rf"<{k} unit=\"m(?:/s)?\">([^<]+)</{k}>", txt)])
    return np.array(T), np.c_[g("X"), g("Y"), g("Z")], np.c_[g("VX"), g("VY"), g("VZ")]

def _fix_v(T, P, V, maxstep=300.0):
    """velocidades ausentes ou inconsistentes -> derivada de Lagrange das posições (mesma regra de E6)."""
    Vd = np.full_like(P, np.nan)
    for i in range(4, len(T) - 4):
        dt = T[i - 4:i + 5] - T[i]
        if np.ptp(dt) <= 8 * maxstep and np.abs(P[i - 4:i + 5]).max() < 1e9:
            Vd[i] = [np.polyder(np.polyfit(dt / 60.0, P[i - 4:i + 5, j], 8))[-1] / 60.0 for j in range(3)]
    ok = np.isfinite(V).all(1) & np.isfinite(Vd).all(1)
    if not np.isfinite(V).any():
        return Vd
    if np.isfinite(V).all(1).mean() < 0.5 or (ok.any() and np.median(np.linalg.norm(V[ok] - Vd[ok], axis=1)) > 1.0):
        V = Vd
    return V

def load(code, f):
    src, par = CODE[code]; _, ref, b, e = f
    loc = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", ref))
    if not os.path.exists(loc):
        url = {"gfz": f"{GFZ}/{ref}", "swarm": SWARM + "?" + UP.urlencode({"do": "download", "file": ref}),
               "cosmic2": f"{C2}/{ref}", "s1": f"{STEP}/{ref}", "gnss": f"{GNSS}/{ref}", "mgex": f"{MGEX}/{ref}"}[src]
        if url.startswith("ftp://"):
            import subprocess
            subprocess.run(f"curl -s -m 600 -o {loc}.tmp {url}", shell=True); os.replace(loc + ".tmp", loc)
        else:
            b_ = _get(url, timeout=600); open(loc + ".tmp", "wb").write(b_); os.replace(loc + ".tmp", loc)
    raw = open(loc, "rb").read()
    if src == "gfz":
        T, P, V = _sp3(gzip.decompress(raw).decode("ascii", "ignore"))
    elif src in ("gnss", "mgex"):
        key = ("gnsstxt", loc)
        if _TXT.get("k") != loc:
            _TXT["k"] = loc; _TXT["v"] = gzip.decompress(raw).decode("ascii", "ignore")
        T, P, V = _sp3(_TXT["v"], par)
        if len(T) and (np.abs(P) > 1e9).any():   # SP3: 0.000000 = ausente
            pass
    elif src == "swarm":
        z = zipfile.ZipFile(io.BytesIO(raw)); n = [x for x in z.namelist() if x.lower().endswith(".sp3")][0]
        T, P, V = _sp3(z.read(n).decode("ascii", "ignore"))
    elif src == "s1":
        if raw[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(raw)); raw = z.read(z.namelist()[0])
        T, P, V = _eof(raw.decode("ascii", "ignore"))
    else:  # cosmic2: arcos curtos sobrepostos; por época, o arco em que ela está mais central
        tf = tarfile.open(fileobj=io.BytesIO(raw)); arcs = []
        for m in tf.getmembers():
            mm = re.search(r"leoOrb_\d{4}\.\d{3}\.(\d{3})\.\d+_", m.name)
            if m.isfile() and mm and int(mm[1]) == par:
                arcs.append(_sp3(tf.extractfile(m).read().decode("ascii", "ignore")))
        if not arcs:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        allT = np.unique(np.concatenate([a[0] for a in arcs])); P = np.full((len(allT), 3), np.nan); V = P.copy(); best = np.full(len(allT), -1.0)
        for Ta, Pa, Va in arcs:
            idx = np.searchsorted(allT, Ta); c = np.minimum(Ta - Ta[0], Ta[-1] - Ta)
            upd = c > best[idx]; best[idx[upd]] = c[upd]; P[idx[upd]] = Pa[upd]; V[idx[upd]] = Va[upd]
        T = allT
    if len(T) == 0:
        return T, P, V
    o = np.argsort(T); T, P, V = T[o], P[o], V[o]
    keep = np.r_[True, np.diff(T) > 0]; T, P, V = T[keep], P[keep], V[keep]
    good = np.isfinite(P).all(1) & (np.linalg.norm(P, axis=1) > 1e6)
    T, P, V = T[good], P[good], V[good]
    return T, P, _fix_v(T, P, V, MAXSTEP.get(src, 300.0))

_TXT = {}

def _interp(T, X, s, maxspan=9 * 60.0):
    i = np.searchsorted(T, s)
    if i < len(T) and T[i] == s:
        return X[i]
    lo, hi = i - 4, i + 5
    if lo < 0 or hi > len(T) or T[hi - 1] - T[lo] > maxspan or not np.isfinite(X[lo:hi]).all():
        return None
    w = T[lo:hi] - s; out = np.zeros(X.shape[1])
    for k in range(9):
        L = np.prod([(0 - w[j]) / (w[k] - w[j]) for j in range(9) if j != k]); out += L * X[lo + k]
    return out

def state(sat, t, files, cache):
    s = (t - T0).total_seconds(); cand = [f for f in files if f[2] <= t < f[3]]
    cand.sort(key=lambda f: -min((t - f[2]).total_seconds(), (f[3] - t).total_seconds()))
    for f in cand:
        key = (sat, f[1])
        if key not in cache:
            cache[key] = load(sat, f)
        T, P, V = cache[key]
        if len(T) == 0:
            continue
        ms = 9 * MAXSTEP.get(CODE[sat][0], 60.0)
        p = _interp(T, P, s, ms); v = _interp(T, V, s, ms)
        if p is not None and v is not None and np.isfinite(p).all() and np.isfinite(v).all():
            return to_inertial(t, p, v)
    return None

def evict(cache, files, tmin):
    """libera arquivos que terminam antes de tmin (varredura longa)."""
    old = {f[1] for f in files if f[3] < tmin}
    for k in [k for k in cache if k[1] in old]:
        del cache[k]
