"""E6c — varredura do detector D em satélites sem rótulos: janelas de +-2 dias centradas a cada 1 dia ao longo de
YEARS; cada janela devolve o melhor candidato em +-1 dia (z, época, dv, tipo). O limiar de z e a fusão de candidatos
vêm da validação (e13_eval.py) e são aplicados depois, sem olhar os escores de associação.
Uso: YEARS=2023,2023 [MAXDAYS=n] python e15_scan.py CODE"""
import os, sys, json
from datetime import timedelta
from datetime import datetime, timezone
import e13_detect as D, e15_src as S
code = sys.argv[1]; Y = os.environ.get("YEARS", "2023,2023").replace(",", "-")
if code.isdigit():  # satélite DORIS com rótulos (validação do modo varredura)
    import e12_fetch as F2
    y0, y1 = [int(x) for x in Y.split("-")]; a = datetime(y0, 1, 1, tzinfo=timezone.utc); b = datetime(y1 + 1, 1, 1, tzinfo=timezone.utc)
    files = [f for f in F2.listing(F2.CODE[int(code)]) if f[3] > a and f[2] < b]
    D.F = F2; code_k = int(code)
    S.evict = lambda cache, files, tmin: [cache.pop(k) for k in [k for k in cache if any(f[1] == k[1] and f[3] < tmin for f in files)]]
    files = sorted(files, key=lambda f: f[2])
    lo, hi = max(files[0][2], a), min(max(f[3] for f in files), b)
    files_rng = (lo, hi)
else:
    D.F = S; code_k = code
    files = S.listing(code); files_rng = None
print(code, "files", len(files), flush=True)
out = []
if files:
    b0, b1 = files_rng if files_rng else (files[0][2], files[-1][3]); cache = {}
    tm = (b0 + timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0); n = 0
    maxd = int(os.environ.get("MAXDAYS", "100000"))
    while tm <= b1 - timedelta(days=2) and n < maxd:
        S.evict(cache, files, tm - timedelta(days=3))
        try:
            r = D.window(code_k, tm, files, cache)
        except Exception as e:
            r = None; print("err", tm, repr(e)[:200], flush=True)
        if r is not None and r[0] is not None:
            d, t0 = r; d["t_abs"] = (t0 + timedelta(seconds=d["t"])).isoformat(); out.append(dict(tm=tm.isoformat(), det=d))
        n += 1
        if n % 10 == 0:
            print(code, n, len(out), flush=True); json.dump(out, open(f"e15_scan_{code}_{Y}.partial.json", "w"))
        tm += timedelta(days=1)
json.dump(out, open(f"e15_scan_{code}_{Y}.json", "w")); print("ok", code, len(out))
