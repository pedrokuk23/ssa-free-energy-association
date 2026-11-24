import re, gzip, urllib.request as U, datetime as D
B = "https://isdc-data.gfz.de/gnss/products/final/"
def ls(u):
    h = U.urlopen(u, timeout=60).read().decode("latin1"); return list(dict.fromkeys(re.findall(r'href="([^"?/][^"?]*)"', h)))
L = ls(B); print("root", L[:4], L[-3:], len(L))
def gpsweek(d): return (d - D.date(1980, 1, 6)).days // 7
for (prn, day) in [("G16", D.date(2023, 2, 17)), ("J02", D.date(2018, 5, 20)), ("E24", D.date(2023, 6, 1))]:
    for k in range(-2, 3):
        d = day + D.timedelta(days=k); w = gpsweek(d); doy = d.timetuple().tm_yday
        try:
            fs = [f for f in ls(f"{B}w{w}/") if re.search(rf"{d.year}{doy:03d}.*(ORB\.SP3|sp3)", f, re.I) or re.search(rf"gf[zm]{w}{(d.weekday()+1)%7}\.sp3", f)]
        except Exception as e:
            print(prn, d, "ERR", e); continue
        if not fs: print(prn, d, "no file in", f"w{w}"); continue
        f = fs[0]; txt = gzip.decompress(U.urlopen(f"{B}w{w}/{f}", timeout=120).read()).decode("ascii", "ignore")
        n = sum(1 for l in txt.splitlines() if l.startswith("P" + prn)); hdr = txt.splitlines()[0][:60]
        print(prn, d, f, "epochs", n, "|", hdr, "| sats", len(set(l[1:4] for l in txt.splitlines() if l.startswith("P"))))
