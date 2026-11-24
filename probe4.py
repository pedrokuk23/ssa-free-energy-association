import re, sys, io, gzip, tarfile, zipfile, urllib.request as U
roots = ["https://isdc-data.gfz.de/grace-fo/ORBIT/L64/RSO/", "https://isdc-data.gfz.de/tsxtdx/ORBIT/L13/RSO/", "https://isdc-data.gfz.de/tsxtdx/ORBIT/L20/", "https://isdc-data.gfz.de/champ/ORBIT/L06/RSO/",
         "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/S1A/", "https://isdc-data.gfz.de/grace-fo/Level-1B/JPL/INSTRUMENT/RL04/2022/"]
def ls(u):
    h = U.urlopen(u, timeout=60).read().decode("latin1")
    return [x for x in dict.fromkeys(re.findall(r'href="([^"?/][^"?]*)"', h)) if not x.startswith("http") and x not in ("../",)]
def show(b, name):
    if name.endswith((".tgz", ".tar.gz", ".tar")):
        t = tarfile.open(fileobj=io.BytesIO(b)); m = [x for x in t.getmembers() if x.isfile()]; print("   tar:", [x.name for x in m][:6])
        b = t.extractfile(m[0]).read(); name = m[0].name
    elif name.endswith((".zip", ".ZIP")):
        z = zipfile.ZipFile(io.BytesIO(b)); print("   zip:", z.namelist()[:6]); n = [x for x in z.namelist() if not x.endswith("HDR")][0]; b = z.read(n); name = n
    elif name.endswith((".gz", ".Z")):
        try: b = gzip.decompress(b)
        except Exception as e: print("   gz err", e)
    txt = b[:3000].decode("latin1").splitlines()
    for l in txt[:14]: print("   |", l[:130])
for r in roots:
    u = r; print("==", r)
    try:
        for d in range(5):
            L = ls(u); print("  ", d, len(L), L[:5], "...", L[-2:])
            dirs = [x for x in L if x.endswith("/")]
            if dirs and len(dirs) == len(L): u = u + dirs[len(dirs) // 2]; continue
            f = [x for x in L if not x.endswith("/")][0]; print("   file", u + f)
            show(U.urlopen(u + f, timeout=120).read(), f); break
    except Exception as e: print("   ERR", e)
