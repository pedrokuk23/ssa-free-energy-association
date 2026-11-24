"""E6c: seleção pré-registrada (no máximo 8 manobras >= 5 cm/s por satélite, sorteio com semente 17) e registro de
congelamento (sha256 do plano, dos arquivos de eventos e do código; data UTC)."""
import json, hashlib, random, datetime, os
def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()
groups = {"meo": "e17_events_gnss.json", "qzss": ["e17_events_qzss.json", "e17_events_qzss2.json"], "leo": "e17_events_leo.json"}
BAD_QZSS = {49336, 62876}  # PRN mal identificado na 1a passagem; vêm de e17_events_qzss2.json
rec = {"utc": datetime.datetime.utcnow().isoformat() + "Z", "files": {}}
for g, fs in groups.items():
    fs = fs if isinstance(fs, list) else [fs]; ev = []
    for f in fs:
        if not os.path.exists(f): continue
        rec["files"][f] = sha(f); E = json.load(open(f))["events"]
        if f == "e17_events_qzss.json": E = [e for e in E if e["sat"] not in BAD_QZSS]
        ev += E
    man = [e for e in ev if e["man"] and e["s"] >= 0.05]
    rnd = random.Random(17); sel = []
    for s in sorted({e["sat"] for e in man}):
        m = [e for e in man if e["sat"] == s]; rnd.shuffle(m); sel += m[:8]
    json.dump(dict(events=sel), open(f"e17sel_{g}.json", "w"))
    rec[g] = dict(events=len(sel), satellites=len({e["sat"] for e in sel}))
    rec["files"][f"e17sel_{g}.json"] = sha(f"e17sel_{g}.json")
for f in ["PREREG_E6c.md", "e17_scoreK.py", "e10_numeric.py", "e11_numeric.py", "numprop.py", "numprop_gpu.py", "forces3b.py", "e17_gnss_events.py", "e15_src.py", "e12_fetch.py"]:
    rec["files"][f] = sha(f)
json.dump(rec, open("PREREG_E6c.freeze.json", "w"), indent=1); print(json.dumps({k: v for k, v in rec.items() if k != "files"}))
