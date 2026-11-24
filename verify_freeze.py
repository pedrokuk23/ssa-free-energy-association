"""Verifies the E6c freeze record (PREREG_E6c.freeze.json, 2026-11-08 04:02 UTC).
For each frozen file, the SHA-256 is checked against the copy in frozen_E6c/ if present, otherwise against the
top-level file. Files changed after freezing are described in the dated amendments of PREREG_E6c.md.
Usage: python verify_freeze.py"""
import hashlib, json, os
F = json.load(open("PREREG_E6c.freeze.json"))
NOTE = {"e17_scoreK.py": "changed by Amendment 3 (regularize-on-failure, resume); frozen copy not retained",
        "numprop_gpu.py": "changed by Amendment 2 (host-side Sun/Moon grid); frozen copy not retained"}
print("freeze utc:", F["utc"]); ok = 0
for f, h in F["files"].items():
    p = os.path.join("frozen_E6c", f) if os.path.exists(os.path.join("frozen_E6c", f)) else f
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    st = "OK" if got == h else ("CHANGED: " + NOTE.get(f, "see amendments") if got else "MISSING")
    ok += got == h; print(f"{p:32s} {st}")
print(f"{ok}/{len(F['files'])} frozen hashes reproduced")
