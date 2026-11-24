import json
for s in ("0.5", "0.05"):
    r = json.load(open(f"e9_rows_s{s}_seed1_K41.json")) + json.load(open(f"e9_rows_s{s}_seed2_K41.json"))
    json.dump(r, open(f"e9_rows_s{s}_all_K41.json", "w")); print(s, len(r))
