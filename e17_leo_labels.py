"""Rótulos IDS dos LEO novos (SWOT, Sentinel-3C, Sentinel-6B) no formato maneuver-detect, com NORAD fictícios
90001-90003 (somente internos). Saída: data/labels_leo_new.json"""
import json
M = {"swoman.txt": 90001, "s3cman.txt": 90002, "s6bman.txt": 90003}
L = json.load(open("ids_labels.json")); L = L["labels"] if isinstance(L, dict) else L
out = [dict(norad_id=M[l["file"]], epoch=l["epoch"] + "+00:00", delta_v=l["delta_v"], maneuver_type=l["maneuver_type"],
            orbit_class="LEO", source="DORIS-IDS", source_ref=l["file"]) for l in L if l["file"] in M]
json.dump(out, open("data/labels_leo_new.json", "w")); print(len(out), sum(o["delta_v"] >= 0.05 for o in out))
