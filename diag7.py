import json, sys, re, pandas as pd, numpy as np
from datetime import timedelta
import e12_fetch as F
L=pd.DataFrame(json.load(open('data/maneuver_labels_v0.3.json')))
L=L[(L.orbit_class=='LEO')&L.norad_id.isin(F.CODE)].copy(); L['t']=pd.to_datetime(L.epoch,utc=True,format='ISO8601')
acs=[x for x in F.sh(f"curl -s -m 60 --list-only {F.BASE}/").split() if re.fullmatch(r'[a-z0-9]{3}',x)]
print('ACs', acs, flush=True)
tot={}
for sat in F.CODE:
    B=L[(L.norad_id==sat)&L.delta_v.notna()&(L.delta_v>=0.05)]
    ok=np.zeros(len(B),bool); per={}
    for ac in acs:
        txt=F.sh(f"curl -s -m 60 --list-only {F.BASE}/{ac}/{F.CODE[sat]}/")
        iv=[]
        for f in txt.split():
            m=re.search(r"\.b(\d{5})\.e(\d{5})\..*sp3",f)
            if m: iv.append((F.yd2dt(m.group(1)), F.yd2dt(m.group(2))+timedelta(days=1)))
        if not iv: continue
        c=np.array([any(b<=t.to_pydatetime()-timedelta(hours=12)<e for b,e in iv) and any(b<=t.to_pydatetime()+timedelta(hours=12)<e for b,e in iv) for t in B.t])
        per[ac]=int(c.sum()); ok|=c
    print(sat, F.CODE[sat], len(B), per, 'union', int(ok.sum()), flush=True)
