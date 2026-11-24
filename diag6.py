import json, sys, pandas as pd, numpy as np
from datetime import timedelta
import e12_fetch as F
L=pd.DataFrame(json.load(open('data/maneuver_labels_v0.3.json')))
L=L[(L.orbit_class=='LEO')&L.norad_id.isin(F.CODE)].copy(); L['t']=pd.to_datetime(L.epoch,utc=True,format='ISO8601')
for sat in [int(x) for x in sys.argv[1].split(',')]:
    files=F.listing(F.CODE[sat])
    B=L[(L.norad_id==sat)&L.delta_v.notna()&(L.delta_v>=0.05)]
    cov=lambda t: any(b<=t<e for _,_,b,e in files)
    a=[(cov(t.to_pydatetime()-timedelta(hours=12)), cov(t.to_pydatetime()+timedelta(hours=12))) for t in B.t]
    a=np.array(a)
    print(sat, len(B), 'cov before',a[:,0].sum(),'after',a[:,1].sum(),'both',(a.all(1)).sum(), 'range', files[0][2].date() if files else None, files[-1][3].date() if files else None, 'big years', sorted(set(B.t.dt.year))[:3], sorted(set(B.t.dt.year))[-3:], flush=True)
    # check one file's epoch step
    if files:
        P,V=F.load_sp3(*files[len(files)//2][:1], F.CODE[sat], files[len(files)//2][1]) if False else (None,None)
