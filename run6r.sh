source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p evb_v1 && cp e12_evb_20436.json e12_evb_25260.json e12_evb_27421.json evb_v1/ 2>/dev/null
echo 20436 25260 27421 | tr ' ' '\n' | \
 xargs -P 3 -I{} sh -c 'ACS=ssa,gsc,grg,lca,ign NDRAW=12 MINDV=0.05 EXCL=e12_events_seed7.json timeout 10800 python e12_fetch.py 1000 40 11 {} e12_evb_{}.json > log6_fetch_{}.txt 2>&1; echo END >> log6_fetch_{}.txt'
python -c "
import json,glob
ev=[];log={}
for f in sorted(glob.glob('e12_evb_*.json')):
    d=json.load(open(f)); ev+=d['events']; log.update(d['log'])
json.dump(dict(log=log,events=ev),open('e12_events_b.json','w'))
print(len(ev), sum(e['man'] for e in ev)); print(log)
" > log6_merge.txt 2>&1
python e12_qc.py e12_events_b.json e12_events_bq.json > log6_qc.txt 2>&1
for m in full tracklet; do
 (EVENTS=e12_events_bq.json MEAS=$m python e12_score.py 0.05 11 > log6_$m.txt 2>&1; echo DONE >> log6_$m.txt) &
done
wait
SEED=11 python e12_analyze.py > an6.log 2>&1; echo FIM6 >> an6.log
