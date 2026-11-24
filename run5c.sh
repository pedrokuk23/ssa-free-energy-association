source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
pkill -f e12_score; sleep 2
for s in 41240 41335 43437 46984; do
 (python e12_fetch.py 40 25 7 $s e12_ev_$s.json > log5_fetch_$s.txt 2>&1) &
done
wait
python -c "
import json,glob
ev=[];log={}
for f in sorted(glob.glob('e12_ev_*.json')):
    d=json.load(open(f)); ev+=d['events']; log.update(d['log'])
json.dump(dict(log=log,events=ev),open('e12_events.json','w')); print(len(ev), log)
" > log5_merge.txt 2>&1
for m in full tracklet; do
 (MEAS=$m python e12_score.py 0.05 7 > log5_$m.txt 2>&1; echo DONE >> log5_$m.txt) &
done
wait
python e12_analyze.py > an5.log 2>&1; echo FIM5 >> an5.log
