# E6c: eventos dos LEO novos (1 dia, como E6) e QZSS-1R/QZS-6 com PRN corrigido
~/venv/bin/python e17_leo_labels.py > log17l.txt 2>&1
for s in 90001 90002 90003; do
  (ACS=ssa,gsc,grg,lca,ign NDRAW=12 MINDV=0.05 LABELS=data/labels_leo_new.json timeout 10800 ~/venv/bin/python e12_fetch.py 1000 20 17 $s e17_evleo_$s.json >> log17l.txt 2>&1) &
done
wait
~/venv/bin/python -c "
import json,glob
ev=[];log={}
for f in sorted(glob.glob('e17_evleo_*.json')):
    d=json.load(open(f)); ev+=d['events']; log.update(d['log'])
json.dump(dict(log=log,events=ev),open('e17_events_leo.json','w')); print('LEO', len(ev), sum(e['man'] for e in ev), log)
" >> log17l.txt 2>&1
echo FIM >> log17l.txt
