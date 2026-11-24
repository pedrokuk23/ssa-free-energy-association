source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python e12_qc.py e12_events_b.json e12_events_bq.json > log6_qc.txt 2>&1
for m in full tracklet; do
 (EVENTS=e12_events_bq.json MEAS=$m python e12_score.py 0.05 11 > log6_$m.txt 2>&1; echo DONE >> log6_$m.txt) &
done
wait
SEED=11 python e12_analyze.py > an6.log 2>&1; echo FIM6 >> an6.log
