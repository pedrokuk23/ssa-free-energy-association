source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 KDRAW=20
for i in 0 1 2; do (CHUNK=$i/3 MEAS=full python e12_scoreK.py 0.05 13 > log7_full_$i.txt 2>&1; echo DONE >> log7_full_$i.txt) & done
for i in 0 1 2 3 4 5; do (CHUNK=$i/6 MEAS=tracklet python e12_scoreK.py 0.05 13 > log7_trk_$i.txt 2>&1; echo DONE >> log7_trk_$i.txt) & done
wait
python e12K_analyze.py > an7.log 2>&1; echo FIM7 >> an7.log
