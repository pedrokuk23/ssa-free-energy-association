source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in tracklet full; do for s in 0.5 0.05; do
 (MEAS=$m python e11_numeric.py 1500 $s 5 > log4_${m}_$s.txt 2>&1; echo DONE >> log4_${m}_$s.txt) &
done; done
for be in 0.2 0.4; do
 (MEAS=full MISMATCH=1 SIG_BETA_EST=$be python e11_numeric.py 600 0.5 6 > log4_mis_$be.txt 2>&1; echo DONE >> log4_mis_$be.txt) &
done
wait
python e11_analyze.py > an4.log 2>&1; echo FIM4 >> an4.log
