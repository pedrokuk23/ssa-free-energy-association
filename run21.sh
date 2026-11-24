source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for v in "MEAS=tracklet MISMATCH=1" "MEAS=tracklet MISMATCH=0" "MEAS=full MISMATCH=1"; do
  ( env $v NOBS=6 EXTRA=1 EXTRA2=1 TAG=od2 python e22_od.py 150 0.5 41 > "log21_od2_$(echo $v | tr " =" "__").txt" 2>&1 ) &
done
wait
echo FIM > log21_od2.txt
