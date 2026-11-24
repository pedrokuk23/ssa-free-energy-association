source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
( while kill -0 3265462 2>/dev/null; do sleep 60; done
  for v in "MEAS=full MISMATCH=1" "MEAS=tracklet MISMATCH=0" "MEAS=tracklet MISMATCH=1 COV_INFL=50"; do
    env $v NOBS=6 EXTRA=1 EXTRA2=1 TAG=od python e22_od.py 150 0.5 41 > "log20_od_$(echo $v | tr " =" "__").txt" 2>&1; done
  echo FIM > log20_od.txt ) &
( for m in tracklet full; do env MEAS=$m NS=20000 python e24_decomp.py 60 0.5 51 > log20_e24_${m}_0.5.txt 2>&1; done
  echo FIM > log20_e24.txt ) &
wait
