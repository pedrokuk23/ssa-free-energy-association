source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 KDRAW=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in 0 1; do (EVENTS=e17sel_meo.json MEAS=tracklet T_DAYS=2 A_REF=26.56e6 LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO CHUNK=$c/2 python e17_scoreK.py 0.05 17 > log18c_meo_tracklet_$c.txt 2>&1; echo FIM >> log18c_meo_tracklet_$c.txt) & done
wait; echo TUDO > log18c_all.txt
