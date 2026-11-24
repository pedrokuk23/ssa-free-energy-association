# E6c: pontuação confirmatória (depois de PREREG_E6c.freeze.json)
source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 KDRAW=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in full tracklet; do
  for c in 0 1; do (EVENTS=e17sel_meo.json MEAS=$m T_DAYS=2 A_REF=26.56e6 LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO CHUNK=$c/2 python e17_scoreK.py 0.05 17 > log18_meo_${m}_$c.txt 2>&1; echo FIM >> log18_meo_${m}_$c.txt) & done
  (EVENTS=e17sel_qzss.json MEAS=$m T_DAYS=2 A_REF=42.164e6 LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO CHUNK=0/1 python e17_scoreK.py 0.05 17 > log18_qzss_${m}.txt 2>&1; echo FIM >> log18_qzss_${m}.txt) &
  (EVENTS=e17sel_leo.json MEAS=$m T_DAYS=1 CHUNK=0/1 python e17_scoreK.py 0.05 17 > log18_leo_${m}.txt 2>&1; echo FIM >> log18_leo_${m}.txt) &
done
wait; echo TUDO > log18_all.txt
