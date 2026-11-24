# Pós-E6c e fase D (exploratórios): E6c com CTRLp, IMM e campo F2 em escala do regime; sensibilidades do E5
source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 KDRAW=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G="LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO T_DAYS=2"
for m in full tracklet; do
  for c in 0 1; do (env $G EVENTS=e17sel_meo.json MEAS=$m A_REF=26.56e6 CHUNK=$c/2 python e17_scoreK2.py 0.05 17 > log19_meo_${m}_$c.txt 2>&1; echo FIM >> log19_meo_${m}_$c.txt) & done
  (env $G EVENTS=e17sel_qzss.json MEAS=$m A_REF=42.164e6 CHUNK=0/1 python e17_scoreK2.py 0.05 17 > log19_qzss_${m}.txt 2>&1; echo FIM >> log19_qzss_${m}.txt) &
  (EVENTS=e17sel_leo.json MEAS=$m T_DAYS=1 CHUNK=0/1 python e17_scoreK2.py 0.05 17 > log19_leo_${m}.txt 2>&1; echo FIM >> log19_leo_${m}.txt) &
done
N=300; S=31
( for v in "MEAS=tracklet" "MEAS=tracklet NJ=48" "MEAS=tracklet T_DAYS=2" "MEAS=tracklet COV_SCALE=2" "MEAS=full"; do
    env $v TAG=pd python e11_numeric.py $N 0.5 $S > "log19_pd_$(echo $v | tr ' =' '__').txt" 2>&1; done; echo FIM > log19_pdA.txt ) &
( for v in "MEAS=tracklet SIGNED=1" "MEAS=tracklet2" "MEAS=full NJ=48" "MEAS=full T_DAYS=2"; do
    env $v TAG=pd python e11_numeric.py $N 0.5 $S > "log19_pd_$(echo $v | tr ' =' '__').txt" 2>&1; done; echo FIM > log19_pdB.txt ) &
wait; echo TUDO > log19_all.txt
