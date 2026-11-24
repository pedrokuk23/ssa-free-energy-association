# Audit rerun (2026-11-24): every tracklet scoring repeated with the angle fix of e11_numeric.ev()
# (right ascension averaged on the circle; symmetric sigma-point square root). Full-state UCTs are not affected.
# Run in a copy of the workspace: ~/workspace/ssa_entropy_circ (original rows stay in ~/workspace/ssa_entropy).
source ~/venv/bin/activate; cd ~/workspace/ssa_entropy_circ
export USE_GPU=1 K1=41 KDRAW=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CIRC_MEAN=1 EQ_SQRT=1
echo "esperando GPU livre $(date)" > logc_status.txt
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do sleep 600; done
echo "inicio $(date)" >> logc_status.txt
G="LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO T_DAYS=2"
k3(){ # grupo chunk nchunks arquivo-de-eventos conf   (tracklet only)
  local g=$1 c=$2 n=$3 ev=$4 cf=$5 env0
  if [ $g = leo ]; then env0="T_DAYS=1"; elif [ $g = meo ]; then env0="$G A_REF=26.56e6"; else env0="$G A_REF=42.164e6"; fi
  env $env0 EVENTS=$ev ${cf:+CONF=$cf} MEAS=tracklet CHUNK=$c/$n python e17_scoreK3.py 0.05 17 > logc_${ev%.json}_$c.txt 2>&1
  echo FIM >> logc_${ev%.json}_$c.txt
}
# ---- wave 1: E5 Study B (Table E5b) and E6c (Table E6c, S5)
for s in 0.5 0.05; do (MEAS=tracklet python e11_numeric.py 1500 $s 5 > logc_e5b_$s.txt 2>&1; echo FIM >> logc_e5b_$s.txt) & done
for c in 0 1 2; do (k3 meo $c 3 e17sel_meo.json e21_conf_meo.json) & done
(k3 qzss 0 1 e17sel_qzss.json e21_conf_qzss.json; k3 leo 0 1 e17sel_leo.json) &
(k3 meo 0 1 e17extra_meo.json; k3 qzss 0 2 e17extra_qzss.json; k3 qzss 1 2 e17extra_qzss.json; k3 leo 0 1 e17extra_leo.json) &
wait; echo "onda 1 fim $(date)" >> logc_status.txt
# ---- wave 2: E6a/E6b (Supplementary Tables e6, paired3), E6s, S3 sensitivity and OD runs
(EVENTS=e12_events_seed7.json MEAS=tracklet python e12_score.py 0.05 7 > logc_e6a.txt 2>&1; EVENTS=e12_events_bq.json MEAS=tracklet python e12_score.py 0.05 11 > logc_e6b.txt 2>&1; echo FIM >> logc_e6b.txt) &
for c in 0 1 2; do (CHUNK=$c/3 MEAS=tracklet python e12_scoreK.py 0.05 13 > logc_e6s_$c.txt 2>&1; echo FIM >> logc_e6s_$c.txt) & done
( for v in "MEAS=tracklet" "MEAS=tracklet NJ=48" "MEAS=tracklet T_DAYS=2" "MEAS=tracklet COV_SCALE=2" "MEAS=tracklet SIGNED=1" "MEAS=tracklet2"; do
    env $v TAG=pd python e11_numeric.py 300 0.5 31 > "logc_pd_$(echo $v | tr ' =' '__').txt" 2>&1; done
  env MEAS=tracklet EXTRA=1 EXTRA2=1 TAG=pdx python e11_numeric.py 300 0.5 31 > logc_pdx.txt 2>&1
  for v in "MEAS=tracklet" "MEAS=tracklet2"; do env $v EXTRA=1 EXTRA2=1 IPLF_IT=5 TAG=pdit5 python e11_numeric.py 300 0.5 31 > "logc_pdit5_${v#MEAS=}.txt" 2>&1; done
  echo FIM > logc_pd.txt ) &
( for v in "MEAS=tracklet MISMATCH=1" "MEAS=tracklet MISMATCH=0"; do
    env $v NOBS=6 EXTRA=1 EXTRA2=1 TAG=od2 python e22_od.py 150 0.5 41 > "logc_od2_$(echo $v | tr ' =' '__').txt" 2>&1; done; echo FIM > logc_od2.txt ) &
wait; echo "onda 2 fim $(date)" >> logc_status.txt
# ---- summaries (full-state rows are the original ones, copied into this directory)
python e11_analyze.py > logc_an_e11.txt 2>&1
python e12_pool.py > logc_an_pool.txt 2>&1
for s in 7 11 pool; do SEED=$s python e12_analyze.py > logc_an_e12_$s.txt 2>&1; done
python e12K_analyze.py > logc_an_e12K.txt 2>&1
python e14_recal.py > logc_an_e14.txt 2>&1
python e19_pd_analyze.py > logc_an_e19.txt 2>&1
python e23_analyze.py 'e11_rows_pdx_*_seed31_*_x.json' e23_pdx.json > logc_an_pdx.txt 2>&1
python e23_analyze.py 'e11_rows_pdit5_*_seed31_*_x.json' e23_pdit5.json > logc_an_pdit5.txt 2>&1
python e23_analyze.py 'e11_rows_od2_*_seed41_*.json' e23_od2.json > logc_an_od2.txt 2>&1
python e25_k3_analyze.py > logc_an_e25.txt 2>&1
echo "TUDO $(date)" >> logc_status.txt
