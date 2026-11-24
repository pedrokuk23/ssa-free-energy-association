# Revisão (exploratórios, após o desbloqueio do E6c)
source ~/venv/bin/activate; cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41 KDRAW=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G="LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO T_DAYS=2"
k3(){ # grupo meas chunk nchunks arquivo-de-eventos conf
  local g=$1 m=$2 c=$3 n=$4 ev=$5 cf=$6 env0
  if [ $g = leo ]; then env0="T_DAYS=1"; elif [ $g = meo ]; then env0="$G A_REF=26.56e6"; else env0="$G A_REF=42.164e6"; fi
  env $env0 EVENTS=$ev ${cf:+CONF=$cf} MEAS=$m CHUNK=$c/$n python e17_scoreK3.py 0.05 17 > log20_${ev%.json}_${m}_$c.txt 2>&1
  echo FIM >> log20_${ev%.json}_${m}_$c.txt
}
( YEARS=2020,2026 python e21_conf.py meo > log20_conf_meo.txt 2>&1
  for m in full tracklet; do for c in 0 1 2; do
    ( k3 meo $m $c 3 e17sel_meo.json e21_conf_meo.json; [ $c = 2 ] && k3 meo $m 0 1 e17extra_meo.json ) & done; done; wait ) &
( YEARS=2018,2026 python e21_conf.py qzss > log20_conf_qzss.txt 2>&1
  for m in full tracklet; do
    ( k3 qzss $m 0 1 e17sel_qzss.json e21_conf_qzss.json; k3 qzss $m 0 2 e17extra_qzss.json ) &
    ( k3 qzss $m 1 2 e17extra_qzss.json ) & done; wait ) &
( python e21_conf.py leo > log20_conf_leo.txt 2>&1
  for m in full tracklet; do ( k3 leo $m 0 1 e17sel_leo.json; k3 leo $m 0 1 e17extra_leo.json ) & done; wait ) &
( for v in "MEAS=tracklet" "MEAS=full"; do env $v EXTRA=1 EXTRA2=1 TAG=pdx python e11_numeric.py 300 0.5 31 > "log20_pdx_${v#MEAS=}.txt" 2>&1; done
  for v in "MEAS=tracklet" "MEAS=tracklet2"; do env $v EXTRA=1 EXTRA2=1 IPLF_IT=5 TAG=pdit5 python e11_numeric.py 300 0.5 31 > "log20_pdit5_${v#MEAS=}.txt" 2>&1; done
  echo FIM > log20_pd.txt ) &
( for v in "MEAS=tracklet MISMATCH=1" "MEAS=full MISMATCH=1" "MEAS=tracklet MISMATCH=0" "MEAS=full MISMATCH=0" "MEAS=tracklet MISMATCH=1 COV_INFL=10" "MEAS=tracklet MISMATCH=1 COV_INFL=50"; do
    env $v NOBS=6 EXTRA=1 EXTRA2=1 TAG=od python e22_od.py 300 0.5 41 > "log20_od_$(echo $v | tr ' =' '__').txt" 2>&1; done
  echo FIM > log20_od.txt ) &
( for m in tracklet full; do for s in 0.5 0.05; do env MEAS=$m NS=50000 python e24_decomp.py 150 $s 51 > log20_e24_${m}_$s.txt 2>&1; done; done
  echo FIM > log20_e24.txt ) &
wait
python e25_k3_analyze.py > log20_e25.txt 2>&1
python e23_analyze.py 'e11_rows_pd*_seed31_*_x.json' e23_pd_summary.json > log20_e23pd.txt 2>&1
python e23_analyze.py 'e11_rows_od_*_seed41_*.json' e23_od_summary.json > log20_e23od.txt 2>&1
echo TUDO > log20_all.txt
