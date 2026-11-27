# E6c covariance check (audit revision 3, R03). Draws actually run for e33_*.txt; one core per stream.
# LEO: events 0,8,9, two draws each; MEO: event 4, one draw; QZSS: event 0, one draw. Both UCT types.
B="K1=41 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
G="LABELS=data/labels_gnss.json PRIOR_CLASS=MEO,GEO,IGSO T_DAYS=2"
T="MEAS=tracklet CIRC_MEAN=1 EQ_SQRT=1"
run(){ env $B $1 python3 e33_e6c_covariance_check.py $2 $3 $4 . ; }
( run "T_DAYS=1 MEAS=full" e17sel_leo.json 0,8,9 2
  run "$G A_REF=26.56e6 MEAS=full" e17sel_meo.json 4 1
  run "$G A_REF=42.164e6 MEAS=full" e17sel_qzss.json 0 1 ) > e33_full.txt 2>&1 &
( run "T_DAYS=1 $T" e17sel_leo.json 0,8,9 2
  run "$G A_REF=26.56e6 $T" e17sel_meo.json 4 1
  run "$G A_REF=42.164e6 $T" e17sel_qzss.json 0 1 ) > e33_tracklet.txt 2>&1 &
wait

# Revision 4 (R4-01, R4-03): target and neighbour scores vs archive, decisions, asymmetry, rounding sensitivity.
P4="PERTURB_EPS=1e-15,1e-14,1e-13,1e-12"
( env $B T_DAYS=1 $T $P4 python3 e33_e6c_covariance_check.py e17sel_leo.json 0,8,9 4 . > e33_r4_leo_tracklet.txt 2>&1
  env $B T_DAYS=1 MEAS=full PERTURB_EPS=1e-15 python3 e33_e6c_covariance_check.py e17sel_leo.json 0,8,9 2 . > e33_r4_leo_full.txt 2>&1
  env $B $G A_REF=26.56e6 MEAS=full PERTURB_EPS=1e-15 python3 e33_e6c_covariance_check.py e17sel_meo.json 4 1 . > e33_r4_meo_full.txt 2>&1 ) &
( env $B $G A_REF=42.164e6 $T PERTURB_EPS=1e-15,1e-13 python3 e33_e6c_covariance_check.py e17sel_qzss.json 0 1 . > e33_r4_qzss_tracklet.txt 2>&1
  env $B $G A_REF=42.164e6 MEAS=full PERTURB_EPS=1e-15 python3 e33_e6c_covariance_check.py e17sel_qzss.json 0 1 . > e33_r4_qzss_full.txt 2>&1 ) &
wait
