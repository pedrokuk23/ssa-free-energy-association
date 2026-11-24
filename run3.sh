source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1
(K1=41 python e10_numeric.py 3000 0.5 3 > log3_s05.txt 2>&1; echo DONE >> log3_s05.txt) &
(K1=41 python e10_numeric.py 3000 0.05 3 > log3_s005.txt 2>&1; echo DONE >> log3_s005.txt) &
wait
(K1=41 MISMATCH=1 python e10_numeric.py 1000 0.5 3 > log3_mis.txt 2>&1; echo DONE >> log3_mis.txt) &
(for k in 11 21 41 81; do K1=$k TAG=kconv python e10_numeric.py 200 0.5 4; done > log3_kconv.txt 2>&1; echo DONE >> log3_kconv.txt) &
wait
echo FIM3 > fim3.txt
