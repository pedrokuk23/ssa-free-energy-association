source ~/venv/bin/activate
cd ~/workspace/ssa_entropy
export USE_GPU=1 K1=41
python e9_numeric.py 3000 0.5 2 > log2_s05.txt 2>&1
python e9_numeric.py 3000 0.05 2 > log2_s005.txt 2>&1
python combine.py > final2.txt 2>&1
python finalize.py e9_rows_s0.5_all_K41.json e9_rows_s0.05_all_K41.json >> final2.txt 2>&1
cp e9_gpu_summary.json e9_gpu_summary_all.json
echo FIM >> final2.txt
