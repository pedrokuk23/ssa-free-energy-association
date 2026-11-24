#!/bin/bash
# Uso: bash run_gpu.sh   (rodar dentro de tmux)
set -e
cd ~/workspace/ssa_entropy
source ~/venv/bin/activate
pip install -q numba "numba-cuda[cu12]" 2>&1 | tail -1
if [ ! -f data/maneuver_labels_v0.3.json ]; then
  mkdir -p data
  git clone -q --depth 1 https://github.com/astro-tools/maneuver-detect /tmp/md_$$
  cp /tmp/md_$$/dataset/v0.3/labels.json data/maneuver_labels_v0.3.json
fi
python - <<'PY'
import numpy as np, time, numprop as C, numprop_gpu as G
from numba import cuda
print("GPU:", cuda.is_available(), cuda.get_current_device().name if cuda.is_available() else "")
from orbit import ref_state
rng=np.random.default_rng(0)
X=ref_state(alt_km=550)+rng.standard_normal((400,6))*[100,300,100,.2,.5,.2]; b=0.2*rng.standard_normal(400)
A=C.integrate(X,b,0.,86400.,30.); B=G.integrate(X,b,0.,86400.,30.)
print("validacao GPU vs CPU, 1 dia: max |dpos| = %.3e m" % np.abs(A-B)[:,:3].max())
Xb=np.repeat(X,500,0); bb=np.repeat(b,500)
t=time.time(); G.integrate(Xb,bb,0.,86400.,30.); el=time.time()-t
print("GPU: %d estados-dia em %.2f s -> %.0f estados-dia/s" % (len(Xb), el, len(Xb)/el))
PY
export USE_GPU=1 K1=41
python e9_numeric.py 500 0.5 1 > log_s05.txt 2>&1
python e9_numeric.py 300 0.05 1 > log_s005.txt 2>&1
python finalize.py e9_rows_s0.5_seed1_K41.json e9_rows_s0.05_seed1_K41.json | tee final.txt
echo FIM
