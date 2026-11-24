# E6c encadeado: espera QZSS (1a passagem) e LEO novos; refaz QZS-1R/QZS-6 com PRN corrigido; rótulos; congela; pontua
cd ~/workspace/ssa_entropy
while ! grep -q FIM log17q.txt 2>/dev/null || ! grep -q FIM log17l.txt 2>/dev/null; do sleep 60; done
GSRC=mgex SOURCES=QZSS-OHI YEARS=2018,2026 NORADS=49336,62876 OUT=e17_events_qzss2.json ~/venv/bin/python e17_gnss_events.py 6 33 > log17q2.txt 2>&1
~/venv/bin/python e17_labels.py > log17lab.txt 2>&1
~/venv/bin/python e17_select.py > log17sel.txt 2>&1
bash run18.sh
