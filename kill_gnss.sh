for p in $(pgrep -f 'e17_scoreK.py 0.05 17'); do if tr '\0' '\n' < /proc/$p/environ | grep -qE 'EVENTS=e17sel_(meo|qzss)'; then kill $p; echo killed $p; fi; done
