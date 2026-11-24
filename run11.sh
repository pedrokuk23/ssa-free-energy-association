# E6c: varredura anual do detector D nos satélites novos (prioridade baixa; a validação continua na frente)
jobs="swa:2023 swb:2023 swc:2023 s1a:2023 s1b:2021 s1c:2025 c2fm1:2023 c2fm2:2023 c2fm3:2023 c2fm4:2023 c2fm5:2023 c2fm6:2023 tsx:2023 tdx:2023 gfo1:2023 gfo2:2023 champ:2005"
echo $jobs | tr ' ' '\n' | xargs -P 3 -I{} sh -c 'c=${1%%:*}; y=${1##*:}; YEARS=$y,$y nice -n 10 ~/venv/bin/python e15_scan.py $c > log11_$c.txt 2>&1; echo FIM >> log11_$c.txt' _ {}
echo TUDO > log11_all.txt
