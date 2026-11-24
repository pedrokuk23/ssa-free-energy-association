set -x
mkdir -p data/probe && cd data/probe
curl -s -m 60 -o sw.zip 'https://swarm-diss.eo.esa.int/?do=download&file=swarm%2FLevel2daily%2FLatest_baselines%2FPOD%2FRD%2FSat_A%2FSW_OPER_SP3ACOM_2__20140101T235944_20140102T235944_0201.ZIP'
ls -la sw.zip; unzip -o -q sw.zip && ls; for f in *.sp3 *.SP3 SW_*; do [ -f "$f" ] && head -c 600 "$f" && sed -n '23,26p' "$f"; done 2>/dev/null | head -30
curl -s -m 60 -o c2.tgz https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level1b/2024/136/leoOrb_nrt_2024_136.tar.gz; ls -la c2.tgz; tar tzf c2.tgz | head -5
mkdir -p c2 && tar xzf c2.tgz -C c2 && f=$(find c2 -type f | head -1) && ~/venv/bin/python -c "import netCDF4,sys;d=netCDF4.Dataset(sys.argv[1]);print(d);[print(k,v.dimensions,getattr(v,'units','')) for k,v in d.variables.items()]" "$f" 2>&1 | head -40
curl -s -m 60 'https://isdcftp.gfz-potsdam.de/grace-fo/Level-1B/JPL/INSTRUMENT/RL04/2020/' | grep -o 'gracefo_1B_[^"<]*' | head -3
