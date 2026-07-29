#!/bin/bash
# Full manifold-geometry sweep: all 5 embodiments, ALL episodes, both
# granularities, then charts + tables. CPU only, no model inference.
#
#   conda activate gr00t-actlat && bash run_all.sh
#
# Cheap embodiments first so failures surface early. Logs land in logs/.
set -e
cd "$(dirname "$0")"
mkdir -p logs results figs tables

COMMON="--granularity single chunk --ambient gauss uniform \
        --nn-subsample 10000 --nn-boot 5 --nn-stride 25 \
        --n-ambient 100000 --occ-ref 15000 --corrdim-sub 5000"

for EMB in dexjoco_dual dexjoco_single gr1_tabletop robocasa_mg bridge; do
  echo "############## $EMB ##############"
  python -W ignore mg_run.py --embodiment "$EMB" $COMMON 2>&1 | tee "logs/${EMB}.log"
done

echo "############## PLOTS ##############"
python -W ignore mg_plots.py --granularity single chunk --ambient gauss 2>&1 | tee logs/plots.log

echo "ALL COMPLETE"
