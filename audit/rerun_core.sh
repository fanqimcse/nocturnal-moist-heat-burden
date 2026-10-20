#!/bin/bash
PY="C:/Users/Esther/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
CORE="step30_humidity_attribution.py step34_humidity_decomp_v2.py step35_nwh_detection_attribution.py step36_trend_robustness.py step38_tropical_threshold.py step39_south_north_stratified.py step43_decisive_spec.py step46_first_crossing_europe.py step47_exposure_projection.py step49_design_sensitivity.py step51_tropical_power.py step52_reference_implementation.py step53_graded_scale.py step54_warning_skill.py"
for s in $CORE; do
  t0=$(date +%s)
  timeout 900 "$PY" "$s" > "audit/rerun_${s%.py}.log" 2>&1
  ec=$?
  t1=$(date +%s)
  echo "$s EXIT=$ec TIME=$((t1-t0))s"
done
