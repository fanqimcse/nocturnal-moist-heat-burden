#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检验 v2 指标是否消除了观测频次混淆"""
import sys, csv
sys.stdout.reconfigure(encoding="utf-8")
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

rows = list(csv.DictReader(open("metrics_v2.csv", encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["station"], []).append(r)
for k in by:
    by[k].sort(key=lambda x: int(x["year"]))

for st in ["FMW00040308", "USW00021504", "USW00012918"]:
    if st not in by:
        continue
    print(f"\n=== {st} {by[st][0].get('name','')} ===")
    print(f"  {'年':<6}{'原始观测':>8}{'有效小时':>8}{'覆盖率':>8}"
          f"{'有效夜':>7}{'NRF计数':>8}{'NRF比率':>9}{'夜超阈h':>9}{'DTR_Tw':>8}{'合格':>5}")
    for r in by[st]:
        print("  %-6s%8s%8s%8.2f%7s%8s%9s%9s%8s%5s" % (
            r["year"], r["n_obs"], r["n_hours"], float(r["coverage"]),
            r["n_valid_nights"], r.get("nrf_elderly", ""),
            ("%.3f" % float(r["nrfr_elderly"])) if r.get("nrfr_elderly") else "-",
            ("%.0f" % float(r["nhrs_elderly"])) if r.get("nhrs_elderly") else "-",
            ("%.2f" % float(r["dtr_tw"])) if r.get("dtr_tw") else "-",
            r["ok_trend"]))
