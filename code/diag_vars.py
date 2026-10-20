#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断：各变量的真实可得性
inventory 统计的是"记录总数"，但湿球字段可能大量缺失。
若 temperature + relative_humidity / dew_point 可得，
则可自算湿球，大幅扩展可用数据，且保证全站全年代算法一致。
"""
import sys, glob, os, collections
sys.stdout.reconfigure(encoding="utf-8")
import duckdb

os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob("ghcn_data/GHCNh_*.parquet"))
con = duckdb.connect()

print(f"{'文件':<34}{'总记录':>8}{'T':>8}{'Tw':>8}{'RH':>8}{'Td':>8}"
      f"{'T&RH':>8}{'T&Td':>8}{'气压':>8}")
print("-" * 100)
sample = files[:12] + [f for f in files if "THI0000VTBD" in f][:4]
tot = collections.Counter()
for p in sample:
    b = os.path.basename(p)
    con.execute("CREATE OR REPLACE VIEW h AS SELECT * FROM read_parquet(['%s'])" % p)
    q = lambda s: con.execute(s).fetchall()[0][0]
    n = q("SELECT COUNT(*) FROM h")
    t = q("SELECT COUNT(temperature) FROM h")
    tw = q("SELECT COUNT(wet_bulb_temperature) FROM h")
    rh = q("SELECT COUNT(TRY_CAST(relative_humidity AS DOUBLE)) FROM h")
    td = q("SELECT COUNT(dew_point_temperature) FROM h")
    trh = q("""SELECT COUNT(*) FROM h WHERE temperature IS NOT NULL
               AND TRY_CAST(relative_humidity AS DOUBLE) IS NOT NULL""")
    ttd = q("""SELECT COUNT(*) FROM h WHERE temperature IS NOT NULL
               AND dew_point_temperature IS NOT NULL""")
    pr = q("""SELECT COUNT(*) FROM h WHERE station_level_pressure IS NOT NULL
              OR sea_level_pressure IS NOT NULL""")
    print(f"{b:<34}{n:>8}{t:>8}{tw:>8}{rh:>8}{td:>8}{trh:>8}{ttd:>8}{pr:>8}")

print("\n--- 全样本汇总 (所有已下载文件) ---")
N = T = TW = RH = TD = TRH = TTD = 0
for p in files:
    con.execute("CREATE OR REPLACE VIEW h AS SELECT * FROM read_parquet(['%s'])" % p)
    q = lambda s: con.execute(s).fetchall()[0][0]
    try:
        N += q("SELECT COUNT(*) FROM h")
        T += q("SELECT COUNT(temperature) FROM h")
        TW += q("SELECT COUNT(wet_bulb_temperature) FROM h")
        RH += q("SELECT COUNT(TRY_CAST(relative_humidity AS DOUBLE)) FROM h")
        TD += q("SELECT COUNT(dew_point_temperature) FROM h")
        TRH += q("""SELECT COUNT(*) FROM h WHERE temperature IS NOT NULL
                    AND TRY_CAST(relative_humidity AS DOUBLE) IS NOT NULL""")
        TTD += q("""SELECT COUNT(*) FROM h WHERE temperature IS NOT NULL
                    AND dew_point_temperature IS NOT NULL""")
    except Exception:
        pass
print(f"  总记录           {N:>12,}")
print(f"  有干球 T         {T:>12,}  ({100*T/max(N,1):.1f}%)")
print(f"  有原生 Tw        {TW:>12,}  ({100*TW/max(N,1):.1f}%)")
print(f"  有相对湿度 RH    {RH:>12,}  ({100*RH/max(N,1):.1f}%)")
print(f"  有露点 Td        {TD:>12,}  ({100*TD/max(N,1):.1f}%)")
print(f"  T 且 RH (可算Tw) {TRH:>12,}  ({100*TRH/max(N,1):.1f}%)")
print(f"  T 且 Td (可算Tw) {TTD:>12,}  ({100*TTD/max(N,1):.1f}%)")
