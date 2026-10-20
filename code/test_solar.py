#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验太阳高度角算法与昼夜判据"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from step5_metrics import solar_elevation, ymd_to_doy

print("太阳高度角算法校验 (理论值 vs 计算值):")
tests = [
    ('赤道春分正午', 0.0, 0.0, 80, 12.0, 90.0),
    ('北回归线夏至正午', 23.44, 0.0, 172, 12.0, 90.0),
    ('赤道夏至正午', 0.0, 0.0, 172, 12.0, 66.56),
    ('北极圈夏至午夜', 66.56, 0.0, 172, 0.0, 0.0),
    ('赤道秋分正午', 0.0, 0.0, 266, 12.0, 90.0),
    ('北纬40冬至正午', 40.0, 0.0, 355, 12.0, 26.56),
    ('赤道午夜', 0.0, 0.0, 80, 0.0, -90.0),
]
for name, lat, lon, doy, h, exp in tests:
    got = solar_elevation(lat, lon, doy, h)
    print('  %-18s 理论 %7.2f  计算 %7.2f  差 %+6.2f' % (name, exp, got, got - exp))

print()
print("DOY 校验: 2023-01-01 -> %d, 2023-12-31 -> %d, 2024-12-31 -> %d"
      % (ymd_to_doy(2023, 1, 1), ymd_to_doy(2023, 12, 31), ymd_to_doy(2024, 12, 31)))

print()
print("菲尼克斯 (33.4N, -112.0) 2023-07-15 昼夜判据:")
for h in range(24):
    e = solar_elevation(33.4, -112.0, ymd_to_doy(2023, 7, 15), h + 0.5)
    print("   UTC %02d:30  高度角 %7.2f  %s" % (h, e, "夜" if e < -0.833 else "昼"))
print("   注: 菲尼克斯地方时约 UTC-7 -> 当地正午约 19 UTC, 当地日出约 12 UTC(05当地), 日落约 02 UTC(19当地)")
