#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 v3 自算湿球求解器"""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from step5c_metrics_v3 import WetBulbSolver, pressure_from_elev, esat

sv = WetBulbSolver(1013.25)

print("自算湿球 vs 参考值 (海平面气压):")
tests = [
    (30.0, 20.0, 15.6),
    (30.0, 50.0, 21.9),
    (30.0, 80.0, 26.4),
    (35.0, 50.0, 26.0),
    (40.0, 10.0, 21.2),
    (25.0, 100.0, 25.0),
    (20.0, 60.0, 15.3),
    (45.0, 30.0, 29.0),
]
print(f"  {'T(°C)':>7}{'RH(%)':>8}{'自算Tw':>9}{'参考Tw':>9}{'差':>8}")
mx = 0
for T, RH, ref in tests:
    got = sv.from_rh(T, RH)
    d = got - ref
    mx = max(mx, abs(d))
    print(f"  {T:>7.1f}{RH:>8.0f}{got:>9.2f}{ref:>9.2f}{d:>+8.2f}")
print(f"\n  最大偏差 {mx:.2f} °C  (参考值本身有 ±0.3 °C 读取误差)")

print("\n饱和态自洽 (RH=100 -> Tw=T):")
for T in [10, 20, 30, 40]:
    print(f"  T={T}  Tw={sv.from_rh(T, 100.0):.4f}")

print("\n单调性 (Tw <= T 且 RH 越高 Tw 越高):")
for T in [25, 35]:
    prev = -99
    ok = True
    for RH in range(10, 101, 10):
        tw = sv.from_rh(T, RH)
        if tw > T + 1e-6 or tw < prev - 1e-6:
            ok = False
        prev = tw
    print(f"  T={T}: {'通过' if ok else '失败'}")

print("\n海拔 -> 本站气压:")
for h in [0, 500, 1500, 3000, 4000]:
    print(f"  {h:>5} m -> {pressure_from_elev(h):>7.1f} hPa")

print("\n记忆化有效性:")
import time
sv2 = WetBulbSolver(1013.25)
t0 = time.time()
n = 0
for i in range(200):
    for j in range(200):
        sv2.from_rh(10 + i * 0.1, 10 + j * 0.45)
        n += 1
dt = time.time() - t0
print(f"  {n:,} 次调用 (含 40,000 个去重组合) 用时 {dt:.2f} 秒")
print(f"  缓存条目 {len(sv2.cache):,}")
sv3 = WetBulbSolver(1013.25)
t0 = time.time()
for i in range(200):
    for j in range(200):
        sv3.from_rh(10 + i * 0.1, 10 + j * 0.45)
print(f"  冷启动同等规模用时 {time.time()-t0:.2f} 秒")
