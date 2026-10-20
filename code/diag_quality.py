#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 v2 质量门槛的淘汰模式：被剔除的是哪些年、哪些站？"""
import sys, csv, collections
sys.stdout.reconfigure(encoding="utf-8")
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

rows = list(csv.DictReader(open("metrics_v2.csv", encoding="utf-8")))
for r in rows:
    r["year"] = int(r["year"])
    r["coverage"] = float(r["coverage"])
    r["ok"] = int(float(r.get("ok_trend") or 0))
    r["n_nights"] = int(float(r.get("n_valid_nights") or 0))

print(f"总站年 {len(rows)}, 合格 {sum(r['ok'] for r in rows)} "
      f"({100*sum(r['ok'] for r in rows)/len(rows):.0f}%)")

print("\n--- 按年代窗口 ---")
win = collections.defaultdict(lambda: [0, 0, []])
for r in rows:
    y = r["year"]
    k = "E1 1976-1980" if y <= 1980 else ("E2 2000-2004" if 2000 <= y <= 2004
         else ("E3 2019-2023" if y >= 2019 else f"其他 {y}"))
    win[k][0] += 1
    win[k][1] += r["ok"]
    win[k][2].append(r["coverage"])
for k in sorted(win):
    n, ok, cov = win[k]
    print(f"  {k:<14} {n:>4} 站年  合格 {ok:>4} ({100*ok/n:>3.0f}%)  "
          f"中位覆盖率 {sorted(cov)[len(cov)//2]:.2f}")

print("\n--- 按纬度带 ---")
band = collections.defaultdict(lambda: [0, 0])
for r in rows:
    b = r.get("band") or "?"
    band[b][0] += 1
    band[b][1] += r["ok"]
for b, (n, ok) in sorted(band.items()):
    print(f"  {b:<12} {n:>4} 合格 {ok:>4} ({100*ok/max(n,1):>3.0f}%)")

print("\n--- 不合格记录的原因分布 ---")
bad_cov = sum(1 for r in rows if not r["ok"] and r["coverage"] < 0.60)
bad_nig = sum(1 for r in rows if not r["ok"] and r["coverage"] >= 0.60)
print(f"  覆盖率 < 0.60            : {bad_cov}")
print(f"  覆盖率达标但有效夜数 < 250: {bad_nig}")

print("\n--- 完全没有任何合格年份的站点 ---")
bys = collections.defaultdict(list)
for r in rows:
    bys[r["station"]].append(r)
none_ok = [s for s, v in bys.items() if not any(x["ok"] for x in v)]
print(f"  {len(none_ok)} / {len(bys)} 站")
for s in none_ok[:25]:
    v = bys[s]
    print(f"    {s:<12}{v[0].get('name','')[:20]:<22}"
          f" 年数 {len(v)}  中位覆盖率 "
          f"{sorted(x['coverage'] for x in v)[len(v)//2]:.2f}")

print("\n--- 覆盖率最高的年份 (说明观测加密的时间点) ---")
yc = collections.defaultdict(list)
for r in rows:
    yc[r["year"]].append(r["coverage"])
for y in sorted(yc):
    v = yc[y]
    print(f"  {y}  n={len(v):>3}  中位覆盖率 {sorted(v)[len(v)//2]:.2f}")
