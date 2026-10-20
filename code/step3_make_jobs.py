#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 3: 生成下载任务清单

设计要点（可直接写进论文方法）:
  - 三时相窗口: 1976-1980 / 2000-2004 / 2019-2023，跨约 45 年
  - 每窗口取最多 5 年，用多年平均抑制 ENSO 等年际振荡对趋势的污染
  - 站点按纬度带分层抽样，避免面板被美国温带站主导
  - 每站每窗口至少 3 个有效年，否则剔除（保证统计量可比）
  - 年观测数门槛 5000 条 (约 14 条/日)，保证昼夜统计可靠
"""
import io, sys, csv, collections, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

random.seed(20261018)

INV = "ghcn_data/ghcnh-inventory.txt"
PANEL = "station_panel.csv"

WINDOWS = [
    ("E1", 1976, 1980),
    ("E2", 2000, 2004),
    ("E3", 2019, 2023),
]
MIN_REC = 5000      # 单站单年最少观测数
MIN_YRS = 3         # 每个窗口最少有效年数

# 纬度带分层目标站数（核心面板，用于趋势）
TARGETS = [
    ("tropics", 0.0, 23.5, 34),
    ("subtropics", 23.5, 35.0, 36),
    ("temperate", 35.0, 50.0, 24),
    ("boreal", 50.0, 90.0, 8),
]


def load_inv():
    inv = collections.defaultdict(dict)
    with open(INV, encoding="utf-8", errors="replace") as f:
        next(f)
        for l in f:
            if len(l) < 30:
                continue
            sid = l[0:11].strip()
            try:
                yr = int(l[11:17])
            except ValueError:
                continue
            tot = 0
            for i in range(12):
                p = 17 + i * 6
                try:
                    tot += int(l[p:p + 6])
                except ValueError:
                    pass
            inv[sid][yr] = tot
    return inv


def band_of(lat):
    a = abs(lat)
    for name, lo, hi, _ in TARGETS:
        if lo <= a < hi:
            return name
    return "boreal"


def main():
    inv = load_inv()
    rows = list(csv.DictReader(open(PANEL, encoding="utf-8")))
    for r in rows:
        r["lat"] = float(r["lat"])
        r["lon"] = float(r["lon"])
        r["band"] = band_of(r["lat"])

    by_band = collections.defaultdict(list)
    for r in rows:
        by_band[r["band"]].append(r)

    selected = []
    for name, lo, hi, want in TARGETS:
        cand = by_band.get(name, [])
        # 优先记录年份多、年均记录多的站
        cand.sort(key=lambda r: (-int(r["nyears"]), -float(r["mean_rec"])))
        picked = []
        for r in cand:
            sid = r["station"]
            yrs = {}
            ok = True
            for tag, y0, y1 in WINDOWS:
                good = [y for y in range(y0, y1 + 1) if inv[sid].get(y, 0) >= MIN_REC]
                if len(good) < MIN_YRS:
                    ok = False
                    break
                yrs[tag] = sorted(good)[-5:]   # 取窗口内最好的 5 年
            if not ok:
                continue
            r["_yrs"] = yrs
            picked.append(r)
            if len(picked) >= want:
                break
        print(f"  {name:<12} 纬度 {lo:>4.1f}-{hi:<4.1f} 目标 {want:>3}  实选 {len(picked):>3}")
        selected.extend(picked)

    print(f"\n核心面板共 {len(selected)} 站")

    jobs = []
    for r in selected:
        for tag, _, _ in WINDOWS:
            for y in r["_yrs"][tag]:
                jobs.append((r["station"], y))
    print(f"下载任务 {len(jobs)} 个站年 (每站约 {len(jobs)/max(len(selected),1):.1f} 年)")

    with open("jobs_core.txt", "w", encoding="utf-8") as f:
        for s, y in jobs:
            f.write(f"{s} {y}\n")

    meta = [dict(station=r["station"], name=r["name"], country=r["country"],
                 lat=r["lat"], lon=r["lon"], elev=r["elev"], band=r["band"],
                 nyears=int(r["nyears"]), first=int(r["first"]), last=int(r["last"]),
                 **{f"yr_{t}": ",".join(str(x) for x in r["_yrs"][t])
                    for t, _, _ in WINDOWS})
            for r in selected]
    with open("panel_core.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("已写出 jobs_core.txt 与 panel_core.json")

    cc = collections.Counter(r["country"] for r in selected)
    print(f"涉及国家/地区 {len(cc)} 个, TOP10: {cc.most_common(10)}")


if __name__ == "__main__":
    main()
