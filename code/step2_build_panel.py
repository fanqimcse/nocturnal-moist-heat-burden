#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 2: 从 103.8 万行 GHCNh 记录清单中筛选可用于趋势分析的站点面板

筛选逻辑（审稿人会盯的几点）:
  1. 时间跨度足够: 有效年份数 >= MIN_YEARS  (趋势需要样本量)
  2. 观测密度足够: 年均记录数 >= MIN_REC   (太稀的站算不出可靠的日/夜统计)
  3. 起止年份覆盖: 最早年 <= EARLY_MAX 且 最晚年 >= LATE_MIN  (能比早晚两个时段)
  4. 按纬度带分层: 保证热带/副热带/温带都有代表 (湿球热应激主要发生在低纬湿润区)

输出: 候选站点表 (CSV) 供下一步下载
"""
import io, sys, csv, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INV = "ghcn_data/ghcnh-inventory.txt"
STL = "ghcn_data/ghcnh-station-list.txt"

MIN_YEARS = 30     # 至少 30 个有效年份
MIN_REC = 5000     # 年均至少 5000 条观测 (约 14 条/天)
EARLY_MAX = 1985   # 最早有效年 <= 1985
LATE_MIN = 2015    # 最晚有效年 >= 2015


def load_stations():
    """解析 GHCNh 站点清单 (固定宽度)"""
    out = {}
    for l in open(STL, encoding="utf-8", errors="replace"):
        if len(l) < 80:
            continue
        sid = l[0:11].strip()
        try:
            lat = float(l[12:20])
            lon = float(l[21:30])
        except ValueError:
            continue
        try:
            elev = float(l[31:37])
        except ValueError:
            elev = None
        name = l[41:71].strip()
        state = l[38:40].strip()
        out[sid] = dict(lat=lat, lon=lon, elev=elev, name=name, state=state,
                        country=sid[:2])
    return out


def load_inventory():
    """按站点聚合记录清单"""
    st = collections.defaultdict(lambda: dict(years=[], recs=0, byyear={}))
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
            if tot == 0:
                continue
            d = st[sid]
            d["years"].append(yr)
            d["recs"] += tot
            d["byyear"][yr] = tot
    return st


def main():
    stations = load_stations()
    print(f"站点清单: {len(stations):,}")

    inv = load_inventory()
    print(f"有记录的站点: {len(inv):,}")

    rows = []
    for sid, d in inv.items():
        if sid not in stations:
            continue
        ys = sorted(d["years"])
        nyr = len(ys)
        if nyr < MIN_YEARS:
            continue
        if ys[0] > EARLY_MAX or ys[-1] < LATE_MIN:
            continue
        mean_rec = d["recs"] / nyr
        if mean_rec < MIN_REC:
            continue
        m = stations[sid]
        rows.append(dict(
            station=sid, name=m["name"], country=m["country"], state=m["state"],
            lat=m["lat"], lon=m["lon"], elev=m["elev"],
            first=ys[0], last=ys[-1], nyears=nyr,
            mean_rec=mean_rec, total_rec=d["recs"],
            span=ys[-1] - ys[0] + 1,
        ))

    print(f"\n通过筛选 (年份>={MIN_YEARS}, 年均记录>={MIN_REC}, "
          f"起<={EARLY_MAX}, 止>={LATE_MIN}): {len(rows):,}")

    # 地理分布
    cc = collections.Counter(r["country"] for r in rows)
    print(f"国家/地区数: {len(cc)}")
    print("TOP15:", cc.most_common(15))

    # 纬度带分布
    bands = collections.Counter()
    for r in rows:
        a = abs(r["lat"])
        if a <= 23.5:
            b = "热带 (0-23.5)"
        elif a <= 35:
            b = "副热带 (23.5-35)"
        elif a <= 50:
            b = "温带 (35-50)"
        else:
            b = "高纬 (>50)"
        bands[b] += 1
    print("\n纬度带分布:")
    for k in ["热带 (0-23.5)", "副热带 (23.5-35)", "温带 (35-50)", "高纬 (>50)"]:
        print(f"  {k:<20} {bands.get(k,0):,}")

    # 年份覆盖
    print("\n年份覆盖: 最早 %d, 最晚 %d" %
          (min(r["first"] for r in rows), max(r["last"] for r in rows)))
    nspan = collections.Counter(r["span"] for r in rows)
    print("记录跨度分布(前8):", sorted(nspan.items(), reverse=True)[:8])

    rows.sort(key=lambda r: -r["nyears"])
    with open("station_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 station_panel.csv ({len(rows):,} 站)")

    print("\n前 20 个最长序列站点:")
    print(f"  {'站点':<12}{'名称':<26}{'国家':<5}{'纬度':>7}{'经度':>8}"
          f"{'起':>6}{'止':>6}{'年数':>5}{'年均记录':>10}")
    for r in rows[:20]:
        print(f"  {r['station']:<12}{r['name'][:24]:<26}{r['country']:<5}"
              f"{r['lat']:>7.1f}{r['lon']:>8.1f}{r['first']:>6}{r['last']:>6}"
              f"{r['nyears']:>5}{r['mean_rec']:>10,.0f}")


if __name__ == "__main__":
    main()
