#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 7: 人口暴露估算

用 GeoNames cities5000（人口 >= 5000 的城市点，含经纬度与人口数）
为每个观测站计算邻近人口，从而把"站点上的物理量"转换成"多少人受影响"。

方法：对每个站点，累加半径 R 内所有城市的人口，按距离权重衰减：
    P(R) = Σ_i  pop_i * w(d_i),   w = 1 for d<=R0,  exp(-(d-R0)/L) for d>R0
这样避免硬边界导致的人为跳变。

局限（必须在论文中声明）：
  - GeoNames 只含城市点，系统性低估农村人口；
  - 但热应激的健康负担主要集中于城市，城市人口暴露是相关量；
  - 站点多为机场，位于城市边缘，R0 取 50 km 是折中。
"""
import sys, os, math, json, csv
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CITIES = "ext/cities5000.txt"
R0 = 50.0      # km 内全额计入
RMAX = 300.0   # km 以外忽略
L = 100.0      # km 衰减尺度


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def load_cities(path=CITIES):
    out = []
    if not os.path.exists(path):
        print("缺少", path)
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 15:
            continue
        try:
            lat = float(f[4]); lon = float(f[5]); pop = int(f[14])
        except (ValueError, IndexError):
            continue
        if pop <= 0:
            continue
        out.append((lat, lon, pop, f[1] if len(f) > 1 else ""))
    return out


def station_population(lat, lon, cities):
    tot = 0.0
    for cla, clo, pop, _ in cities:
        d = haversine(lat, lon, cla, clo)
        if d > RMAX:
            continue
        w = 1.0 if d <= R0 else math.exp(-(d - R0) / L)
        tot += pop * w
    return tot


def main():
    cities = load_cities()
    if not cities:
        return
    print(f"载入城市 {len(cities):,} 个, 总人口 {sum(c[2] for c in cities):,}")

    panel = json.load(open("panel_all.json", encoding="utf-8"))
    print(f"面板站点 {len(panel)} 个")

    res = []
    for m in panel:
        p = station_population(m["lat"], m["lon"], cities)
        res.append(dict(station=m["station"], name=m["name"], country=m["country"],
                        lat=m["lat"], lon=m["lon"], band=m.get("band", ""),
                        region=m.get("region", ""), pop_50km=p))
    res.sort(key=lambda x: -x["pop_50km"])
    with open("station_population.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\n已写出 station_population.csv")
    print(f"\n邻近人口最多的 20 个站点 (50km 全额 + 指数衰减至 300km):")
    print(f"  {'站点':<12}{'名称':<24}{'国':<4}{'纬度':>7}{'经度':>8}{'暴露人口':>14}")
    for r in res[:20]:
        print(f"  {r['station']:<12}{r['name'][:22]:<24}{r['country']:<4}"
              f"{r['lat']:>7.1f}{r['lon']:>8.1f}{r['pop_50km']:>14,.0f}")

    tot = sum(r["pop_50km"] for r in res)
    print(f"\n面板覆盖的邻近人口合计 {tot:,.0f}")


if __name__ == "__main__":
    main()
