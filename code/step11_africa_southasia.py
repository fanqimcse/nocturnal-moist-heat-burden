#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 11: 补齐致命的站网空白 —— 非洲与南亚

第 5 轮审稿人指出: 现有 72 站中非洲 0 站、南亚 0 站,
而这两地恰好是全球夜间湿润热负担最重、人口最密集的区域。
本脚本从 GHCNh 全站目录中筛选非洲/南亚的长序列候选站,
生成下载任务并输出站表。
"""
import csv, os, sys, json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = [1976, 1977, 1978, 1979, 1980,
         2000, 2001, 2002, 2003, 2004,
         2019, 2020, 2021, 2022, 2023]

# ---- 地理分区 (排除地中海欧洲属地, 只保留真正的非洲/南亚) ----
EURO_CODES = {"SP", "GR", "PO", "IT", "MT", "FR", "CY", "GI"}


def region_of(lat, lon, code):
    if code in EURO_CODES:
        return None
    # 南亚: 印度次大陆 + 阿富汗/乌兹别克(中亚边缘)
    if 5.0 <= lat <= 38.0 and 60.0 <= lon <= 92.0:
        return "south_asia"
    # 撒哈拉以南非洲
    if -35.0 <= lat <= 15.0 and -18.0 <= lon <= 52.0:
        return "sub_saharan_africa"
    # 北非与萨赫勒
    if 15.0 < lat <= 37.0 and -18.0 <= lon <= 52.0:
        return "north_africa"
    return None


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "station_panel.csv"),
                                    encoding="utf-8")))
    cand = []
    for r in rows:
        lat = float(r["lat"]); lon = float(r["lon"])
        code = r["country"][:2].upper()
        g = region_of(lat, lon, code)
        if not g:
            continue
        first = int(r["first"]); last = int(r["last"]); ny = int(r["nyears"])
        # 必须跨越三个时相窗口
        if first > 1980 or last < 2023:
            continue
        # 记录密度太低的站无法支撑亚小时分析
        if ny < 40:
            continue
        cand.append({
            "station": r["station"], "name": r["name"], "country": r["country"],
            "lat": lat, "lon": lon, "elev": r["elev"],
            "region": g, "nyears": ny,
            "yr_E1": ",".join(str(y) for y in YEARS[0:5]),
            "yr_E2": ",".join(str(y) for y in YEARS[5:10]),
            "yr_E3": ",".join(str(y) for y in YEARS[10:15]),
        })

    # 按区域内记录年数排序, 优先下载最有希望的站
    cand.sort(key=lambda d: -d["nyears"])
    json.dump(cand, open(os.path.join(HERE, "panel_afrsa.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(os.path.join(HERE, "jobs_afrsa.txt"), "w", encoding="utf-8") as f:
        for c in cand:
            for y in YEARS:
                f.write(f"{c['station']} {y}\n")

    import collections
    cnt = collections.Counter(c["region"] for c in cand)
    print(f"候选站 {len(cand)} 个, 下载任务 {len(cand) * len(YEARS)} 条")
    for k, v in cnt.most_common():
        print(f"  {k}: {v} 站")
    print("\n南亚候选:")
    for c in cand:
        if c["region"] == "south_asia":
            print(f"  {c['station']} {c['name'][:28]:30s} "
                  f"{c['country']} {c['lat']:7.2f},{c['lon']:8.2f} {c['nyears']}y")
    print("\n撒哈拉以南非洲候选 (前 25):")
    n = 0
    for c in cand:
        if c["region"] == "sub_saharan_africa":
            print(f"  {c['station']} {c['name'][:28]:30s} "
                  f"{c['country']} {c['lat']:7.2f},{c['lon']:8.2f} {c['nyears']}y")
            n += 1
            if n >= 25:
                break


if __name__ == "__main__":
    main()
