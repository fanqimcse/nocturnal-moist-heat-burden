#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 12: 合并站点元数据 —— 把非洲/南亚候选站并入主站表,
并为所有站点补齐纬度带 (band) 与地理分区 (region)。

分区用于第 5 轮审稿人 M22 要求的站网代表性核算,
以及 M23 要求的区域块 bootstrap 分组。
"""
import csv, os, sys, json, math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

EURO = {"SP", "GR", "PO", "IT", "MT", "FR", "CY", "GI", "IC"}


def band_of(lat):
    a = abs(lat)
    if a <= 23.5:
        return "tropics"
    if a <= 35.0:
        return "subtropics"
    return "temperate"


def region_of(lat, lon, code=""):
    code = (code or "")[:2].upper()
    if 5.0 <= lat <= 38.0 and 60.0 <= lon <= 92.0:
        return "南亚"
    if -35.0 <= lat <= 15.0 and -18.0 <= lon <= 52.0:
        return "撒哈拉以南非洲"
    if 15.0 < lat <= 37.0 and -18.0 <= lon <= 52.0:
        return None if code in EURO else "北非与萨赫勒"
    if 12.0 <= lat <= 42.0 and 32.0 <= lon <= 60.0:
        return "西亚与波斯湾"
    if -11.0 <= lat <= 25.0 and 92.0 <= lon <= 141.0:
        return "东南亚"
    if 20.0 <= lat <= 50.0 and 100.0 <= lon <= 146.0:
        return "东亚"
    if 14.0 <= lat <= 33.0 and -118.0 <= lon <= -86.0:
        return "中美洲与加勒比"
    if -60.0 <= lat < 15.0 and -82.0 <= lon <= -34.0:
        return "南美"
    if -50.0 <= lat <= -10.0 and 110.0 <= lon <= 180.0:
        return "大洋洲"
    if 25.0 <= lat <= 72.0 and -170.0 <= lon <= -50.0:
        return "北美"
    if 36.0 <= lat <= 72.0 and -25.0 <= lon <= 50.0:
        return "欧洲"
    return "其他"


def main():
    main_path = os.path.join(HERE, "panel_all.json")
    base = json.load(open(main_path, encoding="utf-8"))
    idx = {m["station"]: m for m in base}

    afr = os.path.join(HERE, "panel_afrsa.json")
    added = 0
    if os.path.exists(afr):
        for m in json.load(open(afr, encoding="utf-8")):
            if m["station"] in idx:
                continue
            m.setdefault("band", band_of(m["lat"]))
            m.setdefault("region", region_of(m["lat"], m["lon"], m.get("country", "")))
            m["elev"] = float(m.get("elev", 0) or 0)
            base.append(m); idx[m["station"]] = m; added += 1

    # 为全部站点补齐 band / region
    for m in base:
        lat = float(m["lat"]); lon = float(m["lon"])
        if not m.get("band"):
            m["band"] = band_of(lat)
        if not m.get("region"):
            m["region"] = region_of(lat, lon, m.get("country", ""))

    json.dump(base, open(main_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"站表 {len(base)} 个站点 (新增非洲/南亚 {added})")

    # 已下载数据的站点分区统计
    have = {}
    for f in os.listdir(os.path.join(HERE, "ghcn_data")):
        if f.endswith(".parquet"):
            p = f[:-8].split("_")
            have.setdefault(p[1], set()).add(int(p[2]))
    cnt = {}
    for st, yrs in have.items():
        m = idx.get(st, {})
        g = m.get("region") or "未分类"
        cnt.setdefault(g, []).append(len(yrs))
    print("\n已下载站年按分区:")
    for g, v in sorted(cnt.items(), key=lambda x: -len(x[1])):
        print(f"  {g:<16} {len(v):>4} 站, 中位 {sorted(v)[len(v)//2]} 年")


if __name__ == "__main__":
    main()
