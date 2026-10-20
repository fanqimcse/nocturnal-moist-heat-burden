#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 3b: 检查湿热关键区（南亚/波斯湾/西非/东南亚/中非）的观测可用性

这些区域正是湿球热应激最极端、人口最密集的地方。
若 GHCNh 在这些区长序列稀缺，必须在论文中明确说明并量化这一偏倚，
而不是假装全球均匀覆盖。
"""
import io, sys, csv, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INV = "ghcn_data/ghcnh-inventory.txt"
STL = "ghcn_data/ghcnh-station-list.txt"

# 关键湿热区 (名称, 纬度范围, 经度范围)
REGIONS = [
    ("南亚 (印度/孟加拉/巴基斯坦/斯里兰卡)", 5, 35, 60, 95),
    ("波斯湾 / 阿拉伯半岛", 12, 32, 34, 60),
    ("西非 / 几内亚湾", 0, 20, -18, 20),
    ("东南亚", -10, 25, 95, 145),
    ("中非 / 刚果盆地", -12, 10, 8, 35),
    ("东亚季风区 (中国东部/朝鲜半岛/日本)", 20, 45, 100, 145),
    ("中美洲 / 加勒比", 5, 30, -105, -60),
    ("南美北部 (亚马逊)", -15, 10, -80, -35),
]


def load():
    st = {}
    for l in open(STL, encoding="utf-8", errors="replace"):
        if len(l) < 80:
            continue
        sid = l[0:11].strip()
        try:
            st[sid] = dict(lat=float(l[12:20]), lon=float(l[21:30]),
                           name=l[41:71].strip(), country=sid[:2])
        except ValueError:
            pass
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
    return st, inv


def main():
    st, inv = load()
    print(f"{'区域':<34}{'总站点':>7}{'有记录':>8}{'>=30y':>7}{'>=20y':>7}{'>=10y':>7}"
          f"{'可用(20y/4k)':>12}")
    print("-" * 90)
    picks = {}
    for name, la0, la1, lo0, lo1 in REGIONS:
        tot_s = 0; has = 0; c30 = 0; c20 = 0; c10 = 0; usable = []
        for sid, m in st.items():
            if not (la0 <= m["lat"] <= la1 and lo0 <= m["lon"] <= lo1):
                continue
            tot_s += 1
            yrs = [y for y, v in inv.get(sid, {}).items() if v >= 4000]
            if not yrs:
                continue
            has += 1
            n = len(yrs)
            if n >= 30: c30 += 1
            if n >= 20: c20 += 1
            if n >= 10: c10 += 1
            if n >= 20 and min(yrs) <= 1995 and max(yrs) >= 2015:
                usable.append((sid, m, n, min(yrs), max(yrs)))
        picks[name] = usable
        print(f"{name:<34}{tot_s:>7}{has:>8}{c30:>7}{c20:>7}{c10:>7}{len(usable):>12}")

    print("\n各区域可用长序列站点样例 (最多 8 个):")
    for name, u in picks.items():
        if not u:
            continue
        print(f"\n  【{name}】共 {len(u)} 站")
        u.sort(key=lambda x: -x[2])
        for sid, m, n, y0, y1 in u[:8]:
            print(f"    {sid:<12}{m['name'][:22]:<24}{m['country']:<3}"
                  f" {m['lat']:>6.1f} {m['lon']:>7.1f}  {n:>3}年 ({y0}-{y1})")

    # 汇总可补入面板的站点
    with open("panel_supplement.txt", "w", encoding="utf-8") as f:
        for name, u in picks.items():
            for sid, m, n, y0, y1 in u:
                f.write(f"{sid}\t{m['name']}\t{m['country']}\t{m['lat']}\t{m['lon']}\t{n}\t{y0}\t{y1}\n")
    print("\n已写出 panel_supplement.txt")


if __name__ == "__main__":
    main()
