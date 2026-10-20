#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 3c: 生成补充面板任务（湿热关键区）

核心面板按纬度带抽样，结果南亚/波斯湾/西非等"湿球热应激最极端、人口最密集"
区域代表性不足 —— 这是审稿人一定会攻击的点。因此按区域定向补充。

这些站点多从 1970s-80s 才有稳定逐小时记录，因此采用**自适应时相窗口**:
  取该站有效年份序列的 前5年 / 中5年 / 后5年，而不是固定日历年份。
  这样每站都能给出跨约 40 年的三个时相，最大化可用信息。
"""
import io, sys, csv, collections, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

INV = "ghcn_data/ghcnh-inventory.txt"
MIN_REC = 4000
MIN_YRS = 20          # 该站有效年份总数下限
REGION_MAX = {        # 各区域最多取多少站（按有效年数排序）
    "波斯湾": 16,
    "南亚": 6,
    "东南亚": 12,
    "东亚": 10,
    "西非": 6,
    "中非": 4,
}
REGIONS = [
    ("南亚", 5, 35, 60, 95),
    ("波斯湾", 12, 32, 34, 60),
    ("西非", 0, 20, -18, 20),
    ("东南亚", -10, 25, 95, 145),
    ("中非", -12, 10, 8, 35),
    ("东亚", 20, 45, 100, 145),
]
CORE = set()


def load():
    st = {}
    for l in open("ghcn_data/ghcnh-station-list.txt", encoding="utf-8", errors="replace"):
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


def band_of(lat):
    a = abs(lat)
    if a < 23.5: return "tropics"
    if a < 35: return "subtropics"
    if a < 50: return "temperate"
    return "boreal"


def main():
    try:
        for m in json.load(open("panel_core.json", encoding="utf-8")):
            CORE.add(m["station"])
    except Exception:
        pass

    st, inv = load()
    out_meta, jobs = [], []
    for name, la0, la1, lo0, lo1 in REGIONS:
        cand = []
        for sid, m in st.items():
            if sid in CORE:
                continue
            if not (la0 <= m["lat"] <= la1 and lo0 <= m["lon"] <= lo1):
                continue
            yrs = sorted([y for y, v in inv.get(sid, {}).items() if v >= MIN_REC])
            if len(yrs) < MIN_YRS:
                continue
            if yrs[0] > 1995:
                continue
            cand.append((sid, m, yrs))
        cand.sort(key=lambda x: -len(x[2]))
        pick = cand[:REGION_MAX.get(name, 6)]
        print(f"  {name:<8} 候选 {len(cand):>4}  取 {len(pick):>3}")
        for sid, m, yrs in pick:
            e1 = yrs[:5]
            mid = len(yrs) // 2
            e2 = yrs[max(0, mid - 2):mid + 3]
            e3 = yrs[-5:]
            sel = sorted(set(e1 + e2 + e3))
            for y in sel:
                jobs.append((sid, y))
            out_meta.append(dict(station=sid, name=m["name"], country=m["country"],
                                 lat=m["lat"], lon=m["lon"], band=band_of(m["lat"]),
                                 region=name, nyears=len(yrs),
                                 first=yrs[0], last=yrs[-1],
                                 yr_E1=",".join(map(str, e1)),
                                 yr_E2=",".join(map(str, e2)),
                                 yr_E3=",".join(map(str, e3))))

    seen, uniq = set(), []
    for j in jobs:
        if j not in seen:
            seen.add(j); uniq.append(j)
    print(f"\n补充站点 {len(out_meta)} 个, 下载任务 {len(uniq)} 个站年")
    with open("jobs_supp.txt", "w", encoding="utf-8") as f:
        for s, y in uniq:
            f.write(f"{s} {y}\n")
    with open("panel_supp.json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=1)
    print("已写出 jobs_supp.txt / panel_supp.json")


if __name__ == "__main__":
    main()
