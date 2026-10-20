#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 17: 可业务化的夜间热预警指标（回应 R6-7）

审稿人指出: 从"发现一个现象"到"改变一种实践", 需要一个业务可直接采用的指标。

定义 **NHRI（夜间热恢复指数, Nighttime Heat Recovery Index）**:

    NHRI = Σ_{t ∈ 夜间小时} max(0, Tw(t) − 21.9) Δt     单位: °C·小时

即"夜间累积的、人体靠蒸发散热无法排掉的湿球热负荷"。
它是《柳叶刀-星球健康》东亚 28 城研究所用"热夜强度"
(暖季夜间超过阈值的最低气温小时累积量) 的**湿球版本**,
因此与已发表的健康证据同构, 可直接对接流行病学文献。

业务含义:
  - NHRI = 0        该夜人体可完成降温恢复
  - NHRI 越大        夜间热负荷越重, 恢复越不完全
  - 只需逐小时气温与相对湿度, 任何气象业务系统都能实时计算

本脚本:
  1) 计算各站当代 (2019-2023) 的 NHRI 年总量与每夜均值
  2) 给出分级阈值
  3) 与现行主流指标(年最高干球气温)排名对比, 找出**分歧最大的站点**
     —— 这些正是现行预警体系会漏掉的城市

输出: operational_index.txt
"""
import sys, os, csv, collections, math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import median, theil_sen, sign_test

TH = 21.9


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"),
                                           encoding="utf-8"))
            if r.get("ok_trend") == "1"]
    panel = {}
    pp = os.path.join(HERE, "station_population.csv")
    if os.path.exists(pp):
        for r in csv.DictReader(open(pp, encoding="utf-8")):
            panel[r["station"]] = r
    pop = {k: float(v["pop_50km"]) for k, v in panel.items()}
    name = {k: v["name"] for k, v in panel.items()}
    ctry = {k: v["country"] for k, v in panel.items()}

    # ---- 当代 NHRI ----
    cur = collections.defaultdict(list)
    for r in rows:
        if 2019 <= int(r["year"]) <= 2023:
            try:
                nrd = float(r["nrd_elderly"]); nh = float(r["n_night_hours"])
            except Exception:
                continue
            cur[r["station"]].append(nrd * nh)      # °C·小时/年
    nhs = {s: median(v) for s, v in cur.items() if len(v) >= 2}
    if not nhs:
        print("数据不足")
        return
    vals = sorted(nhs.values())

    def q(p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    print("=" * 92)
    print("【业务指标】NHRI = 夜间累积湿球超阈量 (°C·小时/年), 阈值 21.9 °C")
    print("=" * 92)
    print(f"  站数 {len(nhs)}, 覆盖人口 {sum(pop.get(s,0) for s in nhs)/1e8:.2f} 亿")
    print(f"\n  当代 (2019-2023) NHRI 分布:")
    print(f"    中位 {median(vals):>9.0f}")
    print(f"    四分位 [{q(0.25):.0f}, {q(0.75):.0f}]")
    print(f"    90 分位 {q(0.90):.0f},  99 分位 {q(0.99):.0f},  最大 {vals[-1]:.0f}")
    print(f"\n  建议业务分级 (按 NHRI 非零站的分位数切分):")
    nz = sorted(v for v in vals if v > 0)
    z = len(vals) - len(nz)
    print(f"    （{z} 个站 NHRI 恒为 0，即整年夜间均可完成恢复）")

    def qn(p):
        return nz[min(len(nz) - 1, int(p * len(nz)))]
    for lab, a, b in [("II 级 轻度未恢复", 0, qn(0.33)),
                      ("III 级 中度未恢复", qn(0.33), qn(0.66)),
                      ("IV 级 重度未恢复", qn(0.66), qn(0.90)),
                      ("V 级 极端未恢复", qn(0.90), 1e18)]:
        print(f"    {lab:<18} NHRI ∈ [{a:.0f}, {b:.0f}) °C·h/年")

    # ---- 与现行指标的分歧 ----
    tmax = collections.defaultdict(list)
    for r in rows:
        if 2019 <= int(r["year"]) <= 2023:
            try:
                tmax[r["station"]].append(float(r["t_max"]))
            except Exception:
                pass
    tm = {s: median(v) for s, v in tmax.items() if len(v) >= 2 and s in nhs}
    common = sorted(tm)
    if len(common) < 20:
        print("  对比样本不足")
        return
    rn = {s: i for i, s in enumerate(sorted(common, key=lambda s: -nhs[s]), 1)}
    rt = {s: i for i, s in enumerate(sorted(common, key=lambda s: -tm[s]), 1)}
    n = len(common)
    # rt 大 = 日间最高温排名靠后(现行指标认为不热); rn 小 = NHRI 排名靠前(夜间负荷重)
    # 二者之差越大 = 现行指标越会漏掉
    gap = sorted(common, key=lambda s: -(rt[s] - rn[s]))
    big = sum(1 for s in common if abs(rt[s] - rn[s]) >= n * 0.25)

    print("\n" + "=" * 92)
    print(f"【分歧分析】按 NHRI 排名 vs 按现行指标(年最高干球气温)排名, 共 {n} 站")
    print("=" * 92)
    print(f"  排名位移 ≥ 25% 的站点: {big}/{n} = {100*big/n:.0f}%")
    print(f"\n  现行指标认为不严重、但夜间恢复最差的 12 个站点:")
    print(f"  {'站点':<13}{'名称':<22}{'国':<4}{'最高干球':>9}{'干球排名':>9}"
          f"{'NHRI':>10}{'NHRI排名':>9}{'位移':>7}{'人口(万)':>10}")
    print("  " + "-" * 88)
    for s in gap[:12]:
        print(f"  {s:<13}{name.get(s,'')[:20]:<22}{ctry.get(s,''):<4}"
              f"{tm[s]:>9.1f}{rt[s]:>9}{nhs[s]:>10.0f}{rn[s]:>9}"
              f"{rt[s]-rn[s]:>7}{pop.get(s,0)/1e4:>10.0f}")

    print(f"\n  含义: 这些城市的日间最高气温并不突出, 现行以日间最高温/热指数")
    print(f"        为判据的预警体系不会发出警报, 但其夜间湿球热负荷位居全球前列。")

    # ---- 趋势 ----
    print("\n" + "=" * 92)
    print("【NHRI 趋势】")
    print("=" * 92)
    seq = collections.defaultdict(dict)
    for r in rows:
        try:
            seq[r["station"]][int(r["year"])] = float(r["nrd_elderly"]) * \
                float(r["n_night_hours"])
        except Exception:
            pass
    sl = []
    for s, d in seq.items():
        ys = sorted(d)
        if len(ys) >= 8:
            b = theil_sen([float(y) for y in ys], [d[y] for y in ys])
            if b is not None:
                sl.append(b)
    pos = sum(1 for v in sl if v > 0); neg = sum(1 for v in sl if v < 0)
    print(f"  站数 {len(sl)}, 中位 {median(sl):+.1f} °C·h/年², "
          f"正/负 {pos}/{neg}, 符号检验 p = {sign_test(pos,neg):.2e}")


if __name__ == "__main__":
    main()
