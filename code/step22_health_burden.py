#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R7-6（回应 M39）: 人·夜暴露量与归因负担量级

分两层, 严格区分:

  【第一层 严格观测】人·夜暴露量
    逐站: 热夜数 = fn_tnXX * n_valid_nights
    人·夜 = 站点人口 * 热夜数
    这一时期对比 (1976-1980 vs 2019-2023), 人口加权。
    这是本研究自己的数据, 无外推。

  【第二层 文献外推】归因死亡量级
    用已发表的暴露-反应关系, 不是本研究建立的剂量-反应:
      RR_low  = 1.09  Kim & Hashizume 2023 (日本全国, 2472 万例死亡, 1973-2015)
                      夜间最低温 >=25 C 热夜, 独立于日均温, 全因死亡 RR = 1.09-1.10
      RR_high = 1.50  复旦大学 / Lancet Planet Health (东亚 28 城)
                      控制日均温后, 热夜暴露使死亡风险增加 40-50%
    归因分数 AF = Pexp*(RR-1) / (1 + Pexp*(RR-1)),  Pexp = 暴露人·夜 / 总人·夜
    基线死亡率取 7/1000/年 (全球粗死亡率量级)
    *** 结果仅作量级参考, 不确定性 >=50%, 不得作为健康影响的精确估计 ***
"""
import sys, os, csv, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import median

E1 = (1976, 1980)
E3 = (2019, 2023)
CDR = 7.0 / 1000.0           # 基线粗死亡率
RRs = [("低 (Kim & Hashizume 2023, RR=1.09)", 1.09),
       ("高 (复旦大学/柳叶刀东亚28城, RR=1.50)", 1.50)]


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"),
                                           encoding="utf-8")) if r.get("ok_trend") == "1"]
    pop = {r["station"]: float(r["pop_50km"]) for r in
           csv.DictReader(open(os.path.join(HERE, "station_population.csv"), encoding="utf-8"))}
    return rows, pop


def paired_stations(rows, y0, y1, minyr=3):
    """两期都有 >=minyr 个合格年的站集, 用于消除站网构成变化的污染。"""
    c1 = collections.Counter(); c3 = collections.Counter()
    for r in rows:
        y = int(r["year"])
        if y0[0] <= y <= y0[1]:
            c1[r["station"]] += 1
        if y1[0] <= y <= y1[1]:
            c3[r["station"]] += 1
    return sorted(s for s in c1 if c1[s] >= minyr and c3[s] >= minyr)


def epoch_pn(rows, pop, key, y0, y1, only=None):
    """返回 (年化人·夜, 年化人·夜基数, 站数)
    先逐站对本站有数据的年份取年均值, 再跨站求和 —— 避免各站年数不同造成的偏差。"""
    per_ex = collections.defaultdict(list)   # st -> [该年 人·夜]
    per_base = collections.defaultdict(list)
    for r in rows:
        y = int(r["year"])
        if not y0 <= y <= y1:
            continue
        st = r["station"]
        if only is not None and st not in only:
            continue
        w = pop.get(st, 0.0)
        if w <= 0:
            continue
        f = r.get(key)
        nv = r.get("n_valid_nights")
        if f in (None, "", "None") or nv in (None, "", "None"):
            continue
        try:
            f = float(f); nv = float(nv)
        except Exception:
            continue
        per_ex[st].append(w * f * nv)
        per_base[st].append(w * nv)
    tot_ex = sum(sum(v) / len(v) for v in per_ex.values())
    tot_base = sum(sum(v) / len(v) for v in per_base.values())
    return tot_ex, tot_base, len(per_ex)


def main():
    rows, pop = load()
    sts_all = set(r["station"] for r in rows)
    tot_pop = sum(pop.get(s, 0.0) for s in sts_all)
    print("=" * 100)
    print("R7-6 / M39  人·夜暴露量与归因负担")
    print(f"  参与站 {len(sts_all)}, 站域人口合计 {tot_pop/1e8:.2f} 亿")
    print("=" * 100)

    pair = paired_stations(rows, E1, E3)
    ppop = sum(pop.get(s, 0.0) for s in pair)
    print(f"  配对站集 (两期均 >=3 合格年): {len(pair)} 站, 人口 {ppop/1e8:.2f} 亿")
    print("  —— 以下全部使用配对站集, 以消除站网构成变化造成的假信号 ——")

    print(f"\n【第一层 · 严格观测】人·夜暴露量 (人·夜/年, 配对站集合计)")
    print(f"{'指标':<34}{'1976-1980':>18}{'2019-2023':>18}{'变化':>12}{'相对':>10}")
    print("-" * 100)
    first = {}
    for key, lab in [("fn_tn20", "热夜 TN>=20 °C"),
                     ("fn_tn24", "热夜 TN>=24 °C (巴塞罗那夜间热浪阈值)"),
                     ("fn_tn25", "热夜 TN>=25 °C (日本全国死亡风险阈值)"),
                     ("nrfr_elderly", "夜间恢复失效 (湿球未降至 21.9 °C)")]:
        a, ab, na = epoch_pn(rows, pop, key, *E1, only=set(pair))
        b, bb, nb = epoch_pn(rows, pop, key, *E3, only=set(pair))
        if not a or not b:
            continue
        first[key] = (a, b, ab, bb)
        rel = (b - a) / a * 100 if a else float("nan")
        print(f"{lab:<34}{a/1e8:>16.2f}亿{b/1e8:>16.2f}亿"
              f"{(b-a)/1e8:>+10.2f}亿{rel:>+9.1f}%")
    print("-" * 100)
    print("  口径: 人·夜 = 站点人口 x 该年满足条件夜数; 两期同站、同人口基数。")

    # 人均夜数
    print(f"\n  人均 (配对站集内每位居民每年经历的夜数):")
    for key, lab in [("fn_tn20", "热夜 TN>=20"), ("fn_tn25", "热夜 TN>=25"),
                     ("nrfr_elderly", "夜间恢复失效")]:
        if key not in first:
            continue
        a, b, ab, bb = first[key]
        n0 = a / ppop; n1 = b / ppop
        print(f"    {lab:<26} {n0:>6.1f} 夜/年  ->  {n1:>6.1f} 夜/年 "
              f"({(n1-n0):+.1f} 夜, {(n1/n0-1)*100:+.1f}%)")

    print(f"\n【第二层 · 文献外推】归因死亡量级 (不确定度 >=50%, 仅供量级参考)")
    print("-" * 100)
    annual_deaths = ppop * CDR
    print(f"  配对站集年死亡基数 (粗死亡率 {CDR*1000:.0f}‰): {annual_deaths/1e4:.1f} 万/年")
    for key, lab in [("fn_tn25", "TN>=25 °C 热夜"),
                     ("nrfr_elderly", "夜间恢复失效夜")]:
        if key not in first:
            continue
        a, b, ab, bb = first[key]
        p0 = a / ab; p1 = b / bb
        print(f"\n  暴露指标: {lab}")
        print(f"    暴露人·夜占比: {p0*100:.2f}%  ->  {p1*100:.2f}%")
        for rlab, rr in RRs:
            af0 = p0 * (rr - 1) / (1 + p0 * (rr - 1))
            af1 = p1 * (rr - 1) / (1 + p1 * (rr - 1))
            d0 = af0 * annual_deaths; d1 = af1 * annual_deaths
            print(f"    {rlab}")
            print(f"      归因分数 AF: {af0*100:.3f}% -> {af1*100:.3f}%")
            print(f"      归因死亡:   {d0/1e4:>7.2f} 万/年 -> {d1/1e4:>7.2f} 万/年 "
                  f"(净增 {(d1-d0)/1e4:+.2f} 万/年, {(d1/d0-1)*100:+.0f}%)")
    print("-" * 100)
    print("  *** 第二层不是本研究建立的剂量-反应关系, 是引用已发表 RR 的量级外推。")
    print("  *** 面板仅覆盖站点周边城市人口, 不代表全球; 不得外推为全球归因死亡。")


if __name__ == "__main__":
    main()
