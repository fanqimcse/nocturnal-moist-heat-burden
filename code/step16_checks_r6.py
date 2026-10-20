#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 16: 第 6 轮要求的三项检验（M30 / M31 / M32）

  A. 标度稳定性: 分两段(1976-1999, 2000-2023)估计"每 °C 全球增暖"的标度,
     检验线性外推的前提是否成立。
  B. 分样本报告: 核心站 / 非洲南亚站 / 合并, 分别给趋势与显著性。
  C. 人口口径更正: 只对"实际参与分析的站"汇总人口, 重算暴露量。

输出: checks_r6.txt
"""
import sys, os, csv, math, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import theil_sen, mann_kendall, sign_test, median, lag1_autocorr
from step14_attribution import load_gmst, load_series, composite

METRICS = ["tw_night_mean", "dtr_tw", "usable_frac", "nrfr_elderly",
           "nrd_elderly", "relief_elderly", "nhrs_elderly"]
LABEL = {
    "tw_night_mean": "夜间湿球均值 (°C)",
    "dtr_tw": "湿球日较差 (°C)",
    "usable_frac": "可用降温比例",
    "nrfr_elderly": "夜间恢复失效率",
    "nrd_elderly": "夜间恢复亏缺 (°C)",
    "relief_elderly": "每日可恢复窗口 (h)",
    "nhrs_elderly": "夜间超阈等效小时 (h)",
}


def ols(x, y):
    n = len(x)
    if n < 5:
        return None, None
    mx = sum(x) / n; my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None, None
    b = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / sxx
    r = b * math.sqrt(sxx / syy)
    return b, r * r


def main():
    gm = load_gmst()
    pre = [v for y, v in gm.items() if 1850 <= y <= 1900]
    base = sum(pre) / len(pre)
    rel = {y: v - base for y, v in gm.items()}
    series = load_series()

    # ---------- A. 标度稳定性 ----------
    # 固定站集: 只保留两段都有充分覆盖的站, 否则构成变化会污染标度比较
    def has_span(st, y0, y1, need):
        s = {y for (y, k), v in series[st].items() if k == m}
        return sum(1 for y in s if y0 <= y <= y1) >= need

    print("=" * 90)
    print("【R6-A】标度稳定性：线性外推的前提检验（固定站集，两段均需 ≥8 年）")
    print("=" * 90)
    print(f"{'指标':<20}{'站数':>5}{'全期标度':>11}{'前期(76-99)':>13}{'后期(00-23)':>13}"
          f"{'后期/前期':>11}{'R²(全期,连续)':>15}")
    print("-" * 90)
    stab = {}
    for m in METRICS:
        keep = {st: dd for st, dd in series.items()
                if has_span(st, 1976, 1999, 8) and has_span(st, 2000, 2023, 8)}
        if len(keep) < 10:
            continue
        comp, _ = composite(keep, m)
        ys = sorted(comp)
        if len(ys) < 12:
            continue
        def fit(y0, y1):
            yy = [y for y in ys if y0 <= y <= y1 and y in rel]
            if len(yy) < 6:
                return None, None
            xs = [rel[y] for y in yy]; vs = [comp[y] for y in yy]
            mx = sum(xs)/len(xs); my = sum(vs)/len(vs)
            return ols([v-mx for v in xs], [v-my for v in vs])
        b_all, r2_all = fit(1976, 2023)
        b1, _ = fit(1976, 1999)
        b2, _ = fit(2000, 2023)
        ratio = (b2 / b1) if (b1 and b2) else None
        stab[m] = {"all": b_all, "early": b1, "late": b2, "r2_all": r2_all,
                   "ratio": ratio, "n_years": len(ys), "n_stations": len(keep)}
        print(f"{LABEL[m]:<20}{len(keep):>5}"
              f"{(b_all if b_all else float('nan')):>11.4f}"
              f"{(b1 if b1 else float('nan')):>13.4f}"
              f"{(b2 if b2 else float('nan')):>13.4f}"
              f"{(ratio if ratio else float('nan')):>11.2f}"
              f"{(r2_all if r2_all else float('nan')):>15.3f}")
    print("\n  若 '后期/前期' 显著偏离 1, 则线性外推有偏, 预估须改为分段。")

    # ---------- B. 分样本 ----------
    afr = set()
    p = os.path.join(HERE, "panel_afrsa.json")
    if os.path.exists(p):
        import json
        afr = {m["station"] for m in json.load(open(p, encoding="utf-8"))}
    groups = {"合并": None, "核心站": lambda s: s not in afr,
              "非洲/南亚站": lambda s: s in afr}
    print("\n" + "=" * 90)
    print("【R6-B】分样本趋势（连续序列）")
    print("=" * 90)
    print(f"{'指标':<20}{'样本':<14}{'站数':>5}{'年数中位':>9}"
          f"{'中位斜率':>12}{'正/负':>9}{'符号p':>11}")
    print("-" * 90)
    for m in METRICS:
        for gname, filt in groups.items():
            sl = []
            nyears = []
            for st, dd in series.items():
                if filt and not filt(st):
                    continue
                s = {y: v for (y, k), v in dd.items() if k == m}
                ys = sorted(s)
                if len(ys) < 8:
                    continue
                b = theil_sen([float(y) for y in ys], [s[y] for y in ys])
                if b is not None:
                    sl.append(b); nyears.append(len(ys))
            if len(sl) < 5:
                continue
            pos = sum(1 for v in sl if v > 0); neg = sum(1 for v in sl if v < 0)
            print(f"{LABEL[m]:<20}{gname:<14}{len(sl):>5}"
                  f"{median(nyears):>9.0f}{median(sl):>12.5f}"
                  f"{str(pos)+'/'+str(neg):>9}{sign_test(pos,neg):>11.2e}")
        print("-" * 90)

    # ---------- C. 人口口径更正 ----------
    print("\n" + "=" * 90)
    print("【R6-C】人口暴露口径更正：只对实际参与分析的站汇总")
    print("=" * 90)
    pop = {r["station"]: float(r["pop_50km"])
           for r in csv.DictReader(open(os.path.join(HERE,
                                                     "station_population.csv"),
                                        encoding="utf-8"))}
    used = set(series)
    tot = sum(pop.get(s, 0.0) for s in used)
    print(f"  参与分析的站数 {len(used)}, 覆盖人口 {tot/1e8:.2f} 亿")
    print("  （更正前报告中误用了整个站表 218 站的合计 14.48 亿）\n")

    # 时相对比: E1 vs E3 的人口加权 NRF 与夜间超阈人·小时
    def pw(metric, y0, y1, weight=True):
        num = 0.0; den = 0.0
        for st, dd in series.items():
            s = {y: v for (y, k), v in dd.items() if k == metric and y0 <= y <= y1}
            if not s:
                continue
            v = median(list(s.values()))
            if v is None:
                continue
            w = pop.get(st, 0.0) if weight else 1.0
            if w <= 0 and weight:
                continue
            num += w * v; den += w
        return (num / den) if den > 0 else None

    for metric, unit in [("nrfr_elderly", ""),
                         ("nrd_elderly", " °C"),
                         ("nhrs_elderly", " h/年"),
                         ("tw_night_mean", " °C"),
                         ("usable_frac", "")]:
        e1 = pw(metric, 1976, 1980)
        e3 = pw(metric, 2019, 2023)
        if e1 is None or e3 is None:
            continue
        print(f"  {LABEL.get(metric, metric):<20} "
              f"E1 {e1:>9.4f}{unit}  ->  E3 {e3:>9.4f}{unit}  "
              f"({(e3-e1)/abs(e1)*100 if e1 else float('nan'):+.1f}%)")
    # 人·小时
    for metric in ["nhrs_elderly", "hrs_elderly"]:
        v1 = pw(metric, 1976, 1980); v3 = pw(metric, 2019, 2023)
        if v1 is None or v3 is None:
            continue
        print(f"\n  人口加权 {metric}:")
        print(f"    E1 人·小时/年 = {v1*tot/1e8:.1f} 亿")
        print(f"    E3 人·小时/年 = {v3*tot/1e8:.1f} 亿  "
              f"({(v3-v1)/v1*100 if v1 else 0:+.1f}%)")


if __name__ == "__main__":
    main()
