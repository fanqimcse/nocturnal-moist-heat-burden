#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R7-3（回应 M38）: 出现时间 Time of Emergence

48 年连续序列最有价值的产物: 不回答"变化了多少", 而回答"什么时候变的"。
把统计显著性翻译成人类可感知的年份。

判据 (Hawkins & Sutton 2012 传统):
  - 基准期 1976-1995 (20 年), 先线性去趋势以隔离年际变率, 取残差标准差 sigma
  - 阈值 = 基准期均值 + 2*sigma
  - 5 年居中滑动平均
  - ToE = 平滑序列首次超过阈值、且此后永不再回落到阈值以下的年份
  - 若截至 2023 仍未满足, 记为"尚未浮现"
"""
import sys, os, csv, collections, json, math
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import median

BASE = (1976, 1995)
METRICS = [
    ("tw_night_mean", "夜间湿球均值", +1),
    ("nrfr_elderly", "夜间恢复失效率", +1),
    ("nhrs_elderly", "夜间超阈小时", +1),
    ("dtr_tw", "湿球日较差", -1),
    ("usable_frac", "可用降温比例", -1),
    ("fn_tn25", "热夜占比(TN>=25)", +1),
]


def series(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = collections.defaultdict(dict)
    for r in rows:
        if r.get("ok_trend") != "1":
            continue
        d[r["station"]][int(r["year"])] = r
    return d


def smooth(ys, vs, w=5):
    out = {}
    h = w // 2
    for i, y in enumerate(ys):
        lo = max(0, i - h); hi = min(len(ys), i + h + 1)
        if hi - lo >= max(3, w - 1):
            out[y] = sum(vs[lo:hi]) / (hi - lo)
    return out


def toe_one(yrs, vals, sign, w=5):
    """返回 (toe_year or None, sigma, base_mean, threshold)"""
    b = [(y, v) for y, v in zip(yrs, vals) if BASE[0] <= y <= BASE[1]]
    if len(b) < 8:
        return None, None, None, None
    by = [x[0] for x in b]; bv = [x[1] for x in b]
    n = len(by); my = sum(by) / n; mv = sum(bv) / n
    sxy = sum((y - my) * (v - mv) for y, v in b)
    sxx = sum((y - my) ** 2 for y in by)
    sl = sxy / sxx if sxx else 0.0
    res = [v - (mv + sl * (y - my)) for y, v in b]
    sigma = math.sqrt(sum(r * r for r in res) / max(1, n - 2))
    if sigma <= 1e-12:
        return None, None, None, None
    thr = mv + sign * 2 * sigma
    sm = smooth(yrs, vals, w)
    sy = sorted(sm)
    toe = None
    for i, y in enumerate(sy):
        if all((sm[z] - thr) * sign > 0 for z in sy[i:]):
            toe = y
            break
    return toe, sigma, mv, thr


def main():
    print("=" * 100)
    print("R7-3 / M38  出现时间分析 (Time of Emergence)")
    print(f"  基准期 {BASE[0]}-{BASE[1]}, 2-sigma 判据, 5 年居中滑动平均")
    print("=" * 100)
    d = series(os.path.join(HERE, "metrics_v3.csv"))
    sts = sorted(d)
    print(f"站数 {len(sts)}")

    print(f"\n{'指标':<20}{'可用站':>7}{'已浮现':>8}{'占比':>8}"
          f"{'出现年中位':>11}{'四分位区间':>16}{'最早':>7}{'最晚':>7}")
    print("-" * 100)
    out = {}
    for m, lab, sign in METRICS:
        toes = []
        for st in sts:
            ys = sorted(d[st])
            vs = []
            yy = []
            for y in ys:
                v = d[st][y].get(m)
                if v in (None, "", "None"):
                    continue
                try:
                    vs.append(float(v)); yy.append(y)
                except Exception:
                    pass
            if len(yy) < 20:
                continue
            t, sg, bm, thr = toe_one(yy, vs, sign)
            if t is not None:
                toes.append(t)
        if not toes:
            print(f"{lab:<20}{'—':>7}")
            continue
        toes.sort()
        q1 = toes[len(toes) // 4]; q3 = toes[3 * len(toes) // 4]
        n_avail = sum(1 for st in sts if len([y for y in d[st]
                      if d[st][y].get(m) not in (None, "", "None")]) >= 20)
        print(f"{lab:<20}{n_avail:>7}{len(toes):>8}{100*len(toes)/max(1,n_avail):>7.0f}%"
              f"{median(toes):>11.0f}{f'{q1:.0f}-{q3:.0f}':>16}"
              f"{toes[0]:>7}{toes[-1]:>7}")
        out[m] = dict(n_avail=n_avail, emerged=len(toes),
                      toe_median=median(toes), q1=q1, q3=q3,
                      earliest=toes[0], latest=toes[-1])

    # 逐十年: 已浮现站点累计数
    print(f"\n【累计浮现曲线】按'夜间恢复失效率'计, 每隔 5 年已浮现站数占比:")
    m, sign = "nrfr_elderly", +1
    toes = []
    for st in sts:
        ys = sorted(d[st]); vs = []; yy = []
        for y in ys:
            v = d[st][y].get(m)
            if v in (None, "", "None"):
                continue
            try:
                vs.append(float(v)); yy.append(y)
            except Exception:
                pass
        if len(yy) < 20:
            continue
        t, *_ = toe_one(yy, vs, sign)
        if t is not None:
            toes.append((t, st))
    if toes:
        tot = len(toes)
        print("-" * 60)
        for yr in range(1990, 2026, 5):
            c = sum(1 for t, _ in toes if t <= yr)
            bar = "#" * int(40 * c / tot)
            print(f"  {yr}: {c:>3}/{tot} ({100*c/tot:>5.1f}%) {bar}")
        print("-" * 60)
        # 人口加权的已浮现人口
        try:
            pop = {r["station"]: float(r["pop_50km"]) for r in
                   csv.DictReader(open(os.path.join(HERE, "station_population.csv"),
                                       encoding="utf-8"))}
            totp = sum(pop.get(s, 0) for _, s in toes)
            print(f"  已浮现站点覆盖人口: {totp/1e8:.2f} 亿")
            for yr in (2000, 2010, 2020):
                p = sum(pop.get(s, 0) for t, s in toes if t <= yr)
                print(f"    {yr} 年前浮现: {p/1e8:.2f} 亿人")
        except Exception as e:
            print("  (人口数据不可用)", e)

    with open(os.path.join(HERE, "toe_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存 toe_results.json")


if __name__ == "__main__":
    main()
