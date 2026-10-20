#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 15: 未来预估与政策含义

用步骤 14 标定的"每 °C 全球增暖"标度关系, 把观测到的夜间湿润热负担
外推到 IPCC AR6 各情景, 回答一个政策问题:

    全球再增暖 0.5 / 1.0 / 2.0 °C, 会有多少人进入"夜间无法恢复"的状态?

采用两条独立途径并交叉检验:
  途径 1 (线性标度):  指标异常 = 标度 x GMST 异常, 直接用逐年回归斜率外推
  途径 2 (站内标定):  用本站"夜间湿球 -> 夜间恢复失效率"的经验曲线,
                      把未来夜间湿球映射到 NRF, 避免线性外推越界

输出: projections.txt
"""
import sys, os, csv, math, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import theil_sen, median
from step14_attribution import load_gmst, load_series, composite, ols_slope_se

# IPCC AR6 2081-2100 全球地表温度相对 1850-1900 的最佳估计
SSP = {"SSP1-2.6": 1.8, "SSP2-4.5": 2.7, "SSP3-7.0": 3.6, "SSP5-8.5": 4.4}


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[i]


def main():
    gm = load_gmst()
    pre = [v for y, v in gm.items() if 1850 <= y <= 1900]
    base = sum(pre) / len(pre)
    rel = {y: v - base for y, v in gm.items()}
    cur = rel.get(2023, rel[max(rel)])
    print(f"GISTEMP: 1850-1900 基准 {base:+.3f} (1951-80 基)"
          f" -> 2023 相对工业化前 {cur:+.2f} °C")

    series = load_series()
    # ---------- 标度关系 ----------
    scales = {}
    for m in ["tw_night_mean", "usable_frac", "nrd_elderly", "dtr_tw",
              "nrfr_elderly", "relief_elderly"]:
        comp, _ = composite(series, m)
        ys = [y for y in sorted(comp) if y in rel]
        if len(ys) < 8:
            continue
        xs = [rel[y] for y in ys]
        vs = [comp[y] for y in ys]
        mx = sum(xs) / len(xs); my = sum(vs) / len(vs)
        xs = [v - mx for v in xs]; vs = [v - my for v in vs]
        sxx = sum(v * v for v in xs)
        if sxx <= 0:
            continue
        b = sum(x * v for x, v in zip(xs, vs)) / sxx
        scales[m] = b

    print("\n每 °C 全球增暖的标度（合成序列回归）:")
    for k, v in scales.items():
        print(f"  {k:<18} {v:+.4f}")

    # ---------- 标度稳定性 -> 区间外推 (回应 M40) ----------
    # R6-4 已证明标度不稳定(后期/前期 0.47-0.74), 单一标度外推系统性有偏。
    # 故对每一指标分别估计 全期/前期(1976-1999)/后期(2000-2023) 三个标度,
    # 用其 min/max 作为预估的上下界。
    def scale_window(m, y0, y1):
        comp, _ = composite(series, m)
        ys = [y for y in sorted(comp) if y0 <= y <= y1 and y in rel]
        if len(ys) < 8:
            return None
        xs = [rel[y] for y in ys]; vs = [comp[y] for y in ys]
        mx = sum(xs) / len(xs); my = sum(vs) / len(vs)
        xs = [v - mx for v in xs]; vs = [v - my for v in vs]
        sxx = sum(v * v for v in xs)
        if sxx <= 0:
            return None
        return sum(x * v for x, v in zip(xs, vs)) / sxx

    sc_lo, sc_hi = {}, {}
    print("\n标度稳定性 (区间外推的依据):")
    print(f"  {'指标':<20}{'全期':>11}{'前期76-99':>12}{'后期00-23':>12}{'下界':>11}{'上界':>11}")
    for m in list(scales):
        be = scale_window(m, 1976, 1999)
        bl = scale_window(m, 2000, 2023)
        cand = [v for v in (scales.get(m), be, bl) if v is not None]
        if len(cand) < 2:
            sc_lo[m] = sc_hi[m] = scales.get(m)
            continue
        sc_lo[m] = min(cand); sc_hi[m] = max(cand)
        f = lambda v: (f"{v:+.4f}" if v is not None else "   n/a")
        print(f"  {m:<20}{f(scales.get(m)):>11}{f(be):>12}{f(bl):>12}"
              f"{sc_lo[m]:>+11.4f}{sc_hi[m]:>+11.4f}")
    print("  —— 预估改用 [下界, 上界] 区间, 不再给单一数值 ——")

    # ---------- 途径 1: 线性外推 ----------
    print("\n" + "=" * 88)
    print("【预估 1】线性标度外推：相对 2023 年, 全球再增暖 Δ 时的变化")
    print("=" * 88)
    print(f"{'ΔGMST':>8}{'夜间湿球(区间)':>22}"
          f"{'可用降温比例':>14}{'湿球日较差':>12}{'夜间恢复亏缺':>14}{'可恢复窗口':>12}")
    print("-" * 88)
    for d in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:

        def rng(k):
            a = sc_lo.get(k, scales.get(k, 0.0)) * d
            b = sc_hi.get(k, scales.get(k, 0.0)) * d
            return f"[{a:+.2f},{b:+.2f}]"

        print(f"{d:>8.1f}{rng('tw_night_mean'):>22}"
              f"{scales.get('usable_frac',0)*d:>14.3f}"
              f"{scales.get('dtr_tw',0)*d:>12.2f}"
              f"{scales.get('nrd_elderly',0)*d:>14.3f}"
              f"{scales.get('relief_elderly',0)*d:>12.2f}")

    # ---------- 途径 2: 站内标定 NRF ----------
    print("\n" + "=" * 88)
    print("【预估 2】站内标定：夜间湿球 -> 夜间恢复失效率, 再按人口加权")
    print("=" * 88)
    pop = {}
    pp = os.path.join(HERE, "station_population.csv")
    if os.path.exists(pp):
        pop = {r["station"]: float(r["pop_50km"])
               for r in csv.DictReader(open(pp, encoding="utf-8"))}

    # 每站: 用本站 (夜间湿球, NRF) 的年度配对, Theil-Sen 稳健估计 dNRF/dTw
    curves = {}
    for st, dd in series.items():
        tw = {y: v for (y, k), v in dd.items() if k == "tw_night_mean"}
        nf = {y: v for (y, k), v in dd.items() if k == "nrfr_elderly"}
        common = sorted(set(tw) & set(nf))
        if len(common) < 8:
            continue
        b = theil_sen([tw[y] for y in common], [nf[y] for y in common])
        if b is None:
            continue
        # 当代基准: 最近 5 个有数据年份
        last = common[-5:]
        tw0 = sum(tw[y] for y in last) / len(last)
        p0 = sum(nf[y] for y in last) / len(last)
        curves[st] = (b, tw0, p0)

    def nrf_at(st, tw):
        b, tw0, p0 = curves[st]
        return min(1.0, max(0.0, p0 + b * (tw - tw0)))

    if curves:
        tot = sum(pop.get(st, 0.0) for st in curves)
        print(f"  标定站数 {len(curves)}, 覆盖人口 {tot/1e8:.2f} 亿")
        print(f"  dNRF/dTw 中位 {median([curves[s][0] for s in curves]):+.4f} /°C\n")

    for name, tgt in SSP.items():
        d = tgt - cur
        if d <= 0:
            continue
        dlt = scales.get("tw_night_mean", 0.0) * d
        dlt_lo = sc_lo.get("tw_night_mean", dlt / max(d, 1e-9)) * d
        dlt_hi = sc_hi.get("tw_night_mean", dlt / max(d, 1e-9)) * d
        # 区间: 下界/上界标度各跑一遍站内 logistic
        res = {}
        for tag, dd in (("下界", dlt_lo), ("中位", dlt), ("上界", dlt_hi)):
            vals = []
            wsum = 0.0; wsum_base = 0.0
            for st, (b, tw0, p0) in curves.items():
                p1 = nrf_at(st, tw0 + dd)
                vals.append((p1 - p0, p1))
                w = pop.get(st, 0.0)
                wsum += w * p1; wsum_base += w * p0
            res[tag] = (vals, wsum, wsum_base)
        if not res["中位"][0]:
            continue
        tot = sum(pop.get(st, 0.0) for st in curves)
        print(f"\n  {name}: 2081-2100 全球相对工业化前 {tgt:+.1f} °C, "
              f"即相对 2023 年再增暖 {d:+.2f} °C")
        print(f"    夜间湿球上升: 中位 {dlt:+.2f} °C, "
              f"区间 [{dlt_lo:+.2f}, {dlt_hi:+.2f}] °C")
        for tag in ("下界", "中位", "上界"):
            vals, wsum, wsum_base = res[tag]
            d_nrf = sorted(v for v, _ in vals)
            s = (f"    夜间恢复失效率变化 [{tag}]: 中位 {median(d_nrf):+.3f} "
                 f"(四分位 [{quantile(d_nrf,0.25):+.3f}, {quantile(d_nrf,0.75):+.3f}])")
            if tot > 0:
                s += (f"   人口加权 {wsum_base/tot:.3f} -> {wsum/tot:.3f} "
                      f"({100*(wsum/tot - wsum_base/tot):+.1f} 个百分点)")
            print(s)

    print("\n  口径: '夜间无法恢复' = 该夜最低湿球未降到 21.9 °C 以下;")
    print("        人口为站点 50 km 全额 + 指数衰减至 300 km 的邻近城市人口。")


if __name__ == "__main__":
    main()
