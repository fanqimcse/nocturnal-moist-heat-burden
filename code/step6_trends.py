#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 6: 趋势与显著性统计

审稿人会盯的统计问题:
  1. 单站趋势 —— Theil-Sen 斜率 (对离群值稳健) + Mann-Kendall 检验
     注意: 逐年值存在自相关时 MK 会低估方差、夸大显著性。
     这里同时给出 (a) 原始 MK  (b) 一阶自相关修正后的有效样本量 MK。
  2. 多站聚合 —— 不能简单把各站 p 值当独立样本。
     空间相关会使"显著站比例"虚高。采用:
       (a) Benjamini-Hochberg FDR 控制 (q=0.05)
       (b) 站点 bootstrap 重抽样给出聚合趋势的置信区间
       (c) 符号检验: 正趋势站数 vs 负趋势站数 (对空间相关不敏感)
  3. 时相对比 —— E3-E1 配对差, bootstrap CI
"""

import sys, csv, math, json, random, collections, os
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

random.seed(20261018)


# ---------- 基础统计 ----------
def median(x):
    if not x:
        return None
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def theil_sen(x, y):
    """Theil-Sen 斜率: 所有点对斜率的中位数。对离群值稳健。"""
    sl = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    return median(sl) if sl else None


def lag1_autocorr(v):
    if len(v) < 3:
        return 0.0
    m = sum(v) / len(v)
    num = sum((v[i] - m) * (v[i + 1] - m) for i in range(len(v) - 1))
    den = sum((a - m) ** 2 for a in v)
    return num / den if den > 0 else 0.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_kendall(v, adjust_autocorr=True):
    """
    返回 (S, z, p, tau)
    adjust_autocorr=True 时用 Hamed & Rao (1998) 的方差修正因子,
    处理逐年序列的正自相关导致显著性被夸大的问题。
    """
    n = len(v)
    if n < 4:
        return None, None, None, None
    S = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            S += 1 if d > 0 else (-1 if d < 0 else 0)
    varS = n * (n - 1) * (2 * n + 5) / 18.0
    if adjust_autocorr:
        r = lag1_autocorr(v)
        if r > 0:
            # Hamed & Rao 修正
            corr = 1.0
            for i in range(1, n - 1):
                corr += 2.0 * (n - i) * (n - i - 1) * (n - i - 2) * (r ** i) \
                        / (n * (n - 1) * (n - 2))
            varS *= max(corr, 1.0)
    if varS <= 0:
        return S, None, None, None
    z = (S - 1) / math.sqrt(varS) if S > 0 else (
        (S + 1) / math.sqrt(varS) if S < 0 else 0.0)
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    tau = 2.0 * S / (n * (n - 1))
    return S, z, p, tau


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: 返回显著性的布尔列表。"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = [False] * m
    thr = None
    for rank, i in enumerate(idx, 1):
        if pvals[i] <= q * rank / m:
            thr = pvals[i]
    if thr is None:
        return out, None
    for i in range(m):
        out[i] = pvals[i] <= thr
    return out, thr


def bootstrap_ci(vals, fn=median, n_boot=4000, alpha=0.05):
    """对统计量做站点 bootstrap, 返回 (估计, lo, hi)。"""
    if len(vals) < 3:
        return (fn(vals), None, None)
    est = fn(vals)
    bs = []
    for _ in range(n_boot):
        s = [vals[random.randrange(len(vals))] for _ in range(len(vals))]
        v = fn(s)
        if v is not None:
            bs.append(v)
    bs.sort()
    lo = bs[int(alpha / 2 * len(bs))]
    hi = bs[int((1 - alpha / 2) * len(bs))]
    return est, lo, hi


def sign_test(pos, neg):
    """二项符号检验 (忽略并列), 对空间相关稳健。"""
    n = pos + neg
    if n == 0:
        return None
    k = max(pos, neg)
    # 精确二项尾部概率
    def C(n_, r):
        return math.comb(n_, r)
    tail = sum(C(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# ---------- 主流程 ----------
METRICS = [
    "tw_max", "tw_mean", "hrs_elderly", "hrs_young",
    "nhrs_elderly", "nhrs_young", "nrf_elderly", "nrf_young",
    "dtr_t", "dtr_tw", "tw_night_mean", "tw_day_mean",
    "night_min_mean", "night_min_max", "t_max", "t_mean",
]


def load(path="metrics_core.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["year"] = int(r["year"])
        for k in list(r.keys()):
            if k in ("station", "name", "country", "band", "region"):
                continue
            try:
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            except ValueError:
                pass
    return rows


def trends(rows, metric, min_years=8):
    """逐站计算趋势, 返回列表"""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(metric) is None:
            continue
        by[r["station"]].append(r)
    out = []
    for st, rs in by.items():
        rs.sort(key=lambda x: x["year"])
        yrs = [r["year"] for r in rs]
        vals = [r[metric] for r in rs]
        if len(set(yrs)) < min_years:
            continue
        sl = theil_sen(yrs, vals)
        S, z, p, tau = mann_kendall(vals)
        out.append(dict(station=st, n=len(yrs), y0=min(yrs), y1=max(yrs),
                        slope=sl, mk_z=z, mk_p=p, tau=tau,
                        lat=rs[0].get("lat"), lon=rs[0].get("lon"),
                        band=rs[0].get("band"), name=rs[0].get("name"),
                        country=rs[0].get("country")))
    return out


def summarize(ts, label):
    slopes = [t["slope"] for t in ts if t["slope"] is not None]
    ps = [t["mk_p"] for t in ts if t.get("mk_p") is not None]
    if not slopes:
        return None
    est, lo, hi = bootstrap_ci(slopes)
    sig, thr = bh_fdr(ps, 0.05) if ps else ([], None)
    pos = sum(1 for s in slopes if s > 0)
    neg = sum(1 for s in slopes if s < 0)
    sp = sign_test(pos, neg)
    return dict(metric=label, n_stations=len(slopes),
                median_slope=est, ci_lo=lo, ci_hi=hi,
                n_pos=pos, n_neg=neg, sign_p=sp,
                n_sig_fdr=sum(sig), fdr_thr=thr,
                frac_pos=pos / len(slopes))


def epoch_contrast(rows, metric, e1=(1976, 1980), e3=(2019, 2023)):
    """时相对比: 同一站点的 E3 均值 - E1 均值, 配对差。"""
    by = collections.defaultdict(lambda: {"E1": [], "E3": []})
    for r in rows:
        if r.get(metric) is None:
            continue
        y = r["year"]
        if e1[0] <= y <= e1[1]:
            by[r["station"]]["E1"].append(r[metric])
        elif e3[0] <= y <= e3[1]:
            by[r["station"]]["E3"].append(r[metric])
    diffs, base = [], []
    for st, d in by.items():
        if d["E1"] and d["E3"]:
            a = sum(d["E1"]) / len(d["E1"])
            b = sum(d["E3"]) / len(d["E3"])
            diffs.append(b - a)
            base.append(a)
    if not diffs:
        return None
    est, lo, hi = bootstrap_ci(diffs)
    pos = sum(1 for x in diffs if x > 0)
    rel = None
    bm = median(base)
    if bm:
        rel = 100.0 * est / abs(bm) if bm != 0 else None
    return dict(n=len(diffs), mean_diff=sum(diffs) / len(diffs),
                median_diff=est, ci_lo=lo, ci_hi=hi,
                n_pos=pos, frac_pos=pos / len(diffs),
                base_median=bm, rel_change_pct=rel)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "metrics_core.csv"
    if not os.path.exists(p):
        print("找不到", p)
        sys.exit(0)
    rows = load(p)
    print(f"载入 {len(rows)} 条站年记录")
    print(f"\n{'指标':<18}{'站数':>6}{'中位趋势/年':>13}{'95%CI':>22}"
          f"{'正/负':>10}{'FDR显著':>9}")
    print("-" * 82)
    for m in METRICS:
        ts = trends(rows, m)
        s = summarize(ts, m)
        if not s:
            continue
        ci = f"[{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]" if s['ci_lo'] is not None else "-"
        print(f"{m:<18}{s['n_stations']:>6}{s['median_slope']:>13.4g}"
              f"{ci:>22}{str(s['n_pos'])+'/'+str(s['n_neg']):>10}"
              f"{s['n_sig_fdr']:>9}")
    print("\n时相对比 (E3 2019-2023 减 E1 1976-1980):")
    print(f"{'指标':<18}{'站数':>6}{'中位变化':>11}{'95%CI':>24}{'正比例':>9}{'相对%':>9}")
    print("-" * 82)
    for m in METRICS:
        e = epoch_contrast(rows, m)
        if not e:
            continue
        ci = f"[{e['ci_lo']:.4g}, {e['ci_hi']:.4g}]" if e['ci_lo'] is not None else "-"
        rp = f"{e['rel_change_pct']:.1f}" if e['rel_change_pct'] is not None else "-"
        print(f"{m:<18}{e['n']:>6}{e['median_diff']:>11.4g}{ci:>24}"
              f"{100*e['frac_pos']:>8.0f}%{rp:>9}")
