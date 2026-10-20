#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 8: 综合统计分析 —— 生成论文级结果

输出:
  results_summary.txt   人读的结果摘要
  results.json          机读的完整结果（供绘图与报告使用）
"""
import sys, os, csv, json, math, collections, random
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step6_trends import (theil_sen, mann_kendall, bh_fdr, bootstrap_ci,
                          sign_test, median)

random.seed(20261018)

METRICS = {
    "tw_p99": "湿球年 99 分位 (°C)",
    "tw_mean": "湿球年均 (°C)",
    "tw_night_mean": "夜间湿球均值 (°C)",
    "tw_day_mean": "白天湿球均值 (°C)",
    "rate_elderly": "全年超阈小时占比 (21.9°C)",
    "nrate_elderly": "夜间超阈小时占比 (21.9°C)",
    "drate_elderly": "白天超阈小时占比 (21.9°C)",
    "nrfr_elderly": "夜间恢复失效率 NRF (21.9°C)",
    "nrfr_young": "夜间恢复失效率 NRF (25.8°C)",
    "nhrs_elderly": "夜间超阈等效小时 (21.9°C)",
    "hrs_elderly": "全年超阈等效小时 (21.9°C)",
    "dtr_t": "干球日较差 (°C)",
    "dtr_tw": "湿球日较差 (°C)",
    "t_mean": "干球年均 (°C)",
}
KEY = ["nrfr_elderly", "nrd_elderly", "nrelief_elderly", "relief_elderly",
       "nrate_elderly", "drate_elderly", "tw_night_mean",
       "tw_day_mean", "dtr_tw", "dtr_t", "tw_p99", "hrs_elderly", "nhrs_elderly",
       "nrfr_young", "nrd_young", "unusable", "usable_frac",
       "nrfr_psu", "nrfr_classic"]


def composite(rows, metric, min_stations=5, use_anomaly=True):
    """
    综合序列趋势。

    use_anomaly=True 时先对每个站减去**该站自身记录期的均值**，再跨站合成。
    这一步是必需的：本面板的站点构成随年代变化（后期热带站更多），
    若直接合成原始值，会把"网络构成变化"误读为气候趋势。
    转成距平后，每个站贡献的是"相对于自己常态的偏离"，构成变化被消除。
    """
    by = collections.defaultdict(list)
    for r in rows:
        if not r["ok_trend"] or r.get(metric) is None:
            continue
        by[r["station"]].append(r)
    base = {}
    for st, rs in by.items():
        v = [x[metric] for x in rs]
        base[st] = sum(v) / len(v)

    yy = collections.defaultdict(list)
    for st, rs in by.items():
        for r in rs:
            yy[r["year"]].append(r[metric] - base[st] if use_anomaly else r[metric])
    ys = sorted(y for y, v in yy.items() if len(v) >= min_stations)
    if len(ys) < 8:
        return None
    vals = [median(yy[y]) for y in ys]
    sl = theil_sen(ys, vals)
    S, z, p, tau = mann_kendall(vals)
    nst = {y: len(yy[y]) for y in ys}
    return dict(y0=ys[0], y1=ys[-1], n=len(ys), slope=sl, z=z, p=p,
                n_stations=nst, series=list(zip(ys, vals)))


def by_station(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["station"]].append(r)
    return d


def load(path="metrics_v3.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        r["year"] = int(float(r["year"]))
        r["ok_trend"] = int(float(r.get("ok_trend", 0) or 0))
        for k in list(r.keys()):
            if k in ("station", "name", "country", "band", "region",
                     "year", "ok_trend"):
                continue
            v = r[k]
            r[k] = float(v) if v not in ("", "None", None) else None
        out.append(r)
    return out


def station_trends(rows, metric, min_years=6):
    by = collections.defaultdict(list)
    for r in rows:
        if not r["ok_trend"] or r.get(metric) is None:
            continue
        by[r["station"]].append(r)
    res = []
    for st, rs in by.items():
        rs.sort(key=lambda x: x["year"])
        yrs = [r["year"] for r in rs]
        vals = [r[metric] for r in rs]
        if len(set(yrs)) < min_years:
            continue
        sl = theil_sen(yrs, vals)
        S, z, p, tau = mann_kendall(vals)
        res.append(dict(station=st, n=len(yrs), y0=min(yrs), y1=max(yrs),
                        slope=sl, p=p, z=z, lat=rs[0]["lat"], lon=rs[0]["lon"],
                        band=rs[0].get("band", ""), region=rs[0].get("region", ""),
                        name=rs[0].get("name", ""), country=rs[0].get("country", "")))
    return res


def agg(ts):
    slopes = [t["slope"] for t in ts if t["slope"] is not None]
    ps = [t["p"] for t in ts if t.get("p") is not None]
    if not slopes:
        return None
    est, lo, hi = bootstrap_ci(slopes)
    sig, thr = bh_fdr(ps, 0.05) if ps else ([], None)
    pos = sum(1 for s in slopes if s > 0)
    neg = sum(1 for s in slopes if s < 0)
    d = dict(n=len(slopes), median_slope=est, ci_lo=lo, ci_hi=hi,
             n_pos=pos, n_neg=neg, frac_pos=pos / len(slopes),
             sign_p=sign_test(pos, neg),
             n_sig=sum(sig) if sig else 0, fdr_thr=thr)
    return d


def epoch(rows, metric, e1, e3):
    by = collections.defaultdict(lambda: {"a": [], "b": []})
    for r in rows:
        if not r["ok_trend"] or r.get(metric) is None:
            continue
        if e1[0] <= r["year"] <= e1[1]:
            by[r["station"]]["a"].append(r[metric])
        elif e3[0] <= r["year"] <= e3[1]:
            by[r["station"]]["b"].append(r[metric])
    d, base = [], []
    for st, v in by.items():
        if v["a"] and v["b"]:
            a = sum(v["a"]) / len(v["a"]); b = sum(v["b"]) / len(v["b"])
            d.append(b - a); base.append(a)
    if not d:
        return None
    est, lo, hi = bootstrap_ci(d)
    bm = median(base)
    return dict(n=len(d), mean=sum(d) / len(d), median=est, ci_lo=lo, ci_hi=hi,
                n_pos=sum(1 for x in d if x > 0), frac_pos=sum(1 for x in d if x > 0) / len(d),
                base=bm, rel=100 * est / bm if bm else None)


def main(src="metrics_v3.csv"):
    rows = load(src)
    if not rows:
        print("无数据")
        return
    ok = [r for r in rows if r["ok_trend"]]
    print(f"载入 {len(rows)} 条站年, 其中 {len(ok)} 条通过质量门槛")
    st_n = len(set(r['station'] for r in ok))
    print(f"涉及站点 {st_n} 个, 年份 {min(r['year'] for r in ok)}-{max(r['year'] for r in ok)}")

    out = {"meta": dict(n_stationyear=len(rows), n_ok=len(ok), n_station=st_n,
                        y0=min(r["year"] for r in ok), y1=max(r["year"] for r in ok))}

    # ---- 1. 全球趋势 ----
    print("\n" + "=" * 92)
    print("【表 1】全球站点趋势 (Theil-Sen 斜率 / 年, Mann-Kendall + BH-FDR)")
    print("=" * 92)
    print(f"{'指标':<18}{'站数':>5}{'中位斜率':>11}{'95%CI':>22}{'正/负':>9}{'正比例':>8}{'FDR显著':>8}")
    print("-" * 92)
    out["trends"] = {}
    for m in KEY:
        ts = station_trends(rows, m)
        a = agg(ts)
        if not a:
            continue
        a["sign_p"] = sign_test(a["n_pos"], a["n_neg"])
        out["trends"][m] = a
        ci = f"[{a['ci_lo']:.4g}, {a['ci_hi']:.4g}]"
        sp = f"{a['sign_p']:>11.2g}" if a["sign_p"] is not None else f"{'—':>11}"
        print(f"{m:<18}{a['n']:>5}{a['median_slope']:>11.4g}{ci:>22}"
              f"{str(a['n_pos'])+'/'+str(a['n_neg']):>9}"
              f"{100*a['frac_pos']:>7.0f}%{a['n_sig']:>8}"
              f"{sp}")

    # ---- 1b. 综合序列趋势（功效最高）----
    print("\n" + "=" * 92)
    print("【表 1b】多站中位数合成序列的趋势 (Mann-Kendall, 功效高于逐站检验)")
    print("=" * 92)
    print(f"{'指标':<18}{'年份跨度':>12}{'年数':>5}{'斜率/年':>12}{'MK z':>9}{'p值':>11}")
    print("-" * 92)
    out["composite"] = {}
    for m in KEY:
        c = composite(rows, m)
        if not c:
            continue
        out["composite"][m] = {k: v for k, v in c.items() if k != "series"}
        out["composite"][m]["series"] = c["series"]
        print(f"{m:<18}{str(c['y0'])+'-'+str(c['y1']):>12}{c['n']:>5}"
              f"{c['slope']:>12.4g}{c['z']:>9.2f}{c['p']:>11.3g}")

    # ---- 2. 昼夜增暖不对称 ----
    print("\n" + "=" * 92)
    print("【表 2】昼夜增暖不对称：同一批站点，夜间 vs 白天 湿球趋势对比")
    print("=" * 92)
    tn = {t["station"]: t["slope"] for t in station_trends(rows, "tw_night_mean")}
    td = {t["station"]: t["slope"] for t in station_trends(rows, "tw_day_mean")}
    com = sorted(set(tn) & set(td))
    if com:
        dif = [tn[s] - td[s] for s in com]
        est, lo, hi = bootstrap_ci(dif)
        pos = sum(1 for x in dif if x > 0)
        print(f"  共同站点数: {len(com)}")
        print(f"  夜间湿球中位趋势: {median([tn[s] for s in com]):+.4f} °C/年")
        print(f"  白天湿球中位趋势: {median([td[s] for s in com]):+.4f} °C/年")
        print(f"  差值 (夜-昼) 中位: {est:+.4f} °C/年  95%CI [{lo:.4g}, {hi:.4g}]")
        print(f"  夜间增暖更快的站: {pos}/{len(com)} = {100*pos/len(com):.0f}%")
        print(f"  符号检验 p = {sign_test(pos, len(com)-pos):.3g}")
        out["asymmetry"] = dict(n=len(com), night=median([tn[s] for s in com]),
                                day=median([td[s] for s in com]), diff=est,
                                ci_lo=lo, ci_hi=hi, n_pos=pos,
                                frac=pos / len(com),
                                p=sign_test(pos, len(com) - pos))
    # 干球对照
    tnd = {t["station"]: t["slope"] for t in station_trends(rows, "tw_night_mean")}
    print()
    tt_n = {t["station"]: t["slope"] for t in station_trends(rows, "dtr_t")}
    tt_w = {t["station"]: t["slope"] for t in station_trends(rows, "dtr_tw")}
    com2 = sorted(set(tt_n) & set(tt_w))
    if com2:
        print(f"  干球日较差中位趋势: {median([tt_n[s] for s in com2]):+.4f} °C/年")
        print(f"  湿球日较差中位趋势: {median([tt_w[s] for s in com2]):+.4f} °C/年")
        out["dtr"] = dict(n=len(com2), dtr_t=median([tt_n[s] for s in com2]),
                          dtr_tw=median([tt_w[s] for s in com2]))

    # ---- 3. 时相对比 ----
    print("\n" + "=" * 92)
    print("【表 3】时相对比：E3 (2019-2023) 减 E1 (1976-1980)，配对差")
    print("=" * 92)
    print(f"{'指标':<18}{'站数':>5}{'中位变化':>11}{'95%CI':>24}{'正比例':>8}{'E1基准':>10}{'相对%':>8}")
    print("-" * 92)
    out["epoch"] = {}
    for m in KEY:
        e = epoch(rows, m, (1976, 1980), (2019, 2023))
        if not e:
            continue
        out["epoch"][m] = e
        ci = f"[{e['ci_lo']:.4g}, {e['ci_hi']:.4g}]"
        print(f"{m:<18}{e['n']:>5}{e['median']:>11.4g}{ci:>24}"
              f"{100*e['frac_pos']:>7.0f}%{(e['base'] or 0):>10.3g}"
              f"{(e['rel'] if e['rel'] is not None else 0):>8.1f}")

    # ---- 4. 纬度带 / 区域 ----
    print("\n" + "=" * 92)
    print("【表 4】按纬度带的 NRF 与昼夜湿球趋势")
    print("=" * 92)
    print(f"{'纬度带':<10}{'站数':>5}{'NRF中位':>10}{'NRF趋势/年':>12}{'夜间Tw趋势':>12}{'白天Tw趋势':>12}")
    print("-" * 92)
    out["bands"] = {}
    for b in ["tropics", "subtropics", "temperate", "boreal"]:
        sub = [r for r in rows if r.get("band") == b]
        if not sub:
            continue
        ts = station_trends(sub, "nrfr_elderly")
        a = agg(ts)
        vals = [r["nrfr_elderly"] for r in sub if r["ok_trend"]
                and r.get("nrfr_elderly") is not None]
        tsn = agg(station_trends(sub, "tw_night_mean"))
        tsd = agg(station_trends(sub, "tw_day_mean"))
        mv = median(vals)
        print(f"{b:<10}{(a['n'] if a else 0):>5}{(mv if mv is not None else 0):>10.3f}"
              f"{(a['median_slope'] if a else 0):>12.4g}"
              f"{(tsn['median_slope'] if tsn else 0):>12.4g}"
              f"{(tsd['median_slope'] if tsd else 0):>12.4g}")
        out["bands"][b] = dict(n=(a['n'] if a else 0), nrf_median=mv,
                               nrf_trend=(a['median_slope'] if a else None),
                               night=(tsn['median_slope'] if tsn else None),
                               day=(tsd['median_slope'] if tsd else None))

    # ---- 5. 当代气候态排名 ----
    print("\n" + "=" * 92)
    print("【表 5】当代 (2019-2023) 夜间恢复失效率最高的 20 个站点")
    print("=" * 92)
    recent = collections.defaultdict(list)
    for r in rows:
        if r["ok_trend"] and 2019 <= r["year"] <= 2023:
            if r.get("nrfr_elderly") is not None:
                recent[r["station"]].append(r)
    tab = []
    for st, rs in recent.items():
        if len(rs) < 2:
            continue
        v = sum(x["nrfr_elderly"] for x in rs) / len(rs)
        nm = sum(x["nhrs_elderly"] for x in rs if x.get("nhrs_elderly")) / \
             max(1, sum(1 for x in rs if x.get("nhrs_elderly")))
        tab.append((v, st, rs[0].get("name", ""), rs[0].get("country", ""),
                    rs[0]["lat"], rs[0]["lon"], nm, len(rs)))
    tab.sort(reverse=True)
    print(f"{'排名':<5}{'站点':<12}{'名称':<24}{'国':<4}{'纬度':>7}{'NRF率':>8}{'夜间超阈h/年':>13}")
    print("-" * 92)
    for i, (v, st, nm, c, la, lo, nh, ny) in enumerate(tab[:20], 1):
        print(f"{i:<5}{st:<12}{nm[:22]:<24}{c:<4}{la:>7.1f}{v:>8.3f}{nh:>13,.0f}")
    out["top_nrf"] = [dict(rank=i, station=st, name=nm, country=c, lat=la, lon=lo,
                           nrf=v, night_hours=nh)
                      for i, (v, st, nm, c, la, lo, nh, ny) in enumerate(tab[:30], 1)]

    # ---- 6. 干湿指标倒置 ----
    print("\n" + "=" * 92)
    print("【表 6】干球 vs 湿球指标的城市排序倒置")
    print("=" * 92)
    dd = collections.defaultdict(list)
    ww = collections.defaultdict(list)
    for r in rows:
        if r["ok_trend"] and 2019 <= r["year"] <= 2023:
            if r.get("t_mean") is not None:
                dd[r["station"]].append(r["t_mean"])
            if r.get("tw_mean") is not None:
                ww[r["station"]].append(r["tw_mean"])
    sd = sorted(((sum(v) / len(v), k) for k, v in dd.items() if len(v) >= 2), reverse=True)
    sw = sorted(((sum(v) / len(v), k) for k, v in ww.items() if len(v) >= 2), reverse=True)
    rd = {k: i + 1 for i, (v, k) in enumerate(sd)}
    rw = {k: i + 1 for i, (v, k) in enumerate(sw)}
    com = sorted(set(rd) & set(rw))
    shifts = [(rd[k] - rw[k], k) for k in com]
    thr_inv = max(5, int(0.15 * len(com)))
    big = [x for x in shifts if abs(x[0]) >= thr_inv]
    print(f"  共同站点 {len(com)} 个")
    print(f"  排名倒置 >= {thr_inv} 位 的站点: {len(big)} "
          f"({100*len(big)/max(1,len(com)):.0f}%)")
    print(f"\n  {'站点':<12}{'干球排名':>9}{'湿球排名':>9}{'位移':>7}")
    for s, k in sorted(shifts)[:10]:
        print(f"  {k:<12}{rd[k]:>9}{rw[k]:>9}{s:>7}")
    print("  ...")
    for s, k in sorted(shifts, reverse=True)[:10]:
        print(f"  {k:<12}{rd[k]:>9}{rw[k]:>9}{s:>7}")
    out["inversion"] = dict(n=len(com), n_invert=len(big),
                            frac=len(big) / max(1, len(com)))

    # ---- 6b. 干球判据 vs 湿球判据：幻影缓解 ----
    print("\n" + "=" * 92)
    print("【表 6b】不可用降温：干球的昼夜降幅中，人体真正能用上的有多少？")
    print("        (当代 2019-2023)")
    print("=" * 92)
    rec = [r for r in rows if r["ok_trend"] and 2019 <= r["year"] <= 2023]
    ag = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rec:
        for f in ["dtr_t", "dtr_tw", "unusable", "usable_frac"]:
            if r.get(f) is not None:
                ag[r["station"]][f].append(r[f])
    if ag:
        def m(f):
            v = [sum(x[f]) / len(x[f]) for x in ag.values() if x.get(f)]
            return median(v), len(v)
        a, na = m("dtr_t"); b, nb = m("dtr_tw")
        c, nc = m("unusable"); d, nd = m("usable_frac")
        print(f"  站点数 {len(ag)}")
        print(f"  干球日较差中位            {a:.2f} °C")
        print(f"  湿球日较差中位            {b:.2f} °C")
        print(f"  不可用降温中位            {c:.2f} °C")
        print(f"  **可用比例 (湿球/干球) 中位 {100*d:.0f}%**")
        print(f"  -> 干球呈现的昼夜降温中，约 {100*(1-d):.0f}% 是人体用不上的")
        print(f"\n  干球昼夜温差最大的站点，可用比例反而最低:")
        print(f"  {'站点':<12}{'名称':<24}{'干球日较差':>10}{'湿球日较差':>10}"
              f"{'不可用':>9}{'可用%':>8}")
        tab = []
        for st, x in ag.items():
            if not x.get("usable_frac"):
                continue
            tab.append((sum(x["dtr_t"]) / len(x["dtr_t"]), st))
        tab.sort(reverse=True)
        for dt, st in tab[:12]:
            x = ag[st]
            nm = [r.get("name", "") for r in rec if r["station"] == st]
            dtw = sum(x["dtr_tw"]) / len(x["dtr_tw"])
            uf = sum(x["usable_frac"]) / len(x["usable_frac"])
            print(f"  {st:<12}{(nm[0] if nm else '')[:22]:<24}{dt:>10.2f}"
                  f"{dtw:>10.2f}{dt-dtw:>9.2f}{100*uf:>8.0f}")
        # 趋势
        tu = agg(station_trends(rows, "unusable"))
        tf = agg(station_trends(rows, "usable_frac"))
        if tu:
            tu["sign_p"] = sign_test(tu["n_pos"], tu["n_neg"])
            print(f"\n  不可用降温趋势: {tu['median_slope']:+.4f} °C/年  "
                  f"95%CI [{tu['ci_lo']:.4g}, {tu['ci_hi']:.4g}]  "
                  f"{tu['n_pos']}正/{tu['n_neg']}负  "
                  f"符号检验 p={tu['sign_p']:.3g}")
        if tf:
            tf["sign_p"] = sign_test(tf["n_pos"], tf["n_neg"])
            print(f"  可用比例趋势  : {tf['median_slope']:+.5f} /年  "
                  f"95%CI [{tf['ci_lo']:.4g}, {tf['ci_hi']:.4g}]  "
                  f"{tf['n_pos']}正/{tf['n_neg']}负  "
                  f"符号检验 p={tf['sign_p']:.3g}")
        out["unusable"] = dict(n=len(ag), dtr_t=a, dtr_tw=b, unusable=c,
                               usable_frac=d,
                               trend_unusable=(tu["median_slope"] if tu else None),
                               trend_frac=(tf["median_slope"] if tf else None))

    # ---- 6c. 阈值敏感性阶梯 ----
    # 回应审稿人最强质疑："21.9 °C 源自日间暴晒研究，用于夜间是否成立？"
    # 若趋势在整个阶梯上都成立、且能定位夜间负担所处的温度区间，则结论稳健。
    print("\n" + "=" * 92)
    print("【表 6c】阈值敏感性阶梯：夜间恢复失效率对阈值取值的依赖")
    print("=" * 92)
    print(f"  {'阈值':>7}  {'来源':<26}{'E1 中位':>9}{'E3 中位':>9}"
          f"{'变化':>9}{'趋势/年':>11}{'符号检验p':>11}")
    lad = []
    src = {"elderly": "老年下限 (PK 2026)", "t23": "插值档", "t24": "插值档",
           "young": "青年下限 (PK 2026)", "psu": "PSU HEAT 实测临界",
           "classic": "经典不可生存阈值"}
    for k, th in sorted([("elderly", 21.9), ("t23", 23.0), ("t24", 24.0),
                         ("young", 25.8), ("psu", 30.6), ("classic", 35.0)],
                        key=lambda x: x[1]):
        e1 = [r[f"nrfr_{k}"] for r in rows
              if r["ok_trend"] and 1976 <= r["year"] <= 1980
              and r.get(f"nrfr_{k}") is not None]
        e3 = [r[f"nrfr_{k}"] for r in rows
              if r["ok_trend"] and 2019 <= r["year"] <= 2023
              and r.get(f"nrfr_{k}") is not None]
        tr = station_trends(rows, f"nrfr_{k}")
        agg_ = agg(tr)
        if not e1 or not e3:
            continue
        p = agg_.get("sign_p") if agg_ else None
        d = median(e3) - median(e1)
        print(f"  {th:>7.1f}  {src[k]:<26}{median(e1):>9.3f}{median(e3):>9.3f}"
              f"{d:>+9.3f}{(agg_['median_slope'] if agg_ else 0):>+11.5f}"
              f"{(f'{p:.2g}' if p is not None else '—'):>11}")
        lad.append(dict(threshold=th, source=src[k], e1=median(e1),
                        e3=median(e3), delta=d,
                        slope=(agg_["median_slope"] if agg_ else None),
                        sign_p=p))
    out["ladder"] = lad

    # ---- 7. 网络构成与非均一性诊断 ----
    print("\n" + "=" * 92)
    print("【表 7】网络构成与非均一性诊断")
    print("=" * 92)
    cnt = collections.Counter(r["year"] for r in ok)
    print("  逐年有效站数: " + "  ".join(f"{y}:{cnt[y]}" for y in sorted(cnt)))
    out["network"] = {int(y): cnt[y] for y in sorted(cnt)}

    # 跳变检测：逐年序列中出现远大于典型年际变幅的跃变，提示可能的非均一性
    jumps = []
    for st, rs in sorted(by_station(ok).items()):
        rs.sort(key=lambda x: x["year"])
        for m in ["tw_night_mean", "tw_day_mean"]:
            pairs = [(rs[i], rs[i + 1]) for i in range(len(rs) - 1)
                     if rs[i + 1]["year"] - rs[i]["year"] <= 3
                     and rs[i].get(m) is not None and rs[i + 1].get(m) is not None]
            if len(pairs) < 6:
                continue
            d = [abs(b[m] - a[m]) for a, b in pairs]
            med = median(d)
            if med and med > 0:
                for (a, b), x in zip(pairs, d):
                    if x > 4 * med and x > 0.5:
                        jumps.append((st, a["year"], b["year"], m, x))
    print(f"\n  潜在非均一跳变 (相邻年差 > 4 倍中位年际变幅且 > 0.5 °C): {len(jumps)} 处")
    seen = set()
    for st, y0, y1, m, x in jumps:
        if st in seen:
            continue
        seen.add(st)
        print(f"    {st:<12}{y0}->{y1}  {m}  跳变 {x:.2f} °C")
        if len(seen) >= 12:
            break
    out["jumps"] = [dict(station=s, y0=int(a), y1=int(b), metric=m, jump=round(x, 2))
                    for s, a, b, m, x in jumps]

    # ---- 8. 人口加权暴露 ----
    print("\n" + "=" * 92)
    print("【表 8】人口加权暴露 (50 km 全额 + 指数衰减至 300 km 的邻近城市人口)")
    print("=" * 92)
    pop = {}
    if os.path.exists("station_population.csv"):
        for r in csv.DictReader(open("station_population.csv", encoding="utf-8")):
            pop[r["station"]] = float(r["pop_50km"])
    if pop:
        def weighted(lo, hi, field):
            num = den = 0.0
            for st, p in pop.items():
                v = [r[field] for r in rows
                     if r["station"] == st and r["ok_trend"]
                     and lo <= r["year"] <= hi and r.get(field) is not None]
                if v:
                    x = sum(v) / len(v)
                    num += p * x
                    den += p
            return (num / den, den) if den else (None, 0)

        for field, unit, scale in [
                ("nrfr_elderly", "夜间恢复失效率", 1.0),
                ("nrd_elderly", "夜间恢复亏缺 °C", 1.0),
                ("nhrs_elderly", "夜间超阈小时/年", 1.0),
                ("relief_elderly", "每日可恢复窗口 小时", 1.0)]:
            a, pa = weighted(1976, 1980, field)
            b, pb = weighted(2019, 2023, field)
            if a is None or b is None:
                continue
            print(f"  {unit:<18} E1 {a*scale:>10.3f}   E3 {b*scale:>10.3f}   "
                  f"变化 {b-a:>+9.3f}  ({100*(b-a)/a if a else 0:>+6.1f}%)")
        print()
        # 人·小时：总夜间热暴露
        for lo, hi, lab in [(1976, 1980, "E1"), (2019, 2023, "E3")]:
            tot = 0.0
            for st, p in pop.items():
                v = [r["nhrs_elderly"] for r in rows
                     if r["station"] == st and r["ok_trend"]
                     and lo <= r["year"] <= hi and r.get("nhrs_elderly") is not None]
                if v:
                    tot += p * (sum(v) / len(v))
            print(f"  {lab} 夜间超阈人·小时/年: {tot/1e9:>10.1f} 十亿")
        out["population"] = dict(
            n_stations_with_pop=len(pop),
            total_pop=sum(pop.values()))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print("\n已写出 results.json")


if __name__ == "__main__":
    main()
