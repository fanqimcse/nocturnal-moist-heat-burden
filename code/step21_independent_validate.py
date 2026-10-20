#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R7-4 / R7-5（回应 M36）: 独立外部验证 —— 完全独立处理链的两套数据

上一轮 MERRA-2 趋势同号率仅 56%, 且 POWER 在 18 站一律为正、量级 3 倍。
本轮用两条与 GHCNh 亚小时链**完全独立**的证据来裁决:

  A. NASA GISTEMP v4 格点 (2 度, GHCNv4 月值)
     —— 逐站提取所在格点的年距平趋势, 检验我们的台站夜间干球增暖是否被独立确认。
     —— 同时用它裁决 POWER: 若 POWER 趋势系统性高于 GISTEMP 同格点趋势,
        则"POWER 一律为正且偏大"得到解释(再分析漂移), 而非我们的算法有问题。

  B. HadEX3 (v3.0.4, Met Office / Dunn et al. 2020)
     —— 台站基、独立 QC、独立 ADW 格点化、覆盖 37,000+ 站, 1901-2018。
     —— 四个夜间指数, 逐一对应我们的四个核心量:
          HadEX3 TN90p (暖夜占比)   <->  nrfr_elderly / fn_tn25
          HadEX3 TR    (热带夜 TN>20) <->  fn_tn20
          HadEX3 TNn   (最冷夜最低温) <->  tmin_night_med
          HadEX3 DTR   (日较差)      <->  dtr_t / dtr_tw

统一比较期: 1976-2018 (HadEX3 止于 2018)。
"""
import sys, os, csv, math, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
# netCDF-C / HDF5 在 Windows 上无法打开含非 ASCII 字符的路径,
# 故把外部 netCDF 复制到纯 ASCII 临时目录后再读。
NC = os.path.join(os.environ.get("TEMP", "/tmp"), "climnc")
if not os.path.isdir(NC):
    NC = os.path.join(HERE, "ext")
sys.path.insert(0, HERE)
from step6_trends import theil_sen, median

CMP = (1976, 2018)
OUR = {
    "t_mean": "全天干球均值(对照)",
    "t_night_mean": "夜间干球均值",
    "tw_night_mean": "夜间湿球均值",
    "tmin_night_med": "夜间干球最低温",
    "dtr_t": "干球日较差",
    "dtr_tw": "湿球日较差",
    "fn_tn20": "热夜占比 TN>=20",
    "fn_tn25": "热夜占比 TN>=25",
    "nrfr_elderly": "夜间恢复失效率",
}


def load_ours():
    d = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"), encoding="utf-8")):
        if r.get("ok_trend") != "1":
            continue
        y = int(r["year"])
        d[r["station"]][y] = r
        meta[r["station"]] = (float(r["lat"]), float(r["lon"]))
    return d, meta


def slope_ours(recs, key):
    ys, vs = [], []
    for y, r in sorted(recs.items()):
        if not CMP[0] <= y <= CMP[1]:
            continue
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            ys.append(y); vs.append(float(v))
        except Exception:
            pass
    if len(ys) < 12:
        return None
    return theil_sen(ys, vs)


def pearson(x, y):
    n = len(x)
    if n < 5:
        return None
    mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def pval(r, n):
    """相关系数的 t 检验双尾 p 值 (标准库实现)。"""
    if r is None or n < 5 or abs(r) >= 1:
        return None
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    # t -> p, 用不完全 Beta 的连分数展开 (Numerical Recipes betacf)
    df = n - 2
    x = df / (df + t * t)

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab = a + b; qap = a + 1.0; qam = a - 1.0
        c = 1.0; d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d; h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1 - x))
        bt = math.exp(lbeta)
        if x < (a + 1) / (a + b + 2):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1 - x) / b

    return betai(df / 2.0, 0.5, x)


def main():
    # netCDF4 在 Windows 上偶发 DLL 占用失败(并行进程/杀毒扫描), 重试导入
    nc = None
    for _ in range(12):
        try:
            import netCDF4 as nc
            break
        except Exception:
            import time
            time.sleep(2.0)
    if nc is None:
        raise SystemExit("netCDF4 无法导入")
    ours, meta = load_ours()
    print("=" * 104)
    print("R7-4 / R7-5  独立外部验证 (GISTEMP v4 格点 + HadEX3 夜间指数)")
    print(f"  比较期 {CMP[0]}-{CMP[1]}   参与站 {len(meta)}")
    print("=" * 104)

    # ---------------- GISTEMP ----------------
    g = nc.Dataset(os.path.join(NC, "gistemp250.nc"))
    glat = list(g.variables["lat"][:]); glon = list(g.variables["lon"][:])
    gt = list(g.variables["time"][:])
    ga = g.variables["tempanomaly"]
    # 年 -> (月索引列表)
    yr2idx = collections.defaultdict(list)
    for i, t in enumerate(gt):
        y = 1800 + int(t // 365.2425) + 1  # 近似, 下面用严格换算
    # 严格: days since 1800-01-01
    import datetime
    base = datetime.date(1800, 1, 1)
    for i, t in enumerate(gt):
        y = (base + datetime.timedelta(days=float(t))).year
        yr2idx[y].append(i)
    print(f"  GISTEMP 覆盖 {min(yr2idx)}-{max(yr2idx)}, 格点 {len(glat)}x{len(glon)}")

    def gi(lat, lon):
        return min(range(len(glat)), key=lambda i: abs(glat[i] - lat)), \
               min(range(len(glon)), key=lambda j: abs(glon[j] - lon))

    gist_tr = {}
    for st, (la, lo) in meta.items():
        i, j = gi(la, lo)
        ys, vs = [], []
        for y in range(CMP[0], CMP[1] + 1):
            idx = yr2idx.get(y)
            if not idx:
                continue
            s = n = 0.0
            for k in idx:
                v = ga[k, i, j]
                try:
                    v = float(v)
                except Exception:
                    continue
                if v == v and abs(v) < 90:
                    s += v; n += 1
            if n >= 8:
                ys.append(y); vs.append(s / n)
        if len(ys) >= 12:
            gist_tr[st] = theil_sen(ys, vs)
    print(f"  成功提取格点趋势: {len(gist_tr)} 站")

    print(f"\n【A 层】我们的台站 vs GISTEMP 同格点 (单位 C/年)")
    print(f"{'我们的指标':<20}{'站数':>6}{'同号率':>9}{'相关r':>9}{'我们中位':>12}{'GISTEMP中位':>13}{'比值':>8}")
    print("-" * 104)
    A = {}
    for k, lab in OUR.items():
        xs, ys_, oo = [], [], []
        for st in gist_tr:
            a = slope_ours(ours[st], k)
            if a is None or gist_tr[st] is None:
                continue
            xs.append(a); ys_.append(gist_tr[st]); oo.append(st)
        if len(xs) < 20:
            continue
        same = sum(1 for a, b in zip(xs, ys_) if (a > 0) == (b > 0))
        r = pearson(xs, ys_)
        ma, mb = median(xs), median(ys_)
        ratio = ma / mb if abs(mb) > 1e-12 else float("nan")
        A[k] = dict(n=len(xs), same=same / len(xs), r=r, ours=ma, gist=mb, ratio=ratio)
        print(f"{lab:<20}{len(xs):>6}{100*same/len(xs):>8.0f}%"
              f"{(r if r is not None else float('nan')):>9.3f}{ma:>+12.5f}{mb:>+13.5f}{ratio:>8.2f}")
    print("-" * 104)
    # 诊断: 我们的台站是否"代表性偏冷" —— 逐站列出差异最大的与最一致的
    dev = []
    for st, gsl in gist_tr.items():
        a = slope_ours(ours[st], "t_night_mean")
        if a is None or gsl is None:
            continue
        dev.append((a - gsl, st, a, gsl, meta[st]))
    dev.sort()
    print(f"\n  诊断: 夜间干球趋势 − GISTEMP 同格点趋势 (C/年)")
    print(f"    中位差 {median([d[0] for d in dev]):+.5f}   "
          f"(我们的台站系统性偏 {'低' if median([d[0] for d in dev])<0 else '高'})")
    print("    最不一致的 6 站 (我们 − GISTEMP):")
    for d, st, a, g_, (la, lo) in dev[:6]:
        print(f"      {st:<16} 我们 {a:+.5f}  GISTEMP {g_:+.5f}  差 {d:+.5f}  ({la:.1f},{lo:.1f})")

    # ---------------- HadEX3 ----------------
    H = {
        "TN90p": ("HadEX3 暖夜占比", "%/年", ["nrfr_elderly", "fn_tn25"]),
        "TR": ("HadEX3 热带夜 TN>20", "夜/年", ["fn_tn20", "nrfr_elderly"]),
        "TNn": ("HadEX3 最冷夜最低温", "C/年", ["tmin_night_med", "t_night_mean"]),
        "DTR": ("HadEX3 日较差", "C/年", ["dtr_t", "dtr_tw"]),
    }
    print(f"\n【B 层】HadEX3 夜间指数 (独立台站基格点, Dunn et al. 2020)")
    B = {}
    for idx, (lab, unit, ours_keys) in H.items():
        p = os.path.join(NC, f"hadex3_{idx}.nc")
        if not os.path.exists(p):
            print(f"  缺 {p}")
            continue
        d = nc.Dataset(p)
        ha = d.variables["Ann"]
        hlat = [float(x) for x in d.variables["latitude"][:]]
        hlon = [float(x) for x in d.variables["longitude"][:]]
        ht = [float(x) for x in d.variables["time"][:]]
        y0, y1 = int(ht[0]), int(ht[-1])

        def hi(la, lo):
            L = lo % 360.0
            return (min(range(len(hlat)), key=lambda i: abs(hlat[i] - la)),
                    min(range(len(hlon)), key=lambda j: abs(hlon[j] - L)))

        cache = {}
        htr = {}
        for st, (la, lo) in meta.items():
            i, j = hi(la, lo)
            key = (i, j)
            if key not in cache:
                ys, vs = [], []
                for t in range(y0, y1 + 1):
                    if not CMP[0] <= t <= CMP[1]:
                        continue
                    k = t - y0
                    v = ha[k, i, j]
                    try:
                        v = float(v)
                    except Exception:
                        v = float("nan")
                    if v == v and abs(v) < 1e6:
                        ys.append(t); vs.append(v)
                cache[key] = theil_sen(ys, vs) if len(ys) >= 12 else None
            if cache[key] is not None:
                htr[st] = cache[key]
        d.close()

        print(f"\n  -- {lab} ({unit}), 可用站 {len(htr)}")
        print(f"  {'我们的指标':<20}{'站数':>6}{'同号率':>9}{'相关r':>9}{'r的p值':>10}"
              f"{'我们的中位':>13}{'HadEX3中位':>13}")
        for k in ours_keys:
            xs, ys_ = [], []
            for st in htr:
                a = slope_ours(ours[st], k)
                if a is None:
                    continue
                xs.append(a); ys_.append(htr[st])
            if len(xs) < 20:
                continue
            same = sum(1 for a, b in zip(xs, ys_) if (a > 0) == (b > 0))
            r = pearson(xs, ys_)
            # 仅用趋势非退化的站 (我们的 |斜率| > 0) 再看一次 —— 大量恒 0 会污染同号率
            nz = [(a, b) for a, b in zip(xs, ys_) if abs(a) > 1e-12]
            nzr = pearson([a for a, _ in nz], [b for _, b in nz]) if len(nz) >= 20 else None
            nzs = (sum(1 for a, b in nz if (a > 0) == (b > 0)) / len(nz)) if nz else float("nan")
            pv = pval(r, len(xs))
            B[(idx, k)] = dict(n=len(xs), same=same / len(xs), r=r, p=pv,
                               nz_n=len(nz), nz_r=nzr, nz_same=nzs,
                               ours=median(xs), hadex=median(ys_))
            print(f"  {OUR[k]:<20}{len(xs):>6}{100*same/len(xs):>8.0f}%"
                  f"{(r if r is not None else float('nan')):>9.3f}"
                  f"{(pv if pv is not None else float('nan')):>10.4f}"
                  f"{median(xs):>+13.5f}{median(ys_):>+13.5f}")
            if nzr is not None:
                print(f"  {'  └ 仅非零站 ('+str(len(nz))+' 站)':<20}{len(nz):>6}"
                      f"{100*nzs:>8.0f}%{nzr:>9.3f}")

    # ---------------- POWER 裁决 ----------------
    print(f"\n【C 层】POWER/MERRA-2 漂移裁决: POWER 趋势 vs GISTEMP 同格点趋势")
    print("  逻辑: 若 POWER 在同格点上系统性高于 GISTEMP, 则上一轮 56% 同号率")
    print("        应归因于再分析暖漂移, 而非本研究算法。")
    txt = os.path.join(HERE, "external_validation.txt")
    pw = {}
    if os.path.exists(txt):
        started = False
        for line in open(txt, encoding="utf-8"):
            s = line.strip()
            if "外部验证 B" in s:
                started = True; continue
            if not started:
                continue
            p = s.split()
            # 表列: 站点, 纬度, 站点斜率, POWER斜率, [95%CI], 同号
            if len(p) >= 4 and p[0][:1].isalnum() and p[0][:3].isupper():
                try:
                    pw[p[0]] = float(p[3])     # POWER 斜率
                except Exception:
                    pass
    print(f"  解析到 POWER 逐站趋势 {len(pw)} 站")

    # 同格点 GISTEMP 趋势 (POWER 比较期为 1981-2023)
    def gist_slope_range(la, lo, y0, y1):
        i, j = gi(la, lo)
        ys, vs = [], []
        for y in range(y0, y1 + 1):
            idx = yr2idx.get(y)
            if not idx:
                continue
            s = n = 0.0
            for k in idx:
                try:
                    v = float(ga[k, i, j])
                except Exception:
                    continue
                if v == v and abs(v) < 90:
                    s += v; n += 1
            if n >= 8:
                ys.append(y); vs.append(s / n)
        return theil_sen(ys, vs) if len(ys) >= 12 else None

    rows = []
    for st, sl in pw.items():
        if st not in meta:
            continue
        la, lo = meta[st]
        gsl = gist_slope_range(la, lo, 1981, 2023)
        if gsl is None:
            continue
        osl = None
        ys, vs = [], []
        for y, r in sorted(ours[st].items()):
            if not 1981 <= y <= 2023:
                continue
            v = r.get("tw_night_mean")
            if v in (None, "", "None"):
                continue
            try:
                ys.append(y); vs.append(float(v))
            except Exception:
                pass
        if len(ys) >= 12:
            osl = theil_sen(ys, vs)
        rows.append((st, osl, sl, gsl))
    if rows:
        med_o = median([r[1] for r in rows if r[1] is not None])
        med_p = median([r[2] for r in rows])
        med_g = median([r[3] for r in rows])
        print(f"{'站点':<16}{'我们(湿球夜间)':>16}{'POWER':>10}{'GISTEMP同格点':>15}")
        print("-" * 104)
        for st, o, p_, g_ in rows:
            print(f"{st:<16}{(o if o is not None else float('nan')):>+16.5f}"
                  f"{p_:>+10.5f}{g_:>+15.5f}")
        print("-" * 104)
        print(f"  中位:  我们 {med_o:+.5f}   POWER {med_p:+.5f}   GISTEMP {med_g:+.5f} °C/年")
        print(f"  POWER / GISTEMP  = {med_p/med_g if med_g else float('nan'):.2f}")
        print(f"  我们   / GISTEMP = {med_o/med_g if med_g else float('nan'):.2f}")
        if med_g and med_p / med_g > 1.5:
            print("  >>> 判定: POWER 增暖趋势显著高于同格点 GISTEMP, 存在系统性暖漂移。")
            print("      上一轮 56% 同号率应归因于 MERRA-2, 而非本研究算法。")
        else:
            print("  >>> 判定: POWER 与 GISTEMP 量级相当, 漂移假设未获支持。")
    print()

    g.close()

    # 保存
    import json
    with open(os.path.join(HERE, "independent_validation.json"), "w", encoding="utf-8") as f:
        json.dump({"gistemp": A,
                   "hadex3": {f"{a}|{b}": v for (a, b), v in B.items()},
                   "gistemp_station_trend": {k: v for k, v in list(gist_tr.items())[:200]}},
                  f, ensure_ascii=False, indent=2)
    print("\n已保存 independent_validation.json")


if __name__ == "__main__":
    main()
