#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 9: 稳健性三重检验（回应审稿人 M23 / M24 / M25）

1) 空间自相关校正
   - 站间去趋势异常的相关矩阵 -> 平均成对相关 r_bar
   - Bretherton 有效样本量  N_eff = N / (1 + (N-1) * r_bar)
   - 空间校正后的符号检验（把计数按 N_eff/N 缩放）
   - 20°x20° 网格区域块 bootstrap 重估中位斜率的显著性与 CI

2) 相对均一化（Pettitt + 邻站参考序列）
   - 每站构造邻站参考异常序列（距离 < 1500 km 且相关最高）
   - 差分序列 d = 本站异常 - 参考异常，Pettitt 检验变点
   - 显著则订正：断点后整段减去位移量
   - 报告订正前后的趋势与显著性

3) 城市化（UHI）混杂分层
   - 按站点 50 km 人口分四层，比较各层中位趋势与符号检验

输出: robust_summary.txt, robust.json
"""
import sys, os, csv, json, math, random, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step6_trends import theil_sen, mann_kendall, bh_fdr, sign_test, median

random.seed(20261019)

KEY = ["dtr_tw", "dtr_t", "usable_frac", "unusable", "nrfr_elderly",
       "nrd_elderly", "tw_night_mean", "relief_elderly", "nhrs_elderly"]
LABEL = {
    "dtr_tw": "湿球日较差 (°C)",
    "dtr_t": "干球日较差 (°C)",
    "usable_frac": "可用降温比例",
    "unusable": "不可用降温 (°C)",
    "nrfr_elderly": "夜间恢复失效率 (21.9°C)",
    "nrd_elderly": "夜间恢复亏缺 (°C)",
    "tw_night_mean": "夜间湿球均值 (°C)",
    "relief_elderly": "每日可恢复窗口 (h)",
    "nhrs_elderly": "夜间超阈等效小时 (h)",
}


# ---------------- 基础统计 ----------------
def pearson(a, b):
    n = len(a)
    if n < 8:
        return None
    ma = sum(a) / n; mb = sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((x - mb) ** 2 for x in b) ** 0.5
    if sa <= 0 or sb <= 0:
        return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (sa * sb)


def detrend_anom(years, vals):
    """去掉线性趋势后的距平序列。"""
    sl = theil_sen(years, vals)
    if sl is None:
        sl = 0.0
    m = sum(vals) / len(vals)
    ms = sum(years) / len(years)
    return [v - (m + sl * (y - ms)) for y, v in zip(years, vals)], sl


def pettitt(x):
    """Pettitt 变点检验: 返回 (变点下标, 统计量 K, 近似 p)。"""
    n = len(x)
    if n < 8:
        return None, 0.0, 1.0
    order = sorted(range(n), key=lambda i: x[i])
    # 并列取平均秩
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    S = 0.0; best_k = None; best_U = 0.0
    for k in range(1, n):
        S += rank[k - 1]
        U = 2.0 * S - k * (n + 1)
        if abs(U) > abs(best_U):
            best_U = U; best_k = k
    K = abs(best_U)
    p = min(1.0, 2.0 * math.exp(-6.0 * K * K / (n ** 3 + n ** 2)))
    return best_k, K, p


def hav_km(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = p2 - p1
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def grid_cell(lat, lon):
    return (int(math.floor(lat / 20.0)), int(math.floor(lon / 20.0)))


# ---------------- 载入 ----------------
def load(path="metrics_v3.csv"):
    rows = list(csv.DictReader(open(os.path.join(HERE, path), encoding="utf-8")))
    out = collections.defaultdict(dict)   # station -> year -> {metric: val}
    meta = {}
    for r in rows:
        st = r["station"]; y = int(r["year"])
        d = {}
        for k in KEY:
            v = r.get(k, "")
            try:
                d[k] = float(v)
            except Exception:
                d[k] = None
        out[st][y] = d
        meta[st] = {"lat": float(r["lat"]), "lon": float(r["lon"]),
                    "elev": float(r.get("elev", 0) or 0)}
    return out, meta


def load_pop():
    p = os.path.join(HERE, "station_population.csv")
    if not os.path.exists(p):
        return {}
    return {r["station"]: float(r["pop_50km"])
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def series(data, st, metric, min_years=8):
    ys = sorted(y for y, d in data[st].items() if d.get(metric) is not None)
    if len(ys) < min_years:
        return None, None
    return ys, [data[st][y][metric] for y in ys]


# ---------------- 1) 空间自相关 ----------------
def spatial_neff(data, meta, metric, min_overlap=10):
    """站间去趋势异常的平均成对相关 -> N_eff。"""
    anom = {}; years = {}
    for st in data:
        ys, vs = series(data, st, metric)
        if ys is None:
            continue
        a, _ = detrend_anom(ys, vs)
        anom[st] = a; years[st] = ys
    sts = sorted(anom)
    rs = []
    pairs = 0
    for i in range(len(sts)):
        for j in range(i + 1, len(sts)):
            A, B = sts[i], sts[j]
            yi, yj = set(years[A]), set(years[B])
            common = sorted(yi & yj)
            if len(common) < min_overlap:
                continue
            ai = [anom[A][years[A].index(y)] for y in common]
            bj = [anom[B][years[B].index(y)] for y in common]
            r = pearson(ai, bj)
            if r is not None:
                rs.append(r); pairs += 1
    if not rs:
        return len(sts), 0.0, 0
    rbar = sum(rs) / len(rs)
    N = len(sts)
    neff = N / (1.0 + (N - 1) * rbar) if rbar > 0 else float(N)
    return max(2.0, neff), rbar, pairs


def scaled_sign_test(pos, neg, neff):
    """把计数缩放到有效样本量后再做精确二项检验。"""
    N = pos + neg
    if N == 0 or neff >= N:
        return sign_test(pos, neg), pos, neg
    f = neff / N
    p2 = max(1, int(round(pos * f)))
    n2 = max(1, int(round(neg * f)))
    while p2 + n2 > int(round(neff)) and (p2 + n2) > 2:
        if p2 >= n2:
            p2 -= 1
        else:
            n2 -= 1
    return sign_test(p2, n2), p2, n2


def block_bootstrap(data, meta, metric, n_boot=4000, min_years=8):
    """按 20°x20° 网格分块, 重采样地块 -> 中位斜率的分布。"""
    slopes = {}
    for st in data:
        ys, vs = series(data, st, metric, min_years)
        if ys is None:
            continue
        sl = theil_sen(ys, vs)
        if sl is not None:
            slopes[st] = sl
    cells = collections.defaultdict(list)
    for st, sl in slopes.items():
        cells[grid_cell(meta[st]["lat"], meta[st]["lon"])].append(sl)
    cell_keys = list(cells)
    if len(cell_keys) < 3:
        return None
    obs = median(list(slopes.values()))
    reps = []
    for _ in range(n_boot):
        pick = [cells[cell_keys[random.randrange(len(cell_keys))]]
                for _ in range(len(cell_keys))]
        flat = [v for c in pick for v in c]
        if flat:
            reps.append(median(flat))
    reps.sort()
    lo = reps[int(0.025 * len(reps))]
    hi = reps[int(0.975 * len(reps))]
    npos = sum(1 for v in reps if v > 0)
    nneg = sum(1 for v in reps if v < 0)
    ntot = max(1, npos + nneg)
    p = min(1.0, 2.0 * min(npos, nneg) / ntot)
    frac_neg = nneg / ntot
    return {"obs": obs, "lo": lo, "hi": hi, "p": p,
            "frac_neg": frac_neg, "n_cells": len(cell_keys),
            "n_stations": len(slopes)}


# ---------------- 2) 相对均一化 ----------------
def relative_homogenize(data, meta, metric, max_km=1500.0,
                        n_ref=5, min_overlap=10, p_thr=0.05):
    """邻站参考 + Pettitt。返回每站订正前后的斜率。"""
    anom = {}; years = {}; raw = {}
    for st in data:
        ys, vs = series(data, st, metric)
        if ys is None:
            continue
        a, sl = detrend_anom(ys, vs)
        anom[st] = a; years[st] = ys; raw[st] = (ys, vs, sl)
    sts = sorted(anom)
    # 相关矩阵
    corr = collections.defaultdict(dict)
    for i in range(len(sts)):
        for j in range(i + 1, len(sts)):
            A, B = sts[i], sts[j]
            common = sorted(set(years[A]) & set(years[B]))
            if len(common) < min_overlap:
                continue
            ai = [anom[A][years[A].index(y)] for y in common]
            bj = [anom[B][years[B].index(y)] for y in common]
            r = pearson(ai, bj)
            if r is not None:
                corr[A][B] = r; corr[B][A] = r

    result = {}
    for st in sts:
        ys, vs, sl_raw = raw[st]
        dist = {o: hav_km(meta[st]["lat"], meta[st]["lon"],
                          meta[o]["lat"], meta[o]["lon"])
                for o in corr.get(st, {})}
        cand = sorted([o for o in corr.get(st, {}) if dist.get(o, 9e9) <= max_km],
                      key=lambda o: -corr[st][o])[:n_ref]
        if len(cand) < 3:
            result[st] = {"slope_raw": sl_raw, "slope_adj": sl_raw,
                          "break": None, "p": None, "shift": 0.0,
                          "n_ref": len(cand)}
            continue
        # 参考序列 = 邻站异常的中位（对齐到本站年份）
        refs = []
        for y in ys:
            col = []
            for o in cand:
                if y in years[o]:
                    col.append(anom[o][years[o].index(y)])
            refs.append(median(col) if len(col) >= 2 else None)
        ok = [i for i, v in enumerate(refs) if v is not None]
        if len(ok) < 10:
            result[st] = {"slope_raw": sl_raw, "slope_adj": sl_raw,
                          "break": None, "p": None, "shift": 0.0,
                          "n_ref": len(cand)}
            continue
        d = [anom[st][i] - refs[i] for i in ok]
        dy = [ys[i] for i in ok]
        k, K, p = pettitt(d)
        shift = 0.0; brk = None
        if p is not None and p < p_thr and k is not None:
            before = d[:k]; after = d[k:]
            if len(before) >= 3 and len(after) >= 3:
                shift = median(after) - median(before)
                brk = dy[k]
        # 订正：断点之后的原始值减去位移
        vs_adj = list(vs)
        if brk is not None:
            for i, y in enumerate(ys):
                if y >= brk:
                    vs_adj[i] = vs[i] - shift
        sl_adj = theil_sen(ys, vs_adj)
        result[st] = {"slope_raw": sl_raw, "slope_adj": sl_adj,
                      "break": brk, "p": p, "shift": shift,
                      "n_ref": len(cand)}
    return result


# ---------------- 3) UHI 分层 ----------------
def uhi_strata(data, meta, metric, pop):
    vals = []
    for st in data:
        if st not in pop:
            continue
        ys, vs = series(data, st, metric)
        if ys is None:
            continue
        sl = theil_sen(ys, vs)
        if sl is not None:
            vals.append((pop[st], sl, st))
    if len(vals) < 12:
        return None
    vals.sort()
    q = len(vals) // 4
    out = []
    for i in range(4):
        seg = vals[i * q:(i + 1) * q] if i < 3 else vals[3 * q:]
        if len(seg) < 4:
            continue
        sls = [s for _, s, _ in seg]
        pos = sum(1 for s in sls if s > 0); neg = sum(1 for s in sls if s < 0)
        out.append({"q": i + 1, "n": len(seg),
                    "pop_lo": seg[0][0], "pop_hi": seg[-1][0],
                    "med": median(sls), "pos": pos, "neg": neg,
                    "p_sign": sign_test(pos, neg)})
    return out


# ---------------- 主流程 ----------------
def main():
    data, meta = load()
    pop = load_pop()
    print("=" * 90)
    print("【稳健性表 R1】空间自相关校正：朴素符号检验 vs 空间校正后")
    print("=" * 90)
    print(f"{'指标':<26}{'站数':>5}{'r_bar':>8}{'N_eff':>7}"
          f"{'正/负':>9}{'朴素p':>11}{'校正p':>12}{'块boot p':>11}")
    print("-" * 90)

    res = {}
    for m in KEY:
        ys_all = {}
        pos = neg = 0
        slopes = []
        for st in data:
            ys, vs = series(data, st, m)
            if ys is None:
                continue
            sl = theil_sen(ys, vs)
            if sl is None:
                continue
            slopes.append(sl)
            if sl > 0:
                pos += 1
            elif sl < 0:
                neg += 1
        if len(slopes) < 10:
            continue
        neff, rbar, npair = spatial_neff(data, meta, m)
        p_naive = sign_test(pos, neg)
        p_adj, p2, n2 = scaled_sign_test(pos, neg, neff)
        bb = block_bootstrap(data, meta, m)
        res[m] = {"n": len(slopes), "r_bar": rbar, "n_eff": neff,
                  "pos": pos, "neg": neg, "p_naive": p_naive,
                  "p_spatial": p_adj, "p_block": bb["p"] if bb else None,
                  "med": median(slopes), "boot_lo": bb["lo"] if bb else None,
                  "boot_hi": bb["hi"] if bb else None,
                  "n_cells": bb["n_cells"] if bb else None,
                  "frac_neg": bb["frac_neg"] if bb else None}
        print(f"{LABEL[m]:<26}{len(slopes):>5}{rbar:>8.3f}{neff:>7.1f}"
              f"{str(pos)+'/'+str(neg):>9}{p_naive:>11.2e}{p_adj:>12.2e}"
              f"{(bb['p'] if bb else float('nan')):>11.2e}")

    print("\n说明: r_bar 为去趋势年距平的平均站间相关; "
          "N_eff = N/(1+(N-1)r_bar) 为有效独立样本量;")
    print("      校正 p 把计数缩放到 N_eff 后重算二项检验; "
          "块 boot 按 20°x20° 网格重采样地块(保留空间结构)。")

    print("\n" + "=" * 90)
    print("【稳健性表 R2】相对均一化（邻站参考 + Pettitt）：订正前后趋势对比")
    print("=" * 90)
    print(f"{'指标':<26}{'站数':>5}{'检出断点':>9}{'中位位移':>10}"
          f"{'订正前中位斜率':>16}{'订正后中位斜率':>16}{'比值':>8}")
    print("-" * 90)
    hom = {}
    for m in KEY:
        h = relative_homogenize(data, meta, m)
        hom[m] = h
        raws = [v["slope_raw"] for v in h.values() if v["slope_raw"] is not None]
        adjs = [v["slope_adj"] for v in h.values() if v["slope_adj"] is not None]
        nb = sum(1 for v in h.values() if v["break"] is not None)
        shifts = [v["shift"] for v in h.values() if v["break"] is not None]
        mr = median(raws); ma = median(adjs)
        ratio = (ma / mr) if (mr not in (0, None) and ma is not None) else None
        res[m]["homog"] = {"n_break": nb, "n": len(raws),
                           "med_raw": mr, "med_adj": ma,
                           "med_shift": median(shifts) if shifts else 0.0,
                           "ratio": ratio}
        print(f"{LABEL[m]:<26}{len(raws):>5}{nb:>9}"
              f"{(median(shifts) if shifts else 0.0):>10.3f}"
              f"{mr:>16.5f}{ma:>16.5f}"
              f"{(ratio if ratio is not None else float('nan')):>8.2f}")

    print("\n" + "=" * 90)
    print("【稳健性表 R3】城市化混杂：按站点 50 km 人口四分位分层")
    print("=" * 90)
    for m in KEY:
        s = uhi_strata(data, meta, m, pop)
        if not s:
            continue
        print(f"\n  {LABEL[m]}")
        print(f"    {'四分位':<8}{'站数':>5}{'人口区间(百万)':>22}"
              f"{'中位斜率':>14}{'正/负':>9}{'符号p':>10}")
        for row in s:
            print(f"    Q{row['q']:<7}{row['n']:>5}"
                  f"{row['pop_lo']/1e6:>10.2f}-{row['pop_hi']/1e6:<11.2f}"
                  f"{row['med']:>14.5f}{str(row['pos'])+'/'+str(row['neg']):>9}"
                  f"{(row['p_sign'] if row['p_sign'] is not None else float('nan')):>10.2e}")
        res[m]["uhi"] = s

    json.dump(res, open(os.path.join(HERE, "robust.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写出 robust.json")


if __name__ == "__main__":
    main()
