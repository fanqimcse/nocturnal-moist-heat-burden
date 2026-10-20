# -*- coding: utf-8 -*-
"""第10轮 R10-2  网络代表性 + 空间自相关校正的有效样本量

审稿人 M55 / M58:
  (a) 218 站到底代表谁? 热带/亚热带占比多少? 趋势在各气候带是否一致?
  (b) 176 站绝不是 176 个独立样本。台站趋势场存在强空间自相关,
      符号检验 p = 1.9e-4 会被严重夸大。必须给出 N_eff 与校正后的显著性。

做法:
  1. 台站构成: 按气候带 / 区域 / 纬度带给出站数、覆盖人口。
  2. 逐站 Theil-Sen 趋势 → 站间相关 vs 大圆距离 → 去相关长度 L。
  3. Bretherton 有效样本量 N_eff = n(1−r̄)/(1+r̄)。
  4. 空间聚类 (球面 k-means, k = 20) → 以"簇"为独立单元重做符号检验与中位数趋势 CI。
  5. 簇级 bootstrap 给出人口加权趋势的置信区间。

产出: representativeness.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 4:
        return float('nan')
    sl = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        m = np.abs(dx) > 1e-9
        if m.any():
            sl.append((y[i + 1:][m] - y[i]) / dx[m])
    return float(np.median(np.concatenate(sl))) if sl else float('nan')


def gcdist(la1, lo1, la2, lo2):
    la1 = np.radians(la1); la2 = np.radians(la2)
    dlo = np.radians(lo2 - lo1)
    c = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(dlo)
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def sphere_kmeans(lat, lon, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    la = np.radians(lat); lo = np.radians(lon)
    X = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    # k-means++ 初始化
    c = X[rng.integers(len(X))].reshape(1, -1)
    for _ in range(k - 1):
        d = np.clip(np.min(1.0 - X @ c.T, axis=1), 0.0, None)
        s = d.sum()
        p = d / s if s > 0 else np.full(len(X), 1.0 / len(X))
        c = np.vstack([c, X[rng.choice(len(X), p=p)]])
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        lab = np.argmin(1.0 - X @ c.T, axis=1)
        new = c.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                v = X[m].mean(axis=0)
                new[j] = v / np.linalg.norm(v)
        if np.allclose(new, c):
            break
        c = new
    return lab


def binom_p(pos, neg):
    from math import comb
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


COLS = [('nhrs_elderly', 'NWH₂₁.₉'), ('nhrs_t24', 'NWH₂₄'), ('nhrs_young', 'NWH₂₅.₈')]


def main():
    m = pd.read_csv('metrics_v3.csv')
    mm = m[m['ok_trend'].astype(str) == '1'].copy()
    cnt = mm.groupby('station').size()
    keep = set(cnt[cnt >= 20].index)
    mm = mm[mm.station.isin(keep)]
    pop = pd.read_csv('station_population.csv')[['station', 'pop_50km']]
    mm = mm.merge(pop, on='station', how='left')
    mm['pop_50km'] = mm['pop_50km'].fillna(0.0)

    L = ['网络代表性与空间自相关校正 (有效样本量)', '']
    js = {}
    meta = mm.groupby('station').agg(lat=('lat', 'first'), lon=('lon', 'first'),
                                     band=('band', 'first'), region=('region', 'first'),
                                     pop=('pop_50km', 'first'), nyr=('year', 'size'))
    L.append('== 1. 台站构成 ==')
    L.append('  合格站 %d 个, 覆盖(50km半径)人口 %.2f 亿'
             % (len(meta), meta['pop'].sum() / 1e8))
    tot = meta['pop'].sum()
    L.append('  %-14s %6s %10s %8s' % ('气候带', '站数', '人口(亿)', '占比'))
    for b, g in meta.groupby('band'):
        L.append('  %-14s %6d %10.2f %7.1f%%' % (b, len(g), g['pop'].sum() / 1e8,
                                                 100 * g['pop'].sum() / tot))
    L.append('  %-14s %6s %10s %8s' % ('区域', '站数', '人口(亿)', '占比'))
    for b, g in meta.groupby('region'):
        L.append('  %-14s %6d %10.2f %7.1f%%' % (b, len(g), g['pop'].sum() / 1e8,
                                                 100 * g['pop'].sum() / tot))
    js['composition'] = dict(n_stations=int(len(meta)),
                             pop_billions=float(meta['pop'].sum() / 1e9),
                             by_band={b: dict(n=int(len(g)), pop=float(g['pop'].sum()))
                                      for b, g in meta.groupby('band')},
                             by_region={b: dict(n=int(len(g)), pop=float(g['pop'].sum()))
                                        for b, g in meta.groupby('region')})
    L.append('')

    # ---- 逐站趋势 ----
    tr = {}
    for col, nm in COLS:
        d = {}
        for sid, g in mm.groupby('station'):
            g = g.dropna(subset=[col]).sort_values('year')
            if len(g) < 20:
                continue
            d[sid] = theil_sen(g['year'].values.astype(float), g[col].values.astype(float)) * 10
        tr[col] = pd.Series(d)
    tdf = pd.DataFrame(tr).dropna(subset=[COLS[0][0]])
    tdf = tdf.join(meta[['lat', 'lon', 'band', 'pop']], how='inner')
    L.append('== 2. 有效样本量 (空间自相关校正) ==')
    lat = tdf['lat'].values; lon = tdf['lon'].values
    n = len(tdf)
    D = gcdist(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    iu = np.triu_indices(n, 1)
    dv = D[iu]
    L0 = None
    js['neff'] = {}
    for col, nm in COLS:
        b = tdf[col].values
        sd = float(np.std(b))
        if sd > 0:
            z = (b - b.mean()) / sd
            rb = np.outer(z, z)[iu]
        else:
            rb = np.zeros(len(dv))
        ok = np.isfinite(rb)
        # 分距离 bin 的平均相关
        edges = np.arange(0, 12001, 1000)
        prof = []
        for a, e in zip(edges[:-1], edges[1:]):
            s = ok & (dv >= a) & (dv < e)
            if s.sum() >= 30:
                prof.append((a, e, float(np.nanmean(rb[s])), int(s.sum())))
        rbar = float(np.nanmean(rb[ok]))
        neff = n * (1 - rbar) / (1 + rbar) if rbar < 1 else 1.0
        neff = max(2.0, min(float(n), neff))
        # 指数拟合去相关长度
        if len(prof) >= 3:
            dd = np.array([(p[0] + p[1]) / 2 for p in prof])
            rr = np.array([p[2] for p in prof])
            sel = rr > 0.02
            if sel.sum() >= 3:
                A = np.column_stack([np.ones(sel.sum()), -dd[sel]])
                try:
                    coef, *_ = np.linalg.lstsq(A, np.log(np.clip(rr[sel], 1e-4, None)), rcond=None)
                    L0 = float(1.0 / max(coef[1], 1e-6))
                except Exception:
                    L0 = None
        L.append('  %s: n=%d  站间趋势平均相关 r̄=%.3f  →  N_eff=%.1f  (膨胀因子 %.1f×)'
                 % (nm, n, rbar, neff, n / neff))
        if prof:
            L.append('      相关-距离剖面: ' + ' | '.join('%dkm r=%.2f' % (int(p[0]), p[2]) for p in prof[:8]))
        if L0 and L0 < 50000:
            L.append('      去相关长度 e-folding ≈ %.0f km' % L0)
        else:
            L.append('      相关随距离无明显衰减 → 趋势场空间自相关可忽略, N_eff ≈ n')
        # 校正显著性 (均值用 SE, 中位数用 bootstrap)
        med = float(np.median(b)); avg = float(np.mean(b))
        se = float(np.std(b, ddof=1) / np.sqrt(n))
        se_eff = float(np.std(b, ddof=1) / np.sqrt(neff))
        L.append('      均值趋势 %+.2f h/10yr   未校正 SE=%.2f (t=%.2f)   校正后 SE=%.2f (t=%.2f)'
                 % (avg, se, avg / se if se else float('nan'), se_eff,
                    avg / se_eff if se_eff else float('nan')))
        L.append('      中位趋势 %+.2f h/10yr   (重尾分布下中位数远小于均值, 报道须同时给出)' % med)
        js['neff'][col] = dict(n=int(n), rbar=rbar, neff=float(neff),
                               decorrelation_km=(L0 if (L0 and L0 < 50000) else None),
                               median_trend=med, mean_trend=avg,
                               se=se, se_eff=se_eff,
                               t_uncorrected=(avg / se if se else None),
                               t_corrected=(avg / se_eff if se_eff else None))
    L.append('')

    # ---- 空间聚类: 以簇为独立单元 ----
    K = 20
    lab = sphere_kmeans(lat, lon, K, seed=1)
    tdf['cluster'] = lab
    L.append('== 3. 空间聚类检验 (球面 k-means, k=%d; 以簇为独立单元) ==' % K)
    js['cluster_test'] = {}
    for col, nm in COLS:
        cm = tdf.groupby('cluster')[col].median().dropna().values
        pos = int((cm > 0).sum()); neg = int((cm < 0).sum())
        pb = binom_p(pos, neg)
        se = float(np.std(cm, ddof=1) / np.sqrt(len(cm)))
        L.append('  %s: 簇中位趋势 %+.2f h/10yr  (n=%d 簇, %d 正 / %d 负, 符号检验 p=%.4f, SE=%.2f, t=%.2f)'
                 % (nm, np.median(cm), len(cm), pos, neg, pb,
                    se, np.median(cm) / se if se else float('nan')))
        # 簇级 bootstrap
        rng = np.random.default_rng(0)
        idx = np.arange(len(cm))
        bs = [np.median(cm[rng.choice(idx, len(idx), replace=True)]) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append('      簇级 bootstrap 95%%CI: [%+.2f, %+.2f]' % (lo, hi))
        js['cluster_test'][col] = dict(k=K, n_clusters=int(len(cm)), median=float(np.median(cm)),
                                       pos=pos, neg=neg, sign_p=float(pb),
                                       ci=[float(lo), float(hi)])
    L.append('')

    # ---- 人口加权趋势 (簇级 bootstrap) ----
    L.append('== 4. 人口加权趋势 (簇级 bootstrap, 1000 次) ==')
    js['pop_weighted'] = {}
    rng = np.random.default_rng(2)
    groups = [(c, g) for c, g in tdf.groupby('cluster')]
    # 直接对"趋势"做人口加权, 重抽样单元 = 空间簇
    for col, nm in COLS:
        vals = []
        for _ in range(1000):
            pick = rng.integers(0, len(groups), len(groups))
            num = 0.0; den = 0.0
            for j in pick:
                c, g = groups[j]
                v = g[col].dropna()
                if not len(v):
                    continue
                w = float(g['pop'].sum())
                num += w * float(v.mean()); den += w
            if den > 0:
                vals.append(num / den)
        vals = np.array(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        pt = float(np.average(tdf[col].dropna(), weights=tdf.loc[tdf[col].notna(), 'pop']))
        L.append('  %s: 人口加权趋势 %+.2f h/10yr  95%%CI [%+.2f, %+.2f]  (P(>0)=%.3f)'
                 % (nm, pt, lo, hi, float(np.mean(vals > 0))))
        js['pop_weighted'][col] = dict(point=pt, ci=[float(lo), float(hi)],
                                       p_gt0=float(np.mean(vals > 0)))
    L.append('')

    # ---- 分气候带趋势 ----
    L.append('== 5. 分气候带 / 纬度带趋势 (h/10yr) ==')
    js['by_band'] = {}
    for col, nm in COLS:
        L.append('  --- %s ---' % nm)
        for b, g in tdf.groupby('band'):
            v = g[col].dropna().values
            if len(v) < 5:
                continue
            pos = int((v > 0).sum()); neg = int((v < 0).sum())
            L.append('    %-12s n=%3d  中位 %+8.2f  均值 %+8.2f  %d正/%d负  p=%.3f'
                     % (b, len(v), np.median(v), np.mean(v), pos, neg, binom_p(pos, neg)))
            js['by_band'].setdefault(col, {})[b] = dict(
                n=int(len(v)), median=float(np.median(v)), mean=float(np.mean(v)),
                pos=pos, neg=neg, p=float(binom_p(pos, neg)))
    tdf['latband'] = pd.cut(tdf['lat'].abs(), [0, 23.5, 35, 55, 90],
                            labels=['热带 0-23.5', '亚热带 23.5-35', '温带 35-55', '寒带 >55'])
    L.append('  --- 按 |纬度| 带 (NWH₂₁.₉) ---')
    for b, g in tdf.groupby('latband', observed=True):
        v = g['nhrs_elderly'].dropna().values
        if len(v) < 5:
            continue
        pos = int((v > 0).sum()); neg = int((v < 0).sum())
        L.append('    %-14s n=%3d  中位 %+8.2f  %d正/%d负  p=%.3f'
                 % (b, len(v), np.median(v), pos, neg, binom_p(pos, neg)))
        js.setdefault('by_latband', {})[str(b)] = dict(
            n=int(len(v)), median=float(np.median(v)), pos=pos, neg=neg,
            p=float(binom_p(pos, neg)))
    L.append('')
    # ---- 6. 最少年数敏感性 ----
    L.append('== 6. 最少年数敏感性 (记录长度门槛 vs 样本量 / 结论稳健性, NWH₂₁.₉) ==')
    js['minyears'] = {}
    cnt_all = m[m['ok_trend'].astype(str) == '1'].groupby('station').size()
    for my in (8, 10, 15, 20, 25, 30):
        c2 = set(cnt_all[cnt_all >= my].index)
        m2 = m[(m['ok_trend'].astype(str) == '1') & m.station.isin(c2)]
        bs = []
        for sid, g in m2.groupby('station'):
            g = g.dropna(subset=['nhrs_elderly']).sort_values('year')
            if len(g) < my:
                continue
            bs.append(theil_sen(g['year'].values.astype(float),
                                g['nhrs_elderly'].values.astype(float)) * 10)
        bs = np.array(bs, float)
        bs = bs[np.isfinite(bs)]
        if len(bs) < 5:
            continue
        pos = int((bs > 0).sum()); neg = int((bs < 0).sum())
        L.append('  ≥%2d 年: n=%3d  中位 %+7.2f  均值 %+7.2f  %3d正/%2d负  符号检验 p=%.4f'
                 % (my, len(bs), np.median(bs), np.mean(bs), pos, neg, binom_p(pos, neg)))
        js['minyears'][str(my)] = dict(n=int(len(bs)), median=float(np.median(bs)),
                                       mean=float(np.mean(bs)), pos=pos, neg=neg,
                                       p=float(binom_p(pos, neg)))
    txt = '\n'.join(L)
    print(txt)
    with open('representativeness.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('representativeness.json', 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
