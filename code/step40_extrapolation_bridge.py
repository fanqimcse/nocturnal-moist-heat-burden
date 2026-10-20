# -*- coding: utf-8 -*-
"""第12轮 R12-2  外推桥: 区域级梯度 meta 回归 + 剂量-反应 + 暴露覆盖度

step39 暴露出三个必须修掉的问题:
  (1) 归因死亡出现负值 —— 用区域特异 β 逐区域求和, β 噪声大的区域会翻号。
      修正: 经验贝叶斯收缩 (区域 β 向随机效应汇总值收缩, 权重 = 1/(se²+τ²))。
  (2) 最凉四分位整层被剔除 —— 那些区域 NWH21.9 从不越阈 (SD=0), z 全为 NaN。
      这本身是结果(指标在该气候带无分辨力), 但不能让梯度检验失去样本。
      修正: 改用**区域级** meta 回归 (n = 区域数, 而不是 4 个组均值), 功效高得多。
  (3) 外推跨度: 旧口径"欧洲年均 71 h vs 台站年均 1487 h = 21×"是**均值对均值**,
      而 RR 是在欧洲暴露分布的**尾部**识别的。正确口径是**尾部对尾部**。
      step39 已显示: 南欧 P99 周 = 54 h/周, 热带台站典型周 = 18.2 h/周。
      本脚本量化: 热带站月分布中有多大比例落在欧洲已识别的暴露区间内。

产出: extrapolation_bridge.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, fit_region, meta

BASE = os.path.dirname(os.path.abspath(__file__))
WARM = (5, 9)
EX = ['tx_max', 'nh_tn20', 'nh219']


def log(*a):
    print(*a, flush=True)


def prep():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM[0]) & (mo <= WARM[1])].copy()
    cent = pd.read_csv(os.path.join(BASE, 'ext', 'health', 'nuts2_centroids.csv'))
    m = m.merge(cent[['nuts2', 'cntr', 'name', 'lat', 'lon']], on='nuts2', how='left')
    return m


def region_fits(m, exps):
    """逐区域拟合, 返回每区域每个暴露的 β(每 1 SD) 与 se, 以及区域属性"""
    for e in exps:
        sd = m.groupby('nuts2')[e].transform('std')
        m[e + '_sd'] = sd
        m[e + '_z'] = m[e] / sd.replace(0.0, np.nan)
    zz = [e + '_z' for e in exps]
    rows = []
    for k, g in m.groupby('nuts2'):
        gg = g.dropna(subset=zz)
        if len(gg) < 100 or gg['deaths'].sum() < 200:
            continue
        r = fit_region(gg, zz)
        if not r:
            continue
        o = {'nuts2': k, 'n': len(gg), 'deaths': float(gg['deaths'].sum()),
             'lat': float(g['lat'].iloc[0]), 'cntr': str(g['cntr'].iloc[0]),
             'tw0': float(g['tw_night_min_max'].mean()),
             'tn0': float(g['tn_max'].mean())}
        for e in exps:
            o[e + '_b'] = float(r.get(e + '_z', np.nan))
            o[e + '_se'] = float(r.get(e + '_z_se', np.nan))
            o[e + '_sd'] = float(g[e + '_sd'].iloc[0])
            o[e + '_nz'] = float(np.nanmean(g[e].fillna(0) != 0))
        rows.append(o)
    return pd.DataFrame(rows)


def wls(x, y, se):
    """加权最小二乘 y = a + b x, 权重 1/se²; 返回 b, se_b, z, p, n"""
    k = np.isfinite(x) & np.isfinite(y) & np.isfinite(se) & (se > 0)
    x, y, se = np.asarray(x)[k], np.asarray(y)[k], np.asarray(se)[k]
    n = len(x)
    if n < 8:
        return None
    w = 1.0 / se ** 2
    X = np.column_stack([np.ones(n), x])
    W = np.diag(w)
    A = X.T @ W @ X
    Ai = np.linalg.inv(A + 1e-12 * np.eye(2))
    b = Ai @ X.T @ W @ y
    resid = y - X @ b
    dof = max(n - 2, 1)
    s2 = float(np.sum(w * resid ** 2) / dof)
    cov = s2 * Ai
    sb = math.sqrt(max(cov[1, 1], 0.0))
    z = b[1] / sb if sb > 0 else float('nan')
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return dict(slope=float(b[1]), se=float(sb), z=float(z), p=float(p),
                intercept=float(b[0]), n=int(n))


def shrink_burden(m, fits, exps, pooled):
    """经验贝叶斯收缩后的归因死亡: β_shrunk = (β/se² + μ/τ²)/(1/se² + 1/τ²)"""
    out = {}
    denom = 0.0
    for e in exps:
        mu = pooled[e]['re']
        tau2 = max(pooled[e]['tau2'], 1e-12)
        tot = 0.0
        for k, g in m.groupby('nuts2'):
            f = fits[fits['nuts2'] == k]
            if len(f) == 0:
                continue
            b = float(f[e + '_b'].iloc[0]); se = float(f[e + '_se'].iloc[0])
            if not (np.isfinite(b) and np.isfinite(se) and se > 0):
                continue
            bs = (b / se ** 2 + mu / tau2) / (1.0 / se ** 2 + 1.0 / tau2)
            x = g[e].values.astype(float)
            ref = np.nanpercentile(x, 5)
            af = 1.0 - np.exp(-bs * (x - ref))
            tot += float(np.nansum(g['deaths'].values.astype(float) * af))
            if e == exps[0]:
                denom += float(g['deaths'].sum())
        out[e] = tot
    return out, denom


def tropical_monthly_weekly():
    """热带/亚热带台站: 每月 NWH21.9 -> 折合每周 (月/4.345)"""
    c = pd.read_csv(os.path.join(BASE, 'transfer_calib.csv'))
    try:
        import step33_nwh_projection as S
        st = S.stations()
        NH = S.night_hours_table(st)
        c['nh'] = [NH.get((s, int(m)), np.nan) for s, m in zip(c['station'], c['month'])]
    except Exception:
        c['nh'] = c['nhours_obs']
    c['nwh_m'] = c['f_obs'] * c['nh']
    c['nwh_w'] = c['nwh_m'] / 4.345
    return c


def main():
    m = prep()
    nyears = max(x.year for x in m['mon']) - min(x.year for x in m['mon']) + 1
    L = ['外推桥: 区域级梯度检验 + 剂量-反应 + 暴露覆盖度 (step40)',
         '面板: %d NUTS2 / %d 区域-周 / 暖季 5-9 月 / %d 年'
         % (m['nuts2'].nunique(), len(m), nyears), '']
    js = {}

    fits = region_fits(m, EX)
    log('区域拟合完成: %d 个区域 (面板共 %d 个)' % (len(fits), m['nuts2'].nunique()))
    js['n_regions_fitted'] = int(len(fits))
    js['n_regions_total'] = int(m['nuts2'].nunique())
    js['n_regions_dropped_no_variance'] = int(m['nuts2'].nunique() - len(fits))
    # meta() 需要 '<e>_z' 形式的键
    recs = []
    for _, r in fits.iterrows():
        o = dict(r)
        for e in EX:
            o[e + '_z'] = r[e + '_b']
            o[e + '_z_se'] = r[e + '_se']
        recs.append(o)
    pooled = {}
    for e in EX:
        mt = meta(recs, e + '_z')
        pooled[e] = mt
    # τ² (DerSimonian-Laird 近似, 用于收缩)
    for e in EX:
        b = fits[e + '_b'].values; se = fits[e + '_se'].values
        k = np.isfinite(b) & np.isfinite(se) & (se > 0)
        b, se = b[k], se[k]
        w = 1.0 / se ** 2
        mu = np.sum(w * b) / np.sum(w)
        Q = float(np.sum(w * (b - mu) ** 2))
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max((Q - (len(b) - 1)) / C, 0.0) if C > 0 else 0.0
        pooled[e]['tau2'] = float(tau2)
        pooled[e]['mu_fixed'] = float(mu)

    # ---------- 【1】区域级 meta 回归: 效应是否随湿热本底增强 ----------
    L.append('【1】区域级 meta 回归 —— 效应是否随区域湿热本底增强')
    L.append('   被解释量: 该区域各暴露的 β(每 1 个区域内 SD); 解释量: 该区域暖季最热夜湿球本底 tw0')
    L.append('   权重 1/se²; 这是比"四分位组均值"功效高得多的梯度检验 (第 8 轮 p=0.206 的低功效问题)')
    L.append('')
    for e in EX + ['tw_night_min_max']:
        if e + '_b' not in fits.columns:
            continue
        r = wls(fits['tw0'].values, fits[e + '_b'].values, fits[e + '_se'].values)
        if r is None:
            continue
        L.append('   %-18s 斜率 %+.5f ± %.5f 每 °C 本底   z=%+5.2f  p=%.4f  (n=%d 区域)'
                 % (e, r['slope'], r['se'], r['z'], r['p'], r['n']))
        js.setdefault('meta_regression', {})[e] = r
    L.append('')
    L.append('   同样用纬度做解释量(独立性检验):')
    for e in EX:
        r = wls(fits['lat'].values, fits[e + '_b'].values, fits[e + '_se'].values)
        if r is None:
            continue
        L.append('   %-18s 斜率 %+.5f ± %.5f 每 °N       z=%+5.2f  p=%.4f'
                 % (e, r['slope'], r['se'], r['z'], r['p']))
        js.setdefault('meta_regression_lat', {})[e] = r
    L.append('')

    # ---------- 【2】南欧 / 北欧 收缩后归因负担 ----------
    L.append('【2】经验贝叶斯收缩后的归因死亡 (修正 step39 的负值问题)')
    L.append('   β_shrunk = (β/se² + μ/τ²) / (1/se² + 1/τ²), 参考 = 该区域暴露 5 分位')
    L.append('')
    tw0_med = float(fits['tw0'].median())
    js['burden'] = {}
    for tag, sel in (('南欧型 tw0≥%.2f' % tw0_med, fits['tw0'] >= tw0_med),
                     ('北欧型 tw0<%.2f' % tw0_med, fits['tw0'] < tw0_med),
                     ('全欧洲', fits['tw0'].notna())):
        ids = set(fits[sel]['nuts2'])
        sub = m[m['nuts2'].isin(ids)]
        f2 = fits[fits['nuts2'].isin(ids)]
        if sub['nuts2'].nunique() < 20:
            continue
        tot, denom = shrink_burden(sub, f2, EX, pooled)
        L.append('  %s : %d 区域 / %.0f 暖季死亡' % (tag, sub['nuts2'].nunique(), denom))
        d = {'n_regions': int(sub['nuts2'].nunique()), 'deaths': float(denom)}
        for e in EX:
            v = tot.get(e, 0.0)
            L.append('     %-10s %10.0f  (%7.2f 万/年, 占暖季死亡 %6.2f%%)'
                     % (e, v, v / nyears / 1e4, 100 * v / max(denom, 1e-9)))
            d[e] = dict(total=float(v), per_year=float(v / nyears),
                        pct=float(100 * v / max(denom, 1e-9)))
        if d.get('nh219', {}).get('per_year') and d.get('tx_max', {}).get('per_year'):
            L.append('     比值 nh219 / tx_max = %.1f%%'
                     % (100 * d['nh219']['per_year'] / d['tx_max']['per_year']))
        L.append('')
        js['burden'][tag] = d

    # ---------- 【3】剂量-反应: 在南欧子样本内, 风险随暴露分箱如何变化 ----------
    L.append('【3】剂量-反应 (南欧型子样本, 暖季周)')
    L.append('   每周 NWH21.9 分箱: 0 / (0,2] / (2,6] / (6,20] / >20 h; 参考 = 0 箱')
    L.append('   同模型控制白天 TX 与夜间干球度夜; 分箱 RR 经随机效应汇总')
    L.append('')
    ids = set(fits[fits['tw0'] >= tw0_med]['nuts2'])
    sub = m[m['nuts2'].isin(ids)].copy()
    sub['bin'] = pd.cut(sub['nh219'], [-1e-9, 0, 2, 6, 20, 1e9],
                        labels=['0', '0-2', '2-6', '6-20', '>20'])
    sub['tx_z'] = sub['tx_max'] / sub.groupby('nuts2')['tx_max'].transform('std').replace(0, np.nan)
    sub['tn_z'] = sub['nh_tn20'] / sub.groupby('nuts2')['nh_tn20'].transform('std').replace(0, np.nan)
    cats = ['0-2', '2-6', '6-20', '>20']
    res = []
    for k, g in sub.groupby('nuts2'):
        gg = g.dropna(subset=['tx_z', 'tn_z'])
        if len(gg) < 100:
            continue
        if gg['bin'].nunique() < 3:
            continue
        dm = pd.get_dummies(gg['bin'].astype(str), prefix='b').reindex(
            columns=['b_' + c for c in cats], fill_value=0.0).astype(float)
        X0, nm0 = design(gg, ['tx_z', 'tn_z'])
        X = np.column_stack([X0] + [dm[c].values for c in dm.columns])
        names = nm0 + list(dm.columns)
        y = gg['deaths'].values.astype(float)
        b, se, phi = poisson_glm(X, y)
        if not np.all(np.isfinite(se)):
            continue
        o = {'nuts2': k, 'n': len(gg)}
        for c in dm.columns:
            i = names.index(c)
            o[c] = float(b[i]); o[c + '_se'] = float(se[i])
        res.append(o)
    if len(res) >= 20:
        L.append('   汇总区域数 %d' % len(res))
        js['dose_response'] = {'n_regions': len(res)}
        for c in cats:
            key = 'b_' + c
            mt = meta(res, key)
            if not mt:
                continue
            rr = math.exp(mt['re'])
            lo = math.exp(mt['re'] - 1.96 * mt['re_se'])
            hi = math.exp(mt['re'] + 1.96 * mt['re_se'])
            z = mt['re'] / mt['re_se'] if mt['re_se'] > 0 else float('nan')
            L.append('   NWH21.9 %-6s h/周  RR=%6.4f [%6.4f,%6.4f]  z=%+5.2f  I²=%2.0f%%'
                     % (c, rr, lo, hi, z, 100 * mt['I2']))
            js['dose_response'][c] = dict(rr=float(rr), lo=float(lo), hi=float(hi),
                                          z=float(z), I2=float(mt['I2']))
    else:
        L.append('   可用区域不足 (%d), 跳过' % len(res))
    L.append('')

    # ---------- 【4】暴露覆盖度: 欧洲尾部 vs 热带站月分布 ----------
    L.append('【4】暴露覆盖度 —— 欧洲 RR 到底覆盖了多少热带站月')
    tr = tropical_monthly_weekly()
    tr = tr[tr['year'] >= 2000]
    tw = tr['nwh_w'].dropna().values
    south_ids = set(fits[fits['tw0'] >= tw0_med]['nuts2'])
    s = m[m['nuts2'].isin(south_ids)]['nh219'].dropna().values
    allv = m['nh219'].dropna().values
    L.append('   热带/亚热带台站 (n=%d 站月, 2000 年后): 每周 NWH21.9' % len(tw))
    L.append('     中位 %.1f  IQR %.1f-%.1f  P90 %.1f  P99 %.1f  h/周'
             % (np.median(tw), np.percentile(tw, 25), np.percentile(tw, 75),
                np.percentile(tw, 90), np.percentile(tw, 99)))
    L.append('   欧洲暖季周 NWH21.9:')
    L.append('     全欧洲   中位 %.1f  P95 %.1f  P99 %.1f  P99.9 %.1f  最大 %.1f'
             % (np.median(allv), np.percentile(allv, 95), np.percentile(allv, 99),
                np.percentile(allv, 99.9), allv.max()))
    L.append('     南欧型   中位 %.1f  P95 %.1f  P99 %.1f  P99.9 %.1f  最大 %.1f'
             % (np.median(s), np.percentile(s, 95), np.percentile(s, 99),
                np.percentile(s, 99.9), s.max()))
    cuts = {'欧洲 P95 (=%.0f)' % np.percentile(allv, 95): float(np.percentile(allv, 95)),
            '欧洲 P99 (=%.0f)' % np.percentile(allv, 99): float(np.percentile(allv, 99)),
            '南欧 P95 (=%.0f)' % np.percentile(s, 95): float(np.percentile(s, 95)),
            '南欧 P99 (=%.0f)' % np.percentile(s, 99): float(np.percentile(s, 99)),
            '南欧 P99.9 (=%.0f)' % np.percentile(s, 99.9): float(np.percentile(s, 99.9))}
    L.append('   → 热带站月中有多大比例落在"欧洲已识别的暴露区间"(≤该切点)内:')
    js['coverage'] = {}
    for k, v in cuts.items():
        frac = float(np.mean(tw <= v))
        L.append('     ≤ %-18s 的热带站月占比: %5.1f%%' % (k, 100 * frac))
        js['coverage'][k] = dict(cut=float(v), frac=frac)
    L.append('')
    L.append('   判读提示: 若南欧 P99 已能覆盖多数热带站月, 则"欧洲 RR → 热带"')
    L.append('   不再是 21 倍外推, 而是**在已观测区间内插值**。')
    L.append('')

    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'extrapolation_bridge.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'extrapolation_bridge.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
