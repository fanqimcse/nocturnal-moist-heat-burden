# -*- coding: utf-8 -*-
"""第12轮 R12-1  南欧 vs 北欧分层 —— 用现有欧洲数据闭合"暴露梯度"

背景(致命缺口):
  欧洲死亡面板年均 NWH21.9 ~ 71 h/yr, 而热带/亚热带台站网络 ~ 1487 h/yr (21x)。
  RR 是在暴露分布的第 0-5 分位上估计的(非零率仅 1.5-5.1%),
  因此第 8 轮的气候梯度检验 p = 0.206 不显著, 无法支撑向热带外推。

本轮方案: 欧洲内部本身就存在巨大的湿热梯度(斯堪的纳维亚 vs 地中海)。
如果"夜间湿球越致命"这一效应随湿热本底增强, 那么:
  (1) 南欧子样本的非零率应显著高于北欧 -> 识别样本不再是"少数区域-周";
  (2) 南欧的 RR/SD 应 >= 北欧 -> 梯度在欧洲内部被证实;
  (3) 南欧顶部周的暴露量应接近热带台站的典型周暴露 -> 外推跨度从 21x 缩小到可辩护范围。

产出: south_north_stratified.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, fit_region, meta

BASE = os.path.dirname(os.path.abspath(__file__))
WARM = (5, 9)
MED = {'ES', 'IT', 'GR', 'PT', 'HR', 'CY', 'MT'}


def log(*a):
    print(*a, flush=True)


def two_group_test(b1, s1, b2, s2):
    """两组随机效应系数之差及其 z / p"""
    d = b1 - b2
    se = math.sqrt(s1 ** 2 + s2 ** 2)
    if not np.isfinite(se) or se <= 0:
        return d, float('nan'), float('nan')
    z = d / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return d, z, p


def prep(exps):
    """载入面板 + 暖季过滤 + 区域内标准化"""
    m = load()
    m = m[m['n_days'] >= 5].copy()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM[0]) & (mo <= WARM[1])]
    cent = pd.read_csv(os.path.join(BASE, 'ext', 'health', 'nuts2_centroids.csv'))
    m = m.merge(cent[['nuts2', 'cntr', 'name', 'lat', 'lon']], on='nuts2', how='left')
    for e in exps:
        sd = m.groupby('nuts2')[e].transform('std')
        m[e + '_z'] = m[e] / sd.replace(0.0, np.nan)
    return m


def fit_stratum(sub, exps, min_n=100):
    """在子样本内逐区域拟合, 返回 (结果列表, 归因死亡)"""
    zz = [e + '_z' for e in exps]
    res = []
    for k, g in sub.groupby('nuts2'):
        gg = g.dropna(subset=zz)
        if len(gg) < min_n:
            continue
        r = fit_region(gg, zz)
        if r:
            res.append(r)
    burden = {}
    denom = 0.0
    for k, g in sub.groupby('nuts2'):
        r = [x for x in res if x['nuts2'] == k]
        if not r:
            continue
        r = r[0]
        denom += float(g['deaths'].sum())
        for e in exps:
            b = r.get(e + '_z', np.nan)
            if not np.isfinite(b):
                continue
            x = g[e].values.astype(float)
            ref = np.nanpercentile(x, 5)
            af = 1.0 - np.exp(-b * (x - ref))
            burden[e] = burden.get(e, 0.0) + float(np.nansum(g['deaths'].values.astype(float) * af))
    return res, burden, denom


def report_stratum(tag, sub, exps, L, js, nyears):
    zz = [e + '_z' for e in exps]
    res, burden, denom = fit_stratum(sub, exps)
    if len(res) < 20:
        L.append('  %-14s 区域数不足(%d), 跳过' % (tag, len(res)))
        return None
    L.append('  ---- %s : %d 区域 / %d 区域-周 / 暖季死亡 %.0f ----'
             % (tag, len(res), len(sub), denom))
    out = {'n_regions': len(res), 'n_weeks': int(len(sub)), 'deaths': float(denom)}
    for e in exps:
        mt = meta(res, e + '_z')
        if not mt:
            continue
        rr = math.exp(mt['re'])
        lo = math.exp(mt['re'] - 1.96 * mt['re_se'])
        hi = math.exp(mt['re'] + 1.96 * mt['re_se'])
        z = mt['re'] / mt['re_se'] if mt['re_se'] > 0 else float('nan')
        nz = float(np.nanmean((sub[e].fillna(0) != 0)))
        sd = float(np.nanmedian([g[e].std() for _, g in sub.groupby('nuts2')
                                 if g[e].notna().sum() > 50]))
        L.append('    %-18s RR/SD=%.4f [%.4f,%.4f] z=%+5.2f I²=%2.0f%%  非零率%5.1f%%  SD=%.3f'
                 % (e, rr, lo, hi, z, 100 * mt['I2'], 100 * nz, sd))
        out[e] = dict(rr_per_sd=rr, lo=lo, hi=hi, beta=float(mt['re']),
                      se=float(mt['re_se']), z=float(z), I2=mt['I2'],
                      nonzero=float(nz), sd=sd)
    L.append('    -- 归因死亡 (区域特异 β, 参考=该区域 5 分位) --')
    out['burden'] = {}
    for e in exps:
        v = burden.get(e, 0.0)
        L.append('       %-16s %9.0f  (%6.2f 万/年, 占暖季死亡 %5.2f%%)'
                 % (e, v, v / nyears / 1e4, 100 * v / max(denom, 1e-9)))
        out['burden'][e] = dict(total=float(v), per_year=float(v / nyears),
                                pct=float(100 * v / max(denom, 1e-9)))
    if 'nh219' in out and 'tx_max' in out:
        a = out['burden'].get('nh219', {}).get('per_year', 0.0)
        b = out['burden'].get('tx_max', {}).get('per_year', 0.0)
        if b:
            L.append('       比值 nh219 / tx_max = %.1f%%' % (100 * a / b))
    L.append('')
    js[tag] = out
    return out


def tropical_weekly_ref():
    """热带/亚热带台站网络的典型"每周 NWH21.9" (h/week), 用于与欧洲顶部周比较"""
    c = pd.read_csv(os.path.join(BASE, 'transfer_calib.csv'))
    try:
        import step33_nwh_projection as S
        st = S.stations()
        NH = S.night_hours_table(st)
        c['nh'] = [NH.get((s, int(m)), np.nan) for s, m in zip(c['station'], c['month'])]
    except Exception:
        c['nh'] = c['nhours_obs']
    c['nwh'] = c['f_obs'] * c['nh']
    a = c.groupby(['station', 'year'])['nwh'].sum().reset_index()
    a = a[a['year'] >= 2000]
    per_st = a.groupby('station')['nwh'].mean()
    return per_st


def main():
    m = prep(['tx_max', 'tw_night_min_max', 'nh_tn20', 'nh219'])
    nyears = max(x.year for x in m['mon']) - min(x.year for x in m['mon']) + 1
    L = ['南欧 vs 北欧分层: 用欧洲内部湿热梯度闭合"暴露梯度"缺口 (step39)',
         '面板: %d 个 NUTS2, %d 区域-周, 暖季 5-9 月, %d-%d (%d 年)'
         % (m['nuts2'].nunique(), len(m),
            min(x.year for x in m['mon']), max(x.year for x in m['mon']), nyears),
         '暴露已在各区域内除以自身 SD -> RR/SD 可跨气候区直接比较',
         '']
    js = {'n_regions': int(m['nuts2'].nunique()), 'n_rows': int(len(m)), 'nyears': int(nyears)}

    # ---- 区域湿热本底 ----
    g = m.groupby('nuts2').agg(tw0=('tw_night_min_max', 'mean'),
                               tn0=('tn_max', 'mean'),
                               nh0=('nh219', 'mean'),
                               nh_p95=('nh219', lambda v: float(np.nanpercentile(v, 95))),
                               nh_p99=('nh219', lambda v: float(np.nanpercentile(v, 99))),
                               lat=('lat', 'first'), cntr=('cntr', 'first'),
                               n=('deaths', 'size')).reset_index()
    g = g[g['n'] >= 100]
    L.append('【1】区域湿热本底暖季"最热一夜"最低湿球 tw0 的分布')
    q = g['tw0'].quantile([0.25, 0.5, 0.75]).to_dict()
    L.append('  n=%d 区域  中位 %.2f °C  IQR %.2f-%.2f  全距 %.2f-%.2f'
             % (len(g), q[0.5], q[0.25], q[0.75], g.tw0.min(), g.tw0.max()))
    js['tw0'] = dict(n=int(len(g)), median=float(q[0.5]),
                     q1=float(q[0.25]), q3=float(q[0.75]),
                     min=float(g.tw0.min()), max=float(g.tw0.max()))
    L.append('')

    # ---- 三种分层方案 ----
    schemes = {}
    schemes['湿热本底中位数切分'] = {
        '南欧型 湿热': set(g[g['tw0'] >= q[0.5]]['nuts2']),
        '北欧型 凉爽': set(g[g['tw0'] < q[0.5]]['nuts2'])}
    latmed = float(g['lat'].median())
    schemes['纬度中位数切分 (%.1f°N)' % latmed] = {
        '南部 <%.1f°N' % latmed: set(g[g['lat'] < latmed]['nuts2']),
        '北部 ≥%.1f°N' % latmed: set(g[g['lat'] >= latmed]['nuts2'])}
    schemes['地中海国家集合'] = {
        '地中海国家': set(g[g['cntr'].isin(MED)]['nuts2']),
        '其余欧洲': set(g[~g['cntr'].isin(MED)]['nuts2'])}

    L.append('【2】三套分层方案下的暴露对比度 (暖季周)')
    L.append('  方案 / 层            区域数  非零率nh219  周nh219均值  P95   P99   年化h/yr')
    js['schemes'] = {}
    for sname, groups in schemes.items():
        js['schemes'][sname] = {}
        for gname, ids in groups.items():
            sub = m[m['nuts2'].isin(ids)]
            if len(sub) == 0:
                continue
            nz = float(np.nanmean((sub['nh219'].fillna(0) != 0)))
            mn = float(sub['nh219'].mean())
            ann = mn * 52.0
            L.append('  %-22s %5d   %6.1f%%      %6.2f     %5.1f %5.1f   %6.0f'
                     % (gname[:22], sub['nuts2'].nunique(), 100 * nz, mn,
                        np.nanpercentile(sub['nh219'], 95),
                        np.nanpercentile(sub['nh219'], 99), ann))
            js['schemes'][sname][gname] = dict(
                n_regions=int(sub['nuts2'].nunique()), nonzero=float(nz),
                mean_weekly=mn, p95=float(np.nanpercentile(sub['nh219'], 95)),
                p99=float(np.nanpercentile(sub['nh219'], 99)), annualized=ann)
    L.append('')

    # ---- 主分层: 湿热本底中位数切分 ----
    EX = ['tx_max', 'tw_night_min_max', 'nh_tn20', 'nh219']
    L.append('【3】分层回归: 规格 = 区域FE + 年 + Fourier(4) + 白天TX + 夜间干球度夜 + 夜间湿球小时数')
    L.append('    (四个暴露同时进入, 每个都已在该区域内除以自身 SD)')
    L.append('')
    strat_res = {}
    for sname in ['湿热本底中位数切分', '纬度中位数切分 (%.1f°N)' % latmed, '地中海国家集合']:
        groups = schemes[sname]
        L.append('== 方案: %s ==' % sname)
        d = {}
        for gname in sorted(groups, reverse=True):
            ids = groups[gname]
            sub = m[m['nuts2'].isin(ids)]
            if sub['nuts2'].nunique() < 20:
                continue
            d[gname] = report_stratum(gname, sub, EX, L, d.setdefault('_js', {}), nyears)
        js['schemes'][sname]['_fit'] = d.pop('_js', {})
        # 层间差异检验
        ks = [k for k in d if d.get(k)]
        if len(ks) == 2:
            a, b = ks[0], ks[1]
            L.append('  -- 层间差异 (%s − %s) --' % (a, b))
            for e in EX:
                if e in d[a] and e in d[b]:
                    dd, z, p = two_group_test(d[a][e]['beta'], d[a][e]['se'],
                                              d[b][e]['beta'], d[b][e]['se'])
                    L.append('     %-18s Δβ/SD=%+.4f  z=%+5.2f  p=%.3f' % (e, dd, z, p))
                    js['schemes'][sname].setdefault('contrast', {})[e] = dict(
                        dbeta=float(dd), z=float(z), p=float(p))
            L.append('')
        strat_res[sname] = d

    # ---- 四分位梯度 + meta 回归趋势 ----
    L.append('【4】湿热本底四分位梯度 (第 8 轮同款检验, 用本轮更大样本重做)')
    g['grp'] = pd.qcut(g['tw0'], 4, labels=['Q1 最凉', 'Q2', 'Q3', 'Q4 最湿热'])
    trend_x, trend_y, trend_se = [], [], []
    js['quartile'] = {}
    for grp in ['Q1 最凉', 'Q2', 'Q3', 'Q4 最湿热']:
        ids = set(g[g['grp'] == grp]['nuts2'])
        sub = m[m['nuts2'].isin(ids)]
        d = {}
        r = report_stratum(str(grp), sub, EX, L, d, nyears)
        if r:
            js['quartile'][str(grp)] = r
            if 'nh219' in r:
                trend_x.append(float(g[g['grp'] == grp]['tw0'].mean()))
                trend_y.append(r['nh219']['beta'])
                trend_se.append(r['nh219']['se'])
    if len(trend_x) >= 4:
        x = np.array(trend_x); y = np.array(trend_y); se = np.array(trend_se)
        w = 1.0 / np.maximum(se, 1e-9) ** 2
        Xd = np.column_stack([np.ones(len(x)), x])
        W = np.diag(w)
        A = Xd.T @ W @ Xd
        Ai = np.linalg.inv(A + 1e-12 * np.eye(2))
        bb = Ai @ Xd.T @ W @ y
        resid = y - Xd @ bb
        dof = max(len(x) - 2, 1)
        s2 = float(np.sum(w * resid ** 2) / dof)
        cov = s2 * Ai
        slope_se = math.sqrt(max(cov[1, 1], 0.0))
        zs = bb[1] / slope_se if slope_se > 0 else float('nan')
        ps = 2 * (1 - 0.5 * (1 + math.erf(abs(zs) / math.sqrt(2))))
        L.append('  nh219 的 β/SD 对区域湿热本底 tw0 的加权 meta 回归:')
        L.append('    斜率 %+.4f ± %.4f 每 °C 本底   z=%+.2f  p=%.3f' % (bb[1], slope_se, zs, ps))
        js['meta_regression'] = dict(slope=float(bb[1]), se=float(slope_se),
                                     z=float(zs), p=float(ps),
                                     x=[float(v) for v in x], y=[float(v) for v in y])
    L.append('')

    # ---- 暴露重叠度: 欧洲顶部周暴露 vs 热带台站典型周暴露 ----
    L.append('【5】暴露重叠度: 欧洲的"最热周"与热带台站"典型周"差多远')
    per_st = tropical_weekly_ref()
    trop_weekly = float(per_st.median()) / 52.0
    trop_q25, trop_q75 = float(per_st.quantile(0.25)) / 52.0, float(per_st.quantile(0.75)) / 52.0
    eu = g.copy()
    L.append('  热带/亚热带台站网络 (n=%d 站, 2000 年后年均):' % len(per_st))
    L.append('    年均 NWH21.9 中位 %.0f h/yr  ->  折合典型周 %.1f h/周 (IQR %.1f-%.1f)'
             % (per_st.median(), trop_weekly, trop_q25, trop_q75))
    allsub = m
    L.append('  欧洲暖季周 nh219:')
    for lab, vals in [('全欧洲', allsub['nh219'].values),
                      ('南欧型(湿热本底上半)', m[m['nuts2'].isin(schemes['湿热本底中位数切分']['南欧型 湿热'])]['nh219'].values),
                      ('北欧型(湿热本底下半)', m[m['nuts2'].isin(schemes['湿热本底中位数切分']['北欧型 凉爽'])]['nh219'].values)]:
        v = vals[np.isfinite(vals)]
        L.append('    %-20s 均值 %5.2f  P95 %5.1f  P99 %5.1f  最大 %5.1f h/周'
                 % (lab, v.mean(), np.percentile(v, 95), np.percentile(v, 99), v.max()))
    south = m[m['nuts2'].isin(schemes['湿热本底中位数切分']['南欧型 湿热'])]['nh219'].values
    south = south[np.isfinite(south)]
    s_p999 = float(np.percentile(south, 99.9))
    s_p99 = float(np.percentile(south, 99))
    L.append('  外推跨度重估:')
    L.append('    旧口径(欧洲年均 vs 台站年均):        71 / 1487  = 21.0×  <- 不可辩护')
    L.append('    新口径(南欧 P99 周 vs 台站典型周):   %.1f / %.1f = %.1f×'
             % (s_p99, trop_weekly, trop_weekly / max(s_p99, 1e-9)))
    L.append('    新口径(南欧 P99.9 周 vs 台站典型周): %.1f / %.1f = %.1f×'
             % (s_p999, trop_weekly, trop_weekly / max(s_p999, 1e-9)))
    js['overlap'] = dict(tropical_weekly_median=trop_weekly,
                         tropical_q25=trop_q25, tropical_q75=trop_q75,
                         south_p99=s_p99, south_p999=s_p999,
                         south_mean=float(south.mean()),
                         ratio_p99=float(trop_weekly / max(s_p99, 1e-9)),
                         ratio_p999=float(trop_weekly / max(s_p999, 1e-9)))
    L.append('')

    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'south_north_stratified.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'south_north_stratified.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
