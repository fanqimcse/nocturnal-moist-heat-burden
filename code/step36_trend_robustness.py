# -*- coding: utf-8 -*-
"""
step36: 头条趋势的稳健性 + 按检测归因缩放因子修正未来投影
========================================================
第 11 轮审稿人提出的最尖锐问题: 头条趋势 +20.5 h/10yr 是否稳健?
    - 阈值计数型指标在"常年为 0"的站上大量并列, 中位数会被锁死
    - 1976 年是记录首年, 数值异常低(1247 h/yr), 会主导端点式估计
    - 1976-2014 与 1976-2023 两个窗口给出相差 5 倍的趋势
本脚本系统扫描窗口与聚合口径, 并把 D&A 得到的缩放因子 b_ALL 应用到投影上。
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from step30_humidity_attribution import theil_sen, binom_p
import step33_nwh_projection as S

Y0 = 1976


def obs_panel():
    st = S.stations()
    NH = S.night_hours_table(st)
    c = pd.read_csv('transfer_calib.csv')
    c['nh'] = [NH.get((s, int(m)), np.nan) for s, m in zip(c['station'], c['month'])]
    c['nwh'] = c['f_obs'] * c['nh']
    a = c.groupby(['station', 'year'])['nwh'].sum().reset_index()
    return a.pivot(index='year', columns='station', values='nwh')


def net_trend(p, lo, hi, minfrac=0.9, weights=None):
    q = p.loc[lo:hi]
    nyr = len(q)
    frac = q.notna().sum(axis=0) / nyr
    keep = q.columns[frac >= minfrac]
    if len(keep) < 5:
        return np.nan, np.nan, 0
    s = q[keep]
    if weights is None:
        ser = s.mean(axis=1).values
    else:
        w = np.array([weights.get(k, 0.0) for k in keep])
        ser = np.array([np.average(s.iloc[i].values, weights=w)
                        if np.isfinite(s.iloc[i].values).all() else np.nan
                        for i in range(nyr)])
    y = q.index.values.astype(float)
    k = np.isfinite(ser)
    if k.sum() < 15:
        return np.nan, np.nan, len(keep)
    base = float(np.nanmean(ser))
    return float(theil_sen(y[k], ser[k]) * 10), base, len(keep)


def station_median(p, lo, hi):
    q = p.loc[lo:hi]
    y = q.index.values.astype(float)
    ts = []
    for s in q.columns:
        v = q[s].values
        k = np.isfinite(v)
        if k.sum() < 15:
            continue
        ts.append(theil_sen(y[k], v[k]) * 10)
    if not ts:
        return np.nan, 0
    ts = np.array(ts)
    pos = int((ts > 0).sum()); neg = int((ts < 0).sum())
    return np.nanmedian(ts), len(ts), binom_p(pos, neg)


def main():
    p = obs_panel()
    pop = pd.read_csv('station_population.csv').set_index('station')
    w = pop['pop_50km'].to_dict()
    L = ['NWH₂₁.₉ 观测趋势稳健性 (step36)',
         '观测序列: Σ月 f_obs × 天文夜间小时数; 阈值 21.9 °C',
         '',
         '【A】窗口敏感性 (平衡站集: 窗口内覆盖率≥90%的站; 网络等权均值)',
         '  窗口          站数   基线 h/yr   趋势 h/10yr']
    js = {'windows': {}}
    for lo, hi in ((1976, 2023), (1976, 2014), (1980, 2023), (1985, 2023),
                   (1990, 2023), (1995, 2023), (2000, 2023), (1977, 2023)):
        t, base, n = net_trend(p, lo, hi)
        L.append('  %d-%d    %3d   %7.0f    %+8.2f' % (lo, hi, n, base, t))
        js['windows']['%d-%d' % (lo, hi)] = dict(n=n, base=base, trend=t)
    L.append('')
    L.append('【B】聚合口径对比 (1976-2023)')
    t1, b1, n1 = net_trend(p, 1976, 2023)
    t2, b2, n2 = net_trend(p, 1976, 2023, weights=w)
    med, nmed, pmed = station_median(p, 1976, 2023)
    L.append('  网络等权均值    n=%3d  基线 %6.0f  趋势 %+8.2f h/10yr' % (n1, b1, t1))
    L.append('  人口加权均值    n=%3d  基线 %6.0f  趋势 %+8.2f h/10yr' % (n2, b2, t2))
    L.append('  逐站趋势中位数  n=%3d              趋势 %+8.2f h/10yr  (符号检验 p=%.1e)'
             % (nmed, med, pmed))
    L.append('  → 三者差异来自阈值计数在常年为 0 的站上并列, 中位数被锁死。')
    L.append('    论文应以网络均值为主口径, 并同时报告逐站中位数。')
    js['aggregation'] = dict(net_mean=dict(n=n1, base=b1, trend=t1),
                             pop_weighted=dict(n=n2, base=b2, trend=t2),
                             station_median=dict(n=nmed, trend=float(med), p=float(pmed)))
    # 逐年序列
    L.append('')
    L.append('【C】网络等权均值年序列 (h/yr)')
    ser = p.mean(axis=1)
    for y in p.index:
        L.append('    %d  %6.0f' % (y, ser[y]))
    js['series'] = {int(y): float(ser[y]) for y in p.index}

    # ---- 按 D&A 缩放因子修正投影 ----
    L.append('')
    L.append('【D】未来投影的观测约束修正')
    try:
        da = json.load(open('nwh_detection_attribution.json'))
        b_all_2 = da['two_signal']['b_ALL']
        se_2 = da['two_signal']['se_ALL']
        b_all_1 = da['single']['ALL']['b']
        se_1 = da['single']['ALL']['se']
    except Exception:
        b_all_2 = b_all_1 = None
    proj = json.load(open('nwh_projection_summary.json'))
    L.append('  D&A 缩放因子: 双信号 b_ALL=%+.3f ± %.3f ; 单信号 b_ALL=%+.3f ± %.3f'
             % (b_all_2, se_2, b_all_1, se_1))
    L.append('  → 模式对 1976-2014 的 NWH 增速高估约 %.1f 倍 (1/b_ALL)' % (1 / b_all_2))
    L.append('  → 把该因子应用于 SSP 投影的"变化量"(相对 2015-34 基线):')
    js['rescaled'] = {}
    for tag, (kb, km, ke) in (('主样本 67 站', ('base67', 'mid67', 'end67')),
                              ('全样本 176 站', ('base_all', 'mid_all', 'end_all'))):
        for sc in ('ssp245', 'ssp585'):
            b = np.array([v[kb] for k, v in proj.items() if sc in k])
            m = np.array([v[km] for k, v in proj.items() if sc in k])
            e = np.array([v[ke] for k, v in proj.items() if sc in k])
            dm = np.median(m - b); de = np.median(e - b)
            L.append('    %-14s %-8s 原始: 中期 %+6.0f 末期 %+6.0f h/yr | '
                     '× b_ALL: 中期 %+6.0f 末期 %+6.0f h/yr'
                     % (tag, sc, dm, de, dm * b_all_2, de * b_all_2))
            js['rescaled']['%s|%s' % (tag, sc)] = dict(
                raw_mid=float(dm), raw_end=float(de),
                scaled_mid=float(dm * b_all_2), scaled_end=float(de * b_all_2),
                scaled_mid_lo=float(dm * (b_all_2 - 1.645 * se_2)),
                scaled_mid_hi=float(dm * (b_all_2 + 1.645 * se_2)),
                scaled_end_lo=float(de * (b_all_2 - 1.645 * se_2)),
                scaled_end_hi=float(de * (b_all_2 + 1.645 * se_2)))
    txt = '\n'.join(L)
    print(txt)
    open('nwh_trend_robustness.txt', 'w', encoding='utf-8').write(txt + '\n')
    json.dump(js, open('nwh_trend_robustness.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
