# -*- coding: utf-8 -*-
"""第8轮 R8-6 头条指标重构: NWH — 夜间湿球超阈小时数 (Nocturnal Wet-bulb Hours)

为什么必须换头条指标:
  原头条 NRF(夜间恢复失效率) 在 30.6 / 35 °C 下全样本恒为 0 —— 一个恒等于零的
  指标不能当头条。欧洲死亡面板的阈值扫描显示: 死亡信号在湿球 21.9 °C 处最强,
  到 25.8 °C 完全消失, 且"持续小时数"远优于"是否越阈"的二元判据(ΔLL 1585 vs 208)。
  因此把头条改为与流行病学锚定的连续量:
      NWH_T = 一年内夜间(太阳高度角<-0.833°)湿球超过 T °C 的小时数
  metrics_v3 中已存在: nhrs_elderly(21.9)/nhrs_t23(23)/nhrs_t24(24)/
                       nhrs_young(25.8)/nhrs_psu(30.6)/nhrs_classic(35)

产出: headline_metric.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from step6_trends import theil_sen, mann_kendall, median as med

METRICS = [
    ('nhrs_elderly', 21.9, 'NWH₂₁.₉ 老年人下限 (流行病学锚定)'),
    ('nhrs_t23', 23.0, 'NWH₂₃'),
    ('nhrs_t24', 24.0, 'NWH₂₄'),
    ('nhrs_young', 25.8, 'NWH₂₅.₈ 青年下限 (Perkins-Kirkpatrick)'),
    ('nhrs_psu', 30.6, 'NWH₃₀.₆ PSU 可补偿性上限'),
    ('nhrs_classic', 35.0, 'NWH₃₅ 经典生存极限'),
]
# 对照: 干球夜间热负荷(小时)
DRY = [('fn_tn20', '夜数 TN≥20 (热带夜)'), ('fn_tn25', '夜数 TN≥25')]


def neff_bretherton(r):
    """空间自相关校正的有效样本量 (Bretherton 等)"""
    r = np.clip(r, -0.999, 0.999)
    n = len(r)
    if n < 3:
        return n
    # 用站点间相关的均值估计有效自由度
    rr = np.array(r, dtype=float)
    rr = rr[np.isfinite(rr)]
    rbar = float(np.mean(rr))
    neff = n * (1 - rbar) / (1 + rbar) if rbar < 1 else 1.0
    return max(2.0, min(float(n), neff))


def station_trend(m, col, min_years=20):
    """逐站 Theil-Sen 趋势 (小时/10年)"""
    out = {}
    for sid, g in m.groupby('station'):
        g = g.dropna(subset=[col]).sort_values('year')
        if len(g) < min_years:
            continue
        b = theil_sen(g['year'].values.astype(float), g[col].values.astype(float))
        _s, _z, p, _t = mann_kendall(g[col].values.astype(float))
        out[sid] = (b * 10.0, p, len(g))
    return out


def main():
    m = pd.read_csv('metrics_v3.csv')
    pop = pd.read_csv('station_population.csv')
    m = m.merge(pop[['station', 'pop_50km']], on='station', how='left')
    m['pop_50km'] = m['pop_50km'].fillna(0.0)
    ok = m[m['ok_trend'].astype(str) == '1'].copy()
    lines = ['头条指标重构: NWH —— 夜间湿球超阈小时数 (Nocturnal Wet-bulb Hours)',
             '样本: %d 站 / %d 站年 (合格), %d–%d'
             % (ok['station'].nunique(), len(ok), int(ok.year.min()), int(ok.year.max())),
             '']
    out = {'n_stations': int(ok['station'].nunique()), 'n_station_years': int(len(ok)),
           'year_range': [int(ok.year.min()), int(ok.year.max())]}
    lines.append('%-26s %10s %10s %10s %10s %8s %8s'
                 % ('指标', '1976-85均', '2014-23均', '倍数', '趋势/10yr', 'p', '站数'))
    for col, th, nm in METRICS:
        if col not in ok.columns:
            continue
        tr = station_trend(ok, col)
        if len(tr) < 10:
            continue
        bs = np.array([v[0] for v in tr.values()])
        ps = np.array([v[1] for v in tr.values()])
        med_b = float(np.median(bs))
        # 人口加权
        pw = {}
        for sid, (b, p, n) in tr.items():
            w = float(ok[ok.station == sid]['pop_50km'].iloc[0])
            pw[sid] = (b, w)
        wsum = sum(w for _, w in pw.values())
        wm = sum(b * w for b, w in pw.values()) / wsum if wsum > 0 else float('nan')
        e1 = ok[(ok.year >= 1976) & (ok.year <= 1985)][col].mean()
        e2 = ok[(ok.year >= 2014) & (ok.year <= 2023)][col].mean()
        # 符号检验
        pos = int(np.sum(bs > 0)); neg = int(np.sum(bs < 0))
        ntr = len(bs)
        # 二项 p (双侧)
        from math import comb
        k = max(pos, neg)
        pb = min(1.0, 2 * sum(comb(ntr, i) for i in range(k, ntr + 1)) / 2 ** ntr)
        lines.append('%-26s %10.2f %10.2f %10s %+10.2f %8.1e %8d'
                     % (nm, e1 if np.isfinite(e1) else 0,
                        e2 if np.isfinite(e2) else 0,
                        ('%.2fx' % (e2 / e1)) if (e1 and e1 > 1e-9) else 'n/a',
                        med_b, pb, ntr))
        out[col] = dict(threshold=th, name=nm, n_stations=ntr,
                        mean_1976_85=float(e1), mean_2014_23=float(e2),
                        ratio=(float(e2 / e1) if e1 and e1 > 1e-9 else None),
                        trend_median_per_decade=med_b, trend_pop_weighted=float(wm),
                        sign_pos=pos, sign_neg=neg, sign_binom_p=float(pb))
    lines.append('')
    lines.append('对照: 现行业务使用的干球夜间指标')
    for col, nm in DRY:
        if col not in ok.columns:
            continue
        tr = station_trend(ok, col, min_years=20)
        if len(tr) < 10:
            continue
        bs = np.array([v[0] for v in tr.values()])
        e1 = ok[(ok.year >= 1976) & (ok.year <= 1985)][col].mean()
        e2 = ok[(ok.year >= 2014) & (ok.year <= 2023)][col].mean()
        lines.append('  %-26s 1976-85 = %.4f  2014-23 = %.4f  趋势 %+.5f /10yr (n=%d)'
                     % (nm, e1, e2, np.median(bs), len(bs)))
        out[col] = dict(name=nm, mean_1976_85=float(e1), mean_2014_23=float(e2),
                        trend_median_per_decade=float(np.median(bs)))
    # 关键: 各阈值的"零值站年比例" —— 量纲错位的直接证据
    lines.append('')
    lines.append('=== 量纲错位: 各阈值下"全年 0 小时"的站年比例 ===')
    for col, th, nm in METRICS:
        if col not in ok.columns:
            continue
        v = ok[col].fillna(0)
        z = float(np.mean(v <= 0))
        lines.append('  %-26s 零值站年 %.1f%%  (99分位 %.1f h/yr, 最大 %.0f h/yr)'
                     % (nm, 100 * z, np.nanpercentile(v, 99), np.nanmax(v)))
        out[col]['zero_fraction'] = z
        out[col]['p99'] = float(np.nanpercentile(v, 99))
        out[col]['max'] = float(np.nanmax(v))
    txt = '\n'.join(lines)
    print(txt)
    with open('headline_metric.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('headline_metric.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
