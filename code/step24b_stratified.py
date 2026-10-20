# -*- coding: utf-8 -*-
"""第8轮 R8-2 深化: 夜间湿球相对"夜间干球"的增量价值是否依赖湿度环境?
  审稿人最尖锐的一条: 控制 TN(夜间最低干球) 后夜间湿球不再显著 -> 湿球是否有增量价值?
  检验: 按区域"干湿程度"(干球-湿球差) 分层, 比较 Tw 与 TN 的独立效应。
产出: health_stratified.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, fit_region, meta

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    m = m[m['mon'].map(lambda d: d.year) < 2020]          # 剔除 COVID
    # 区域干湿指标: 暖季夜间 (干球最低 - 湿球) 的均值, 越大越干
    g = m.groupby('nuts2').apply(
        lambda d: pd.Series({'deficit': float((d['tn_max'] - d['tw_night_min_max']).mean()),
                             'rh': float(d.get('rh', pd.Series([np.nan])).mean())
                             if 'rh' in d else np.nan,
                             'tw': float(d['tw_night_mean_mean'].mean()),
                             'n': len(d)}), include_groups=False)
    g = g[g['n'] >= 200]
    med = g['deficit'].median()
    g['grp'] = np.where(g['deficit'] <= med, 'WET(湿, 蒸发受限)', 'DRY(干, 蒸发有效)')
    print('区域 %d, 干湿中位数 deficit = %.2f °C' % (len(g), med))
    print(g.groupby('grp')['deficit'].describe()[['count', 'mean', 'min', 'max']].round(2).to_string())

    specs = {
        'G_twmean_only': ['tw_night_mean_mean'],
        'H_twmean_vs_tx': ['tw_night_mean_mean', 'tx_mean'],
        'E_tn_only': ['tn_max'],
        'F_tw_vs_tn': ['tw_night_min_max', 'tn_max'],
    }
    out = {'median_deficit': float(med)}
    lines = []
    lines.append('按区域干湿分层 (deficit = 夜间干球最低 − 夜间湿球, 越大越干), 剔除 2020-2022')
    lines.append('干湿中位数 = %.2f °C' % med)
    lines.append('')
    for grp in ['WET(湿, 蒸发受限)', 'DRY(干, 蒸发有效)']:
        ids = set(g[g['grp'] == grp].index)
        sub = m[m['nuts2'].isin(ids)]
        lines.append('== %s : %d 区域, %d 周 ==' % (grp, len(ids), len(sub)))
        out[grp] = {'n_regions': len(ids)}
        for sp, exps in specs.items():
            rr = []
            for nid, gg in sub.groupby('nuts2'):
                r = fit_region(gg, exps)
                if r:
                    rr.append(r)
            if len(rr) < 15:
                continue
            parts = []
            for e in exps:
                mm = meta(rr, e)
                if not mm:
                    continue
                z = mm['re'] / mm['re_se'] if mm['re_se'] > 0 else float('nan')
                p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
                parts.append('%s: RR=%.4f [%.4f,%.4f] p=%.3g'
                             % (e, math.exp(mm['re']),
                                math.exp(mm['re'] - 1.96 * mm['re_se']),
                                math.exp(mm['re'] + 1.96 * mm['re_se']), p))
                out[grp]['%s|%s' % (sp, e)] = dict(beta=float(mm['re']), se=float(mm['re_se']),
                                                   rr=float(math.exp(mm['re'])), p=float(p),
                                                   n=mm['n_regions'])
            lines.append('  %-16s %s' % (sp, '   |   '.join(parts)))
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open('health_stratified.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('health_stratified.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
