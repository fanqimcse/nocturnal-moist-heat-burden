# -*- coding: utf-8 -*-
"""第10轮 R10-4  白天 vs 夜间: 同一面板内的效应量与归因负担对比

审稿人 M57: 夜间湿球效应相对白天热有多大? 若只有白天的 1/10, 政策含义有限。
RR/单位 不能直接比较(单位不同: °C 干球 vs °C 湿球 vs 度夜), 必须换成可比口径:
  (1) 每 1 个区域内标准差的 RR (RR/SD) —— 标准化效应量
  (2) 归因死亡数 —— 用各区域实际暴露分布, 以该区域 5 分位为最低风险参考,
      计算 Σ deaths × [1 − exp(−β(x − x_ref))], 折算为 每年归因死亡

产出: daynight_burden.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, fit_region, meta

SPECS = {
    'A4 白天tx + 夜间最低湿球(°C)': ['tx_max', 'tw_night_min_max'],
    'A5 白天tx + 夜间湿球超阈小时数': ['tx_max', 'nh219'],
    'A6 白天tx + 夜间干球度夜 + 湿球小时数(决定性)': ['tx_max', 'nh_tn20', 'nh219'],
    'A1 白天tx + 夜间湿球度夜(欧洲近退化, 仅作对照)': ['tx_max', 'dn219'],
    'A2 白天tx + 夜间干球度夜': ['tx_max', 'nh_tn20'],
}
WARM = (5, 9)


def main():
    m = load()
    m = m[m['n_days'] >= 5]
    mm = m['mon'].map(lambda d: d.month)
    m = m[(mm >= WARM[0]) & (mm <= WARM[1])]
    L = ['白天 vs 夜间: 标准化效应量与归因负担 (欧洲 %d 个 NUTS2, 暖季 5–9 月)' % m['nuts2'].nunique(),
         '样本: %d 区域-周' % len(m), '']
    js = {'n_regions': int(m['nuts2'].nunique()), 'n_rows': int(len(m)),
          'total_deaths': float(m['deaths'].sum()),
          'years': [int(min(x.year for x in m['mon'])), int(max(x.year for x in m['mon']))]}
    L.append('总死亡(暖季, 全期) %.0f' % m['deaths'].sum())
    nyears = js['years'][1] - js['years'][0] + 1
    L.append('覆盖年份 %d–%d (%d 年)' % (js['years'][0], js['years'][1], nyears))
    L.append('')
    js['specs'] = {}
    for nm, exps in SPECS.items():
        res = []
        for k, g in m.groupby('nuts2'):
            r = fit_region(g, exps)
            if r:
                res.append(r)
        if len(res) < 50:
            continue
        L.append('===== %s =====' % nm)
        d = dict(n_regions=len(res))
        sds = {}
        for e in exps:
            mt = meta(res, e)
            # 区域内标准差 (按区域算, 取中位)
            sd = np.median([g[e].std() for k, g in m.groupby('nuts2') if g[e].notna().sum() > 50])
            nz = float(np.mean((m[e].fillna(0) != 0)))
            sds[e] = float(sd)
            if mt is None:
                continue
            rr = math.exp(mt['re'])
            rrsd = math.exp(mt['re'] * sd)
            lo = math.exp((mt['re'] - 1.96 * mt['re_se']) * sd)
            hi = math.exp((mt['re'] + 1.96 * mt['re_se']) * sd)
            z = mt['re'] / mt['re_se']
            L.append('  %-16s RR/unit=%.5f [%.5f,%.5f] SD=%.3f 非零率%4.1f%%  RR/SD=%.4f [%.4f,%.4f] z=%+.1f I²=%.0f%%'
                     % (e, rr, math.exp(mt['re'] - 1.96 * mt['re_se']),
                        math.exp(mt['re'] + 1.96 * mt['re_se']), sd, 100 * nz, rrsd, lo, hi, z, 100 * mt['I2']))
            d[e] = dict(rr_unit=rr, se=mt['re_se'], sd=float(sd), nonzero_frac=float(nz),
                        rr_per_sd=rrsd, rr_per_sd_ci=[lo, hi], z=float(z), I2=mt['I2'])
        # ---- 归因负担 ----
        L.append('  -- 归因死亡 (区域特异 β, 参考=该区域暴露 5 分位) --')
        tot = {}
        for e in exps:
            tot[e] = 0.0
        denom = 0.0
        for k, g in m.groupby('nuts2'):
            r = [x for x in res if x['nuts2'] == k]
            if not r:
                continue
            r = r[0]
            denom += float(g['deaths'].sum())
            for e in exps:
                b = r.get(e, np.nan)
                if not np.isfinite(b):
                    continue
                x = g[e].values.astype(float)
                ref = np.nanpercentile(x, 5)
                af = 1.0 - np.exp(-b * (x - ref))
                tot[e] += float(np.nansum(g['deaths'].values.astype(float) * af))
        for e in exps:
            L.append('     %-18s 归因死亡 %9.0f   (%7.1f 万/年, 占暖季死亡 %.2f%%)'
                     % (e, tot[e], tot[e] / nyears / 1e4, 100 * tot[e] / max(denom, 1e-9)))
            d.setdefault('burden', {})[e] = dict(total=float(tot[e]),
                                                 per_year=float(tot[e] / nyears),
                                                 pct_warm_deaths=float(100 * tot[e] / max(denom, 1e-9)))
        # 夜间/白天 比
        for pair in (('nh219', 'tx_max'), ('nh_tn20', 'tx_max'),
                     ('tw_night_min_max', 'tx_max'), ('dn219', 'tx_max')):
            a, b = pair
            if a in d.get('burden', {}) and b in d.get('burden', {}):
                va = d['burden'][a]['per_year']; vb = d['burden'][b]['per_year']
                L.append('     比值 %s / %s = %.1f%%' % (a, b, 100 * va / vb if vb else float('nan')))
                d.setdefault('ratio', {})['%s_over_%s' % (a, b)] = float(va / vb) if vb else None
        L.append('')
        js['specs'][nm] = d
    txt = '\n'.join(L)
    print(txt)
    with open('daynight_burden.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('daynight_burden.json', 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
