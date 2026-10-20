# -*- coding: utf-8 -*-
"""第8轮 R8-4: 湿度的增量价值是否随"区域湿热本底"增强 —— 向热带外推的直接证据

欧洲样本内部就存在巨大的湿热梯度: 斯堪的纳维亚暖季夜间湿球 ~11 °C,
地中海 ~18 °C。若"湿度在高温下才致命"成立, 则 F9 规格下 nh219 的系数
应随区域湿热本底单调放大。这是对"热带地区湿球更致命"这一外推的**内部检验**。

产出: health_climate_gradient.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta, log
from step24d_operational_compare import fit_region

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    m = m[m['mon'].map(lambda d: d.year) < 2020]
    # 区域湿热本底: 暖季"最热一夜"湿球的均值 + 干球均值
    g = m.groupby('nuts2').agg(
        tw0=('tw_hot', 'mean'), tn0=('tn_hot', 'mean'),
        nh219_0=('nh219', 'mean'), n=('deaths', 'size')).reset_index()
    g = g[g['n'] >= 200]
    m = m[m['nuts2'].isin(set(g['nuts2']))]
    q = g['tw0'].quantile([0.25, 0.5, 0.75]).to_dict()
    log('区域 %d, 暖季最热夜湿球本底: 中位 %.2f, IQR %.2f-%.2f, 全距 %.2f-%.2f'
        % (len(g), q[0.5], q[0.25], q[0.75], g.tw0.min(), g.tw0.max()))

    g['grp'] = pd.cut(g['tw0'], [g.tw0.min() - 1, q[0.25], q[0.5], q[0.75], g.tw0.max() + 1],
                      labels=['Q1 最凉', 'Q2', 'Q3', 'Q4 最湿热'])
    lines = ['湿度增量价值随区域湿热本底的梯度 (欧洲 NUTS2, 暖季周 2001-2019, 剔除COVID)',
             '规格 F9: 死亡 ~ 区域FE + 年 + 季节 + 日最高TX + 夜间干球负荷Σ(TN−20)+ + 夜间湿球负荷Σ(Tw−21.9)+',
             '暴露已在各区域内除以自身标准差 -> 系数可跨气候区直接比较(避免"湿热区超阈常见"的机械稀释)',
             '']
    out = {'quantiles': {str(k): float(v) for k, v in q.items()}}
    rows = []
    for grp in ['Q1 最凉', 'Q2', 'Q3', 'Q4 最湿热']:
        ids = set(g[g['grp'] == grp]['nuts2'])
        sub = m[m['nuts2'].isin(ids)]
        if len(ids) < 10:
            continue
        # 关键: 暴露在区域内标准化(除以该区域自身 SD), 否则"湿热区超阈常见"
        # 会机械地稀释每单位效应, 造成虚假的负梯度
        EX = ['tx_max', 'nh_tn20', 'nh219']
        sub = sub.copy()
        sds = {}
        for e in EX:
            sd = sub.groupby('nuts2')[e].transform('std')
            sds[e] = float(np.nanmedian(sd))
            sub[e + '_z'] = sub[e] / sub.groupby('nuts2')[e].transform('std')
        rr = [r for _, gg in sub.groupby('nuts2')
              for r in [fit_region(gg, [e + '_z' for e in EX], min_n=100)] if r]
        rec = {'n_regions': len(rr), 'n_weeks': int(len(sub)),
               'tw0_mean': float(g[g['grp'] == grp]['tw0'].mean()),
               'sd': sds,
               'nh219_mean': float(g[g['grp'] == grp]['nh219_0'].mean())}
        parts = []
        for e in EX:
            mm = meta(rr, e + '_z')
            if not mm:
                continue
            z = mm['re'] / mm['re_se'] if mm['re_se'] > 0 else float('nan')
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            rec[e + '_z'] = dict(rr_per_sd=float(math.exp(mm['re'])), beta=float(mm['re']),
                                 se=float(mm['re_se']), p=float(p), I2=mm['I2'])
            parts.append('%s/SD RR=%.4f p=%.3g' % (e, math.exp(mm['re']), p))
        out[str(grp)] = rec
        rows.append((str(grp), rec))
        lines.append('%-12s n=%2d区 %5d周 夜湿球本底%.2f°C 度夜均值%.2f | %s'
                     % (grp, rec['n_regions'], rec['n_weeks'], rec['tw0_mean'],
                        rec['nh219_mean'], '   '.join(parts)))

    # 单调性检验: nh219 系数 vs 区域湿热本底 的相关
    xs, ys, ss = [], [], []
    for grp, rec in rows:
        if 'nh219_z' in rec:
            xs.append(rec['tw0_mean']); ys.append(rec['nh219_z']['beta']); ss.append(rec['nh219_z']['se'])
    if len(xs) >= 3:
        xa = np.array(xs); ya = np.array(ys); sa = np.array(ss)
        w = 1 / sa ** 2
        A = np.column_stack([np.ones(len(xa)), xa])
        try:
            C = np.linalg.inv(A.T @ (A * w[:, None]))
            c = C @ (A.T @ (w * ya))
            se_slope = math.sqrt(C[1, 1])
            z = c[1] / se_slope if se_slope > 0 else float('nan')
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            lines.append('')
            lines.append('加权回归: nh219 标准化系数(每1SD) ~ 区域夜湿球本底')
            lines.append('  斜率 = %+.5f ± %.5f /°C (z=%+.2f, p=%.3g)' % (c[1], se_slope, z, p))
            lines.append('  即: 区域暖季夜间湿球本底每高 1 °C, 湿球负荷每 1SD 的死亡 RR 变化 %+.2f%%'
                         % (100 * c[1]))
            out['gradient'] = dict(slope=float(c[1]), se=float(se_slope), z=float(z), p=float(p))
        except Exception as e:
            lines.append('  梯度回归失败 %s' % e)

    txt = '\n'.join(lines)
    print()
    print(txt)
    with open('health_climate_gradient.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('health_climate_gradient.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
