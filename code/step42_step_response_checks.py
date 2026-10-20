# -*- coding: utf-8 -*-
"""第13轮 R13-1  阶跃效应的阴性对照、滞后检验与阈值敏感性

step41 出现一个极强的信号: 把 nh219(连续小时数) 换成 any219(二元是否跨越) 后,
  连续规格: RR/SD=1.0033  z=+5.24  I²=84%   (跨区域高度不一致)
  二元规格: RR   =1.0181  z=+4.62  I²= 1%   (跨区域几乎完全一致)

审稿人必问的三件事, 本脚本全部回答:
  (a) 阴性对照: 这是"湿球跨越"特有的, 还是"任何二元阈值变量"的通性?
      -> 用夜间干球 any_tn20 / any_tn25、白天 tx 的二元版做同样规格, 看 I² 是否也降到 1%。
         若干球的二元版 I² 依然高, 则"湿球跨越"确实是特殊的。
  (b) 死亡位移: 阶跃效应会不会只是把死亡提前几天(harvesting)?
      -> 加入滞后一周的 any219_lag, 看同期效应是否保持、滞后项是否为负。
  (c) 阈值敏感性: any(20.0) / any(21.9) / any(24.0) 的 RR 与 I² 各是多少?
         若 21.9 是最优点则支持该阈值; 若 24.0 更强则应改用更高阈值。
  (d) 用二元 RR 重算南欧/北欧归因死亡 —— 与 step40 的连续口径对照。

产出: step_response_checks.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta

BASE = os.path.dirname(os.path.abspath(__file__))
WARM = (5, 9)


def log(*a):
    print(*a, flush=True)


def prep():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM[0]) & (mo <= WARM[1])].copy()
    cent = pd.read_csv(os.path.join(BASE, 'ext', 'health', 'nuts2_centroids.csv'))
    m = m.merge(cent[['nuts2', 'cntr', 'name', 'lat', 'lon']], on='nuts2', how='left')
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    return m


def fit_binary(m, var, extra_z=('tx_max', 'nh_tn20'), lag=False):
    """二元 var 以原始 0/1 进入; extra_z 以区域内 z 进入。返回逐区域 (b, se, ll)"""
    sub = m.copy()
    for e in extra_z:
        sub[e + '_z'] = sub[e] / sub.groupby('nuts2')[e].transform('std').replace(0, np.nan)
    cols = [e + '_z' for e in extra_z] + ([var + '_lag'] if lag else []) + [var]
    use = [e + '_z' for e in extra_z] + ([var + '_lag'] if lag else []) + [var]
    rows = []
    for k, g in sub.groupby('nuts2'):
        gg = g.dropna(subset=use)
        if len(gg) < 100 or gg['deaths'].sum() < 200:
            continue
        if gg[var].std() <= 0:
            continue
        X, names = design(gg, use)
        y = gg['deaths'].values.astype(float)
        b, se, phi = poisson_glm(X, y)
        if not np.all(np.isfinite(se)):
            continue
        mu = np.exp(np.clip(X @ b, -30, 30))
        o = {'nuts2': k, 'n': len(gg),
             'll': float(np.sum(y * np.log(np.maximum(mu, 1e-9)) - mu))}
        for c in ([var + '_lag'] if lag else []) + [var]:
            i = names.index(c)
            o[c] = float(b[i]); o[c + '_se'] = float(se[i])
        rows.append(o)
    return rows


def show(tag, rows, var, L, js):
    mt = meta(rows, var)
    if not mt:
        L.append('  %-26s 区域不足' % tag)
        return None
    rr = math.exp(mt['re'])
    lo = math.exp(mt['re'] - 1.96 * mt['re_se'])
    hi = math.exp(mt['re'] + 1.96 * mt['re_se'])
    z = mt['re'] / mt['re_se'] if mt['re_se'] > 0 else float('nan')
    nz = float(np.nanmean([r['n'] for r in rows]))
    L.append('  %-26s 区域 %3d  RR=%.4f [%.4f,%.4f]  z=%+5.2f  I²=%3.0f%%  AF=%5.2f%%'
             % (tag, len(rows), rr, lo, hi, z, 100 * mt['I2'], 100 * (1 - 1 / rr)))
    js[tag] = dict(n_regions=len(rows), rr=float(rr), lo=float(lo), hi=float(hi),
                   z=float(z), I2=float(mt['I2']), af=float(1 - 1 / rr))
    return js[tag]


def main():
    m = prep()
    nyears = max(x.year for x in m['mon']) - min(x.year for x in m['mon']) + 1
    # 二元暴露
    m['any219'] = (m['nh219'] > 0).astype(float)
    m['any240'] = (m['nh240'] > 0).astype(float)
    m['any_tn20'] = (m['n_tn20'] > 0).astype(float)
    m['any_tn25'] = (m['n_tn25'] > 0).astype(float)
    m['any_tx30'] = (m['tx_max'] >= 30.0).astype(float)
    m['any219_lag'] = m.groupby('nuts2')['any219'].shift(1)

    L = ['阶跃效应的阴性对照 / 滞后检验 / 阈值敏感性 (step42)',
         '面板: %d NUTS2 / %d 区域-周 / 暖季 5-9 月 / %d 年'
         % (m['nuts2'].nunique(), len(m), nyears), '']
    js = {}

    # (a) 阴性对照
    L.append('【1】阴性对照 —— 低 I² 是"湿球跨越"特有, 还是"任何二元阈值变量"的通性?')
    L.append('     规格均控制: 白天 TX(z) + 夜间干球度夜(z)')
    L.append('')
    for tag, var in (('湿球 跨越 21.9 °C', 'any219'),
                     ('湿球 跨越 24.0 °C', 'any240'),
                     ('干球 热带夜 ≥20 °C', 'any_tn20'),
                     ('干球 热带夜 ≥25 °C', 'any_tn25'),
                     ('白天 TX ≥30 °C', 'any_tx30')):
        rows = fit_binary(m, var)
        show(tag, rows, var, L, js.setdefault('neg_control', {}))
    L.append('')
    L.append('     判读: 若只有湿球跨越的二元版 I² 极低, 而干球/白天同类二元版 I² 仍高,')
    L.append('           则"夜间湿球跨越"对应一个跨区域高度一致的生理阈值, 不是统计假象。')
    L.append('')

    # (b) 死亡位移
    L.append('【2】死亡位移检验 —— 阶跃效应是不是只是把死亡提前一周?')
    rows = fit_binary(m.dropna(subset=['any219_lag']), 'any219', lag=True)
    if rows:
        mt = meta(rows, 'any219')
        mtl = meta(rows, 'any219_lag')
        if mt and mtl:
            L.append('     同期 any219   RR=%.4f [%.4f,%.4f] z=%+.2f'
                     % (math.exp(mt['re']), math.exp(mt['re'] - 1.96 * mt['re_se']),
                        math.exp(mt['re'] + 1.96 * mt['re_se']), mt['re'] / mt['re_se']))
            L.append('     滞后 any219   RR=%.4f [%.4f,%.4f] z=%+.2f'
                     % (math.exp(mtl['re']), math.exp(mtl['re'] - 1.96 * mtl['re_se']),
                        math.exp(mtl['re'] + 1.96 * mtl['re_se']), mtl['re'] / mtl['re_se']))
            L.append('     若滞后项 ≈ 1 且同期项保持, 则不是 harvesting。')
            js['harvesting'] = dict(same=dict(rr=float(math.exp(mt['re'])),
                                              z=float(mt['re'] / mt['re_se'])),
                                    lag=dict(rr=float(math.exp(mtl['re'])),
                                             z=float(mtl['re'] / mtl['re_se'])))
    L.append('')

    # (c) 阈值敏感性 (用夜间最低湿球本身, 不依赖 nh219 分箱)
    L.append('【3】阈值敏感性 —— 用"该周最热一夜的最低湿球"直接构造二元暴露')
    for th in (20.0, 21.0, 21.9, 23.0, 24.0, 25.0):
        v = 'bin_%.1f' % th
        m[v] = (m['tw_night_min_max'] >= th).astype(float)
        rows = fit_binary(m, v)
        show('湿球 ≥ %.1f °C' % th, rows, v, L, js.setdefault('threshold', {}))
    L.append('')

    # (d) 用二元 RR 重算南欧/北欧归因死亡
    L.append('【4】用二元 RR 重算归因死亡 (与 step40 连续口径对照)')
    tw0 = m.groupby('nuts2')['tw_night_min_max'].mean()
    med = float(tw0.median())
    rows = fit_binary(m, 'any219')
    mu = None
    mt = meta(rows, 'any219')
    if mt:
        b = {r['nuts2']: r['any219'] for r in rows}
        se = {r['nuts2']: r['any219_se'] for r in rows}
        w = 1.0 / np.array([se[k] for k in b]) ** 2
        bb = np.array([b[k] for k in b])
        mu0 = float(np.sum(w * bb) / np.sum(w))
        Q = float(np.sum(w * (bb - mu0) ** 2))
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max((Q - (len(bb) - 1)) / C, 0.0) if C > 0 else 0.0
        for tag, ids in (('南欧型', set(tw0[tw0 >= med].index)), ('北欧型', set(tw0[tw0 < med].index))):
            sub = m[m['nuts2'].isin(ids)]
            tot = 0.0
            den = 0.0
            for k, g in sub.groupby('nuts2'):
                if k not in b:
                    continue
                bs = (b[k] / se[k] ** 2 + mu0 / max(tau2, 1e-12)) / (1.0 / se[k] ** 2 + 1.0 / max(tau2, 1e-12))
                x = g['any219'].values.astype(float)
                af = 1.0 - np.exp(-bs * x)
                tot += float(np.nansum(g['deaths'].values.astype(float) * af))
                den += float(g['deaths'].sum())
            L.append('     %s : %d 区域  归因死亡 %8.0f  (%6.2f 万/年, 占暖季死亡 %5.2f%%)'
                     % (tag, sub['nuts2'].nunique(), tot, tot / nyears / 1e4,
                        100 * tot / max(den, 1e-9)))
            js.setdefault('burden_binary', {})[tag] = dict(
                n_regions=int(sub['nuts2'].nunique()), total=float(tot),
                per_year=float(tot / nyears), pct=float(100 * tot / max(den, 1e-9)))
        L.append('     (对照 step40 连续口径: 南欧 0.24 万/年, 北欧 0.01 万/年)')
    L.append('')

    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'step_response_checks.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'step_response_checks.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
