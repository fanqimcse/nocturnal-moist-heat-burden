# -*- coding: utf-8 -*-
"""第8轮 R8-3 决定性检验: 本研究提出的"夜间湿球负荷"vs 现行业务夜间指标

审稿人的核心质疑: 你说湿球重要, 但控制夜间干球后它就不显著了 —— 那它相对
现行指标(日最高干球 TX、热带夜 TN≥20°C / TN≥25°C)究竟有没有增量价值?

设计(全部控制日最高干球 tx_max, 即"夜间还能在白天之上解释多少"):
  F1  tx + 夜间湿球度夜  Σ(Tw−21.9)+       <- 本研究 NHRI 口径
  F2  tx + 最热夜干球    tn_hot             <- 最强干球对照
  F3  tx + 夜间湿球度夜 + 最热夜干球        <- 湿球能否在干球之上再加
  F4  tx + 热带夜计数    N(TN≥20)
  F5  tx + 热带夜计数    N(TN≥25)
  F6  tx + 夜间干球度夜  Σ(TN−20)+
  F7  tx + 夜间湿球计数  N(Tw≥21.9)         <- 阈值计数 vs 连续负荷(量纲错位检验)
  F8  tx + 夜间湿球度夜  Σ(Tw−25.8)+        <- 更高阈值的连续负荷

每个规格同时报告: 暴露的合并 RR、以及相对 F0(tx only) 的准似然改善(ΔQ, 越大越好)
产出: health_operational_compare.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta, log

BASE = os.path.dirname(os.path.abspath(__file__))


def fit_region(g, exposures, min_n=100):
    X, names = design(g, exposures)
    y = g['deaths'].values.astype(float)
    if len(g) < min_n or y.sum() < 200:
        return None
    b, se, phi = poisson_glm(X, y)
    if not np.all(np.isfinite(se)):
        return None
    mu = np.exp(np.clip(X @ b, -30, 30))
    # 准似然(用于跨规格的信息增益比较)
    ll = float(np.sum(y * np.log(np.maximum(mu, 1e-9)) - mu))
    out = {'nuts2': g['nuts2'].iloc[0], 'n': len(g), 'll': ll}
    for e in exposures:
        i = names.index(e)
        out[e] = float(b[i]); out[e + '_se'] = float(se[i])
    return out


def run_spec(m, exposures, label, lines, out):
    rr = [r for _, g in m.groupby('nuts2') for r in [fit_region(g, exposures)] if r]
    tot_ll = sum(r['ll'] for r in rr)
    rec = {'n_regions': len(rr), 'quasi_ll': tot_ll,
           '_ll_by_region': {r['nuts2']: r['ll'] for r in rr}}
    parts = []
    for e in exposures:
        mm = meta(rr, e)
        if not mm:
            continue
        z = mm['re'] / mm['re_se'] if mm['re_se'] > 0 else float('nan')
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        parts.append('%s RR=%.5f [%.5f,%.5f] p=%.3g'
                     % (e, math.exp(mm['re']), math.exp(mm['re'] - 1.96 * mm['re_se']),
                        math.exp(mm['re'] + 1.96 * mm['re_se']), p))
        rec[e] = dict(rr=float(math.exp(mm['re'])), beta=float(mm['re']),
                      se=float(mm['re_se']), p=float(p), I2=mm['I2'],
                      n_regions=mm['n_regions'])
    lines.append('%-42s %s' % (label, '   '.join(parts)))
    out[label] = rec
    return rec


def main():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    m = m[m['mon'].map(lambda d: d.year) < 2020]
    log('样本: %d 周 / %d 区域' % (len(m), m['nuts2'].nunique()))

    lines = ['夜间湿球负荷 vs 现行业务夜间指标 —— 全部控制"日最高干球 TX"',
             '欧洲 NUTS2 暖季周 2001-2019, %d 周 / %d 区域, 区域固定效应+年份因子+Fourier季节'
             % (len(m), m['nuts2'].nunique()),
             '']
    # 滞后暴露(死亡位移/harvesting 检验): 前一周的暴露
    for c in ['dn219', 'nh_tn20', 'tx_max']:
        m[c + '_lag'] = m.groupby('nuts2')[c].shift(1)
    m = m.dropna(subset=['dn219_lag'])

    out = {'n_weeks': int(len(m)), 'n_regions': int(m['nuts2'].nunique())}

    specs = [
        ('F0 仅白天 tx_max', ['tx_max']),
        ('F1 +夜间湿球度夜 Σ(Tw−21.9)+', ['tx_max', 'dn219']),
        ('F2 +最热夜干球 tn_hot', ['tx_max', 'tn_hot']),
        ('F3 +湿球度夜 +最热夜干球', ['tx_max', 'dn219', 'tn_hot']),
        ('F4 +热带夜计数 N(TN≥20)', ['tx_max', 'n_tn20']),
        ('F5 +热夜计数   N(TN≥25)', ['tx_max', 'n_tn25']),
        ('F6 +夜间干球度夜 Σ(TN−20)+', ['tx_max', 'nh_tn20']),
        ('F7 +夜间湿球计数 N(Tw≥21.9)', ['tx_max', 'n_tw219']),
        ('F8 +夜间湿球度夜 Σ(Tw−25.8)+', ['tx_max', 'dn258']),
        # 决定性: 湿球负荷能否在"干球负荷"之上再加信息
        ('F9 干球度夜 +湿球度夜(决定性)', ['tx_max', 'nh_tn20', 'dn219']),
        ('F10 干球度夜 +湿球度夜(阈24)', ['tx_max', 'nh_tn20', 'dn240']),
        ('F11 干球度夜 +最热夜干球', ['tx_max', 'nh_tn20', 'tn_hot']),
        # 死亡位移(harvesting)检验
        ('F12 仅滞后1周: 干球+湿球度夜', ['tx_max_lag', 'nh_tn20_lag', 'nh219_lag']),
        ('F13 同期+滞后: 干球+湿球度夜', ['tx_max', 'nh_tn20', 'dn219', 'dn219_lag']),
    ]
    for label, exps in specs:
        miss = [e for e in exps if e not in m.columns]
        if miss:
            lines.append('%-42s 跳过(缺列 %s)' % (label, miss))
            continue
        run_spec(m, exps, label, lines, out)

    # ΔLL 必须在"所有规格都收敛的共同区域集"上比较, 否则区域数不同会把量级搞乱
    common = None
    for label, _ in specs:
        if label in out and '_ll_by_region' in out[label]:
            s0 = set(out[label]['_ll_by_region'])
            common = s0 if common is None else (common & s0)
    common = sorted(common) if common else []
    lines.append('')
    lines.append('=== 相对"仅白天 TX"的准似然增益 ΔLL (共同区域集 n=%d, 越大 = 夜间信息越多) ===' % len(common))
    base = sum(out['F0 仅白天 tx_max']['_ll_by_region'][r] for r in common) \
        if 'F0 仅白天 tx_max' in out and common else float('nan')
    for label, _ in specs:
        if label not in out or '_ll_by_region' not in out[label]:
            continue
        tot = sum(out[label]['_ll_by_region'][r] for r in common)
        dl = tot - base
        out[label]['delta_ll_common'] = float(dl)
        lines.append('  %-42s ΔLL = %+.1f  (n=%d 区)' % (label, dl, len(common)))
    for label in list(out.keys()):
        if isinstance(out[label], dict):
            out[label].pop('_ll_by_region', None)

    txt = '\n'.join(lines)
    print()
    print(txt)
    with open('health_operational_compare.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('health_operational_compare.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
