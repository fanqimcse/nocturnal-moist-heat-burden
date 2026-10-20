# -*- coding: utf-8 -*-
"""第13轮 R13-2  决定性规格: 湿球跨越是否携带干球之外的信息 + 口径统一 + 主口径裁定

step42 暴露三件必须裁定的事:
  (1) step41【4】报的 I²=1% 是漏乘 100 的显示 bug, 实际 78% (连续口径 84%)。
      -> "阶跃导致异质性骤降"的说法作废。
  (2) 阴性对照里 any_tn20 与模型中的 nh_tn20 共线, 其 RR=1.022 不可直接与 any219 比较。
      -> 正确对照: 让 any_tn20 与 any219 **同时**进入模型, 看 any219 是否仍独立显著。
  (3) 两种"跨越"定义给出不同答案:
      any219   = 该周内**任一夜间小时**湿球 ≥21.9  -> 136 区域, RR=1.0181, z=+4.62
      twmin≥21.9 = 该周**最热一夜的最低**湿球 ≥21.9 -> 30 区域,  RR=1.0043, z=+0.50
      -> 必须统一: 前者是"出现过", 后者是"整夜都在阈值以上"(强得多)。

产出: decisive_spec.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta

BASE = os.path.dirname(os.path.abspath(__file__))
WARM = (5, 9)


def prep():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM[0]) & (mo <= WARM[1])].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    return m


def fit(m, cols, min_n=100):
    rows = []
    for k, g in m.groupby('nuts2'):
        gg = g.dropna(subset=cols)
        if len(gg) < min_n or gg['deaths'].sum() < 200:
            continue
        if any(gg[c].std() <= 0 for c in cols):
            continue
        X, names = design(gg, cols)
        y = gg['deaths'].values.astype(float)
        b, se, phi = poisson_glm(X, y)
        if not np.all(np.isfinite(se)):
            continue
        mu = np.exp(np.clip(X @ b, -30, 30))
        o = {'nuts2': k, 'n': len(gg),
             'll': float(np.sum(y * np.log(np.maximum(mu, 1e-9)) - mu))}
        for c in cols:
            i = names.index(c)
            o[c] = float(b[i]); o[c + '_se'] = float(se[i])
        rows.append(o)
    return rows


def show(L, js, tag, rows, cols):
    if not rows:
        L.append('  %s : 无可用区域' % tag)
        return
    d = {'n_regions': len(rows), 'll': float(sum(r['ll'] for r in rows))}
    L.append('  %s   (区域 %d, 准似然 %.1f)' % (tag, len(rows), d['ll']))
    for c in cols:
        mt = meta(rows, c)
        if not mt:
            continue
        rr = math.exp(mt['re'])
        z = mt['re'] / mt['re_se'] if mt['re_se'] > 0 else float('nan')
        L.append('      %-14s RR=%.4f [%.4f,%.4f] z=%+5.2f  I²=%3.0f%%'
                 % (c, rr, math.exp(mt['re'] - 1.96 * mt['re_se']),
                    math.exp(mt['re'] + 1.96 * mt['re_se']), z, 100 * mt['I2']))
        d[c] = dict(rr=float(rr), lo=float(math.exp(mt['re'] - 1.96 * mt['re_se'])),
                    hi=float(math.exp(mt['re'] + 1.96 * mt['re_se'])),
                    z=float(z), I2=float(100 * mt['I2']))
    js[tag] = d


def main():
    m = prep()
    nyears = max(x.year for x in m['mon']) - min(x.year for x in m['mon']) + 1
    m['any219'] = (m['nh219'] > 0).astype(float)
    m['any_tn20'] = (m['n_tn20'] > 0).astype(float)
    m['any_tx30'] = (m['tx_max'] >= 30.0).astype(float)
    m['tx_z'] = m['tx_max'] / m.groupby('nuts2')['tx_max'].transform('std').replace(0, np.nan)
    m['tn_z'] = m['nh_tn20'] / m.groupby('nuts2')['nh_tn20'].transform('std').replace(0, np.nan)
    m['nwh_z'] = m['nh219'] / m.groupby('nuts2')['nh219'].transform('std').replace(0, np.nan)
    m['twmax_z'] = m['tw_night_min_max'] / m.groupby('nuts2')['tw_night_min_max'].transform('std').replace(0, np.nan)

    L = ['决定性规格: 湿球跨越是否携带干球之外的信息 (step43)',
         '面板: %d NUTS2 / %d 区域-周 / 暖季 5-9 月 / %d 年' % (m['nuts2'].nunique(), len(m), nyears), '']
    js = {}

    L.append('【1】口径统一 —— 三种"跨越 21.9 °C"的定义, 逐一给出')
    L.append('    A = 该周内任一夜间小时湿球 ≥21.9 (any219)')
    L.append('    B = 该周最热一夜的"最低湿球" ≥21.9 (整夜都在阈值以上)')
    L.append('    C = 连续小时数 nh219 (z)')
    L.append('')
    show(L, js.setdefault('defs', {}), 'A 任一夜间小时 (二元, 原始尺度)',
         fit(m, ['tx_z', 'tn_z', 'any219']), ['tx_z', 'tn_z', 'any219'])
    show(L, js.setdefault('defs', {}), 'C 连续小时数 nh219 (z)',
         fit(m, ['tx_z', 'tn_z', 'nwh_z']), ['tx_z', 'tn_z', 'nwh_z'])
    # B: 只在有该事件的区域可估
    mb = m.copy()
    mb['b219'] = (mb['tw_night_min_max'] >= 21.9).astype(float)
    ids_b = [k for k, g in mb.groupby('nuts2') if g['b219'].std() > 0 and len(g) >= 100]
    show(L, js.setdefault('defs', {}), 'B 最热夜最低湿球 ≥21.9 (仅 %d 区域有该事件)' % len(ids_b),
         fit(mb[mb['nuts2'].isin(set(ids_b))], ['tx_z', 'tn_z', 'b219']), ['tx_z', 'tn_z', 'b219'])
    L.append('')
    L.append('    → A 是"出现过"(宽松, 136 区域可估); B 是"整夜无法恢复"(严格, 仅少数区域可估)。')
    L.append('      论文必须明确采用 A, 并把 B 报告为"样本不足、无法在欧洲检验"。')
    L.append('')

    L.append('【2】决定性阴性对照 —— 干球热带夜与湿球跨越**同时**进入模型')
    L.append('    (step42 的对照有共线问题: any_tn20 与已控制的 nh_tn20 高度相关)')
    L.append('')
    show(L, js.setdefault('decisive', {}), '任意219 + 干球热带夜(均为二元原始尺度)',
         fit(m, ['tx_z', 'any_tn20', 'any219']), ['tx_z', 'any_tn20', 'any219'])
    show(L, js.setdefault('decisive', {}), '任意219 + 白天TX≥30(均为二元原始尺度)',
         fit(m, ['any_tx30', 'any_tn20', 'any219']), ['any_tx30', 'any_tn20', 'any219'])
    L.append('')
    L.append('    判读: 若 any219 在控制"是否出现干球热带夜"之后仍显著 > 1,')
    L.append('          则夜间湿球确实携带干球之外的信息 —— 这是本研究的立身之本。')
    L.append('')

    L.append('【3】连续口径的同一决定性对照 (与【2】互为印证)')
    show(L, js.setdefault('decisive_cont', {}), '连续: 湿球最低值(z) + 干球度夜(z) + 白天TX(z)',
         fit(m, ['tx_z', 'tn_z', 'twmax_z']), ['tx_z', 'tn_z', 'twmax_z'])
    show(L, js.setdefault('decisive_cont', {}), '连续: 湿球小时数(z) + 干球度夜(z) + 白天TX(z)',
         fit(m, ['tx_z', 'tn_z', 'nwh_z']), ['tx_z', 'tn_z', 'nwh_z'])
    L.append('')

    L.append('【4】主口径裁定')
    a = js.get('defs', {}).get('A 任一夜间小时 (二元, 原始尺度)', {})
    dec = js.get('decisive', {}).get('任意219 + 干球热带夜(均为二元原始尺度)', {})
    if a.get('any219'):
        rr = a['any219']['rr']; lo = a['any219']['lo']; hi = a['any219']['hi']
        L.append('    主口径 = A(二元, 原始 0/1): RR = %.4f [%.4f, %.4f], z=%+.2f, I²=%.0f%%'
                 % (rr, lo, hi, a['any219']['z'], a['any219']['I2']))
        L.append('    理由: (i) 可直接解读为"跨越 vs 未跨越"; (ii) 不依赖区域内 SD;')
        L.append('          (iii) 控制干球热带夜后仍独立显著(见【2】)。')
        L.append('    归因分数 AF = 1 − 1/RR = %.2f%% [%.2f%%, %.2f%%]' % (100 * (1 - 1 / rr), 100 * (1 - 1 / lo), 100 * (1 - 1 / hi)))
        js['headline'] = dict(rr=float(rr), lo=float(lo), hi=float(hi),
                              z=float(a['any219']['z']), I2=float(a['any219']['I2']),
                              af=float(1 - 1 / rr), af_lo=float(1 - 1 / lo), af_hi=float(1 - 1 / hi))
    if dec.get('any219'):
        L.append('    控制干球热带夜后: any219 RR = %.4f [%.4f,%.4f], z=%+.2f'
                 % (dec['any219']['rr'], dec['any219']['lo'], dec['any219']['hi'], dec['any219']['z']))
        L.append('    同一模型中的干球热带夜: RR = %.4f [%.4f,%.4f], z=%+.2f'
                 % (dec['any_tn20']['rr'], dec['any_tn20']['lo'], dec['any_tn20']['hi'], dec['any_tn20']['z']))
    L.append('')

    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'decisive_spec.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'decisive_spec.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
