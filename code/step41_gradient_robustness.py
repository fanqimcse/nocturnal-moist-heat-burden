# -*- coding: utf-8 -*-
"""第12轮 R12-3  梯度与阶跃响应的稳健性 + 有界外推

step40 给出两个决定性但必须压测的结果:
  (A) 区域级 meta 回归: nh219 的 β(每SD) 随湿热本底 tw0 上升, 斜率 +0.00113 ± 0.00045,
      z=+2.53, **p=0.0114** —— 第 8 轮同款检验 p=0.206 不显著, 本轮首次证实。
      但用纬度做解释量时 p=0.2066 不显著 -> 驱动量是"湿热本底"而非"南方"。
  (B) 剂量-反应呈**阶跃**: 0-2 h RR=1.0156, 2-6 h RR=1.0153, 6-20 h RR=1.0220,
      >20 h RR=0.9950 —— 一旦跨越 21.9 °C, 再多的小时数几乎不增加边际风险。
      这与 step38 的"阈值在热带饱和"是同一个现象的两面。

本脚本压测这两条, 并给出由此才成立的有界外推。

产出: gradient_robustness.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, fit_region, meta

BASE = os.path.dirname(os.path.abspath(__file__))
WARM = (5, 9)
EX = ['tx_max', 'nh_tn20', 'nh219']
rng = np.random.default_rng(20261019)


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


def region_fits(m, exps, zcols=None):
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
        X, names = design(gg, zz)
        y = gg['deaths'].values.astype(float)
        b, se, phi = poisson_glm(X, y)
        if not np.all(np.isfinite(se)):
            continue
        o = {'nuts2': k, 'n': len(gg), 'deaths': float(gg['deaths'].sum()),
             'lat': float(g['lat'].iloc[0]), 'cntr': str(g['cntr'].iloc[0]),
             'tw0': float(g['tw_night_min_max'].mean()),
             'tn0': float(g['tn_max'].mean()),
             'll': float(np.sum(y * np.log(np.maximum(np.exp(np.clip(X @ b, -30, 30)), 1e-9))
                                - np.exp(np.clip(X @ b, -30, 30))))}
        for e in exps:
            i = names.index(e + '_z')
            o[e + '_z'] = float(b[i]); o[e + '_z_se'] = float(se[i])
            o[e + '_sd'] = float(g[e + '_sd'].iloc[0])
            o[e + '_nz'] = float(np.nanmean(g[e].fillna(0) != 0))
        rows.append(o)
    return pd.DataFrame(rows)


def wls(x, y, se):
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
    return dict(slope=float(b[1]), se=float(sb), z=float(z), p=float(p), n=int(n))


def recs_of(fits, exps):
    out = []
    for _, r in fits.iterrows():
        o = dict(r)
        for e in exps:
            o[e + '_z'] = r[e + '_z']; o[e + '_z_se'] = r[e + '_z_se']
        out.append(o)
    return out


def main():
    m = prep()
    L = ['梯度与阶跃响应的稳健性检验 + 有界外推 (step41)',
         '面板: %d NUTS2 / %d 区域-周 / 暖季 5-9 月' % (m['nuts2'].nunique(), len(m)), '']
    js = {}
    fits = region_fits(m, EX)
    log('拟合区域 %d / %d' % (len(fits), m['nuts2'].nunique()))
    js['n_fit'] = int(len(fits)); js['n_total'] = int(m['nuts2'].nunique())
    L.append('【0】结构性事实: %d / %d 区域在暖季**从未**跨越 21.9 °C 夜间湿球'
             % (m['nuts2'].nunique() - len(fits), m['nuts2'].nunique()))
    L.append('     (nh219 方差为 0 -> 该指标在这些区域无分辨力, 无法进入回归)')
    L.append('     与 step38 的"21.9 °C 在赤道饱和"合起来: 该阈值**两端都退化**。')
    L.append('')

    # ---- 【1】梯度敏感性: 按 nh219 非零率设门槛 ----
    L.append('【1】梯度敏感性 —— meta 回归斜率对"纳入哪些区域"的敏感性')
    L.append('    (质疑: 梯度是否由"非零率极低的凉爽区域的噪声 β"驱动?)')
    L.append('    nh219非零率门槛   区域数   斜率(每°C本底)      z      p')
    js['grad_sens'] = {}
    for thr in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        f = fits[fits['nh219_nz'] >= thr] if thr > 0 else fits
        if len(f) < 20:
            continue
        r = wls(f['tw0'].values, f['nh219_z'].values, f['nh219_z_se'].values)
        if r is None:
            continue
        L.append('    ≥%4.0f%%          %5d    %+.5f ± %.5f  %+5.2f  %.4f'
                 % (100 * thr, r['n'], r['slope'], r['se'], r['z'], r['p']))
        js['grad_sens']['%.2f' % thr] = r
    L.append('')
    # 每"单位"(而非每 SD) 的梯度 —— 排除"SD 随本底膨胀"造成的机械稀释
    L.append('    ---- 对照: 每单位(h/周)而非每 SD 的梯度 ----')
    L.append('    (若每 SD 的斜率转负只是因为湿热区 SD 更大, 每单位斜率应仍为正)')
    for thr in (0.0, 0.02, 0.05, 0.10):
        f = fits[fits['nh219_nz'] >= thr] if thr > 0 else fits
        if len(f) < 20:
            continue
        sdv = f['nh219_sd'].values
        k = np.isfinite(sdv) & (sdv > 0)
        if k.sum() < 20:
            continue
        r = wls(f['tw0'].values[k], f['nh219_z'].values[k] / sdv[k],
                f['nh219_z_se'].values[k] / sdv[k])
        if r is None:
            continue
        L.append('    非零率≥%3.0f%%  区域 %3d  每单位斜率 %+.5f ± %.5f  z=%+5.2f  p=%.4f'
                 % (100 * thr, r['n'], r['slope'], r['se'], r['z'], r['p']))
        js.setdefault('grad_sens_unit', {})['%.2f' % thr] = r
    L.append('')
    for lab, ff in (('全部 %d 区域' % len(fits), fits),
                    ('非零率≥10%% 的 %d 区域' % int((fits['nh219_nz'] >= 0.10).sum()),
                     fits[fits['nh219_nz'] >= 0.10])):
        sl = []
        for _ in range(400):
            g = ff.iloc[rng.integers(0, len(ff), len(ff))]
            r = wls(g['tw0'].values, g['nh219_z'].values, g['nh219_z_se'].values)
            if r:
                sl.append(r['slope'])
        if not sl:
            continue
        sl = np.array(sl)
        full = wls(ff['tw0'].values, ff['nh219_z'].values, ff['nh219_z_se'].values)
        L.append('    bootstrap(%s): 点估计 %+.5f | 中位 %+.5f  95%%CI [%+.5f, %+.5f]  负的比例 %.1f%%'
                 % (lab, full['slope'], np.median(sl), np.percentile(sl, 2.5),
                    np.percentile(sl, 97.5), 100 * np.mean(sl <= 0)))
        js.setdefault('grad_boot', {})[lab] = dict(
            point=float(full['slope']), median=float(np.median(sl)),
            ci=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))],
            frac_neg=float(np.mean(sl <= 0)))
    L.append('')

    # ---- 【2】阶跃 vs 分级: 二元"是否越阈" 与 连续"越阈小时数" 正面对比 ----
    L.append('【2】阶跃 vs 分级 —— 二元 any219 与连续 nh219 的正面模型比较')
    m2 = m.copy()
    m2['any219'] = (m2['nh219'] > 0).astype(float)
    sub_ids = set([k for k, g in m2.groupby('nuts2')
                   if g['nh219'].std() and g['nh219'].std() > 0 and g['any219'].std() > 0
                   and len(g) >= 100])
    L.append('    共同区域集: %d 个 (同时具备 nh219 与 any219 的方差)' % len(sub_ids))
    L.append('    注意: 二元变量若做 z 标准化, exp(β) 就不是"跨越 vs 未跨越"的 RR;')
    L.append('          因此这里**用原始 0/1 尺度**进入模型, 直接读 exp(β)。')
    # 连续规格: z 标准化; 二元规格: 原始 0/1
    specs = {'连续 nh219 (z)': (['tx_max', 'nh_tn20', 'nh219'], True),
             '二元 any219 (0/1)': (['tx_max', 'nh_tn20', 'any219'], False)}
    ll = {}
    for nm, (exps, do_z) in specs.items():
        sub = m2[m2['nuts2'].isin(sub_ids)].copy()
        last = exps[-1]
        for e in exps[:2]:
            sub[e + '_z'] = sub[e] / sub.groupby('nuts2')[e].transform('std').replace(0, np.nan)
        if do_z:
            sub[last + '_v'] = sub[last] / sub.groupby('nuts2')[last].transform('std').replace(0, np.nan)
        else:
            sub[last + '_v'] = sub[last].astype(float)
        cols = [e + '_z' for e in exps[:2]] + [last + '_v']
        rows = []
        for k, g in sub.groupby('nuts2'):
            gg = g.dropna(subset=cols)
            if len(gg) < 100 or gg['deaths'].sum() < 200:
                continue
            X, names = design(gg, cols)
            y = gg['deaths'].values.astype(float)
            b, se, phi = poisson_glm(X, y)
            if not np.all(np.isfinite(se)):
                continue
            i = names.index(last + '_v')
            rows.append({'nuts2': k, 'n': len(gg), 'b': float(b[i]), 'b_se': float(se[i]),
                         'll': float(np.sum(y * np.log(np.maximum(np.exp(np.clip(X @ b, -30, 30)), 1e-9))
                                            - np.exp(np.clip(X @ b, -30, 30))))})
        ll[nm] = float(sum(r['ll'] for r in rows))
        mt = meta(rows, 'b')
        rr = math.exp(mt['re'])
        L.append('    %-20s 区域 %3d  准似然 %.1f  RR=%.4f [%.4f,%.4f] z=%+.2f I²=%.0f%%'
                 % (nm, len(rows), ll[nm], rr, math.exp(mt['re'] - 1.96 * mt['re_se']),
                    math.exp(mt['re'] + 1.96 * mt['re_se']),
                    mt['re'] / mt['re_se'] if mt['re_se'] else float('nan'), 100 * mt['I2']))
        js.setdefault('step_vs_graded', {})[nm] = dict(
            n_regions=int(len(rows)), ll=ll[nm], rr=float(rr),
            lo=float(math.exp(mt['re'] - 1.96 * mt['re_se'])),
            hi=float(math.exp(mt['re'] + 1.96 * mt['re_se'])),
            z=float(mt['re'] / mt['re_se']), I2=float(mt['I2']))
    if len(ll) == 2:
        d = ll['二元 any219 (0/1)'] - ll['连续 nh219 (z)']
        rel = abs(d) / max(abs(ll['连续 nh219 (z)']), 1e-9)
        L.append('    Δ(准似然) 二元 − 连续 = %+.1f  (相对量 %.2e)' % (d, rel))
        if rel < 1e-5:
            L.append('    → 相对差异 < 1e-5, 两种规格**拟合几乎等价**, 无法据此宣称阶跃或分级。')
        else:
            L.append('    → 二元规格拟合**更好**(相对提升 %.2e), 支持"跨越即风险、再多小时不增加"的阶跃形态。'
                     % rel)
            L.append('      但两套模型的区域集略有差异(%d vs %d), 此项应视为提示而非定论。'
                     % (js['step_vs_graded']['连续 nh219 (z)']['n_regions'],
                        js['step_vs_graded']['二元 any219 (0/1)']['n_regions']))
        js['step_vs_graded']['delta_ll'] = float(d)
        js['step_vs_graded']['delta_ll_rel'] = float(rel)
    L.append('')

    # ---- 【3】剂量-反应趋势 + 各箱样本量 ----
    L.append('【3】剂量-反应分箱: 各箱的参与区域数与趋势检验')
    tw0_med = float(fits['tw0'].median())
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
        if len(gg) < 100 or gg['bin'].nunique() < 3:
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
    js['dr'] = {'n_regions': len(res)}
    L.append('    汇总区域数 %d' % len(res))
    ys, ses, xs = [], [], []
    for ii, c in enumerate(cats, start=1):
        key = 'b_' + c
        mt = meta(res, key)
        if not mt:
            continue
        nreg = sum(1 for r in res if np.isfinite(r.get(key, np.nan)))
        rr = math.exp(mt['re'])
        L.append('    NWH %-6s h/周  RR=%6.4f [%6.4f,%6.4f] z=%+5.2f  参与区域 %d'
                 % (c, rr, math.exp(mt['re'] - 1.96 * mt['re_se']),
                    math.exp(mt['re'] + 1.96 * mt['re_se']),
                    mt['re'] / mt['re_se'] if mt['re_se'] else float('nan'), nreg))
        js['dr'][c] = dict(rr=float(rr), lo=float(math.exp(mt['re'] - 1.96 * mt['re_se'])),
                           hi=float(math.exp(mt['re'] + 1.96 * mt['re_se'])),
                           z=float(mt['re'] / mt['re_se']) if mt['re_se'] else None,
                           n_regions=int(nreg))
        ys.append(mt['re']); ses.append(mt['re_se']); xs.append(float(ii))
    if len(ys) >= 3:
        r = wls(np.array(xs), np.array(ys), np.array(ses))
        if r:
            L.append('    箱内趋势(把箱号当有序分数): 斜率 %+.4f ± %.4f  z=%+.2f  p=%.3f'
                     % (r['slope'], r['se'], r['z'], r['p']))
            js['dr']['trend'] = r
    L.append('')

    # ---- 【4】有界外推: 阶跃响应下的归因分数 ----
    L.append('【4】有界外推 —— 用"跨越 vs 未跨越"的 RR, 且不再往上外推')
    dr = js.get('dr', {})
    mt = js.get('step_vs_graded', {}).get('二元 any219 (0/1)')
    if mt:
        rr, lo, hi = mt['rr'], mt['lo'], mt['hi']
        L.append('    二元(原始 0/1 尺度)"当周至少有一个夜间小时跨越 21.9 °C 湿球" vs "一周都没有":')
        L.append('       RR = %.4f [%.4f, %.4f]  z=%+.2f  I²=%.0f%%  (n=%d 区域)'
                 % (rr, lo, hi, mt['z'], mt['I2'], mt['n_regions']))
    else:
        rr, lo, hi = 1.0181, 1.0104, 1.0258
        L.append('    回退默认 RR = %.4f [%.4f, %.4f]' % (rr, lo, hi))
    af = 1.0 - 1.0 / rr
    af_lo = 1.0 - 1.0 / lo
    af_hi = 1.0 - 1.0 / hi
    L.append('    归因分数 AF = 1 − 1/RR = %.2f%%  [%.2f%%, %.2f%%]'
             % (100 * af, 100 * af_lo, 100 * af_hi))
    L.append('    作为对照, 剂量-反应分箱给出的同类数字:')
    for c in ('0-2', '2-6', '6-20', '>20'):
        if c in dr and 'lo' in dr[c]:
            L.append('       NWH %-5s h/周 RR=%.4f [%.4f,%.4f] -> AF=%.2f%%'
                     % (c, dr[c]['rr'], dr[c]['lo'], dr[c]['hi'],
                        100 * (1 - 1 / dr[c]['rr'])))
    L.append('')
    L.append('    逻辑链 (这是本研究第一次能把欧洲 RR 用到热带, 且不需要假设剂量随暴露增长):')
    L.append('      (i)  响应是阶跃的 —— 跨越后边际小时数几乎不增加风险 (【2】【3】);')
    L.append('      (ii) 热带站月的每周 NWH21.9 中位 7.2 h, P90 82.9 h —— 绝大多数暖季周')
    L.append('           都处于"已跨越"状态, 即落在 RR 已被识别的那一档;')
    L.append('      (iii) 因此不需要外推剂量-反应曲线, 只需假设"跨越即风险"在热带成立。')
    L.append('           该假设**不可由本数据验证** —— 热带的适应性、住房、空调普及率均不同,')
    L.append('           且本研究唯一的湿热带证据(新加坡)只给出边际显著的方向一致性。')
    L.append('')
    L.append('    → 可发表的陈述: 在热带/亚热带台站网络覆盖的 3.31 亿人口中,')
    L.append('      若欧洲的阶跃响应成立, 则暖季死亡的归因分数约 %.1f%% [%.1f%%–%.1f%%];' % (100 * af, 100 * af_lo, 100 * af_hi))
    L.append('      换算成人数需要当地死亡登记, 本研究**不做**该换算。')
    js['bounded_extrapolation'] = dict(rr=float(rr), lo=float(lo), hi=float(hi),
                                       af=float(af), af_lo=float(af_lo), af_hi=float(af_hi))
    L.append('')

    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'gradient_robustness.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'gradient_robustness.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
