# -*- coding: utf-8 -*-
"""第8轮 R8-2 关键检验: 湿球的增量价值是否集中在"热端"?

背景: 全样本与干湿分层均显示 —— 控制夜间最低干球 TN 后, 夜间湿球 Tw 不再独立显著。
      但这不等于湿球无用。生理学与热浪流行病学的经典假说是:
      **蒸发散热在高温下才成为限速环节**, 因此湿度的增量价值应集中在高温区。
      若在全温度域平均, 该效应会被大量凉爽周稀释(且夏季周多为舒适温度)。

本脚本做三个递进检验:
  A. 温度分位分层: 每区域把暖季周按 TN 分十层, 逐层拟合 Tw vs TN 头对头
  B. 热端效应修饰: 只取最热十分位周, 再按夜间"干湿亏缺"(TN-Tw)分两组,
     比较同 TN 下湿组的 TN 斜率是否更大
  C. 热端阈值: 最热十分位周内, 夜间最低湿球超 21.9/24.0°C 的夜数是否带来额外风险
产出: health_heattail.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta, log

BASE = os.path.dirname(os.path.abspath(__file__))


def design_pars(df, exposures):
    """简约设计: 常数 + 线性年趋势 + Fourier(周)四季 + 暴露项
    分层内每区域仅 ~55 周, 不能用年份虚拟变量(24个)"""
    y = df['mon'].map(lambda d: d.year).values.astype(float)
    woy = df['mon'].map(lambda d: d.isocalendar()[1]).values.astype(float)
    n = len(df)
    X = [np.ones(n), (y - y.mean()) / max(y.std(), 1e-9)]
    names = ['const', 'trend']
    for k in range(1, 4):
        X.append(np.cos(2 * math.pi * k * woy / 52.0)); names.append('cos%d' % k)
        X.append(np.sin(2 * math.pi * k * woy / 52.0)); names.append('sin%d' % k)
    for e in exposures:
        X.append(df[e].values.astype(float)); names.append(e)
    return np.column_stack(X), names


def fit_region_strat(g, exposures, min_n=40, min_deaths=100):
    """分层内样本更少, 放宽门槛 + 简约设计"""
    X, names = design_pars(g, exposures)
    y = g['deaths'].values.astype(float)
    if len(g) < min_n or y.sum() < min_deaths or X.shape[0] < X.shape[1] + 8:
        return None
    b, se, phi = poisson_glm(X, y)
    if not np.all(np.isfinite(se)):
        return None
    out = {'nuts2': g['nuts2'].iloc[0], 'n': len(g), 'deaths': float(y.sum())}
    for e in exposures:
        i = names.index(e)
        out[e] = float(b[i]); out[e + '_se'] = float(se[i])
    return out


def pooled(m_sub, exposures, label, min_n=40, quiet=False):
    """多区域随机效应合并"""
    rr = [r for _, g in m_sub.groupby('nuts2')
          for r in [fit_region_strat(g, exposures, min_n)] if r]
    parts = []
    rec = {'n_regions': len(rr), 'n_weeks': int(len(m_sub))}
    for e in exposures:
        mm = meta(rr, e)
        if not mm:
            continue
        z = mm['re'] / mm['re_se'] if mm['re_se'] > 0 else float('nan')
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        parts.append('%s RR=%.4f [%.4f,%.4f] p=%.3g'
                     % (e, math.exp(mm['re']), math.exp(mm['re'] - 1.96 * mm['re_se']),
                        math.exp(mm['re'] + 1.96 * mm['re_se']), p))
        rec[e] = dict(rr=float(math.exp(mm['re'])), beta=float(mm['re']),
                      se=float(mm['re_se']), p=float(p), I2=mm['I2'],
                      n_regions=mm['n_regions'])
    if not quiet:
        log('%-28s n=%d周 %d区域 | %s' % (label, len(m_sub), len(rr), '   '.join(parts)))
    return rec


def pooled_interaction(df, tag):
    """合并准泊松: 区域固定效应 + 年趋势 + Fourier + TN异常 + TN异常×湿夜
    返回 干夜TN斜率 与 湿夜的额外增量"""
    d = df.dropna(subset=['tn_a', 'humid', 'deaths']).copy()
    regs = sorted(d['nuts2'].unique())
    if len(regs) < 10 or len(d) < 500:
        return None
    n = len(d)
    y = d['deaths'].values.astype(float)
    yr = d['mon'].map(lambda x: x.year).values.astype(float)
    woy = d['mon'].map(lambda x: x.isocalendar()[1]).values.astype(float)
    cols = [np.ones(n), (yr - yr.mean()) / max(yr.std(), 1e-9)]
    names = ['const', 'trend']
    for k in range(1, 4):
        cols.append(np.cos(2 * math.pi * k * woy / 52.0)); names.append('cos%d' % k)
        cols.append(np.sin(2 * math.pi * k * woy / 52.0)); names.append('sin%d' % k)
    for r in regs[1:]:
        cols.append((d['nuts2'] == r).values.astype(float)); names.append('R_' + r)
    cols.append(d['tn_a'].values.astype(float)); names.append('tn_a')
    cols.append((d['tn_a'] * d['humid']).values.astype(float)); names.append('tn_x_humid')
    X = np.column_stack(cols)
    b, se, phi = poisson_glm(X, y, maxiter=80)
    i1 = names.index('tn_a'); i2 = names.index('tn_x_humid')
    b1, s1 = b[i1], se[i1]
    b2, s2 = b[i2], se[i2]
    z = b2 / s2 if s2 > 0 else float('nan')
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return dict(n=int(n), n_regions=len(regs),
                rr_dry=float(math.exp(b1)), rr_dry_lo=float(math.exp(b1 - 1.96 * s1)),
                rr_dry_hi=float(math.exp(b1 + 1.96 * s1)),
                rr_int=float(math.exp(b2)), rr_int_lo=float(math.exp(b2 - 1.96 * s2)),
                rr_int_hi=float(math.exp(b2 + 1.96 * s2)),
                beta_int=float(b2), se_int=float(s2), p_int=float(p), phi=float(phi))


def main():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    m = m[m['mon'].map(lambda d: d.year) < 2020]
    m['dec'] = m.groupby('nuts2')['tn_hot'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
    m['deficit'] = m['tn_hot'] - m['tw_hot']
    # 湿度异常: deficit 对 TN 做区域内回归后的残差(负值=该温度下异常湿)
    # 直接用 deficit 会被"越热越干"污染(热周里 TN 高的往往 deficit 也大)
    m['def_res'] = np.nan
    for nid, gg in m.groupby('nuts2'):
        x = gg['tn_hot'].values.astype(float)
        yv = gg['deficit'].values.astype(float)
        if len(x) < 50 or np.std(x) < 1e-9:
            m.loc[gg.index, 'def_res'] = yv - np.mean(yv)
            continue
        A = np.column_stack([np.ones(len(x)), x])
        try:
            c = np.linalg.lstsq(A, yv, rcond=None)[0]
            m.loc[gg.index, 'def_res'] = yv - A @ c
        except Exception:
            m.loc[gg.index, 'def_res'] = yv - np.mean(yv)
    m['humid'] = (m['def_res'] <= 0).astype(int)
    m['tn_a'] = m['tn_hot'] - m.groupby('nuts2')['tn_hot'].transform('mean')
    log('样本: %d 周, %d 区域, 年份 %d-%d'
        % (len(m), m['nuts2'].nunique(),
           m['mon'].map(lambda d: d.year).min(), m['mon'].map(lambda d: d.year).max()))

    lines = ['夜间湿球的增量价值是否集中在热端 (欧洲 NUTS2, 暖季周, 剔除2020后)',
             '样本 %d 周 / %d 区域' % (len(m), m['nuts2'].nunique()), '']
    out = {'n_weeks': int(len(m)), 'n_regions': int(m['nuts2'].nunique())}

    # ---------- A. 按 TN 十分位分层 ----------
    # 主口径: 配对夜 —— 该周最热一夜的(TN, Tw) 取自同一夜
    m['dec'] = m.groupby('nuts2')['tn_hot'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
    lines.append('=== A. 按区域"最热一夜"的干球(TN)十分位分层: 逐层头对头 (每 1°C) ===')
    lines.append('    [配对] TN 与 Tw 取自同一夜(该周最热夜);  [周最大] 各自取周内最大, 可能来自不同夜')
    out['decile'] = {}
    out['decile_unmatched'] = {}
    for d in range(10):
        sub = m[m['dec'] == d]
        tnmed = float(np.median(sub['tn_hot']))
        lo, hi = np.percentile(sub['tn_hot'], [10, 90])
        rec = pooled(sub, ['tw_hot', 'tn_hot'], 'D%d' % d)
        rec_u = pooled(sub, ['tw_night_min_max', 'tn_max'], 'D%d-u' % d, quiet=True)
        out['decile'][str(d)] = dict(tn_median=tnmed, tn_p10=float(lo), tn_p90=float(hi), **rec)
        out['decile_unmatched'][str(d)] = rec_u
        tw = rec.get('tw_hot', {}); tn = rec.get('tn_hot', {})
        twu = rec_u.get('tw_night_min_max', {})
        lines.append('  D%-2d 中位TN=%.1f∈[%.1f,%.1f] | 配对Tw RR=%.4f p=%-9.3g | 配对TN RR=%.4f p=%-9.3g | 周最大Tw RR=%.4f p=%.3g'
                     % (d, tnmed, lo, hi, tw.get('rr', float('nan')), tw.get('p', float('nan')),
                        tn.get('rr', float('nan')), tn.get('p', float('nan')),
                        twu.get('rr', float('nan')), twu.get('p', float('nan'))))
    lines.append('')

    # ---------- B. 热端效应修饰 (合并模型 + 区域固定效应 + 交互) ----------
    hot = m[m['dec'] == 9].copy()
    lines.append('=== B. 最热十分位周: 同 TN 下"湿夜"是否放大 TN 的风险? (合并模型) ===')
    out['hot_tail'] = {'n_weeks': int(len(hot))}
    for sub, tag in [(hot, 'hot'), (m, 'all')]:
        r = pooled_interaction(sub, tag)
        out['hot_tail'][tag] = r
        if r:
            lines.append('  [%s] n=%d周 干夜 TN斜率 RR=%.4f [%.4f,%.4f];  湿夜增量 RR=%.4f [%.4f,%.4f] p=%.3g'
                         % (tag, r['n'], r['rr_dry'], r['rr_dry_lo'], r['rr_dry_hi'],
                            r['rr_int'], r['rr_int_lo'], r['rr_int_hi'], r['p_int']))
    lines.append('  (湿夜 = 该温度下"干湿亏缺 TN−Tw"低于区域回归预期值者, 已剔除"越热越干"混淆)')
    lines.append('')

    # ---------- C. 热端阈值型暴露 ----------
    lines.append('=== C. 最热十分位周内: 夜间超阈夜数的额外风险 (控制 TN) ===')
    out['hot_threshold'] = {}
    for col, nm in [('n_tw219', '夜数 Tw_min>21.9°C'), ('n_tw240', '夜数 Tw_min>24.0°C'),
                    ('nh219', '累积强度 Σ(Tw−21.9)+ ·°C')]:
        if col not in hot.columns:
            continue
        rec = pooled(hot, ['tn_hot', col], nm)
        out['hot_threshold'][col] = rec
        c = rec.get(col, {})
        lines.append('  %-22s 控制TN后 RR=%.4f [%.4f,%.4f] p=%.3g'
                     % (nm, c.get('rr', float('nan')),
                        math.exp(c.get('beta', 0) - 1.96 * c.get('se', 0)),
                        math.exp(c.get('beta', 0) + 1.96 * c.get('se', 0)),
                        c.get('p', float('nan'))))
    # 最热十分位 vs 其余 的差异
    cold = m[m['dec'] <= 7]
    lines.append('=== D. 冷端对照 (D0-D7) ===')
    rec = pooled(cold, ['tw_hot', 'tn_hot'], 'D0-D7')
    out['cold'] = rec
    twc = rec.get('tw_hot', {}); tnc = rec.get('tn_hot', {})
    lines.append('  冷端 80%% 的暖季周: Tw RR=%.4f p=%.3g | TN RR=%.4f p=%.3g'
                 % (twc.get('rr', float('nan')), twc.get('p', float('nan')),
                    tnc.get('rr', float('nan')), tnc.get('p', float('nan'))))
    lines.append('')
    # 热端 vs 冷端的 Tw 效应对比(正式差值检验)
    try:
        h = out['decile']['9']['tw_hot']
        c0 = out['cold']['tw_hot']
        dl = h['beta'] - c0['beta']
        sd = math.sqrt(h['se'] ** 2 + c0['se'] ** 2)
        z = dl / sd if sd > 0 else float('nan')
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        lines.append('=== E. 关键差值检验: 最热十分位 vs 冷端(D0-D7) 的湿球独立效应 ===')
        lines.append('  热端 Tw 斜率 = %+.5f/°C, 冷端 = %+.5f/°C, 差 = %+.5f ± %.5f (z=%+.2f, p=%.3g)'
                     % (h['beta'], c0['beta'], dl, sd, z, p))
        out['diff_hot_vs_cold'] = dict(beta_hot=float(h['beta']), beta_cold=float(c0['beta']),
                                       delta=float(dl), se=float(sd), z=float(z), p=float(p))
    except Exception as e:
        lines.append('  差值检验失败: %s' % e)

    txt = '\n'.join(lines)
    print()
    print(txt)
    with open('health_heattail.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('health_heattail.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
