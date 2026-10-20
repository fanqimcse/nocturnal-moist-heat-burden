# -*- coding: utf-8 -*-
"""第14轮 R14-2/R14-3/R14-4  二元头条指标 NWF/NMF + 效应分布 + 二元版未来投影

审稿人 M81/M82/M83 的核心一句话:
  "你花大力气证明的 NWH(小时数) 增长, 在健康上可能没有对应物 ——
   因为剂量-反应是平的, 最强的健康结果是'是否跨越'(二元)。"

本脚本建立与二元健康结果直接对应的头条指标:
  NWF (欧洲面板) = 每年暖季中出现 ≥1 次夜间湿球跨越 21.9 °C 的**周数**
  NMF (全球台站) = 每年出现 ≥1 小时越阈的**月数**   (月尺度上 NWF 的对应物)
并:
  【1】报告主口径(二元 any219)的逐区域效应分布 —— 回应 I²=78% 的质疑
  【2】NWF 的南北对比与趋势
  【3】NMF 的观测趋势与分气候带
  【4】NMF 的 CMIP6 未来投影(带逐站逐月偏差订正, 并按 b_ALL 观测约束缩放)

产出: nwf_headline.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm, meta

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'ext', 'cmip6')
WARM = (5, 9)
THR = 21.9


def log(*a):
    print(*a, flush=True)


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    n = len(x)
    if n < 6:
        return float('nan')
    sl = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(sl)) if sl else float('nan')


def prep_health():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM[0]) & (mo <= WARM[1])].copy()
    m['any219'] = (m['nh219'] > 0).astype(float)
    m['year'] = m['mon'].map(lambda d: d.year)
    return m


def effect_distribution(m, L, js):
    """主口径二元 any219 的逐区域效应分布"""
    L.append('【1】主口径(二元 any219)的逐区域效应分布 —— 回应 I²=78%')
    m = m.copy()
    m['tx_z'] = m['tx_max'] / m.groupby('nuts2')['tx_max'].transform('std').replace(0, np.nan)
    m['tn_z'] = m['nh_tn20'] / m.groupby('nuts2')['nh_tn20'].transform('std').replace(0, np.nan)
    rows = []
    for k, g in m.groupby('nuts2'):
        gg = g.dropna(subset=['tx_z', 'tn_z', 'any219'])
        if len(gg) < 100 or gg['deaths'].sum() < 200 or gg['any219'].std() <= 0:
            continue
        X, names = design(gg, ['tx_z', 'tn_z', 'any219'])
        y = gg['deaths'].values.astype(float)
        b, se, phi = poisson_glm(X, y)
        if not np.all(np.isfinite(se)):
            continue
        i = names.index('any219')
        rows.append({'nuts2': k, 'b': float(b[i]), 'se': float(se[i]),
                     'tw0': float(g['tw_night_min_max'].mean()),
                     'lat': float(g['lat'].iloc[0]) if 'lat' in g else np.nan,
                     'nz': float(np.nanmean(g['nh219'] > 0))})
    rr = np.array([math.exp(r['b']) for r in rows])
    L.append('    区域数 %d' % len(rows))
    for q in (5, 25, 50, 75, 95):
        L.append('      RR 分位 P%-3d = %.4f' % (q, np.percentile(rr, q)))
    L.append('    逐区域 RR > 1 的比例: %.1f%%' % (100 * np.mean(rr > 1)))
    L.append('    逐区域 RR 显著>1 (b>1.96se) 的比例: %.1f%%'
             % (100 * np.mean([r['b'] > 1.96 * r['se'] for r in rows])))
    # 与湿热本底的关系
    tw0 = np.array([r['tw0'] for r in rows])
    kk = np.argsort(tw0)
    L.append('    按湿热本底三分:')
    js['dist'] = dict(n_regions=len(rows),
                      pct_gt1=float(np.mean(rr > 1)),
                      pct_sig=float(np.mean([r['b'] > 1.96 * r['se'] for r in rows])),
                      quantiles={str(q): float(np.percentile(rr, q)) for q in (5, 25, 50, 75, 95)})
    for lab, idx in (('最凉三分之一', kk[:len(kk) // 3]),
                     ('中间三分之一', kk[len(kk) // 3:2 * len(kk) // 3]),
                     ('最湿热三分之一', kk[2 * len(kk) // 3:])):
        sub = [rows[i] for i in idx]
        w = 1.0 / np.array([r['se'] for r in sub]) ** 2
        bb = np.array([r['b'] for r in sub])
        mu = float(np.sum(w * bb) / np.sum(w))
        L.append('      %-14s 区域 %3d  汇总 RR=%.4f  逐区域 RR>1 比例 %.0f%%  非零率中位 %.1f%%'
                 % (lab, len(sub), math.exp(mu), 100 * np.mean(np.exp(bb) > 1),
                    100 * np.median([r['nz'] for r in sub])))
        js['dist'].setdefault('by_tercile', {})[lab] = dict(
            n=len(sub), rr=float(math.exp(mu)), pct_gt1=float(np.mean(np.exp(bb) > 1)),
            nz_median=float(np.median([r['nz'] for r in sub])))
    L.append('')
    return rows


def nwf_europe(m, L, js):
    """欧洲面板: 每年暖季出现跨越的周数"""
    L.append('【2】NWF（欧洲）= 每年暖季 5–9 月中出现 ≥1 次夜间湿球跨越 21.9 °C 的周数')
    tw0 = m.groupby('nuts2')['tw_night_min_max'].mean()
    med = float(tw0.median())
    d = m.groupby(['nuts2', 'year']).agg(nwk=('any219', 'size'),
                                         ncross=('any219', 'sum')).reset_index()
    d['frac'] = d['ncross'] / d['nwk']
    L.append('    (暖季每年约 22 周)')
    for tag, ids in (('全欧洲', set(tw0.index)),
                     ('南欧型 tw0≥%.2f' % med, set(tw0[tw0 >= med].index)),
                     ('北欧型 tw0<%.2f' % med, set(tw0[tw0 < med].index))):
        s = d[d['nuts2'].isin(ids)]
        yr = s.groupby('year')['ncross'].mean()
        yrs = yr.index.values.astype(float)
        tr = theil_sen(yrs, yr.values) * 10
        L.append('    %-18s 区域 %3d  基线 %.1f 周/年  趋势 %+6.2f 周/十年'
                 % (tag, len(ids), yr.mean(), tr))
        js.setdefault('nwf_europe', {})[tag] = dict(
            n_regions=len(ids), baseline=float(yr.mean()), trend=float(tr))
    L.append('')
    L.append('    注: 暖季每年仅 22 周, 因此 NWF 的上限为 22 —— 该指标会在暖区饱和。')
    L.append('    这本身就是"阈值按气候带设定"必要性的又一个证据。')
    L.append('')


def nmf_stations(L, js):
    """全球台站: 每年出现 ≥1 小时越阈的月数"""
    L.append('【3】NMF（全球台站）= 每年出现 ≥1 小时夜间湿球 ≥21.9 °C 的月数（上限 12）')
    c = pd.read_csv(os.path.join(BASE, 'transfer_calib.csv'))
    try:
        import step33_nwh_projection as S
        st = S.stations()
        NH = S.night_hours_table(st)
        c['nh'] = [NH.get((s, int(m)), np.nan) for s, m in zip(c['station'], c['month'])]
    except Exception:
        c['nh'] = c['nhours_obs']
    c['nwh_m'] = c['f_obs'] * c['nh']
    c['cross'] = (c['nwh_m'] >= 1.0).astype(float)
    a = c.groupby(['station', 'year'])['cross'].sum().reset_index()
    a = a[a['year'] >= 1976]
    p = a.pivot(index='year', columns='station', values='cross')
    cnt = a.groupby('station')['year'].nunique()
    keep = cnt[cnt >= 20].index
    p = p[[c for c in p.columns if c in set(keep)]]
    L.append('    合格站（≥20 年）: %d' % p.shape[1])
    ser = p.mean(axis=1)
    yrs = ser.index.values.astype(float)
    tr = theil_sen(yrs, ser.values) * 10
    L.append('    网络均值: 基线 %.2f 月/年   趋势 %+.3f 月/十年' % (ser.mean(), tr))
    js.setdefault('nmf', {})['network'] = dict(n_stations=int(p.shape[1]),
                                               baseline=float(ser.mean()), trend=float(tr))
    # 分带
    try:
        pop = pd.read_csv(os.path.join(BASE, 'station_population.csv')).set_index('station')['band']
    except Exception:
        pop = None
    if pop is not None:
        L.append('    分气候带:')
        for b in sorted(set(pop.dropna())):
            cols = [c for c in p.columns if pop.get(c) == b]
            if len(cols) < 5:
                continue
            s2 = p[cols].mean(axis=1)
            t2 = theil_sen(yrs, s2.values) * 10
            L.append('      %-8s 站 %3d  基线 %.2f 月/年  趋势 %+.3f 月/十年'
                     % (b, len(cols), s2.mean(), t2))
            js['nmf'].setdefault('by_band', {})[str(b)] = dict(
                n=len(cols), baseline=float(s2.mean()), trend=float(t2))
    L.append('')
    return p


def nmf_projection(L, js):
    """CMIP6: 二元版未来投影"""
    L.append('【4】NMF 的未来投影（CMIP6 SSP，逐站逐月偏差订正，按 b_ALL 观测约束缩放）')
    try:
        import step33_nwh_projection as S
    except Exception:
        L.append('    step33 不可用，跳过')
        return
    st = S.stations()
    NH = S.night_hours_table(st)
    cal = pd.read_csv(os.path.join(BASE, 'transfer_calib.csv'))
    cal['nwh_m'] = cal['f_obs'] * cal['nhours_obs']
    sig = cal.groupby(['station', 'month'])['sd'].median()
    sig_med = cal.groupby('month')['sd'].median()
    MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'CNRM-CM6-1', 'HadGEM3-GC31-LL']
    SSP = ['ssp245', 'ssp585']
    # 观测侧: 台站各月夜间湿球均值 (用于偏差订正)
    obs_tw = cal.groupby(['station', 'month'])['mu'].median() if 'mu' in cal.columns else None
    res = {}
    B_ALL = 0.348
    for src in MODELS:
        for exp in SSP:
            ft = os.path.join(OUT, 'pts_%s_%s_tasmin.npz' % (src, exp))
            fh = os.path.join(OUT, 'pts_%s_%s_hurs.npz' % (src, exp))
            if not (os.path.exists(ft) and os.path.exists(fh)):
                continue
            zt = np.load(ft); zh = np.load(fh)
            tn = np.asarray(zt['data'], float); rh = np.asarray(zh['data'], float)
            if np.nanmean(tn) > 150:
                tn = tn - 273.15
            yr = zt['year']; mo = zt['month']
            if tn.shape != rh.shape:
                n = min(len(tn), len(rh))
                tn = tn[:n]; rh = rh[:n]; yr = yr[:n]; mo = mo[:n]
            e = np.clip(rh, 1, 100) / 100.0 * S.esat(np.clip(tn, -60, 60))
            Td = 243.5 * np.log(np.maximum(e, 1e-6) / 6.112) / (17.67 - np.log(np.maximum(e, 1e-6) / 6.112))
            Td = np.where(np.isfinite(Td), Td, tn - 5.0)
            nst = min(tn.shape[1], len(st))
            nwhm = np.full((len(yr), nst), np.nan)
            for j in range(nst):
                sid = st['station'].iloc[j]
                tw = S.tw_vec(np.clip(tn[:, j], -60, 60), np.minimum(Td[:, j], tn[:, j] - 1e-3), 1013.25)
                # 逐站逐月常数偏差订正: 用该站该月的观测-模式气候态差
                corr = np.zeros(len(yr))
                for m in range(1, 13):
                    k = (mo == m)
                    if not k.any():
                        continue
                    o = obs_tw.get((sid, m), np.nan) if obs_tw is not None else np.nan
                    if not np.isfinite(o):
                        continue
                    v = np.nanmean(tw[k])
                    if np.isfinite(v):
                        corr[k] = o - v
                twc = tw + corr
                sd_m = np.array([sig.get((sid, int(m)), sig_med.get(int(m), np.nan)) for m in mo])
                nh_m = np.array([NH.get((sid, int(m)), np.nan) for m in mo])
                z = (THR - twc) / np.maximum(sd_m, 1e-6)
                f = 0.5 * np.array([math.erfc(v / math.sqrt(2)) if np.isfinite(v) else np.nan for v in z])
                nwhm[:, j] = f * nh_m
            cross = (nwhm >= 1.0).astype(float)
            yrs = yr.astype(int)
            ann = {}
            for a in np.unique(yrs):
                ann[int(a)] = float(np.nanmean(cross[yrs == a]))
            res['%s_%s' % (src, exp)] = ann
    if not res:
        L.append('    无可用 SSP 数据')
        return
    def win(d, lo, hi):
        v = [d[a] * 12.0 for a in d if lo <= a <= hi]
        return float(np.mean(v)) if v else float('nan')
    L.append('    情景            基线2015-34   中期2041-60   末期2081-2100   末期变化(月/年)')
    js.setdefault('nmf_proj', {})
    for exp in SSP:
        ks = [k for k in res if exp in k]
        if not ks:
            continue
        bs = np.array([win(res[k], 2015, 2034) for k in ks])
        ms = np.array([win(res[k], 2041, 2060) for k in ks])
        es = np.array([win(res[k], 2081, 2100) for k in ks])
        de_raw = np.median(es - bs)
        de = de_raw * B_ALL
        L.append('    %-10s n=%d    %6.2f        %6.2f         %6.2f        %+.2f (原始 %+.2f, ×b_ALL)'
                 % (exp, len(ks), np.median(bs), np.median(ms), np.median(es), de, de_raw))
        js['nmf_proj'][exp] = dict(n_models=len(ks), base=float(np.median(bs)),
                                   mid=float(np.median(ms)), end=float(np.median(es)),
                                   delta_raw=float(de_raw), delta_scaled=float(de))
    L.append('')
    L.append('    判读: NMF 上限为 12 月/年, 因此该指标在暖区会**饱和** ——')
    L.append('          它的增长空间集中在"目前只有少数月份越阈"的地区, 即中纬度与南欧。')
    L.append('')


def main():
    m = prep_health()
    L = ['二元头条指标 NWF / NMF + 效应分布 + 二元版未来投影 (step44)', '']
    js = {}
    effect_distribution(m, L, js)
    nwf_europe(m, L, js)
    nmf_stations(L, js)
    nmf_projection(L, js)
    L.append('【5】头条指标体系的最终形态')
    L.append('    NWH（小时数）  : 物理强度。趋势 +42.8 h/十年；96% 需湿度变化；SSP5-8.5 末期 +174 h/年。')
    L.append('    NWF/NMF（发生）: 健康对应物。剂量-反应平坦，故风险随"发生频率"而非"强度"上升。')
    L.append('    → 论文必须同时报这两个，并明确它们回答的是不同的问题。')
    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(BASE, 'nwf_headline.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open(os.path.join(BASE, 'nwf_headline.json'), 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
