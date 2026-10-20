# -*- coding: utf-8 -*-
"""
step35: 在头条指标 NWH 本身上做正式检测-归因 (D&A)
==================================================
第 8 轮的 D&A 是做在"月尺度夜间湿球"上的, 而论文的头条量是 NWH(超阈小时数)。
二者非线性相关, 不能默认等价。本脚本:

  1. 用第 10 轮标定的传递函数 (月 μ,σ → 超阈比例) 把 CMIP6 月尺度场
     (tasmin + hurs) 换算成逐站逐年的 NWH, 并对 historical / hist-nat /
     piControl / ssp 全部实验施加同一套逐站逐月常数偏差订正
     (修正: 第 10 轮算了 delta 却从未应用)。
  2. 验证: 模式 historical 能否复现观测到的 NWH 趋势。
  3. 双信号最优指纹: 观测 39 年(1976-2014)趋势向量(热带, 亚热带)
     = b_ALL * ALL 趋势 + b_NAT * NAT 趋势, 噪声协方差取自 piControl
     非重叠 39 年窗的趋势分布。
  4. S/N: 观测趋势相对 piControl 内部变率。
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import step33_nwh_projection as S

THR = S.THR
OUT = 'ext/cmip6'
MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'CNRM-CM6-1']
PI_MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'MRI-ESM2-0']
HIST_MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'CNRM-CM6-1', 'HadGEM3-GC31-LL']
Y0, Y1 = 1976, 2014
WIN = Y1 - Y0 + 1


def erfc_vec(z):
    from scipy.special import erfc
    return erfc(z)


def obs_annual():
    """观测: 由 transfer_calib.csv 重建逐站逐年 NWH。
    **关键修正**: 不能用 calib 表里的 nhours_obs 做缩放 —— 那是当年"实际被采样到的
    夜间小时数", 随 GHCNh 观测频次从 3 小时一次提高到 1 小时一次而单调增长,
    会让年 NWH 从 796 涨到 1600 h/yr 而站数恒定(纯覆盖漂移, 非物理)。
    改用与模式侧完全一致的**天文夜间小时数** NH(站,月) 做常数缩放。"""
    import step33_nwh_projection as S
    st = S.stations()
    NH = S.night_hours_table(st)
    c = pd.read_csv('transfer_calib.csv')
    c['nh'] = [NH.get((s, int(m)), np.nan) for s, m in zip(c['station'], c['month'])]
    c['nwh'] = c['f_obs'] * c['nh']
    a = c.groupby(['station', 'year'])['nwh'].sum().reset_index()
    return a.pivot(index='year', columns='station', values='nwh')


def model_monthly_tw(src, exp):
    ft = os.path.join(OUT, 'pts_%s_%s_tasmin.npz' % (src, exp))
    fh = os.path.join(OUT, 'pts_%s_%s_hurs.npz' % (src, exp))
    if not (os.path.exists(ft) and os.path.exists(fh)):
        return None
    zt = np.load(ft); zh = np.load(fh)
    tn = np.asarray(zt['data'], float); rh = np.asarray(zh['data'], float)
    if np.nanmean(tn) > 150:
        tn = tn - 273.15
    yr = np.asarray(zt['year'], int); mo = np.asarray(zt['month'], int)
    n = min(len(tn), len(rh))
    tn, rh, yr, mo = tn[:n], rh[:n], yr[:n], mo[:n]
    e = np.clip(rh, 1.0, 100.0) / 100.0 * S.esat(np.clip(tn, -60, 60))
    with np.errstate(divide='ignore', invalid='ignore'):
        Td = 243.5 * np.log(np.maximum(e, 1e-6) / 6.112) / (17.67 - np.log(np.maximum(e, 1e-6) / 6.112))
    Td = np.where(np.isfinite(Td), Td, tn - 5.0)
    tw = S.tw_vec(np.clip(tn, -60, 60), np.minimum(Td, tn - 1e-3), 1013.25)
    return tw, yr, mo


def build():
    st = S.stations()
    cal = pd.read_csv('transfer_calib.csv')
    sig = cal.groupby(['station', 'month'])['sd'].median()
    sig_med = cal.groupby('month')['sd'].median()
    obs_mu = cal.groupby(['station', 'month'])['mu'].mean()
    NH = S.night_hours_table(st)
    nst = len(st)
    sids = st['station'].tolist()
    # ---- 偏差订正 delta: 用各模式自身的 historical 期 ----
    delta = {}
    dinfo = []
    for src in MODELS + HIST_MODELS:
        src = src
        if src in [m for m in dict.fromkeys(MODELS + HIST_MODELS)]:
            pass
    for src in dict.fromkeys(HIST_MODELS + MODELS):
        r = model_monthly_tw(src, 'historical')
        if r is None:
            continue
        tw, yr, mo = r
        for j in range(min(tw.shape[1], nst)):
            sid = sids[j]
            for m in range(1, 13):
                s = mo == m
                v = float(np.nanmean(tw[s, j]))
                o = obs_mu.get((sid, m), np.nan)
                if np.isfinite(o) and np.isfinite(v):
                    delta[(src, sid, m)] = float(o - v)
                    dinfo.append(o - v)
    dinfo = np.array(dinfo)
    print('偏差订正 delta: n=%d 中位 %+.2f °C  四分位 [%+.2f, %+.2f]'
          % (len(dinfo), np.median(dinfo), np.percentile(dinfo, 25), np.percentile(dinfo, 75)),
          flush=True)

    out = {}
    exps = [('historical', HIST_MODELS), ('hist-nat', HIST_MODELS)]
    for src in MODELS:
        for e in ('ssp245', 'ssp585'):
            exps.append((e, [src]))
    for src in PI_MODELS:
        exps.append(('piControl', [src]))
    done = set()
    for exp, mods in exps:
        for src in mods:
            key = (src, exp)
            if key in done:
                continue
            done.add(key)
            r = model_monthly_tw(src, exp)
            if r is None:
                continue
            tw, yr, mo = r
            nw = np.full(tw.shape, np.nan)
            for j in range(min(tw.shape[1], nst)):
                sid = sids[j]
                dvec = np.array([delta.get((src, sid, int(m)), 0.0) for m in mo])
                twc = tw[:, j] + dvec
                sdv = np.array([sig.get((sid, int(m)), sig_med.get(int(m), np.nan)) for m in mo])
                nhv = np.array([NH.get((sid, int(m)), np.nan) for m in mo])
                z = (THR - twc) / np.maximum(sdv, 1e-6)
                f = 0.5 * erfc_vec(z / math.sqrt(2.0))
                nw[:, j] = f * nhv * 12.0
            yrs = np.unique(yr)
            ann = np.full((len(yrs), nst), np.nan)
            for i, a in enumerate(yrs):
                s = yr == a
                ann[i] = np.nanmean(nw[s], axis=0)
            out['%s|%s' % (src, exp)] = dict(years=yrs.tolist(),
                                             ann=np.round(ann, 2).tolist())
            print('  built %s %s  %d-%d' % (src, exp, yrs.min(), yrs.max()), flush=True)
    json.dump(dict(stations=sids, delta_n=int(len(dinfo)),
                   delta_med=float(np.median(dinfo)),
                   data=out), open('cmip6_nwh_series.json', 'w'), ensure_ascii=False)
    print('saved cmip6_nwh_series.json')


def load():
    d = json.load(open('cmip6_nwh_series.json'))
    st = d['stations']
    out = {}
    for k, v in d['data'].items():
        src, exp = k.split('|')
        out[(src, exp)] = (np.array(v['years']), np.array(v['ann'], float))
    return st, out


def region_masks(stations, obs_cols):
    pop = pd.read_csv('station_population.csv')[['station', 'band']].set_index('station')['band']
    idx = {s: i for i, s in enumerate(stations)}
    have = [s for s in obs_cols if s in idx]
    reg = {}
    for b in ('tropics', 'subtropics'):
        sel = [s for s in have if pop.get(s, '') == b]
        reg[b] = np.array([idx[s] for s in sel], int), sel
    return reg, have


def station_trends(mat, years, cols):
    """逐站 Theil-Sen 趋势 (h/10yr)。"""
    out = []
    for j in cols:
        v = mat[:, j]
        k = np.isfinite(v)
        if k.sum() < 15:
            continue
        out.append(theil_sen(years[k], v[k]) * 10)
    return np.array(out, float)


def region_stat(mat, years, cols, minfrac=0.90):
    """区域统计量 = 固定(平衡)站集上的区域逐年均值, 再取 Theil-Sen 趋势。

    为什么不能用"逐站趋势的中位数":
      NWH 是阈值计数, 约一半站点常年为 0, 逐站趋势大量并列在 0,
      中位被并列值锁死 —— 实测亚热带中位 +0.77 而均值 +37.9。
    为什么不能用"逐年对全部可用站取平均":
      台站构成随年代变化会把构成漂移误当趋势。
    → 先筛出窗口内覆盖率 ≥90% 的站, 再对该固定站集取逐年均值。"""
    sub = mat[:, cols]
    nyr = len(years)
    frac = np.isfinite(sub).sum(axis=0) / nyr
    keep = cols[frac >= minfrac]
    if len(keep) < 5:
        return np.nan
    s = sub[:, frac >= minfrac]
    ser = np.array([np.nanmean(s[i]) if np.isfinite(s[i]).any() else np.nan
                    for i in range(nyr)])
    k = np.isfinite(ser)
    if k.sum() < 15:
        return np.nan
    return float(theil_sen(years[k], ser[k]) * 10)


def region_nstation(mat, years, cols, minfrac=0.90):
    sub = mat[:, cols]
    frac = np.isfinite(sub).sum(axis=0) / len(years)
    return int((frac >= minfrac).sum())


def regional_series(mat, years, cols, ref_years=None):
    """(保留, 用于绘图) 区域逐年异常。ref_years 给定时以该窗口为参考期。"""
    sub = mat[:, cols]
    if ref_years is None:
        m0 = (years >= Y0) & (years <= Y1)
    else:
        m0 = (years >= ref_years[0]) & (years <= ref_years[1])
    ref = np.nanmean(sub[m0], axis=0)
    an = sub - ref
    out = np.full(len(years), np.nan)
    for i in range(len(years)):
        v = an[i]
        k = np.isfinite(v)
        out[i] = np.mean(v[k]) if k.any() else np.nan
    return out


def theil_sen(x, y):
    from step30_humidity_attribution import theil_sen as ts
    return ts(np.asarray(x, float), np.asarray(y, float))


def trend(years, ser):
    k = np.isfinite(ser)
    if k.sum() < 15:
        return np.nan
    return theil_sen(years[k], ser[k]) * 10


def analyse():
    st, data = load()
    OA = obs_annual()
    OA = OA.loc[(OA.index >= Y0) & (OA.index <= Y1)]      # 窗口必须与模式一致
    obs_cols = [c for c in OA.columns]
    reg, have = region_masks(st, obs_cols)
    L = ['NWH₂₁.₉ 检测与归因 (step35)',
         '传递函数: 月(μ,σ)→超阈比例, 偏差订正 n=%d 个点, 中位 %+.2f °C' %
         (json.load(open('cmip6_nwh_series.json'))['delta_n'],
          json.load(open('cmip6_nwh_series.json'))['delta_med']),
         '窗口 %d-%d (%d 年); 区域 = 热带 / 亚热带 (观测可用站 %d)' % (Y0, Y1, WIN, len(have)),
         '']
    js = {}

    # ---------- 1. 观测 vs 模式 historical ----------
    L.append('【1】模式能否复现观测到的 NWH 趋势 (h/10yr)')
    obs_tr = {}
    obs_cols_arr = {s: i for i, s in enumerate(obs_cols)}
    for b, (cols, sel) in reg.items():
        oc = np.array([obs_cols_arr[s] for s in sel], int)
        obs_tr[b] = region_stat(OA.values, OA.index.values.astype(int), oc)
    L.append('  观测       热带 %+7.2f   亚热带 %+7.2f' % (obs_tr['tropics'], obs_tr['subtropics']))
    js['obs_trend'] = {k: float(v) for k, v in obs_tr.items()}
    all_tr = {}; nat_tr = {}
    for b in reg:
        all_tr[b] = []; nat_tr[b] = []
    for src in HIST_MODELS:
        if (src, 'historical') not in data:
            continue
        yrs, mat = data[(src, 'historical')]
        tt = {}
        for b, (cols, sel) in reg.items():
            tt[b] = region_stat(mat, yrs, cols)
        L.append('  %-18s 热带 %+7.2f   亚热带 %+7.2f' % (src, tt['tropics'], tt['subtropics']))
        for b in reg:
            all_tr[b].append(tt[b])
    L.append('  多模式中位  热带 %+7.2f   亚热带 %+7.2f'
             % (np.nanmedian(all_tr['tropics']), np.nanmedian(all_tr['subtropics'])))
    js['model_hist_trend'] = {b: [float(v) for v in all_tr[b]] for b in reg}
    js['model_hist_median'] = {b: float(np.nanmedian(all_tr[b])) for b in reg}

    # ---------- 2. piControl 内部变率 ----------
    L.append('')
    L.append('【2】内部变率 (piControl 非重叠 %d 年窗趋势分布)' % WIN)
    seg = []
    for src in PI_MODELS:
        if (src, 'piControl') not in data:
            continue
        yrs, mat = data[(src, 'piControl')]
        n = len(yrs) // WIN
        for k in range(n):
            y = yrs[k * WIN:(k + 1) * WIN]
            mm = mat[k * WIN:(k + 1) * WIN]
            v = []
            okk = True
            for b in ('tropics', 'subtropics'):
                cols = reg[b][0]
                t = region_stat(mm, y, cols)   # 窗内逐站趋势的中位
                if not np.isfinite(t):
                    okk = False
                v.append(t)
            if okk:
                seg.append(v)
        L.append('  %-18s %d 段' % (src, n))
    seg = np.array(seg, float)
    Sig = np.cov(seg.T)
    L.append('  段数 n=%d' % len(seg))
    L.append('  趋势分布 热带 SD %.2f   亚热带 SD %.2f   相关 %.3f'
             % (np.sqrt(Sig[0, 0]), np.sqrt(Sig[1, 1]),
                Sig[0, 1] / math.sqrt(Sig[0, 0] * Sig[1, 1])))
    js['picontrol'] = dict(n=int(len(seg)), sd_tropics=float(np.sqrt(Sig[0, 0])),
                           sd_subtropics=float(np.sqrt(Sig[1, 1])),
                           corr=float(Sig[0, 1] / math.sqrt(Sig[0, 0] * Sig[1, 1])),
                           cov=Sig.tolist())

    y = np.array([obs_tr['tropics'], obs_tr['subtropics']])
    sn = y / np.sqrt(np.diag(Sig))
    L.append('  观测趋势 S/N: 热带 %.2f  亚热带 %.2f' % (sn[0], sn[1]))
    js['obs_SN'] = dict(tropics=float(sn[0]), subtropics=float(sn[1]))

    # ---------- 3. 双信号指纹 ----------
    L.append('')
    L.append('【3】双信号最优指纹 (热带, 亚热带) × (ALL, NAT)')
    XA = []; XN = []
    for src in HIST_MODELS:
        if (src, 'historical') not in data or (src, 'hist-nat') not in data:
            continue
        ya, ma = data[(src, 'historical')]
        yn, mn = data[(src, 'hist-nat')]
        va = []; vn = []
        for b in ('tropics', 'subtropics'):
            cols = reg[b][0]
            va.append(region_stat(ma, ya, cols))
            vn.append(region_stat(mn, yn, cols))
        XA.append(va); XN.append(vn)
    XA = np.array(XA, float); XN = np.array(XN, float)
    XAm = np.nanmedian(XA, axis=0); XNm = np.nanmedian(XN, axis=0)
    L.append('  ALL 指纹 (多模式中位): 热带 %+.2f  亚热带 %+.2f' % (XAm[0], XAm[1]))
    L.append('  NAT 指纹 (多模式中位): 热带 %+.2f  亚热带 %+.2f' % (XNm[0], XNm[1]))

    def gls(X, yv, Sig):
        Si = np.linalg.inv(Sig)
        A = X.T @ Si @ X
        bb = np.linalg.solve(A, X.T @ Si @ yv)
        cov = np.linalg.inv(A)
        return bb, cov

    # 单信号 ALL
    for nm, Xv in (('ALL', XAm), ('NAT', XNm)):
        bb, cov = gls(Xv.reshape(2, 1), y, Sig)
        se = math.sqrt(cov[0, 0])
        L.append('  单信号 %s: 缩放因子 b=%+.3f ± %.3f (90%%CI [%+.3f, %+.3f])  t=%.2f'
                 % (nm, bb[0], se, bb[0] - 1.645 * se, bb[0] + 1.645 * se, bb[0] / se))
        js.setdefault('single', {})[nm] = dict(b=float(bb[0]), se=float(se), t=float(bb[0] / se))
    # 双信号
    X = np.column_stack([XAm, XNm])
    bb, cov = gls(X, y, Sig)
    L.append('  双信号: b_ALL=%+.3f ± %.3f (t=%.2f)   b_NAT=%+.3f ± %.3f (t=%.2f)'
             % (bb[0], math.sqrt(cov[0, 0]), bb[0] / math.sqrt(cov[0, 0]),
                bb[1], math.sqrt(cov[1, 1]), bb[1] / math.sqrt(cov[1, 1])))
    js['two_signal'] = dict(b_ALL=float(bb[0]), se_ALL=float(math.sqrt(cov[0, 0])),
                            b_NAT=float(bb[1]), se_NAT=float(math.sqrt(cov[1, 1])),
                            cov=cov.tolist(),
                            ci_ALL=[float(bb[0] - 1.645 * math.sqrt(cov[0, 0])),
                                    float(bb[0] + 1.645 * math.sqrt(cov[0, 0]))],
                            ci_NAT=[float(bb[1] - 1.645 * math.sqrt(cov[1, 1])),
                                    float(bb[1] + 1.645 * math.sqrt(cov[1, 1]))])
    # 可归因增量的比例
    if np.isfinite(bb[0]) and np.isfinite(bb[1]):
        L.append('  → 由 ALL 解释的观测趋势份额: %.0f%%' % (100 * bb[0]))
    txt = '\n'.join(L)
    print(txt)
    open('nwh_detection_attribution.txt', 'w', encoding='utf-8').write(txt + '\n')
    json.dump(js, open('nwh_detection_attribution.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'build':
        build()
    elif cmd == 'analyse':
        analyse()
    else:
        build(); analyse()
