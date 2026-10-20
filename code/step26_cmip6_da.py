# -*- coding: utf-8 -*-
"""第8轮 R8-5: CMIP6 正式检出与归因 (D&A)

月尺度 (Amon) tasmin + hurs -> 模式同口径"夜间湿球" -> 与 GHCNh 观测双信号最优指纹
阶段: download / obs / da
"""
import os, sys, json, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'ext', 'cmip6')
os.makedirs(OUT, exist_ok=True)

MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'CNRM-CM6-1', 'HadGEM3-GC31-LL', 'MRI-ESM2-0']
VARS = ['tasmin', 'hurs']
EXPS = ['historical', 'hist-nat', 'hist-GHG', 'hist-aer']
T0, T1 = '1976-01-01', '2014-12-31'


def stations():
    m = pd.read_csv('metrics_v3.csv', usecols=['station', 'lat', 'lon', 'ok_trend'])
    m = m[m['ok_trend'].astype(str) == '1']
    g = m.groupby('station').first().reset_index()
    return g[['station', 'lat', 'lon']].reset_index(drop=True)


def pick(cat, src, exp, var):
    r = cat[(cat.table_id == 'Amon') & (cat.source_id == src) &
            (cat.experiment_id == exp) & (cat.variable_id == var)]
    if not len(r):
        return None, None
    def mkey(m):
        try:
            a = int(m[1:].split('i')[0]); b = int(m.split('i')[1].split('p')[0])
            c = int(m.split('p')[1].split('f')[0])
        except Exception:
            a, b, c = 999, 999, 999
        return (a, b, c)
    r = r.assign(_k=r.member_id.map(mkey)).sort_values('_k')
    return r.zstore.iloc[0], r.member_id.iloc[0]


def download(srcs, exps):
    import fsspec, xarray as xr
    cat = pd.read_csv(os.path.join(OUT, 'cmip6_zarr_stores.csv'))
    st = stations()
    slat = st['lat'].values; slon = st['lon'].values
    meta = {'stations': st['station'].tolist(), 'lat': slat.tolist(), 'lon': slon.tolist()}
    json.dump(meta, open(os.path.join(OUT, 'stations.json'), 'w'))
    for src in srcs:
        for exp in exps:
            for var in VARS:
                dst = os.path.join(OUT, 'pts_%s_%s_%s.npz' % (src, exp, var))
                if os.path.exists(dst):
                    print('  skip', dst, flush=True); continue
                url, mem = pick(cat, src, exp, var)
                if url is None:
                    print('  MISSING', src, exp, var, flush=True); continue
                try:
                    mp = fsspec.get_mapper(url)
                    ds = xr.open_zarr(mp, consolidated=True)
                    v = ds[var]
                    lat = ds['lat'].values; lon = ds['lon'].values
                    ii = np.abs(lat[:, None] - slat[None, :]).argmin(axis=0)
                    jj = np.abs(lon[:, None] - slon[None, :]).argmin(axis=0)
                    arr = np.asarray(v.values, dtype=np.float32)   # 触发下载 (T, lat, lon)
                    try:
                        yrs = np.asarray(v.time.dt.year.values)
                        mos = np.asarray(v.time.dt.month.values)
                    except Exception:
                        tt0 = v.time.values
                        yrs = np.array([getattr(t, 'year', 0) for t in tt0])
                        mos = np.array([getattr(t, 'month', 0) for t in tt0])
                    yrs = yrs.astype(int); mos = mos.astype(int)
                    if exp == 'piControl':
                        msk = np.ones(len(yrs), bool)          # 控制试验: 全时段(最多前 12000 月)
                        if len(yrs) > 12000:
                            msk = np.zeros(len(yrs), bool); msk[:12000] = True
                    else:
                        if yrs.max() < 1900:
                            raise RuntimeError('year parse failed: max=%s' % yrs.max())
                        msk = (yrs >= 1976) & (yrs <= 2014)
                    arr = arr[msk][:, ii, jj]
                    yrs = yrs[msk].astype(np.int16)
                    mos = mos[msk].astype(np.int8)
                    np.savez_compressed(dst, data=arr, lat=lat[ii], lon=lon[jj],
                                        year=yrs, month=mos)
                    print('  OK %s %s %s member=%s shape=%s' % (src, exp, var, mem, arr.shape), flush=True)
                except Exception as e:
                    print('  ERR %s %s %s: %s %s' % (src, exp, var, type(e).__name__, str(e)[:150]), flush=True)
    print('download stage done')


def wetbulb_monthly(tasmin_k, hurs_pct):
    """模式同口径: 月 tasmin + 月 hurs -> 湿球 (Stull 2011, 与本研究湿球交叉验证过)."""
    T = np.asarray(tasmin_k, dtype=float) - 273.15
    rh = np.clip(np.asarray(hurs_pct, dtype=float), 1.0, 100.0)
    return ((T * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
             + np.arctan(T + rh) - np.arctan(rh - 1.676331)
             + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
             - 4.686035))


def seasonal_index(year, month, data, summer_north=(6, 7, 8)):
    """北半球夏季(JJA)/南半球(DJF) 平均 -> 逐年序列。"""
    dfy = pd.DataFrame({'y': year, 'm': month, 'v': data})
    north = dfy['m'].isin(summer_north)
    south = dfy['m'].isin([12, 1, 2])
    is_north = None
    return None


def _warm_months(obs):
    """每站气候态最热 3 个月 (由观测月均温确定, 观测与模式共用)"""
    g = obs.groupby(['station', 'month'])['t_mean'].mean().reset_index()
    sel = {}
    for sid, gg in g.groupby('station'):
        sel[sid] = list(gg.sort_values('t_mean', ascending=False).month[:3])
    return sel


def _warm_series(year, month, data, months=None):
    """逐年均值序列 -> (years, values); months=None 表示全年"""
    df = pd.DataFrame({'y': year, 'm': month, 'v': data})
    if months:
        df = df[df['m'].isin(months)]
    s = df.groupby('y')['v'].mean()
    return s.index.values.astype(int), s.values.astype(float)


def theil_sen(x, y):
    n = len(x)
    sl = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(sl)) if sl else float('nan')


def load_model(src, exp):
    p1 = os.path.join(OUT, 'pts_%s_%s_tasmin.npz' % (src, exp))
    p2 = os.path.join(OUT, 'pts_%s_%s_hurs.npz' % (src, exp))
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return None
    a = np.load(p1); b = np.load(p2)
    tm = a['data']; hr = b['data']
    tw = wetbulb_monthly(tm, hr)
    stj = json.load(open(os.path.join(OUT, 'stations.json')))
    return dict(year=a['year'], month=a['month'], tw=tw, tasmin=tm - 273.15,
                lat=a['lat'], lon=a['lon'], st=stj['stations'])


def da():
    obs = pd.read_csv('obs_monthly.csv')
    warm = _warm_months(obs)
    stlist = sorted(warm.keys())
    # 两套口径分别计算 (全年 / 暖季)
    srcs = [d.split('_')[1] for d in os.listdir(OUT) if d.startswith('pts_') and d.endswith('tasmin.npz')]
    srcs = sorted(set(srcs))
    lines = ['CMIP6 检出与归因 · 夜间湿球趋势 (°C/每10年), 1976-2014', '']
    out = {'models': srcs}
    for label, months in [('全年', None), ('暖季', warm)]:
        # 观测趋势
        obs_tr = {}
        for sid in stlist:
            g = obs[obs.station == sid]
            mm = None if months is None else months[sid]
            for var in ('tw_stull', 'tw_night', 'tasmin', 't_mean'):
                yrs, vals = _warm_series(g.year.values, g.month.values, g[var].values, mm)
                if len(yrs) >= 20:
                    obs_tr.setdefault(var, {})[sid] = theil_sen(yrs.astype(float), vals) * 10.0
        # 模式趋势
        sig = {}
        for src in srcs:
            for exp in ('historical', 'hist-nat'):
                M = load_model(src, exp)
                if M is None:
                    continue
                vec = {}
                for k, sid in enumerate(M['st']):
                    mm = None if months is None else warm.get(sid, [6, 7, 8])
                    yrs, vals = _warm_series(M['year'], M['month'], M['tw'][:, k], mm)
                    if len(yrs) >= 20:
                        vec[sid] = theil_sen(yrs.astype(float), vals) * 10.0
                if vec:
                    sig.setdefault(exp, {})[src] = vec
        lines.append('===== %s口径 =====' % label)
        for exp in ('historical', 'hist-nat'):
            if exp not in sig:
                continue
            df = pd.DataFrame(sig[exp])
            common = [s for s in stlist if s in df.index and df.loc[s].notna().all()]
            if not common:
                continue
            mmv = df.loc[common].mean(axis=1)
            lines.append('  %s (%d 模式: %s)' % (exp, df.shape[1], ', '.join(df.columns)))
            lines.append('     多模式平均趋势 中位 %+.4f / 均值 %+.4f °C/10年 (n=%d 站)'
                         % (np.median(mmv), np.mean(mmv), len(common)))
            out['%s_%s' % (label, exp)] = dict(models=list(df.columns), n_stations=len(common),
                                               median_trend=float(np.median(mmv)),
                                               mean_trend=float(np.mean(mmv)))
        for var in ('tw_stull', 'tw_night', 'tasmin', 't_mean'):
            v = pd.Series(obs_tr.get(var, {}))
            if len(v):
                lines.append('  观测 %-9s 中位趋势 %+.4f °C/10年 (n=%d 站)'
                             % (var, np.median(v.values), len(v)))
                out['%s_obs_%s' % (label, var)] = dict(median_trend=float(np.median(v.values)),
                                                       n=len(v), mean_trend=float(np.mean(v.values)))
        if 'historical' in sig and 'hist-nat' in sig and 'tw_stull' in obs_tr:
            da_ = pd.DataFrame(sig['historical']); dn = pd.DataFrame(sig['hist-nat'])
            common = sorted(set(da_.index) & set(dn.index) & set(obs_tr['tw_stull']))
            if len(common) >= 20:
                Y = np.array([obs_tr['tw_stull'][s] for s in common])
                XA = da_.loc[common].mean(axis=1).values
                XN = dn.loc[common].mean(axis=1).values
                X = np.column_stack([XA, XN])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                resid = Y - X @ beta
                s2 = resid @ resid / (len(common) - 2)
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(XtX_inv) * s2)
                lines.append('  双信号最优指纹 Y_obs ~ b_ALL*X_ALL + b_NAT*X_NAT (n=%d 站)' % len(common))
                for nm, b, s in zip(['b_ALL', 'b_NAT'], beta, se):
                    t = b / s if s > 0 else float('nan')
                    lines.append('     %s = %+.3f ± %.3f  [%.3f, %.3f]  t=%+.2f'
                                 % (nm, b, s, b - 1.96 * s, b + 1.96 * s, t))
                for nm, Xs in [('ALL only', XA), ('NAT only', XN)]:
                    bb = float(np.dot(Xs, Y) / np.dot(Xs, Xs))
                    lines.append('     单信号 %s: b = %+.3f' % (nm, bb))
                out['%s_fingerprint' % label] = dict(
                    n=len(common), b_ALL=float(beta[0]), b_ALL_se=float(se[0]),
                    b_NAT=float(beta[1]), b_NAT_se=float(se[1]))
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open('cmip6_da.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('cmip6_da.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


def ivar():
    """piControl 内部变率: 39 年暖季夜间湿球趋势的零分布 -> 正式检出检验"""
    obs = pd.read_csv('obs_monthly.csv')
    warm = _warm_months(obs)
    stlist = sorted(warm.keys())
    obs_tr = {}
    for sid in stlist:
        g = obs[obs.station == sid]
        yrs, vals = _warm_series(g.year.values, g.month.values, g['tw_stull'].values, warm[sid])
        if len(yrs) >= 20:
            obs_tr[sid] = theil_sen(yrs.astype(float), vals) * 10.0
    srcs = sorted({s.split('_')[1] for s in os.listdir(OUT)
                   if s.startswith('pts_') and '_piControl_tasmin' in s})
    if not srcs:
        print('no piControl'); return
    all_lines = []
    all_out = {'models': {}}
    for src in srcs:
        _one_ivar(src, obs_tr, warm, all_lines, all_out)
    txt = '\n'.join(all_lines)
    print(txt)
    open('cmip6_internal_variability.txt', 'w', encoding='utf-8').write(txt + '\n')
    json.dump(all_out, open('cmip6_internal_variability.json', 'w'), ensure_ascii=False, indent=1)
    return


def _one_ivar(src, obs_tr, warm, all_lines, all_out):
    M = load_model(src, 'piControl')
    if M is None:
        return
    tw = M['tw']
    T = tw.shape[0]
    nyr = T // 12
    seg = 39
    stnames = M['st']
    idx = [k for k, s in enumerate(stnames) if s in obs_tr]   # 与观测同站集
    # 逐站暖季年序列
    ser = {}
    for k in idx:
        sid = stnames[k]
        mo = warm.get(sid, [6, 7, 8])
        cols = [m - 1 for m in mo]
        v2 = tw[:nyr * 12, k].reshape(nyr, 12)
        ser[sid] = v2[:, cols].mean(axis=1)
    # (a) 单站趋势分布
    tr = []
    for sid, s in ser.items():
        for st0 in range(0, nyr - seg + 1, seg):
            segv = s[st0:st0 + seg]
            if np.isfinite(segv).all():
                tr.append(theil_sen(np.arange(seg, dtype=float), segv) * 10.0)
    tr = np.array(tr); tr = tr[np.isfinite(tr)]
    # (b) 网络聚合趋势分布 (关键: 保留空间相关)
    net = []
    for st0 in range(0, nyr - seg + 1, seg):
        vals = []
        for sid, s in ser.items():
            segv = s[st0:st0 + seg]
            if np.isfinite(segv).all():
                vals.append(theil_sen(np.arange(seg, dtype=float), segv) * 10.0)
        if vals:
            net.append(np.mean(vals))
    net = np.array(net)
    ov = np.array([obs_tr[s] for s in obs_tr])
    obs_net = float(np.mean(ov))
    lines = []
    lines.append('内部变率检验 (piControl: %s, %d 年 -> %d 个非重叠 39 年段 x %d 站)'
                 % (src, nyr, nyr // seg, len(idx)))
    lines.append('  单站 39 年暖季夜间湿球趋势: 均值 %+.4f, 标准差 %.4f °C/10年 (n=%d)'
                 % (tr.mean(), tr.std(), len(tr)))
    lines.append('    95%% 分位 %+.4f, 99%% 分位 %+.4f' % (np.percentile(tr, 95), np.percentile(tr, 99)))
    p95 = np.percentile(tr, 95)
    k = int(np.sum(ov > p95)); n = len(ov)
    # 二项检验: 期望 5%
    from math import comb
    pval = sum(comb(n, i) * 0.05 ** i * 0.95 ** (n - i) for i in range(k, n + 1))
    lines.append('  观测 (n=%d 站): 中位 %+.4f, 网络均值 %+.4f °C/10年' % (n, np.median(ov), obs_net))
    lines.append('  超过内部变率 95%% 分位的站: %d/%d = %.1f%% (纯内部变率期望 5%%), 二项检验 p = %.3g'
                 % (k, n, 100 * k / n, pval))
    # 网络尺度 S/N (保留空间相关)
    sn_net = abs(obs_net) / net.std() if net.std() > 0 else float('nan')
    lines.append('  网络聚合的内部变率: 均值 %+.4f, 标准差 %.4f °C/10年 (n=%d 段)'
                 % (net.mean(), net.std(), len(net)))
    lines.append('  **网络尺度检测 S/N = %.2f**  (观测网络均值 / 内部变率网络标准差)' % sn_net)
    lines.append('  单站尺度 S/N = %.2f' % (abs(np.median(ov)) / tr.std()))
    ok = (sn_net > 2) and (pval < 0.05)
    lines.append('  结论: %s' % ('观测的夜间湿球上升无法由内部变率解释 -> 检出成立' if ok
                                 else '未达到检出门槛'))
    all_lines.extend(lines)
    all_lines.append('')
    all_out['models'][src] = dict(n_segments=nyr // seg, n_samples=int(len(tr)),
                                  pi_mean=float(tr.mean()), pi_std=float(tr.std()),
                                  p95=float(p95), obs_median=float(np.median(ov)), n_obs=n,
                                  frac_above_p95=float(k / n), binom_p=float(pval),
                                  net_pi_std=float(net.std()), obs_net_mean=obs_net,
                                  sn_network=float(sn_net),
                                  sn_station=float(abs(np.median(ov)) / tr.std()),
                                  detected=bool(ok))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'download'
    if mode == 'download':
        srcs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['CanESM5', 'IPSL-CM6A-LR', 'CNRM-CM6-1']
        exps = sys.argv[3].split(',') if len(sys.argv) > 3 else ['historical', 'hist-nat']
        download(srcs, exps)
    elif mode == 'da':
        da()
    elif mode == 'ivar':
        ivar()
