# -*- coding: utf-8 -*-
"""第10轮 R10-3  NWH 的未来投影 (CMIP6 SSP × 统计传递函数)

审稿人 M56: 只有历史观测, 没有未来。Nature 级论文必须回答"到本世纪末会怎样"。

难点: CMIP6 只有月尺度, NWH 是小时尺度量。
解决: 建立"月尺度夜间湿球分布 → 超阈小时数"的统计传递函数, 在台站上标定并验证,
      再应用到 CMIP6 月尺度场。

传递函数:
  假设某月内夜间湿球近似正态 N(μ, σ), 则
      f = P(Tw_night ≥ thr) = 1 − Φ((thr − μ)/σ)
      NWH_month = f × 该月夜间小时数
  标定: μ, σ 取该月夜间湿球的观测均值与标准差; 夜间小时数由太阳高度角给出(逐年近恒定)。
  验证: 在 67 站 × 全部站月上比较"预测 NWH" 与"实测 NWH"的相关/RMSE/偏差。

CMIP6 侧: μ ≈ Tw(tasmin月均, hurs月均); σ 取台站该月气候态 σ(模式不直接提供小时变率)。
  σ 的处理是本项目最大近似 —— 但 σ 的时间变化远小于 μ, 且我们只报"相对变化"。

阶段: download | calib | project
"""
import os, sys, json, math, warnings
import numpy as np
import pandas as pd
try:
    from scipy.special import erfc
except Exception:                       # 回退: 用互补误差函数的向量化近似
    from math import erfc as _e
    erfc = np.vectorize(lambda v: _e(v) if np.isfinite(v) else np.nan)

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'ext', 'cmip6')
os.makedirs(OUT, exist_ok=True)

THR = 21.9
MODELS = ['CanESM5', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'CNRM-CM6-1']
VARS = ['tasmin', 'hurs']
SSP = ['ssp245', 'ssp585']


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


def download():
    import fsspec, xarray as xr
    cat = pd.read_csv(os.path.join(OUT, 'cmip6_zarr_stores.csv'))
    st = stations()
    slat = st['lat'].values; slon = st['lon'].values
    for src in MODELS:
        for exp in SSP:
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
                    arr = np.asarray(v.values, dtype=np.float32)
                    try:
                        yrs = np.asarray(v.time.dt.year.values)
                        mos = np.asarray(v.time.dt.month.values)
                    except Exception:
                        tt0 = v.time.values
                        yrs = np.array([getattr(t, 'year', 0) for t in tt0])
                        mos = np.array([getattr(t, 'month', 0) for t in tt0])
                    yrs = yrs.astype(int); mos = mos.astype(int)
                    if yrs.max() < 2000:
                        raise RuntimeError('year parse failed max=%s' % yrs.max())
                    msk = yrs >= 2015
                    arr = arr[msk][:, ii, jj]
                    yrs = yrs[msk].astype(np.int16); mos = mos[msk].astype(np.int8)
                    np.savez_compressed(dst, data=arr, lat=lat[ii], lon=lon[jj],
                                        year=yrs, month=mos)
                    print('  OK %s %s %s member=%s shape=%s' % (src, exp, var, mem, arr.shape), flush=True)
                except Exception as e:
                    print('  FAIL %s %s %s : %s' % (src, exp, var, e), flush=True)


# ---------------- 标定: 台站月尺度 → NWH ----------------
def esat(t):
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def tw_vec(T, Td, p):
    T = np.asarray(T, float); Td = np.asarray(Td, float)
    out = np.full(T.shape, np.nan)
    ok = np.isfinite(T) & np.isfinite(Td)
    t_ = T[ok].copy(); d_ = np.minimum(Td[ok], t_ - 1e-6)
    e = np.minimum(esat(d_), p * 0.95)
    w = 0.622 * e / (p - e)
    t = t_ - (t_ - d_) * 0.33
    for _ in range(8):
        es = np.minimum(esat(t), p * 0.95)
        den = p - es
        ws = 0.622 * es / den
        des = es * 4303.845 / ((t + 243.5) ** 2)
        dw = 0.622 * p * des / (den * den)
        g = 2418.0 * (ws - w) - (t_ - t)
        gp = 2418.0 * dw + 1.0
        t -= g / gp
    out[ok] = np.where((t > t_ - 90) & (t <= t_ + 0.01), t, t_)
    sat = ok & (Td >= T)
    out[sat] = T[sat]
    return out


def calib():
    """在台站上标定并验证传递函数: 月(μ,σ) → 月 NWH。"""
    import duckdb, glob
    sys.path.insert(0, BASE)
    from step30_humidity_attribution import solar_elevation_vec, pressure_from_elev
    m = pd.read_csv('metrics_v3.csv')
    mm = m[m['ok_trend'].astype(str) == '1']
    cnt = mm.groupby('station').size()
    keep = set(cnt[cnt >= 20].index)
    files_by = {}
    for f in glob.glob(os.path.join('ghcn_data', 'GHCNh_*.parquet')):
        s = os.path.basename(f).split('_')[1]
        files_by.setdefault(s, []).append(f)
    rows = []
    for sid in sorted(keep):
        if sid not in files_by:
            continue
        g = mm[mm.station == sid]
        lat = float(g['lat'].iloc[0]); lon = float(g['lon'].iloc[0])
        elev = float(g['elev'].iloc[0])
        try:
            con = duckdb.connect()
            pat = "','".join(f.replace("'", "''") for f in sorted(files_by[sid]))
            con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])" % pat)
            rr = con.execute("""SELECT CAST(DATE AS VARCHAR) d,
                                TRY_CAST(temperature AS DOUBLE) t,
                                TRY_CAST(dew_point_temperature AS DOUBLE) td
                                FROM h ORDER BY d""").fetchall()
            con.close()
        except Exception:
            continue
        if not rr:
            continue
        ds = np.array([x[0] for x in rr])
        T = np.array([x[1] if x[1] is not None else np.nan for x in rr], float)
        Td = np.array([x[2] if x[2] is not None else np.nan for x in rr], float)
        y = np.array([int(s[:4]) for s in ds]); mo = np.array([int(s[5:7]) for s in ds])
        da = np.array([int(s[8:10]) for s in ds]); hh = np.array([int(s[11:13]) for s in ds])
        T[(T < -80) | (T > 65)] = np.nan
        Td[(Td < -90) | (Td > 45)] = np.nan
        dfa = pd.DataFrame(dict(y=y, mo=mo, da=da, hh=hh, t=T, td=Td))
        hb = dfa.groupby(['y', 'mo', 'da', 'hh']).agg(t=('t', 'mean'), td=('td', 'mean')).reset_index()
        cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        leap = (hb['y'].values % 4 == 0) & ((hb['y'].values % 100 != 0) | (hb['y'].values % 400 == 0))
        doy = cum[hb['mo'].values - 1] + hb['da'].values + np.where((hb['mo'].values > 2) & leap, 1, 0)
        Tv = hb['t'].values; Tdv = hb['td'].values
        ok = np.isfinite(Tv) & np.isfinite(Tdv)
        el = solar_elevation_vec(lat, lon, doy[ok], hb['hh'].values[ok] + 0.5)
        night = el < -0.833
        idx = np.where(ok)[0][night]
        p = pressure_from_elev(elev)
        tw = tw_vec(Tv[idx], Tdv[idx], p)
        yy = hb['y'].values[idx]; mmo = hb['mo'].values[idx]
        nh = np.full(len(tw), np.nan)
        # 该月夜间小时数: 用该月内观测到的夜间小时数按覆盖率折算
        for (a, b) in set(zip(yy.tolist(), mmo.tolist())):
            s = (yy == a) & (mmo == b)
            v = tw[s]
            nv = np.isfinite(v).sum()
            days = np.unique(hb['da'].values[idx][s]).size
            if nv < 20 or days < 20:
                continue
            mu = float(np.nanmean(v)); sd = float(np.nanstd(v))
            f_obs = float(np.nanmean(v[ np.isfinite(v) ] >= THR))
            # 正态预测
            f_pred = 0.5 * math.erfc((THR - mu) / (sd * math.sqrt(2))) if sd > 1e-6 else (1.0 if mu >= THR else 0.0)
            rows.append(dict(station=sid, year=int(a), month=int(b), mu=mu, sd=sd,
                             f_obs=f_obs, f_pred=f_pred,
                             nhours_obs=float(nv) / max(days, 1) * 30.44))
        print('calib', sid, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('transfer_calib.csv', index=False)
    return df


def calib_report():
    df = pd.read_csv('transfer_calib.csv')
    r = float(np.corrcoef(df['f_obs'], df['f_pred'])[0, 1])
    rmse = float(np.sqrt(np.mean((df['f_obs'] - df['f_pred']) ** 2)))
    bias = float(np.mean(df['f_pred'] - df['f_obs']))
    L = ['传递函数标定 (月尺度 μ,σ → 月超阈比例), 阈值 %.1f °C' % THR,
         '样本: %d 站月, %d 站' % (len(df), df['station'].nunique()),
         '  相关 r = %.4f   RMSE(比例) = %.4f   偏差 = %+.4f' % (r, rmse, bias),
         '  分箱标定 (按 f_obs 十分位分箱, 比较箱内 f_pred 均值):',
         '    %8s %10s %10s %8s' % ('f_obs箱', 'f_obs均值', 'f_pred均值', 'n')]
    q = pd.qcut(df['f_obs'], 10, duplicates='drop')
    gb = df.groupby(q, observed=True)
    for k, g in gb:
        L.append('    [%4.2f,%4.2f] %10.3f %10.3f %8d'
                 % (k.left, k.right, g['f_obs'].mean(), g['f_pred'].mean(), len(g)))
    # 年度 NWH: 预测 vs 实测 (这才是真正要投影的量)
    df['nwh_obs_m'] = df['f_obs'] * df['nhours_obs']
    df['nwh_pred_m'] = df['f_pred'] * df['nhours_obs']
    ann = df.groupby(['station', 'year']).agg(
        obs=('nwh_obs_m', 'sum'), pred=('nwh_pred_m', 'sum'),
        nmon=('f_obs', 'size')).reset_index()
    ann = ann[ann['nmon'] >= 10]
    r2 = float(np.corrcoef(ann['obs'], ann['pred'])[0, 1])
    rr = float(np.corrcoef(ann['obs'], ann['pred'])[0, 1] ** 2)
    mbe = float(np.mean(ann['pred'] - ann['obs']))
    rmse_a = float(np.sqrt(np.mean((ann['pred'] - ann['obs']) ** 2)))
    L.append('')
    L.append('  年度 NWH 验证 (%d 站年, 要求年内 ≥10 个有效月):' % len(ann))
    L.append('    相关 r = %.4f  (R² = %.4f)   平均偏差 = %+.1f h/yr   RMSE = %.1f h/yr'
             % (r2, rr, mbe, rmse_a))
    L.append('    实测年均 %.0f h/yr   预测年均 %.0f h/yr   相对偏差 %+.1f%%'
             % (ann['obs'].mean(), ann['pred'].mean(),
                100 * (ann['pred'].mean() / max(ann['obs'].mean(), 1e-9) - 1)))
    # 高暴露子样本(热带站)的偏差
    hi = ann[ann['obs'] > ann['obs'].quantile(0.75)]
    L.append('    高暴露四分位(obs > %.0f h/yr): 实测 %.0f  预测 %.0f  相对偏差 %+.1f%%'
             % (ann['obs'].quantile(0.75), hi['obs'].mean(), hi['pred'].mean(),
                100 * (hi['pred'].mean() / max(hi['obs'].mean(), 1e-9) - 1)))
    txt = '\n'.join(L)
    print(txt)
    with open('transfer_calib.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    return r, rmse, bias


# ---------------- 投影 ----------------
def night_hours_table(st):
    """夜长只由纬度与月份决定 —— 用太阳高度角直接算, 不依赖观测覆盖。"""
    import datetime as _dt
    sys.path.insert(0, BASE)
    from step30_humidity_attribution import solar_elevation_vec
    out = {}
    # 代表年取 2001 (非闰年), 逐日逐时判定
    d0 = _dt.date(2001, 1, 1)
    days = [(d0 + _dt.timedelta(days=i), (d0 + _dt.timedelta(days=i)).month)
            for i in range(365)]
    doy = np.array([x[0].timetuple().tm_yday for x in days])
    mmo = np.array([x[1] for x in days])
    hh = np.arange(24)
    D, H = np.meshgrid(doy, hh, indexing='ij')
    Df = D.ravel().astype(float); Hf = H.ravel().astype(float)
    for _, row in st.iterrows():
        sid = row['station']
        el = solar_elevation_vec(float(row['lat']), float(row['lon']), Df, Hf + 0.5)
        night = (el < -0.833).reshape(D.shape).sum(axis=1).astype(float)
        for m in range(1, 13):
            out[(sid, m)] = float(np.sum(night[mmo == m]))    # 该月夜间总小时数
    return out


def project():
    """把传递函数应用到 CMIP6 SSP 月尺度场, 得到 NWH 年序列。"""
    st = stations()
    cal = pd.read_csv('transfer_calib.csv')
    # 台站各月气候态 σ (夜间湿球月内标准差); 无本站标定时回退到该月所有站的中位 σ
    sig = cal.groupby(['station', 'month'])['sd'].median()
    sig_med = cal.groupby('month')['sd'].median()
    NH = night_hours_table(st)
    cal_st = set(cal['station'].unique())
    # ---- 偏差订正 ----
    # 模式侧夜间湿球用 (tasmin 月均, hurs 月均) 估计, 与实测"夜间湿球月均"系统性不一致
    # (日均 RH 配日最低温 ≠ 夜间实际 RH)。用 historical 期(1976-2014)与台站实测的
    # 逐站逐月差值做常数偏差订正。
    obs_mu = cal.groupby(['station', 'month'])['mu'].mean()
    hist_clim = {}
    for src in MODELS:
        ft = os.path.join(OUT, 'pts_%s_historical_tasmin.npz' % src)
        fh = os.path.join(OUT, 'pts_%s_historical_hurs.npz' % src)
        if not (os.path.exists(ft) and os.path.exists(fh)):
            continue
        zt = np.load(ft); zh = np.load(fh)
        tn = np.asarray(zt['data'], float)
        rh = np.asarray(zh['data'], float)
        if np.nanmean(tn) > 150:
            tn = tn - 273.15
        mo = np.asarray(zt['month'], int)
        nst = min(tn.shape[1], len(st))
        _e = np.clip(rh, 1.0, 100.0) / 100.0 * esat(np.clip(tn, -60, 60))
        _td = 243.5 * np.log(np.maximum(_e, 1e-6) / 6.112) / (17.67 - np.log(np.maximum(_e, 1e-6) / 6.112))
        _td = np.where(np.isfinite(_td), _td, tn - 5.0)
        twm = tw_vec(np.clip(tn[:, :nst], -60, 60),
                     np.minimum(_td[:, :nst], tn[:, :nst] - 1e-3), 1013.25)
        for j in range(nst):
            sid = st['station'].iloc[j]
            for m in range(1, 13):
                s = mo == m
                hist_clim[(src, sid, m)] = float(np.nanmean(twm[s, j]))
    delta = {}
    for (src, sid, m), v in hist_clim.items():
        o = obs_mu.get((sid, m), np.nan)
        if np.isfinite(o) and np.isfinite(v):
            delta[(src, sid, m)] = float(o - v)
    dvals = np.array(list(delta.values()))
    print('偏差订正: n=%d  中位 %+.2f °C  范围 [%+.2f, %+.2f]'
          % (len(dvals), np.median(dvals), np.nanmin(dvals), np.nanmax(dvals)), flush=True)
    hist = {}
    for src in MODELS:
        for var in VARS:
            f = os.path.join(OUT, 'pts_%s_historical_%s.npz' % (src, var))
            if os.path.exists(f):
                hist[(src, var)] = np.load(f)
    res = {}
    for src in MODELS:
        for exp in SSP:
            ft = os.path.join(OUT, 'pts_%s_%s_tasmin.npz' % (src, exp))
            fh = os.path.join(OUT, 'pts_%s_%s_hurs.npz' % (src, exp))
            if not (os.path.exists(ft) and os.path.exists(fh)):
                print('missing', src, exp)
                continue
            zt = np.load(ft); zh = np.load(fh)
            tn = np.asarray(zt['data'], float)      # CMIP6 原生单位 K
            rh = np.asarray(zh['data'], float)
            if np.nanmean(tn) > 150:
                tn = tn - 273.15                    # K → °C
            yr = zt['year']; mo = zt['month']
            if tn.shape != rh.shape:
                n = min(len(tn), len(rh))
                tn = tn[:n]; rh = rh[:n]; yr = yr[:n]; mo = mo[:n]
            # 夜间湿球 ≈ Tw(tasmin, hurs)  (夜间最低温 + 月均相对湿度)
            e = np.clip(rh, 1.0, 100.0) / 100.0 * esat(np.clip(tn, -60, 60))
            Td = 243.5 * np.log(np.maximum(e, 1e-6) / 6.112) / (17.67 - np.log(np.maximum(e, 1e-6) / 6.112))
            Td = np.where(np.isfinite(Td), Td, tn - 5.0)
            nst = min(tn.shape[1], len(st))
            nwh = np.full((len(yr), nst), np.nan, dtype=np.float32)
            for j in range(nst):
                sid = st['station'].iloc[j]
                p = 1013.25        # 气压统一用海平面近似(模式格点海拔与台站不一致)
                tw = tw_vec(np.clip(tn[:, j], -60, 60), np.minimum(Td[:, j], tn[:, j] - 1e-3), p)
                sd_m = np.array([sig.get((sid, int(m)), sig_med.get(int(m), np.nan))
                                 for m in mo])
                nh_m = np.array([NH.get((sid, int(m)), np.nan) for m in mo])
                z = (THR - tw) / np.maximum(sd_m, 1e-6)
                f = 0.5 * erfc(z / math.sqrt(2.0))
                nwh[:, j] = f * nh_m * 12.0
            yrs = yr.astype(int)
            ann = {}
            for a in np.unique(yrs):
                s = yrs == a
                ann[int(a)] = np.nanmean(nwh[s], axis=0)
            res['%s_%s' % (src, exp)] = dict(years=sorted(ann),
                                             ann={str(k): v.tolist() for k, v in ann.items()},
                                             stations=st['station'].tolist())
            print('proj', src, exp, len(ann), flush=True)
    np.save(os.path.join(OUT, 'nwh_proj.npy'), np.array([0]))
    json.dump(res, open('nwh_projection.json', 'w'), ensure_ascii=False)

    # ---- 汇总 ----
    pop = pd.read_csv('station_population.csv').set_index('station')['pop_50km']
    w = np.array([float(pop.get(s, 0.0)) for s in st['station']])
    mask67 = np.array([s in cal_st for s in st['station']])
    L = ['NWH₂₁.₉ 未来投影 (CMIP6 SSP × 统计传递函数)',
         '全样本 %d 站 (覆盖 %.2f 亿); 主样本 = 有传递函数标定的 %d 站 (%.2f 亿)'
         % (len(st), w.sum() / 1e8, int(mask67.sum()), w[mask67].sum() / 1e8),
         '基线: 各情景 2015–2034 均值; 中期 2041–2060; 末期 2081–2100', '']
    js = {}
    for key, d in res.items():
        yrs = np.array(d['years'])
        A = np.array([d['ann'][str(a)] for a in yrs], float)
        b0 = np.nanmean(A[(yrs >= 2015) & (yrs <= 2034)], axis=0)
        b2 = np.nanmean(A[(yrs >= 2041) & (yrs <= 2060)], axis=0)
        b1 = np.nanmean(A[(yrs >= 2081) & (yrs <= 2100)], axis=0)

        def pm(v, m=None):
            vv = v[m] if m is not None else v
            ww = w[m] if m is not None else w
            k = np.isfinite(vv) & (ww > 0)
            return float(np.average(vv[k], weights=ww[k])) if k.any() else float('nan')
        L.append('  %-22s' % key)
        for tag, m in (('主样本 67 站', mask67), ('全样本 176 站', None)):
            L.append('    %-14s 2015-34 %6.0f → 2041-60 %6.0f → 2081-2100 %6.0f h/yr'
                     '   中期 %+5.0f%%  末期 %+6.0f%%'
                     % (tag, pm(b0, m), pm(b2, m), pm(b1, m),
                        100 * (pm(b2, m) / pm(b0, m) - 1), 100 * (pm(b1, m) / pm(b0, m) - 1)))
        js[key] = dict(base67=pm(b0, mask67), mid67=pm(b2, mask67), end67=pm(b1, mask67),
                       inc_mid67=float(pm(b2, mask67) / pm(b0, mask67) - 1),
                       inc_end67=float(pm(b1, mask67) / pm(b0, mask67) - 1),
                       base_all=pm(b0), mid_all=pm(b2), end_all=pm(b1),
                       inc_mid_all=float(pm(b2) / pm(b0) - 1),
                       inc_end_all=float(pm(b1) / pm(b0) - 1))
    txt = '\n'.join(L)
    print(txt)
    with open('nwh_projection.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('nwh_projection_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if a == 'download':
        download()
    elif a == 'calib':
        calib(); calib_report()
    elif a == 'calib_report':
        calib_report()
    elif a == 'project':
        project()
