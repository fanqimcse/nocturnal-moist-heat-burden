# -*- coding: utf-8 -*-
"""
step34: 湿度必要性分解 —— 方法学修正版
=====================================
第 10 轮用的"气候态锁定"反事实有一个致命混淆:
    把 Td 锁到平滑气候态, 同时去掉了湿度的【趋势】和【变率】。
    而去变率本身就会改变越阈频率(阈值附近的分布被压缩),
    因此 B 的"基线偏低 + 趋势偏小"里混进了变率效应,
    "96% 需要湿度"这个数字被夸大。

本脚本做**保变率去趋势**反事实(只去掉长期趋势, 保留全部季节/天气尺度异常),
并同时保留原版以便对照:

    版本1 (V) 保变率:  Td* = Td - bTd*(t-t0)     只去趋势
    版本2 (L) 锁气候态: Td  = climTd              去趋势+去变率  (第10轮口径)

因子分解统一以 D 为参照:
    温度主效 = tr(B) - tr(D)
    湿度主效 = tr(C) - tr(D)
    交互项   = tr(A) - tr(B) - tr(C) + tr(D)
    (D 为"双去趋势/双锁", 因 Tw 非线性, D 的趋势不严格为 0, 必须报告)
"""
import os, sys, math, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from step30_humidity_attribution import (tw_vec, harmonic_clim, pressure_from_elev,
                                         solar_elevation_vec, theil_sen, binom_p)

SUN = -0.833
THS = [20.0, 21.9, 24.0, 25.8]


def lowess_trend(x, y, frac=0.35):
    """局部线性平滑 (高斯核), 用于提取低频趋势。
    关键: 线性去趋势无法处理"加速型增温", 会在反事实里留下虚假的残余趋势
    (本轮实测到 D = -26 h/10yr 的假象)。平滑去趋势可消除该偏差。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    h = max(frac * n, 5.0)
    out = np.full(n, np.nan)
    for i in range(n):
        w = np.exp(-0.5 * ((x - x[i]) / (h / 2.0)) ** 2)
        X = np.column_stack([np.ones(n), x - x[i]])
        try:
            beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
            out[i] = beta[0]
        except Exception:
            out[i] = np.nan
    k = np.isfinite(out)
    if k.sum() >= 3:                      # 端点用线性外推补齐
        out[~k] = np.interp(x[~k], x[k], out[k])
    return out


def station_decomp(station, lat, lon, elev, files):
    import duckdb
    con = duckdb.connect()
    pat = "','".join(f.replace("'", "''") for f in files)
    con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])" % pat)
    try:
        rows = con.execute("""
            SELECT CAST(DATE AS VARCHAR) d,
                   TRY_CAST(temperature AS DOUBLE) t,
                   TRY_CAST(dew_point_temperature AS DOUBLE) td
            FROM h ORDER BY d""").fetchall()
        e = con.execute("SELECT MAX(ELEVATION) FROM h").fetchall()[0][0]
    except Exception:
        con.close(); return None
    con.close()
    if not rows:
        return None
    elev = float(e) if (e is not None and np.isfinite(float(e))) else elev
    ds = np.array([r[0] for r in rows])
    T = np.array([r[1] if r[1] is not None else np.nan for r in rows], float)
    Td = np.array([r[2] if r[2] is not None else np.nan for r in rows], float)
    yr = np.array([int(s[:4]) for s in ds]); mo = np.array([int(s[5:7]) for s in ds])
    da = np.array([int(s[8:10]) for s in ds]); hh = np.array([int(s[11:13]) for s in ds])
    T[(T < -80) | (T > 65)] = np.nan
    Td[(Td < -90) | (Td > 45)] = np.nan
    df = pd.DataFrame(dict(y=yr, mo=mo, da=da, hh=hh, t=T, td=Td))
    g = df.groupby(['y', 'mo', 'da', 'hh'], sort=True).agg(
        t=('t', 'mean'), td=('td', 'mean')).reset_index()
    yr = g['y'].to_numpy(); mo = g['mo'].to_numpy(); da = g['da'].to_numpy(); hh = g['hh'].to_numpy()
    T = g['t'].to_numpy(float); Td = g['td'].to_numpy(float)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    doy = cum[mo - 1] + da + np.where((mo > 2) & leap, 1, 0)
    ok = np.isfinite(T) & np.isfinite(Td)
    if ok.sum() < 5000:
        return None
    el = solar_elevation_vec(lat, lon, doy[ok], hh[ok] + 0.5)
    idx = np.where(ok)[0][el < SUN]
    if len(idx) < 5000:
        return None
    y = yr[idx].astype(float); T = T[idx]; Td = Td[idx]
    fy = y + (doy[idx] - 1) / 365.0
    years = np.unique(y)
    if len(years) < 20:
        return None
    am_T = np.array([np.nanmean(T[y == yy]) for yy in years])
    am_Td = np.array([np.nanmean(Td[y == yy]) for yy in years])
    bT = theil_sen(years, am_T); bTd = theil_sen(years, am_Td)
    if not (np.isfinite(bT) and np.isfinite(bTd)):
        return None
    p = pressure_from_elev(elev)
    t0 = float(np.median(fy))

    # ---- 版本 S: 保变率 + 平滑去趋势(推荐口径) ----
    sT = lowess_trend(years, am_T)
    sTd = lowess_trend(years, am_Td)
    fT = np.interp(fy, years, sT)          # 逐小时时刻的低频分量
    fTd = np.interp(fy, years, sTd)
    tS_T = float(np.interp(t0, years, sT)) if np.isfinite(sT).all() else np.nanmean(sT)
    tS_Td = float(np.interp(t0, years, sTd)) if np.isfinite(sTd).all() else np.nanmean(sTd)
    TS = T - (fT - tS_T)                   # 温度: 去低频趋势, 保留异常
    TdS = Td - (fTd - tS_Td)               # 湿度: 去低频趋势, 保留异常
    # ---- 版本 V: 保变率 + 线性去趋势(仅作对照, 会因加速增温失真) ----
    TV = T - bT * (fy - t0)
    TdV = Td - bTd * (fy - t0)
    # ---- 版本 L: 锁平滑气候态(去趋势+去变率) ----
    d = doy[idx] - 1; h = hh[idx]
    climT = harmonic_clim(d, h, T); climTd = harmonic_clim(d, h, Td)
    if climT is None or climTd is None:
        return None

    def mk(a, b):
        return tw_vec(a, np.minimum(b, a - 1e-3), p)

    TW = dict(A=mk(T, Td), B=mk(T, TdV), C=mk(TV, Td), D=mk(TV, TdV))
    TS_ = dict(A=mk(T, Td), B=mk(T, TdS), C=mk(TS, Td), D=mk(TS, TdS))
    TL = dict(A=mk(T, Td), B=mk(T, np.minimum(climTd, T - 1e-3)),
              C=mk(climT, np.minimum(Td, climT - 1e-3)),
              D=mk(climT, np.minimum(climTd, climT - 1e-3)))

    def series(twd, th):
        out = {}
        for tag, tw in twd.items():
            num = np.zeros(len(years)); den = np.zeros(len(years))
            for i, yy in enumerate(years):
                s = (y == yy); v = tw[s]; f = np.isfinite(v)
                num[i] = np.nansum((v[f] >= 0).astype(float))  # placeholder
            out[tag] = num
        return out

    rec = {}
    for th in THS:
        r = {}
        for key, twd in (('S', TS_), ('V', TW), ('L', TL)):
            num = {}; den = None
            for tag, tw in twd.items():
                nn = np.zeros(len(years)); dd = np.zeros(len(years))
                for i, yy in enumerate(years):
                    s = (y == yy); v = tw[s]; f = np.isfinite(v)
                    nn[i] = np.nansum((v[f] >= th).astype(float)); dd[i] = f.sum()
                num[tag] = nn
                if tag == 'A':
                    den = dd
            hn = float(np.nanmedian(den))
            with np.errstate(invalid='ignore', divide='ignore'):
                fr = {t: np.where(den > 200, num[t] / np.maximum(den, 1e-9), np.nan) * hn
                      for t in num}
            # 干球对照
            nn = np.zeros(len(years))
            for i, yy in enumerate(years):
                s = (y == yy); v = T[s]; nn[i] = np.nansum((v[np.isfinite(v)] >= th).astype(float))
            with np.errstate(invalid='ignore', divide='ignore'):
                dry = np.where(den > 200, nn / np.maximum(den, 1e-9), np.nan) * hn
            r[key] = fr
            if key == 'V':
                r['dry'] = dry
        rec[th] = dict(years=years, S=r['S'], V=r['V'], L=r['L'], dry=r['dry'],
                       nhours=den, hnight=hn, bT=bT, bTd=bTd)
    return rec


def med_trend(years, v):
    if np.isfinite(v).sum() < 10:
        return np.nan
    return theil_sen(years.astype(float), v.astype(float)) * 10


def agg_trend(sd, tag_sel):
    """网络聚合: 逐年对所有可用站取等权平均, 再对该序列做趋势。
    聚合可消除单站阈值计数的大量并列零值, 比"逐站趋势取中位"稳定得多。"""
    p = sd.pivot_table(index='year', columns='tag', values='val', aggfunc='mean')
    yrs = p.index.values.astype(float)
    return {t: (theil_sen(yrs, p[t].values) * 10) for t in tag_sel}, float(np.nanmean(p['A'].values))


def main():
    m = pd.read_csv('metrics_v3.csv')
    SER = []
    pop = pd.read_csv('station_population.csv')[['station', 'pop_50km', 'band', 'lat', 'lon']]
    keep = set(m.loc[m.get('ok_trend', 1) == 1, 'station']) if 'ok_trend' in m.columns else set(m['station'])
    rows = []
    for sid, r in pop.iterrows():
        st = r['station']
        if st not in keep:
            continue
        fs = sorted(glob.glob('ghcn_data/GHCNh_%s_*.parquet' % st))
        if len(fs) < 20:
            continue
        g = m[m.station == st]
        lat = float(g['lat'].iloc[0]) if len(g) else float(r['lat'])
        lon = float(g['lon'].iloc[0]) if len(g) else float(r['lon'])
        elev = float(g['elev'].iloc[0]) if (len(g) and 'elev' in g.columns) else 20.0
        rec = station_decomp(st, lat, lon, elev, fs)
        if rec is None:
            continue
        for th, d in rec.items():
            o = dict(station=st, band=r['band'], pop=float(r['pop_50km']), thr=th,
                     nyears=len(d['years']), bT=d['bT'] * 10, bTd=d['bTd'] * 10,
                     hnight=d['hnight'],
                     cov=med_trend(d['years'], d['nhours']))
            for key in ('S', 'V', 'L'):
                for tag in 'ABCD':
                    o['%s_%s_tr' % (key, tag)] = med_trend(d['years'], d[key][tag])
                    o['%s_%s_mn' % (key, tag)] = float(np.nanmean(d[key][tag]))
                    for i, yy in enumerate(d['years']):
                        SER.append((st, r['band'], float(r['pop_50km']), th, key, tag,
                                    int(yy), float(d[key][tag][i])))
            o['dry_tr'] = med_trend(d['years'], d['dry'])
            o['dry_mn'] = float(np.nanmean(d['dry']))
            for i, yy in enumerate(d['years']):
                SER.append((st, r['band'], float(r['pop_50km']), th, 'dry', 'X',
                            int(yy), float(d['dry'][i])))
            rows.append(o)
        print('  ok %s (%d)' % (st, len(rows)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('humidity_decomp_v2.csv', index=False)
    sd = pd.DataFrame(SER, columns=['station', 'band', 'pop', 'thr', 'key', 'tag', 'year', 'val'])
    sd.to_csv('humidity_decomp_series.csv', index=False)
    print('saved humidity_decomp_v2.csv', df.shape, 'series', sd.shape)


def report():
    df = pd.read_csv('humidity_decomp_v2.csv')
    L = ['湿度必要性分解 · 方法学修正版 (step34)',
         '样本: %d 站' % df[df.thr == 21.9]['station'].nunique(),
         '',
         '反事实口径说明:',
         '  S = 保变率 + 平滑(LOWESS)去趋势  ← 推荐口径: 保留变率且能处理加速型增温',
         '  V = 保变率 + 线性去趋势          ← 加速增温下 D 会失真(本轮实测 D=-26)',
         '  L = 锁平滑气候态 (去趋势 + 去变率) ← 第10轮口径, 混淆了变率效应',
         '因子分解以 D 为参照: 温度主效=tr(B)-tr(D), 湿度主效=tr(C)-tr(D),',
         '                    交互=tr(A)-tr(B)-tr(C)+tr(D)', '']
    js = {}
    for th in sorted(df.thr.unique()):
        g = df[df.thr == th]
        L.append('=' * 62)
        L.append('阈值 %.1f °C   n=%d 站' % (th, g['station'].nunique()))
        L.append('  夜间干球趋势 中位 %+.3f °C/10yr | 夜间露点趋势 中位 %+.3f °C/10yr'
                 % (g['bT'].median(), g['bTd'].median()))
        L.append('  [自检] 年有效夜间小时趋势 中位 %+.2f h/10yr (基线 %.0f h/yr) —— 应≈0'
                 % (g['cov'].median(), g['hnight'].median()))
        sd = pd.read_csv('humidity_decomp_series.csv')
        sdth = sd[sd.thr == th]
        L.append('  ---- 网络聚合口径 (逐年全站等权平均后再取趋势, 主口径) ----')
        js.setdefault(str(th), {})['aggregate'] = {}
        for key in ('S', 'V', 'L'):
            tr, base = agg_trend(sdth[sdth.key == key], 'ABCD')
            te = tr['B'] - tr['D']; hu = tr['C'] - tr['D']
            ia = tr['A'] - tr['B'] - tr['C'] + tr['D']
            tot = tr['A']
            L.append('    %s: A %+7.2f | B %+7.2f | C %+7.2f | D %+7.2f  (基线 %.0f h/yr)'
                     % (key, tr['A'], tr['B'], tr['C'], tr['D'], base))
            if abs(tot) > 1e-6:
                L.append('        温度主效 %+7.2f (%+.0f%%) | 湿度主效 %+7.2f (%+.0f%%) | 交互 %+7.2f (%+.0f%%)'
                         % (te, 100 * te / tot, hu, 100 * hu / tot, ia, 100 * ia / tot))
                L.append('        → 仅靠增温可解释 %+.0f%% ;  需要湿度变化 %+.0f%%'
                         % (100 * (te + tr['D']) / tot, 100 * (hu + ia) / tot))
            js[str(th)]['aggregate'][key] = dict(
                A=float(tr['A']), B=float(tr['B']), C=float(tr['C']), D=float(tr['D']),
                temp=float(te), humid=float(hu), inter=float(ia), base=float(base),
                share_temp=float(te / tot) if abs(tot) > 1e-6 else None,
                share_humid=float(hu / tot) if abs(tot) > 1e-6 else None,
                share_inter=float(ia / tot) if abs(tot) > 1e-6 else None)
        js[str(th)] = {}
        for key, name in (('S', '保变率+平滑去趋势(推荐)'), ('V', '保变率+线性去趋势'),
                          ('L', '锁气候态(第10轮)')):
            A = g[key + '_A_tr']; B = g[key + '_B_tr']; C = g[key + '_C_tr']; D = g[key + '_D_tr']
            mA = g[key + '_A_mn']; mB = g[key + '_B_mn']; mC = g[key + '_C_mn']; mD = g[key + '_D_mn']
            te = B - D; hu = C - D; ia = A - B - C + D
            L.append('  ---- %s (%s) ----' % (name, key))
            L.append('    版本      基线 h/yr   趋势 h/10yr   p(符号)')
            for tag, v in (('A 实测', A), ('B 仅温度', B), ('C 仅湿度', C), ('D 双去(参照)', D)):
                vv = v.dropna().values
                pos = int((vv > 0).sum()); neg = int((vv < 0).sum())
                L.append('    %-11s %8.0f   %+8.2f   %9.1e'
                         % (tag, g[key + '_' + tag[0] + '_mn'].median(), np.nanmedian(v),
                            binom_p(pos, neg)))
            L.append('    基线: A %.0f | B %.0f | C %.0f | D %.0f'
                     % (mA.median(), mB.median(), mC.median(), mD.median()))
            L.append('    分解(中位, h/10yr): 温度主效 %+.2f | 湿度主效 %+.2f | 交互 %+.2f | 参照D %+.2f'
                     % (te.median(), hu.median(), ia.median(), D.median()))
            tot = np.nanmedian(A)
            if abs(tot) > 1e-9:
                L.append('    占实测: 温度 %+.0f%%  湿度 %+.0f%%  交互 %+.0f%%  (D %+.0f%%)'
                         % (100 * te.median() / tot, 100 * hu.median() / tot,
                            100 * ia.median() / tot, 100 * D.median() / tot))
                L.append('    → 仅靠增温(湿度不去趋势项)能解释的份额: %+.0f%%'
                         % (100 * (te.median() + D.median()) / tot))
                L.append('    → 需要湿度变化的份额: %+.0f%%'
                         % (100 * (hu.median() + ia.median()) / tot))
            js[str(th)][key] = dict(
                n=int(g['station'].nunique()),
                A=float(np.nanmedian(A)), B=float(np.nanmedian(B)),
                C=float(np.nanmedian(C)), D=float(np.nanmedian(D)),
                temp=float(te.median()), humid=float(hu.median()), inter=float(ia.median()),
                share_temp=float(te.median() / tot) if abs(tot) > 1e-9 else None,
                share_humid=float(hu.median() / tot) if abs(tot) > 1e-9 else None,
                share_inter=float(ia.median() / tot) if abs(tot) > 1e-9 else None,
                base_A=float(mA.median()), base_B=float(mB.median()),
                base_C=float(mC.median()), base_D=float(mD.median()))
        # 干球对照
        L.append('  ---- 干球越阈小时 ----')
        L.append('    基线 %.0f h/yr  趋势 %+.2f h/10yr  (湿球 A 趋势 %+.2f)'
                 % (g['dry_mn'].median(), g['dry_tr'].median(), g['V_A_tr'].median()))
        js[str(th)]['dry'] = dict(base=float(g['dry_mn'].median()), trend=float(g['dry_tr'].median()))
        # 分气候带 (S 口径)
        L.append('  ---- 分气候带 (S 口径, 中位 h/10yr) ----')
        js[str(th)]['by_band'] = {}
        for b, gg in g.groupby('band'):
            if len(gg) < 5:
                continue
            A = gg['S_A_tr']; B = gg['S_B_tr']; C = gg['S_C_tr']; D = gg['S_D_tr']
            te = (B - D).median(); hu = (C - D).median(); ia = (A - B - C + D).median()
            L.append('    %-12s n=%3d  A %+7.2f | 温度 %+6.2f | 湿度 %+6.2f | 交互 %+6.2f'
                     % (b, gg['station'].nunique(), A.median(), te, hu, ia))
            js[str(th)]['by_band'][str(b)] = dict(n=int(gg['station'].nunique()),
                                                  A=float(A.median()), temp=float(te),
                                                  humid=float(hu), inter=float(ia))
        L.append('')
    txt = '\n'.join(L)
    print(txt)
    with open('humidity_decomp_v2.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('humidity_decomp_v2.json', 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
