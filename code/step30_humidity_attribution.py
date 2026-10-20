# -*- coding: utf-8 -*-
"""第10轮 R10-1  致命检验: NWH 趋势到底由"温度"还是"湿度"驱动?

审稿人 M54 (致命):
  NWH₂₁.₉ 的趋势可能 100% 由夜间气温上升驱动, 湿度毫无贡献。
  若如此, "湿球"这一卖点是伪创新 —— 用夜间干球 TN 做同样计算即可得到同样结论。
  必须做反事实分解。

设计 (逐站去趋势反事实, 保留天气尺度变率):
  记 b_T / b_Td = 该站夜间平均干球 / 露点的 Theil-Sen 年趋势 (°C/yr)
    A = NWH(T,          Td         )   —— 实测
    B = NWH(T,          Td − b_Td·Δt)  —— 湿度去趋势 → 纯温度驱动
    C = NWH(T − b_T·Δt, Td         )   —— 温度去趋势 → 纯湿度驱动
    D = NWH(T − b_T·Δt, Td − b_Td·Δt)  —— 双去趋势 → 应 ≈ 0 (自检)
  温度贡献 = B,  湿度贡献 = C,  交互项 = A − B − C

同时给出业务上更有意义的"干热夜 → 湿热夜"转化量:
    H_dry = 夜间干球 ≥ 阈值的小时数   (现行"热带夜"类判据的小时版)
    H_wet = 夜间湿球 ≥ 阈值的小时数   (本研究的判据)
    gap   = H_dry − H_wet            —— 干球已越阈、但湿球仍可恢复的小时
  gap 缩小 = 夜间的"干爽缓冲"正在消失, 这是比湿球绝对值更能被现行体系理解的指标。

产出: humidity_attribution.txt / .json
"""
import os, sys, math, glob, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
LC = 2418.0
SUN = -0.833

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------- 向量化湿球 (Davies-Jones Newton) ----------------
def esat(t):
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def tw_vec(T, Td, p):
    """干球+露点 → 湿球, 向量化。NaN 直接透传。"""
    T = np.asarray(T, dtype=float)
    Td = np.asarray(Td, dtype=float)
    out = np.full(T.shape, np.nan)
    ok = np.isfinite(T) & np.isfinite(Td)
    t_ = T[ok].copy()
    d_ = np.minimum(Td[ok], t_ - 1e-6)        # 过饱和按饱和处理
    e = np.minimum(esat(d_), p * 0.95)
    w = 0.622 * e / (p - e)
    t = t_ - (t_ - d_) * 0.33
    for _ in range(8):
        es = np.minimum(esat(t), p * 0.95)
        den = p - es
        ws = 0.622 * es / den
        des = es * 4303.845 / ((t + 243.5) ** 2)
        dw = 0.622 * p * des / (den * den)
        g = LC * (ws - w) - (t_ - t)
        gp = LC * dw + 1.0
        step = g / gp
        t -= step
        if np.nanmax(np.abs(step)) < 1e-7:
            break
    t = np.where((t > t_ - 90) & (t <= t_ + 0.01), t, t_)
    out[ok] = t
    # 饱和: Td >= T → Tw = T
    sat = ok & (Td >= T)
    out[sat] = T[sat]
    return out


def _harm_fit(dd, vv):
    """对日序做年周期+半年周期谐波最小二乘。"""
    X = np.column_stack([np.ones_like(dd),
                         np.sin(2 * np.pi * dd / 365.0), np.cos(2 * np.pi * dd / 365.0),
                         np.sin(4 * np.pi * dd / 365.0), np.cos(4 * np.pi * dd / 365.0)])
    beta, *_ = np.linalg.lstsq(X, vv, rcond=None)
    return beta


def _harm_eval(xi, beta):
    X = np.column_stack([np.ones_like(xi),
                         np.sin(2 * np.pi * xi / 365.0), np.cos(2 * np.pi * xi / 365.0),
                         np.sin(4 * np.pi * xi / 365.0), np.cos(4 * np.pi * xi / 365.0)])
    return X @ beta


def harmonic_clim(d, h, v):
    """平滑的 日序×小时 气候态。缺数据的小时回退到全小时合并拟合。"""
    ok = np.isfinite(v)
    if ok.sum() < 3000:
        return None
    dfp = pd.DataFrame({'d': d[ok], 'h': h[ok], 'v': v[ok]})
    pooled = dfp.groupby('d')['v'].mean()
    bd = pooled.index.to_numpy(float)
    bpool = _harm_fit(bd, pooled.to_numpy(float))
    out = np.full(v.shape, np.nan)
    for hh_ in range(24):
        m = ok & (h == hh_)
        if m.sum() < 150:
            continue
        sub = dfp[dfp['h'] == hh_].groupby('d')['v'].agg(['mean', 'size'])
        sub = sub[sub['size'] >= 2]
        if len(sub) < 60:
            continue
        beta = _harm_fit(sub.index.to_numpy(float), sub['mean'].to_numpy(float))
        xi = d[m].astype(float)
        out[m] = _harm_eval(xi, beta)
    # 回退
    miss = ok & ~np.isfinite(out)
    if miss.any():
        out[miss] = _harm_eval(d[miss].astype(float), bpool)
    return out


def pressure_from_elev(h):
    if h is None or not np.isfinite(h) or h < -500 or h > 6000:
        return 1013.25
    return 1013.25 * (1.0 - 2.25577e-5 * h) ** 5.25588


# ---------------- 太阳高度角 (向量化) ----------------
def solar_elevation_vec(lat, lon, doy, utc_hour):
    lat = np.asarray(lat, float)
    g = 2.0 * np.pi / 365.0 * (doy - 1 + (utc_hour - 12.0) / 24.0)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    lr = np.radians(lat)
    tst = utc_hour * 60.0 + eq + lon * 4.0
    ha = np.radians(tst / 4.0 - 180.0)
    cz = np.sin(lr) * np.sin(dec) + np.cos(lr) * np.cos(dec) * np.cos(ha)
    return 90.0 - np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 4:
        return float('nan')
    if n > 60:                       # 全对斜率在大样本下太慢, 抽样
        idx = np.random.default_rng(0).choice(n, 60, replace=False)
        x = x[idx]; y = y[idx]; n = 60
    sl = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        m = np.abs(dx) > 1e-9
        if m.any():
            sl.append((y[i + 1:][m] - y[i]) / dx[m])
    if not sl:
        return float('nan')
    return float(np.median(np.concatenate(sl)))


def binom_p(pos, neg):
    from math import comb
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


THS = [21.9, 25.8]


def station_nwh(station, lat, lon, elev, files):
    import duckdb
    con = duckdb.connect()
    pat = "','".join(f.replace("'", "''") for f in files)
    con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])" % pat)
    try:
        rows = con.execute("""
            SELECT CAST(DATE AS VARCHAR) d,
                   TRY_CAST(temperature AS DOUBLE) t,
                   TRY_CAST(dew_point_temperature AS DOUBLE) td
            FROM h ORDER BY d
        """).fetchall()
        e = con.execute("SELECT MAX(ELEVATION) FROM h").fetchall()[0][0]
    except Exception:
        con.close()
        return None
    con.close()
    if not rows:
        return None
    elev = float(e) if (e is not None and np.isfinite(float(e))) else elev
    ds = np.array([r[0] for r in rows])
    T = np.array([r[1] if r[1] is not None else np.nan for r in rows], float)
    Td = np.array([r[2] if r[2] is not None else np.nan for r in rows], float)
    # 解析 DATE: YYYY-MM-DDTHH:MM:SS
    yr = np.array([int(s[:4]) for s in ds])
    mo = np.array([int(s[5:7]) for s in ds])
    da = np.array([int(s[8:10]) for s in ds])
    hh = np.array([int(s[11:13]) for s in ds])
    T[(T < -80) | (T > 65)] = np.nan
    Td[(Td < -90) | (Td > 45)] = np.nan
    # ---- 逐小时分箱(与主引擎 step5c 同口径): 同一小时内的多次观测取平均 ----
    df = pd.DataFrame(dict(y=yr, mo=mo, da=da, hh=hh, t=T, td=Td))
    hh_df = df.groupby(['y', 'mo', 'da', 'hh'], sort=True).agg(
        t=('t', 'mean'), td=('td', 'mean'), n=('t', 'size')).reset_index()
    yr = hh_df['y'].to_numpy()
    mo = hh_df['mo'].to_numpy(); da = hh_df['da'].to_numpy(); hh = hh_df['hh'].to_numpy()
    T = hh_df['t'].to_numpy(float); Td = hh_df['td'].to_numpy(float)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    doy = cum[mo - 1] + da + np.where((mo > 2) & leap, 1, 0)
    ok = np.isfinite(T) & np.isfinite(Td)
    if ok.sum() < 5000:
        return None
    el = solar_elevation_vec(lat, lon, doy[ok], hh[ok] + 0.5)
    night = el < SUN
    idx = np.where(ok)[0][night]
    if len(idx) < 5000:
        return None
    y = yr[idx].astype(float)
    T = T[idx]; Td = Td[idx]
    fy = y + (doy[idx] - 1) / 365.0                      # 带小数的时间轴
    years = np.unique(y)
    if len(years) < 20:
        return None
    # 逐站夜间年均值趋势 (报告用, 不参与反事实)
    am_T = np.array([np.nanmean(T[y == yy]) for yy in years])
    am_Td = np.array([np.nanmean(Td[y == yy]) for yy in years])
    bT = theil_sen(years, am_T)
    bTd = theil_sen(years, am_Td)
    p = pressure_from_elev(elev)

    # ---- 季节 × 日变化 气候态 (平滑谐波拟合), 用于构造反事实 ----
    d = doy[idx] - 1          # 0..364
    h = hh[idx]
    climT = harmonic_clim(d, h, T)
    climTd = harmonic_clim(d, h, Td)
    if climT is None or climTd is None:
        return None
    # 四个版本的湿球只算一次, 阈值循环里仅做计数
    TwA = tw_vec(T, Td, p)                                        # 实测
    TwB = tw_vec(T, np.minimum(climTd, T - 1e-3), p)              # 湿度锁气候态
    TwC = tw_vec(climT, np.minimum(Td, climT - 1e-3), p)          # 温度锁气候态
    TwD = tw_vec(climT, np.minimum(climTd, climT - 1e-3), p)      # 双锁(自检)
    rec = {}
    for th in THS:
        res = {}
        for tag, tw in (('A', TwA), ('B', TwB), ('C', TwC), ('D', TwD)):
            num = np.zeros(len(years)); den = np.zeros(len(years))
            for i, yy in enumerate(years):
                s = (y == yy)
                v = tw[s]
                num[i] = np.nansum((v[ np.isfinite(v) ] >= th).astype(float))
                den[i] = np.nansum(np.isfinite(v))
            # 覆盖率无关: 用"越阈夜间小时占有效夜间小时的比例", 而非裸计数
            with np.errstate(invalid='ignore', divide='ignore'):
                fr = np.where(den > 200, num / np.maximum(den, 1e-9), np.nan)
            res[tag] = fr
        # 干球对照 (同样归一化)
        numd = np.zeros(len(years))
        for i, yy in enumerate(years):
            s = (y == yy); v = T[s]
            numd[i] = np.nansum((v[np.isfinite(v)] >= th).astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            frd = np.where(den > 200, numd / np.maximum(den, 1e-9), np.nan)
        # 折算成 小时/年: 乘该站常年的年夜间小时数(取各年中位数, 不随年变化)
        hn = float(np.nanmedian(den))
        rec[th] = dict(years=years,
                       A=res['A'] * hn, B=res['B'] * hn, C=res['C'] * hn, D=res['D'] * hn,
                       dry=frd * hn, gap=(frd - res['A']) * hn, nhours=den,
                       fracA=res['A'], fracB=res['B'], fracC=res['C'], hnight=hn,
                       bT=bT, bTd=bTd)
    return rec


def main():
    m = pd.read_csv('metrics_v3.csv')
    pop = pd.read_csv('station_population.csv')[['station', 'pop_50km', 'band', 'lat', 'lon']]
    span = pd.read_csv('station_panel.csv')[['station', 'elev']]
    mm = m[m['ok_trend'].astype(str) == '1']
    cnt = mm.groupby('station').size()
    keep = set(cnt[cnt >= 20].index)
    meta = pop.merge(span, on='station', how='left').set_index('station')
    files_by = {}
    for f in glob.glob(os.path.join('ghcn_data', 'GHCNh_*.parquet')):
        s = os.path.basename(f).split('_')[1]
        files_by.setdefault(s, []).append(f)
    rows = []
    done = 0
    for sid in sorted(keep):
        if sid not in files_by or sid not in meta.index:
            continue
        r = meta.loc[sid]
        try:
            rec = station_nwh(sid, float(r['lat']), float(r['lon']),
                              float(r.get('elev', np.nan)), sorted(files_by[sid]))
        except Exception as ex:
            print('FAIL', sid, ex)
            continue
        if rec is None:
            continue
        done += 1
        pw = float(r['pop_50km'])
        for th, d in rec.items():
            out = dict(station=sid, band=r['band'], pop=pw, thr=th,
                       nyears=len(d['years']), bT=d['bT'] * 10, bTd=d['bTd'] * 10)
            yy = d['years'].astype(float)
            for tag in ('A', 'B', 'C', 'D', 'dry', 'gap'):
                v = d[tag].astype(float)
                k = np.isfinite(v) & np.isfinite(yy)
                out['trend_' + tag] = theil_sen(yy[k], v[k]) * 10 if k.sum() >= 20 else np.nan
                out['mean_' + tag] = float(np.nanmean(v))
            # 覆盖率漂移自检: 有效夜间小时数本身是否有趋势 (应有 ≈ 无)
            nv = d['nhours'].astype(float)
            out['trend_cov'] = theil_sen(d['years'].astype(float), nv) * 10
            out['mean_cov'] = float(np.nanmean(nv))
            # 相对量: 每十年变化占基线均值的比例
            out['rel_A'] = out['trend_A'] / max(out['mean_A'], 1e-9)
            rows.append(out)
        print('[%d] %s  bT=%+.3f bTd=%+.3f  A=%+.1f B=%+.1f C=%+.1f'
              % (done, sid, d['bT'] * 10, d['bTd'] * 10,
                 out['trend_A'], out['trend_B'], out['trend_C']), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('humidity_attr_station.csv', index=False)

    lines = ['湿度必要性分解: NWH 趋势的温度驱动 vs 湿度驱动',
             '样本: %d 站' % df['station'].nunique(), '']
    js = {'n_stations': int(df['station'].nunique())}
    for th in THS:
        g = df[df.thr == th]
        if len(g) < 20:
            continue
        lines.append('===== 阈值 %.1f °C =====' % th)
        lines.append('  夜间平均干球趋势 中位 %+.3f °C/10yr  |  夜间平均露点趋势 中位 %+.3f °C/10yr'
                     % (g['bT'].median(), g['bTd'].median()))
        lines.append('  [自检] 年有效夜间小时数趋势 中位 %+.2f h/10yr (基线 %.0f h/yr) —— 应≈0,'
                     ' 否则覆盖率漂移污染' % (g['trend_cov'].median(), g['mean_cov'].mean()))
        lines.append('  %-6s %12s %12s %10s %10s' % ('版本', '基线均值h/yr', '趋势h/10yr', 'p(符号)', '站数'))
        js['thr_%.1f' % th] = {}
        store = {}
        for tag, nm in (('A', '实测(温+湿)'), ('B', '仅温度(湿度去趋势)'),
                        ('C', '仅湿度(温度去趋势)'), ('D', '双去趋势(自检≈0)'),
                        ('dry', '干球越阈小时'), ('gap', '干球越阈但湿球可恢复')):
            v = g['trend_' + tag].dropna().values
            pos = int((v > 0).sum()); neg = int((v < 0).sum())
            pb = binom_p(pos, neg)
            lines.append('  %-6s %12.1f %12.2f %10.1e %10d'
                         % (nm, g['mean_' + tag].mean(), np.median(v), pb, len(v)))
            store[tag] = dict(mean=float(g['mean_' + tag].mean()),
                              trend_median=float(np.median(v)),
                              trend_mean=float(np.mean(v)),
                              sign_pos=pos, sign_neg=neg, sign_p=float(pb))
        a = store['A']['trend_median']; b = store['B']['trend_median']
        c = store['C']['trend_median']; dd = store['D']['trend_median']
        inter = a - b - c
        lines.append('  ---- 分解 (h/10yr) ----')
        lines.append('    温度驱动 (B)      %+.2f   (占实测 %.0f%%)' % (b, 100 * b / a if a else float('nan')))
        lines.append('    湿度驱动 (C)      %+.2f   (占实测 %.0f%%)' % (c, 100 * c / a if a else float('nan')))
        lines.append('    交互项 (A−B−C)    %+.2f   (占实测 %.0f%%)' % (inter, 100 * inter / a if a else float('nan')))
        lines.append('    自检 D (应≈0)     %+.2f' % dd)
        lines.append('  ---- 干热夜 → 湿热夜 转化 ----')
        lines.append('    干球越阈小时趋势 %+.2f h/10yr' % store['dry']['trend_median'])
        lines.append('    湿球越阈小时趋势 %+.2f h/10yr' % store['A']['trend_median'])
        lines.append('    gap(干热但仍可恢复) 趋势 %+.2f h/10yr  基线 %.1f h/yr'
                     % (store['gap']['trend_median'], store['gap']['mean']))
        store['decomp'] = dict(temp=b, humid=c, inter=inter, selfcheck_D=dd,
                               frac_temp=(b / a if a else None), frac_humid=(c / a if a else None))
        js['thr_%.1f' % th] = store
        # 分气候带
        lines.append('  ---- 分气候带 (实测 A 趋势 h/10yr) ----')
        for band, gb in g.groupby('band'):
            v = gb['trend_A'].dropna().values
            if len(v) < 5:
                continue
            lines.append('    %-12s n=%3d  中位 %+.2f  C(湿度) %+.2f  B(温度) %+.2f'
                         % (band, len(v), np.median(v),
                            np.median(gb['trend_C'].dropna().values),
                            np.median(gb['trend_B'].dropna().values)))
            js['thr_%.1f' % th].setdefault('by_band', {})[band] = dict(
                n=int(len(v)), A=float(np.median(v)),
                C=float(np.median(gb['trend_C'].dropna().values)),
                B=float(np.median(gb['trend_B'].dropna().values)))
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open('humidity_attribution.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('humidity_attribution.json', 'w', encoding='utf-8') as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
