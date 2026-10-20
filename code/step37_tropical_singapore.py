# -*- coding: utf-8 -*-
"""
step37: 湿热带死亡面板 —— 新加坡(1.35°N) 月尺度检验
====================================================
第 9-11 轮最大的未关闭缺口: 欧洲死亡面板的 NWH₂₁.₉ 暴露中位数约为 71 h/yr,
而热带台站网络为 ~1500 h/yr(21×)。欧洲 RR 落在全球暴露分布的 0-5 分位,
"热带更严重"因此一直是假说而非结论。

本脚本首次把死亡结局放进湿热带:
  - 结局: 新加坡按月总死亡(1960-01 .. 2026-06, 798 个月), 来源 SingStat M810121
    (Registry of Births and Deaths)
  - 暴露: NASA POWER/MERRA-2 逐小时 T2M/T2MDEW(1.35°N, 103.82°E),
    按太阳高度角 < -0.833° 定义夜间, 计算逐月夜间湿球均值与 NWH₂₁.₉
  - 模型: Poisson(死亡) ~ 偏移(log 人口) + 年样条(df=8) + 月 Fourier(2 对)
          + 暴露; 报 RR/SD 与 RR/unit
  - 对照: 同时用白天最高温、夜间干球做头对头, 判断湿球是否独立
"""
import os, sys, json, math, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from step30_humidity_attribution import (tw_vec, solar_elevation_vec,
                                         pressure_from_elev)
import step33_nwh_projection as S33

OUT = 'ext/tropical'
LAT, LON = 1.3521, 103.8198
SUN = -0.833
THR = 21.9
UA = {'User-Agent': 'Mozilla/5.0 (research; climate-health)'}


def parse_deaths():
    rows = {}
    for f, tag in (('sg_deaths.json', 'total'), ('sgr2.json', 'male'), ('sgr3.json', 'female')):
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding='utf-8'))['Data']
        for r in d.get('row', []):
            for c in r.get('columns', []):
                k = c.get('key')            # e.g. '1960 Jan'
                try:
                    y, m = k.split()
                    ym = '%s-%02d' % (y, 'JanFebMarAprMayJunJulAugSepOctNovDec'.find(m) // 3 + 1)
                except Exception:
                    continue
                v = c.get('value')
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                rows.setdefault(ym, {})[tag] = v
    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = 'ym'
    return df


def fetch_power_year(y):
    s = '%d0101' % y
    e = '%d1231' % y
    url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW'
           '&community=RE&longitude=%.4f&latitude=%.4f&start=%s&end=%s&format=JSON'
           % (LON, LAT, s, e))
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                js = json.loads(r.read().decode('utf-8'))
            p = js['properties']['parameter']
            t2, td = p['T2M'], p['T2MDEW']
            keys = sorted(t2.keys())
            rec = []
            for kk in keys:
                a = t2[kk]; b = td.get(kk, -999)
                if a > -900 and b > -900:
                    rec.append(('%s-%s-%sT%s:00' % (kk[0:4], kk[4:6], kk[6:8], kk[8:10]), a, b))
            return rec
        except Exception as ex:
            time.sleep(3 + 4 * k)
    return None


def build_wx(y0=2001, y1=2025):   # POWER 逐小时(MERRA-2)仅从 2001-01-01 起
    dst = os.path.join(OUT, 'power_singapore_hourly.csv')
    os.makedirs(OUT, exist_ok=True)
    done = {}
    if os.path.exists(dst):
        try:
            old = pd.read_csv(dst)
            for yy, g in old.groupby('year'):
                done[int(yy)] = g[['dt', 't', 'td']].values.tolist()
        except Exception:
            pass
    recs = []
    for y in range(y0, y1 + 1):
        if y in done:
            recs.extend(done[y]); continue
        r = fetch_power_year(y)
        if r is None:
            print('  FAIL', y, flush=True); continue
        for dt, a, b in r:
            recs.append([dt, a, b])
        print('  ok', y, len(r), flush=True)
        time.sleep(1.0)
    df = pd.DataFrame(recs, columns=['dt', 't', 'td']).drop_duplicates('dt')
    df['year'] = df['dt'].str[:4].astype(int)
    df.to_csv(dst, index=False)
    print('hourly rows', len(df))
    return df


def monthly_exposure(df):
    ds = df['dt'].values
    yr = np.array([int(s[:4]) for s in ds]); mo = np.array([int(s[5:7]) for s in ds])
    da = np.array([int(s[8:10]) for s in ds]); hh = np.array([int(s[11:13]) for s in ds])
    T = df['t'].values.astype(float); Td = df['td'].values.astype(float)
    T[(T < -80) | (T > 60)] = np.nan; Td[(Td < -90) | (Td > 40)] = np.nan
    ok = np.isfinite(T) & np.isfinite(Td)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    doy = cum[mo - 1] + da + np.where((mo > 2) & leap, 1, 0)
    el = solar_elevation_vec(LAT, LON, doy[ok], hh[ok] + 0.5)
    idx = np.where(ok)[0]
    night = el < SUN
    inight = idx[night]
    iday = idx[~night]
    p = 1013.25
    # 夜间湿球
    Twn = tw_vec(T[inight], np.minimum(Td[inight], T[inight] - 1e-3), p)
    ynm = yr[inight]; mnm = mo[inight]
    # 白天干球
    Tx = T[iday]; ydm = yr[iday]; mdm = mo[iday]
    keys = []
    for y in np.unique(yr):
        for m in range(1, 13):
            keys.append((int(y), m))
    out = []
    nh_night = []      # 每月夜间小时数(天文, 常数)
    for (y, m) in keys:
        s = (ynm == y) & (mnm == m)
        sd = (ydm == y) & (mdm == m)
        if s.sum() < 100:
            continue
        v = Twn[s]
        fr = float(np.mean(v >= THR)) if np.isfinite(v).any() else np.nan
        # 该月天文夜间小时数
        d0 = np.array([doy[i] for i in range(len(doy)) if yr[i] == y and mo[i] == m])
        out.append(dict(ym='%d-%02d' % (y, m), year=y, month=m,
                        tw_night=float(np.nanmean(v)), tw_night_sd=float(np.nanstd(v)),
                        f_nwh=fr, n_hrs=int(s.sum()),
                        tx_day=float(np.nanmean(Tx[sd])) if sd.sum() > 100 else np.nan,
                        tn_night=float(np.nanmean(T[inight][s]))))
    return pd.DataFrame(out)


def night_hours_const():
    """各月天文夜间小时数 (2001 年, 365 天)。"""
    import datetime as _dt
    cum = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    res = {}
    d0 = _dt.date(2001, 1, 1)
    all_d = [(d0 + _dt.timedelta(days=i)) for i in range(365)]
    doy = np.array([d.timetuple().tm_yday for d in all_d])
    mo = np.array([d.month for d in all_d])
    hh = np.tile(np.arange(24), 365)
    dy = np.repeat(doy, 24); dm = np.repeat(mo, 24)
    el = solar_elevation_vec(LAT, LON, dy, hh + 0.5)
    n = el < SUN
    for m in range(1, 13):
        res[m] = float(n[dm == m].sum())
    return res


def run_model(d):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = d.dropna(subset=['deaths', 'pop']).copy()
    d['t'] = (d['year'] - d['year'].min()) / 10.0
    # 年样条 (自然三次, df=8)
    from patsy import dmatrix
    B = np.asarray(dmatrix('bs(t, df=8, degree=3) - 1', d, return_type='dataframe'))
    for i in range(B.shape[1]):
        d['sp%d' % i] = B[:, i]
    d['sin1'] = np.sin(2 * np.pi * d['month'] / 12)
    d['cos1'] = np.cos(2 * np.pi * d['month'] / 12)
    d['sin2'] = np.sin(4 * np.pi * d['month'] / 12)
    d['cos2'] = np.cos(4 * np.pi * d['month'] / 12)
    base = ['sp%d' % i for i in range(B.shape[1])] + ['sin1', 'cos1', 'sin2', 'cos2']
    res = {}
    specs = {
        'E1 夜间湿球均值 (°C)': ['tw_night'],
        'E2 夜间干球均值 (°C)': ['tn_night'],
        'E3 白天干球均值 (°C)': ['tx_day'],
        'E4 夜间湿球 + 夜间干球': ['tw_night', 'tn_night'],
        'E5 夜间湿球 + 白天干球': ['tw_night', 'tx_day'],
        'E6 三者同现': ['tw_night', 'tn_night', 'tx_day'],
        'E7 NWH₂₁.₉ (h/月)': ['nwh'],
        'E8 NWH₂₁.₉ + 白天干球': ['nwh', 'tx_day'],
    }
    for name, exps in specs.items():
        cols = base + exps
        sub = d.dropna(subset=cols)
        if len(sub) < 100:
            continue
        X = sm.add_constant(sub[cols].astype(float))
        try:
            m = sm.GLM(sub['deaths'].astype(float), X, family=sm.families.Poisson(),
                       offset=np.log(sub['pop'].astype(float))).fit(cov_type='HC1')
        except Exception as e:
            print('  fit fail', name, e); continue
        rr = {}
        for e in exps:
            b = m.params[e]; se = m.bse[e]
            sd = float(sub[e].std())
            rr[e] = dict(rr_unit=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                         hi=float(np.exp(b + 1.96 * se)), z=float(b / se),
                         p=float(2 * (1 - 0.5 * (1 + math.erf(abs(b / se) / math.sqrt(2))))),
                         sd=sd, rr_sd=float(np.exp(b * sd)),
                         rr_sd_lo=float(np.exp((b - 1.96 * se) * sd)),
                         rr_sd_hi=float(np.exp((b + 1.96 * se) * sd)))
        res[name] = dict(n=int(len(sub)), rr=rr, aic=float(m.aic), ll=float(m.llf))
    return res, d


def main():
    os.makedirs(OUT, exist_ok=True)
    dea = parse_deaths()
    dea.to_csv(os.path.join(OUT, 'sg_deaths_monthly.csv'))
    print('deaths', dea.shape, dea.index.min(), dea.index.max())
    # 人口
    popf = os.path.join(OUT, 'sg_population.csv')
    if not os.path.exists(popf):
        rec = None
        for src in ('sg.json',):
            if os.path.exists(src):
                try:
                    js = json.load(open(src, encoding='utf-8'))
                    for row in js['Data']['row']:
                        if (row.get('rowText') or '').strip() == 'Total Population':
                            rec = [(c['key'], c['value']) for c in row['columns']]
                            break
                except Exception:
                    rec = None
            if rec:
                break
        if rec is None:
            url = ('https://tablebuilder.singstat.gov.sg/api/table/tabledata/M810001'
                   '?seriesNoORrowNo=1&sortBy=key%20asc&offset=0&limit=200')
            for k in range(4):
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=90) as r:
                        js = json.loads(r.read().decode('utf-8'))
                    for row in js['Data']['row']:
                        if (row.get('rowText') or '').strip() == 'Total Population':
                            rec = [(c['key'], c['value']) for c in row['columns']]
                            break
                    if rec:
                        break
                except Exception:
                    time.sleep(3)
        if rec is None:
            raise SystemExit('population unavailable')
        pd.DataFrame(rec, columns=['year', 'pop']).to_csv(popf, index=False)
    pop = pd.read_csv(popf, dtype=str)
    pop['year'] = pop['year'].astype(int); pop['pop'] = pd.to_numeric(pop['pop'], errors='coerce')
    print('pop', pop.shape, pop.year.min(), pop.year.max())
    wx = build_wx()
    mn = monthly_exposure(wx)
    NH = night_hours_const()
    mn['nwh'] = mn['f_nwh'] * mn['month'].map(NH)
    mn.to_csv(os.path.join(OUT, 'sg_monthly_exposure.csv'), index=False)
    print(mn.groupby('month')[['tw_night', 'nwh', 'tx_day']].mean().round(1))

    d = mn.merge(dea.reset_index(), on='ym', how='inner')
    d['year'] = d['year_x'] if 'year_x' in d.columns else d['year']
    d = d[['ym', 'year', 'month', 'deaths' if 'deaths' in d.columns else 'total',
           'tw_night', 'tn_night', 'tx_day', 'nwh']].rename(columns={'total': 'deaths'})
    d = d.merge(pop, on='year', how='left')
    d['pop'] = d['pop'].astype(float)
    d = d.dropna(subset=['deaths', 'pop', 'tw_night'])
    print('panel', d.shape, d['ym'].min(), d['ym'].max(),
          'deaths/月 中位 %.0f' % d['deaths'].median())
    print('NWH 月均 %.0f h/月 → 年化 %.0f h/yr' % (d['nwh'].mean(), d['nwh'].mean() * 12))
    print('夜间湿球月均 %.2f °C (SD %.2f)' % (d['tw_night'].mean(), d['tw_night'].std()))
    res, dd = run_model(d)
    L = ['湿热带死亡面板: 新加坡 (1.35°N) 月尺度',
         '结局: SingStat M810121 按月总死亡 (Registry of Births and Deaths)',
         '暴露: NASA POWER/MERRA-2 逐小时 → 夜间湿球(太阳高度角<-0.833°)',
         '面板: %d 月, %s .. %s' % (len(dd), dd['ym'].min(), dd['ym'].max()),
         '月死亡中位数 %.0f; 人口 %.2f 百万(年均)' % (dd['deaths'].median(), dd['pop'].mean() / 1e6),
         '',
         '【暴露水平 —— 这是本研究第一次进入湿热带】',
         '  夜间湿球月均 %.2f °C  年内 SD %.2f  区间 [%.1f, %.1f]'
         % (dd['tw_night'].mean(), dd['tw_night'].std(), dd['tw_night'].min(), dd['tw_night'].max()),
         '  NWH₂₁.₉ 月均 %.0f h  →  **年化 %.0f h/yr**   (欧洲面板 ~71 h/yr)'
         % (dd['nwh'].mean(), dd['nwh'].mean() * 12),
         '  NWH₂₁.₉ 非零月比例 %.0f%%' % (100 * (dd['nwh'] > 1).mean()),
         '',
         '【模型】Poisson: 死亡 ~ 偏移(log 人口) + 年样条(df=8) + 月 Fourier(2对) + 暴露',
         '  规格                        n    暴露          RR/unit [95%CI]            RR/SD [95%CI]        z']
    js = {}
    for name, r in res.items():
        for e, v in r['rr'].items():
            L.append('  %-26s %4d  %-12s %8.5f [%.5f,%.5f]  %8.4f [%.4f,%.4f]  %+6.2f'
                     % (name, r['n'], e, v['rr_unit'], v['lo'], v['hi'],
                        v['rr_sd'], v['rr_sd_lo'], v['rr_sd_hi'], v['z']))
        js[name] = r
    txt = '\n'.join(L)
    print(txt)
    open('tropical_singapore.txt', 'w', encoding='utf-8').write(txt + '\n')
    json.dump(js, open('tropical_singapore.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
