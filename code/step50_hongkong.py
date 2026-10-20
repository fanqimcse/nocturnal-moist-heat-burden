# -*- coding: utf-8 -*-
"""
step50 —— 亚年尺度湿热带健康证据 II：香港（第二座湿热城市）+ 与新加坡的双城 meta
=============================================================================
背景（第 17–19 轮）
  R17：湿热带健康证据只有新加坡 1 城（z=+2.03），是通往 Nature 正刊的唯一实质台阶。
  step48：WHO 多国**年度**面板失败（ICD-10 时代 z=+0.08）。
  step49：原因=P 年度聚合稀释 93% 效应（欧洲周 z=+34.38 → 年 z=+1.11）。
  => 缺口精确化为"缺**亚年尺度**的热带死亡数据"。

本步新增第二座湿热带城市：**中国香港**
  死亡：香港政府统计处 Table 115-01022A「按月死亡人数」1976-01 ~ 2026-12（逐月，按发生时间计）
  气象：NASA POWER 逐小时 MERRA-2（T2M / T2MDEW），2001-01 起
用**与新加坡完全相同**的管线重跑新加坡，再做双城随机效应 meta。

v2 修订（第 20 轮自查发现并修正）
  1. **冬季混杂**：香港冬季死亡高峰 + 夜间湿球最低 ⇒ 全月份单系数模型被"冷保护"主导
     （tw_night RR/SD=0.902, z=−19）。修正：主规格限定**暖季（5–10 月）**，
     并显式给出冷季对照以证明这是效应异质性而非噪声。
  2. **cov_type 被 statsmodels 忽略**（GLM.fit 不接受该 kwarg）⇒ 之前 SE 非稳健。
     改用 get_robustcov_results(cov_type='HC1')。
  3. 增加**局地分位数阈值**（NWH_p95 / NWH_p99）与绝对阈值并列，落实第 11 轮
     "绝对阈值在赤道饱和、须按当地分位数设阈"的结论。
"""
import os, sys, json, math, time, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from step30_humidity_attribution import tw_vec, solar_elevation_vec

OUT = 'ext/tropical'
SUN = -0.833
UA = {'User-Agent': 'Mozilla/5.0 (research; climate-health)'}

CITY = {
    'HK': dict(name='Hong Kong, China', lat=22.3193, lon=114.1694,
               deaths='ext/hk/hk_deaths_monthly.csv',
               power='ext/tropical/power_hongkong_hourly.csv',
               warm=list(range(5, 11))),          # 5–10 月
    'SG': dict(name='Singapore', lat=1.3521, lon=103.8198,
               deaths='ext/tropical/sg_deaths_monthly.csv',
               power='ext/tropical/power_singapore_hourly.csv',
               warm=list(range(1, 13))),          # 赤道，无冷季
}
Y0, Y1 = 2001, 2025
COVID = (2020, 2022)


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 气象
def fetch_power_year(lat, lon, y):
    url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW'
           '&community=RE&longitude=%.4f&latitude=%.4f&start=%d0101&end=%d1231&format=JSON'
           % (lon, lat, y, y))
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=240) as r:
                js = json.loads(r.read().decode('utf-8'))
            p = js['properties']['parameter']
            t2, td = p['T2M'], p['T2MDEW']
            rec = []
            for kk in sorted(t2.keys()):
                a = t2[kk]; b = td.get(kk, -999)
                if a > -900 and b > -900:
                    rec.append(('%s-%s-%sT%s:00' % (kk[0:4], kk[4:6], kk[6:8], kk[8:10]), a, b))
            return rec
        except Exception:
            time.sleep(3 + 4 * k)
    return None


def build_power(city, y0=Y0, y1=Y1):
    dst = CITY[city]['power']
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    lat, lon = CITY[city]['lat'], CITY[city]['lon']
    recs, done = [], set()
    if os.path.exists(dst):
        try:
            old = pd.read_csv(dst)
            for dt, a, b in old[['dt', 't', 'td']].values:
                recs.append([dt, a, b]); done.add(int(str(dt)[:4]))
        except Exception:
            pass
    for y in [y for y in range(y0, y1 + 1) if y not in done]:
        r = fetch_power_year(lat, lon, y)
        if r is None:
            P('  FAIL', city, y); continue
        recs.extend([[a, b, c] for a, b, c in r])
        P('  power ok', city, y, len(r))
    df = pd.DataFrame(recs, columns=['dt', 't', 'td']).drop_duplicates('dt')
    df['year'] = df['dt'].str[:4].astype(int)
    df = df[(df.year >= y0) & (df.year <= y1)]
    df.to_csv(dst, index=False)
    return df


def monthly_exposure(df, lat, lon):
    ds = df['dt'].values
    yr = np.array([int(s[:4]) for s in ds]); mo = np.array([int(s[5:7]) for s in ds])
    da = np.array([int(s[8:10]) for s in ds]); hh = np.array([int(s[11:13]) for s in ds])
    T = df['t'].values.astype(float); Td = df['td'].values.astype(float)
    T[(T < -80) | (T > 60)] = np.nan; Td[(Td < -90) | (Td > 40)] = np.nan
    ok = np.isfinite(T) & np.isfinite(Td)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    doy = cum[mo - 1] + da + np.where((mo > 2) & leap, 1, 0)
    el = solar_elevation_vec(lat, lon, doy[ok], hh[ok] + 0.5)
    idx = np.where(ok)[0]
    night = el < SUN
    inight, iday = idx[night], idx[~night]
    p = 1013.25
    Twn = tw_vec(T[inight], np.minimum(Td[inight], T[inight] - 1e-3), p)
    ynm, mnm = yr[inight], mo[inight]
    Tn = T[inight]
    # 局地分位数阈值（全记录夜间湿球的 p95 / p99）
    q95 = float(np.nanpercentile(Twn, 95)); q99 = float(np.nanpercentile(Twn, 99))
    out = []
    for y in range(int(yr.min()), int(yr.max()) + 1):
        for m in range(1, 13):
            s = (ynm == y) & (mnm == m)
            sd = (yr[iday] == y) & (mo[iday] == m)
            if s.sum() < 100:
                continue
            v = Twn[s]; tn = Tn[s]
            out.append(dict(ym='%d-%02d' % (y, m), year=y, month=m,
                            tw_night=float(np.nanmean(v)),
                            tn_night=float(np.nanmean(tn)),
                            tx_day=float(np.nanmean(T[iday][sd])) if sd.sum() > 100 else np.nan,
                            nwh219=float(np.nansum(v >= 21.9)),
                            nwh25=float(np.nansum(v >= 25.0)),
                            nwh26=float(np.nansum(v >= 26.0)),
                            nwh_p95=float(np.nansum(v >= q95)),
                            nwh_p99=float(np.nansum(v >= q99)),
                            hotnight28=float(np.nansum(tn >= 28.0)),
                            n_night_obs=int(s.sum())))
    dfx = pd.DataFrame(out)
    dfx.attrs['q95'], dfx.attrs['q99'] = q95, q99
    return dfx


# ---------------------------------------------------------------- 死亡
def load_deaths(city):
    f = CITY[city]['deaths']
    d = pd.read_csv(f)
    if city == 'HK':
        d = d[['year', 'month', 'deaths']].copy()
    else:
        d = d.rename(columns={'total': 'deaths'})
        d['ym'] = d['ym'].astype(str)
        d['year'] = d['ym'].str[:4].astype(int)
        d['month'] = d['ym'].str[5:7].astype(int)
        d = d[['year', 'month', 'deaths']]
    d['deaths'] = pd.to_numeric(d['deaths'], errors='coerce')
    d = d.dropna(subset=['deaths'])
    d['ym'] = d.year.astype(str) + '-' + d.month.astype(str).str.zfill(2)
    return d


# ---------------------------------------------------------------- 模型
def fit(d, expo, cov='HC1', covid_excl=True, months=None):
    import statsmodels.api as sm
    dd = d.copy()
    if months is not None:
        dd = dd[dd.month.isin(months)]
    if covid_excl:
        dd = dd[~((dd.year >= COVID[0]) & (dd.year <= COVID[1]))]
    cols = [e for e in expo if e in dd.columns]
    sub = dd.dropna(subset=cols)
    if len(sub) < 60:
        return {}, 0
    yv = pd.get_dummies(sub['year'], prefix='y', drop_first=True).astype(float)
    mv = pd.get_dummies(sub['month'], prefix='m', drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([sub[cols].astype(float), yv, mv], axis=1))
    m = sm.GLM(sub['deaths'].astype(float), X, family=sm.families.Poisson()).fit()
    try:
        m = m.get_robustcov_results(cov_type=cov)
    except Exception:
        pass
    par = np.asarray(m.params).ravel(); se = np.asarray(m.bse).ravel()
    nm = list(X.columns)
    res = {}
    for e in cols:
        i = nm.index(e)
        b, s_ = float(par[i]), float(se[i])
        if not np.isfinite(s_) or s_ <= 0:
            continue
        sd = float(sub[e].std())
        z = b / s_
        res[e] = dict(rr_unit=float(np.exp(b)), z=float(z),
                      p=float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))),
                      sd=sd, rr_sd=float(np.exp(b * sd)),
                      rr_sd_lo=float(np.exp((b - 1.96 * s_) * sd)),
                      rr_sd_hi=float(np.exp((b + 1.96 * s_) * sd)))
    return res, int(len(sub))


SPECS = {
    'A 夜间湿球均值': ['tw_night'],
    'B 夜间湿球 + 夜间干球 + 白天干球': ['tw_night', 'tn_night', 'tx_day'],
    'C NWH21.9 + 干球控制': ['nwh219', 'tn_night', 'tx_day'],
    'D NWH25.0 + 干球控制': ['nwh25', 'tn_night', 'tx_day'],
    'E NWH26.0 + 干球控制': ['nwh26', 'tn_night', 'tx_day'],
    'F NWH_p95 + 干球控制': ['nwh_p95', 'tn_night', 'tx_day'],
    'G NWH_p99 + 干球控制': ['nwh_p99', 'tn_night', 'tx_day'],
    'H 夜间干球≥28℃ + 湿球控制': ['hotnight28', 'tw_night', 'tx_day'],
}
KEY = {'A 夜间湿球均值': 'tw_night', 'B 夜间湿球 + 夜间干球 + 白天干球': 'tw_night',
       'C NWH21.9 + 干球控制': 'nwh219', 'D NWH25.0 + 干球控制': 'nwh25',
       'E NWH26.0 + 干球控制': 'nwh26', 'F NWH_p95 + 干球控制': 'nwh_p95',
       'G NWH_p99 + 干球控制': 'nwh_p99', 'H 夜间干球≥28℃ + 湿球控制': 'hotnight28'}


def city_report(city):
    P('\n' + '=' * 78)
    P('【%s】%s' % (city, CITY[city]['name']))
    P('=' * 78)
    wx = build_power(city)
    ex = monthly_exposure(wx, CITY[city]['lat'], CITY[city]['lon'])
    q95, q99 = ex.attrs['q95'], ex.attrs['q99']
    ex.to_csv(os.path.join(OUT, '%s_monthly_exposure.csv' % city.lower()), index=False)
    d = ex.merge(load_deaths(city)[['ym', 'deaths']], on='ym', how='inner')
    P('  面板 %d 月 (%s~%s)；夜间湿球月均 %.2f℃，局地 p95=%.2f / p99=%.2f ℃'
      % (len(d), d.ym.min(), d.ym.max(), d.tw_night.mean(), q95, q99))
    P('  NWH21.9 月均 %.1f h（饱和比例 %.1f%%）→ 绝对阈值可用性检验'
      % (d.nwh219.mean(), 100 * np.mean(d.nwh219 >= d.n_night_obs * 0.999)))
    warm = CITY[city]['warm']
    out = {}
    for tag, mo in (('暖季', warm), ('全年', None),
                    ('冷季', [m for m in range(1, 13) if m not in warm])):
        if city == 'SG' and tag in ('冷季',):
            continue
        out[tag] = {}
        P('  --- %s ---' % tag)
        for nm, exps in SPECS.items():
            r, n = fit(d, exps, months=mo)
            if not r:
                continue
            out[tag][nm] = dict(n=n, rr=r)
            s = '    %-28s n=%3d ' % (nm, n)
            for e, v in r.items():
                s += '| %s RR/SD=%.4f[%.4f,%.4f] z=%+.2f ' % (e, v['rr_sd'], v['rr_sd_lo'], v['rr_sd_hi'], v['z'])
            P(s)
    # 滞后（暖季，主规格 E）
    lag = {}
    dd = d[d.month.isin(warm)].copy()
    if city != 'SG' or True:
        dd = dd[~((dd.year >= COVID[0]) & (dd.year <= COVID[1]))]
        dd = dd.sort_values(['year', 'month'])
        for L in (1, 2):
            dd['l%d' % L] = dd['nwh26'].shift(L)
        for nm, exps in (('E+lag1', ['nwh26', 'l1', 'tn_night', 'tx_day']),
                         ('E+lag1+2', ['nwh26', 'l1', 'l2', 'tn_night', 'tx_day'])):
            r, n = fit(dd, exps, months=None, covid_excl=False)
            lag[nm] = dict(n=n, rr=r)
            P('    %-28s n=%3d ' % (nm, n) + ' '.join(
                '| %s z=%+.2f' % (e, v['z']) for e, v in r.items()))
    return dict(city=city, name=CITY[city]['name'], n=len(d), q95=q95, q99=q99,
                tw_night_mean=float(d.tw_night.mean()),
                nwh219_mean=float(d.nwh219.mean()),
                sat_frac=float(np.mean(d.nwh219 >= d.n_night_obs * 0.999)),
                specs=out, lag=lag,
                panel=d[['ym', 'deaths', 'tw_night', 'tn_night', 'tx_day',
                         'nwh219', 'nwh25', 'nwh26', 'nwh_p95', 'nwh_p99', 'hotnight28']])


def meta(reps, tag, spec, key):
    bs, ses = [], []
    for r in reps:
        v = r['specs'].get(tag, {}).get(spec, {}).get('rr', {}).get(key)
        if not v:
            continue
        bs.append(math.log(v['rr_sd']))
        ses.append((math.log(v['rr_sd_hi']) - math.log(v['rr_sd_lo'])) / (2 * 1.96))
    bs = np.array(bs); ses = np.array(ses)
    if len(bs) < 2:
        return None
    w = 1 / ses ** 2
    mu = float((w * bs).sum() / w.sum())
    Q = float((w * (bs - mu) ** 2).sum()); k = len(bs)
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (ses ** 2 + tau2)
    mur = float((wr * bs).sum() / wr.sum()); ser = float(math.sqrt(1 / wr.sum()))
    z = mur / ser
    return dict(k=k, per_city=[[float(x) for x in bs], [float(x) for x in ses]],
                fixed=dict(rr=float(np.exp(mu)), lo=float(np.exp(mu - 1.96 / math.sqrt(w.sum()))),
                           hi=float(np.exp(mu + 1.96 / math.sqrt(w.sum())))),
                random=dict(rr=float(np.exp(mur)), lo=float(np.exp(mur - 1.96 * ser)),
                            hi=float(np.exp(mur + 1.96 * ser)), z=float(z),
                            p=float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))),
                Q=Q, I2=float(I2), tau2=float(tau2))


def main():
    os.makedirs(OUT, exist_ok=True)
    reps = []
    for c in ['HK', 'SG']:
        try:
            reps.append(city_report(c))
        except Exception:
            import traceback
            P('CITY FAIL', c); traceback.print_exc()
    P('\n' + '=' * 78)
    P('【双城随机效应 meta】暖季主规格')
    P('=' * 78)
    metas = {}
    for spec in ['E NWH26.0 + 干球控制', 'D NWH25.0 + 干球控制',
                 'F NWH_p95 + 干球控制', 'B 夜间湿球 + 夜间干球 + 白天干球']:
        m = meta(reps, '暖季', spec, KEY[spec])
        if not m:
            continue
        metas[spec] = m
        P('  %-30s k=%d' % (spec, m['k']))
        for i, (b, se) in enumerate(zip(*m['per_city'])):
            P('      city%d RR/SD=%.4f [%.4f, %.4f]' %
              (i, math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se)))
        P('      固定 %.4f [%.4f,%.4f] | 随机 %.4f [%.4f,%.4f] z=%+.2f p=%.4g | I²=%.1f%%'
          % (m['fixed']['rr'], m['fixed']['lo'], m['fixed']['hi'],
             m['random']['rr'], m['random']['lo'], m['random']['hi'],
             m['random']['z'], m['random']['p'], m['I2']))
    json.dump(dict(cities=reps, meta=metas), open('twin_cities.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    for r in reps:
        r['panel'].to_csv('twin_cities_panel_%s.csv' % r['city'], index=False)
    P('\n已保存 twin_cities.json / twin_cities_panel_*.csv')


if __name__ == '__main__':
    main()
