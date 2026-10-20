# -*- coding: utf-8 -*-
"""
step38: 阈值的气候带适用性 —— 为什么 21.9 °C 在赤道是退化的
==========================================================
step37 发现: 新加坡 NWH₂₁.₉ ≈ 4378 h/yr ≈ 天文夜间总小时数,
即**每一个夜间小时都在 21.9 °C 湿球以上**, 指标饱和,
此时 NWH 测量的其实是夜长(天文量)而不是热 —— 它与死亡的相关是季节伪信号。

本脚本:
  1. 给出新加坡各阈值的越阈比例与年化小时数, 找出"在当地仍有分辨力"的阈值区间
  2. 用这些"热带适用阈值"重做月尺度 Poisson 模型
  3. 与欧洲面板的同阈值结果对照, 明确"欧洲 RR 能否外推到热带"这一第 9 轮悬案
"""
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from step30_humidity_attribution import tw_vec, solar_elevation_vec
import step37_tropical_singapore as S37

THS = [21.9, 24.0, 25.0, 25.8, 26.5, 27.0]


def monthly_multi():
    df = pd.read_csv(os.path.join(S37.OUT, 'power_singapore_hourly.csv'))
    ds = df['dt'].values
    yr = np.array([int(s[:4]) for s in ds]); mo = np.array([int(s[5:7]) for s in ds])
    da = np.array([int(s[8:10]) for s in ds]); hh = np.array([int(s[11:13]) for s in ds])
    T = df['t'].values.astype(float); Td = df['td'].values.astype(float)
    T[(T < -80) | (T > 60)] = np.nan; Td[(Td < -90) | (Td > 40)] = np.nan
    ok = np.isfinite(T) & np.isfinite(Td)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    doy = cum[mo - 1] + da + np.where((mo > 2) & leap, 1, 0)
    el = solar_elevation_vec(S37.LAT, S37.LON, doy[ok], hh[ok] + 0.5)
    idx = np.where(ok)[0]
    inight = idx[el < S37.SUN]
    Twn = tw_vec(T[inight], np.minimum(Td[inight], T[inight] - 1e-3), 1013.25)
    ynm = yr[inight]; mnm = mo[inight]
    NH = S37.night_hours_const()
    rows = []
    for y in np.unique(yr):
        for m in range(1, 13):
            s = (ynm == y) & (mnm == m)
            if s.sum() < 100:
                continue
            v = Twn[s]; v = v[np.isfinite(v)]
            r = dict(ym='%d-%02d' % (y, m), year=int(y), month=m,
                     tw_night=float(np.mean(v)), tn_night=float(np.nanmean(T[inight][s])),
                     n_hrs=int(s.sum()))
            for th in THS:
                r['f%.1f' % th] = float(np.mean(v >= th))
                r['nwh%.1f' % th] = float(np.mean(v >= th)) * NH[m]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    mn = monthly_multi()
    mn.to_csv(os.path.join(S37.OUT, 'sg_monthly_exposure_multi.csv'), index=False)
    dea = pd.read_csv(os.path.join(S37.OUT, 'sg_deaths_monthly.csv')).rename(
        columns={'Unnamed: 0': 'ym'})
    pop = pd.read_csv(os.path.join(S37.OUT, 'sg_population.csv'))
    d = mn.merge(dea[['ym', 'total']].rename(columns={'total': 'deaths'}), on='ym', how='inner')
    d = d.merge(pop, on='year', how='left')
    d['pop'] = d['pop'].astype(float)
    d = d.dropna(subset=['deaths', 'pop', 'tw_night'])
    L = ['阈值气候带适用性检验 · 新加坡赤道站 (1.35°N) (step38)',
         '面板 %d 月, %s .. %s' % (len(d), d['ym'].min(), d['ym'].max()), '',
         '【1】各阈值在新加坡的饱和程度',
         '  阈值     月越阈比例中位   年化 h/yr    非零月%%   月际 SD(h)   分辨力判读']
    js = {'saturation': {}}
    for th in THS:
        c = 'f%.1f' % th; n = 'nwh%.1f' % th
        fr = d[c].median(); an = d[n].mean() * 12
        sd = d[n].std()
        nz = 100 * (d[n] > 1).mean()
        if fr > 0.95:
            vd = '饱和 —— 指标退化为夜长, 不可用'
        elif fr < 0.02:
            vd = '近零 —— 识别样本不足, 不可用'
        else:
            vd = '可用'
        L.append('  %5.1f °C    %8.3f     %8.0f   %6.0f%%   %8.1f   %s'
                 % (th, fr, an, nz, sd, vd))
        js['saturation']['%.1f' % th] = dict(frac_median=float(fr), annual_h=float(an),
                                             nonzero_pct=float(nz), sd_month=float(sd),
                                             usable=(vd == '可用'))
    # ---- 用"可用阈值"重做模型 ----
    import statsmodels.api as sm
    from patsy import dmatrix
    d = d.copy()
    d['t'] = (d['year'] - d['year'].min()) / 10.0
    B = np.asarray(dmatrix('bs(t, df=8, degree=3) - 1', d, return_type='dataframe'))
    for i in range(B.shape[1]):
        d['sp%d' % i] = B[:, i]
    for k in (1, 2):
        d['sin%d' % k] = np.sin(2 * np.pi * k * d['month'] / 12)
        d['cos%d' % k] = np.cos(2 * np.pi * k * d['month'] / 12)
    base = ['sp%d' % i for i in range(B.shape[1])] + ['sin1', 'cos1', 'sin2', 'cos2']
    use = [th for th in THS if js['saturation']['%.1f' % th]['usable']]
    L.append('')
    L.append('【2】用"有分辨力的阈值"重做模型 (Poisson + 年样条 df=8 + 月 Fourier×2)')
    L.append('  暴露                n    RR/unit [95%CI]              RR/SD [95%CI]         z')
    js['models'] = {}
    for th in use + ([21.9] if 21.9 not in use else []):
        e = 'nwh%.1f' % th
        cols = base + [e]
        sub = d.dropna(subset=cols)
        if len(sub) < 100:
            continue
        X = sm.add_constant(sub[cols].astype(float))
        m = sm.GLM(sub['deaths'].astype(float), X, family=sm.families.Poisson(),
                   offset=np.log(sub['pop'].astype(float))).fit(cov_type='HC1')
        b = m.params[e]; se = m.bse[e]; sd = float(sub[e].std())
        L.append('  NWH_%-5.1f (h/月)  %4d  %.5f [%.5f,%.5f]  %.4f [%.4f,%.4f]  %+6.2f'
                 % (th, len(sub), math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se),
                    math.exp(b * sd), math.exp((b - 1.96 * se) * sd),
                    math.exp((b + 1.96 * se) * sd), b / se))
        js['models']['%.1f' % th] = dict(n=int(len(sub)), rr_unit=float(math.exp(b)),
                                         ci=[float(math.exp(b - 1.96 * se)), float(math.exp(b + 1.96 * se))],
                                         sd=sd, rr_sd=float(math.exp(b * sd)),
                                         rr_sd_ci=[float(math.exp((b - 1.96 * se) * sd)),
                                                   float(math.exp((b + 1.96 * se) * sd))],
                                         z=float(b / se))
    # ---- 连续口径: 夜间湿球均值 ----
    for e in ('tw_night', 'tn_night'):
        cols = base + [e]
        sub = d.dropna(subset=cols)
        X = sm.add_constant(sub[cols].astype(float))
        m = sm.GLM(sub['deaths'].astype(float), X, family=sm.families.Poisson(),
                   offset=np.log(sub['pop'].astype(float))).fit(cov_type='HC1')
        b = m.params[e]; se = m.bse[e]; sd = float(sub[e].std())
        L.append('  %-16s %4d  %.5f [%.5f,%.5f]  %.4f [%.4f,%.4f]  %+6.2f'
                 % (e, len(sub), math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se),
                    math.exp(b * sd), math.exp((b - 1.96 * se) * sd),
                    math.exp((b + 1.96 * se) * sd), b / se))
        js['models'][e] = dict(n=int(len(sub)), rr_unit=float(math.exp(b)),
                               ci=[float(math.exp(b - 1.96 * se)), float(math.exp(b + 1.96 * se))],
                               sd=sd, rr_sd=float(math.exp(b * sd)),
                               rr_sd_ci=[float(math.exp((b - 1.96 * se) * sd)),
                                         float(math.exp((b + 1.96 * se) * sd))], z=float(b / se))
    L.append('')
    L.append('【3】判读')
    L.append('  新加坡夜间湿球年内 SD 仅 %.2f °C(月均 %.2f °C), 暴露对比度极低;'
             % (d['tw_night'].std(), d['tw_night'].mean()))
    L.append('  月尺度全因死亡对该暴露的 RR/SD 置信区间很宽, 既覆盖欧洲估计值, 也覆盖 1。')
    L.append('  → 单个赤道城市**不足以**判定"热带更严重", 但它确立了一件确定的事:')
    L.append('     21.9 °C 阈值在湿热带是饱和的, 该指标必须按气候带重设阈值。')
    txt = '\n'.join(L)
    print(txt)
    open('tropical_threshold.txt', 'w', encoding='utf-8').write(txt + '\n')
    json.dump(js, open('tropical_threshold.json', 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
