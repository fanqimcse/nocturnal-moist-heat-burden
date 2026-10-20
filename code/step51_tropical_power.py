# -*- coding: utf-8 -*-
"""
step51 —— 湿热带亚年尺度检测的**功效**与**共线性**诊断（step50 的配套自检）
=====================================================================
step50 给出双城 meta：NWH25.0 暖季 RR/SD = 1.0096 [1.0027,1.0165] z=+2.72 p=0.0066 I²=0%。
但审稿人会立刻问两个问题，本步逐一回答：

Q1 **共线性**：HK 暖季把 tw_night / tn_night / tx_day 同时放进模型后，
    tn_night 系数变负（z=−4.90）——这是湿球与干球高度共线导致的符号翻转，
    不是真实结果。修正：把湿球暴露**对干球正交化**（取残差）后再入模。

Q2 **功效**：新加坡夜间湿球月均 SD 极小（赤道，全年 23.7–27.1℃），
    亚年尺度可识别的效应本就很小。若热带检测不到效应，
    到底是"没有效应"还是"没有功效"？必须量化。

输出：每城的可检测最小效应（MDE, 80% power）+ 正交化后的湿球效应 + 双城 meta。
"""
import os, json, math
import numpy as np
import pandas as pd
import statsmodels.api as sm

WARM = {'HK': list(range(5, 11)), 'SG': list(range(1, 13))}
COVID = (2020, 2022)


def P(*a):
    print(*a, flush=True)


def prep(city):
    d = pd.read_csv('twin_cities_panel_%s.csv' % city)
    d['month'] = d['ym'].str[5:7].astype(int)
    d['year'] = d['ym'].str[:4].astype(int)
    d = d[d.month.isin(WARM[city])]
    d = d[~((d.year >= COVID[0]) & (d.year <= COVID[1]))]
    return d.reset_index(drop=True)


def design(d, cols):
    yv = pd.get_dummies(d['year'], prefix='y', drop_first=True).astype(float)
    mv = pd.get_dummies(d['month'], prefix='m', drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[cols].astype(float), yv, mv], axis=1))
    return X


def glm(d, cols, target='deaths'):
    X = design(d, cols)
    m = sm.GLM(d[target].astype(float), X, family=sm.families.Poisson()).fit()
    try:
        m = m.get_robustcov_results(cov_type='HC1')
    except Exception:
        pass
    par = np.asarray(m.params).ravel(); se = np.asarray(m.bse).ravel()
    nm = list(X.columns)
    return {c: (float(par[nm.index(c)]), float(se[nm.index(c)])) for c in cols if c in nm}


def main():
    res = {}
    for city in ['HK', 'SG']:
        d = prep(city)
        P('\n' + '=' * 76)
        P('【%s】暖季 n=%d 月，月均死亡 %.0f' % (city, len(d), d.deaths.mean()))
        P('=' * 76)
        # ---- 暴露变异性 ----
        P('  夜间湿球月均 %.2f ℃  SD=%.3f ℃  极差 %.2f' %
          (d.tw_night.mean(), d.tw_night.std(), d.tw_night.max() - d.tw_night.min()))
        P('  NWH25.0 月均 %.1f h  SD=%.1f h' % (d.nwh25.mean(), d.nwh25.std()))
        # ---- MDE（80% power, 双侧 0.05 → 2.80）----
        r = glm(d, ['nwh25', 'tn_night', 'tx_day'])
        b, se = r['nwh25']
        sd = float(d.nwh25.std())
        mde_sd = math.exp(2.80 * se * sd)
        mde_unit = math.exp(2.80 * se)
        P('  NWH25.0：RR/SD=%.4f (z=%+.2f)  → 80%%功效下可检测的最小 RR/SD = %.4f'
          % (math.exp(b * sd), b / se, mde_sd))
        P('           即每增加 1 h 需达到 RR=%.5f 才能被检出（实测 %.5f）' % (mde_unit, math.exp(b)))
        # ---- 正交化：nwh25 对干球 + 年月FE 取残差 ----
        Xr = design(d, ['tn_night', 'tx_day'])
        yr = d['nwh25'].astype(float).values
        ols = sm.OLS(yr, Xr).fit()
        d['nwh25_orth'] = ols.resid
        P('  正交化：nwh25 对干球回归 R²=%.3f，残差 SD=%.1f h' %
          (ols.rsquared, d.nwh25_orth.std()))
        r2 = glm(d, ['nwh25_orth', 'tn_night', 'tx_day'])
        b2, se2 = r2['nwh25_orth']
        sd2 = float(d.nwh25_orth.std())
        z2 = b2 / se2
        P('  正交化湿球残差：RR/SD=%.4f [%.4f, %.4f]  z=%+.2f  p=%.4g' %
          (math.exp(b2 * sd2), math.exp((b2 - 1.96 * se2) * sd2),
           math.exp((b2 + 1.96 * se2) * sd2), z2,
           2 * (1 - 0.5 * (1 + math.erf(abs(z2) / math.sqrt(2))))))
        # 干球残差对照（把 hotnight28 对湿球正交化）
        Xr2 = design(d, ['tw_night', 'tx_day'])
        d['hot_orth'] = sm.OLS(d['hotnight28'].astype(float).values, Xr2).fit().resid
        r3 = glm(d, ['hot_orth', 'tw_night', 'tx_day'])
        b3, se3 = r3['hot_orth']; sd3 = float(d.hot_orth.std())
        P('  对照·正交化干球热夜(≥28℃)：RR/SD=%.4f  z=%+.2f' % (math.exp(b3 * sd3), b3 / se3))
        res[city] = dict(n=int(len(d)), deaths=float(d.deaths.mean()),
                         tw_sd=float(d.tw_night.mean()), tw_sd_v=float(d.tw_night.std()),
                         tw_range=float(d.tw_night.max() - d.tw_night.min()),
                         nwh25_mean=float(d.nwh25.mean()), nwh25_sd=float(d.nwh25.std()),
                         rr_sd=float(math.exp(b * sd)), z=float(b / se),
                         mde_sd=float(mde_sd),
                         orth_rr_sd=float(math.exp(b2 * sd2)), orth_z=float(z2),
                         orth_sd=float(sd2),
                         hot_orth_rr_sd=float(math.exp(b3 * sd3)), hot_orth_z=float(b3 / se3))

    # ---- 双城 meta（正交化后）----
    P('\n' + '=' * 76)
    P('【双城随机效应 meta】正交化夜间湿球（剔除干球共线部分）')
    P('=' * 76)
    bs = np.array([math.log(res[c]['orth_rr_sd']) for c in ['HK', 'SG']])
    from scipy.stats import norm
    ses = np.array([abs(math.log(res[c]['orth_rr_sd']) - 0) / abs(res[c]['orth_z']) for c in ['HK', 'SG']])
    w = 1 / ses ** 2
    mu = float((w * bs).sum() / w.sum())
    Q = float((w * (bs - mu) ** 2).sum()); k = 2
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (ses ** 2 + tau2)
    mur = float((wr * bs).sum() / wr.sum()); ser = float(math.sqrt(1 / wr.sum()))
    z = mur / ser
    P('  HK  RR/SD=%.4f (z=%+.2f) ; SG RR/SD=%.4f (z=%+.2f)' %
      (res['HK']['orth_rr_sd'], res['HK']['orth_z'], res['SG']['orth_rr_sd'], res['SG']['orth_z']))
    P('  随机效应 RR/SD = %.4f [%.4f, %.4f]  z=%+.2f  p=%.4g  I²=%.1f%%' %
      (math.exp(mur), math.exp(mur - 1.96 * ser), math.exp(mur + 1.96 * ser),
       z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), I2))
    res['meta_orth'] = dict(rr=float(math.exp(mur)), lo=float(math.exp(mur - 1.96 * ser)),
                            hi=float(math.exp(mur + 1.96 * ser)), z=float(z), I2=float(I2))
    json.dump(res, open('tropical_power.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    P('\n已保存 tropical_power.json')


if __name__ == '__main__':
    main()
