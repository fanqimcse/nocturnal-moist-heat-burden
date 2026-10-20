# -*- coding: utf-8 -*-
"""第8轮 R8-1/R8-2: 欧洲 NUTS2 面板 —— 夜间湿球 vs 日最高干球的头对头死亡风险检验

数据:
  ext/health/deaths_weekly_nuts3.csv   Eurostat demo_r_mweek3 (NUTS3 x ISO周)
  ext/health/wx_nightly_nuts2.csv      ERA5(Open-Meteo) 逐夜指标, 已按本研究的
                                       湿球求解器 + 太阳高度角夜间定义计算
产出:
  health_er.txt / health_er.json
"""
import os, sys, json, math, datetime
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'ext', 'health')


def log(*a):
    print(*a, flush=True)


# ---------------- 载入 ----------------
def load():
    import datetime as _dt
    cent = pd.read_csv(os.path.join(OUT, 'nuts2_centroids.csv'))
    keep = set(cent['nuts2'])
    d = pd.read_csv(os.path.join(OUT, 'deaths_weekly_nuts3.csv'))
    d['nuts2'] = d['nuts3'].str[:4]
    d = d[d['nuts2'].isin(keep)]
    dw = d.groupby(['nuts2', 'week'], as_index=False)['deaths'].sum()
    # 解析 ISO 周 -> 该周周一
    def wk2date(w):
        y, wn = int(w[:4]), int(w[-2:])
        return _dt.date.fromisocalendar(y, wn, 1)
    dw['mon'] = dw['week'].map(wk2date)
    dw = dw.dropna(subset=['mon'])
    dw['year'] = dw['mon'].map(lambda x: x.year)
    # 区域-年 覆盖过滤: 夏季周(5-9月) 至少 18 周
    dw['month'] = dw['mon'].map(lambda x: x.month)
    cnt = dw[(dw['month'] >= 5) & (dw['month'] <= 9)].groupby(['nuts2', 'year']).size()
    good_ry = set(cnt[cnt >= 18].index)
    dw['ry'] = list(zip(dw['nuts2'], dw['year']))
    dw = dw[dw['ry'].isin(good_ry)].drop(columns=['ry', 'month'])

    w = pd.read_csv(os.path.join(OUT, os.environ.get('WXFILE', 'wx_power_nuts2.csv')))
    w['date'] = pd.to_datetime(w['date']).dt.date
    w['iso'] = w['date'].map(lambda x: x.isocalendar()[:2])
    w['isoyear'] = w['iso'].map(lambda t: t[0])
    w['isoweek'] = w['iso'].map(lambda t: t[1])
    w['week'] = w['isoyear'].astype(str) + '-W' + w['isoweek'].astype(str).str.zfill(2)
    # 周聚合: 夏季(5-9月)内的暴露
    w['f219'] = (w['tw_night_min'] > 21.9).astype(int)
    w['f240'] = (w['tw_night_min'] > 24.0).astype(int)
    agg = w.groupby(['nuts2', 'week']).agg(
        n_days=('tx', 'size'),
        tx_max=('tx', 'max'), tx_mean=('tx', 'mean'),
        tn_max=('tn', 'max'), tn_mean=('tn', 'mean'),
        tw_night_min_max=('tw_night_min', 'max'),     # 最糟一夜的"最低湿球" -> 恢复能力最差
        tw_night_min_mean=('tw_night_min', 'mean'),
        tw_night_mean_max=('tw_night_mean', 'max'),
        tw_night_mean_mean=('tw_night_mean', 'mean'),
        tw_day_max=('tw_day_max', 'max'),
        nh219=('nh219', 'sum'), nh240=('nh240', 'sum'), nh258=('nh258', 'sum'),
        n_tw219=('f219', 'sum'), n_tw240=('f240', 'sum'),
    ).reset_index()
    # 现行业务口径的夜间指标(对照用): 热带夜计数与干球度夜
    w['e_tn20'] = np.maximum(w['tn'] - 20.0, 0.0)
    w['e_tn25'] = np.maximum(w['tn'] - 25.0, 0.0)
    w['f_tn20'] = (w['tn'] >= 20.0).astype(int)
    w['f_tn25'] = (w['tn'] >= 25.0).astype(int)
    agg['nh_tn20'] = w.groupby(['nuts2', 'week'])['e_tn20'].sum().values
    agg['nh_tn25'] = w.groupby(['nuts2', 'week'])['e_tn25'].sum().values
    agg['n_tn20'] = w.groupby(['nuts2', 'week'])['f_tn20'].sum().values
    agg['n_tn25'] = w.groupby(['nuts2', 'week'])['f_tn25'].sum().values
    # 与干球"度夜"严格可比的湿球指标:
    #   度夜 = Σ_夜 max(0, 该夜最低湿球 − 阈值)   (夜间最低湿球 = 恢复能力上限)
    #   小时数 nh219 = 夜间"小时"计数(量纲不同, 仅作对照)
    for th, tag in [(21.9, '219'), (24.0, '240'), (25.8, '258')]:
        w['dn' + tag] = np.maximum(w['tw_night_min'] - th, 0.0)
        agg['dn' + tag] = w.groupby(['nuts2', 'week'])['dn' + tag].sum().values
    agg['nh219'] = agg.get('nh219', 0)
    # 配对夜: 取该周"最热一夜"(TN 最高)的干球与湿球, 避免两个指标来自不同夜晚
    w['_rk'] = w.groupby(['nuts2', 'week'])['tn'].rank(method='first', ascending=False)
    mt = w[w['_rk'] == 1][['nuts2', 'week', 'tn', 'tw_night_min']].rename(
        columns={'tn': 'tn_hot', 'tw_night_min': 'tw_hot'})
    agg = agg.merge(mt, on=['nuts2', 'week'], how='left')
    agg['deficit_hot'] = agg['tn_hot'] - agg['tw_hot']
    agg['mon'] = agg['week'].map(wk2date)
    m = dw.merge(agg, on=['nuts2', 'week'], how='inner', suffixes=('', '_x'))
    return m.dropna(subset=['deaths', 'tw_night_min_max', 'tx_max'])


# ---------------- 设计矩阵 ----------------
def design(df, exposures):
    """区域固定效应 + 年份因子 + Fourier(周) 季节 + 暴露项"""
    y = df['mon'].map(lambda d: d.year).values.astype(float)
    woy = df['mon'].map(lambda d: d.isocalendar()[1]).values.astype(float)
    n = len(df)
    years = sorted(set(y.astype(int)))
    cols = {}
    X = [np.ones(n)]
    names = ['const']
    for yy in years[1:]:
        X.append((y == yy).astype(float)); names.append('y%d' % yy)
    for k in range(1, 5):
        X.append(np.cos(2 * math.pi * k * woy / 52.0)); names.append('cos%d' % k)
        X.append(np.sin(2 * math.pi * k * woy / 52.0)); names.append('sin%d' % k)
    for e in exposures:
        X.append(df[e].values.astype(float)); names.append(e)
    return np.column_stack(X), names


def poisson_glm(X, y, maxiter=60):
    """IRLS Poisson, 返回 beta 与 稳健(准泊松)协方差"""
    n, p = X.shape
    b = np.zeros(p)
    b[0] = math.log(max(y.mean(), 1e-6))
    for it in range(maxiter):
        eta = X @ b
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-8)
        z = (y - mu) / w
        try:
            A = X.T @ (X * w[:, None])
            b_new = np.linalg.solve(A + 1e-10 * np.eye(p), X.T @ (w * z)) + b
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new; break
        b = b_new
    eta = np.clip(X @ b, -30, 30); mu = np.exp(eta)
    # 三明治(准泊松)方差
    try:
        A = X.T @ (X * mu[:, None])
        Ai = np.linalg.inv(A + 1e-10 * np.eye(p))
        meat = X.T @ (X * ((y - mu) ** 2)[:, None])
        V = Ai @ meat @ Ai
    except np.linalg.LinAlgError:
        V = np.full((p, p), np.nan)
    # 离散参数
    chi2 = np.sum((y - mu) ** 2 / np.maximum(mu, 1e-8))
    phi = chi2 / max(n - p, 1)
    return b, np.sqrt(np.diag(V)), phi


def fit_region(g, exposures):
    X, names = design(g, exposures)
    y = g['deaths'].values.astype(float)
    if len(g) < 100 or y.sum() < 200:
        return None
    b, se, phi = poisson_glm(X, y)
    if not np.all(np.isfinite(se)):
        return None
    out = {'nuts2': g['nuts2'].iloc[0], 'n': len(g), 'deaths': float(y.sum()), 'phi': float(phi)}
    for e in exposures:
        i = names.index(e)
        out[e] = float(b[i]); out[e + '_se'] = float(se[i])
    return out


def meta(results, key):
    """固定效应逆方差 + DerSimonian-Laird 随机效应"""
    b = np.array([r[key] for r in results])
    se = np.array([r[key + '_se'] for r in results])
    ok = np.isfinite(b) & np.isfinite(se) & (se > 0) & (se < 0.5)
    b, se = b[ok], se[ok]
    if len(b) == 0:
        return None
    w = 1.0 / se ** 2
    fe = float(np.sum(w * b) / np.sum(w)); fe_se = float(1 / math.sqrt(np.sum(w)))
    Q = float(np.sum(w * (b - fe) ** 2))
    dfQ = len(b) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfQ) / C) if C > 0 else 0.0
    w2 = 1.0 / (se ** 2 + tau2)
    re = float(np.sum(w2 * b) / np.sum(w2)); re_se = float(1 / math.sqrt(np.sum(w2)))
    return dict(n_regions=len(b), fe=fe, fe_se=fe_se, re=re, re_se=re_se,
                tau2=tau2, Q=Q, dfQ=dfQ,
                I2=float(max(0.0, (Q - dfQ) / Q)) if Q > 0 else 0.0)


def run():
    m = load()
    log('merged rows: %d   regions: %d' % (len(m), m['nuts2'].nunique()))
    log('years: %d-%d' % (m['mon'].map(lambda d: d.year).min(), m['mon'].map(lambda d: d.year).max()))
    # 只保留夏季完整周(>=5天) 且 死亡数合理
    m = m[m['n_days'] >= 5]
    log('after n_days>=5: %d rows, %d regions' % (len(m), m['nuts2'].nunique()))

    res = {}
    out_lines = []
    R = {}
    specs = {
        'A_tw_only': ['tw_night_min_max'],
        'B_tx_only': ['tx_max'],
        'C_headtohead': ['tw_night_min_max', 'tx_max'],
        'D_headtohead_lag': ['tw_night_min_max', 'tx_max', 'hw_persist'],
        'E_tn_only': ['tn_max'],
        'F_tw_vs_tn': ['tw_night_min_max', 'tn_max'],
        'K_matched_hot': ['tw_hot', 'tn_hot'],
        # 以下用于负担计算(与本研究台站指标量纲一致: 暖季夜间湿球均值)
        'G_twmean_only': ['tw_night_mean_mean'],
        'H_twmean_vs_tx': ['tw_night_mean_mean', 'tx_mean'],
        # 阈值型: 夜间最低湿球超阈的夜数(与 NRF 指标对应)
        'I_twnights219': ['n_tw219'],
        'J_twnights240': ['n_tw240'],
    }
    # 热浪持续性: 周内 tx 超过该区域 90 分位的天数
    thr = m.groupby('nuts2')['tx_max'].quantile(0.90)
    m['hw_persist'] = 0.0
    for k, g in m.groupby('nuts2'):
        pass
    m['_thr'] = m['nuts2'].map(thr)
    m['hot_day'] = (m['tx_mean'] > m['_thr']).astype(float)
    # 用前一周暴露代理持续性: 按 (region, year) 排序滞后
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    m['tx_lag'] = m.groupby('nuts2')['tx_mean'].shift(1)
    m['tw_lag'] = m.groupby('nuts2')['tw_night_min_mean'].shift(1)
    m['hw_persist'] = m.groupby('nuts2')['hot_day'].transform(
        lambda s: s.rolling(2, min_periods=1).sum())

    R['_n'] = dict(rows=len(m), regions=int(m['nuts2'].nunique()))
    subsets = {
        'ALL': m,
        'NO_COVID': m[m['mon'].map(lambda d: d.year) < 2020],
    }
    for sname, sub in subsets.items():
        if len(sub) == 0:
            continue
        for sp, exps in specs.items():
            if 'hw_persist' in exps:
                s2 = sub.dropna(subset=['hw_persist'])
            else:
                s2 = sub
            rr = []
            for nid, g in s2.groupby('nuts2'):
                r = fit_region(g, exps)
                if r:
                    rr.append(r)
            if len(rr) < 20:
                continue
            line = '== %s / %s : %d regions' % (sname, sp, len(rr))
            log(line); out_lines.append(line)
            key = '%s__%s' % (sname, sp)
            R[key] = {'n_regions': len(rr)}
            for e in exps:
                mm = meta(rr, e)
                if not mm:
                    continue
                R[key][e] = mm
                rr_ = math.exp(mm['re'])
                lo = math.exp(mm['re'] - 1.96 * mm['re_se'])
                hi = math.exp(mm['re'] + 1.96 * mm['re_se'])
                z = mm['re'] / mm['re_se'] if mm['re_se'] > 0 else float('nan')
                p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
                s = ('   %-18s RR/unit = %.4f  [%.4f, %.4f]  z=%+.2f  p=%.3g  I2=%.0f%%  (FE %.4f)'
                     % (e, rr_, lo, hi, z, p, mm['I2'] * 100, math.exp(mm['fe'])))
                log(s); out_lines.append(s)
    # 暴露单位说明
    unit = {
        'tw_night_min_max': 'per +1 C 周内"最差一夜"的最低湿球 (夜间恢复能力)',
        'tx_max': 'per +1 C 周内日最高干球 (现行预警依据)',
        'tn_max': 'per +1 C 周内日最低干球 (现行热夜依据)',
        'hw_persist': 'per +1 热日(前两周窗口)',
    }
    with open('health_er.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    R['_units'] = unit
    R['_desc'] = ('夏季(5-9月) NUTS2 x ISO周面板; 每区域 Poisson: 年份因子 + Fourier(周,k=1..4) 季节 '
                  '+ 暴露; 区域系数随机效应(DL)合并; 准泊松三明治标准误')
    with open('health_er.json', 'w', encoding='utf-8') as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    log('-> health_er.txt / health_er.json')


if __name__ == '__main__':
    run()
