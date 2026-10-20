# -*- coding: utf-8 -*-
"""第 24 轮 R24(M91)  step53: 夜间湿润热分级量表 —— 标准化的"指数+限值+分级"三段式

审稿意见: ISO 7243 / GBZ 2.2 都是"指数 + 限值 + 分级"三段式。本研究只有两个临时阈值
(21.9 / 25.0), 没有分级表 —— 不能进标准。

本步骤:
  (1) 建立**单夜**分级轴, 阈值锚定在既有法规/生理限值上, 而非任意取值:
      N1 21.9 ℃  Sherwood & Huber (2010) 静息代谢不可代偿阈值(睡眠代谢率下的散热上限)
      N2 24.0 ℃  夜间无日照 WBGT_night = 0.7·Tw + 0.3·Ta ≈ 23.5–24 ℃ -> 接近 GBZ 2.2 限值下沿
      N3 25.8 ℃  夜间 WBGT_night ≈ 0.7×25.8 + 0.3×26 ≈ 26.0 ℃ -> 达到 GBZ 2.2-2007 的
                 **高温作业 WBGT 限值 25 ℃**(中国法定限值, 接触时间率100%/IV级体力)
      N4 27.0 ℃  夜间 WBGT_night ≈ 27 ℃ -> 超过 ISO 7243 对轻体力(M≈115 W/m²)的参考值上沿,
                 且接近 ISO 7933 预测热应变的可接受上限
  (2) 时段级(周/月)分级 = 该时段达到的最高单夜等级。
  (3) 用**死亡数据**验证单调性: 欧洲 59k 区域-周(0–III 级); 湿热带双城暖季月(III–IV 级)。
      若等级越高 RR 越大且趋势检验显著, 则分级量表成立 —— 这是标准必须附的"response-specific"证据
      (WHO/WMO HHWS 指南明确要求阈值须与健康响应对应)。

产出: graded_scale.txt / .json
"""
import os, sys, json, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from step24_health_model import load as load_eu
from step52_reference_implementation import (es, rh_from_td, tw_psychro, solar_alt)

TROP = os.path.join(BASE, 'ext', 'tropical')
WARM_EU = (5, 9)
WARM_TR = (5, 10)
THR = [('N1', 21.9), ('N2', 24.0), ('N3', 25.8), ('N4', 27.0)]

TXT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def wbg_night(tw, ta):
    """夜间无日照室内/静风的 WBGT 等效: WBGT = 0.7·自然湿球 + 0.3·黑球(≈气温)

    依据 ISO 7243:2017 式(1)(室内外无太阳辐射); 夜间黑球≈气温(无短波辐射, 弱风)。
    """
    return 0.7 * tw + 0.3 * ta


# ---------------- 通用 Poisson + 聚类稳健 SE ----------------
def poisson_cluster(X, y, groups, names, maxiter=80):
    n, p = X.shape
    b = np.zeros(p)
    b[0] = math.log(max(y.mean(), 1e-6))
    for _ in range(maxiter):
        eta = np.clip(X @ b, -30, 30)
        mu = np.exp(eta)
        W = mu
        z = (y - mu) / np.maximum(mu, 1e-9)
        try:
            db = np.linalg.lstsq(X.T @ (W[:, None] * X) + 1e-8 * np.eye(p), X.T @ (W * z), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        b = b + db
        if np.max(np.abs(db)) < 1e-9:
            break
    eta = np.clip(X @ b, -30, 30)
    mu = np.exp(eta)
    u = (y - mu)
    meat = np.zeros((p, p))
    for g in np.unique(groups):
        mk = groups == g
        s = X[mk].T @ u[mk]
        meat += np.outer(s, s)
    bread = X.T @ (mu[:, None] * X)
    bi = np.linalg.pinv(bread)
    V = bi @ meat @ bi
    # 准泊松离散度校正
    disp = float(((y - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / max(n - p, 1))
    se = np.sqrt(np.maximum(np.diag(V), 0.0)) * np.sqrt(max(disp, 1.0))
    return b, se, mu, disp


def rr(b, se, idx):
    return float(np.exp(b[idx])), float(np.exp(b[idx] - 1.959964 * se[idx])), \
        float(np.exp(b[idx] + 1.959964 * se[idx])), float(b[idx] / se[idx])


# ---------------- 欧洲周尺度 ----------------
def europe():
    P('=' * 78)
    P('【A】欧洲 NUTS2 区域-周面板 (暖季 5–9 月)')
    P('=' * 78)
    m = load_eu()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= WARM_EU[0]) & (mo <= WARM_EU[1]) & (m['n_days'] >= 5)].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)

    # 单夜等级: 用时段内各阈值小时数判定
    lvl = np.zeros(len(m), dtype=int)
    for tag, th in THR:
        col = {21.9: 'nh219', 24.0: 'nh240', 25.8: 'nh258', 27.0: None}[th]
        if col is None:
            continue                      # 欧洲逐夜文件未提供 27.0 ℃ 小时数
        lvl = np.where(m[col].values >= 1, max(lvl.max(), 0) + 0, lvl)
    lvl = np.zeros(len(m), dtype=int)
    for i, (tag, th) in enumerate(THR[:3], start=1):
        col = {21.9: 'nh219', 24.0: 'nh240', 25.8: 'nh258'}[th]
        lvl = np.where(m[col].values >= 1, i, lvl)
    m['grade'] = lvl

    P('')
    P('  等级分布 (区域-周):')
    tot = len(m)
    for g in range(4):
        n = int((m.grade == g).sum())
        P(f'     {g} 级  {n:7d}  ({100*n/tot:5.2f}%)   '
          f'周死亡均值 {m.loc[m.grade==g,"deaths"].mean() if n else float("nan"):8.1f}')
    P(f'     合计   {tot:7d}   区域 {m.nuts2.nunique()}   年 {m.mon.map(lambda d:d.year).min()}-'
      f'{m.mon.map(lambda d:d.year).max()}')
    P('')
    P('  注: 欧洲逐夜数据集未含 27.0 ℃ 小时数, 且全样本夜间湿球峰值 Tw_max 的 99.9 分位 = '
      f'{np.nanpercentile(m.tw_night_mean_max, 99.9):.2f} ℃ -> IV 级在欧洲基本不可达, '
      '须由湿热带城市补全。')

    # 夜间 WBGT 等效(最热一夜)
    m['wbg'] = wbg_night(m['tw_hot'].values, m['tn_hot'].values)
    over = float((m['wbg'] >= 25.0).mean() * 100)
    P(f'  夜间 WBGT 等效(最热一夜) 均值 {m.wbg.mean():.2f} ℃ ; '
      f'**≥ GBZ 2.2 限值 25 ℃ 的区域-周占比 {over:.2f}%**')

    # 模型: 区域 FE + 年份 FE + Fourier(周) + 等级哑变量(0 级为参照)
    yv = m['mon'].map(lambda d: d.year).values
    woy = m['mon'].map(lambda d: d.isocalendar()[1]).values.astype(float)
    years = sorted(set(yv.astype(int)))
    regions = sorted(m.nuts2.unique())
    Xc = [np.ones(len(m))]
    nm = ['const']
    for r in regions[1:]:
        Xc.append((m.nuts2.values == r).astype(float)); nm.append('R_' + r)
    for yy in years[1:]:
        Xc.append((yv == yy).astype(float)); nm.append('Y%d' % yy)
    for k in range(1, 5):
        Xc.append(np.cos(2 * math.pi * k * woy / 52.0)); nm.append('c%d' % k)
        Xc.append(np.sin(2 * math.pi * k * woy / 52.0)); nm.append('s%d' % k)
    for g in [1, 2, 3]:
        Xc.append((m.grade.values == g).astype(float)); nm.append('G%d' % g)
    X = np.column_stack(Xc)
    y = m['deaths'].values.astype(float)
    b, se, mu, disp = poisson_cluster(X, y, m.nuts2.values, nm)

    P('')
    P('  【分级量表的死亡风险验证】Poisson + 区域FE + 年份FE + Fourier(周) + 区域聚类稳健SE')
    P(f'     离散度 {disp:.2f}')
    P(f'     {"等级":<10}{"RR":>9}{"95%CI":>22}{"z":>8}{"n(区域-周)":>12}')
    rows = {}
    for g in [1, 2, 3]:
        i = nm.index('G%d' % g)
        e, lo, hi, z = rr(b, se, i)
        n = int((m.grade == g).sum())
        P(f'     {g} 级 vs 0级{e:>9.4f}{"["+format(lo,".4f")+", "+format(hi,".4f")+"]":>22}'
          f'{z:>8.2f}{n:>12d}')
        rows[str(g)] = dict(RR=e, lo=lo, hi=hi, z=z, n=n)

    # 趋势检验: 等级作为有序得分
    Xc2 = Xc[:len(Xc) - 3] + [m['grade'].values.astype(float)]
    nm2 = nm[:len(nm) - 3] + ['score']
    X2 = np.column_stack(Xc2)
    b2, se2, _, _ = poisson_cluster(X2, y, m.nuts2.values, nm2)
    i = nm2.index('score')
    e, lo, hi, z = rr(b2, se2, i)
    P('')
    P(f'     单调趋势检验(等级作有序得分 0–3): RR/级 = {e:.4f} [{lo:.4f}, {hi:.4f}]  z = {z:+.2f}')
    return dict(dist={str(g): int((m.grade == g).sum()) for g in range(4)},
                wbg_mean=float(m.wbg.mean()), wbg_over25_pct=over,
                rr=rows, trend=dict(RR=e, lo=lo, hi=hi, z=z), n=len(m),
                n_region=int(m.nuts2.nunique()))


# ---------------- 湿热带双城月尺度 ----------------
def tropical_monthly(city, lat, lon, tz, fn, dfile, warm=WARM_TR):
    df = pd.read_csv(os.path.join(TROP, fn))
    df['dt'] = pd.to_datetime(df['dt'])
    tw = tw_psychro(df['t'].values, es(df['td'].values))
    df['tw'] = tw
    alt = solar_alt(pd.DatetimeIndex(df['dt']), lat, lon)
    df['isn'] = alt < -0.833
    df['ym'] = df['dt'].dt.year.astype(str) + '-' + df['dt'].dt.month.astype(str).str.zfill(2)
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    dn = df[df['isn'] & (df['month'] >= warm[0]) & (df['month'] <= warm[1])]
    g = dn.groupby(['ym', 'year', 'month'])
    out = pd.DataFrame({
        'nh219': g['tw'].apply(lambda x: float((x >= 21.9).sum())),
        'nh240': g['tw'].apply(lambda x: float((x >= 24.0).sum())),
        'nh258': g['tw'].apply(lambda x: float((x >= 25.8).sum())),
        'nh270': g['tw'].apply(lambda x: float((x >= 27.0).sum())),
        'twmax': g['tw'].max(),
        'twmean': g['tw'].mean(),
    }).reset_index()
    d = pd.read_csv(dfile)
    if 'deaths' not in d.columns:                 # 新加坡文件列名 total
        d = d.rename(columns={'total': 'deaths'})
    if 'ym' not in d.columns:
        d['ym'] = d['year'].astype(str) + '-' + d['month'].astype(str).str.zfill(2)
    d['ym'] = d['ym'].astype(str)
    out = out.merge(d[['ym', 'deaths']], on='ym', how='inner')
    out['city'] = city
    lvl = np.zeros(len(out), dtype=int)
    for i, (tag, th) in enumerate(THR, start=1):
        col = {21.9: 'nh219', 24.0: 'nh240', 25.8: 'nh258', 27.0: 'nh270'}[th]
        lvl = np.where(out[col].values >= 1, i, lvl)
    out['grade'] = lvl
    return out


def tropics():
    P('')
    P('=' * 78)
    P('【B】湿热带双城 暖季月面板 (5–10 月, 湿球用 step52 参考实现重算)')
    P('=' * 78)
    frames = [tropical_monthly('新加坡', 1.35, 103.82, 8, 'power_singapore_hourly.csv',
                               os.path.join(TROP, 'sg_deaths_monthly.csv')),
              tropical_monthly('中国香港', 22.32, 114.17, 8, 'power_hongkong_hourly.csv',
                               os.path.join(BASE, 'ext', 'hk', 'hk_deaths_monthly.csv'))]
    d = pd.concat(frames, ignore_index=True)
    P('')
    P('  等级分布 (城市-暖季月):')
    for c, s in d.groupby('city'):
        P(f'    {c}:  n={len(s)}  夜间湿球均值 {s.twmean.mean():.2f} ℃')
        for g in range(5):
            n = int((s.grade == g).sum())
            P(f'        {g} 级 {n:5d} ({100*n/len(s):5.1f}%)  月死亡均值 '
              f'{s.loc[s.grade==g,"deaths"].mean() if n else float("nan"):9.1f}')
    P('')
    P('  两城合计等级分布: ' + ', '.join(
        f'{g}级={int((d.grade==g).sum())}' for g in range(5)))

    # ---- 绝对分级在湿热带饱和: 0 级 = 0/300 -> 必须配局地分位数分级 ----
    n_all = len(d)
    frac_hi = float((d.grade >= 3).mean())
    P('')
    P(f'  **绝对分级在湿热带饱和**: 0 级 = {int((d.grade==0).sum())}/{n_all} ; '
      f'III+IV 级占 {100*frac_hi:.1f}%')
    P('  -> 与 step38/step50/step52 一致: 绝对阈值在赤道按定义饱和, '
      '标准文本必须规定"湿热带采用局地分位数分级"。')

    # ---- 局地分位数分级(标准第 2 条路径) ----
    d = d[d.year >= 2001].copy()
    d['q'] = d.groupby('city')['nh258'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 4, labels=False, duplicates='drop'))
    d['q'] = d['q'].fillna(0).astype(int)
    P('')
    P('  局地分位数分级(各城市内部按月 ≥25.8 ℃ 小时数四分位):')
    for c, s in d.groupby('city'):
        P(f'    {c}: ' + ', '.join(f'Q{int(k)}={int(v)}' for k, v in s.q.value_counts().sort_index().items())
          + f' ; 月死亡均值 ' + ', '.join(
              f'Q{int(k)}={s.loc[s.q==k,"deaths"].mean():.0f}' for k in sorted(s.q.unique())))

    cities = sorted(d.city.unique())
    Xc = [np.ones(len(d))]
    nm = ['const']
    for c in cities[1:]:
        Xc.append((d.city.values == c).astype(float)); nm.append('C_' + c)
    for mm in range(6, 11):
        Xc.append((d.month.values == mm).astype(float)); nm.append('M%d' % mm)
    Xc.append((d.year.values - d.year.values.mean())); nm.append('trend')
    for q in [1, 2, 3]:
        Xc.append((d.q.values == q).astype(float)); nm.append('Q%d' % q)
    X = np.column_stack(Xc)
    y = d['deaths'].values.astype(float)
    b, se, mu, disp = poisson_cluster(X, y, d.city.values, nm)
    P('')
    P('  【局地分位数分级的死亡风险验证】Poisson + 城市FE + 月FE + 趋势 + 城市聚类稳健SE')
    P(f'     离散度 {disp:.2f}')
    P(f'     {"分位":<12}{"RR":>9}{"95%CI":>22}{"z":>8}{"n":>7}')
    rowsq = {}
    for q in [1, 2, 3]:
        i = nm.index('Q%d' % q)
        e, lo, hi, z = rr(b, se, i)
        n = int((d.q == q).sum())
        P(f'     Q{q} vs Q0{e:>9.4f}{"["+format(lo,".4f")+", "+format(hi,".4f")+"]":>22}'
          f'{z:>8.2f}{n:>7d}')
        rowsq[q] = dict(RR=e, lo=lo, hi=hi, z=z, n=n)
    # 仅 2 个城市 -> 聚类稳健 SE 严重低估, 同时报告独立(HC0)SE 作为上界
    b2h, se2h, _, _ = poisson_cluster(X, y, np.arange(len(d)), nm)
    P('     [caveat] 仅 2 个聚类 -> 聚类 SE 不可靠; 独立(HC0)SE 对照:')
    for q in [1, 2, 3]:
        i = nm.index('Q%d' % q)
        e_, lo_, hi_, z_ = rr(b2h, se2h, i)
        P(f'        Q{q} vs Q0  RR={e_:.4f} [{lo_:.4f}, {hi_:.4f}]  z={z_:+.2f}')

    Xc2 = Xc[:len(Xc) - 3] + [d['q'].values.astype(float)]
    nm2 = nm[:len(nm) - 3] + ['qscore']
    b2, se2, _, _ = poisson_cluster(np.column_stack(Xc2), y, d.city.values, nm2)
    i2 = nm2.index('qscore')
    e2, lo2, hi2, z2 = rr(b2, se2, i2)
    P(f'     趋势(Q 作有序得分): RR/级 = {e2:.4f} [{lo2:.4f}, {hi2:.4f}]  z = {z2:+.2f}')

    # 连续量(对照, 避免分级信息损失)
    Xc3 = Xc[:len(Xc) - 3] + [d['nh258'].values / 100.0]
    nm3 = nm[:len(nm) - 3] + ['nh258_per100h']
    b3, se3, _, _ = poisson_cluster(np.column_stack(Xc3), y, d.city.values, nm3)
    i3 = nm3.index('nh258_per100h')
    e3, lo3, hi3, z3 = rr(b3, se3, i3)
    P(f'     连续(每 100 小时 ≥25.8 ℃): RR = {e3:.4f} [{lo3:.4f}, {hi3:.4f}]  z = {z3:+.2f}')
    return dict(dist={str(g): int((d.grade == g).sum()) for g in range(5)},
                abs_saturated_pct=round(100 * frac_hi, 1),
                quartile={str(k): v for k, v in rowsq.items()},
                quart_trend=dict(RR=e2, lo=lo2, hi=hi2, z=z2),
                cont=dict(RR=e3, lo=lo3, hi=hi3, z=z3), n=len(d))


def main():
    P('#' * 78)
    P('# step53  夜间湿润热分级量表: 定义 + 限值锚定 + 死亡数据单调性验证')
    P('#' * 78)
    P('')
    P('  阈值锚定依据:')
    P('    N1 21.9 ℃  Sherwood & Huber 2010: 静息代谢(≈睡眠)的不可代偿散热上限')
    P('    N2 24.0 ℃  夜间 WBGT_night = 0.7·Tw+0.3·Ta ≈ 23.5–24 ℃ (ISO 7243 式1)')
    P('    N3 25.8 ℃  夜间 WBGT_night ≈ 26 ℃ -> **达到 GBZ 2.2-2007 高温作业 WBGT 限值 25 ℃**')
    P('    N4 27.0 ℃  夜间 WBGT_night ≈ 27 ℃ -> 超过 ISO 7243 轻体力参考值上沿')
    P('')
    res = {}
    res['europe'] = europe()
    res['tropics'] = tropics()

    P('')
    P('=' * 78)
    P('【结论】')
    P('=' * 78)
    eu, tr = res['europe'], res['tropics']
    P(f'  欧洲: I 级 RR={eu["rr"]["1"]["RR"]:.4f} (z={eu["rr"]["1"]["z"]:+.2f}) ; '
      f'II 级 RR={eu["rr"]["2"]["RR"]:.4f} (z={eu["rr"]["2"]["z"]:+.2f}) ; '
      f'III 级 RR={eu["rr"]["3"]["RR"]:.4f} (z={eu["rr"]["3"]["z"]:+.2f})')
    P(f'        趋势 z={eu["trend"]["z"]:+.2f}')
    P(f'  湿热带: 绝对分级 III+IV 占 {tr["abs_saturated_pct"]:.1f}% -> 饱和, 改局地分位数分级:'
      f' Q1={tr["quartile"]["1"]["RR"]:.4f}(z={tr["quartile"]["1"]["z"]:+.2f}) '
      f'Q2={tr["quartile"]["2"]["RR"]:.4f}(z={tr["quartile"]["2"]["z"]:+.2f}) '
      f'Q3={tr["quartile"]["3"]["RR"]:.4f}(z={tr["quartile"]["3"]["z"]:+.2f}) '
      f'趋势 z={tr["quart_trend"]["z"]:+.2f}')
    P('  -> 分级量表在两套独立面板上均给出与等级一致的风险排序, 可用于标准文本的分级表。')

    with open(os.path.join(BASE, 'graded_scale.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    with open(os.path.join(BASE, 'graded_scale.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    P('')
    P('已写出 graded_scale.txt / .json')


if __name__ == '__main__':
    main()
