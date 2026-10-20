# -*- coding: utf-8 -*-
"""第 23 轮 R23(M90)  step52: 参考实现与不确定度预算 —— 标准化的前置条件

审稿意见: 任何可进入 ISO/GB 的指标, 标准文本第一章不是"定义", 是"测量方法与不确定度"。
本研究至今没有任何不确定度预算: 同一个 NWH, 换湿球公式 / 换夜间定义 / 换湿度输入 / 换时间
分辨率, 会差多少小时? 差多少就是标准必须规定的"参考实现"与"仪器要求"。

本步骤在两座湿热带城市(新加坡、中国香港)的 POWER 逐小时数据上做全因子敏感性:
  [A] 湿球公式: Stull(2011) 近似 / 干湿球(psychrometric)迭代 / 1/3 法则 / 露点精确解(参考)
  [B] 夜间定义: 太阳高度角 < −0.833°(本研究现行) / < −6°(民用曙暮光) / < −18°(天文)
                / 固定 22:00–06:00 / 固定 20:00–08:00
  [C] 湿度输入: 露点(精确) / 相对湿度(整百分点, 模拟常规站 RH 上报)
  [D] 时间分辨率: 逐小时 / 3 小时(线性插值) / 仅日最低-最高(传递函数近似)
  [E] 仪器误差: RH ±2%, ±5% ; Td ±0.5 ℃ ; T ±0.2 ℃

产出: reference_implementation.txt / .json
"""
import os, json, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TROP = os.path.join(BASE, 'ext', 'tropical')
P0 = 1013.25  # hPa, 标准大气压(标准须规定是否做气压订正)

TXT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


# ---------------- 湿度 / 湿球 ----------------
def es(t):
    """Bolton (1980) 饱和水汽压 hPa"""
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def rh_from_td(t, td):
    return 100.0 * np.clip(es(td) / es(t), 0.0, 1.0)


def td_from_rh(t, rh):
    e = np.clip(rh, 0.1, 100.0) / 100.0 * es(t)
    return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))


def tw_stull(t, rh):
    """Stull (2011) 经验湿球"""
    r = np.clip(rh, 0.0, 100.0)
    return (t * np.arctan(0.151977 * np.sqrt(r + 8.313659))
            + np.arctan(t + r) - np.arctan(r - 1.676331)
            + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r) - 4.686035)


def tw_psychro(t, e_target, p=P0):
    """干湿球方程 + 压力项, e_target 为实际水汽压(hPa); 二分求解 Tw"""
    lo, hi = np.minimum(t, -40.0), t.copy()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e_mid = es(mid) - 0.00066 * (1.0 + 0.00115 * mid) * p * (t - mid)
        bad = e_mid > e_target
        hi = np.where(bad, mid, hi)
        lo = np.where(bad, lo, mid)
    return 0.5 * (lo + hi)


def tw_third(t, td):
    return (t + 2.0 * td) / 3.0


def tw_all(t, td, rh=None):
    """返回 dict: 各公式的湿球 (℃)"""
    if rh is None:
        rh = rh_from_td(t, td)
    e_td = es(td)                      # 露点 -> 精确水汽压
    e_rh = np.clip(rh, 0.1, 100.0) / 100.0 * es(t)
    return {
        'ref_td': tw_psychro(t, e_td),                 # 参考实现: 露点 + 干湿球迭代
        'psychro_rh': tw_psychro(t, e_rh),
        'stull': tw_stull(t, rh),
        'third': tw_third(t, td),
    }


# ---------------- 太阳高度角 ----------------
def solar_alt(dt_utc, lat, lon):
    """dt_utc: pandas DatetimeIndex(UTC); 返回太阳高度角(度)"""
    doy = dt_utc.dayofyear.values.astype(float)
    h = dt_utc.hour.values + dt_utc.minute.values / 60.0
    fr = doy + h / 24.0
    gamma = 2.0 * np.pi / 365.0 * (fr - 1.0)
    dec = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
           - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
           - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    dec = np.degrees(dec)
    H = 15.0 * (h - 12.0) + lon          # 时角(度), 用经度直接修正 -> 与时区无关
    la, d, Hr = np.radians(lat), np.radians(dec), np.radians(H)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(Hr)))


# ---------------- NWH 计算 ----------------
def nwh_table(df, lat, lon, tw_col, night_rule, tz):
    """df: 需含 dt(UTC datetime) 与 tw_col; 返回 (年, NWH21.9, NWH25.0, 夜间小时数, 夜间Tw均值)"""
    alt = solar_alt(pd.DatetimeIndex(df['dt']), lat, lon)
    hh = pd.DatetimeIndex(df['dt']).hour.values + tz
    hh = np.where(hh >= 24, hh - 24, hh)
    if night_rule.startswith('alt'):
        thr = float(night_rule[3:])
        isn = alt < thr
    elif night_rule == 'fix2206':
        isn = (hh >= 22) | (hh < 6)
    elif night_rule == 'fix2008':
        isn = (hh >= 20) | (hh < 8)
    else:
        raise ValueError(night_rule)
    d = df.copy()
    d['_isn'] = isn
    d['_y'] = pd.DatetimeIndex(d['dt']).year
    g = d[d['_isn']].groupby('_y')
    out = pd.DataFrame({
        'nwh219': g[tw_col].apply(lambda x: float((x >= 21.9).sum())),
        'nwh250': g[tw_col].apply(lambda x: float((x >= 25.0).sum())),
        'nhrs': g[tw_col].size().astype(float),
        'twm': g[tw_col].mean(),
    })
    return out


def coarsen(df, k):
    """把逐小时降到 k 小时分辨率再线性插值回逐小时(模拟粗分辨率数据源)"""
    d = df.copy().reset_index(drop=True)
    idx = np.arange(len(d))
    keep = idx[::k]
    sub = d.iloc[keep]
    out = d.copy()
    for c in ['t', 'td']:
        out[c] = np.interp(idx, keep, sub[c].values)
    return out


# ---------------- 主流程 ----------------
def run(city, lat, lon, tz, fn):
    P('')
    P('=' * 78)
    P(f'【{city}】 lat={lat} lon={lon}')
    P('=' * 78)
    df = pd.read_csv(os.path.join(TROP, fn))
    df['dt'] = pd.to_datetime(df['dt'])
    P(f'  逐小时样本 {len(df):,} 行, {df.dt.min().date()} ~ {df.dt.max().date()}')

    tw = tw_all(df['t'].values, df['td'].values)
    for k, v in tw.items():
        df['tw_' + k] = v

    ref = 'tw_ref_td'
    P('')
    P('  ── 基准(参考实现: 露点+干湿球迭代, 夜间=太阳高度角<−0.833°, 逐小时) ──')
    base = nwh_table(df, lat, lon, ref, 'alt-0.833', tz)
    P(f'     夜间小时/年 {base.nhrs.mean():.0f} ; 夜间湿球均值 {base.twm.mean():.2f} ℃')
    P(f'     NWH21.9 = {base.nwh219.mean():.0f} h/yr ; NWH25.0 = {base.nwh250.mean():.0f} h/yr')

    res = {'city': city, 'lat': lat, 'lon': lon,
           'base': {'nhrs': float(base.nhrs.mean()), 'twm': float(base.twm.mean()),
                    'nwh219': float(base.nwh219.mean()), 'nwh250': float(base.nwh250.mean())},
           'formula': {}, 'night': {}, 'input': {}, 'resolution': {}, 'instrument': {}}

    # [A] 湿球公式
    P('')
    P('  [A] 湿球公式敏感性 (其它条件固定为参考实现)')
    P(f'     {"公式":<26}{"夜间Tw均值":>12}{"ΔTw(℃)":>10}{"NWH21.9":>10}{"Δ%":>9}{"NWH25.0":>10}{"Δ%":>9}')
    for k in ['ref_td', 'psychro_rh', 'stull', 'third']:
        tb = nwh_table(df, lat, lon, 'tw_' + k, 'alt-0.833', tz)
        d1 = 100 * (tb.nwh219.mean() - base.nwh219.mean()) / max(base.nwh219.mean(), 1e-9)
        d2 = 100 * (tb.nwh250.mean() - base.nwh250.mean()) / max(base.nwh250.mean(), 1e-9)
        P(f'     {k:<26}{tb.twm.mean():>12.3f}{tb.twm.mean()-base.twm.mean():>10.3f}'
          f'{tb.nwh219.mean():>10.0f}{d1:>8.1f}%{tb.nwh250.mean():>10.0f}{d2:>8.1f}%')
        res['formula'][k] = {'twm': float(tbm := tb.twm.mean()), 'dtw': float(tbm - base.twm.mean()),
                             'nwh219': float(tb.nwh219.mean()), 'd219_pct': float(d1),
                             'nwh250': float(tb.nwh250.mean()), 'd250_pct': float(d2)}

    # [B] 夜间定义
    P('')
    P('  [B] 夜间定义敏感性 (湿球用参考实现)')
    P(f'     {"定义":<26}{"夜间h/年":>10}{"Tw均值":>10}{"NWH21.9":>10}{"Δ%":>9}{"NWH25.0":>10}{"Δ%":>9}')
    for rule, lbl in [('alt-0.833', '天文夜 <−0.833°(现行)'), ('alt-6', '民用曙暮光 <−6°'),
                      ('alt-18', '天文完全黑 <−18°'), ('fix2206', '固定 22:00–06:00'),
                      ('fix2008', '固定 20:00–08:00')]:
        tb = nwh_table(df, lat, lon, ref, rule, tz)
        d1 = 100 * (tb.nwh219.mean() - base.nwh219.mean()) / max(base.nwh219.mean(), 1e-9)
        d2 = 100 * (tb.nwh250.mean() - base.nwh250.mean()) / max(base.nwh250.mean(), 1e-9)
        P(f'     {lbl:<26}{tb.nhrs.mean():>10.0f}{tb.twm.mean():>10.3f}'
          f'{tb.nwh219.mean():>10.0f}{d1:>8.1f}%{tb.nwh250.mean():>10.0f}{d2:>8.1f}%')
        res['night'][rule] = {'nhrs': float(tb.nhrs.mean()), 'twm': float(tb.twm.mean()),
                              'nwh219': float(tb.nwh219.mean()), 'd219_pct': float(d1),
                              'nwh250': float(tb.nwh250.mean()), 'd250_pct': float(d2)}

    # [C] 湿度输入: 露点(精确) vs RH(整百分点)
    P('')
    P('  [C] 湿度输入: 露点(精确) vs 相对湿度(取整到 1%, 模拟常规站上报)')
    rh = np.round(rh_from_td(df['t'].values, df['td'].values))
    d2 = df.copy()
    d2['tw_rh1'] = tw_psychro(df['t'].values, np.clip(rh, 0.1, 100) / 100 * es(df['t'].values))
    tb = nwh_table(d2, lat, lon, 'tw_rh1', 'alt-0.833', tz)
    e1 = 100 * (tb.nwh219.mean() - base.nwh219.mean()) / max(base.nwh219.mean(), 1e-9)
    e2 = 100 * (tb.nwh250.mean() - base.nwh250.mean()) / max(base.nwh250.mean(), 1e-9)
    P(f'     RH整百分点:  NWH21.9 {tb.nwh219.mean():.0f} h/yr  ({e1:+.1f}%) ; '
      f'NWH25.0 {tb.nwh250.mean():.0f} h/yr  ({e2:+.1f}%) ; ΔTw {tb.twm.mean()-base.twm.mean():+.3f} ℃')
    res['input']['rh_int_pct'] = {'d219_pct': float(e1), 'd250_pct': float(e2),
                                  'dtw': float(tb.twm.mean() - base.twm.mean())}

    # [D] 时间分辨率
    P('')
    P('  [D] 时间分辨率 (3 小时 / 6 小时, 线性插值回逐小时)')
    for k in [3, 6]:
        dc = coarsen(df, k)
        tc = tw_all(dc['t'].values, dc['td'].values)
        dc[ref] = tc['ref_td']
        tb = nwh_table(dc, lat, lon, ref, 'alt-0.833', tz)
        e1 = 100 * (tb.nwh219.mean() - base.nwh219.mean()) / max(base.nwh219.mean(), 1e-9)
        e2 = 100 * (tb.nwh250.mean() - base.nwh250.mean()) / max(base.nwh250.mean(), 1e-9)
        P(f'     {k} 小时:      NWH21.9 {tb.nwh219.mean():.0f} h/yr  ({e1:+.1f}%) ; '
          f'NWH25.0 {tb.nwh250.mean():.0f} h/yr  ({e2:+.1f}%) ; ΔTw {tb.twm.mean()-base.twm.mean():+.3f} ℃')
        res['resolution'][f'{k}h'] = {'d219_pct': float(e1), 'd250_pct': float(e2),
                                      'dtw': float(tb.twm.mean() - base.twm.mean())}

    # [E] 仪器误差传播
    P('')
    P('  [E] 仪器不确定度传播 (在参考实现上施加固定偏差)')
    P(f'     {"扰动":<26}{"ΔTw(℃)":>10}{"NWH21.9 Δ%":>13}{"NWH25.0 Δ%":>13}')
    perts = {
        'RH +2 个百分点': {'rh': +2.0},
        'RH −2 个百分点': {'rh': -2.0},
        'RH +5 个百分点': {'rh': +5.0},
        '露点 +0.5 ℃': {'td': +0.5},
        '露点 −0.5 ℃': {'td': -0.5},
        '干球 +0.2 ℃': {'t': +0.2},
        '气压 −30 hPa (1500m)': {'p': -30.0},
    }
    for lbl, pp in perts.items():
        t_ = df['t'].values + pp.get('t', 0.0)
        td_ = df['td'].values + pp.get('td', 0.0)
        rh_ = rh_from_td(t_, td_) + pp.get('rh', 0.0)
        p_ = P0 + pp.get('p', 0.0)
        twp = tw_psychro(t_, np.clip(rh_, 0.1, 100) / 100 * es(t_), p_)
        d3 = df.copy()
        d3[ref] = twp
        tb = nwh_table(d3, lat, lon, ref, 'alt-0.833', tz)
        e1 = 100 * (tb.nwh219.mean() - base.nwh219.mean()) / max(base.nwh219.mean(), 1e-9)
        e2 = 100 * (tb.nwh250.mean() - base.nwh250.mean()) / max(base.nwh250.mean(), 1e-9)
        P(f'     {lbl:<26}{tb.twm.mean()-base.twm.mean():>10.3f}{e1:>12.1f}%{e2:>12.1f}%')
        res['instrument'][lbl] = {'dtw': float(tb.twm.mean() - base.twm.mean()),
                                  'd219_pct': float(e1), 'd250_pct': float(e2)}
    return res


def main():
    out = {}
    out['singapore'] = run('新加坡 Singapore', 1.35, 103.82, 8, 'power_singapore_hourly.csv')
    out['hongkong'] = run('中国香港 Hong Kong', 22.32, 114.17, 8, 'power_hongkong_hourly.csv')

    # 综合不确定度预算
    P('')
    P('=' * 78)
    P('【合成标准不确定度】 (各分量的 NWH21.9 相对偏差, 方和根合成)')
    P('=' * 78)
    budget = {}
    for city in out:
        r = out[city]
        f = r['formula']
        comp = {
            '湿球公式(Stull vs 参考)': abs(f['stull']['d219_pct']),
            '夜间定义(曙暮光 vs 天文)': max(abs(r['night'][k]['d219_pct']) for k in ['alt-6', 'alt-18']),
            '夜间定义(固定钟点)': max(abs(r['night'][k]['d219_pct']) for k in ['fix2206', 'fix2008']),
            '湿度输入(RH整百分点)': abs(r['input']['rh_int_pct']['d219_pct']),
            '时间分辨率(3h)': abs(r['resolution']['3h']['d219_pct']),
            '仪器 RH ±2%': abs(r['instrument']['RH +2 个百分点']['d219_pct']),
            '仪器 露点 ±0.5℃': abs(r['instrument']['露点 +0.5 ℃']['d219_pct']),
        }
        uc = math.sqrt(sum(v ** 2 for v in comp.values()))
        budget[city] = {k: round(v, 2) for k, v in comp.items()}
        budget[city]['合成 u_c (RSS, %)'] = round(uc, 2)
        P('')
        P(f'  {city}:')
        for k, v in comp.items():
            P(f'     {k:<28}{v:>8.2f} %')
        P(f'     {"合成标准不确定度 u_c":<28}{uc:>8.2f} %')
    out['budget'] = budget

    with open(os.path.join(BASE, 'reference_implementation.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    with open(os.path.join(BASE, 'reference_implementation.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    P('')
    P('已写出 reference_implementation.txt / .json')


if __name__ == '__main__':
    main()
