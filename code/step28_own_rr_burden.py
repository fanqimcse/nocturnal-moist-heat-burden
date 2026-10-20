# -*- coding: utf-8 -*-
"""第8轮 R8-6: 用本研究自身建立的暴露-反应关系重算全球归因负担

负担 = 基线死亡 x (exp(beta * dTw) - 1)
  beta : 来自欧洲 NUTS2 面板的 Poisson 暴露-反应 (每 +1 C 暖季夜间湿球)
  dTw  : 本研究 176 站观测到的夜间湿球上升 (人口加权)
产出: own_rr_burden.txt / .json
"""
import os, json, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CDR = 0.0076          # 全球粗死亡率 (2023, 世界银行), 敏感性 0.006-0.010


def theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); sl = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(sl)) if sl else float('nan')


def obs_dtw():
    """人口加权的夜间湿球上升: 两套口径"""
    m = pd.read_csv('metrics_v3.csv')
    m = m[m['ok_trend'].astype(str) == '1']
    pop = pd.read_csv('station_population.csv')[['station', 'pop_50km']]
    m = m.merge(pop, on='station', how='left')
    m['pop_50km'] = m['pop_50km'].fillna(0)
    out = {}
    # (1) 全年夜间湿球均值趋势 (1976-2023)
    tr = {}
    for sid, g in m.groupby('station'):
        g = g.sort_values('year')
        if len(g) >= 20:
            tr[sid] = theil_sen(g.year.values, g.tw_night_mean.values)
    t = pd.Series(tr)
    sub = m.groupby('station').first().reset_index()
    sub['tw_slope'] = sub.station.map(t)
    sub = sub.dropna(subset=['tw_slope'])
    w = sub.pop_50km.values
    out['annual_median_slope'] = float(np.median(t.values))
    out['annual_pop_weighted_slope'] = float(np.sum(w * sub.tw_slope.values) / np.sum(w))
    out['n_stations'] = int(len(sub))
    # 人口基数: 全部合格站(不要求有趋势) —— 与摘要的 9.01 亿口径一致
    allst = m.groupby('station').first().reset_index()
    out['pop_total'] = float(allst['pop_50km'].sum())
    out['n_stations_all'] = int(len(allst))
    # (2) 暖季夜间湿球趋势 (obs_monthly, 1976-2014)
    if os.path.exists('obs_monthly.csv'):
        o = pd.read_csv('obs_monthly.csv')
        warm = {}
        for sid, g in o.groupby(['station', 'month'])['t_mean'].mean().reset_index().groupby('station'):
            warm[sid] = list(g.sort_values('t_mean', ascending=False).month[:3])
        tr2 = {}
        for sid, g in o.groupby('station'):
            mo = warm.get(sid, [6, 7, 8])
            gg = g[g.month.isin(mo)].groupby('year')['tw_night'].mean()
            if len(gg) >= 20:
                tr2[sid] = theil_sen(gg.index.values, gg.values)
        t2 = pd.Series(tr2)
        s2 = sub.copy()
        s2['s'] = s2.station.map(t2)
        s2 = s2.dropna(subset=['s'])
        w2 = s2.pop_50km.values
        out['warm_season_median_slope'] = float(np.median(t2.values))
        out['warm_season_pop_weighted_slope'] = float(np.sum(w2 * s2.s.values) / np.sum(w2))
        out['n_stations_warm'] = int(len(s2))
    return out


def main():
    er = json.load(open('health_er.json'))
    dt = obs_dtw()
    lines = []
    lines.append('用本研究自身暴露-反应关系重算的全球归因负担')
    lines.append('')
    lines.append('一、暴露变化 (本研究 176 站观测, 人口加权)')
    lines.append('  全年夜间湿球均值趋势   : %+.4f °C/年 (中位 %+.4f), %d 站, 覆盖人口 %.2f 亿'
                 % (dt['annual_pop_weighted_slope'], dt['annual_median_slope'],
                    dt['n_stations'], dt['pop_total'] / 1e8))
    if 'warm_season_pop_weighted_slope' in dt:
        lines.append('  暖季夜间湿球趋势       : %+.4f °C/年 (中位 %+.4f), %d 站 [1976-2014]'
                     % (dt['warm_season_pop_weighted_slope'], dt['warm_season_median_slope'],
                        dt['n_stations_warm']))
    span = 2023 - 1976
    dTw = dt['annual_pop_weighted_slope'] * span
    lines.append('  1976-2023 累计上升 dTw : %+.3f °C (人口加权全年口径)' % dTw)
    lines.append('')
    lines.append('二、暴露-反应系数 (欧洲 NUTS2 面板, Poisson + 区域随机效应合并)')
    res = {}
    for key in ['NO_COVID__G_twmean_only', 'ALL__G_twmean_only', 'NO_COVID__H_twmean_vs_tx']:
        if key not in er:
            continue
        for var, mm in er[key].items():
            if not isinstance(mm, dict) or 're' not in mm:
                continue
            lines.append('  %s [%s]: beta = %+.5f ± %.5f /°C  -> RR = %.4f [%.4f, %.4f]  (n=%d 区域, I2=%.0f%%)'
                         % (key, var, mm['re'], mm['re_se'], math.exp(mm['re']),
                            math.exp(mm['re'] - 1.96 * mm['re_se']),
                            math.exp(mm['re'] + 1.96 * mm['re_se']),
                            mm['n_regions'], mm['I2'] * 100))
            res[key + '|' + var] = mm
    lines.append('')
    lines.append('三、归因负担 (基线死亡 = 覆盖人口 x 粗死亡率 %.1f‰)' % (CDR * 1000))
    D0 = dt['pop_total'] * CDR
    lines.append('  覆盖人口 %.2f 亿 -> 基线年死亡 %.1f 万' % (dt['pop_total'] / 1e8, D0 / 1e4))
    out = {'dTw': dTw, 'pop': dt['pop_total'], 'CDR': CDR, 'baseline_deaths': D0, 'exposure': dt}
    for k, mm in res.items():
        b = mm['re']; lo = mm['re'] - 1.96 * mm['re_se']; hi = mm['re'] + 1.96 * mm['re_se']
        for lab, bb in [('点估计', b), ('下界', lo), ('上界', hi)]:
            inc = D0 * (math.exp(bb * dTw) - 1.0)
            lines.append('  %s [%s]: RR(per dTw) = %.4f -> 归因死亡增量 = %.1f 万/年 (占基线 %.2f%%)'
                         % (k, lab, math.exp(bb * dTw), inc / 1e4, 100 * (math.exp(bb * dTw) - 1)))
        out['burden_' + k] = dict(beta=b, lo=lo, hi=hi,
                                  attributable=float(D0 * (math.exp(b * dTw) - 1)),
                                  attributable_lo=float(D0 * (math.exp(lo * dTw) - 1)),
                                  attributable_hi=float(D0 * (math.exp(hi * dTw) - 1)))
    # 未来情景 (用第7轮标度区间)
    scal = [(0.795, 1.557)]     # 夜间湿球 °C 每 °C GMST (下界, 上界)
    scen = {'SSP1-2.6': 0.41, 'SSP2-4.5': 1.31, 'SSP3-7.0': 2.21, 'SSP5-8.5': 3.01}
    lines.append('')
    lines.append('四、未来情景下的额外负担 (夜间湿球标度 0.795-1.557 °C/°C GMST)')
    for s, dg in scen.items():
        for k, mm in res.items():
            if 'tw_night_mean_mean' not in k:
                continue
            b = mm['re']
            a1 = D0 * (math.exp(b * scal[0][0] * dg) - 1)
            a2 = D0 * (math.exp(b * scal[0][1] * dg) - 1)
            lines.append('  %s (+%.2f °C): 夜间湿球再升 %.2f-%.2f °C -> 额外归因死亡 %.1f-%.1f 万/年'
                         % (s, dg, scal[0][0] * dg, scal[0][1] * dg, a1 / 1e4, a2 / 1e4))
    txt = '\n'.join(lines)
    print(txt)
    with open('own_rr_burden.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('own_rr_burden.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
