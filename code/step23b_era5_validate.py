# -*- coding: utf-8 -*-
"""第8轮补充验证: 再分析(POWER/MERRA-2) 能否代表台站实测的夜间湿球?
(Open-Meteo/ERA5 触发每小时限额, 改用 NASA POWER; 已下载的 ERA5 部分作交叉样本)
用 GHCNh 欧洲/北非台站的亚小时实测 vs 同经纬度 ERA5 逐小时, 比较暖季夜间湿球。
产出: era5_validation.txt
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import concurrent.futures as cf

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import step23_health_data as S
from step23c_power import fetch_power as fetch_one   # POWER(MERRA-2) 逐小时 T2M/T2MDEW

SITES = [
    ('AGI0000DAAG', 36.6910, 3.2154),    # Alger 25m
    ('AGI0000DAOO', 35.6239, -0.6212),   # Es Senia 90m
    ('AGU060360-1', 36.8300, 7.8200),    # Annaba 4m
    ('AUI0000LOWW', 48.1103, 16.5697),   # Wien-Schwechat 183m
    ('GII0000LXGB', 36.1512, -5.3497),   # Gibraltar 5m
    ('GMN00005100', 49.7479, 6.6583),    # Trier 265m
    ('MOI0000GMTA', 35.1771, -3.8395),   # Cherif El Idrissi 27m
    ('MOI0000GMTT', 35.7269, -5.9169),   # Tanger 19m
    ('TSI0000DTKA', 36.9800, 8.8769),    # Tabarka 70m
    ('TUI0000LTAI', 36.8987, 30.8005),   # Antalya 54m
    ('UKI0000EGHE', 49.9167, -6.3000),   # Scilly 31m
]


def main():
    S.CENTROIDS.update({sid: (lat, lon) for sid, lat, lon in SITES})
    obs = pd.read_csv('obs_monthly.csv')
    warm = {}
    for sid, g in obs.groupby(['station', 'month'])['t_mean'].mean().reset_index().groupby('station'):
        warm[sid] = list(g.sort_values('t_mean', ascending=False).month[:3])
    tasks = []
    for sid, lat, lon in SITES:
        yrs = sorted(obs[obs.station == sid].year.unique())
        for y in yrs:
            tasks.append((sid, lat, lon, '%d-05-01' % y, '%d-10-01' % y))
    print('POWER tasks', len(tasks), flush=True)
    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for nid, times, T, TD, elev in ex.map(fetch_one, tasks):
            if times is None:
                continue
            yr = int(times[0][:4])
            for r in S.process(nid, times, T, TD, elev):
                rows.append((nid, yr, r[1], r[5]))   # date, tw_night_mean
    print('fetched %.0fs' % (time.time() - t0))
    e = pd.DataFrame(rows, columns=['station', 'year', 'date', 'tw_night'])
    e['month'] = e['date'].str[5:7].astype(int)
    lines = []
    allx, ally = [], []
    recs = []
    for (sid, yr), g in e.groupby(['station', 'year']):
        mo = warm.get(sid, [6, 7, 8])
        gg = g[g.month.isin(mo)]
        if len(gg) < 60:
            continue
        ev = gg.tw_night.mean()
        o = obs[(obs.station == sid) & (obs.year == yr) & (obs.month.isin(mo))]
        if len(o) < 2:
            continue
        ov = o.tw_night.mean()
        allx.append(ev); ally.append(ov)
        recs.append((sid, yr, ev, ov))
    x = np.array(allx); y = np.array(ally)
    d = y - x
    lines.append('再分析(POWER/MERRA-2) vs 台站实测 暖季夜间湿球 对比  (n=%d 站年, %d 站)' %
                 (len(x), len(set(r[0] for r in recs))))
    lines.append('  均值: 实测 %.2f °C, ERA5 %.2f °C, 偏差(Obs-ERA5) 中位 %+.3f, 均值 %+.3f °C' %
                 (y.mean(), x.mean(), np.median(d), d.mean()))
    lines.append('  相关 r = %.4f, RMSE = %.3f °C, |偏差|<0.5 °C 比例 %.1f%%' %
                 (np.corrcoef(x, y)[0, 1], np.sqrt((d ** 2).mean()), 100 * np.mean(np.abs(d) < 0.5)))
    # 分站
    df = pd.DataFrame(recs, columns=['station', 'year', 'era5', 'obs'])
    lines.append('')
    lines.append('分站偏差 (Obs - 再分析):')
    for sid, g in df.groupby('station'):
        if len(g) >= 3:
            dd = (g.obs - g.era5)
            lines.append('  %-12s n=%2d  偏差 %+.3f °C  RMSE %.3f  r=%.3f'
                         % (sid, len(g), dd.mean(), np.sqrt((dd ** 2).mean()),
                            np.corrcoef(g.era5, g.obs)[0, 1] if len(g) > 2 else float('nan')))
    # 年际变率一致性(相关性)
    lines.append('')
    lines.append('年际异常相关 (去均值后逐站):')
    cors = []
    for sid, g in df.groupby('station'):
        if len(g) >= 8:
            a = g.era5 - g.era5.mean(); b = g.obs - g.obs.mean()
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
    if cors:
        lines.append('  中位 r = %.3f  (n=%d 站, 范围 %.2f..%.2f)' %
                     (np.median(cors), len(cors), min(cors), max(cors)))
    txt = '\n'.join(lines)
    print(txt)
    with open('reanalysis_validation.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('reanalysis_validation.json', 'w', encoding='utf-8') as fh:
        json.dump(dict(n=len(x), bias_median=float(np.median(d)), bias_mean=float(d.mean()),
                       r=float(np.corrcoef(x, y)[0, 1]), rmse=float(np.sqrt((d ** 2).mean())),
                       interannual_r_median=float(np.median(cors)) if cors else None),
                  fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
