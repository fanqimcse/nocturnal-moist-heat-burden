# -*- coding: utf-8 -*-
"""第8轮补充: 两套独立再分析(ERA5 vs MERRA-2/POWER) 在 NUTS2 网格上的一致性
健康面板的暴露来自再分析而非台站。若两套独立再分析给出的逐夜湿球高度一致,
则"暴露来自再分析"这一选择不改变结论。
产出: reanalysis_crosscheck.txt / .json
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'ext', 'health')

COLS = ['tw_night_min', 'tw_night_mean', 'tw_night_max', 'tx', 'tn', 'nh219', 'nh240']


def main():
    a = pd.read_csv(os.path.join(OUT, 'wx_era5_nuts2.csv'))
    b = pd.read_csv(os.path.join(OUT, 'wx_power_nuts2.csv'))
    for d in (a, b):
        d['date'] = d['date'].astype(str)
    keys = ['nuts2', 'date']
    j = a[keys + COLS].merge(b[keys + COLS], on=keys, suffixes=('_era5', '_pwr'))
    lines = ['两套独立再分析交叉检验 (ERA5/Open-Meteo  vs  MERRA-2/NASA POWER)',
             '重叠样本: %d 区域-日, %d 区域' % (len(j), j['nuts2'].nunique()),
             '']
    out = {'n_rows': int(len(j)), 'n_regions': int(j['nuts2'].nunique())}
    lines.append('%-16s %8s %8s %8s %8s %8s' % ('变量', '均值A', '均值B', '偏差', 'RMSE', 'r'))
    for c in COLS:
        x = j[c + '_era5'].values.astype(float)
        y = j[c + '_pwr'].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 100:
            continue
        d = y - x
        r = float(np.corrcoef(x, y)[0, 1])
        rmse = float(np.sqrt((d ** 2).mean()))
        lines.append('%-16s %8.3f %8.3f %+8.3f %8.3f %8.4f'
                     % (c, x.mean(), y.mean(), d.mean(), rmse, r))
        out[c] = dict(n=int(len(x)), mean_era5=float(x.mean()), mean_power=float(y.mean()),
                      bias=float(d.mean()), rmse=rmse, r=r)
    # 关键: 周聚合后的指标是否一致(健康模型实际用的是周聚合)
    for d in (a, b):
        d['iso'] = pd.to_datetime(d['date']).map(lambda t: t.isocalendar()[:2])
        d['week'] = d['iso'].map(lambda t: '%d-W%02d' % t)
    wa = a.groupby(['nuts2', 'week']).agg(nh219=('nh219', 'sum'), tn=('tn', 'max'),
                                          twmin=('tw_night_min', 'max')).reset_index()
    wb = b.groupby(['nuts2', 'week']).agg(nh219=('nh219', 'sum'), tn=('tn', 'max'),
                                          twmin=('tw_night_min', 'max')).reset_index()
    wj = wa.merge(wb, on=['nuts2', 'week'], suffixes=('_a', '_b'))
    lines.append('')
    lines.append('周聚合后(健康模型实际使用的口径), n=%d 区域-周' % len(wj))
    out['weekly'] = {}
    for c in ['nh219', 'tn', 'twmin']:
        x = wj[c + '_a'].values.astype(float); y = wj[c + '_b'].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        r = float(np.corrcoef(x, y)[0, 1])
        lines.append('  %-8s r = %.4f   偏差 %+.3f   RMSE %.3f'
                     % (c, r, (y - x).mean(), np.sqrt(((y - x) ** 2).mean())))
        out['weekly'][c] = dict(r=r, bias=float((y - x).mean()),
                                rmse=float(np.sqrt(((y - x) ** 2).mean())), n=int(len(x)))
    txt = '\n'.join(lines)
    print(txt)
    with open('reanalysis_crosscheck.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('reanalysis_crosscheck.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
