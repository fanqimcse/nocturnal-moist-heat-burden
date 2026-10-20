# -*- coding: utf-8 -*-
"""第8轮 R8-5 配套: 由 GHCNh 亚小时观测重算"模式同口径"月值
  - 月 tasmin (月内日最低干球的月均)
  - 月 hurs   (月内逐小时相对湿度的月均)
  - 月 tw_night (夜间湿球真值, 亚小时 + 太阳高度角)
  - 月 tw_stull (模式同口径: wetbulb(月tasmin, 月hurs))
输出: obs_monthly.csv
"""
import os, sys, math, json, glob
import numpy as np
import pandas as pd
import duckdb

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from step5c_metrics_v3 import WetBulbSolver, solar_elevation, doy_of

DATA = os.path.join(BASE, 'ghcn_data')


def stull(T, rh):
    rh = min(max(rh, 1.0), 100.0)
    return (T * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(T + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def stn_year_month(path):
    con = duckdb.connect()
    con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])" % path.replace("'", "''"))
    rows = con.execute("""
        SELECT CAST(DATE AS VARCHAR) d, TRY_CAST(temperature AS DOUBLE) t,
               TRY_CAST(dew_point_temperature AS DOUBLE) td,
               TRY_CAST(relative_humidity AS DOUBLE) rh
        FROM h WHERE temperature IS NOT NULL
    """).fetchall()
    con.close()
    return rows


def do_file(path, lat, lon, press):
    rows = stn_year_month(path)
    if len(rows) < 500:
        return []
    solver = WetBulbSolver(press if press and press > 500 else 1013.25)
    mon = {}
    daymin = {}
    for d, t, td, rh in rows:
        if t is None or not (-80 < t < 60):
            continue
        y = int(d[0:4]); m = int(d[5:7]); dd = int(d[8:10]); hh = int(d[11:13])
        k = (y, m)
        rec = mon.get(k)
        if rec is None:
            rec = [0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0]
            mon[k] = rec
        rec[0] += t; rec[1] += 1                      # 干球均值
        if rh is not None and 0 <= rh <= 100:
            rec[2] += rh; rec[3] += 1                 # RH
        # 夜间湿球
        alt = solar_elevation(lat, lon, doy_of(y, m, dd), hh + 0.5)
        if alt < -0.833 and td is not None and -80 < td < 60:
            tw = solver.from_td(t, td)
            if tw is not None:
                rec[4] += tw; rec[5] += 1
        # 日最低温
        dk = (y, m, dd)
        cur = daymin.get(dk)
        if cur is None or t < cur:
            daymin[dk] = t
    out = []
    for (y, m), rec in mon.items():
        if rec[1] < 200 or rec[3] < 100 or rec[5] < 30:
            continue
        t_mean = rec[0] / rec[1]
        rh_mean = rec[2] / rec[3]
        tw_night = rec[4] / rec[5]
        dm = [v for (yy, mm, _), v in daymin.items() if yy == y and mm == m]
        tasmin = sum(dm) / len(dm) if dm else None
        if tasmin is None:
            continue
        out.append((y, m, round(t_mean, 3), round(tasmin, 3), round(rh_mean, 2),
                    round(tw_night, 3), round(stull(tasmin, rh_mean), 3), rec[1], rec[5]))
    return out


def main():
    m = pd.read_csv('metrics_v3.csv', usecols=['station', 'lat', 'lon', 'year', 'press'])
    st = m.groupby('station').first().reset_index()[['station', 'lat', 'lon', 'press']]
    latmap = dict(zip(st.station, st.lat)); lonmap = dict(zip(st.station, st.lon))
    pmap = dict(zip(st.station, st.press))
    yrs = sorted(set(m.year.astype(int)))
    yrs = [y for y in yrs if 1976 <= y <= 2014]
    files = []
    for sid in st.station:
        for y in yrs:
            p = os.path.join(DATA, 'GHCNh_%s_%d.parquet' % (sid, y))
            if os.path.exists(p):
                files.append((sid, y, p))
    print('files:', len(files), flush=True)
    out = open('obs_monthly.csv', 'w')
    out.write('station,year,month,t_mean,tasmin,rh,tw_night,tw_stull,n_hr,n_night\n')
    done = 0
    import time
    t0 = time.time()
    for sid, y, p in files:
        try:
            for r in do_file(p, latmap[sid], lonmap[sid], pmap[sid]):
                out.write('%s,%d,%d,%s,%s,%s,%s,%s,%d,%d\n' % ((sid,) + r))
        except Exception as e:
            print('ERR', sid, y, type(e).__name__, str(e)[:80], flush=True)
        done += 1
        if done % 400 == 0:
            print('  %d/%d  %.0fs' % (done, len(files), time.time() - t0), flush=True)
    out.close()
    print('done %.0fs -> obs_monthly.csv' % (time.time() - t0))


if __name__ == '__main__':
    main()
