# -*- coding: utf-8 -*-
"""第8轮: 用 NASA POWER (MERRA-2) 逐小时 T2M/T2MDEW 补齐完整 NUTS2 面板
(Open-Meteo/ERA5 触发每小时额度限制, 已下载部分转为交叉验证样本)
产出: ext/health/wx_power_nuts2.csv   (主数据, 2001-2024)
"""
import os, sys, json, time, math
import pandas as pd
import concurrent.futures as cf
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
OUT = os.path.join(BASE, 'ext', 'health')
import step23_health_data as S

UA = {'User-Agent': 'Mozilla/5.0 (climate-research)'}


def fetch_power(args):
    nid, lat, lon, sd, ed = args
    s = sd.replace('-', ''); e = ed.replace('-', '')
    url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW'
           '&community=RE&longitude=%.4f&latitude=%.4f&start=%s&end=%s&format=JSON' % (lon, lat, s, e))
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=150) as r:
                js = json.loads(r.read().decode('utf-8'))
            p = js['properties']['parameter']
            t2 = p['T2M']; td = p['T2MDEW']
            keys = sorted(t2.keys())
            times = ['%s-%s-%sT%s:00' % (kk[0:4], kk[4:6], kk[6:8], kk[8:10]) for kk in keys]
            T = [(t2[kk] if t2[kk] > -900 else None) for kk in keys]
            TD = [(td[kk] if td.get(kk, -999) > -900 else None) for kk in keys]
            elev = js['geometry']['coordinates'][2] if 'geometry' in js else 0.0
            return (nid, times, T, TD, elev)
        except Exception as ex:
            time.sleep(2 + 3 * k)
    return (nid, None, None, None, None)


def main():
    cent = pd.read_csv(os.path.join(OUT, 'nuts2_centroids.csv'))
    cent = [(r['nuts2'], float(r['lat']), float(r['lon'])) for _, r in cent.iterrows()]
    S.CENTROIDS.update({a: (b, c) for a, b, c in cent})
    dst = os.path.join(OUT, 'wx_power_nuts2.csv')
    done = set()
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        try:
            df = pd.read_csv(dst, usecols=['nuts2', 'date'])
            done = set(zip(df['nuts2'], df['date'].astype(str).str[:4]))
        except Exception as e:
            print('resume err', e)
    tasks = []
    for nid, lat, lon in cent:
        for yr in range(2001, 2025):
            if (nid, str(yr)) in done:
                continue
            tasks.append((nid, lat, lon, '%d-05-01' % yr, '%d-10-01' % yr))
    print('POWER tasks: %d (skip %d)' % (len(tasks), len(done)), flush=True)
    out = open(dst, 'a' if done else 'w', encoding='utf-8')
    if not done:
        out.write('nuts2,date,tx,tn,tw_night_min,tw_night_mean,tw_night_max,tw_day_max,'
                  'nh219,nh230,nh240,nh258,n_hrs,n_night\n')
    t0 = time.time(); n = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_power, t) for t in tasks]
        for fu in cf.as_completed(futs):
            try:
                nid, times, T, TD, elev = fu.result()
            except Exception:
                nid, times = None, None
            n += 1
            if times is None:
                continue
            for row in S.process(nid, times, T, TD, elev):
                out.write('%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d\n' % row)
            if n % 250 == 0:
                out.flush()
                print('  %d/%d  %.0fs' % (n, len(tasks), time.time() - t0), flush=True)
    out.close()
    print('POWER done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
