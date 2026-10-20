# -*- coding: utf-8 -*-
"""第8轮: 欧洲 NUTS2 质心 + Eurostat NUTS3 周死亡 + Open-Meteo(ERA5)逐小时气象
产出: ext/health/nuts2_centroids.csv, ext/health/deaths_weekly_nuts2.csv, ext/health/wx_daily_nuts2.parquet
"""
import os, json, math, time, sys, gzip
import urllib.request, urllib.parse
import concurrent.futures as cf

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
OUT = os.path.join(BASE, 'ext', 'health')
os.makedirs(OUT, exist_ok=True)
CENTROIDS = {}
from step5c_metrics_v3 import WetBulbSolver, solar_elevation, doy_of   # 与主管线完全一致

UA = {'User-Agent': 'Mozilla/5.0 (climate-research)'}


def get(url, timeout=120, retries=3, binary=False):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
            return b if binary else b.decode('utf-8', 'replace')
        except Exception as e:
            if k == retries - 1:
                print('  FAIL', url[:90], type(e).__name__, e, flush=True)
                return None
            time.sleep(2 + 3 * k)
    return None


# ---------------- 1. NUTS2 质心 ----------------
def build_centroids():
    dst = os.path.join(OUT, 'nuts2_centroids.csv')
    if os.path.exists(dst):
        print('centroids exist'); return dst
    url = 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_2.geojson'
    print('downloading NUTS2 geojson ...', flush=True)
    txt = get(url, timeout=300)
    if not txt:
        sys.exit('no geojson')
    gj = json.loads(txt)
    rows = []
    for f in gj['features']:
        p = f['properties']
        nid = p.get('NUTS_ID') or p.get('id')
        if not nid or p.get('LEVL_CODE', 2) != 2:
            continue
        g = f.get('geometry')
        if not g:
            continue
        # 取最大多边形的顶点平均作为质心
        polys = []
        if g['type'] == 'Polygon':
            polys = [g['coordinates'][0]]
        elif g['type'] == 'MultiPolygon':
            polys = [c[0] for c in g['coordinates']]
        best = max(polys, key=len)
        xs = [c[0] for c in best]; ys = [c[1] for c in best]
        lon = sum(xs) / len(xs); lat = sum(ys) / len(ys)
        # 只保留欧洲主体（排除海外领地）
        if not (33.0 <= lat <= 72.5 and -26.0 <= lon <= 46.0):
            continue
        rows.append((nid, p.get('CNTR_CODE', ''), p.get('NAME_LATN', ''), round(lat, 4), round(lon, 4)))
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write('nuts2,cntr,name,lat,lon\n')
        for r in sorted(rows):
            fh.write('%s,%s,%s,%.4f,%.4f\n' % (r[0], r[1], r[2].replace(',', ' '), r[3], r[4]))
    print('NUTS2 centroids:', len(rows), '->', dst)
    return dst


# ---------------- 2. Eurostat 周死亡 ----------------
def iso_weeks(year):
    import datetime
    # ISO 8601: 第1周含该年第一个周四
    d = datetime.date(year, 1, 4)
    start = d - datetime.timedelta(days=d.weekday())
    ws = []
    y = year
    cur = start
    while True:
        ws.append('%04d-W%02d' % (y, ((cur - start).days // 7) + 1))
        cur += datetime.timedelta(days=7)
        nxt = cur + datetime.timedelta(days=3)
        if nxt.year > y:
            break
    return ws


def download_deaths(years):
    dst = os.path.join(OUT, 'deaths_weekly_nuts3.csv')
    if os.path.exists(dst):
        print('deaths exist'); return dst
    allrows = {}
    for yr in years:
        ws = iso_weeks(yr)
        # 分块，每块 40 周，避免 URL 过长
        got = 0
        for i in range(0, len(ws), 40):
            chunk = ws[i:i + 40]
            q = '&'.join(['time=' + urllib.parse.quote(w) for w in chunk])
            url = ('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_r_mweek3'
                   '?format=JSON&sex=T&age=TOTAL&unit=NR&' + q)
            txt = get(url, timeout=180)
            if not txt:
                continue
            try:
                js = json.loads(txt)
            except Exception as e:
                print('  json err', yr, e); continue
            dims = js['dimension']
            ids = js.get('ids') or js.get('id')
            if ids is None:
                print('  keys:', list(js.keys()), flush=True)
                continue
            gidx = dims['geo']['category']['index']
            ginv = {v: k for k, v in gidx.items()}
            tidx = dims['time']['category']['index']
            tinv = {v: k for k, v in tidx.items()}
            # 计算 stride
            stride = {}
            s = 1
            for k in reversed(ids):
                stride[k] = s
                s *= len(dims[k]['category']['index'])
            for kstr, v in js['value'].items():
                k = int(kstr)
                rem = k
                rec = {}
                for kk in ids:
                    n = len(dims[kk]['category']['index'])
                    rec[kk] = rem // stride[kk]
                    rem = rem % stride[kk]
                g = ginv.get(rec['geo'], '?')
                t = tinv.get(rec['time'], '?')
                if v is None:
                    continue
                allrows[(g, t)] = v
            got += len(js['value'])
        print('  %d deaths: %d cells' % (yr, got), flush=True)
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write('nuts3,week,deaths\n')
        for (g, t), v in sorted(allrows.items()):
            fh.write('%s,%s,%d\n' % (g, t, v))
    print('deaths total cells:', len(allrows), '->', dst)
    return dst


# ---------------- 3. Open-Meteo 逐小时 ----------------
def solar_alt(lat, lon, utc_hours):
    """utc_hours: 自 epoch 的小时数(UTC) -> 太阳高度角(度)"""
    import datetime
    days = utc_hours / 24.0
    d = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=days)
    n = d.toordinal() - datetime.date(d.year, 1, 1).toordinal() + 1
    # 更精确: 用 Julian day
    jd = days + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + math.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + math.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = math.radians(true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega)))
    eps0 = 23 + (26 + ((21.448 - T * (46.815 + T * (0.00059 - T * 0.001813)))) / 60) / 60
    e = math.radians(eps0 + 0.00256 * math.cos(math.radians(omega)))
    dec = math.asin(math.sin(e) * math.sin(lam))
    # 格林尼治真太阳时
    minutes = d.hour * 60 + d.minute + d.second / 60.0
    true_solar = (minutes / 4.0 + math.degrees(math.asin(math.sin(math.radians(L0)) * 0)) )
    eqtime = 4 * math.degrees(
        math.sin(2 * lam) * (-0.0) )  # 用简化: NOAA 近似
    # NOAA equation of time (minutes)
    b = math.radians(360.0 * (n - 81) / 364.0)
    eqt = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    mins = d.hour * 60 + d.minute + d.second / 60.0
    true_solar_time = mins + eqt + 4 * lon
    ha = math.radians(true_solar_time / 4.0 - 180.0)
    latr = math.radians(lat)
    cosz = (math.sin(latr) * math.sin(dec) + math.cos(latr) * math.cos(dec) * math.cos(ha))
    cosz = max(-1.0, min(1.0, cosz))
    return math.degrees(math.asin(cosz))


def wetbulb_from_td(tdb, td, p=1013.25):
    """Davies-Jones/Bolton 湿绝热线: 二分求解 es(Tw) - p*0.00066*(T-Tw)*(1+0.00115Tw) = e(Td)"""
    if tdb is None or td is None:
        return None
    if td > tdb:
        td = tdb
    e = 0.6108 * math.exp(17.27 * td / (td + 237.3))          # kPa (Tetens)

    def f(tw):
        ew = 0.6108 * math.exp(17.27 * tw / (tw + 237.3))
        return ew - 0.00066 * p * (tdb - tw) * (1 + 0.00115 * tw) - e

    lo, hi = td - 0.5, tdb + 0.05
    if f(hi) < 0:
        return tdb
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fetch_one(args):
    """下载 (region, year) 夏季逐小时 ERA5，就地计算逐夜湿球指标，返回日行列表。"""
    nid, lat, lon, sd, ed = args
    url = ('https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f'
           '&start_date=%s&end_date=%s&hourly=temperature_2m,dew_point_2m&timezone=UTC' % (lat, lon, sd, ed))
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                js = json.loads(r.read().decode('utf-8'))
            ho = js['hourly']
            return (nid, ho['time'], ho['temperature_2m'], ho['dew_point_2m'], js.get('elevation', 0.0))
        except Exception as e:
            time.sleep(1.5 + 2 * k)
    return (nid, None, None, None, None)


THRESH = (21.9, 23.0, 24.0, 25.8)


def process(nid, times, T, TD, elev):
    """逐夜窗口(当日12UTC起24h)聚合。"""
    lat = CENTROIDS[nid][0]; lon = CENTROIDS[nid][1]
    p = 1013.25 * math.exp(-(elev or 0.0) / 8434.0)
    sv = WetBulbSolver(p)
    n = len(times)
    alt = [0.0] * n
    tw = [None] * n
    hrs = []
    for i, ts in enumerate(times):
        y = int(ts[0:4]); m = int(ts[5:7]); d = int(ts[8:10]); h = int(ts[11:13])
        hrs.append((y, m, d, h))
    # 太阳高度角
    doys = {}
    for i, (y, m, d, h) in enumerate(hrs):
        key = (y, m, d)
        v = doys.get(key)
        if v is None:
            v = doy_of(y, m, d); doys[key] = v
        alt[i] = solar_elevation(lat, lon, v, h + 0.5)
    for i in range(n):
        if T[i] is None or TD[i] is None:
            continue
        tw[i] = sv.from_td(T[i], TD[i])
    out = []
    # 按 "当日 12:00 UTC 起 24h" 归为一夜
    idx_by_day = {}
    for i, (y, m, d, h) in enumerate(hrs):
        idx_by_day.setdefault((y, m, d), []).append(i)
    days = sorted(idx_by_day)
    for j, (y, m, d) in enumerate(days):
        # 窗口: 本日 12:00 起 -> 次日 11:59
        idxs = [i for i in idx_by_day[(y, m, d)] if hrs[i][3] >= 12]
        nxt = days[j + 1] if j + 1 < len(days) else None
        if nxt:
            idxs += [i for i in idx_by_day[nxt] if hrs[i][3] < 12]
        nh = [i for i in idxs if alt[i] < -0.833]
        if len(idxs) < 20 or len(nh) < 3:
            continue
        tvals = [T[i] for i in idxs if T[i] is not None]
        if not tvals:
            continue
        tx = max(tvals); tn = min(tvals)
        nn = [i for i in nh if tw[i] is not None]
        if len(nn) < 3:
            continue
        twn = [tw[i] for i in nn]
        twd = [tw[i] for i in idxs if tw[i] is not None]
        out.append((
            nid, '%04d-%02d-%02d' % (y, m, d),
            round(tx, 2), round(tn, 2),
            round(min(twn), 2), round(sum(twn) / len(twn), 2), round(max(twn), 2),
            round(max(twd), 2) if twd else '',
            sum(1 for v in twn if v > THRESH[0]),
            sum(1 for v in twn if v > THRESH[1]),
            sum(1 for v in twn if v > THRESH[2]),
            sum(1 for v in twn if v > THRESH[3]),
            len(idxs), len(nn)))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'cent'):
        build_centroids()
    if mode in ('all', 'deaths'):
        download_deaths(range(2000, 2025))
    if mode in ('all', 'wx'):
        global CENTROIDS
        import csv
        import pandas as pd
        CENTROIDS = {}
        cent = []
        with open(os.path.join(OUT, 'nuts2_centroids.csv'), encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                CENTROIDS[r['nuts2']] = (float(r['lat']), float(r['lon']))
                cent.append((r['nuts2'], float(r['lat']), float(r['lon'])))
        print('regions:', len(cent))
        tasks = []
        for nid, lat, lon in cent:
            for yr in range(2000, 2025):
                tasks.append((nid, lat, lon, '%d-05-01' % yr, '%d-10-01' % yr))
        # 断点续传: 跳过已完成的 (区域, 年)
        done = set()
        wp = os.path.join(OUT, 'wx_nightly_nuts2.csv')
        if os.path.exists(wp) and os.path.getsize(wp) > 1000:
            try:
                df = pd.read_csv(wp, usecols=['nuts2', 'date'])
                done = set(zip(df['nuts2'], df['date'].astype(str).str[:4]))
            except Exception as e:
                print('resume scan err', e)
        t_all = len(tasks)
        tasks = [t for t in tasks if (t[0], t[3][:4]) not in done]
        print('wx tasks: %d (skip %d done)' % (len(tasks), t_all - len(tasks)), flush=True)
        out = open(wp, 'a' if done else 'w', encoding='utf-8')
        if not done:
            out.write('nuts2,date,tx,tn,tw_night_min,tw_night_mean,tw_night_max,tw_day_max,'
                      'nh219,nh230,nh240,nh258,n_hrs,n_night\n')
        done = 0
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(fetch_one, t) for t in tasks]
            for fu in cf.as_completed(futs):
                try:
                    nid, times, T, TD, elev = fu.result()
                except Exception as e:
                    nid, times = None, None
                done += 1
                if times is None:
                    continue
                for row in process(nid, times, T, TD, elev):
                    out.write('%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d\n' % row)
                if done % 250 == 0:
                    out.flush()
                    print('  %d/%d  %.0fs' % (done, len(tasks), time.time() - t0), flush=True)
        out.close()
        print('wx done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
