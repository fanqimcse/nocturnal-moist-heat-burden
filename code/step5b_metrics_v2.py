#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 5 v2: 核心指标引擎 —— 抗观测频次变化的版本

==========================================================================
为什么必须重写 (v1 的致命缺陷，由"审稿人"环节发现):
==========================================================================
v1 用的是**计数型**指标：
    NRF 夜数 = 湿球最低值 >= 阈值的夜的个数
    hrs      = 湿球 >= 阈值的观测条数

但 GHCNh 的观测频次在几十年里发生了巨大变化。雅浦岛 (FMW00040308):
    1976 年  1,120 条观测 -> 只能识别出  22 个夜 -> NRF =  22
    2000 年  8,781 条观测 -> 识别出    366 个夜 -> NRF = 365
这个"增长"完全是观测加密造成的假象，与气候无关。
更隐蔽的问题是：每夜观测越少，观测到的最小值越高于真实最小值，
因此稀疏年份的 NRF 会被系统性高估 —— 偏差方向与真实趋势相反或相同，无法预知。

==========================================================================
v2 的三道防线:
==========================================================================
1. 逐小时分箱：同一小时内的多条观测取**均值**，
   使结果不受"每小时几次观测"影响（1970s 常见每小时多次报告）。
2. 覆盖率归一化：所有计数型量改成**比率**，再乘以年小时数还原为"等效小时"。
   比率对"每小时/每天观测几次"不敏感，只对"哪些小时有数据"敏感。
3. 严格的年度质量门槛：
   - 年有效小时覆盖率 >= COV_MIN
   - 有效夜数 >= NIGHT_MIN_VALID
   - 单个夜至少 NIGHT_MIN_HOURS 个有效小时，否则该夜不计入 NRF
   不满足的年份打 ok_trend=False，不进入趋势分析。
   并在论文中报告被剔除的比例与敏感性检验。
"""

import sys, os, math, csv, glob, json, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import duckdb

TH = {"elderly": 21.9, "young": 25.8, "psu": 30.6, "classic": 35.0}
SUN_RISE_SET = -0.833

COV_MIN = 0.60            # 年小时覆盖率下限（进入趋势分析）
NIGHT_MIN_HOURS = 5       # 单个夜至少 5 个有效小时
NIGHT_MIN_VALID = 250     # 一年至少 250 个有效夜
DAY_MIN_HOURS = 18        # 计算日较差至少 18 个有效小时

HOURS_YEAR = 8760.0


def solar_decl_eqtime(doy, fh):
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (fh - 12.0) / 24.0)
    eq = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                   - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
           - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
           - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    return dec, eq


def solar_elevation(lat, lon, doy, utc_hour):
    dec, eq = solar_decl_eqtime(doy, utc_hour)
    lr = math.radians(lat)
    tst = utc_hour * 60.0 + eq + lon * 4.0
    ha = math.radians(tst / 4.0 - 180.0)
    cz = math.sin(lr) * math.sin(dec) + math.cos(lr) * math.cos(dec) * math.cos(ha)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cz))))


_C = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
_CL = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]


def doy_of(y, m, d):
    c = _CL if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else _C
    return c[m - 1] + d


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = int(q * (len(s) - 1))
    return s[i]


def analyze(path, lat, lon, station, year):
    con = duckdb.connect()
    con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])"
                % path.replace("'", "''"))
    try:
        rows = con.execute("""
            SELECT CAST(DATE AS VARCHAR), temperature, wet_bulb_temperature,
                   TRY_CAST(relative_humidity AS DOUBLE),
                   wet_bulb_temperature_Quality_Code
            FROM h WHERE wet_bulb_temperature IS NOT NULL ORDER BY DATE
        """).fetchall()
    except Exception:
        return None
    con.close()
    if len(rows) < 200:
        return None

    # ---- 1. 逐小时分箱 (同一小时内取均值) ----
    bin_tw = {}
    bin_t = {}
    for dt, t, tw, rh, qc in rows:
        try:
            y = int(dt[0:4]); mo = int(dt[5:7]); da = int(dt[8:10])
            hh = int(dt[11:13]); mi = int(dt[14:16])
        except Exception:
            continue
        key = ymd_key(y, mo, da, hh)
        b = bin_tw.setdefault(key, [0.0, 0])
        b[0] += tw; b[1] += 1
        if t is not None:
            c = bin_t.setdefault(key, [0.0, 0])
            c[0] += t; c[1] += 1

    if not bin_tw:
        return None

    # ---- 2. 逐小时序列 + 昼夜标记 ----
    hours = []          # (doy, utc_hour, tw, t, is_night)
    for key, (s, n) in bin_tw.items():
        y, mo, da, hh = key
        tw = s / n
        t = None
        if key in bin_t:
            t = bin_t[key][0] / bin_t[key][1]
        uh = hh + 0.5
        el = solar_elevation(lat, lon, doy_of(y, mo, da), uh)
        hours.append((doy_of(y, mo, da), uh, tw, t, el < SUN_RISE_SET))
    hours.sort(key=lambda x: (x[0], x[1]))

    n_hours = len(hours)
    tw_all = [h[2] for h in hours]
    night = [h for h in hours if h[4]]
    day = [h for h in hours if not h[4]]
    tw_n = [h[2] for h in night]
    tw_d = [h[2] for h in day]

    # ---- 3. 覆盖率 ----
    cov = n_hours / HOURS_YEAR
    # 夜/昼的理论小时数随纬度季节变化，用实际样本比例估计
    n_night_h = len(night); n_day_h = len(day)
    f_night = n_night_h / n_hours if n_hours else 0

    # ---- 4. 夜分组 ----
    nights, cur = [], []
    for h in hours:
        if h[4]:
            cur.append(h)
        else:
            if len(cur) >= NIGHT_MIN_HOURS:
                nights.append(cur)
            cur = []
    if len(cur) >= NIGHT_MIN_HOURS:
        nights.append(cur)
    n_valid_nights = len(nights)

    # ---- 5. 日较差 ----
    byday = collections.defaultdict(list)
    for h in hours:
        byday[(h[0])].append(h)
    dtr_t, dtr_tw = [], []
    for d, hs in byday.items():
        if len(hs) < DAY_MIN_HOURS:
            continue
        w = [x[2] for x in hs]
        tv = [x[3] for x in hs if x[3] is not None]
        dtr_tw.append(max(w) - min(w))
        if len(tv) >= DAY_MIN_HOURS:
            dtr_t.append(max(tv) - min(tv))

    def mean(x):
        return sum(x) / len(x) if x else None

    def rate_above(vals, th):
        if not vals:
            return None
        return sum(1 for v in vals if v >= th) / len(vals)

    r = dict(station=station, year=year, lat=lat, lon=lon,
             n_obs=len(rows), n_hours=n_hours, coverage=cov,
             n_night_hours=n_night_h, n_day_hours=n_day_h,
             n_valid_nights=n_valid_nights,
             tw_max=max(tw_all), tw_p99=pct(tw_all, 0.99), tw_mean=mean(tw_all),
             t_max=max([h[3] for h in hours if h[3] is not None], default=None),
             t_mean=mean([h[3] for h in hours if h[3] is not None]),
             tw_night_mean=mean(tw_n), tw_day_mean=mean(tw_d),
             tw_night_max=max(tw_n) if tw_n else None,
             tw_day_max=max(tw_d) if tw_d else None,
             dtr_t=mean(dtr_t), dtr_tw=mean(dtr_tw),
             n_days_dtr=len(dtr_tw))

    for k, th in TH.items():
        r[f"rate_{k}"] = rate_above(tw_all, th)
        r[f"hrs_{k}"] = (r[f"rate_{k}"] * HOURS_YEAR) if r[f"rate_{k}"] is not None else None
        r[f"nrate_{k}"] = rate_above(tw_n, th)
        r[f"nhrs_{k}"] = (r[f"nrate_{k}"] * HOURS_YEAR * f_night) if r[f"nrate_{k}"] is not None else None
        r[f"drate_{k}"] = rate_above(tw_d, th)
        r[f"dhrs_{k}"] = (r[f"drate_{k}"] * HOURS_YEAR * (1 - f_night)) if r[f"drate_{k}"] is not None else None
        # NRF: 夜间湿球最低值 >= 阈值 的夜 占有效夜的比例
        if n_valid_nights:
            nrf = sum(1 for nt in nights if min(x[2] for x in nt) >= th)
            r[f"nrf_{k}"] = nrf
            r[f"nrfr_{k}"] = nrf / n_valid_nights
        else:
            r[f"nrf_{k}"] = None
            r[f"nrfr_{k}"] = None

    # ---- 6. 质量门槛 ----
    r["ok_trend"] = int(cov >= COV_MIN and n_valid_nights >= NIGHT_MIN_VALID)
    return r


def ymd_key(y, m, d, hh):
    return (y, m, d, hh)


def run(pattern="ghcn_data/GHCNh_*_*.parquet", meta_json="panel_all.json",
        out="metrics_all.csv"):
    meta = {}
    if os.path.exists(meta_json):
        for m in json.load(open(meta_json, encoding="utf-8")):
            meta[m["station"]] = m
    files = sorted(glob.glob(pattern))
    print(f"待分析 {len(files)} 个文件")
    rows = []
    for i, p in enumerate(files, 1):
        base = os.path.basename(p).replace(".parquet", "")
        parts = base.split("_")
        if len(parts) < 3:
            continue
        st, yr = parts[1], int(parts[2])
        m = meta.get(st)
        if m:
            lat, lon = m["lat"], m["lon"]
        else:
            try:
                c = duckdb.connect()
                c.execute("CREATE VIEW z AS SELECT * FROM read_parquet(['%s'])" % p)
                a = c.execute("SELECT MAX(LATITUDE), MAX(LONGITUDE) FROM z").fetchall()[0]
                lat, lon = float(a[0]), float(a[1])
                c.close()
            except Exception:
                continue
        try:
            r = analyze(p, lat, lon, st, yr)
        except Exception as e:
            print(f"  [{i}] {base} 错误 {str(e)[:50]}")
            continue
        if r is None:
            print(f"  [{i}] {base} 数据不足")
            continue
        if m:
            r["name"] = m["name"]; r["country"] = m["country"]
            r["band"] = m.get("band", ""); r["region"] = m.get("region", "")
        rows.append(r)
        if i % 200 == 0:
            print(f"  进度 {i}/{len(files)}", flush=True)
    if rows:
        keys = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        nok = sum(r["ok_trend"] for r in rows)
        print(f"\n写出 {out}: {len(rows)} 条, 其中 {nok} 条通过质量门槛 ({100*nok/len(rows):.0f}%)")
    return rows


if __name__ == "__main__":
    run()
