#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 5: 核心指标引擎 —— 逐站年计算昼夜湿球热应激指标

方法学要点（对应审稿人会质疑的每一处）:

1. 夜间定义 —— 用太阳高度角，而非固定钟点
   固定钟点需要时区，而 GHCNh 覆盖 38,870 站 / 253 国家 / 250 年，
   时区与夏令时规则历史多变且元数据不可靠。改用太阳高度角 < -0.833°
   （太阳上边缘完全沉入地平线的标准判据），时区无关、随季节与纬度自动正确。

2. 核心指标: 夜间恢复失效 (Nocturnal Recovery Failure, NRF)
   一个"夜"= 连续太阳在地平线下的一组观测。
   NRF 夜 = 该夜湿球最低值仍 >= 阈值，即全天最凉的时刻也未降到阈值以下。
   这是生理上"无法从白天热负荷中恢复"的直接度量。
   主流业务指标（如日最大湿球）按定义只取白天峰值，对此结构性失明。

3. 昼夜解耦指标
   干球日较差 DTR_T = Tmax - Tmin
   湿球日较差 DTR_Tw = Tw_max - Tw_min
   夜间/白天湿球均值之比 —— 衡量"夜间是否还提供降温窗口"

4. 数据完备性门槛
   单日需 >= 8 条观测且覆盖 >= 12 小时，单夜需 >= 3 条观测，否则不计入统计。
"""

import os, math, csv, glob, json, collections, duckdb
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 阈值 (°C) -------------------------------------------------------------
TH = {
    "elderly": 21.9,   # Perkins-Kirkpatrick & Jay 2026, 老年人下限
    "young": 25.8,     # 青年成人下限
    "psu": 30.6,       # PSU HEAT 健康青年实测临界
    "classic": 35.0,   # Sherwood & Huber 2010 经典不可生存阈值
}
PRIMARY = 21.9

SUN_RISE_SET = -0.833     # 太阳上边缘沉入地平线的标准折射修正角
DAY_MIN_OBS = 8           # 单日最少观测数
DAY_MIN_SPAN = 12.0       # 单日观测至少跨越的小时数
NIGHT_MIN_OBS = 3         # 单夜最少观测数


# ---- 太阳位置 (NOAA / Spencer 1971) ---------------------------------------
def solar_decl_eqtime(doy, frac_hour_utc):
    """返回 (太阳赤纬 rad, 时差 分钟)。doy 为年内日序, frac_hour_utc 为 UTC 小时。"""
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (frac_hour_utc - 12.0) / 24.0)
    eqtime = (229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    return decl, eqtime


def solar_elevation(lat, lon, doy, utc_hour):
    """太阳高度角 (度)。lat/lon 单位度, utc_hour 为 UTC 十进制小时。"""
    decl, eqtime = solar_decl_eqtime(doy, utc_hour)
    latr = math.radians(lat)
    # 真太阳时 (分钟) -> 时角
    tst = utc_hour * 60.0 + eqtime + lon * 4.0     # lon*4: 1度=4分钟
    ha = math.radians(tst / 4.0 - 180.0)
    cz = (math.sin(latr) * math.sin(decl)
          + math.cos(latr) * math.cos(decl) * math.cos(ha))
    cz = max(-1.0, min(1.0, cz))
    return 90.0 - math.degrees(math.acos(cz))


DOY_CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
DOY_CUM_L = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]


def ymd_to_doy(y, m, d):
    c = DOY_CUM_L if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else DOY_CUM
    return c[m - 1] + d


# ---- 主分析 ----------------------------------------------------------------
SQL = """
SELECT
    CAST(DATE AS VARCHAR) AS dt,
    temperature AS t,
    wet_bulb_temperature AS tw,
    TRY_CAST(relative_humidity AS DOUBLE) AS rh,
    wet_bulb_temperature_Quality_Code AS tw_qc
FROM h
WHERE wet_bulb_temperature IS NOT NULL
ORDER BY DATE
"""


def analyze_file(path, lat, lon, station, year):
    con = duckdb.connect()
    lst = "'" + path.replace("'", "''") + "'"
    con.execute(f"CREATE VIEW h AS SELECT * FROM read_parquet([{lst}])")
    try:
        rows = con.execute(SQL).fetchall()
    except Exception as e:
        return None
    con.close()
    if not rows:
        return None

    # 解析时间戳 + 太阳高度角 + 昼夜标记
    recs = []
    qc_bad = 0
    for dt, t, tw, rh, qc in rows:
        try:
            y = int(dt[0:4]); mo = int(dt[5:7]); da = int(dt[8:10])
            hh = int(dt[11:13]); mi = int(dt[14:16])
        except Exception:
            continue
        if qc is not None and str(qc).strip() not in ("", "0", "1", "5"):
            qc_bad += 1
        utc_h = hh + mi / 60.0
        doy = ymd_to_doy(y, mo, da)
        el = solar_elevation(lat, lon, doy, utc_h)
        recs.append((y, mo, da, doy, utc_h, t, tw, el))

    n_obs = len(recs)
    if n_obs < 100:
        return None

    # ---- 按日聚合 ----
    days = collections.defaultdict(list)
    for r in recs:
        days[(r[0], r[1], r[2])].append(r)

    # ---- 昼夜分组: 连续"太阳在地平线下"的观测构成一个夜 ----
    nights = []
    cur = []
    for r in recs:
        if r[7] < SUN_RISE_SET:
            cur.append(r)
        else:
            if len(cur) >= NIGHT_MIN_OBS:
                nights.append(cur)
            cur = []
    if len(cur) >= NIGHT_MIN_OBS:
        nights.append(cur)

    # ---- 指标 ----
    def hrs_above(vals, th):
        return sum(1 for v in vals if v is not None and v >= th)

    tw_all = [r[6] for r in recs]
    t_all = [r[5] for r in recs if r[5] is not None]
    night_recs = [r for r in recs if r[7] < SUN_RISE_SET]
    day_recs = [r for r in recs if r[7] >= SUN_RISE_SET]
    tw_night = [r[6] for r in night_recs]
    tw_day = [r[6] for r in day_recs]

    # 合格日 (用于日较差)
    ok_days = []
    for k, v in days.items():
        if len(v) < DAY_MIN_OBS:
            continue
        ts = [x[4] for x in v]
        if max(ts) - min(ts) < DAY_MIN_SPAN:
            continue
        ok_days.append(v)

    dtr_t, dtr_tw = [], []
    for v in ok_days:
        tv = [x[5] for x in v if x[5] is not None]
        wv = [x[6] for x in v]
        if len(tv) >= DAY_MIN_OBS and wv:
            dtr_t.append(max(tv) - min(tv))
            dtr_tw.append(max(wv) - min(wv))

    # NRF: 夜间湿球最低值 >= 阈值
    night_min = [min(x[6] for x in nt) for nt in nights]
    nrf = {k: sum(1 for m in night_min if m >= th) for k, th in TH.items()}
    # 夜间累积超阈度时 (近似: 观测条数 x 平均超阈量, 再按年均观测间隔折算小时)
    night_obs_per_hr = len(night_recs) / max(1, len(nights)) / 10.0 if nights else 0

    def mean(x):
        return sum(x) / len(x) if x else None

    res = dict(
        station=station, year=year, lat=lat, lon=lon,
        n_obs=n_obs, n_days=len(days), n_ok_days=len(ok_days),
        n_nights=len(nights), qc_flagged=qc_bad,
        tw_max=max(tw_all), tw_mean=mean(tw_all),
        t_max=max(t_all) if t_all else None, t_mean=mean(t_all),
        hrs_elderly=hrs_above(tw_all, TH["elderly"]),
        hrs_young=hrs_above(tw_all, TH["young"]),
        hrs_psu=hrs_above(tw_all, TH["psu"]),
        hrs_classic=hrs_above(tw_all, TH["classic"]),
        nhrs_elderly=hrs_above(tw_night, TH["elderly"]),
        nhrs_young=hrs_above(tw_night, TH["young"]),
        dhrs_elderly=hrs_above(tw_day, TH["elderly"]),
        dhrs_young=hrs_above(tw_day, TH["young"]),
        tw_night_mean=mean(tw_night), tw_day_mean=mean(tw_day),
        tw_night_max=max(tw_night) if tw_night else None,
        tw_day_max=max(tw_day) if tw_day else None,
        dtr_t=mean(dtr_t), dtr_tw=mean(dtr_tw),
        nrf_elderly=nrf["elderly"], nrf_young=nrf["young"],
        nrf_psu=nrf["psu"], nrf_classic=nrf["classic"],
        night_min_mean=mean(night_min),
        night_min_max=max(night_min) if night_min else None,
    )
    return res


def run(pattern="ghcn_data/GHCNh_*_*.parquet", meta_json="panel_core.json",
        out="metrics_core.csv"):
    meta = {}
    if os.path.exists(meta_json):
        for m in json.load(open(meta_json, encoding="utf-8")):
            meta[m["station"]] = m

    files = sorted(glob.glob(pattern))
    print(f"待分析文件 {len(files)} 个")
    out_rows = []
    for i, p in enumerate(files, 1):
        base = os.path.basename(p)
        parts = base.replace(".parquet", "").split("_")
        if len(parts) < 3:
            continue
        st, yr = parts[1], int(parts[2])
        m = meta.get(st)
        if m:
            lat, lon = m["lat"], m["lon"]
        else:
            try:
                import duckdb as d2
                c = d2.connect()
                c.execute(f"CREATE VIEW z AS SELECT * FROM read_parquet(['{p}'])")
                r = c.execute("SELECT MAX(LATITUDE), MAX(LONGITUDE) FROM z").fetchall()[0]
                lat, lon = float(r[0]), float(r[1])
                c.close()
            except Exception:
                continue
        try:
            r = analyze_file(p, lat, lon, st, yr)
        except Exception as e:
            print(f"  [{i}] {base} 失败: {str(e)[:60]}")
            continue
        if r is None:
            print(f"  [{i}] {base} 跳过")
            continue
        if m:
            r["name"] = m["name"]; r["country"] = m["country"]; r["band"] = m["band"]
        out_rows.append(r)
        if i % 100 == 0:
            print(f"  进度 {i}/{len(files)}", flush=True)

    if out_rows:
        keys = list(out_rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n已写出 {out}  ({len(out_rows)} 条站年记录)")
    return out_rows


if __name__ == "__main__":
    run()
