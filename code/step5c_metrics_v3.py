#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 5 v3: 指标引擎 —— 自算湿球温度版本（最终版）

==========================================================================
为什么必须用自算湿球，而不是 NOAA 原生字段
==========================================================================
实测 287 万条记录：
    有原生 wet_bulb_temperature :  1,165,149 条  (40.6%)
    有干球 T + 露点 Td          :  2,724,116 条  (95.0%)

更关键的是，原生湿球字段的**可得性本身随年代剧烈变化**：
    拉巴斯 BLI0000SLLP  1976 年:      0 / 6,372 条
                        1980 年:  2,354 / 7,632 条
                        2003 年:  5,929 / 8,868 条
                        2019 年:  2,854 / 9,025 条
若直接用原生字段做趋势分析，等于把"NOAA 何时开始发布该字段"
这一纯技术史事件注入到气候趋势里 —— 这正是本研究要消除的那类假象。

因此：全部湿球温度由本站干球 + 露点（缺露点时用相对湿度）
经严格能量平衡方程自算，保证全站、全年代算法完全一致。

==========================================================================
湿球求解
==========================================================================
    T - Tw = (Lv/cp) * ( w_sat(Tw) - w )
即干球降到湿球放出的显热 == 蒸发补入水汽的潜热。
Lv/cp = 2418 K。用 Newton 迭代，解析导数，3 次即收敛到 1e-4 °C。

气压：GHCNh 仅有约 30% 记录含气压。为保证一致性，一律用
台站海拔按标准大气换算的**固定本站气压**：
    p = 1013.25 * (1 - 2.25577e-5 * h)^5.25588   (h 单位米)
湿球温度对气压不敏感（100 hPa 约 0.3 °C），此近似的影响可忽略。

加速：湿球只取决于 (T, Td, p)。p 每站固定，T/Td 报告精度 0.1 °C，
因此对 (round(T*10), round(Td*10)) 做记忆化，
把约 1,500 万次求解降到每站约 10 万次。

==========================================================================
抗观测频次混淆（沿用 v2）
==========================================================================
逐小时分箱 + 覆盖率归一化 + 严格年度质量门槛。
"""

import sys, os, math, csv, glob, json, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import duckdb

# 阈值阶梯：21.9 / 25.8 为 Perkins-Kirkpatrick (Nat Comms 2026) 的老年/青年下限，
# 30.6 为 PSU HEAT 实测青年临界 (30.55+-0.98)，35.0 为经典不可生存阈值。
# 追加 23 / 24 两档，用于**阈值敏感性分析**：
#   审稿人必然质疑"21.9 是日间暴晒阈值，能否用于夜间"。
#   阶梯分析可证明趋势在阈值取值上稳健，且能定位夜间负担究竟落在哪个温度区间。
TH = {"elderly": 21.9, "t23": 23.0, "t24": 24.0, "young": 25.8,
      "psu": 30.6, "classic": 35.0}
SUN_RISE_SET = -0.833
COV_MIN = 0.60
NIGHT_MIN_HOURS = 5
NIGHT_MIN_VALID = 250
DAY_MIN_HOURS = 18
HOURS_YEAR = 8760.0
LC = 2418.0          # Lv / cp  [K]


# ---------------- 湿球求解 ----------------
def esat(t):
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


def pressure_from_elev(h):
    if h is None or h < -500 or h > 6000:
        return 1013.25
    return 1013.25 * (1.0 - 2.25577e-5 * h) ** 5.25588


class WetBulbSolver:
    """按台站固定气压的湿球求解器，带记忆化。"""

    def __init__(self, p):
        self.p = p
        self.cache = {}

    def from_td(self, T, Td):
        """由干球 + 露点求湿球。"""
        if T is None or Td is None:
            return None
        if Td >= T:                       # 饱和或过饱和
            return T
        key = (int(round(T * 10)), int(round(Td * 10)))
        v = self.cache.get(key)
        if v is not None:
            return v
        p = self.p
        e = min(esat(Td), p * 0.95)
        w = 0.622 * e / (p - e)
        t = T - (T - Td) * 0.33          # 1/3 法则作初值
        for _ in range(4):
            es = min(esat(t), p * 0.95)
            den = p - es
            ws = 0.622 * es / den
            des = es * 4303.845 / ((t + 243.5) ** 2)
            dw = 0.622 * p * des / (den * den)
            g = LC * (ws - w) - (T - t)
            gp = LC * dw + 1.0
            step = g / gp
            t -= step
            if abs(step) < 1e-6:
                break
        if not (T - 90 < t <= T + 0.01):
            t = T
        self.cache[key] = t
        return t

    def from_rh(self, T, RH):
        """由干球 + 相对湿度求湿球（缺露点时回退）。"""
        if T is None or RH is None:
            return None
        rh = min(max(RH, 0.5), 100.0)
        if rh >= 99.9:
            return T
        e = esat(T) * rh / 100.0
        # 由水汽压反解露点，复用同一求解器
        Td = 243.5 * math.log(max(e, 1e-6) / 6.112) / (17.67 - math.log(max(e, 1e-6) / 6.112))
        return self.from_td(T, Td)


# ---------------- 太阳位置 ----------------
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
    return s[int(q * (len(s) - 1))]


# ---------------- 主分析 ----------------
def analyze(path, lat, lon, station, year, solver_cache=None):
    con = duckdb.connect()
    con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])"
                % path.replace("'", "''"))
    try:
        rows = con.execute("""
            SELECT CAST(DATE AS VARCHAR), temperature, dew_point_temperature,
                   TRY_CAST(relative_humidity AS DOUBLE), wet_bulb_temperature
            FROM h ORDER BY DATE
        """).fetchall()
    except Exception:
        return None
    # 海拔 -> 本站气压
    try:
        a = con.execute("SELECT MAX(ELEVATION) FROM h").fetchall()[0][0]
        elev = float(a) if a is not None else None
    except Exception:
        elev = None
    con.close()
    if len(rows) < 200:
        return None

    p = pressure_from_elev(elev)
    if solver_cache is not None:
        sv = solver_cache.get(station)
        if sv is None:
            sv = WetBulbSolver(p); solver_cache[station] = sv
    else:
        sv = WetBulbSolver(p)

    # ---- 逐小时分箱：对 T, Td, RH 分别取均值，再算湿球 ----
    bt, btd, brh, bn = {}, {}, {}, {}
    n_native = 0
    for dt, T, Td, RH, ntw in rows:
        try:
            y = int(dt[0:4]); mo = int(dt[5:7]); da = int(dt[8:10])
            hh = int(dt[11:13])
        except Exception:
            continue
        k = (y, mo, da, hh)
        if ntw is not None:
            n_native += 1
        if T is None:
            continue
        a = bt.get(k)
        if a is None: bt[k] = [T, 1]
        else: a[0] += T; a[1] += 1
        if Td is not None:
            b = btd.get(k)
            if b is None: btd[k] = [Td, 1]
            else: b[0] += Td; b[1] += 1
        elif RH is not None:
            c = brh.get(k)
            if c is None: brh[k] = [RH, 1]
            else: c[0] += RH; c[1] += 1
        bn[k] = bn.get(k, 0) + 1

    hours = []
    for k, (sT, nT) in bt.items():
        T = sT / nT
        Td = None
        if k in btd:
            Td = btd[k][0] / btd[k][1]
        RH = brh[k][0] / brh[k][1] if k in brh else None
        tw = sv.from_td(T, Td) if Td is not None else sv.from_rh(T, RH)
        if tw is None:
            continue
        y, mo, da, hh = k
        hours.append((doy_of(y, mo, da), hh + 0.5, tw, T, Td,
                      solar_elevation(lat, lon, doy_of(y, mo, da), hh + 0.5) < SUN_RISE_SET))
    hours.sort(key=lambda x: (x[0], x[1]))

    n_hours = len(hours)
    if n_hours < 500:
        return None

    tw_all = [h[2] for h in hours]
    night = [h for h in hours if h[5]]
    day = [h for h in hours if not h[5]]
    tw_n = [h[2] for h in night]
    tw_d = [h[2] for h in day]
    cov = n_hours / HOURS_YEAR
    f_night = len(night) / n_hours

    nights, cur = [], []
    for h in hours:
        if h[5]:
            cur.append(h)
        else:
            if len(cur) >= NIGHT_MIN_HOURS:
                nights.append(cur)
            cur = []
    if len(cur) >= NIGHT_MIN_HOURS:
        nights.append(cur)
    nvn = len(nights)

    # ---- 与流行病学阈值桥接: 夜间湿球越阈时, 干球最低温是多少? ----
    # 独立流行病学研究使用的判据是"干球夜间最低温 TN"
    # (巴塞罗那 TN>=24C 连续 3 天 RR~1.1; 日本全国 TN>=25C RR=1.09-1.10)。
    # 这里把物理阈值(湿球)映射到健康阈值(干球), 使两个体系可对齐。
    def _med(v):
        if not v:
            return None
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    all_tmin, tw_at_tmin, tmin_if = [], [], {}
    for th_key in ("elderly", "t24"):
        tmin_if[th_key] = []
    th_bridge = {"elderly": 21.9, "t24": 24.0}
    # 在"湿球越阈"的夜晚中, 有多少同时超过流行病学已验证的干球夜间阈值
    hit = {"elderly": [0, 0, 0]}      # [夜数, TN>=24, TN>=25]
    for nh in nights:
        w = [x[2] for x in nh]
        tv = [x[3] for x in nh]
        iw = min(range(len(w)), key=lambda i: w[i])
        it = min(range(len(tv)), key=lambda i: tv[i])
        tmin = tv[it]
        all_tmin.append(tmin)
        tw_at_tmin.append(w[it])
        for k, thv in th_bridge.items():
            if w[iw] >= thv:
                tmin_if[k].append(tmin)
        if w[iw] >= 21.9:
            hit["elderly"][0] += 1
            if tmin >= 24.0:
                hit["elderly"][1] += 1
            if tmin >= 25.0:
                hit["elderly"][2] += 1
    nhit = hit["elderly"][0]
    n_tn24 = sum(1 for v in all_tmin if v >= 24.0)
    n_tn25 = sum(1 for v in all_tmin if v >= 25.0)
    n_tn20 = sum(1 for v in all_tmin if v >= 20.0)

    byday = collections.defaultdict(list)
    for h in hours:
        byday[h[0]].append(h)
    dtr_t, dtr_tw = [], []
    for d, hs in byday.items():
        if len(hs) < DAY_MIN_HOURS:
            continue
        w = [x[2] for x in hs]
        tv = [x[3] for x in hs]
        dtr_tw.append(max(w) - min(w))
        if len(tv) >= DAY_MIN_HOURS:
            dtr_t.append(max(tv) - min(tv))

    def mean(x):
        return sum(x) / len(x) if x else None

    def ra(vals, th):
        if not vals:
            return None
        return sum(1 for v in vals if v >= th) / len(vals)

    r = dict(station=station, year=year, lat=lat, lon=lon, elev=elev,
             press=round(p, 1), n_obs=len(rows), n_hours=n_hours, coverage=round(cov, 4),
             n_native_tw=n_native,
             tw_max=round(max(tw_all), 2), tw_p99=round(pct(tw_all, 0.99), 2),
             tw_mean=round(mean(tw_all), 3),
             t_max=round(max(x[3] for x in hours), 2),
             t_mean=round(mean([x[3] for x in hours]), 3),
             tw_night_mean=round(mean(tw_n), 3) if tw_n else None,
             tw_day_mean=round(mean(tw_d), 3) if tw_d else None,
             tw_night_max=round(max(tw_n), 2) if tw_n else None,
             tw_day_max=round(max(tw_d), 2) if tw_d else None,
             t_night_mean=round(mean([h[3] for h in night]), 3) if night else None,
             t_day_mean=round(mean([h[3] for h in day]), 3) if day else None,
             td_night_mean=(round(mean([h[4] for h in night if h[4] is not None]), 3)
                            if any(h[4] is not None for h in night) else None),
             td_day_mean=(round(mean([h[4] for h in day if h[4] is not None]), 3)
                          if any(h[4] is not None for h in day) else None),
             dtr_t=round(mean(dtr_t), 3) if dtr_t else None,
             dtr_tw=round(mean(dtr_tw), 3) if dtr_tw else None,
             n_night_hours=len(night), n_day_hours=len(day),
             n_valid_nights=nvn, n_days_dtr=len(dtr_tw),
             tmin_night_med=round(_med(all_tmin), 2) if all_tmin else None,
             tw_at_tmin=round(_med(tw_at_tmin), 2) if tw_at_tmin else None,
             tmin_if_tw219=(round(_med(tmin_if["elderly"]), 2)
                            if tmin_if["elderly"] else None),
             tmin_if_tw240=(round(_med(tmin_if["t24"]), 2)
                            if tmin_if["t24"] else None),
             fn_tn20=round(n_tn20 / nvn, 4) if nvn else None,
             fn_tn24=round(n_tn24 / nvn, 4) if nvn else None,
             fn_tn25=round(n_tn25 / nvn, 4) if nvn else None,
             fn_tn24_if_twx=(round(hit["elderly"][1] / nhit, 4) if nhit else None),
             fn_tn25_if_twx=(round(hit["elderly"][2] / nhit, 4) if nhit else None))

    for k, th in TH.items():
        rr = ra(tw_all, th)
        r[f"rate_{k}"] = round(rr, 5) if rr is not None else None
        r[f"hrs_{k}"] = round(rr * HOURS_YEAR, 1) if rr is not None else None
        nr = ra(tw_n, th)
        r[f"nrate_{k}"] = round(nr, 5) if nr is not None else None
        r[f"nhrs_{k}"] = round(nr * HOURS_YEAR * f_night, 1) if nr is not None else None
        dr = ra(tw_d, th)
        r[f"drate_{k}"] = round(dr, 5) if dr is not None else None
        r[f"dhrs_{k}"] = round(dr * HOURS_YEAR * (1 - f_night), 1) if dr is not None else None
        # NRD: 夜间恢复亏缺 —— 夜间各小时超出阈值的平均幅度 (°C)。
        # 连续、无上界，解决 NRF 在热带饱和 (接近 1.0) 后失去分辨力的问题。
        if tw_n:
            exc = [max(0.0, v - th) for v in tw_n]
            r[f"nrd_{k}"] = round(sum(exc) / len(exc), 4)
        else:
            r[f"nrd_{k}"] = None
        # RELIEF: 每 24 小时内湿球低于阈值的小时数 —— 即"可恢复窗口"长度
        rr2 = ra([1.0 if v < th else 0.0 for v in tw_all], 0.5)
        r[f"relief_{k}"] = round(rr2 * 24.0, 3) if rr2 is not None else None
        # NRELIEF: 单个夜内湿球低于阈值的小时数
        if nvn:
            tot, cnt = 0, 0
            for nt in nights:
                tot += sum(1 for x in nt if x[2] < th)
                cnt += 1
            r[f"nrelief_{k}"] = round(tot / cnt, 3)
        else:
            r[f"nrelief_{k}"] = None
        # NRF: 夜间湿球最低值仍 >= 阈值的夜 占有效夜的比例
        # 二元判据（是否发生），在热带会饱和至 ~1.0，
        # 因此"严重程度"由上面无上界的 nrd / nrelief 承担。
        if nvn:
            nrf = sum(1 for nt in nights if min(x[2] for x in nt) >= th)
            r[f"nrf_{k}"] = nrf
            r[f"nrfr_{k}"] = round(nrf / nvn, 5)
            # 干球对照 NRF：用同一阈值判据作用于干球。
            # 因 Tw <= T 恒成立，必有 {min(Tw)>=th} ⊆ {min(T)>=th}，
            # 故 nrftr >= nrfr —— 即"干球在同一数值阈值下会判定更多夜晚未能恢复"。
            # 两者之差 = 干球判据与生理判据发生分歧的夜数占比（DISS）。
            nrft = sum(1 for nt in nights if min(x[3] for x in nt) >= th)
            r[f"nrft_{k}"] = nrft
            r[f"nrftr_{k}"] = round(nrft / nvn, 5)
            # DISS：干球判定"未恢复"而湿球判定"已恢复"的夜占比。
            # 这部分是干球指标相对于生理现实的**放宽/收紧分歧**，
            # 量化了"若改用干球温度计会得出多少不同的公共健康结论"。
            dis = sum(1 for nt in nights
                      if min(x[3] for x in nt) >= th and min(x[2] for x in nt) < th)
            r[f"diss_{k}"] = round(dis / nvn, 5)
        else:
            r[f"nrf_{k}"] = None
            r[f"nrfr_{k}"] = None
            r[f"nrft_{k}"] = None
            r[f"nrftr_{k}"] = None
            r[f"diss_{k}"] = None

    r["ok_trend"] = int(cov >= COV_MIN and nvn >= NIGHT_MIN_VALID)
    # ---- 不可用降温：干球日较差中人体无法利用的部分 ----
    if r["dtr_t"] is not None and r["dtr_tw"] is not None:
        r["unusable"] = round(r["dtr_t"] - r["dtr_tw"], 3)
        r["usable_frac"] = (round(r["dtr_tw"] / r["dtr_t"], 4)
                            if r["dtr_t"] > 0 else None)
    else:
        r["unusable"] = None
        r["usable_frac"] = None
    return r


def run(pattern="ghcn_data/GHCNh_*_*.parquet", meta_json="panel_all.json",
        out="metrics_v3.csv"):
    meta = {}
    if os.path.exists(meta_json):
        for m in json.load(open(meta_json, encoding="utf-8")):
            meta[m["station"]] = m
    files = sorted(glob.glob(pattern))
    print(f"待分析 {len(files)} 个文件")
    cache = {}
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
                lat, lon = float(a[0]), float(a[1]); c.close()
            except Exception:
                continue
        try:
            r = analyze(p, lat, lon, st, yr, cache)
        except Exception as e:
            print(f"  [{i}] {base} 错误 {str(e)[:60]}")
            continue
        if r is None:
            continue
        if m:
            r["name"] = m["name"]; r["country"] = m["country"]
            r["band"] = m.get("band", ""); r["region"] = m.get("region", "")
        rows.append(r)
        if i % 300 == 0:
            print(f"  进度 {i}/{len(files)}", flush=True)
    if rows:
        keys = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        nok = sum(r["ok_trend"] for r in rows)
        print(f"\n写出 {out}: {len(rows)} 条, 合格 {nok} ({100*nok/len(rows):.0f}%)")
    return rows


if __name__ == "__main__":
    run()
