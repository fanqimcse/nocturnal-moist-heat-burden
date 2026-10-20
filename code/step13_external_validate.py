#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 13: 外部独立验证（回应审稿人 M26）

NASA POWER (MERRA-2 再分析) 不同化陆面站点气温/湿度观测,
因此构成对 GHCNh 站点分析的**准独立**交叉验证。

  A 层 · 算法与气候态验证 (2019-2023, 逐小时)
      在站点位置上用同一套太阳高度角定义与湿球算法算同样指标,
      与 GHCNh 结果逐站对比 -> 检验算法与量级。

  B 层 · 趋势验证 (1981-2023, 逐日)
      用逐日 T2M_MIN 与 T2MDEW 近似夜间最低湿球,
      计算 Theil-Sen 趋势, 与站点趋势比符号与量级。

输出: power_metrics.csv, external_validation.txt
"""
import sys, os, csv, json, math, time, collections, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step5c_metrics_v3 import WetBulbSolver, solar_elevation, pressure_from_elev, \
    doy_of, pct
from step6_trends import theil_sen, mann_kendall

HOURLY = ("https://power.larc.nasa.gov/api/temporal/hourly/point"
          "?parameters=T2M,T2MDEW&community=RE&longitude={lon}&latitude={lat}"
          "&start={s}&end={e}&format=JSON")
DAILY = ("https://power.larc.nasa.gov/api/temporal/daily/point"
         "?parameters=T2M_MIN,T2M_MAX,T2MDEW&community=RE"
         "&longitude={lon}&latitude={lat}&start={s}&end={e}&format=JSON")

CACHE = os.path.join(HERE, "power_cache")
os.makedirs(CACHE, exist_ok=True)


def fetch(url, tag, tries=4):
    path = os.path.join(CACHE, tag + ".json")
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return json.load(open(path, encoding="utf-8"))
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = r.read().decode("utf-8")
            if '"parameter"' not in txt and "messages" in txt:
                return None
            open(path, "w", encoding="utf-8").write(txt)
            return json.loads(txt)
        except Exception as e:
            last = e
            time.sleep(3 + 3 * k)
    print(f"    取数失败 {tag}: {str(last)[:60]}")
    return None


def median(v):
    if not v:
        return None
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def hourly_metrics(station, lat, lon, years, th=21.9):
    """用 POWER 逐小时数据算与站点完全同一套指标。"""
    sv = None
    elev = None
    hours = []          # (doy, utc_hour, tw, T, is_night)
    for y in years:
        j = fetch(HOURLY.format(lon=lon, lat=lat, s=f"{y}0101", e=f"{y}1231"),
                  f"{station}_{y}")
        if not j:
            continue
        pr = j["properties"]["parameter"]
        if elev is None:
            elev = float(j["geometry"]["coordinates"][2])
            sv = WetBulbSolver(pressure_from_elev(elev))
        t2m = pr.get("T2M", {}); td = pr.get("T2MDEW", {})
        for k, t in t2m.items():
            d = td.get(k)
            if t is None or d is None or t < -80 or d < -80:
                continue
            ymd, hh = k[:8], int(k[8:])
            y0 = int(ymd[:4]); mo = int(ymd[4:6]); da = int(ymd[6:8])
            doy = doy_of(y0, mo, da)
            tw = sv.from_td(float(t), float(d))
            if tw is None:
                continue
            hours.append((y0 * 1000 + doy, hh + 0.5, tw, float(t),
                          solar_elevation(lat, lon, doy, hh + 0.5) < 0))
    if len(hours) < 4000 or sv is None:
        return None
    hours.sort(key=lambda x: (x[0], x[1]))
    # 夜间分组
    nights, cur = [], []
    prev_key = None
    for h in hours:
        key = h[0]
        if prev_key is not None and key != prev_key and not h[4]:
            pass
        if h[4]:
            cur.append(h)
        else:
            if len(cur) >= 3:
                nights.append(cur)
            cur = []
        prev_key = key
    if len(cur) >= 3:
        nights.append(cur)
    tw_n = [h[2] for h in hours if h[4]]
    if not nights:
        return None
    nmins = [min(x[2] for x in nh) for nh in nights]
    # 日较差
    byday = collections.defaultdict(list)
    for h in hours:
        byday[h[0] // 1000 * 1000 + h[0] % 1000].append(h)
    dtr_tw, dtr_t = [], []
    for d, hs in byday.items():
        if len(hs) < 18:
            continue
        dtr_tw.append(max(x[2] for x in hs) - min(x[2] for x in hs))
        dtr_t.append(max(x[3] for x in hs) - min(x[3] for x in hs))
    nrfr = sum(1 for v in nmins if v >= th) / len(nmins)
    uf = (median(dtr_tw) / median(dtr_t)) if dtr_t and median(dtr_t) else None
    return {"elev": elev, "n_hours": len(hours), "n_nights": len(nights),
            "tw_night_mean": round(sum(tw_n) / len(tw_n), 3),
            "night_min_med": round(median(nmins), 3),
            "dtr_tw": round(median(dtr_tw), 3) if dtr_tw else None,
            "dtr_t": round(median(dtr_t), 3) if dtr_t else None,
            "nrfr": round(nrfr, 4), "usable_frac": round(uf, 4) if uf else None}


def daily_series(station, lat, lon, y0=1981, y1=2023):
    """POWER 逐日: 用 T2M_MIN 与日均露点近似夜间最低湿球。"""
    j = fetch(DAILY.format(lon=lon, lat=lat, s=f"{y0}0101", e=f"{y1}1231"),
              f"{station}_daily_{y0}_{y1}")
    if not j:
        return None
    pr = j["properties"]["parameter"]
    elev = float(j["geometry"]["coordinates"][2])
    sv = WetBulbSolver(pressure_from_elev(elev))
    tmin = pr.get("T2M_MIN", {}); td = pr.get("T2MDEW", {})
    byyear = collections.defaultdict(list)
    for k, t in tmin.items():
        d = td.get(k)
        if t is None or d is None or t < -80 or d < -80:
            continue
        tw = sv.from_td(float(t), float(d))
        if tw is None:
            continue
        byyear[int(k[:4])].append(tw)
    out = {}
    for y, v in byyear.items():
        if len(v) >= 300:
            out[y] = sum(v) / len(v)          # 年均夜间最低湿球 (近似)
    return out


def pick_stations(n=16):
    """从已有站点中按纬度带与区域分层抽取。"""
    rows = list(csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"),
                                    encoding="utf-8")))
    info = {}
    for r in rows:
        info[r["station"]] = (float(r["lat"]), float(r["lon"]))
    panel = {m["station"]: m for m in
             json.load(open(os.path.join(HERE, "panel_all.json"), encoding="utf-8"))}
    # 按纬度带分组, 每组内按经度排序后等间隔取样
    groups = collections.defaultdict(list)
    for st, (la, lo) in info.items():
        b = panel.get(st, {}).get("band") or (
            "tropics" if abs(la) <= 23.5 else "subtropics" if abs(la) <= 35 else "temperate")
        groups[b].append((lo, st))
    picks = []
    for b, lst in sorted(groups.items()):
        lst.sort()
        k = max(4, n // len(groups))
        step = max(1, len(lst) // k)
        picks += [st for _, st in lst[::step][:k]]
    return picks[:n], info, panel


def main():
    picks, info, panel = pick_stations(18)
    print(f"选取 {len(picks)} 个站点做外部验证")

    # ---------- A 层: 气候态 ----------
    print("\n" + "=" * 88)
    print("【外部验证 A】POWER 逐小时 vs GHCNh 站点 (2019-2023 平均)")
    print("=" * 88)
    print(f"{'站点':<14}{'纬度':>7}{'夜间Tw 站':>10}{'夜间Tw POWER':>13}"
          f"{'差':>8}{'NRF 站':>9}{'NRF POWER':>11}{'可用比 站':>10}{'可用比 POWER':>13}")
    print("-" * 88)

    # GHCNh 侧: 2019-2023 平均
    gh = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"), encoding="utf-8")):
        if 2019 <= int(r["year"]) <= 2023 and r["station"] in picks:
            for k in ("tw_night_mean", "nrfr_elderly", "usable_frac", "dtr_tw"):
                try:
                    gh[r["station"]][k].append(float(r[k]))
                except Exception:
                    pass

    rowsA = []
    for st in picks:
        la, lo = info[st]
        m = hourly_metrics(st, la, lo, [2019, 2020, 2021, 2022, 2023])
        if not m:
            print(f"{st:<14}{la:>7.1f}  — POWER 无数据")
            continue
        g = gh.get(st, {})
        gtw = median(g.get("tw_night_mean", []))
        gnr = median(g.get("nrfr_elderly", []))
        guf = median(g.get("usable_frac", []))
        rowsA.append((st, la, gtw, m["tw_night_mean"], gnr, m["nrfr"],
                      guf, m["usable_frac"], m))
        print(f"{st:<14}{la:>7.1f}"
              f"{(gtw if gtw else float('nan')):>10.2f}{m['tw_night_mean']:>13.2f}"
              f"{(m['tw_night_mean']-gtw if gtw else float('nan')):>8.2f}"
              f"{(gnr if gnr is not None else float('nan')):>9.3f}{m['nrfr']:>11.3f}"
              f"{(guf if guf else float('nan')):>10.3f}"
              f"{(m['usable_frac'] if m['usable_frac'] else float('nan')):>13.3f}")

    def corr(a, b):
        p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(p) < 5:
            return None
        ma = sum(x for x, _ in p) / len(p); mb = sum(y for _, y in p) / len(p)
        sa = sum((x - ma) ** 2 for x, _ in p) ** 0.5
        sb = sum((y - mb) ** 2 for _, y in p) ** 0.5
        if sa <= 0 or sb <= 0:
            return None
        return sum((x - ma) * (y - mb) for x, y in p) / (sa * sb)

    if rowsA:
        r1 = corr([x[2] for x in rowsA], [x[3] for x in rowsA])
        r2 = corr([x[4] for x in rowsA], [x[5] for x in rowsA])
        r3 = corr([x[6] for x in rowsA], [x[7] for x in rowsA])
        bias = median([x[3] - x[2] for x in rowsA if x[2] is not None])
        print("-" * 88)
        print(f"夜间湿球均值   站间相关 r = {r1:.3f}, POWER 偏差中位 {bias:+.2f} °C")
        print(f"夜间恢复失效率 站间相关 r = {r2:.3f}")
        print(f"可用降温比例   站间相关 r = {r3:.3f}")

    # ---------- B 层: 趋势 ----------
    print("\n" + "=" * 88)
    print("【外部验证 B】1981-2023 夜间最低湿球趋势：POWER 逐日 vs GHCNh 站点")
    print("=" * 88)
    print(f"{'站点':<14}{'纬':>6}{'站点斜率':>11}{'POWER斜率':>11}"
          f"{'POWER 95%CI':>24}{'同号':>6}")
    print("-" * 88)
    gh_series = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"), encoding="utf-8")):
        if r["station"] in picks:
            try:
                gh_series[r["station"]][int(r["year"])] = float(r["tw_night_mean"])
            except Exception:
                pass
    agree = 0; tot = 0
    slopes_s = []; slopes_p = []
    for st in picks:
        la, lo = info[st]
        ds = daily_series(st, la, lo)
        if not ds or len(ds) < 20:
            print(f"{st:<14}{la:>6.1f}   POWER 逐日无数据")
            continue
        ys = sorted(ds); vs = [ds[y] for y in ys]
        sp = theil_sen(ys, vs)
        # bootstrap CI
        import random as _rnd
        _rnd.seed(7)
        bs = []
        for _ in range(400):
            idx = [_rnd.randrange(len(ys)) for _ in range(len(ys))]
            yy = [ys[i] for i in idx]; vv = [vs[i] for i in idx]
            if len(set(yy)) > 3:
                b = theil_sen(yy, vv)
                if b is not None:
                    bs.append(b)
        bs.sort()
        lo_ci = bs[int(0.025 * len(bs))] if bs else None
        hi_ci = bs[int(0.975 * len(bs))] if bs else None
        g = gh_series.get(st, {})
        ys2 = sorted(y for y in g if y >= 1981)
        ss = theil_sen(ys2, [g[y] for y in ys2]) if len(ys2) >= 5 else None
        same = (ss is not None and sp is not None and
                ((ss > 0 and sp > 0) or (ss < 0 and sp < 0)))
        if ss is not None and sp is not None:
            tot += 1; agree += 1 if same else 0
            slopes_s.append(ss); slopes_p.append(sp)
        print(f"{st:<14}{la:>6.1f}"
              f"{(ss if ss is not None else float('nan')):>11.4f}{sp:>11.4f}"
              f"   [{lo_ci:+.4f}, {hi_ci:+.4f}]"
              f"{'是' if same else '否':>6}")
    if tot:
        print("-" * 88)
        print(f"趋势同号率 {agree}/{tot} = {100*agree/tot:.0f}%")
        print(f"站点中位斜率 {median(slopes_s):+.4f} °C/年, "
              f"POWER 中位斜率 {median(slopes_p):+.4f} °C/年")
        print(f"两者相关 r = {corr(slopes_s, slopes_p):.3f}")

    with open(os.path.join(HERE, "power_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "lat", "lon", "gh_tw_night", "pw_tw_night",
                    "gh_nrfr", "pw_nrfr", "gh_uf", "pw_uf",
                    "gh_slope", "pw_slope"])
    print("\n已写出 power_metrics.csv (框架)")


if __name__ == "__main__":
    main()
