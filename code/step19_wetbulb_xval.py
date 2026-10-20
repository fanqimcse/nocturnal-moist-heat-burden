#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R7-2（回应 M37）: 湿球算法三法交叉验证

本研究全部结论是一个"阈值跨越计数"，对 0.1 C 级系统偏差极敏感。
必须回答：在 21.9-26 C 这个决定性区间里，三种算法差多少？

  A. Davies-Jones 型 Newton 迭代 (T, Td, p)   —— 本研究采用
  B. Stull (2011) 经验式 (T, RH)              —— 不依赖气压，广泛引用
  C. GHCNh 原生 wet_bulb_temperature 字段     —— NCEI 内部算法

并做阈值敏感性传播：把 NRF 在 A/B/C 三种算法下各算一遍，
直接看"算法选择"是否改变结论。
"""
import sys, os, json, math, glob, random, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step5c_metrics_v3 as S
from step6_trends import median


def stull_correct(T, RH):
    """Stull (2011) 经验式, J. Appl. Meteor. Climatol. 50(11):
        Tw = T*atan[0.151977*(RH+8.313659)^0.5] + atan(T+RH) - atan(RH-1.676331)
             + 0.00391838*RH^1.5*atan(0.023101*RH) - 4.686035
    RH 以百分数(5-99)代入, 反三角函数取弧度, 结果摄氏度。
    适用: RH 5-99%, T -20~50 C, 气压接近标准海压; 标称精度 +-0.3 C。
    """
    rh = min(max(RH, 5.0), 99.0)
    return (T * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + math.atan(T + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def main():
    print("=" * 96)
    print("R7-2 / M37  湿球算法三法交叉验证（决定性区间 21.9-26 C）")
    print("=" * 96)

    meta = {m["station"]: m for m in
            json.load(open(os.path.join(HERE, "panel_all.json"), encoding="utf-8"))}

    # 抽样：跨区域、跨年代的站年
    files = sorted(glob.glob(os.path.join(HERE, "ghcn_data", "GHCNh_*_*.parquet")))
    by = collections.defaultdict(list)
    for f in files:
        b = os.path.basename(f).replace(".parquet", "").split("_")
        if len(b) >= 3:
            by[b[1]].append((int(b[2]), f))
    sts = sorted(by)
    random.seed(7)
    sample = []
    # 每站取最早与最晚两年，最多 60 站
    for st in sts:
        ys = sorted(y for y, _ in by[st])
        if len(ys) < 20:
            continue
        sample.append((st, ys[0], by[st][0][1]))
        sample.append((st, ys[-1], by[st][-1][1]))
    if len(sample) > 160:
        sample = random.sample(sample, 160)

    import duckdb
    bins = [(18, 20), (20, 21.9), (21.9, 23), (23, 24), (24, 25), (25, 26), (26, 28)]
    acc = {k: {"dj_st": [], "dj_nat": [], "n": 0} for k in bins}
    tot = {"dj_st": [], "dj_nat": [], "n_nat": 0}
    # 阈值计数（夜间最低湿球是否 >= 21.9）
    nrf = {"dj": [0, 0], "stull": [0, 0], "nat": [0, 0]}

    for st, yr, f in sample:
        m = meta.get(st, {})
        lat, lon = m.get("lat"), m.get("lon")
        con = duckdb.connect()
        con.execute("CREATE VIEW h AS SELECT * FROM read_parquet(['%s'])" % f)
        try:
            rows = con.execute("""SELECT CAST(DATE AS VARCHAR), temperature,
                dew_point_temperature, TRY_CAST(relative_humidity AS DOUBLE),
                wet_bulb_temperature FROM h WHERE temperature IS NOT NULL""").fetchall()
            ev = con.execute("SELECT MAX(ELEVATION) FROM h").fetchall()[0][0]
            if lat is None:
                a = con.execute("SELECT MAX(LATITUDE), MAX(LONGITUDE) FROM h").fetchall()[0]
                lat, lon = float(a[0]), float(a[1])
        except Exception:
            con.close(); continue
        con.close()
        if len(rows) < 200 or lat is None:
            continue
        p = S.pressure_from_elev(float(ev) if ev is not None else None)
        sv = S.WetBulbSolver(p)

        # 夜间逐夜最低湿球（三种算法）
        nightly = collections.defaultdict(lambda: {"dj": [], "st": [], "nat": []})
        for dt, T, Td, RH, ntw in rows:
            try:
                y = int(dt[0:4]); mo = int(dt[5:7]); da = int(dt[8:10]); hh = int(dt[11:13])
            except Exception:
                continue
            doy = S.doy_of(y, mo, da)
            if S.solar_elevation(lat, lon, doy, hh + 0.5) >= S.SUN_RISE_SET:
                continue
            tw = sv.from_td(T, Td) if Td is not None else sv.from_rh(T, RH)
            if tw is None:
                continue
            k = (y, mo, da) if hh >= 12 else (y, mo, da)
            nightly[k]["dj"].append(tw)
            if RH is not None:
                nightly[k]["st"].append(stull_correct(T, RH))
            if ntw is not None:
                nightly[k]["nat"].append(float(ntw))
            # 分箱（逐小时，不重复计夜）
            for lo, hi in bins:
                if lo <= tw < hi:
                    if RH is not None:
                        acc[(lo, hi)]["dj_st"].append(tw - stull_correct(T, RH))
                    if ntw is not None:
                        acc[(lo, hi)]["dj_nat"].append(tw - float(ntw))
                    acc[(lo, hi)]["n"] += 1
                    break

        for k, d in nightly.items():
            if not d["dj"]:
                continue
            nrf["dj"][0] += 1
            if min(d["dj"]) >= 21.9:
                nrf["dj"][1] += 1
            if d["st"]:
                nrf["stull"][0] += 1
                if min(d["st"]) >= 21.9:
                    nrf["stull"][1] += 1
            if d["nat"]:
                nrf["nat"][0] += 1
                if min(d["nat"]) >= 21.9:
                    nrf["nat"][1] += 1

    print(f"\n参与站年: {len(sample)}   夜间小时分箱样本量:")
    print(f"\n{'DJ湿球区间 (C)':<18}{'样本小时数':>12}{'DJ-Stull 偏差':>18}{'DJ-原生 偏差':>18}")
    print("-" * 96)
    for lo, hi in bins:
        d = acc[(lo, hi)]
        a = median(d["dj_st"]) if d["dj_st"] else None
        b = median(d["dj_nat"]) if d["dj_nat"] else None
        sa = f"{a:+.3f} C" if a is not None else "  n/a"
        sb = f"{b:+.3f} C" if b is not None else "  n/a"
        print(f"{f'{lo}~{hi}':<18}{d['n']:>12}{sa:>18}{sb:>18}")
    print("-" * 96)
    print("  注: 偏差 = 本研究算法(DJ) - 对照算法;  正值表示本研究算法偏暖。")

    print(f"\n【阈值敏感性传播】夜间最低湿球 >= 21.9 C 的夜数占比 (NRF):")
    print("-" * 96)
    for k, lab in [("dj", "DJ 迭代 (本研究)"), ("stull", "Stull 2011 经验式"),
                   ("nat", "GHCNh 原生字段")]:
        n, x = nrf[k]
        print(f"  {lab:<24} {x}/{n} = {x/n:.4f}" if n else f"  {lab:<24} 无样本")
    base = nrf["dj"][1] / nrf["dj"][0] if nrf["dj"][0] else 0
    if nrf["stull"][0]:
        d1 = nrf["stull"][1] / nrf["stull"][0] - base
        print(f"\n  算法选择造成的 NRF 差异: Stull - DJ = {d1:+.4f} "
              f"({100*d1/max(base,1e-9):+.1f}% 相对)")
    if nrf["nat"][0]:
        d2 = nrf["nat"][1] / nrf["nat"][0] - base
        print(f"                          原生 - DJ = {d2:+.4f} "
              f"({100*d2/max(base,1e-9):+.1f}% 相对)")

    print("\n【结论】见报告。判据: 若关键区间内 |DJ-Stull| < 0.5 C 且")
    print("      NRF 相对差异 < 15%, 则算法选择不改变结论。")


if __name__ == "__main__":
    main()
