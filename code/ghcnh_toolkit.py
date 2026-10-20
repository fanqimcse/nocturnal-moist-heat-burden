#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GHCNh 热应激分析工具包
======================

针对 NOAA GHCNh (Global Historical Climatology Network - hourly) v1.1.0
DOI: 10.25921/jp3d-3v19

解决三个实测踩到的坑：
  1. NCEI / AWS 连接会静默截断（HTTP 200 但文件不全）→ 断点续传 + Content-Length 校验
  2. 带宽极低（实测约 5-7 KB/s）→ 强制走 Parquet（相对 PSV 压缩比约 10:1）
  3. relative_humidity 在 Parquet 中是 VARCHAR → 一律 TRY_CAST

依赖: pip install duckdb
用法:
    python ghcnh_toolkit.py download USW00014739 2023
    python ghcnh_toolkit.py analyze ghcn_data/GHCNh_USW00014739_2023.parquet
    python ghcnh_toolkit.py stations
"""

import os
import sys
import time
import urllib.request
import urllib.error

BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access"
DATA_DIR = "ghcn_data"

# 分人群湿球温度生理阈值（°C）
#
# 出处核对（2026-10-20 第 27 轮审计修正）：
#   21.9 / 25.8 的原始出处是 Vanos et al. 2023, Nature Communications 14, 7653,
#   DOI 10.1038/s41467-023-43121-5（HEAT-Lim 模型）。摘要原文：
#     "Updated survivability limits correspond to Tw~25.8-34.1 °C (young) and
#      ~21.9-33.7 °C (old) - 0.9-13.1 °C lower than Tw = 35 °C."
#   即 21.9 是【老年人在湿热端的生存极限下限】，25.8 是【青年人湿热端下限】。
#   Perkins-Kirkpatrick et al. 2026 (Nat Commun 17:2590, 10.1038/s41467-026-70485-1)
#   是【应用】HEAT-Lim 的后续研究，不是 21.9/25.8 的原始出处——此前误标已修正。
#   注意：35.0 来自 Sherwood & Huber 2010 (PNAS 107:9552)，与 21.9 无关，勿混用。
THRESHOLDS = [
    (35.0, "经典不可生存阈值 (Sherwood & Huber 2010, PNAS 107:9552)"),
    (30.6, "健康青年实测临界值 (PSU HEAT, Vecellio et al. 2022)"),
    (25.8, "青年成人湿热端生存下限 (Vanos et al. 2023, Nat Commun 14:7653)"),
    (21.9, "老年人湿热端生存下限 (Vanos et al. 2023, Nat Commun 14:7653)"),
]

# 致死判据：湿球需持续约 6 小时（Sherwood & Huber 2010）
DURATION_CRITERION_HOURS = 6.0


# ----------------------------------------------------------------------------
# 下载：断点续传 + 完整性校验
# ----------------------------------------------------------------------------

def _head_size(url):
    """取远端文件字节数。失败返回 None。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            v = r.headers.get("Content-Length")
            return int(v) if v else None
    except Exception:
        return None


def download_station_year(station, year, fmt="parquet", data_dir=DATA_DIR,
                          max_retries=25, verbose=True):
    """
    下载单个站点单年数据，支持断点续传。

    关键：实测中 NCEI 与 AWS 都会在传输中途断开，且 HTTP 状态码仍为 200，
    因此每次下载后必须与 Content-Length 比对，不一致就续传。
    """
    os.makedirs(data_dir, exist_ok=True)
    ext = "parquet" if fmt == "parquet" else "psv"
    fname = f"GHCNh_{station}_{year}.{ext}"
    path = os.path.join(data_dir, fname)
    url = f"{BASE}/by-year/{year}/{fmt}/{fname}"

    remote = _head_size(url)
    if remote is None:
        print(f"  [失败] 无法访问 {url}")
        return None
    if verbose:
        print(f"  远端大小: {remote:,} 字节")

    for attempt in range(1, max_retries + 1):
        cur = os.path.getsize(path) if os.path.exists(path) else 0
        if cur == remote:
            if verbose:
                print(f"  [完成] {fname} ({remote:,} 字节)")
            return path
        if cur > remote:
            os.remove(path)
            cur = 0

        headers = {"Range": f"bytes={cur}-"} if cur > 0 else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            mode = "ab" if cur > 0 else "wb"
            with urllib.request.urlopen(req, timeout=180) as r, open(path, mode) as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            if verbose:
                print(f"  第 {attempt} 次中断 ({type(e).__name__})，已下载 "
                      f"{os.path.getsize(path) if os.path.exists(path) else 0:,} 字节")

        got = os.path.getsize(path) if os.path.exists(path) else 0
        if verbose and got < remote:
            pct = 100 * got / remote
            print(f"  第 {attempt} 次后进度: {got:,} / {remote:,} ({pct:.1f}%)")

    got = os.path.getsize(path) if os.path.exists(path) else 0
    if got != remote:
        print(f"  [未完成] {got:,} / {remote:,}，请重跑本命令续传")
        return None
    if verbose:
        print(f"  [完成] {fname} ({remote:,} 字节)")
    return path


# ----------------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------------

def _conn(paths):
    import duckdb
    con = duckdb.connect()
    if isinstance(paths, str):
        paths = [paths]
    lst = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    con.execute(f"CREATE VIEW h AS SELECT * FROM read_parquet([{lst}])")
    return con


def coverage(con):
    """各变量非空率。"""
    q = lambda s: con.execute(s).fetchall()
    total = q("SELECT COUNT(*) FROM h")[0][0]
    print(f"\n总观测行数: {total:,}")
    print("\n变量非空率:")
    print(f"  {'变量':<26}{'非空':>10}{'占比':>9}")
    for c in ["temperature", "dew_point_temperature", "relative_humidity",
              "wet_bulb_temperature", "wind_speed", "wind_gust",
              "sea_level_pressure", "precipitation", "snow_depth"]:
        try:
            n = q(f'SELECT COUNT("{c}") FROM h')[0][0]
            print(f"  {c:<26}{n:>10,}{100*n/total:>8.1f}%")
        except Exception:
            print(f"  {c:<26}{'—':>10}{'—':>9}")
    return total


def qc_check(con):
    """物理一致性检验：湿球必须 <= 干球。"""
    q = lambda s: con.execute(s).fetchall()
    bad = q("SELECT COUNT(*) FROM h WHERE wet_bulb_temperature > temperature + 0.001")[0][0]
    tot = q("SELECT COUNT(*) FROM h "
            "WHERE wet_bulb_temperature IS NOT NULL AND temperature IS NOT NULL")[0][0]
    rate = 100 * bad / tot if tot else 0.0
    print(f"\n物理一致性检验 (Tw <= T):")
    print(f"  违反记录 {bad:,} / {tot:,} = {rate:.3f}%")
    if rate > 1.0:
        print("  [警告] 违反率偏高，建议检查该站数据质量")
    return rate


def threshold_hours(con):
    """按分人群阈值统计超标小时数与折合天数。"""
    q = lambda s: con.execute(s).fetchall()
    r = q("SELECT MAX(wet_bulb_temperature), MIN(wet_bulb_temperature), "
          "AVG(wet_bulb_temperature) FROM h WHERE wet_bulb_temperature IS NOT NULL")[0]
    t = q("SELECT MAX(temperature), MIN(temperature) FROM h WHERE temperature IS NOT NULL")[0]
    print(f"\n年内极值:")
    print(f"  湿球: 最高 {r[0]:.1f} °C   最低 {r[1]:.1f} °C   年均 {r[2]:.1f} °C")
    print(f"  干球: 最高 {t[0]:.1f} °C   最低 {t[1]:.1f} °C")

    print(f"\n{'阈值':>7}  {'含义':<40}{'小时数':>9}{'折合天':>9}")
    out = {}
    for th, lab in THRESHOLDS:
        n = q(f"SELECT COUNT(*) FROM h WHERE wet_bulb_temperature >= {th}")[0][0]
        out[th] = n
        print(f"  {th:>5.1f}  {lab:<40}{n:>9,}{n/24:>9.1f}")
    return out


def longest_events(con, threshold=21.9, top=5):
    """
    连续超标事件。这是日值数据集物理上无法计算的量。

    致死判据要求持续约 6 小时，因此必须识别连续时段而非简单计数。
    """
    q = lambda s: con.execute(s).fetchall()
    rows = q(f"""
        WITH f AS (
            SELECT DATE, wet_bulb_temperature AS tw,
                   CASE WHEN wet_bulb_temperature >= {threshold} THEN 1 ELSE 0 END AS hot
            FROM h WHERE wet_bulb_temperature IS NOT NULL ORDER BY DATE
        ),
        g AS (
            SELECT DATE, tw, hot,
                   SUM(CASE WHEN hot = 0 THEN 1 ELSE 0 END)
                       OVER (ORDER BY DATE ROWS UNBOUNDED PRECEDING) AS grp
            FROM f
        ),
        r AS (
            SELECT grp, COUNT(*) AS n,
                   MIN(DATE) AS s, MAX(DATE) AS e, MAX(tw) AS peak,
                   date_diff('hour', TRY_CAST(MIN(DATE) AS TIMESTAMP),
                                     TRY_CAST(MAX(DATE) AS TIMESTAMP)) AS hrs
            FROM g WHERE hot = 1 GROUP BY grp
        )
        SELECT n, s, e, peak, hrs FROM r ORDER BY n DESC LIMIT {top}
    """)
    print(f"\n最长连续事件 (湿球 >= {threshold} °C):")
    print(f"  致死判据参考: 持续 >= {DURATION_CRITERION_HOURS:.0f} 小时")
    if not rows:
        print("  无")
        return rows
    print(f"  {'持续观测':>8}  {'起始':<18}{'结束':<18}{'持续':>7}{'峰值':>9}")
    for n, s, e, p, hrs in rows:
        hrs = float(hrs) if hrs is not None else float("nan")
        flag = "  <- 超致死判据" if hrs >= DURATION_CRITERION_HOURS else ""
        print(f"  {n:>8,}  {str(s)[:16]:<18}{str(e)[:16]:<18}"
              f"{hrs:>6.0f}h{p:>8.1f} °C{flag}")
    return rows


def daily_smoothing_bias(con):
    """量化日平均相对小时峰值的低估幅度。"""
    q = lambda s: con.execute(s).fetchall()
    d = q("""
        WITH daily AS (
            SELECT substr(CAST(DATE AS VARCHAR), 1, 10) AS d,
                   MAX(wet_bulb_temperature) AS mx,
                   AVG(wet_bulb_temperature) AS av
            FROM h WHERE wet_bulb_temperature IS NOT NULL GROUP BY 1
        )
        SELECT MAX(mx), MAX(av), AVG(mx - av) FROM daily
    """)[0]
    print(f"\n日值相对小时值的抹平效应:")
    print(f"  小时湿球年内最高    : {d[0]:.1f} °C")
    print(f"  日平均湿球的最高日值: {d[1]:.1f} °C")
    print(f"  日最高-日平均 均值  : {d[2]:.1f} °C")
    print(f"  -> 用日平均会低估峰值 {d[0] - d[1]:.1f} °C")
    return d[0] - d[1]


def wbgt_stats(con):
    """
    湿球黑球温度 (WBGT)，户外劳动保护的标准指标。
    简化式（遮阴近似，忽略太阳辐射项）: WBGT = 0.7*Tw + 0.3*Tg
    无黑球温度时用干球近似黑球：WBGT ≈ 0.7*Tw + 0.3*T
    """
    q = lambda s: con.execute(s).fetchall()
    try:
        r = q("""SELECT MAX(0.7*wet_bulb_temperature + 0.3*temperature),
                        AVG(0.7*wet_bulb_temperature + 0.3*temperature)
                 FROM h WHERE wet_bulb_temperature IS NOT NULL
                       AND temperature IS NOT NULL""")[0]
    except Exception as e:
        print(f"\nWBGT 计算失败: {e}")
        return None
    print(f"\n湿球黑球温度 WBGT (0.7*Tw + 0.3*T 近似):")
    print(f"  年内最高 {r[0]:.1f} °C   年均 {r[1]:.1f} °C")
    print("  参考: NIOSH 中度劳动上限约 28 °C；重度劳动约 26 °C")
    for th in [26.0, 28.0, 30.0]:
        n = q(f"""SELECT COUNT(*) FROM h
                  WHERE 0.7*wet_bulb_temperature + 0.3*temperature >= {th}
                    AND wet_bulb_temperature IS NOT NULL""")[0][0]
        print(f"  WBGT >= {th:.0f} °C : {n:,} 小时 ({n/24:.1f} 天)")
    return r


def analyze(paths):
    con = _conn(paths)
    coverage(con)
    qc_check(con)
    threshold_hours(con)
    longest_events(con)
    daily_smoothing_bias(con)
    wbgt_stats(con)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "download":
        station, year = sys.argv[2], sys.argv[3]
        fmt = sys.argv[4] if len(sys.argv) > 4 else "parquet"
        print(f"下载 {station} {year} ({fmt})")
        download_station_year(station, year, fmt)
    elif cmd == "analyze":
        analyze(sys.argv[2:])
    elif cmd == "batch":
        # python ghcnh_toolkit.py batch stations.txt 2023
        st_file, year = sys.argv[2], sys.argv[3]
        sts = [l.strip() for l in open(st_file, encoding="utf-8") if l.strip()]
        print(f"批量下载 {len(sts)} 站, 年份 {year}")
        ok = 0
        for i, s in enumerate(sts, 1):
            print(f"[{i}/{len(sts)}] {s}")
            if download_station_year(s, year, verbose=False):
                ok += 1
                print(f"  完成 ({ok}/{len(sts)})")
            else:
                print("  未完成")
        print(f"共完成 {ok}/{len(sts)}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
