#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤 1: 验证 GHCNh 原生 wet_bulb_temperature 字段的可信度

审稿人必问的问题：这个字段是实测还是推导？如果 NOAA 内部算法有偏差，
整篇论文建立在错误输入上。因此必须用**独立的物理公式**交叉验证。

方法:
  A. Stull (2011) 经验式  : Tw = f(T, RH)              —— 不依赖气压
  B. Davies-Jones 迭代式  : 从 T, RH, p 严格解湿球温度 —— 依赖气压，更严谨
  C. 饱和态检验           : RH=100% 时 Tw 必须 == T
  D. 单调性检验           : Tw <= T 恒成立
"""
import sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import duckdb


def stull_tw(T, RH):
    """Stull (2011) J. Appl. Meteor. Climatol. 湿球温度经验式。
    适用: RH 5~99%, T -20~50 C, 气压接近标准海压。"""
    if RH is None or T is None:
        return None
    rh = min(max(RH, 1.0), 100.0)
    t = T
    return (t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(t + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
            - 4.686035)


def esat(t):
    """Tetens 饱和水汽压 (hPa)"""
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


def davies_jones_tw(T, RH, p):
    """
    湿球温度的严格解（能量平衡 /  psychrometric relation）：

        T - Tw = (Lv / cp) * (w_sat(Tw) - w)

    即：干球降到湿球放出的显热 == 蒸发补入水汽所需的潜热。
    Lv/cp ≈ 2418 K (单位混合比)。用二分法求解，Tw ∈ [T-60, T]。

    注意早期版本误用 f = w_sat(Tw) - w，那解出的是**露点**，
    在干燥空气中会系统性偏低（菲尼克斯可差 10 °C 以上）。
    """
    if T is None or RH is None:
        return None
    p = p if (p is not None and 400 < p < 1100) else 1013.25
    e = esat(T) * min(max(RH, 0.1), 100.0) / 100.0
    e = min(e, p * 0.95)
    w = 0.622 * e / (p - e)

    LC = 2418.0  # Lv / cp  [K]

    def w_sat(t):
        es = esat(t)
        es = min(es, p * 0.95)
        return 0.622 * es / (p - es)

    def g(t):
        return LC * (w_sat(t) - w) - (T - t)

    lo, hi = T - 80.0, T + 0.5
    if g(lo) * g(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


def check(paths, label=""):
    con = duckdb.connect()
    if isinstance(paths, str):
        paths = [paths]
    lst = ", ".join("'" + p + "'" for p in paths)
    con.execute(f"CREATE VIEW h AS SELECT * FROM read_parquet([{lst}])")
    q = lambda s: con.execute(s).fetchall()

    cols = [c[0] for c in con.execute("DESCRIBE SELECT * FROM h").fetchall()]
    qc_cols = [c for c in cols if "wet_bulb" in c.lower()]
    print(f"\n{'='*74}")
    print(f"站点文件: {label or paths[0]}")
    print(f"湿球相关列: {qc_cols}")

    print("\n--- D. 单调性检验: Tw <= T 必须恒成立 ---")
    bad = q("SELECT COUNT(*) FROM h WHERE wet_bulb_temperature > temperature + 0.001")[0][0]
    tot = q("SELECT COUNT(*) FROM h WHERE wet_bulb_temperature IS NOT NULL "
            "AND temperature IS NOT NULL")[0][0]
    print(f"  违反 {bad:,} / {tot:,} = {100*bad/max(tot,1):.4f}%")

    print("\n--- C. 饱和态检验: RH>=99.5% 时 Tw 应约等于 T ---")
    r = q("""SELECT COUNT(*),
                     AVG(ABS(wet_bulb_temperature - temperature))
              FROM h WHERE TRY_CAST(relative_humidity AS DOUBLE) >= 99.5
                AND wet_bulb_temperature IS NOT NULL AND temperature IS NOT NULL""")[0]
    print(f"  饱和样本 {r[0]:,} 条, |Tw - T| 平均 {r[1] if r[1] is not None else 0:.4f} °C")

    print("\n--- A/B. 与独立公式交叉验证 (随机抽样 20000 条) ---")
    rows = q("""
        SELECT temperature, TRY_CAST(relative_humidity AS DOUBLE),
               station_level_pressure, sea_level_pressure, wet_bulb_temperature
        FROM h
        WHERE wet_bulb_temperature IS NOT NULL
          AND temperature IS NOT NULL
          AND TRY_CAST(relative_humidity AS DOUBLE) IS NOT NULL
        USING SAMPLE 20000 ROWS
    """)
    d_stull, d_dj, n = [], [], 0
    for T, RH, psl, pslp, tw in rows:
        if T is None or RH is None or tw is None:
            continue
        p = psl if psl is not None else pslp
        a = stull_tw(T, RH)
        b = davies_jones_tw(T, RH, p)
        if a is None or b is None:
            continue
        d_stull.append(tw - a)
        d_dj.append(tw - b)
        n += 1

    def stat(x, name):
        if not x:
            print(f"  {name}: 无有效样本")
            return
        m = sum(x) / len(x)
        sd = math.sqrt(sum((v - m) ** 2 for v in x) / len(x))
        sx = sorted(x)
        p50 = sx[len(sx) // 2]
        p95 = sx[int(len(sx) * 0.95)]
        p05 = sx[int(len(sx) * 0.05)]
        mx = max(abs(min(x)), abs(max(x)))
        print(f"  {name:<22} 偏差均值 {m:+.3f} °C  标准差 {sd:.3f}  "
              f"中位 {p50:+.3f}  [5%,95%] = [{p05:+.2f},{p95:+.2f}]  最大 |偏差| {mx:.2f}")
        print(f"  {'':<22} 样本 {len(x):,}")

    print(f"  (正偏差 = NOAA 原生值高于公式值)")
    stat(d_stull, "vs Stull(2011)")
    stat(d_dj, "vs Davies-Jones")

    # 判定
    print("\n--- 结论 ---")
    m_dj = sum(d_dj) / len(d_dj) if d_dj else 0
    sd_dj = math.sqrt(sum((v - m_dj) ** 2 for v in d_dj) / len(d_dj)) if d_dj else 0
    if abs(m_dj) < 0.5 and sd_dj < 1.0:
        print("  原生 wet_bulb_temperature 与独立物理公式一致，可直接使用。")
    else:
        print(f"  [警告] 偏差偏大 (均值 {m_dj:+.2f}, 标准差 {sd_dj:.2f})，"
              f"建议改用自算湿球温度以保证可比性。")
    return dict(stull=d_stull, dj=d_dj)


if __name__ == "__main__":
    files = [
        ("波士顿 USW00014739", "ghcn_test/bos_2023.parquet"),
        ("休斯顿 USW00012918", "ghcn_data/GHCNh_USW00012918_2023.parquet"),
        ("菲尼克斯 USW00023183", "ghcn_data/GHCNh_USW00023183_2023.parquet"),
    ]
    for lab, p in files:
        try:
            check(p, lab)
        except Exception as e:
            print(f"{lab} 失败: {e}")
