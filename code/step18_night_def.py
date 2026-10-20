#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R7-1（回应 M35）: 夜间定义敏感性检验

全部结论建立在"太阳高度角 < -0.833 度"这一个夜间定义上。
改用民用曙暮光(-6 度)会切掉黎明前最后约 25 分钟与日落后最初 25 分钟——
而夜间最低湿球恰常出现在黎明前。必须量化这一影响。

做法: 对合格年 >= 30 的站, 分别用两种夜间定义重算全部指标,
      逐站比较 Theil-Sen 斜率与水平。
"""
import sys, os, json, glob, csv, collections
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step5c_metrics_v3 as S
from step6_trends import theil_sen, sign_test, median

METRICS = [
    ("tw_night_mean", "夜间湿球均值"),
    ("dtr_tw", "湿球日较差"),
    ("usable_frac", "可用降温比例"),
    ("nrfr_elderly", "夜间恢复失效率"),
    ("nrd_elderly", "夜间恢复亏缺"),
    ("relief_elderly", "每日可恢复窗口"),
    ("nhrs_elderly", "夜间超阈小时"),
]


def load_qualified(min_years=30):
    rows = list(csv.DictReader(open(os.path.join(HERE, "metrics_v3.csv"), encoding="utf-8")))
    per = collections.Counter(r["station"] for r in rows if r.get("ok_trend") == "1")
    return sorted([s for s, n in per.items() if n >= min_years])


def run_def(stations, thresh, tag):
    S.SUN_RISE_SET = thresh
    meta = {m["station"]: m for m in
            json.load(open(os.path.join(HERE, "panel_all.json"), encoding="utf-8"))}
    out = {}
    cache = {}
    for st in stations:
        m = meta.get(st, {})
        lat, lon = m.get("lat"), m.get("lon")
        recs = []
        for p in sorted(glob.glob(os.path.join(HERE, "ghcn_data", f"GHCNh_{st}_*.parquet"))):
            yr = int(os.path.basename(p).replace(".parquet", "").split("_")[2])
            if lat is None:
                import duckdb
                c = duckdb.connect()
                c.execute("CREATE VIEW z AS SELECT * FROM read_parquet(['%s'])" % p)
                a = c.execute("SELECT MAX(LATITUDE), MAX(LONGITUDE) FROM z").fetchall()[0]
                lat, lon = float(a[0]), float(a[1]); c.close()
            try:
                r = S.analyze(p, lat, lon, st, yr, cache)
            except Exception:
                continue
            if r and r.get("ok_trend") == 1:
                recs.append(r)
        if len(recs) >= 20:
            out[st] = recs
        if len(out) % 10 == 0:
            print(f"    [{tag}] 已完成 {len(out)}/{len(stations)} 站", flush=True)
    return out


def slopes(recs, m):
    ys, vs = [], []
    for r in recs:
        v = r.get(m)
        if v in (None, "", "None"):
            continue
        try:
            ys.append(int(r["year"])); vs.append(float(v))
        except Exception:
            pass
    if len(ys) < 15:
        return None, None
    b = theil_sen(ys, vs)
    return b, median(vs)


def main():
    print("=" * 96)
    print("R7-1 / M35  夜间定义敏感性: 太阳高度角 -0.833 度 vs 民用曙暮光 -6 度")
    print("=" * 96)
    sts = load_qualified(30)
    print(f"合格年 >= 30 的站: {len(sts)}")

    print("\n[1/2] 基准定义 -0.833 度 ...", flush=True)
    A = run_def(sts, -0.833, "A")
    print("\n[2/2] 民用曙暮光 -6 度 ...", flush=True)
    B = run_def(sts, -6.0, "B")

    common = sorted(set(A) & set(B))
    print(f"\n两定义均可用的站: {len(common)}")
    print(f"\n{'指标':<18}{'A中位斜率':>13}{'B中位斜率':>13}{'B/A':>9}"
          f"{'中位|B-A|/年':>14}{'斜率同号率':>12}{'A水平':>10}{'B水平':>10}")
    print("-" * 96)
    res = {}
    for m, lab in METRICS:
        sa, sb, dif, same, la, lb = [], [], [], 0, [], []
        for st in common:
            b1, lv1 = slopes(A[st], m)
            b2, lv2 = slopes(B[st], m)
            if b1 is None or b2 is None:
                continue
            sa.append(b1); sb.append(b2); dif.append(abs(b2 - b1))
            if (b1 > 0) == (b2 > 0):
                same += 1
            if lv1 is not None: la.append(lv1)
            if lv2 is not None: lb.append(lv2)
        if not sa:
            continue
        ma, mb = median(sa), median(sb)
        ratio = mb / ma if abs(ma) > 1e-12 else float("nan")
        res[m] = dict(a=ma, b=mb, ratio=ratio, dif=median(dif),
                      same=same / len(sa), la=median(la), lb=median(lb), n=len(sa))
        print(f"{lab:<18}{ma:>+13.5f}{mb:>+13.5f}{ratio:>9.2f}"
              f"{median(dif):>14.5f}{100*same/len(sa):>11.0f}%"
              f"{median(la):>10.3f}{median(lb):>10.3f}")
    print("-" * 96)
    print("  A = 太阳高度角 < -0.833 度（本研究）;  B = < -6 度（民用曙暮光）")
    print("  判据: 若 B/A 在 [0.7, 1.4] 且同号率 > 90%, 则结论对夜间定义稳健。")

    with open(os.path.join(HERE, "night_def_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items()}, f, ensure_ascii=False, indent=2)
    print("\n已保存 night_def_sensitivity.json")


if __name__ == "__main__":
    main()
