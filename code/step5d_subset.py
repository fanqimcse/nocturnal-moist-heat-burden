#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对指定站点子集重算指标（用于分批验证新下载区域的数据可用性）。"""
import sys, os, json, csv, glob
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step5c_metrics_v3 as S

def run_subset(stations, out, pattern_dir="ghcn_data"):
    meta = {m["station"]: m for m in
            json.load(open(os.path.join(HERE, "panel_all.json"), encoding="utf-8"))}
    rows = []; cache = {}
    for st in stations:
        m = meta.get(st, {})
        lat = m.get("lat"); lon = m.get("lon")
        for p in sorted(glob.glob(os.path.join(HERE, pattern_dir, f"GHCNh_{st}_*.parquet"))):
            yr = int(os.path.basename(p).replace(".parquet", "").split("_")[2])
            if lat is None:
                import duckdb
                c = duckdb.connect()
                c.execute("CREATE VIEW z AS SELECT * FROM read_parquet(['%s'])" % p)
                a = c.execute("SELECT MAX(LATITUDE), MAX(LONGITUDE) FROM z").fetchall()[0]
                lat, lon = float(a[0]), float(a[1]); c.close()
            try:
                r = S.analyze(p, lat, lon, st, yr, cache)
            except Exception as e:
                continue
            if r is None: continue
            if m:
                r["name"] = m.get("name", ""); r["country"] = m.get("country", "")
                r["band"] = m.get("band", ""); r["region"] = m.get("region", "")
            rows.append(r)
    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(HERE, out), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        nok = sum(r["ok_trend"] for r in rows)
        print(f"{out}: {len(rows)} 条, 合格 {nok} ({100*nok/len(rows):.0f}%), 站 {len(set(r['station'] for r in rows))}")
    else:
        print("无结果")
    return rows

if __name__ == "__main__":
    stfile = sys.argv[1]; out = sys.argv[2]
    src = json.load(open(stfile, encoding="utf-8"))
    sts = [m["station"] if isinstance(m, dict) else m for m in src]
    run_subset(sts, out)
