#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GHCNh 高速并行下载器
====================

实测发现（2026-10-18）：
  - NCEI 单连接吞吐仅 ~6.9 KB/s，但这是**每连接**限速，不是总带宽
  - 并发  1 ->   6.9 KB/s
  - 并发 16 ->  40~51 KB/s
  - 并发 32 ->  99 KB/s
  - 并发 64 -> 174 KB/s   <- 最优
  - 并发 96 -> 100 KB/s   <- 服务端开始排队，反而下降

策略：单文件内切块并发 + 多文件并发，总连接数控制在 ~64。
实测单个 1.3 MB 站年文件约 8 秒，相比单连接 190 秒提速约 24 倍。

另一个必须处理的坑：NCEI 会**静默截断**（HTTP 200 但字节数不足），
因此每块下载后必须校验长度，最终文件必须与 Content-Length 完全一致，
并用 pyarrow 尝试打开以确认 Parquet 结构完整。
"""

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access"
DOC = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc"
DATA_DIR = "ghcn_data"

CHUNKS_PER_FILE = 8      # 单文件内切块数
FILE_WORKERS = 8         # 同时下载的文件数
# 总连接数 = 8 * 8 = 64


def _head(url, timeout=60):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        v = r.headers.get("Content-Length")
        return int(v) if v else None


def _get_range(url, start, end, timeout=180, retries=6):
    """下载一个字节区间，失败自动重试。"""
    headers = {"Range": f"bytes={start}-{end}"}
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(1.5 * (a + 1))
    return None


def download_file(url, path, chunks=CHUNKS_PER_FILE, verbose=True, verify_parquet=True):
    """
    并行分块下载单个文件。返回 (ok, bytes, seconds)
    已存在且大小正确的文件直接跳过。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t0 = time.time()
    remote = _head(url)
    if remote is None:
        return False, 0, 0.0
    if os.path.exists(path) and os.path.getsize(path) == remote:
        if not verify_parquet or path.endswith(".parquet") and _parquet_ok(path):
            return True, remote, 0.0

    size = remote
    step = max(size // chunks, 1)
    spans = []
    s = 0
    while s < size:
        e = min(s + step - 1, size - 1)
        spans.append((s, e))
        s = e + 1

    buf = {}

    def work(sid):
        a, b = spans[sid]
        data = _get_range(url, a, b)
        if data is None or len(data) != (b - a + 1):
            return sid, None
        return sid, (a, data)

    ok = True
    with ThreadPoolExecutor(max_workers=chunks) as ex:
        for sid, res in ex.map(work, range(len(spans))):
            if res is None:
                ok = False
            else:
                buf[sid] = res

    if not ok:
        return False, 0, time.time() - t0

    with open(path, "wb") as f:
        for sid in range(len(spans)):
            a, data = buf[sid]
            f.seek(a)
            f.write(data)

    got = os.path.getsize(path)
    if got != remote:
        return False, got, time.time() - t0
    if verify_parquet and path.endswith(".parquet") and not _parquet_ok(path):
        return False, got, time.time() - t0
    return True, got, time.time() - t0


def _parquet_ok(path):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pf.metadata.num_rows > 0
    except Exception:
        try:
            import duckdb
            n = duckdb.connect().execute(
                f"SELECT COUNT(*) FROM read_parquet(['{path}'])").fetchall()[0][0]
            return n > 0
        except Exception:
            return False


def station_year_url(station, year, fmt="parquet"):
    return f"{BASE}/by-year/{year}/{fmt}/GHCNh_{station}_{year}.{fmt}"


def download_many(jobs, data_dir=DATA_DIR, file_workers=FILE_WORKERS,
                  verbose=True):
    """
    jobs: [(station, year), ...]  ->  并行下载
    返回 {station_year: path or None}
    """
    os.makedirs(data_dir, exist_ok=True)

    def one(job):
        st, yr = job
        url = station_year_url(st, yr)
        path = os.path.join(data_dir, f"GHCNh_{st}_{yr}.parquet")
        try:
            ok, n, sec = download_file(url, path, verbose=False)
        except Exception as e:
            return (st, yr), None, str(e)[:60]
        return (st, yr), (path if ok else None), n

    results = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=file_workers) as ex:
        for (st, yr), path, n in ex.map(one, jobs):
            done += 1
            results[f"{st}_{yr}"] = path
            if verbose:
                if path:
                    print(f"  [{done}/{len(jobs)}] {st} {yr} OK  {n:,} B", flush=True)
                else:
                    print(f"  [{done}/{len(jobs)}] {st} {yr} 失败/无数据", flush=True)
    if verbose:
        dt = time.time() - t0
        okc = sum(1 for v in results.values() if v)
        print(f"完成 {okc}/{len(jobs)}，用时 {dt:.0f} 秒")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("用法: python ghcnh_fast.py STATION YEAR [STATION YEAR ...]")
        print("     python ghcnh_fast.py -f jobs.txt   (每行: STATION YEAR)")
        sys.exit(0)
    if sys.argv[1] == "-f":
        jobs = []
        for line in open(sys.argv[2], encoding="utf-8"):
            p = line.split()
            if len(p) >= 2:
                jobs.append((p[0], int(p[1])))
    else:
        a = sys.argv[1:]
        jobs = [(a[i], int(a[i + 1])) for i in range(0, len(a) - 1, 2)]
    print(f"下载 {len(jobs)} 个站年文件")
    download_many(jobs)
