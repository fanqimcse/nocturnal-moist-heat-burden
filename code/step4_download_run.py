#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量下载运行器：读取 jobs 文件，用 ghcnh_fast 并行下载，断点续传。
重复运行会跳过已完整下载的文件，因此可以安全地多次启动直到全部完成。
"""
import io, sys, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghcnh_fast import download_many

def load_jobs(path):
    jobs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 2:
            jobs.append((p[0], int(p[1])))
    return jobs

if __name__ == "__main__":
    jf = sys.argv[1] if len(sys.argv) > 1 else "jobs_core.txt"
    fw = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    jobs = load_jobs(jf)
    print(f"任务 {len(jobs)} 个站年, 文件并发 {fw}", flush=True)
    t0 = time.time()
    res = download_many(jobs, file_workers=fw, verbose=True)
    ok = sum(1 for v in res.values() if v)
    dt = time.time() - t0
    print(f"\n===== 完成 {ok}/{len(jobs)}  用时 {dt/60:.1f} 分钟 =====", flush=True)
    missing = [k for k, v in res.items() if not v]
    if missing:
        print(f"未完成 {len(missing)} 个: {missing[:20]}", flush=True)
        with open(jf.replace(".txt", "_retry.txt"), "w", encoding="utf-8") as f:
            for k in missing:
                s, y = k.rsplit("_", 1)
                f.write(f"{s} {y}\n")
        print("已写出重试清单 " + jf.replace(".txt", "_retry.txt"), flush=True)
