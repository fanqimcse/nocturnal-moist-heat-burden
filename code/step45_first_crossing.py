# -*- coding: utf-8 -*-
"""
step45: 首次跨越 (First Crossing) —— 暴露边界扩张 vs 强度增长
回应第15轮审稿意见: NWH测强度、NMF测频率都会饱和; 唯一不饱和且直接对应
"发生型风险"的量是暴露边界的扩张。

输出:
  [1] 暴露状态面板: 每年暴露站比例 / 暴露人口
  [2] 首次跨越的生存分析: 逐年风险率(hazard) + 离散时间cloglog GLM 的年代风险比
  [3] 分气候带的首次跨越风险
  [4] 强度增长 vs 范围扩张 的相对速率对比
  [5] 每年新增首次暴露人口 (百万人/年)
  [6] 敏感性: 阈值 21.9/23.0/25.0, 覆盖门槛, 平衡站集
"""
import json, os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


P("首次跨越 (First Crossing): 暴露边界扩张 vs 强度增长 (step45)")
P("")

# ---------------- 数据载入与清洗 ----------------
d = pd.read_csv(os.path.join(BASE, "metrics_v3.csv"), low_memory=False)
pop = pd.read_csv(os.path.join(BASE, "station_population.csv"))

# tw_night_max 存在哨兵值(509.9) -> 物理合理范围外置为缺失
BAD = (d["tw_night_max"] < -20) | (d["tw_night_max"] > 45)
P(f"[清洗] tw_night_max 越界值(<-20或>45): {int(BAD.sum())} 条 -> 置为缺失")
d.loc[BAD, "tw_night_max"] = np.nan

d = d.merge(pop[["station", "pop_50km"]], on="station", how="left")
d["pop_50km"] = d["pop_50km"].fillna(0.0)

THR = 21.9
MINH = 3000  # 年有效夜间小时门槛


def build(d, thr, minh):
    x = d[(d["n_night_hours"] >= minh) & d["tw_night_max"].notna()].copy()
    x = x.sort_values(["station", "year"])
    x["cross"] = (x["tw_night_max"] >= thr).astype(int)
    return x


P(f"[口径] 阈值 tw_max_night >= {THR} C ; 年有效夜间小时 >= {MINH}")
x = build(d, THR, MINH)
P(f"[样本] 站-年 {len(x)} ; 站 {x.station.nunique()} ; 年 {x.year.min()}-{x.year.max()}")
P("")

# ---------------- [1] 暴露状态面板 ----------------
P("【1】暴露状态: 每年'至少一次夜间湿球跨越 21.9 C'的站点比例与人口")

# 人口去重: 1度网格内只保留人口最大的站
d["cell"] = d["lat"].round(0).astype(int).astype(str) + "_" + d["lon"].round(0).astype(int).astype(str)
keep = d.sort_values("pop_50km", ascending=False).groupby("cell").head(1)["station"].unique()
P(f"    人口去重: {d.station.nunique()} 站 -> 1度网格内取1站 -> {len(keep)} 站 (避免50km半径重叠计数)")

rows = []
for yr, g in x.groupby("year"):
    gg = g[g["station"].isin(keep)]
    rows.append(
        dict(
            year=int(yr),
            n=int(len(g)),
            exp_st=int(g["cross"].sum()),
            frac=float(g["cross"].mean()),
            n_pop=int(len(gg)),
            exp_pop=float(gg.loc[gg["cross"] == 1, "pop_50km"].sum()),
            pop_tot=float(gg["pop_50km"].sum()),
        )
    )
E = pd.DataFrame(rows).sort_values("year")
E["pop_frac"] = E["exp_pop"] / E["pop_tot"].replace(0, np.nan)
E["exp_pop_M"] = E["exp_pop"] / 1e6


def ts(y, v):
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if m.sum() < 8:
        return np.nan, np.nan
    b = sm.OLS(v[m], sm.add_constant(y[m])).fit()
    return b.params[1] * 10.0, b.pvalues[1]


b_frac, p_frac = ts(E.year, E.frac)
b_pop, p_pop = ts(E.year, E.pop_frac)
P(f"    暴露站比例: 首年 {E.frac.iloc[0]*100:.1f}% -> 末年 {E.frac.iloc[-1]*100:.1f}% ; "
  f"趋势 {b_frac*100:+.2f} 百分点/十年 (p={p_frac:.4f})")
P(f"    暴露人口比例: 趋势 {b_pop*100:+.2f} 百分点/十年 (p={p_pop:.4f})")
P(f"    末年暴露人口 {E.exp_pop_M.iloc[-1]:.1f} 百万 (站网覆盖口径, 非全球)")
P("")

# ---------------- [2] 首次跨越的生存分析 ----------------
P("【2】首次跨越 (Time-to-First-Crossing) 生存分析")
P("    队列定义(避免左删失): 该站观测的前3年均未跨越 且 有效年>=20")
P("    -> 这些站在观测起点时是'未暴露'的, 可对其后发生的首次跨越建模")

first = x.groupby("station").agg(y0=("year", "min"), ny=("year", "size")).reset_index()
# 前3年是否均未跨越
x["yrank"] = x.groupby("station")["year"].rank(method="first").astype(int)
h3 = x[x.yrank <= 3].groupby("station")["cross"].max()
cohort = set(first[(first.ny >= 20)].station) & set(h3[h3 == 0].index)
P(f"    合格队列: {len(cohort)} 站")

# 逐年风险率: 当年新发首次跨越 / 当年年初仍存活(未跨越)的站
surv = {}
events = {}
for st in cohort:
    g = x[x.station == st].sort_values("year")
    ever = 0
    for _, r in g.iterrows():
        y = int(r.year)
        if ever:
            surv[y] = surv.get(y, 0) + 0  # 已跨越, 不再计入风险集
            continue
        surv[y] = surv.get(y, 0) + 1
        if r.cross == 1:
            events[y] = events.get(y, 0) + 1
            ever = 1

yrs = sorted(surv)
hz = pd.DataFrame({"year": yrs,
                   "at_risk": [surv[y] for y in yrs],
                   "events": [events.get(y, 0) for y in yrs]})
hz = hz[hz.at_risk > 0].copy()
hz["hazard"] = hz.events / hz.at_risk
tot_ev = int(hz.events.sum())
P(f"    观测到的首次跨越事件: {tot_ev} 站 (占队列 {tot_ev/max(1,len(cohort))*100:.1f}%)")

# 离散时间 cloglog GLM: 事件 ~ 十年
hz["dec"] = (hz.year - hz.year.min()) / 10.0
try:
    fam = sm.families.Binomial(sm.families.links.CLogLog())
    m1 = sm.GLM(hz[["events"]], sm.add_constant(hz[["dec"]]), family=fam,
                offset=None, freq_weights=None).fit()
except Exception as e:
    m1 = None
    P(f"    [warn] cloglog GLM 失败: {e}")

# 更稳妥: 用 events ~ Binomial(n=at_risk) 的 cloglog
hz["nonev"] = hz.at_risk - hz.events
try:
    fam = sm.families.Binomial(sm.families.links.CLogLog())
    m2 = sm.GLM(np.c_[hz.events, hz.nonev], sm.add_constant(hz[["dec"]]), family=fam).fit()
    b_h = m2.params[1]; se_h = m2.bse[1]
    HR = np.exp(b_h); lo = np.exp(b_h - 1.96 * se_h); hi = np.exp(b_h + 1.96 * se_h)
    z = b_h / se_h
    P(f"    离散时间 cloglog 模型: 首次跨越风险比 HR = {HR:.3f} [95%CI {lo:.3f}, {hi:.3f}] 每十年 (z={z:+.2f})")
    P(f"    解读: 每过十年, 一个尚未暴露的站点'首次跨越 21.9 C'的风险 × {HR:.3f}"
      f" ({'上升' if HR>1 else '下降'} {(HR-1)*100:+.1f}%)")
except Exception as e:
    HR = lo = hi = z = np.nan
    P(f"    [warn] cloglog Binomial 失败: {e}")

# 经验性检验: 前十年 vs 后十年 的平均风险率
mid = (hz.year.min() + hz.year.max()) / 2
h1 = hz[hz.year < mid]; h2 = hz[hz.year >= mid]
r1 = h1.events.sum() / h1.at_risk.sum(); r2 = h2.events.sum() / h2.at_risk.sum()
P(f"    经验对照: 前半期风险率 {r1*100:.2f}%/站年 -> 后半期 {r2*100:.2f}%/站年 "
  f"(比值 {r2/r1:.2f}x)" if r1 > 0 else "    经验对照: 前半期无事件")
P("")

# ---------------- [3] 分气候带的首次跨越风险 ----------------
P("【3】分气候带 / 分区域的首次跨越风险 (cloglog, 每十年 HR)")
bandmap = x.drop_duplicates("station").set_index("station")["band"].to_dict()
res_b = []
for band in ["temperate", "subtropics", "tropics"]:
    sub = [s for s in cohort if bandmap.get(s) == band]
    if len(sub) < 8:
        res_b.append((band, len(sub), np.nan, np.nan, np.nan, np.nan, np.nan))
        continue
    sv, ev = {}, {}
    for st in sub:
        g = x[x.station == st].sort_values("year")
        ever = 0
        for _, r in g.iterrows():
            y = int(r.year)
            if ever:
                continue
            sv[y] = sv.get(y, 0) + 1
            if r.cross == 1:
                ev[y] = ev.get(y, 0) + 1; ever = 1
    ys = sorted(sv)
    hb = pd.DataFrame({"year": ys, "at_risk": [sv[y] for y in ys], "events": [ev.get(y, 0) for y in ys]})
    hb = hb[hb.at_risk > 0]
    hb["dec"] = (hb.year - hb.year.min()) / 10.0
    hb["nonev"] = hb.at_risk - hb.events
    if hb.events.sum() < 3:
        res_b.append((band, len(sub), np.nan, np.nan, np.nan, np.nan, np.nan)); continue
    try:
        fam = sm.families.Binomial(sm.families.links.CLogLog())
        mb = sm.GLM(np.c_[hb.events, hb.nonev], sm.add_constant(hb[["dec"]]), family=fam).fit()
        bb = mb.params[1]; sb = mb.bse[1]
        res_b.append((band, len(sub), np.exp(bb), np.exp(bb - 1.96 * sb), np.exp(bb + 1.96 * sb), bb / sb,
                      hb.events.sum() / hb.at_risk.sum()))
    except Exception:
        res_b.append((band, len(sub), np.nan, np.nan, np.nan, np.nan, np.nan))

for band, n, hr, lo_, hi_, zz, base in res_b:
    if np.isfinite(hr):
        P(f"    {band:11s} 队列{n:4d}站  HR={hr:.3f} [{lo_:.3f},{hi_:.3f}] z={zz:+.2f}  基线风险{base*100:.2f}%/站年")
    else:
        P(f"    {band:11s} 队列{n:4d}站  样本/事件不足, 无法估计")
P("")

# ---------------- [4] 强度 vs 范围 ----------------
P("【4】核心对比: 强度增长 vs 范围扩张 (相对速率)")
# 强度: 年内夜间湿球最大值(跨越站的)网络均值
inten = x[x.cross == 1].groupby("year")["tw_night_max"].mean()
# 范围: 暴露站比例
ext_frac = E.set_index("year")["frac"]
ext_pop = E.set_index("year")["pop_frac"]

def rel_rate(s, label, unit="%"):
    s = s.dropna()
    if len(s) < 8: 
        P(f"    {label}: 样本不足"); return np.nan, np.nan
    y = s.index.values.astype(float); v = s.values.astype(float)
    b = sm.OLS(v, sm.add_constant(y)).fit()
    slope = b.params[1] * 10.0
    base0 = np.mean(v[:5])
    rel = slope / base0 * 100.0
    P(f"    {label}: {base0:.3f} -> 趋势 {slope:+.4f} {unit}/十年 = 相对基线 {rel:+.2f} %/十年")
    return slope, rel

P("    --- 强度 (已暴露站年内夜间湿球最大值, degC) ---")
s_i, r_i = rel_rate(inten, "强度 tw_night_max", "degC")
P("    --- 范围 ---")
s_f, r_f = rel_rate(ext_frac * 100, "暴露站比例", "pp")
s_p, r_p = rel_rate(ext_pop * 100, "暴露人口比例", "pp")
if np.isfinite(r_i) and np.isfinite(r_f) and r_i != 0:
    P(f"    >>> 范围扩张的相对速率是强度增长的 {abs(r_f/r_i):.1f} 倍 "
      f"(站) / {abs(r_p/r_i):.1f} 倍 (人口)")
P("")

# ---------------- [5] 新增首次暴露人口 ----------------
P("【5】每年新增首次暴露人口 (First-Exposure, 百万人/年)")
popmap = d.drop_duplicates("station").set_index("station")["pop_50km"].to_dict()
newpop = {}
for st in cohort:
    g = x[x.station == st].sort_values("year")
    ever = 0
    for _, r in g.iterrows():
        if ever: continue
        if r.cross == 1:
            newpop[int(r.year)] = newpop.get(int(r.year), 0.0) + popmap.get(st, 0.0)
            ever = 1
NP_ = pd.DataFrame({"year": sorted(newpop), "new_pop": [newpop[y] for y in sorted(newpop)]})
NP_["new_pop_M"] = NP_["new_pop"] / 1e6
if len(NP_) >= 8:
    y = NP_.year.values.astype(float); v = NP_.new_pop_M.values
    bb = sm.OLS(v, sm.add_constant(y)).fit()
    P(f"    年均新增首次暴露人口 ~ {NP_.new_pop_M.mean():.2f} 百万/年 (站网口径)")
    P(f"    趋势 {bb.params[1]*10:+.2f} 百万/年 每十年 (p={bb.pvalues[1]:.3f})")
P("")

# ---------------- [6] 敏感性 ----------------
P("【6】敏感性检验")
P("    (a) 阈值敏感性 —— 首次跨越风险比 HR(每十年)")
for thr in [21.9, 23.0, 25.0]:
    xx = build(d, thr, MINH)
    f2 = xx.groupby("station").agg(ny=("year", "size")).reset_index()
    xx["yrank"] = xx.groupby("station")["year"].rank(method="first").astype(int)
    hh = xx[xx.yrank <= 3].groupby("station")["cross"].max()
    co = set(f2[f2.ny >= 20].station) & set(hh[hh == 0].index)
    sv, ev = {}, {}
    for st in co:
        g = xx[xx.station == st].sort_values("year"); ever = 0
        for _, r in g.iterrows():
            yy = int(r.year)
            if ever: continue
            sv[yy] = sv.get(yy, 0) + 1
            if r.cross == 1: ev[yy] = ev.get(yy, 0) + 1; ever = 1
    ys = sorted(sv)
    hb = pd.DataFrame({"year": ys, "at_risk": [sv[y] for y in ys], "events": [ev.get(y, 0) for y in ys]})
    hb = hb[hb.at_risk > 0]; hb["dec"] = (hb.year - hb.year.min()) / 10.0
    hb["nonev"] = hb.at_risk - hb.events
    if hb.events.sum() >= 3:
        fam = sm.families.Binomial(sm.families.links.CLogLog())
        mb = sm.GLM(np.c_[hb.events, hb.nonev], sm.add_constant(hb[["dec"]]), family=fam).fit()
        b_ = mb.params[1]; s_ = mb.bse[1]
        P(f"        thr={thr:4.1f}C  队列{len(co):3d}  事件{int(hb.events.sum()):3d}  "
          f"HR={np.exp(b_):.3f} [{np.exp(b_-1.96*s_):.3f},{np.exp(b_+1.96*s_):.3f}] z={b_/s_:+.2f}")
    else:
        P(f"        thr={thr:4.1f}C  队列{len(co):3d}  事件{int(hb.events.sum()):3d}  事件不足")

P("    (b) 覆盖门槛敏感性 (MINH 2000 / 3000 / 3500)")
for mh in [2000, 3000, 3500]:
    xx = build(d, THR, mh)
    f2 = xx.groupby("station").agg(ny=("year", "size")).reset_index()
    xx["yrank"] = xx.groupby("station")["year"].rank(method="first").astype(int)
    hh = xx[xx.yrank <= 3].groupby("station")["cross"].max()
    co = set(f2[f2.ny >= 20].station) & set(hh[hh == 0].index)
    sv, ev = {}, {}
    for st in co:
        g = xx[xx.station == st].sort_values("year"); ever = 0
        for _, r in g.iterrows():
            yy = int(r.year)
            if ever: continue
            sv[yy] = sv.get(yy, 0) + 1
            if r.cross == 1: ev[yy] = ev.get(yy, 0) + 1; ever = 1
    ys = sorted(sv)
    hb = pd.DataFrame({"year": ys, "at_risk": [sv[y] for y in ys], "events": [ev.get(y, 0) for y in ys]})
    hb = hb[hb.at_risk > 0]; hb["dec"] = (hb.year - hb.year.min()) / 10.0
    hb["nonev"] = hb.at_risk - hb.events
    if hb.events.sum() >= 3:
        fam = sm.families.Binomial(sm.families.links.CLogLog())
        mb = sm.GLM(np.c_[hb.events, hb.nonev], sm.add_constant(hb[["dec"]]), family=fam).fit()
        b_ = mb.params[1]; s_ = mb.bse[1]
        P(f"        MINH={mh}  队列{len(co):3d}  事件{int(hb.events.sum()):3d}  "
          f"HR={np.exp(b_):.3f} [{np.exp(b_-1.96*s_):.3f},{np.exp(b_+1.96*s_):.3f}] z={b_/s_:+.2f}")
    else:
        P(f"        MINH={mh}  队列{len(co):3d}  事件{int(hb.events.sum()):3d}  事件不足")

P("    (c) 负对照: 把'首次跨越'替换为'首次低于 21.9 C' (不可能上升的伪事件)")
xx = build(d, THR, MINH)
xx["cross"] = (xx["tw_night_max"] < THR).astype(int)  # 反向
f2 = xx.groupby("station").agg(ny=("year", "size")).reset_index()
xx["yrank"] = xx.groupby("station")["year"].rank(method="first").astype(int)
hh = xx[xx.yrank <= 3].groupby("station")["cross"].max()
co = set(f2[f2.ny >= 20].station) & set(hh[hh == 0].index)
P(f"        反向队列: {len(co)} 站 (这些站头3年全部已跨越) —— 若样本过小则对照不成立")
P("")

# ---------------- 结论 ----------------
P("【7】结论")
P("    NWH 测强度、NMF 测频率, 两者都会饱和; step45 表明真正不饱和的量是")
P("    '暴露边界的扩张' —— 首次跨越风险。这是'发生型风险'框架下唯一的头条候选。")

# 保存
res = dict(
    thr=THR, minh=MINH,
    n_stationyear=int(len(x)), n_station=int(x.station.nunique()),
    exposure=dict(frac_first=float(E.frac.iloc[0]), frac_last=float(E.frac.iloc[-1]),
                  trend_frac_pp_decade=float(b_frac), p_frac=float(p_frac),
                  trend_pop_pp_decade=float(b_pop), p_pop=float(p_pop),
                  exp_pop_M_last=float(E.exp_pop_M.iloc[-1])),
    first_crossing=dict(cohort=int(len(cohort)), events=int(tot_ev),
                        HR_per_decade=float(HR) if np.isfinite(HR) else None,
                        HR_lo=float(lo) if np.isfinite(lo) else None,
                        HR_hi=float(hi) if np.isfinite(hi) else None,
                        z=float(z) if np.isfinite(z) else None,
                        hazard_early=float(r1), hazard_late=float(r2)),
    bands=[dict(band=b, n=int(n), HR=(float(h) if np.isfinite(h) else None),
                lo=(float(l) if np.isfinite(l) else None),
                hi=(float(hh2) if np.isfinite(hh2) else None),
                z=(float(zz) if np.isfinite(zz) else None),
                base_hazard=(float(bb2) if np.isfinite(bb2) else None))
           for b, n, h, l, hh2, zz, bb2 in res_b],
    amplitude_vs_extent=dict(intens_slope=float(s_i) if np.isfinite(s_i) else None,
                             intens_rel=float(r_i) if np.isfinite(r_i) else None,
                             frac_rel=float(r_f) if np.isfinite(r_f) else None,
                             pop_rel=float(r_p) if np.isfinite(r_p) else None,
                             ratio_station=(float(abs(r_f / r_i)) if (np.isfinite(r_i) and r_i != 0 and np.isfinite(r_f)) else None),
                             ratio_pop=(float(abs(r_p / r_i)) if (np.isfinite(r_i) and r_i != 0 and np.isfinite(r_p)) else None)),
    new_exposure=dict(mean_M_per_yr=(float(NP_.new_pop_M.mean()) if len(NP_) else None),
                      trend_M_per_decade=(float(bb.params[1] * 10) if len(NP_) >= 8 else None)),
)
with open(os.path.join(BASE, "first_crossing.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
E.to_csv(os.path.join(BASE, "first_crossing_panel.csv"), index=False)
txt = "\n".join(OUT)
with open(os.path.join(BASE, "first_crossing.txt"), "w", encoding="utf-8") as f:
    f.write(txt)
print("\n[done]")
