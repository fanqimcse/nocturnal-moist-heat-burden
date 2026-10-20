# -*- coding: utf-8 -*-
"""
step46: 欧洲扩张前沿的首次跨越 (First Crossing) —— 修正 step45 的饱和失效
step45 在全球台站网(176站, 偏湿热)上做首次跨越: 首年即79.5%饱和, 队列仅3站, 失败。
step46 转移到欧洲 225 个 NUTS2 区域 —— step39 已证其中 40% 暖季从未跨越 21.9C,
这才是真正的"未暴露 -> 暴露"扩张前沿。

数据: ext/health/wx_nightly_nuts2.csv (逐夜, nh219 = 夜间湿球>=21.9C 小时数)
人口: 由 ext/health/deaths_weekly_nuts3.csv 死亡数反推 (假设粗死亡率 10/千)
"""
import json, os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(BASE, "ext", "health")
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


P("欧洲扩张前沿的首次跨越 (First Crossing): 暴露边界扩张 vs 强度增长 (step46)")
P("修正 step45: 全球站网在 21.9C 下首年即饱和, 队列仅3站 -> 改用欧洲 225 NUTS2")
P("")

# ---------------- 载入 ----------------
w = pd.read_csv(os.path.join(EXT, "wx_nightly_nuts2.csv"), low_memory=False)
w["date"] = pd.to_datetime(w["date"])
w["year"] = w["date"].dt.year
w["month"] = w["date"].dt.month
P(f"[数据] 逐夜记录 {len(w)} 行 ; 区域 {w.nuts2.nunique()} ; 年 {w.year.min()}-{w.year.max()}")
P(f"       月份覆盖: {sorted(w.month.unique())}  -> {'暖季(5-9月)' if set(w.month.unique())<=set(range(5,10)) else '全年'}")
P(f"       nh219 总计 {int(w.nh219.sum())} 夜间小时")

# 人口代理
dth = pd.read_csv(os.path.join(EXT, "deaths_weekly_nuts3.csv"), low_memory=False)
dth["year"] = dth["week"].astype(str).str[:4].astype(int)
d3 = dth.groupby("nuts3")["deaths"].sum() / dth.groupby("nuts3")["year"].nunique()  # 年均死亡
CDR = 0.010
pop3 = (d3 / CDR).to_dict()
# NUTS3 -> NUTS2 (NUTS3 code 前4位 = NUTS2)
pop2 = {}
for k, v in pop3.items():
    pop2[str(k)[:4]] = pop2.get(str(k)[:4], 0.0) + v
P(f"[人口] 由死亡数反推(粗死亡率 {CDR*1000:.0f}/千): NUTS3 {len(pop3)} -> NUTS2 {len(pop2)}")
P(f"       覆盖人口 {sum(pop2.values())/1e6:.0f} 百万 (欧洲 NUTS2 口径, 一阶估计)")
P("")

# ---------------- 区域-年聚合 ----------------
g = w.groupby(["nuts2", "year"]).agg(
    nwh219=("nh219", "sum"), nwh230=("nh230", "sum"), nwh240=("nh240", "sum"),
    twmax=("tw_night_max", "max"), twmean=("tw_night_mean", "mean"),
    ndays=("date", "size"), nhrs=("n_hrs", "sum"),
).reset_index()

# 覆盖门槛: 暖季有效天数 >= 120 (5-9月约153天)
MIND = 120
g = g[g.ndays >= MIND].copy()
P(f"[聚合] 区域-年 {len(g)} ; 有效年门槛 暖季天数>={MIND} ; 区域 {g.nuts2.nunique()}")

THR = 21.9
g["cross"] = (g.nwh219 >= 1).astype(int)
g["pop"] = g["nuts2"].map(pop2).fillna(0.0)

# 基线湿热本底 tw0 (头3年夜间湿球均值)
y0 = g.year.min()
base = g[g.year <= y0 + 2].groupby("nuts2")["twmean"].mean()
g["tw0"] = g["nuts2"].map(base)
med0 = float(base.median())
P(f"[基线] 夜间湿球本底 tw0 中位 = {med0:.2f} C (用于南欧/北欧切分)")
P("")

# ---------------- [1] 暴露状态 ----------------
P("【1】欧洲暴露状态: 每年'暖季至少一次夜间湿球跨越 21.9 C'的区域比例与人口")
rows = []
for yr, gg in g.groupby("year"):
    rows.append(dict(year=int(yr), n=len(gg), exp=int(gg.cross.sum()),
                     frac=float(gg.cross.mean()),
                     exp_pop=float(gg.loc[gg.cross == 1, "pop"].sum()),
                     pop_tot=float(gg["pop"].sum()),
                     nwh_mean=float(gg.nwh219.mean())))
E = pd.DataFrame(rows).sort_values("year")
E["pop_frac"] = E.exp_pop / E.pop_tot.replace(0, np.nan)
E["exp_pop_M"] = E.exp_pop / 1e6


def ts(y, v):
    y = np.asarray(y, float); v = np.asarray(v, float)
    m = np.isfinite(v)
    if m.sum() < 6: return np.nan, np.nan
    b = sm.OLS(v[m], sm.add_constant(y[m])).fit()
    return float(b.params[1] * 10), float(b.pvalues[1])


b_frac, p_frac = ts(E.year, E.frac * 100)
b_pop, p_pop = ts(E.year, E.pop_frac * 100)
b_nwh, p_nwh = ts(E.year, E.nwh_mean)
P(f"    暴露区域比例: {E.frac.iloc[0]*100:.1f}% -> {E.frac.iloc[-1]*100:.1f}% ; "
  f"趋势 {b_frac:+.2f} pp/十年 (p={p_frac:.4f})")
P(f"    暴露人口: {E.exp_pop_M.iloc[0]:.0f} -> {E.exp_pop_M.iloc[-1]:.0f} 百万 ; "
  f"比例趋势 {b_pop:+.2f} pp/十年 (p={p_pop:.4f})")
P(f"    NWH 网络均值: {E.nwh_mean.iloc[0]:.1f} -> {E.nwh_mean.iloc[-1]:.1f} h/年 ; "
  f"趋势 {b_nwh:+.2f} h/十年 (p={p_nwh:.4f})")
P("")

# ---------------- [2] 首次跨越生存分析 ----------------
P("【2】首次跨越 (Time-to-First-Crossing) 生存分析 —— 欧洲")
P(f"    队列: 头3年({y0}-{y0+2})均未跨越 且 有效年>=18 的区域")


def cohort_hazard(g, thr_col="cross", minyr=18):
    ny = g.groupby("nuts2")["year"].size()
    g = g.copy()
    g["yrank"] = g.groupby("nuts2")["year"].rank(method="first").astype(int)
    h3 = g[g.yrank <= 3].groupby("nuts2")[thr_col].max()
    co = sorted(set(ny[ny >= minyr].index) & set(h3[h3 == 0].index))
    sv, ev = {}, {}
    for st in co:
        gg = g[g.nuts2 == st].sort_values("year")
        ever = 0
        for _, r in gg.iterrows():
            y = int(r.year)
            if ever: continue
            sv[y] = sv.get(y, 0) + 1
            if r[thr_col] == 1:
                ev[y] = ev.get(y, 0) + 1; ever = 1
    ys = sorted(sv)
    h = pd.DataFrame({"year": ys, "at_risk": [sv[y] for y in ys], "events": [ev.get(y, 0) for y in ys]})
    h = h[h.at_risk > 0].copy()
    return co, h


def cloglog_hr(h):
    if len(h) < 5 or h.events.sum() < 3:
        return None
    h = h.copy()
    h["dec"] = (h.year - h.year.min()) / 10.0
    h["nonev"] = h.at_risk - h.events
    fam = sm.families.Binomial(sm.families.links.CLogLog())
    m = sm.GLM(np.c_[h.events, h.nonev], sm.add_constant(h[["dec"]]), family=fam).fit()
    b, s = m.params[1], m.bse[1]
    return dict(HR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * s)), hi=float(np.exp(b + 1.96 * s)),
                z=float(b / s), events=int(h.events.sum()), cohort=int(h.at_risk.iloc[0]))


co, h = cohort_hazard(g)
P(f"    合格队列: {len(co)} 区域 (从未暴露起点出发)")
res_all = cloglog_hr(h)
if res_all:
    P(f"    首次跨越风险比 HR = {res_all['HR']:.3f} [95%CI {res_all['lo']:.3f}, {res_all['hi']:.3f}] "
      f"每十年 (z={res_all['z']:+.2f})")
    P(f"    队列 {res_all['cohort']} 区域, 观测到首次跨越事件 {res_all['events']} 次")
    P(f"    解读: 每过十年, 一个'尚未经历过致命夜间湿热'的欧洲区域, "
      f"首次跨越 21.9C 的风险 × {res_all['HR']:.3f} ({(res_all['HR']-1)*100:+.1f}%)")
mid = (h.year.min() + h.year.max()) / 2
h1 = h[h.year < mid]; h2 = h[h.year >= mid]
r1 = h1.events.sum() / max(1, h1.at_risk.sum()); r2 = h2.events.sum() / max(1, h2.at_risk.sum())
P(f"    经验对照: 前半期 {r1*100:.2f}%/区域年 -> 后半期 {r2*100:.2f}%/区域年 (比值 {r2/r1 if r1>0 else float('nan'):.2f}x)")
P("")

# ---------------- [3] 南北分层 ----------------
P("【3】分层的首次跨越风险 (南欧型 vs 北欧型)")
strat = {}
for lab, sel in [("南欧型 tw0>=%.2f" % med0, g[g.tw0 >= med0]),
                 ("北欧型 tw0< %.2f" % med0, g[g.tw0 < med0])]:
    c2, h2_ = cohort_hazard(sel)
    r = cloglog_hr(h2_)
    strat[lab] = (len(c2), r)
    if r:
        P(f"    {lab}: 队列 {len(c2)} 区域, 事件 {r['events']}, "
          f"HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] z={r['z']:+.2f}")
    else:
        P(f"    {lab}: 队列 {len(c2)} 区域, 事件不足")
P("")

# ---------------- [4] 强度 vs 范围 ----------------
P("【4】核心对比: 强度增长 vs 范围扩张 (相对速率)")
inten = g[g.cross == 1].groupby("year")["nwh219"].mean()   # 已暴露区域的小时数
inten_tw = g.groupby("year")["twmax"].mean()               # 全网络年内夜间湿球峰值


def rel(s, label, unit):
    s = s.dropna()
    if len(s) < 6:
        P(f"    {label}: 样本不足"); return np.nan, np.nan
    y = s.index.values.astype(float); v = s.values.astype(float)
    b = sm.OLS(v, sm.add_constant(y)).fit()
    sl = float(b.params[1] * 10); bs = float(np.mean(v[:5]))
    rel_ = sl / bs * 100 if bs != 0 else np.nan
    P(f"    {label}: 基线 {bs:.3f} -> {sl:+.4f} {unit}/十年 = 相对 {rel_:+.2f} %/十年 (p={b.pvalues[1]:.4f})")
    return sl, rel_


P("    --- 强度 ---")
s_i, r_i = rel(inten, "NWH219 (已暴露区域, h/年)", "h")
_, r_itw = rel(inten_tw, "夜间湿球年内峰值 (degC)", "degC")
P("    --- 范围 ---")
s_f, r_f = rel(E.set_index("year")["frac"] * 100, "暴露区域比例", "pp")
s_p, r_p = rel(E.set_index("year")["pop_frac"] * 100, "暴露人口比例", "pp")
if np.isfinite(r_i) and r_i != 0:
    P(f"    >>> 范围/强度 相对速率之比: 区域 {abs(r_f/r_i):.1f}x ; 人口 {abs(r_p/r_i):.1f}x")
P("")

# ---------------- [5] 新增首次暴露人口 ----------------
P("【5】每年新增'首次暴露'人口 (First Exposure, 百万人/年)")
popmap = pop2
newp = {}
for st in co:
    gg = g[g.nuts2 == st].sort_values("year")
    ever = 0
    for _, r in gg.iterrows():
        if ever: continue
        if r.cross == 1:
            newp[int(r.year)] = newp.get(int(r.year), 0.0) + popmap.get(st, 0.0)
            ever = 1
NP_ = pd.DataFrame({"year": sorted(newp), "pop": [newp[y] for y in sorted(newp)]})
NP_["M"] = NP_["pop"] / 1e6
if len(NP_):
    P(f"    年均新增首次暴露人口 {NP_.M.mean():.2f} 百万/年 (2000-2023 共 {NP_.M.sum():.1f} 百万)")
    if len(NP_) >= 6:
        y = NP_.year.values.astype(float); v = NP_.M.values
        b = sm.OLS(v, sm.add_constant(y)).fit()
        P(f"    趋势 {b.params[1]*10:+.2f} 百万/年 每十年 (p={b.pvalues[1]:.3f})")
    P(f"    逐年: " + ", ".join(f"{int(r.year)}:{r.M:.1f}" for r in NP_.itertuples()))
P("")

# ---------------- [6] 敏感性 ----------------
P("【6】敏感性检验")
P("    (a) 阈值 (21.9 / 23.0 / 24.0 / 25.8) —— 首次跨越 HR 每十年")
for thr, col in [(21.9, "nwh219"), (23.0, "nwh230"), (24.0, "nwh240"), (25.8, None)]:
    gg = g.copy()
    if col is None:
        if "nwh258" not in w.columns: continue
        gg["cx"] = (gg["nwh258"] if "nwh258" in gg else 0).astype(int) >= 1
    else:
        gg["cx"] = (gg[col] >= 1).astype(int)
    c2, h2_ = cohort_hazard(gg, "cx")
    r = cloglog_hr(h2_)
    if r:
        P(f"        thr={thr:4.1f}C 队列{len(c2):3d} 事件{r['events']:3d}  "
          f"HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] z={r['z']:+.2f}")
    else:
        P(f"        thr={thr:4.1f}C 队列{len(c2):3d} 事件不足")

P("    (b) 队列定义 (头2年 / 头3年 / 头5年 未跨越)")
for k in [2, 3, 5]:
    ny = g.groupby("nuts2")["year"].size()
    gg = g.copy()
    gg["yrank"] = gg.groupby("nuts2")["year"].rank(method="first").astype(int)
    hk = gg[gg.yrank <= k].groupby("nuts2")["cross"].max()
    co2 = sorted(set(ny[ny >= 18].index) & set(hk[hk == 0].index))
    sv, ev = {}, {}
    for st in co2:
        s2 = gg[gg.nuts2 == st].sort_values("year"); ever = 0
        for _, r in s2.iterrows():
            y = int(r.year)
            if ever: continue
            sv[y] = sv.get(y, 0) + 1
            if r.cross == 1: ev[y] = ev.get(y, 0) + 1; ever = 1
    ys = sorted(sv)
    hk2 = pd.DataFrame({"year": ys, "at_risk": [sv[y] for y in ys], "events": [ev.get(y, 0) for y in ys]})
    hk2 = hk2[hk2.at_risk > 0]
    r = cloglog_hr(hk2)
    if r:
        P(f"        头{k}年未跨越: 队列{len(co2):3d} 事件{r['events']:3d}  "
          f"HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] z={r['z']:+.2f}")
    else:
        P(f"        头{k}年未跨越: 队列{len(co2):3d} 事件不足")

P("    (c) 负对照 —— 首次'低于'21.9C (伪事件, 应无上升趋势)")
gg = g.copy()
gg["cx"] = (gg.nwh219 == 0).astype(int)  # 首次"无跨越"年
c2, h2_ = cohort_hazard(gg, "cx")
r = cloglog_hr(h2_)
P(f"        反向队列 {len(c2)} 区域" + (f" HR={r['HR']:.3f} z={r['z']:+.2f}" if r else " 事件不足"))
P("")

P("【7】结论")
if res_all:
    P(f"    欧洲'尚未暴露'区域首次跨越 21.9C 的风险 HR={res_all['HR']:.3f} 每十年")
    P(f"    暴露区域比例趋势 {b_frac:+.2f} pp/十年, NWH 强度趋势 {b_nwh:+.2f} h/十年")
    P(f"    -> 检验'扩张主导 vs 强度主导'")

res = dict(
    n_regions=int(g.nuts2.nunique()), years=[int(g.year.min()), int(g.year.max())],
    pop_total_M=float(sum(pop2.values()) / 1e6), tw0_median=med0,
    exposure=dict(frac_first=float(E.frac.iloc[0]), frac_last=float(E.frac.iloc[-1]),
                  trend_frac_pp_dec=float(b_frac), p_frac=float(p_frac),
                  exp_pop_M_first=float(E.exp_pop_M.iloc[0]), exp_pop_M_last=float(E.exp_pop_M.iloc[-1]),
                  trend_pop_pp_dec=float(b_pop), p_pop=float(p_pop),
                  nwh_first=float(E.nwh_mean.iloc[0]), nwh_last=float(E.nwh_mean.iloc[-1]),
                  trend_nwh_h_dec=float(b_nwh), p_nwh=float(p_nwh)),
    first_crossing=dict(cohort=len(co),
                        **{k: v for k, v in (res_all or {}).items() if k != "cohort"},
                        hazard_early=float(r1), hazard_late=float(r2)),
    strat={k: dict(cohort=n, **{kk: vv for kk, vv in (r or {}).items() if kk != "cohort"}) for k, (n, r) in strat.items()},
    amplitude_vs_extent=dict(nwh_slope=float(s_i) if np.isfinite(s_i) else None,
                             nwh_rel=float(r_i) if np.isfinite(r_i) else None,
                             twpeak_rel=float(r_itw) if np.isfinite(r_itw) else None,
                             frac_rel=float(r_f) if np.isfinite(r_f) else None,
                             pop_rel=float(r_p) if np.isfinite(r_p) else None),
    new_exposure=dict(mean_M_per_yr=(float(NP_.M.mean()) if len(NP_) else None),
                      total_M=(float(NP_.M.sum()) if len(NP_) else None)),
)
with open(os.path.join(BASE, "first_crossing_europe.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
E.to_csv(os.path.join(BASE, "first_crossing_europe_panel.csv"), index=False)
with open(os.path.join(BASE, "first_crossing_europe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[done]")
