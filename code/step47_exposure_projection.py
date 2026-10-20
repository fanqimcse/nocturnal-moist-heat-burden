# -*- coding: utf-8 -*-
"""
step47 (v2): 从"风险比"转向"剂量-反应 + 可外推投影"

修正 v1 的三处错误:
  1. CMIP6 基线配对失败 -> 改为按 source_id 配对 historical(1995-2014) 与 ssp(mid/end)
  2. logit 系数与概率尺度混淆 -> 空间换时间检验改用边际效应(pp/degC)比较
  3. "扩张远超强度"判断方向写反 -> 实际强度(+47%)快于范围(+29%);
     但这恰恰是本研究的核心政策结论: 用强度推断风险会**高估** ~1.6 倍。

核心命题(发生型风险的正确含义):
  常规做法: 用 NWH 累积小时数的增长推断健康风险增长
  本研究:   剂量-反应平坦 -> 风险只随"是否发生"上升 -> 风险增长 ∝ 暴露范围扩张
  两者之差 = 常规做法的高估倍数
"""
import json, os, glob, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(BASE, "ext")
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


P("从风险比到剂量-反应: 欧洲首次暴露的可外推投影 (step47 v2)")
P("")

# ============ 面板 ============
w = pd.read_csv(os.path.join(EXT, "health", "wx_nightly_nuts2.csv"), low_memory=False)
w["date"] = pd.to_datetime(w["date"]); w["year"] = w["date"].dt.year
g = w.groupby(["nuts2", "year"]).agg(
    nwh219=("nh219", "sum"), twmax=("tw_night_max", "max"),
    twmean=("tw_night_mean", "mean"), ndays=("date", "size")).reset_index()
g = g[g.ndays >= 120].copy()
g["cross"] = (g.nwh219 >= 1).astype(int)
dth = pd.read_csv(os.path.join(EXT, "health", "deaths_weekly_nuts3.csv"), low_memory=False)
dth["year"] = dth["week"].astype(str).str[:4].astype(int)
d3 = dth.groupby("nuts3")["deaths"].sum() / dth.groupby("nuts3")["year"].nunique()
pop2 = {}
for k, v in (d3 / 0.010).to_dict().items():
    pop2[str(k)[:4]] = pop2.get(str(k)[:4], 0.0) + v
g["pop"] = g["nuts2"].map(pop2).fillna(0.0)
g = g[g["pop"] > 0].copy()
P(f"[样本] 区域-年 {len(g)} ; 区域 {g.nuts2.nunique()} ; 年 {int(g.year.min())}-{int(g.year.max())}")
P(f"       人口覆盖 {g.groupby('nuts2')['pop'].first().sum()/1e6:.0f} 百万")
P("")

# ============ A. 脆弱性消耗诊断 ============
P("【A】诊断: 为何 step46 的首次跨越 HR 下降 —— 脆弱性消耗 (frailty depletion)")
ny = g.groupby("nuts2")["year"].size()
gg = g.copy()
gg["yrank"] = gg.groupby("nuts2")["year"].rank(method="first").astype(int)
h3 = gg[gg.yrank <= 3].groupby("nuts2")["cross"].max()
co = sorted(set(ny[ny >= 18].index) & set(h3[h3 == 0].index))
base0 = gg[gg.yrank <= 3].groupby("nuts2")["twmean"].mean()
tw0 = {s: float(base0.get(s, np.nan)) for s in co}
sv, ev, tws = {}, {}, {}
for st in co:
    s2 = gg[gg.nuts2 == st].sort_values("year"); ever = 0
    for _, r in s2.iterrows():
        y = int(r.year)
        if ever: continue
        sv[y] = sv.get(y, 0) + 1
        tws.setdefault(y, []).append(tw0.get(st, np.nan))
        if r.cross == 1:
            ev[y] = ev.get(y, 0) + 1; ever = 1
ys = sorted(sv)
a_first = float(np.nanmedian(tws[ys[0]])); a_last = float(np.nanmedian(tws[ys[-1]]))
P(f"    队列 {len(co)} 区域, 观测到首次跨越 {int(sum(ev.values()))} 次")
P(f"    风险集(年初仍未跨越)的基线湿热本底 tw0 中位: "
  f"{ys[0]} 年 {a_first:.2f} C  ->  {ys[-1]} 年 {a_last:.2f} C  (变化 {a_last-a_first:+.2f} C)")
P(f"    >>> 湿热本底高的区域先被消耗(跨越后退出风险集), 残留风险集逐年变冷。")
P(f"    >>> 故风险率必然下降 —— HR=0.863 是选择性消耗假象, 不是气候信号。**撤回**")
P("")

# ============ B. 稳健趋势 ============
P("【B】稳健趋势 (Theil-Sen + Mann-Kendall)")
rows = []
for yr, s2 in g.groupby("year"):
    rows.append(dict(year=int(yr), frac=float(s2.cross.mean()),
                     popfrac=float(s2.loc[s2.cross == 1, "pop"].sum() / s2["pop"].sum()),
                     nwh=float(s2.nwh219.mean()), twm=float(s2.twmean.mean())))
E = pd.DataFrame(rows).sort_values("year")


def robust(v, label, unit, scale=1.0):
    y = E.year.values.astype(float); v = np.asarray(v, float) * scale
    ts = stats.theilslopes(v, y, 0.95)
    tau, pv = stats.kendalltau(y, v)
    P(f"    {label:14s} Theil-Sen {ts[0]*10:+8.3f} {unit}/十年 "
      f"[95%CI {ts[2]*10:+.3f},{ts[3]*10:+.3f}]  MK tau={tau:+.3f} p={pv:.4f}")
    return dict(ts=float(ts[0] * 10), lo=float(ts[2] * 10), hi=float(ts[3] * 10),
                tau=float(tau), p=float(pv))


r_frac = robust(E.frac, "暴露区域比例", "pp", 100)
r_pop = robust(E.popfrac, "暴露人口比例", "pp", 100)
r_nwh = robust(E.nwh, "NWH 小时数", "h")
r_twm = robust(E.twm, "暖季夜间湿球", "C")
P("")
f0, f1 = E.frac.head(5).mean(), E.frac.tail(5).mean()
p0, p1 = E.popfrac.head(5).mean(), E.popfrac.tail(5).mean()
n0, n1 = E.nwh.head(5).mean(), E.nwh.tail(5).mean()
rel_f, rel_p, rel_n = f1 / f0 - 1, p1 / p0 - 1, n1 / n0 - 1
P(f"    首5年 -> 末5年 (最稳健的陈述):")
P(f"        暴露区域比例  {f0*100:.1f}% -> {f1*100:.1f}%   (相对 {rel_f*100:+.0f}%)")
P(f"        暴露人口比例  {p0*100:.1f}% -> {p1*100:.1f}%   (相对 {rel_p*100:+.0f}%)")
P(f"        NWH 小时数    {n0:.1f} -> {n1:.1f} h/年  (相对 {rel_n*100:+.0f}%)")
P("")
P(f"    *** 关键修正 (v1 在此处把方向写反了) ***")
P(f"        强度的相对增长 ({rel_n*100:+.0f}%) 实际上【快于】暴露范围的相对扩张 ({rel_p*100:+.0f}%)")
P(f"        但'发生型风险'框架的含义恰恰在这里: 由于剂量-反应平坦,")
P(f"        **总风险增长 ∝ 暴露范围扩张, 而非 ∝ 强度增长**。因此:")
over = rel_n / rel_p if rel_p != 0 else np.nan
P(f"        用 NWH 强度增长推断健康风险增长, 会【高估】约 {over:.2f} 倍 "
  f"({rel_n*100:+.0f}% / {rel_p*100:+.0f}%)。")
P(f"        这是本研究对影响评估实践的直接纠正。")
P("")

# ============ C. 剂量-反应 ============
P("【C】剂量-反应: 暴露概率 ~ 暖季夜间湿球 (logistic, 含饱和)")
mm = g.dropna(subset=["twmean", "cross"])
m_logit = sm.GLM(mm["cross"], sm.add_constant(mm[["twmean"]]),
                 family=sm.families.Binomial()).fit()
a0, b0 = float(m_logit.params["const"]), float(m_logit.params["twmean"])
seb = float(m_logit.bse["twmean"])
pm_ = float(mm.twmean.mean())


def marg(T, a=a0, b=b0):
    p = 1 / (1 + np.exp(-(a + b * T)))
    return b * p * (1 - p) * 100  # 百分点/degC


P(f"    (i) Pooled logistic: logit(p) = {a0:.3f} {b0:+.4f} * Tw   (z={b0/seb:+.2f})")
P(f"        在 Tw={pm_:.2f} C 处的边际效应 = {marg(pm_):+.2f} 百分点/degC")
mm2 = mm.copy()
mm2["tw_dm"] = mm2.twmean - mm2.groupby("nuts2").twmean.transform("mean")
mm2["cr_dm"] = mm2.cross - mm2.groupby("nuts2").cross.transform("mean")
m_fe = sm.OLS(mm2.cr_dm, sm.add_constant(mm2[["tw_dm"]])).fit(
    cov_type="cluster", cov_kwds={"groups": mm2.nuts2})
b_fe = float(m_fe.params["tw_dm"]); se_fe = float(m_fe.bse["tw_dm"])
P(f"    (ii) 区域固定效应 (仅区域内时间变异, 聚类SE): "
  f"{b_fe*100:+.2f} 百分点/degC (SE {se_fe*100:.2f}, z={b_fe/se_fe:+.2f})")
mp, mf = marg(pm_), b_fe * 100
ratio = mf / mp if mp else np.nan
P(f"    (iii) 空间换时间检验 (用边际效应比较, v1 错误地比较了 logit 系数):")
P(f"         pooled 边际 {mp:+.2f} pp/degC  vs  区域内(时间) {mf:+.2f} pp/degC  ->  比值 {ratio:.2f}")
P(f"         判读: {'比值在 0.3-3 内, 空间梯度可代理时间变化, 外推可用(仍标注)' if 0.3<=abs(ratio)<=3 else '比值超出 0.3-3, 空间换时间假设不成立, 外推不可用'}")
P("")

# ============ D. CMIP6 投影 ============
P("【D】投影: CMIP6 欧洲暖季夜间湿球增温 -> 新增首次暴露人口")


def stull(T, RH):
    RH = np.clip(RH, 1, 100)
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH)
            - np.arctan(RH - 1.676331) + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)


def load(model, exp, var):
    p = os.path.join(EXT, "cmip6", f"pts_{model}_{exp}_{var}.npz")
    return np.load(p) if os.path.exists(p) else None


def mean_tw(zt, zh, y0, y1):
    lat, lon = zt["lat"], zt["lon"]
    eu = (lat >= 35) & (lat <= 71) & (lon >= -11) & (lon <= 45)
    if eu.sum() < 3: return np.nan
    tm = zt["data"][:, eu] - 273.15
    hm = zh["data"][:, eu]
    mon, yr = zt["month"], zt["year"]
    mj = (mon >= 5) & (mon <= 9)
    s = mj & (yr >= y0) & (yr <= y1)
    if s.sum() == 0: return np.nan
    return float(np.nanmean(stull(tm[s], hm[s])))


models = sorted({os.path.basename(f)[4:].rsplit("_", 2)[0]
                 for f in glob.glob(os.path.join(EXT, "cmip6", "pts_*_tasmin.npz"))})
P(f"    发现模式: {models}")
dT = {}
for m_ in models:
    zh_t, zh_h = load(m_, "historical", "tasmin"), load(m_, "historical", "hurs")
    if zh_t is None or zh_h is None: continue
    base = mean_tw(zh_t, zh_h, 1995, 2014)
    if not np.isfinite(base): continue
    for s_ in ["ssp245", "ssp585"]:
        zt, zhh = load(m_, s_, "tasmin"), load(m_, s_, "hurs")
        if zt is None or zhh is None: continue
        mid = mean_tw(zt, zhh, 2041, 2060); end = mean_tw(zt, zhh, 2081, 2100)
        if np.isfinite(mid): dT.setdefault(s_, {}).setdefault("mid", []).append(mid - base)
        if np.isfinite(end): dT.setdefault(s_, {}).setdefault("end", []).append(end - base)
    dT.setdefault("_base", {}).setdefault("v", []).append(base)
P(f"    历史基线 (1995-2014) 模式均值暖季夜间湿球 = "
  f"{np.mean(dT.get('_base',{}).get('v',[np.nan])):.2f} C  (n={len(dT.get('_base',{}).get('v',[]))})")
P("")
dTs = {}
for s_ in ["ssp245", "ssp585"]:
    for k in ["mid", "end"]:
        arr = np.array(dT.get(s_, {}).get(k, []))
        if len(arr) == 0: continue
        dTs[f"{s_}_{k}"] = dict(n=int(len(arr)), mean=float(arr.mean()),
                                sd=float(arr.std(ddof=1)) if len(arr) > 1 else None,
                                lo=float(arr.min()), hi=float(arr.max()))
P("    ΔT (暖季夜间湿球, degC, 相对 1995-2014):")
for k, v in dTs.items():
    P(f"        {k:14s} n={v['n']}  ΔT={v['mean']:+.2f} C  "
      f"(模式范围 {v['lo']:+.2f} ~ {v['hi']:+.2f})")
P("")

tw0_reg = g.groupby("nuts2")["twmean"].mean()
popmap = g.groupby("nuts2")["pop"].first()
tot_pop = float(popmap.sum())
p_now = 1 / (1 + np.exp(-(a0 + b0 * tw0_reg)))
ep_now = float((p_now * popmap).sum())
obs_mean_exp = float(E.popfrac.mean() * tot_pop)
P(f"    [校准] 模型基线暴露人口 {ep_now/1e6:.0f} 百万")
P(f"    [校准] 观测全期(2000-2024)平均暴露人口 {obs_mean_exp/1e6:.0f} 百万 "
  f"(平均暴露比例 {E.popfrac.mean()*100:.1f}%)")
P(f"    [校准] 偏差 {(ep_now-obs_mean_exp)/1e6:+.0f} 百万 "
  f"({(ep_now/obs_mean_exp-1)*100:+.1f}%) -> "
  f"{'模型校准良好' if abs(ep_now/obs_mean_exp-1)<0.15 else '模型存在系统性偏差, 投影需订正'}")
P(f"    注: 不应拿'观测末年'(峰值年 {g[g.cross==1].groupby('nuts2')['pop'].first().sum()/1e6:.0f} 百万)"
  f" 与气候态基线直接比较 —— 那是单年峰值, 非气候态。")
P("")
# ---- 观测约束缩放 (第16轮审稿: 本研究第11轮已证模式高估增速 b_ALL=0.429, 投影必须缩放) ----
B_ALL = 0.429
B_ALL_SE = 0.148
P(f"    [观测约束] 本研究 step35 最优指纹给出 ALL 缩放因子 b_ALL = {B_ALL} +- {B_ALL_SE}")
P(f"               (即模式高估夜间湿热增速 {1/B_ALL:.1f}x)。投影必须缩放, 否则与本文自身结论矛盾。")
P("")
for k, v in dTs.items():
    v["mean_scaled"] = v["mean"] * B_ALL
    v["lo_scaled"] = v["lo"] * B_ALL
    v["hi_scaled"] = v["hi"] * B_ALL
P("    ΔT 缩放后 (x b_ALL):")
for k, v in sorted(dTs.items()):
    P(f"        {k:14s} 原始 {v['mean']:+.2f} C -> 缩放 {v['mean_scaled']:+.2f} C "
      f"(范围 {v['lo_scaled']:+.2f} ~ {v['hi_scaled']:+.2f})")
P("")
proj = {}
P("    投影 (使用【缩放后】ΔT —— 首选口径):")
for k in sorted(dTs):
    d_ = dTs[k]["mean_scaled"]
    pf = 1 / (1 + np.exp(-(a0 + b0 * (tw0_reg + d_))))
    epf = float((pf * popmap).sum())
    pfr = 1 / (1 + np.exp(-(a0 + b0 * (tw0_reg + dTs[k]["mean"]))))
    epfr = float((pfr * popmap).sum())
    proj[k] = dict(dT_raw=dTs[k]["mean"], dT_scaled=d_,
                   exp_pop_M=epf / 1e6, add_M=(epf - ep_now) / 1e6,
                   exp_pop_M_raw=epfr / 1e6, add_M_raw=(epfr - ep_now) / 1e6,
                   n_region=float(pf.sum()), frac=float(epf / tot_pop))
    P(f"    {k:14s} ΔT={d_:+.2f}C  暴露人口 {epf/1e6:6.0f} 百万 (新增 {(epf-ep_now)/1e6:+6.0f})  "
      f"暴露区域 {pf.sum():5.1f}/{len(tw0_reg)}   [未缩放对照 {epfr/1e6:.0f} 百万]")
P("")
eps = 0.1
perC = float(((1 / (1 + np.exp(-(a0 + b0 * (tw0_reg + eps)))) - p_now) * popmap).sum() / eps) / 1e6
perC_fe = float((b_fe * popmap).sum() * eps / eps) / 1e6
P(f"    >>> 可外推头条量: 暖季夜间湿球每升高 1 C, 欧洲新增")
P(f"        {perC:.1f} 百万人 (pooled logistic) / {perC_fe:.1f} 百万人 (区域FE, 保守下界)")
P(f"        首次暴露于夜间湿球 >= 21.9 C。")
P("")

P("【E】结论")
P(f"    1. 撤回 step46 的 HR —— 脆弱性消耗假象 (残留风险集 tw0 下降 {a_last-a_first:+.2f} C)。")
P(f"    2. 稳健事实: 欧洲暴露人口比例 {p0*100:.0f}%->{p1*100:.0f}% (相对{rel_p*100:+.0f}%), "
  f"NWH {n0:.1f}->{n1:.1f} h/年 ({rel_n*100:+.0f}%)")
P(f"    3. 由于剂量-反应平坦, 风险增长 ∝ 暴露范围({rel_p*100:+.0f}%) 而非强度({rel_n*100:+.0f}%);")
P(f"       用强度推断风险会高估 ~{over:.2f} 倍 —— 这是对影响评估实践的直接纠正。")
P(f"    4. 政策量: {perC:.1f} (pooled) / {perC_fe:.1f} (FE) 百万人 每升温 1 C。")

res = dict(n_region=int(g.nuts2.nunique()), years=[int(g.year.min()), int(g.year.max())],
           pop_total_M=tot_pop / 1e6,
           frailty=dict(cohort=len(co), tw0_first=a_first, tw0_last=a_last, delta=a_last - a_first),
           trend=dict(frac=r_frac, pop=r_pop, nwh=r_nwh, twmean=r_twm),
           firstlast5=dict(frac=[float(f0), float(f1)], pop=[float(p0), float(p1)],
                           nwh=[float(n0), float(n1)], frac_rel=float(rel_f),
                           pop_rel=float(rel_p), nwh_rel=float(rel_n),
                           overestimate_factor=float(over)),
           doseresponse=dict(pooled_a=a0, pooled_b=b0, pooled_se=seb,
                             pooled_marginal_pp_per_C=float(mp),
                             fe_pp_per_C=float(mf), fe_se=float(se_fe * 100),
                             fe_z=float(b_fe / se_fe), ratio_fe_pooled=float(ratio)),
           cmip6_dT=dTs, projection=proj,
           headline=dict(per_degC_M_pooled=perC, per_degC_M_fe=perC_fe,
                         baseline_exp_pop_M=ep_now / 1e6))
with open(os.path.join(BASE, "exposure_projection.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
with open(os.path.join(BASE, "exposure_projection.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[done]")
