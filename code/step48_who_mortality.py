# -*- coding: utf-8 -*-
"""
step48 (v2): 湿热带健康证据 —— WHO 多国年度死亡面板 (攻击第17轮最大缺口)

数据源与已验证的口径:
  ext/who/tgt_all.pkl  (由 Morticd10_part1..6 / Morticd9 / Morticd8 抽取目标国家)
  全因码: ICD-8='A000' (马来西亚'B000') ; ICD-9='B00' ; ICD-10='AAA'
  **Deaths1 恒为总死亡** (已验证: 巴西1996 男530,068+女376,502+未知2,312=908,882 ≈ 官方90万;
   且 sum(Deaths2..26) 恰等于 Deaths1, 证明其为年龄分组而非并列项)
  人口: ext/who/pop 的 Pop1 (总人口)

关键方法控制:
  (1) 国家FE + 国家特定二次时间趋势 -> 识别来自年际异常 (避免"死亡长期下降 vs NWH长期上升"的虚假负相关)
  (2) **ICD版本固定效应** -> ICD版本切换会改变登记口径, 必须控制
  (3) 粗死亡率合理性筛查
"""
import json, os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
WHO = os.path.join(BASE, "ext", "who")
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


P("WHO 多国年度死亡面板: 湿热带夜间湿球与死亡 (step48 v2)")
P("")

# ---------------- 1. 全因死亡 ----------------
ALLCODE = {"A000", "B000", "B00", "AAA"}
d = pd.read_pickle(os.path.join(WHO, "tgt_all.pkl"))
d = d[d.Cause.isin(ALLCODE)].copy()
d["Deaths1"] = pd.to_numeric(d["Deaths1"], errors="coerce")
d["icdv"] = d.file.map({"tgt8.csv": "icd8", "tgt9.csv": "icd9", "tgt10.csv": "icd10"})
mor = d.groupby(["Country", "Year", "icdv"])["Deaths1"].sum().reset_index()
mor = mor.rename(columns={"Country": "who", "Deaths1": "deaths"})
mor = mor[mor.deaths > 0]
rank = {"icd8": 0, "icd9": 1, "icd10": 2}
mor["rk"] = mor.icdv.map(rank)
mor = mor.sort_values(["who", "Year", "rk"]).drop_duplicates(["who", "Year"], keep="last")
P(f"【1】全因死亡: 国家-年 {len(mor)} ; 国家 {mor.who.nunique()} ; "
  f"年 {int(mor.Year.min())}-{int(mor.Year.max())}")
P(f"    版本分布 {mor.icdv.value_counts().to_dict()}")
P("")

# ---------------- 2. 人口 ----------------
pp = pd.read_csv(os.path.join(WHO, "pop"), low_memory=False,
                 usecols=["Country", "Year", "Sex", "Pop1"])
pp["Pop1"] = pd.to_numeric(pp.Pop1, errors="coerce")
pp = pp.groupby(["Country", "Year"])["Pop1"].sum().reset_index().rename(
    columns={"Country": "who", "Pop1": "pop"})
pp = pp[pp["pop"] > 0]
P(f"【2】人口: 国家-年 {len(pp)}")
P("")

# ---------------- 3. 站 -> 国家 ----------------
GH2WHO = {
    "RP": (3300, "Philippines", "tropics"), "CO": (2130, "Colombia", "tropics"),
    "MY": (3236, "Malaysia", "tropics"), "TH": (3380, "Thailand", "tropics"),
    "BR": (2070, "Brazil", "tropics"), "PE": (2370, "Peru", "subtropics"),
    "CU": (2150, "Cuba", "tropics"), "GT": (2250, "Guatemala", "subtropics"),
    "BL": (2060, "Bolivia", "subtropics"), "RQ": (2380, "Puerto Rico", "tropics"),
    "JA": (3160, "Japan", "temperate"), "CI": (2120, "Chile", "temperate"),
    "AU": (4010, "Austria", "temperate"), "FM": (5108, "Micronesia", "tropics"),
}

P("【3】站级 NWH (key='S', thr=21.9, tag='A')")
hs = pd.read_csv(os.path.join(BASE, "humidity_decomp_series.csv"), low_memory=False)
nwh = hs[(hs.key == "S") & (hs.thr == 21.9) & (hs.tag == "A")][["station", "year", "val"]].copy()
nwh = nwh.rename(columns={"year": "Year", "val": "nwh"})
nwh["ctry2"] = nwh.station.str[:2]
mt = pd.read_csv(os.path.join(BASE, "metrics_v3.csv"), low_memory=False)
tnm = mt[["station", "year", "t_night_mean"]].rename(columns={"year": "Year"})
tnm = tnm[tnm.t_night_mean.between(-30, 45)]
tnm["ctry2"] = tnm.station.str[:2]
tnm = tnm.groupby(["ctry2", "Year"])["t_night_mean"].mean().reset_index()

rows = []
for c2, (w, nm, band) in GH2WHO.items():
    s = nwh[nwh.ctry2 == c2]
    if len(s) == 0: continue
    gs = s.groupby("Year")["nwh"].agg(["mean", "size"]).reset_index()
    gs["who"] = w; gs["name"] = nm; gs["band"] = band; gs["nst"] = gs["size"]
    gs = gs.merge(tnm[tnm.ctry2 == c2][["Year", "t_night_mean"]], on="Year", how="left")
    rows.append(gs)
EXP = pd.concat(rows).rename(columns={"mean": "nwh_nat", "t_night_mean": "tn_nat"})
P(f"    国家-年暴露 {len(EXP)} ; 国家 {EXP.who.nunique()}")
P("")

# ---------------- 4. 面板 ----------------
panel = EXP.merge(mor, on=["who", "Year"], how="inner").merge(pp, on=["who", "Year"], how="inner")
panel = panel[(panel.deaths > 0) & (panel["pop"] > 0) & panel.nwh_nat.notna()].copy()
panel["rate"] = panel.deaths / panel["pop"] * 1e5
bad = (panel.rate < 200) | (panel.rate > 2500)
P(f"【4】面板: 合并后 {len(panel)} 国家-年")
P(f"    质量筛查(粗死亡率 200-2500/10万): 剔除 {int(bad.sum())} 个异常国家-年")
panel = panel[~bad].copy()

panel["nwh_z"] = panel.groupby("who")["nwh_nat"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0)
panel["tn_z"] = panel.groupby("who")["tn_nat"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0)

g = panel.groupby(["name", "band"]).agg(
    nst=("nst", "max"), years=("Year", "size"), y0=("Year", "min"), y1=("Year", "max"),
    nwh=("nwh_nat", "mean"), deaths=("deaths", "mean"), rate=("rate", "mean")).reset_index()
P("")
P(f"    {'国家':14s}{'带':11s}{'站':>3s}{'年数':>5s}{'年份':>12s}{'NWH均值':>10s}{'年均死亡':>12s}{'粗死亡率':>9s}")
for r in g.sort_values("years", ascending=False).itertuples():
    P(f"    {r.name:14s}{r.band:11s}{int(r.nst):3d}{r.years:5d}"
      f"{str(int(r.y0))+'-'+str(int(r.y1)):>12s}{r.nwh:10.1f}{r.deaths:12,.0f}{r.rate:9.1f}")
P(f"    ------------------------------------------------------------------")
P(f"    合计 {len(panel)} 国家-年 / {panel.who.nunique()} 国")
panel.to_csv(os.path.join(BASE, "who_panel.csv"), index=False)
P("")

# ---------------- 5. 模型 ----------------
P("【5】Poisson: 国家FE + 国家特定二次趋势 + ICD版本FE + offset(log pop)")
P("    识别来自 NWH 的**年际异常**; z = 国家内标准化 (每 1 个国内 SD)")


def fit(df, var, label, extra=None):
    d = df.dropna(subset=[var, "deaths", "pop"]).copy()
    if len(d) < 60 or d.who.nunique() < 4:
        P(f"    {label:38s} 样本不足 (n={len(d)}, {d.who.nunique()}国)"); return None
    d["t"] = d.Year - d.Year.min()
    cols = {var: d[var].values.astype(float)}
    if extra:
        for e in extra:
            if e in d: cols[e] = d[e].values.astype(float)
    for c in sorted(d.who.unique()):
        ind = (d.who == c).values.astype(float)
        cols[f"fe_{c}"] = ind
        cols[f"t1_{c}"] = ind * d.t.values
        cols[f"t2_{c}"] = ind * d.t.values ** 2
    for v in sorted(d.icdv.unique())[1:]:
        cols[f"icd_{v}"] = (d.icdv == v).values.astype(float)
    X = pd.DataFrame(cols)
    y = d.deaths.values.astype(float)
    off = np.log(d["pop"].values.astype(float))
    try:
        m = sm.GLM(y, X.values, family=sm.families.Poisson(), offset=off).fit(scale="X2")
        i = list(X.columns).index(var)
        b, se = m.params[i], m.bse[i]
        return dict(label=label, n=int(len(d)), nctry=int(d.who.nunique()),
                    RR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                    hi=float(np.exp(b + 1.96 * se)), z=float(b / se), p=float(m.pvalues[i]),
                    scale=float(m.scale))
    except Exception as e:
        P(f"    {label}: 失败 {e}"); return None


def show(r):
    if not r: return None
    P(f"    {r['label']:38s} RR/SD={r['RR']:.4f} [{r['lo']:.4f},{r['hi']:.4f}] "
      f"z={r['z']:+.2f} p={r['p']:.4f}  n={r['n']}({r['nctry']}国) 离散={r['scale']:.1f}")
    return r


res = {}
res["main"] = show(fit(panel, "nwh_z", "主规格: 夜间湿球 z (全样本)"))
P("")
P("    --- 分气候带 ---")
for band in ["tropics", "subtropics", "temperate"]:
    sub = panel[panel.band == band]
    res[f"band_{band}"] = show(fit(sub, "nwh_z", f"{band} ({sub.who.nunique()}国)"))
P("")
P("    --- 敏感性 ---")
sub = panel[panel.nst >= 2]
res["multi"] = show(fit(sub, "nwh_z", f"仅>=2站国家 ({sub.who.nunique()}国)"))
sub2 = panel[panel.Year >= 1995]
res["recent"] = show(fit(sub2, "nwh_z", f"仅1995+ ICD-10时代 ({sub2.who.nunique()}国)"))
P("")
P("    --- 干球对照 ---")
res["tn"] = show(fit(panel, "tn_z", "对照: 夜间干球 z"))
res["nwh_ctrl_tn"] = show(fit(panel, "nwh_z", "湿球 z (控制干球 z)", extra=["tn_z"]))
res["tn_ctrl_nwh"] = show(fit(panel, "tn_z", "干球 z (控制湿球 z)", extra=["nwh_z"]))
P("")

P("【6】与既有结果的方向一致性")
P("    欧洲(周尺度, 连续夜间最低湿球): RR/SD = 1.0068 [1.0054,1.0083] (z=+9.31)")
P("    新加坡(月尺度, NWH_25.0):        RR/SD = 1.0122 [1.0004,1.0241] (z=+2.03)")
if res.get("main"):
    m_ = res["main"]
    P(f"    本面板(年尺度, {m_['nctry']}国湿热带):      RR/SD = {m_['RR']:.4f} "
      f"[{m_['lo']:.4f}, {m_['hi']:.4f}] (z={m_['z']:+.2f})")
P("")
P("【7】必须随结果报告的局限")
P("    (a) 年度口径识别来自年际异常, 无法完全排除社会经济年际波动的共同驱动;")
P("    (b) 国家暴露由有限站点代理, 单站国家代表性弱;")
P("    (c) 无年龄结构控制 (仅总人口 offset); 发展中国家登记覆盖率随年份变化;")
P("    (d) 已控制 ICD 版本切换, 但登记完整性的渐进变化无法完全消除。")

with open(os.path.join(BASE, "who_mortality.json"), "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in res.items() if v}, f, ensure_ascii=False, indent=2)
with open(os.path.join(BASE, "who_mortality.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[done]")
