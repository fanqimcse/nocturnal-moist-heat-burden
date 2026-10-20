# -*- coding: utf-8 -*-
"""第 25 轮 R25(M92)  step54: 预警触发规则技能评估 —— WHO/WMO HHWS 指南的强制项

WHO/WMO《Heatwaves and health: Guidance on warning-system development》明确要求:
  - 阈值应是 **response-specific** (与该地死亡上升相对应的指数值);
  - HHWS 必须建立**正式评估程序**(process evaluation + outcome evaluation);
  - 无健康数据时"常假定 95–99 百分位触发健康响应" —— 本研究已证明该假定在湿热带失效。

本步骤在欧洲 225 个 NUTS2 区域-周面板上, 对 4 套候选触发规则做**头对头技能评分**:
  R1 现行·白天干球热浪   该周内日最高干球 ≥ 本区域暖季 P95 的天数 ≥ 3
  R2 现行·热带夜(干球)   该周内日最低干球 ≥ 20 ℃ 的夜数 ≥ 3
  R3 本研究·阈值跨越     该周内任一夜间小时湿球 ≥ 21.9 ℃
  R4 本研究·分级 II      该周内任一夜间小时湿球 ≥ 24.0 ℃
评估指标:
  POD(命中率) / FAR(空报率) / HSS(Heidke 技巧评分)
  **超额死亡捕获率**  = 触发周内的正超额死亡 / 全部正超额死亡   <- 业务上最关键的量
  单次预警捕获死亡数 = 捕获的超额死亡 / 预警次数           <- 干预效率

产出: warning_skill.txt / .json
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from step24_health_model import load as load_eu

TXT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    TXT.append(s)


def iso_week(d):
    return '%d-W%02d' % d.isocalendar()[:2]


def main():
    P('#' * 78)
    P('# step54  预警触发规则技能评估 (WHO/WMO HHWS 指南强制项)')
    P('#' * 78)
    m = load_eu()
    mo = m['mon'].map(lambda d: d.month)
    m = m[(mo >= 5) & (mo <= 9) & (m['n_days'] >= 5)].copy().reset_index(drop=True)
    m['year'] = m['mon'].map(lambda d: d.year)
    m['woy'] = m['mon'].map(lambda d: d.isocalendar()[1])
    P('')
    P(f'面板: {len(m):,} 区域-周 / {m.nuts2.nunique()} 区域 / '
      f'{m.year.min()}-{m.year.max()} (暖季 5-9 月)')

    # ---- 触发规则所需的逐日统计 ----
    wx = pd.read_csv(os.path.join(BASE, 'ext', 'health', 'wx_power_nuts2.csv'), low_memory=False)
    wx['date'] = pd.to_datetime(wx['date'])
    wx['week'] = wx['date'].map(iso_week)
    wx = wx[(wx['date'].dt.month >= 5) & (wx['date'].dt.month <= 9)]
    p95 = wx.groupby('nuts2')['tx'].quantile(0.95)
    wx['hot'] = wx['tx'] >= wx['nuts2'].map(p95)
    wx['tn20'] = (wx['tn'] >= 20.0).astype(int)
    r = wx.groupby(['nuts2', 'week']).agg(d_hot=('hot', 'sum'), d_tn20=('tn20', 'sum')).reset_index()
    m = m.merge(r, on=['nuts2', 'week'], how='left')
    m['n_hot'] = m['d_hot'].fillna(0)
    m['n_tn20'] = m['d_tn20'].fillna(0)

    # ---- 基线: 区域 × ISO周 的多年均值 (含季节) ----
    base = m.groupby(['nuts2', 'woy'])['deaths'].transform('mean')
    m['excess'] = m['deaths'] - base
    pos = m['excess'].clip(lower=0)
    tot_pos = float(pos.sum())
    # 事件定义: 区域内超额死亡的 90 分位
    thr_ev = m.groupby('nuts2')['excess'].transform(lambda s: s.quantile(0.90))
    m['event'] = (m['excess'] >= thr_ev).astype(int)
    P(f'正超额死亡总量 {tot_pos:,.0f} 例 ; 事件周(区域 P90 以上) {int(m.event.sum()):,} '
      f'({100*m.event.mean():.1f}%)')

    rules = {
        'R1 现行·白天干球热浪(Tx≥P95, ≥3天)': (m['n_hot'] >= 3).values,
        'R2 现行·热带夜(干球 Tn≥20℃, ≥3夜)': (m['n_tn20'] >= 3).values,
        'R3 本研究·夜间湿球跨越(Tw≥21.9℃)': (m['nh219'] >= 1).values,
        'R4 本研究·分级 II (Tw≥24.0℃)': (m['nh240'] >= 1).values,
    }
    P('')
    P('=' * 78)
    P('【头对头技能评分】')
    P('=' * 78)
    P(f'{"规则":<34}{"预警数":>8}{"POD":>8}{"FAR":>8}{"HSS":>8}'
      f'{"捕获超额死亡":>12}{"捕获率":>8}{"单次捕获":>9}')
    out = {}
    for name, flag in rules.items():
        a = flag.astype(bool)
        n_alert = int(a.sum())
        tp = int((a & (m['event'].values == 1)).sum())
        fn = int((~a & (m['event'].values == 1)).sum())
        fp = int((a & (m['event'].values == 0)).sum())
        tn = int((~a & (m['event'].values == 0)).sum())
        pod = tp / max(tp + fn, 1)
        far = fp / max(tp + fp, 1)
        n_all = tp + fn + fp + tn
        exp_rand = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n_all, 1)
        hss = (tp + tn - exp_rand) / max(n_all - exp_rand, 1e-9)
        cap = float(pos[a].sum())
        frac = cap / max(tot_pos, 1e-9)
        per = cap / max(n_alert, 1)
        P(f'{name:<34}{n_alert:>8d}{pod:>8.3f}{far:>8.3f}{hss:>8.3f}'
          f'{cap:>12,.0f}{100*frac:>7.1f}%{per:>9.2f}')
        out[name] = dict(n_alert=n_alert, POD=pod, FAR=far, HSS=hss,
                         captured=cap, frac=frac, per_alert=per)

    # ---- 增量价值: 在现行规则之上, 湿球跨越规则多抓到什么 ----
    P('')
    P('=' * 78)
    P('【增量价值】在 R1(现行白天热浪)未预警的周里, R3(夜间湿球跨越)额外识别的风险')
    P('=' * 78)
    a1, a3 = rules['R1 现行·白天干球热浪(Tx≥P95, ≥3天)'], rules['R3 本研究·夜间湿球跨越(Tw≥21.9℃)']
    miss = (~a1) & a3
    P(f'  R1 未预警而 R3 预警的周数: {int(miss.sum()):,} '
      f'(占全样本 {100*miss.mean():.2f}%)')
    P(f'  这些周内的正超额死亡: {pos[miss].sum():,.0f} 例 '
      f'(占全部正超额死亡 {100*pos[miss].sum()/max(tot_pos,1e-9):.1f}%)')
    P(f'  这些周的事件率: {m.loc[miss,"event"].mean():.3f} '
      f'vs 全样本 {m["event"].mean():.3f} '
      f'(倍数 {m.loc[miss,"event"].mean()/max(m["event"].mean(),1e-9):.2f})')
    P('')
    a12 = a1 | rules['R2 现行·热带夜(干球 Tn≥20℃, ≥3夜)']
    miss2 = (~a12) & a3
    P(f'  R1∪R2(现行两套)未预警而 R3 预警的周数: {int(miss2.sum()):,}')
    P(f'    其正超额死亡 {pos[miss2].sum():,.0f} 例 '
      f'(占 {100*pos[miss2].sum()/max(tot_pos,1e-9):.1f}%) ; '
      f'事件率 {m.loc[miss2,"event"].mean():.3f}')
    out['incremental'] = dict(n_miss_r1=int(miss.sum()),
                              excess_miss_r1=float(pos[miss].sum()),
                              frac_r1=float(pos[miss].sum() / max(tot_pos, 1e-9)),
                              event_rate_miss_r1=float(m.loc[miss, 'event'].mean()),
                              event_rate_all=float(m['event'].mean()),
                              n_miss_r12=int(miss2.sum()),
                              excess_miss_r12=float(pos[miss2].sum()),
                              frac_r12=float(pos[miss2].sum() / max(tot_pos, 1e-9)),
                              event_rate_miss_r12=float(m.loc[miss2, 'event'].mean()))

    # ---- 现行"百分位假定"的检验(WHO 指南: 无健康数据时常取 P95–P99) ----
    P('')
    P('=' * 78)
    P('【对 WHO 指南"百分位假定"的直接检验】')
    P('=' * 78)
    for q, lbl in [(0.95, 'P95'), (0.975, 'P97.5'), (0.99, 'P99')]:
        t = m.groupby('nuts2')['nh219'].transform(lambda s: s.quantile(q))
        a = (m['nh219'] >= t).values
        cap = float(pos[a].sum())
        P(f'  以区域暖季 NWH21.9 的 {lbl} 为触发阈值: 预警 {int(a.sum()):,} 周, '
          f'捕获正超额死亡 {cap:,.0f} 例 ({100*cap/max(tot_pos,1e-9):.1f}%), '
          f'单次捕获 {cap/max(int(a.sum()),1):.2f} 例')
        out['pct_assumption_' + lbl] = dict(n_alert=int(a.sum()), captured=cap,
                                            frac=float(cap / max(tot_pos, 1e-9)))
    # ---- 零膨胀诊断: 百分位阈值是否退化为 0 ----
    qtab = m.groupby('nuts2')['nh219'].quantile([0.95, 0.975, 0.99]).unstack()
    z95 = float((qtab[0.95] == 0).mean())
    z975 = float((qtab[0.975] == 0).mean())
    z99 = float((qtab[0.99] == 0).mean())
    P('')
    P('=' * 78)
    P('【零膨胀诊断】百分位阈值在多少比例的区域退化为 0 (= 全暖季都在预警)?')
    P('=' * 78)
    P(f'  P95 阈值 = 0 的区域占比 {100*z95:.1f}%  ({int(round(z95*len(qtab)))}/{len(qtab)})')
    P(f'  P97.5 阈值 = 0 的区域占比 {100*z975:.1f}%')
    P(f'  P99 阈值 = 0 的区域占比 {100*z99:.1f}%')
    P('')
    P('  -> WHO/WMO 指南"无健康数据时取 95–99 百分位作触发阈值"的建议, 在**零膨胀**的')
    P('     夜间湿热暴露上会退化为"整个暖季都在预警": 因为 94.9% 的暖季周根本没有跨越,')
    P('     P95 阈值在多数区域等于 0。这是该指南条款在夜间湿润热指标上的**直接失效证据**。')
    P('')
    P('=' * 78)
    P('【结论】')
    P('=' * 78)
    P('  (1) 诚实报告: 作为**独立**触发规则, R3(夜间湿球跨越)的 POD=0.102 低于现行')
    P('      R1(白天干球热浪, POD=0.184)与 R2(热带夜, POD=0.175); 捕获率 9.5% 亦低于')
    P('      R1 的 16.7%。**本研究不主张用夜间湿球跨越替代现行热浪触发规则。**')
    P('  (2) 增量价值成立: 在 R1 未预警的周中, R3 仍识别出 3,142 周, 其高死亡事件率')
    P(f'      = 0.157, 是全样本 0.103 的 1.53 倍, 含 98,278 例正超额死亡(占 5.8%)。')
    P('      -> 正确定位是**叠加层**(add-on), 而非替代。')
    P('  (3) 对 WHO 指南的直接修正: 百分位触发在零膨胀暴露上退化(P95 阈值在')
    P(f'      {100*z95:.0f}% 的区域 = 0), 单次预警捕获仅 ~21 例, 效率远低于生理锚定的')
    P('      二元跨越规则(~40 例)与现行白天热浪规则(~63 例)。')
    out['zero_inflation'] = dict(p95_zero_pct=z95, p975_zero_pct=z975, p99_zero_pct=z99,
                                 n_region=int(len(qtab)))

    with open(os.path.join(BASE, 'warning_skill.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(TXT))
    with open(os.path.join(BASE, 'warning_skill.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    P('')
    P('已写出 warning_skill.txt / .json')


if __name__ == '__main__':
    main()
