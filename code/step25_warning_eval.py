# -*- coding: utf-8 -*-
"""第8轮 R8-4: 现行热预警规则 vs NHRI(夜间湿球) 规则的回顾性检验
  - 用"不含气象项"的基线模型定义超额死亡周
  - 比较三种触发规则的灵敏度/特异度/AUC 与漏报
产出: warning_eval.txt / warning_eval.json
"""
import os, sys, math, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step24_health_model import load, design, poisson_glm

BASE = os.path.dirname(os.path.abspath(__file__))


def baseline_expected(g):
    """年份因子 + Fourier(周) 季节, 不含暴露 -> 预期死亡"""
    X, names = design(g, [])
    y = g['deaths'].values.astype(float)
    b, se, phi = poisson_glm(X, y)
    mu = np.exp(np.clip(X @ b, -30, 30))
    return mu


def auc(score, flag):
    """Mann-Whitney U -> AUC"""
    s0 = score[flag == 0]; s1 = score[flag == 1]
    n0, n1 = len(s0), len(s1)
    if n0 == 0 or n1 == 0:
        return float('nan')
    ranks = pd.Series(np.concatenate([s0, s1])).rank().values
    r1 = ranks[n0:].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def main():
    m = load()
    m = m[m['n_days'] >= 5].copy()
    m = m.sort_values(['nuts2', 'mon']).reset_index(drop=True)
    print('panel rows %d  regions %d' % (len(m), m['nuts2'].nunique()))
    # ---- 1. 基线预期与超额 ----
    exp_list = []
    for nid, g in m.groupby('nuts2'):
        mu = baseline_expected(g)
        exp_list.append(pd.Series(mu, index=g.index))
    m['exp'] = pd.concat(exp_list)
    m = m[m['exp'] > 5].copy()
    m['ratio'] = m['deaths'] / m['exp']
    # 超额死亡周: 观测超过 Poisson 预期的上侧 10% 临界(近似 z>1.28) 且 相对超额>5%
    m['z'] = (m['deaths'] - m['exp']) / np.sqrt(m['exp'])
    m['excess'] = ((m['z'] > 1.2816) & (m['ratio'] > 1.05)).astype(int)
    print('excess weeks: %d / %d = %.1f%%' % (m['excess'].sum(), len(m), 100 * m['excess'].mean()))

    # ---- 2. 三种预警规则 (区域气候态 95 分位, 全期) ----
    rules = {
        'current_TX(日最高干球)': 'tx_max',
        'current_TN(日最低干球/热夜)': 'tn_max',
        'NHRI_Tw(夜间湿球,本研究)': 'tw_night_min_max',
    }
    q = m.groupby('nuts2')[list(rules.values())].quantile(0.95)
    out = {}
    lines = []
    lines.append('夏季 NUTS2 x ISO周面板; 超额死亡周 = 观测 > Poisson基线预期(z>1.28 且 +5%)')
    lines.append('预警触发 = 该区域全期夏季气候态 95 分位以上')
    lines.append('')
    lines.append('%-32s %8s %8s %8s %8s %8s' % ('规则', '灵敏度', '特异度', 'AUC', '漏报率', '精确率'))
    for name, col in rules.items():
        thr = m['nuts2'].map(q[col])
        trig = (m[col] > thr).astype(int)
        tp = int(((trig == 1) & (m['excess'] == 1)).sum())
        fp = int(((trig == 1) & (m['excess'] == 0)).sum())
        fn = int(((trig == 0) & (m['excess'] == 1)).sum())
        tn = int(((trig == 0) & (m['excess'] == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        miss = fn / max(tp + fn, 1)
        a = auc(m[col].values, m['excess'].values)
        lines.append('%-32s %8.3f %8.3f %8.3f %8.3f %8.3f'
                     % (name, sens, spec, a, miss, prec))
        out[name] = dict(sensitivity=sens, specificity=spec, auc=a, miss_rate=miss,
                         precision=prec, tp=tp, fp=fp, fn=fn, tn=tn,
                         n_excess=int(m['excess'].sum()), n_trig=int(trig.sum()))

    # ---- 3b. 联合规则: 补充而非替代 ----
    txc = (m['tx_max'] > m['nuts2'].map(q['tx_max'])).astype(int)
    tnc = (m['tn_max'] > m['nuts2'].map(q['tn_max'])).astype(int)
    twc = (m['tw_night_min_max'] > m['nuts2'].map(q['tw_night_min_max'])).astype(int)
    cur = ((txc == 1) | (tnc == 1)).astype(int)
    cur_tw = ((txc == 1) | (tnc == 1) | (twc == 1)).astype(int)
    lines.append('')
    lines.append('联合规则对比 (关键: 夜间湿球是"补充"还是"替代")')
    for nm, trig in [('现行 (TX 或 TN)', cur), ('现行 + NHRI (TX 或 TN 或 Tw)', cur_tw)]:
        tp = int(((trig == 1) & (m['excess'] == 1)).sum())
        fp = int(((trig == 1) & (m['excess'] == 0)).sum())
        fn = int(((trig == 0) & (m['excess'] == 1)).sum())
        tn = int(((trig == 0) & (m['excess'] == 0)).sum())
        lines.append('  %-32s 灵敏度 %.3f  漏报率 %.3f  精确率 %.3f  触发周数 %d'
                     % (nm, tp / max(tp + fn, 1), fn / max(tp + fn, 1),
                        tp / max(tp + fp, 1), int(trig.sum())))
        out['rule_' + nm] = dict(sensitivity=tp / max(tp + fn, 1), miss=fn / max(tp + fn, 1),
                                 precision=tp / max(tp + fp, 1), n_trig=int(trig.sum()),
                                 tp=tp, fp=fp, fn=fn, tn=tn)
    gain_tp = int((((cur_tw == 1) & (m['excess'] == 1)).sum()) - (((cur == 1) & (m['excess'] == 1)).sum()))
    gain_trig = int(cur_tw.sum() - cur.sum())
    lines.append('  加入夜间湿球后: 多捕获超额死亡周 %d 个 (%.1f%%), 额外触发 %d 周'
                 % (gain_tp, 100 * gain_tp / max(int(((cur == 1) & (m['excess'] == 1)).sum()), 1), gain_trig))
    out['gain'] = dict(extra_excess_weeks=gain_tp, extra_triggers=gain_trig)
    # 组合评分 AUC (样本内 Logit, 仅作上界参考)
    try:
        import statsmodels.api as sm
        zt = (m['tx_max'] - m['tx_max'].mean()) / m['tx_max'].std()
        zw = (m['tw_night_min_max'] - m['tw_night_min_max'].mean()) / m['tw_night_min_max'].std()
        zn = (m['tn_max'] - m['tn_max'].mean()) / m['tn_max'].std()
        for nm, cols in [('Logit: TX', [zt]), ('Logit: Tw', [zw]),
                         ('Logit: TX+Tw', [zt, zw]), ('Logit: TX+TN+Tw', [zt, zn, zw])]:
            Xc = sm.add_constant(np.column_stack(cols))
            r = sm.Logit(m['excess'].values, Xc).fit(disp=0)
            sc = r.predict(Xc)
            lines.append('  %-22s 样本内 AUC = %.4f  (系数 %s)'
                         % (nm, auc(sc, m['excess'].values),
                            np.round(r.params[1:], 4).tolist()))
    except Exception as e:
        lines.append('  (Logit 组合 AUC 计算失败: %s)' % str(e)[:80])

    # ---- 3. 关键交叉: 现行规则未触发的超额周, NHRI 能捕获多少 ----
    blind = (txc == 0) & (tnc == 0) & (m['excess'] == 1)
    n_blind = int(blind.sum())
    n_caught = int((blind & (twc == 1)).sum())
    lines.append('')
    lines.append('现行规则(TX 与 TN 均未触发) 却发生超额死亡的周: %d' % n_blind)
    lines.append('  其中 NHRI(夜间湿球)触发: %d (%.1f%%)' % (n_caught, 100 * n_caught / max(n_blind, 1)))
    lines.append('  即现行体系的盲区中, 夜间湿球指标可挽回 %.1f%%' % (100 * n_caught / max(n_blind, 1)))
    out['blind'] = dict(n_current_missed_excess=n_blind, n_recovered_by_tw=n_caught,
                        recovered_frac=n_caught / max(n_blind, 1))

    # ---- 4. 两种规则都触发 vs 仅 NHRI 触发 的超额强度 ----
    for lab, sel in [('TX触发', txc == 1), ('TN触发', tnc == 1), ('NHRI触发', twc == 1),
                     ('仅NHRI触发(现行未触发)', (twc == 1) & (txc == 0) & (tnc == 0))]:
        sub = m[sel]
        if len(sub):
            lines.append('%s : n=%d  平均死亡比 obs/exp = %.4f  超额周占比 %.1f%%'
                         % (lab, len(sub), sub['ratio'].mean(), 100 * sub['excess'].mean()))
            out['group_' + lab] = dict(n=len(sub), mean_ratio=float(sub['ratio'].mean()),
                                       excess_frac=float(sub['excess'].mean()))

    txt = '\n'.join(lines)
    print(txt)
    with open('warning_eval.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    with open('warning_eval.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
