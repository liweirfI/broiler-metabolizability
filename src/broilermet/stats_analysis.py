"""Descriptive statistics, repeatability, between-week consistency, low- vs. high-RFI groups, and correlations."""
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

WEEKLY_TRAITS = [('ADFI', 'g/d'), ('ADG', 'g/d'), ('FCR', 'g:g'), ('DEW', 'g/d'), ('GEE', 'kcal/d'), ('CPE', 'g/d'),
                 ('MDM', 'g/d'), ('AMEn', 'kcal/d'), ('MCP', 'g/d'), ('cMDM', '%'), ('cAMEn', '%'), ('cMCP', '%'),
                 ('DMout_per_kg', 'g/kg'), ('Nout_per_kg', 'g/kg')]
REP_TRAITS = ['ADFI', 'AMEn', 'DEW', 'CPE', 'RFI', 'ADG', 'cAMEn', 'cMDM', 'cMCP', 'FCR']
GROUP_TRAITS = [('ADFI', 'g/d'), ('ADG', 'g/d'), ('FCR', 'g:g'), ('DEW', 'g/d'), ('GEE', 'kcal/d'), ('CPE', 'g/d'),
                ('AMEn', 'kcal/d'), ('MCP', 'g/d'), ('cMDM', '%'), ('cAMEn', '%'), ('cMCP', '%'),
                ('Nout_per_kg', 'g/kg'), ('BW42', 'g'), ('AbP', '%'), ('BreastP', '%'), ('LegMuscP', '%')]
BIRD_LEVEL = ('BW42', 'AbP', 'BreastP', 'LegMuscP', 'AbF')
BOOT_SEED = 20261011
N_BOOT = 2000


def bh(p):
    """Benjamini-Hochberg adjusted P-values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    cm = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q[o] = np.minimum(cm, 1)
    return q


def week_comparison(d):
    rows = []
    for t, u in WEEKLY_TRAITS:
        a, b = d[t + '_P1'], d[t + '_P2']
        m = a.notna() & b.notna()
        diff = (b - a)[m]
        tt = stats.ttest_rel(b[m], a[m])
        rows.append(dict(trait=t, unit=u, n1=int(a.notna().sum()), n2=int(b.notna().sum()), npair=int(m.sum()),
                         mean1=a.mean(), sd1=a.std(), mean2=b.mean(), sd2=b.std(), diff=diff.mean(),
                         se=diff.std(ddof=1) / np.sqrt(m.sum()), pct=100 * diff.mean() / a[m].mean(),
                         t=tt.statistic, P=tt.pvalue))
    return pd.DataFrame(rows)


def reml_repeatability(y1, y2):
    """REML for y_ij = mu + w_j + b_i + e_ij with 2 records per bird (birds with 1 record included).
    Returns repeatability, bird variance, residual variance."""
    both = ~np.isnan(y1) & ~np.isnan(y2)
    o1 = ~np.isnan(y1) & np.isnan(y2)
    o2 = np.isnan(y1) & ~np.isnan(y2)
    Y2 = np.column_stack([y1[both], y2[both]])
    s1, s2 = y1[o1], y2[o2]
    npair, n1, n2 = len(Y2), len(s1), len(s2)
    nobs, p = 2 * npair + n1 + n2, 2
    Xp = np.array([[1., 0.], [1., 1.]])

    def parts(lam):
        dpair = 1 + 2 * lam
        Hinv = np.array([[1 + lam, -lam], [-lam, 1 + lam]]) / dpair
        hs = 1 + lam
        XtHX = npair * Xp.T @ Hinv @ Xp + n1 * np.outer([1, 0], [1, 0]) / hs + n2 * np.outer([1, 1], [1, 1]) / hs
        XtHy = Xp.T @ Hinv @ Y2.sum(0) + np.array([1, 0]) * s1.sum() / hs + np.array([1, 1]) * s2.sum() / hs
        beta = np.linalg.solve(XtHX, XtHy)
        R = Y2 - Xp @ beta
        q = (np.einsum('ij,jk,ik->', R, Hinv, R) + ((s1 - beta[0]) ** 2).sum() / hs
             + ((s2 - beta.sum()) ** 2).sum() / hs)
        logdet_h = npair * np.log(dpair) + (n1 + n2) * np.log(hs)
        return q, logdet_h, np.linalg.slogdet(XtHX)[1]

    def nll(rho):
        q, ldh, ldx = parts(rho / (1 - rho))
        return (nobs - p) * np.log(q / (nobs - p)) + ldh + ldx

    rho = optimize.minimize_scalar(nll, bounds=(1e-8, 1 - 1e-8), method='bounded', options=dict(xatol=1e-10)).x
    lam = rho / (1 - rho)
    s2e = parts(lam)[0] / (nobs - p)
    return rho, lam * s2e, s2e


def pitman_morgan(a, b):
    """Test of equal variances of 2 paired samples (correlation between sum and difference)."""
    r, pv = stats.pearsonr(a + b, a - b)
    return r, pv


def repeatability(d, n_boot=N_BOOT):
    n = len(d)
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []
    for t in REP_TRAITS:
        y1 = d[t + '_P1'].to_numpy(float)
        y2 = d[t + '_P2'].to_numpy(float)
        r1, sb2, se2 = reml_repeatability(y1, y2)
        bs = np.array([reml_repeatability(y1[i], y2[i])[0] for i in boot_idx]) if n_boot else np.array([np.nan])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        m = ~np.isnan(y1) & ~np.isnan(y2)
        r = stats.pearsonr(y1[m], y2[m])[0]
        a2, b2 = pd.Series(y1[m]), pd.Series(y2[m])
        low1 = a2 <= a2.quantile(0.2)
        stay = 100 * (low1 & (b2 <= b2.quantile(0.2))).sum() / low1.sum()
        move = 100 * (low1 & (b2 >= b2.quantile(0.8))).sum() / low1.sum()
        r2 = 2 * r1 / (1 + r1)
        pm_r, pm_p = pitman_morgan(y1[m], y2[m])
        rows.append(dict(trait=t, r1=r1, lo=lo, hi=hi, r2=r2, gain_pct=100 * (r2 / r1 - 1), r=r,
                         sd_ratio=np.std(y2[m], ddof=1) / np.std(y1[m], ddof=1), pm_P=pm_p,
                         stay=stay, move=move, n_low=int(low1.sum()), npair=int(m.sum()),
                         nrec=int((~np.isnan(y1)).sum() + (~np.isnan(y2)).sum())))
    return pd.DataFrame(rows)


def rfi_consistency(d):
    m = d.RFI_P1.notna() & d.RFI_P2.notna()
    e1 = (d.RFI_P1 <= d.RFI_P1.median())[m]
    e2 = (d.RFI_P2 <= d.RFI_P2.median())[m]
    chi = stats.chi2_contingency(pd.crosstab(e1, e2), correction=False)
    po = (e1 == e2).mean()
    pe = e1.mean() * e2.mean() + (1 - e1.mean()) * (1 - e2.mean())
    out = dict(n=int(m.sum()), eff_both=int((e1 & e2).sum()), ineff_both=int((~e1 & ~e2).sum()),
               changed=int((e1 != e2).sum()), pct_same=100 * po, chi2=chi[0], P=chi[1], kappa=(po - pe) / (1 - pe))
    for a, b in (('P1', 'P2'), ('P2', 'P1')):
        za = (d['RFI_' + a] - d['RFI_' + a].mean()) / d['RFI_' + a].std()
        zb = (d['RFI_' + b] - d['RFI_' + b].mean()) / d['RFI_' + b].std()
        mm = za.notna() & zb.notna()
        sel = za[mm] <= za[mm].quantile(0.25)
        out[f'selected_on_{a}'] = dict(n=int(sel.sum()), retained_pct=100 * zb[mm][sel].mean() / za[mm][sel].mean())
    return out


def rfi_groups(d, frac=0.2):
    rows, info, members = [], {}, {}
    for p in ('P1', 'P2'):
        r = d['RFI_' + p]
        ok = r.notna()
        k = int(round(frac * ok.sum()))
        o = r.rank(method='first')
        lo = ok & (o <= k)
        hi = ok & (o > ok.sum() - k)
        info[p] = dict(k=k, low_mean=r[lo].mean(), high_mean=r[hi].mean())
        members[p] = (lo, hi)
        for t, u in GROUP_TRAITS:
            c = t if t in BIRD_LEVEL else f'{t}_{p}'
            a, b = d.loc[lo, c].dropna(), d.loc[hi, c].dropna()
            w = stats.ttest_ind(a, b, equal_var=False)
            pooled = ((a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(a) + len(b) - 2))
            rows.append(dict(period=p, trait=t, unit=u, low=a.mean(), high=b.mean(), diff=a.mean() - b.mean(),
                             pct=100 * (a.mean() / b.mean() - 1), sed=np.sqrt(a.var() / len(a) + b.var() / len(b)),
                             SEM=np.sqrt(pooled / ((len(a) + len(b)) / 2)), P=w.pvalue, n_low=len(a), n_high=len(b)))
    T = pd.DataFrame(rows)
    T['q'] = np.nan
    for p in ('P1', 'P2'):
        mm = T.period == p
        T.loc[mm, 'q'] = bh(T.loc[mm, 'P'].values)
    # cAME and cAMEn compared in the same birds (birds with both coefficients)
    same = []
    for p in ('P1', 'P2'):
        lo, hi = members[p]
        ok = d[f'cAME_{p}'].notna() & d[f'cAMEn_{p}'].notna()
        for c in ('cAME', 'cAMEn'):
            a, b = d.loc[lo & ok, f'{c}_{p}'], d.loc[hi & ok, f'{c}_{p}']
            same.append(dict(period=p, trait=c, n_low=len(a), n_high=len(b), diff=a.mean() - b.mean(),
                             P=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return T, info, pd.DataFrame(same)


def rfi_classes(d):
    e1 = d.RFI_P1 <= d.RFI_P1.median()
    e2 = d.RFI_P2 <= d.RFI_P2.median()
    grp = pd.Series(np.where(e1 & e2, 'both_eff', np.where(~e1 & ~e2, 'both_ineff',
                                                          np.where(e1 & ~e2, 'w1_only', 'w2_only'))))
    grp[d.RFI_P1.isna() | d.RFI_P2.isna()] = np.nan
    names = ['both_eff', 'w1_only', 'w2_only', 'both_ineff']
    rows = []
    for c in ['BW42', 'ADG_T', 'FCR_T', 'RFI_T', 'cMDM_T', 'cAMEn_T', 'cMCP_T', 'AbP', 'BreastP', 'LegMuscP']:
        vals = [d.loc[grp == g, c].dropna() for g in names]
        F, pv = stats.f_oneway(*vals)
        rows.append(dict(trait=c, **{g: v.mean() for g, v in zip(names, vals)},
                         **{'n_' + g: len(v) for g, v in zip(names, vals)}, F=F, P=pv))
    S = pd.DataFrame(rows)
    S['q'] = bh(S.P)
    return S


def partial_correlations(d):
    def pcor(x, y, z):
        m = x.notna() & y.notna() & z.notna()
        x, y, z = x[m], y[m], z[m]
        Z = np.column_stack([np.ones(m.sum()), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        r = np.corrcoef(rx, ry)[0, 1]
        df = m.sum() - 3
        t = r * np.sqrt(df / (1 - r ** 2))
        return r, 2 * stats.t.sf(abs(t), df), int(m.sum())
    rows = []
    for c in ['AbF', 'AbP', 'BreastP', 'LegMuscP']:
        for e in [f'{t}_{p}' for p in ('P1', 'P2') for t in ('RFI', 'FCR', 'AMEn', 'cMDM', 'cAMEn', 'cMCP')]:
            r, pv, n = pcor(d[c], d[e], d.BW42)
            rows.append(dict(carcass=c, efficiency=e, n=n, r_partial=r, P=pv))
    S = pd.DataFrame(rows)
    S['q'] = bh(S.P)
    return S


def change_vs_intake(d):
    rows = []
    for t in ('cMDM', 'cAMEn', 'cMCP'):
        dA = d.ADFI_P2 - d.ADFI_P1
        dC = d[t + '_P2'] - d[t + '_P1']
        m = dA.notna() & dC.notna()
        r, pv = stats.pearsonr(dA[m], dC[m])
        rows.append(dict(trait=t, r=r, P=pv, n=int(m.sum())))
    return pd.DataFrame(rows)


def excreta_composition(d):
    out = {}
    for p in ('P1', 'P2'):
        dm = 1 - d[f'excr_moist_pct_{p}'] / 100
        ge = d[f'excr_GE_kcal_g_{p}'] / dm
        cp = d[f'excr_CP_pct_{p}'] / dm
        dmr = d[f'DEW_{p}'] / d[f'DMI_{p}']
        mo = d[f'excr_moist_pct_{p}']
        out[p] = dict(GE_per_DM_cv=100 * ge.std() / ge.mean(), CP_per_DM_cv=100 * cp.std() / cp.mean(),
                      DM_output_ratio_cv=100 * dmr.std() / dmr.mean(), moisture_mean=mo.mean(),
                      moisture_cv=100 * mo.std() / mo.mean())
    k = 8.22 * d.feed_CP_pct.iloc[0] / 100 / (6.25 * d.feed_GE_kcal_g.iloc[0])
    m = d.cAME_T.notna() & d.cAMEn_T.notna()
    out.update(k=k, r_cAME_cAMEn_T=np.corrcoef(d.cAME_T[m], d.cAMEn_T[m])[0, 1], sd_Ncorr_T=(k * d.cMCP_T).std(),
               sd_cAMEn_T=d.cAMEn_T.std(), r_cMDM_cAMEn_T=d[['cMDM_T', 'cAMEn_T']].corr().iloc[0, 1])
    return out


CORR_COLS = ['BW28', 'BW35', 'BW42', 'ADG_P1', 'ADG_P2', 'ADFI_P1', 'ADFI_P2', 'FCR_T', 'RFI_T', 'AbF', 'AbP',
             'BreastP', 'LegMuscP', 'EX65_T', 'DEW_T', 'cAME_T', 'cMDM_T', 'cAMEn_T', 'cMCP_T']


def run_all(d, outdir, n_boot=N_BOOT):
    os.makedirs(outdir, exist_ok=True)
    w = lambda df, name: df.to_csv(os.path.join(outdir, name), sep='\t', index=False)
    w(week_comparison(d), 'week_comparison.tsv')
    w(repeatability(d, n_boot), 'repeatability.tsv')
    T, info, same = rfi_groups(d)
    w(T, 'rfi_groups.tsv')
    w(same, 'rfi_groups_cAME_same_birds.tsv')
    w(rfi_classes(d), 'rfi_classes.tsv')
    w(partial_correlations(d), 'partial_correlations.tsv')
    w(change_vs_intake(d), 'change_vs_intake.tsv')
    d[CORR_COLS].corr().to_csv(os.path.join(outdir, 'correlations.tsv'), sep='\t')
    summary = dict(rfi_consistency=rfi_consistency(d), rfi_groups=info, excreta=excreta_composition(d),
                   pooled_vs_mean_of_weeks={t: float(np.corrcoef(*d[[f'{t}_T', f'{t}_P1', f'{t}_P2']].dropna()
                                                                .pipe(lambda x: (x.iloc[:, 0], x.iloc[:, 1:].mean(1))))[0, 1])
                                            for t in ('cMDM', 'cAMEn', 'cMCP')})
    with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary
