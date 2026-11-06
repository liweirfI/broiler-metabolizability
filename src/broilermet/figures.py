"""Figures 1 to 5 (vector PDF; TIFF at 600 dpi for submission systems that require raster files)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats

rcParams.update({'font.family': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'custom',
                 'mathtext.rm': 'Liberation Serif', 'mathtext.it': 'Liberation Serif:italic', 'font.size': 8,
                 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
                 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600})
WEEK1, WEEK2 = '#2a78d6', '#eb6834'
GRAY, INK, INK2 = '#9a9a96', '#0b0b0b', '#52514e'
BAR_DARK, BAR_LIGHT = '#1c5cab', '#86b6ef'
# diverging scale used for all heat maps (low = blue, middle = white, high = salmon)
HEAT_LOW, HEAT_MID, HEAT_HIGH = '#4A57C7', '#FFFFFF', '#FC9E8E'
HEAT = LinearSegmentedColormap.from_list('heat', [HEAT_LOW, HEAT_MID, HEAT_HIGH])


def _save(fig, outdir, name, tif=True):
    fig.savefig(os.path.join(outdir, f'{name}.pdf'), bbox_inches='tight')
    if tif:
        fig.savefig(os.path.join(outdir, f'{name}.tif'), dpi=600, bbox_inches='tight',
                    pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(os.path.join(outdir, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _panel(ax, label, x=-0.18, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')


def _fmtp(p):
    return '$P$ < 0.001' if p < 0.001 else f'$P$ = {p:.3f}'


def fig_week_change(d, outdir):
    traits = [('ADFI', 'ADFI, g/d', '{:+.1f}'), ('ADG', 'ADG, g/d', '{:+.1f}'), ('FCR', 'FCR, g:g', '{:+.3f}'),
              ('cMDM', 'cMDM, %', '{:+.2f}'), ('cAMEn', 'cAMEn, %', '{:+.2f}'), ('cMCP', 'cMCP, %', '{:+.2f}')]
    fig, axs = plt.subplots(2, 3, figsize=(6.9, 4.2))
    for ax, (t, lab, fmt), L in zip(axs.flat, traits, 'ABCDEF'):
        a, b = d[t + '_P1'], d[t + '_P2']
        m = a.notna() & b.notna()
        a, b = a[m].to_numpy(), b[m].to_numpy()
        for i in range(len(a)):
            ax.plot([0, 1], [a[i], b[i]], color='#c9c8c3', lw=0.35, zorder=1)
        ax.plot([0, 1], [a.mean(), b.mean()], color=INK, lw=1.6, zorder=3)
        ax.scatter([0, 1], [a.mean(), b.mean()], s=28, color=[WEEK1, WEEK2], zorder=4, edgecolor='white', linewidth=0.8)
        p = stats.ttest_rel(b, a).pvalue
        ax.text(0.97, 0.04, 'Δ = ' + fmt.format((b - a).mean()).replace('-', '−') + '\n' + _fmtp(p),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['28 to 35 d', '35 to 42 d'])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(lab)
        _panel(ax, L, x=-0.3)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    _save(fig, outdir, 'Figure1')


def fig_week_scatter(d, outdir):
    e1 = d.RFI_P1 <= d.RFI_P1.median()
    e2 = d.RFI_P2 <= d.RFI_P2.median()
    ok = d.RFI_P1.notna() & d.RFI_P2.notna()
    both, neither, changed = ok & e1 & e2, ok & ~e1 & ~e2, ok & (e1 != e2)
    tr = [('ADFI', 'g/d'), ('FCR', 'g:g'), ('RFI', 'g/d'), ('cMDM', '%'), ('cAMEn', '%'), ('cMCP', '%')]
    fig, axs = plt.subplots(2, 3, figsize=(6.9, 4.5))
    for ax, (t, u), L in zip(axs.flat, tr, 'ABCDEF'):
        a, b = d[t + '_P1'], d[t + '_P2']
        m = a.notna() & b.notna()
        if t == 'RFI':
            ax.scatter(a[changed], b[changed], s=11, facecolor='white', edgecolor=GRAY, linewidth=0.7, zorder=2)
            ax.scatter(a[both], b[both], s=10, color=WEEK1, edgecolor='white', linewidth=0.3, zorder=3)
            ax.scatter(a[neither], b[neither], s=10, color=WEEK2, edgecolor='white', linewidth=0.3, zorder=3)
            ax.axvline(a.median(), color=GRAY, lw=0.6, ls='--', zorder=1)
            ax.axhline(b.median(), color=GRAY, lw=0.6, ls='--', zorder=1)
        else:
            ax.scatter(a[m], b[m], s=7, color=WEEK1, alpha=0.75, edgecolor='none', zorder=2)
        sl, ic, r, _, _ = stats.linregress(a[m], b[m])
        xx = np.linspace(a[m].min(), a[m].max(), 10)
        ax.plot(xx, ic + sl * xx, color=INK, lw=1.0, zorder=4)
        ax.text(0.97, 0.04, f'$r$ = {r:.2f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.85), zorder=5)
        ax.set_xlabel(f'{t} (28 to 35 d), {u}')
        ax.set_ylabel(f'{t} (35 to 42 d), {u}')
        _panel(ax, L, x=-0.3)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    _save(fig, outdir, 'Figure2')


def fig_groups_repeatability(groups, rep, outdir):
    fig = plt.figure(figsize=(6.9, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.45)
    names = {'cMDM': 'cMDM', 'cAMEn': 'cAMEn', 'cMCP': 'cMCP', 'AbP': 'Abdominal\nfat',
             'BreastP': 'Breast\nmuscle', 'LegMuscP': 'Leg\nmuscle'}
    for k, traits in enumerate([['cMDM', 'cAMEn', 'cMCP'], ['AbP', 'BreastP', 'LegMuscP']]):
        ax = fig.add_subplot(gs[0, k])
        x = np.arange(3)
        w = 0.36
        ymax = ymin = 0
        for j, (p, c) in enumerate([('P1', WEEK1), ('P2', WEEK2)]):
            sub = groups[groups.period == p].set_index('trait').loc[traits]
            xs = x + (j - 0.5) * w
            ax.bar(xs, sub['diff'], width=w - 0.04, color=c, label='28 to 35 d' if p == 'P1' else '35 to 42 d', zorder=2)
            ax.errorbar(xs, sub['diff'], yerr=sub['sed'], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
            for xi, (dv, se, q) in zip(xs, sub[['diff', 'sed', 'q']].to_numpy()):
                ymax, ymin = max(ymax, dv + se), min(ymin, dv - se)
                if q < 0.05:
                    off = 0.04 * (1 if k == 1 else 3)
                    ax.text(xi, dv + se + off if dv >= 0 else dv - se - off, '*', ha='center',
                            va='bottom' if dv >= 0 else 'top', fontsize=9)
        ax.axhline(0, color='#333333', lw=0.6)
        ax.set_ylim(ymin * 1.35 - 0.05, ymax * 1.25 + 0.1)
        ax.set_xticks(x)
        ax.set_xticklabels([names[t] for t in traits])
        ax.set_ylabel('Low RFI minus high RFI,\npercentage points')
        if k == 0:
            ax.legend(frameon=False, loc='upper left', fontsize=6.5)
        _panel(ax, 'AB'[k], x=-0.32)
    ax = fig.add_subplot(gs[0, 2])
    R = rep.sort_values('r1', ascending=False).set_index('trait')
    y = np.arange(len(R))[::-1]
    h2 = ax.barh(y, R.r2, height=0.7, color=BAR_LIGHT, label='Mean of 2 records', zorder=2)
    h1 = ax.barh(y, R.r1, height=0.7, color=BAR_DARK, label='7-d record', zorder=3)
    ax.errorbar(R.r1, y, xerr=[R.r1 - R.lo, R.hi - R.r1], fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([{'AMEn': 'AMEn intake'}.get(t, t) for t in R.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Repeatability or reliability')
    ax.legend(handles=[h1, h2], frameon=False, loc='lower right', ncol=1, fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
    _panel(ax, 'C', x=-0.42)
    _save(fig, outdir, 'Figure3')


HEAT_ROWS = [('Growth', 'Growth'), ('Growth+FI', '+ Feed intake, efficiency'), ('Growth+FI+Carcass', '+ Carcass'),
             ('EX65', '+ Dried excreta weight'), ('DM', '+ DM'), ('GE', '+ GE'), ('DM+CP', '+ DM, CP'),
             ('DM+GE', '+ DM, GE'), ('GE+CP', '+ GE, CP')]
HEAT_MODELS = [('EN', 'Elastic net'), ('GPR', 'Gaussian process'), ('BAL', 'Balance model')]


def fig_heatmap(summary, outdir):
    targets = ['cMDM', 'cAMEn', 'cMCP']
    S = summary[summary.scope == 'T']
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.5), sharey=True)
    for ax, (alg, title), L in zip(axs, HEAT_MODELS, 'ABC'):
        for i, (scen, _) in enumerate(HEAT_ROWS):
            for j, t in enumerate(targets):
                x = S[(S.alg == alg) & (S.scenario == scen) & (S.target == t)]
                if x.empty:                      # target measured directly, or model not applicable
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#f0f0ee', edgecolor='white',
                                           linewidth=0.8))
                    continue
                v = float(x.R2.iloc[0])
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=HEAT(np.clip(v, 0, 1)),
                                       edgecolor='white', linewidth=0.8))
                ax.text(j, i, f'{v:.3f}'.replace('-', '−'), ha='center', va='center', fontsize=6.6, color=INK)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(len(HEAT_ROWS) - 0.5, -0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(targets)
        ax.set_yticks(range(len(HEAT_ROWS)))
        ax.set_yticklabels([lab for _, lab in HEAT_ROWS])
        ax.set_title(title, pad=4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(2.5, color=INK2, lw=0.8)
        ax.text(-0.05 if L != 'A' else -0.9, 1.0, L, transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='bottom', ha='left')
    fig.subplots_adjust(left=0.2, right=0.98, wspace=0.08, bottom=0.08, top=0.9)
    _save(fig, outdir, 'Figure4')


def fig_observed_predicted(d, oof, outdir):
    cases = [('cMDM', 'EX65', 'Dried excreta weight'), ('cAMEn', 'DM', 'DM only'),
             ('cAMEn', 'GE', 'GE only'), ('cMCP', 'DM+GE', 'DM and GE')]
    fig, axs = plt.subplots(1, 4, figsize=(6.9, 1.95))
    for ax, (t, s, ttl), L in zip(axs, cases, 'ABCD'):
        o = oof[(oof.scope == 'T') & (oof.scenario == s) & (oof.target == t) & (oof.alg == 'BAL') & (oof.rep == 0)]
        o = o.merge(d[['id', f'{t}_T']], on='id')
        yv, pr = o[f'{t}_T'].to_numpy(), o.pred.to_numpy()
        r2 = 1 - ((yv - pr) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        lo, hi = min(yv.min(), pr.min()), max(yv.max(), pr.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls='--', color=GRAY, lw=0.7, zorder=1)
        ax.scatter(pr, yv, s=6, color=WEEK1, alpha=0.8, edgecolor='none', zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal')
        ax.set_xlabel(f'Predicted {t}, %')
        ax.set_ylabel(f'Observed {t}, %')
        ax.set_title(ttl, fontsize=7)
        ax.text(0.04, 0.96, f'$R^2$ = {r2:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=7)
        _panel(ax, L, x=-0.42, y=1.08)
    fig.tight_layout(w_pad=0.6)
    _save(fig, outdir, 'Figure5')


def make_all(d, stats_dir, summary, oof, outdir):
    os.makedirs(outdir, exist_ok=True)
    groups = pd.read_csv(os.path.join(stats_dir, 'rfi_groups.tsv'), sep='\t')
    rep = pd.read_csv(os.path.join(stats_dir, 'repeatability.tsv'), sep='\t')
    fig_week_change(d, outdir)
    fig_week_scatter(d, outdir)
    fig_groups_repeatability(groups, rep, outdir)
    fig_heatmap(summary, outdir)
    fig_observed_predicted(d, oof, outdir)
