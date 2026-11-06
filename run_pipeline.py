"""Run the complete analysis.

    python run_pipeline.py                     # full analysis (10 x 10-fold CV, 2,000 bootstrap samples)
    python run_pipeline.py --quick             # fast check (1 repetition, 50 bootstrap samples)

Steps
  1  traits and quality control          results/model_table.tsv, results/qc_*.{tsv,json}
  2  statistics                          results/stats/
  3  prediction (14-d, week 1, week 2)   results/prediction/
  4  sensitivity analysis (189 birds)    results/sensitivity/
  5  figures                             results/figures/
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from broilermet import figures, prediction, stats_analysis  # noqa: E402
from broilermet.qc import run_qc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/base_records.tsv')
    ap.add_argument('--out', default='results')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--steps', default='1,2,3,4,5')
    a = ap.parse_args()
    if a.quick:
        a.reps, a.boot = 1, 50
    steps = set(a.steps.split(','))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    base = pd.read_csv(a.data, sep='\t')
    d, weekly, summary = run_qc(base)
    d.to_csv(os.path.join(a.out, 'model_table.tsv'), sep='\t', index=False)
    weekly.to_csv(os.path.join(a.out, 'qc_weekly_missing.tsv'), sep='\t', index=False)
    with open(os.path.join(a.out, 'qc_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('[1] quality control:', {k: v for k, v in summary.items() if k != 'excluded_ids'})

    sdir = os.path.join(a.out, 'stats')
    if '2' in steps:
        stats_analysis.run_all(d, sdir, n_boot=a.boot)
        print('[2] statistics written to', sdir)

    pdir = os.path.join(a.out, 'prediction')
    if '3' in steps:
        os.makedirs(pdir, exist_ok=True)
        for scope in ('T', 'P1', 'P2'):
            summ, reps, oof = prediction.run_scope(d, scope, reps=a.reps, folds=a.folds, jobs=a.jobs)
            summ.to_csv(os.path.join(pdir, f'summary_{scope}.tsv'), sep='\t', index=False)
            reps.to_csv(os.path.join(pdir, f'repetitions_{scope}.tsv'), sep='\t', index=False)
            oof.to_csv(os.path.join(pdir, f'oof_{scope}.tsv.gz'), sep='\t', index=False)
            print(f'[3] prediction {scope}: {len(summ)} scenario-target-model combinations')

    if '4' in steps:
        vdir = os.path.join(a.out, 'sensitivity')
        os.makedirs(vdir, exist_ok=True)
        d_all = run_qc(base, apply=False)[0]
        stats_analysis.repeatability(d_all, a.boot).to_csv(os.path.join(vdir, 'repeatability.tsv'), sep='\t', index=False)
        stats_analysis.rfi_groups(d_all)[0].to_csv(os.path.join(vdir, 'rfi_groups.tsv'), sep='\t', index=False)
        summ, _, _ = prediction.run_scope(d_all, 'T', reps=a.reps, folds=a.folds, jobs=a.jobs)
        summ.to_csv(os.path.join(vdir, 'prediction_summary_T.tsv'), sep='\t', index=False)
        print('[4] sensitivity analysis written to', vdir)

    if '5' in steps:
        summ = pd.read_csv(os.path.join(pdir, 'summary_T.tsv'), sep='\t')
        oof = pd.read_csv(os.path.join(pdir, 'oof_T.tsv.gz'), sep='\t')
        figures.make_all(d, sdir, summ, oof, os.path.join(a.out, 'figures'))
        print('[5] figures written')
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
