import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from broilermet import prediction, stats_analysis  # noqa: E402
from broilermet.qc import run_qc  # noqa: E402


@pytest.fixture(scope='module')
def data():
    base = pd.read_csv(os.path.join(ROOT, 'data', 'base_records.tsv'), sep='\t')
    d, weekly, summary = run_qc(base)
    return base, d, weekly, summary


def test_quality_control(data):
    base, d, weekly, s = data
    assert len(base) == 189
    assert s['n_excluded'] == 7 and s['n_retained'] == 182
    assert s['n_weekly_na'] == 10 and s['n_birds_weekly_na'] == 9
    assert d['cAMEn_T'].notna().sum() == 181


def test_identities(data):
    d = data[1]
    k = 8.22 * d.feed_CP_pct.iloc[0] / 100 / (6.25 * d.feed_GE_kcal_g.iloc[0])
    assert np.allclose(d.cAMEn_T, d.cAME_T - k * d.cMCP_T, equal_nan=True)
    assert abs(k - 0.0634) < 1e-4
    m = d.RFI_P1.notna()
    assert abs(d.RFI_P1[m].mean()) < 1e-9


def test_descriptives(data):
    w = stats_analysis.week_comparison(data[1]).set_index('trait')
    assert round(w.loc['ADFI', 'mean1'], 1) == 159.5
    assert round(w.loc['cMCP', 'diff'], 2) == -3.06


def test_repeatability(data):
    r = stats_analysis.repeatability(data[1], n_boot=0).set_index('trait')
    assert round(r.loc['ADFI', 'r1'], 3) == 0.688
    assert round(r.loc['FCR', 'r1'], 3) == 0.162
    assert round(r.loc['cAMEn', 'r1'], 3) == 0.377


def test_balance_model_is_exact_with_true_concentrations(data):
    """With the unmeasured concentration known, the identities must reproduce the coefficient."""
    d = data[1]
    sub = d[d.cAMEn_T.notna()].reset_index(drop=True)
    aux = sub[[f'{a}_T' for a in prediction.AUX]].to_numpy(float)
    X = np.column_stack([np.zeros(len(sub)), aux])
    for assays in ('DM', 'GE', 'DM+GE', 'DM+CP', 'EX65'):
        m = prediction.BalanceModel(target='cAMEn', assays=assays, n_feat=1, learner='mean').fit(X, sub.cAMEn_T)
        base = sub.DEW_T if assays.startswith('DM') else sub.EX65_T
        truth = {'moist': None, 'ge': (sub.GEE_T / base).to_numpy(), 'cp': (sub.CPE_T / base).to_numpy()}
        for u, mm in m.m_.items():
            mm.m_ = 0.0
            mm.predict = (lambda F, v=truth[u]: v)
        assert np.allclose(m.predict(X), sub.cAMEn_T)


def test_prediction_quick(data):
    d = data[1]
    scen = [s for s in prediction.scenarios('T') if s['name'] == 'DM+GE'][0]
    out = prediction.run_one(d, 'T', scen, 'cAMEn', 'BAL', rep=0)
    assert out['R2'] > 0.97
    base = prediction.run_one(d, 'T', scen, 'cAMEn', 'BAL0', rep=0)
    assert out['RMSE'] < base['RMSE']
