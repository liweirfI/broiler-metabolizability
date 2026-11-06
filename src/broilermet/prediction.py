"""Prediction of metabolizability coefficients from routinely recorded traits and reduced excreta analyses.

Models
  EN        elastic net
  GPR       Gaussian process regression (constant x RBF + white-noise kernel)
  BAL       balance model: ridge regression for the unmeasured excreta concentration(s), then the
            intake-excretion identities that define the coefficient
  BAL0      identity baseline: as BAL, but the unmeasured concentration is the training-set mean

Validation
  Outer loop: 10-fold cross-validation over birds, repeated 10 times (partition seed 100 + repetition).
  Inner loop: 5-fold cross-validation within each outer training set (seed 7) for the elastic-net penalty and
  mixing parameter and for the ridge penalty of BAL. RFI estimation, median imputation, and standardization are
  refitted in every inner and outer training set. GPR kernel parameters are estimated by maximizing the marginal
  likelihood in the outer training set.
  R2_CV = 1 - SSE / SST with SST around the mean of all observed values of a repetition.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, enet_path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

K_N = 8.22 / 6.25            # kcal per g of retained CP
INNER_SEED = 7
OUTER_SEED0 = 100
L1_RATIOS = (0.1, 0.5, 0.9, 1.0)
N_ALPHAS = 50
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
AUX = ['DMI', 'GEI', 'CPI', 'DEW', 'EX65', 'GEE', 'CPE']
CARCASS = ['AbF', 'AbP', 'BreastP', 'LegMuscP']


# ----------------------------------------------------------------------------------------------- scenarios
def scenarios(scope):
    S = scope
    if S == 'T':
        growth = ['BW28', 'BW35', 'BW42', 'ADG_P1', 'ADG_P2']
        intake = ['ADFI_P1', 'ADFI_P2', 'FCR_T']
    else:
        growth = ['BW28', 'BW35', 'ADG_P1'] if S == 'P1' else ['BW35', 'BW42', 'ADG_P2']
        intake = [f'ADFI_{S}', f'FCR_{S}']
    base = growth + intake
    sc = lambda name, x, targets, assays, rfi=True: dict(name=name, x=x, targets=targets, assays=assays, rfi=rfi)
    return [
        sc('Growth', growth, ['cMDM', 'cAMEn', 'cMCP'], 'none', rfi=False),
        sc('Growth+FI', base, ['cMDM', 'cAMEn', 'cMCP'], 'none'),
        sc('Growth+FI+Carcass', base + CARCASS, ['cMDM', 'cAMEn', 'cMCP'], 'none'),
        sc('EX65', base + [f'EX65_{S}'], ['cMDM', 'cAMEn', 'cMCP'], 'EX65'),
        sc('DM', base + [f'DEW_{S}', f'cMDM_{S}'], ['cAMEn', 'cMCP'], 'DM'),
        sc('GE', base + [f'EX65_{S}', f'cAME_{S}'], ['cMDM', 'cAMEn', 'cMCP'], 'GE'),
        sc('DM+CP', base + [f'DEW_{S}', f'cMDM_{S}', f'cMCP_{S}'], ['cAMEn'], 'DM+CP'),
        sc('DM+GE', base + [f'DEW_{S}', f'cMDM_{S}', f'cAME_{S}'], ['cAMEn', 'cMCP'], 'DM+GE'),
        sc('GE+CP', base + [f'EX65_{S}', f'cAMEn_{S}', f'cMCP_{S}'], ['cMDM'], 'GE+CP'),
    ]


# ----------------------------------------------------------------------------------------------- building blocks
class RFIStep(BaseEstimator, TransformerMixin):
    """The last 3 columns are ADFI, metabolic BW, and ADG; they are replaced by RFI estimated on the fitted rows."""

    def __init__(self, use=True):
        self.use = use

    def fit(self, X, y=None):
        if self.use:
            A = X[:, -3:]
            ok = ~np.isnan(A).any(1)
            Z = np.column_stack([np.ones(ok.sum()), A[ok, 1], A[ok, 2]])
            self.b_ = np.linalg.lstsq(Z, A[ok, 0], rcond=None)[0]
        return self

    def transform(self, X):
        if not self.use:
            return X
        A = X[:, -3:]
        r = A[:, 0] - (self.b_[0] + self.b_[1] * A[:, 1] + self.b_[2] * A[:, 2])
        return np.column_stack([X[:, :-3], r])


def preprocessor(rfi):
    return Pipeline([('rfi', RFIStep(rfi)), ('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])


class NestedLinear(BaseEstimator, RegressorMixin):
    """Penalized linear regression whose penalty is tuned by inner K-fold cross-validation, with
    preprocessing (RFI, imputation, standardization) and target centring refitted in each inner training fold.
    kind = 'enet' (elastic net, penalty and mixing parameter tuned) or 'ridge' (penalty tuned)."""

    def __init__(self, kind='enet', rfi=True, inner_folds=5, seed=INNER_SEED):
        self.kind, self.rfi, self.inner_folds, self.seed = kind, rfi, inner_folds, seed

    # coefficient paths on standardized X and centred y ----------------------------------------
    def _grid(self, Xs, yc):
        n = len(yc)
        if self.kind == 'ridge':
            return [(None, RIDGE_ALPHAS)]
        grid = []
        for l1 in L1_RATIOS:
            amax = np.abs(Xs.T @ yc).max() / (n * l1)
            grid.append((l1, np.logspace(np.log10(amax), np.log10(amax * 1e-3), N_ALPHAS)))
        return grid

    def _path(self, Xs, yc, l1, alphas):
        if self.kind == 'ridge':
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            Uy = U.T @ yc
            return np.stack([Vt.T @ (s / (s ** 2 + a) * Uy) for a in alphas], axis=1)
        return enet_path(Xs, yc, l1_ratio=l1, alphas=alphas, max_iter=20000)[1]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.pre_ = preprocessor(self.rfi).fit(X)
        Xs = self.pre_.transform(X)
        ym, ys = y.mean(), y.std()
        grid = self._grid(Xs, (y - ym) / ys)
        mse = [np.zeros(len(a)) for _, a in grid]
        for tr, va in KFold(self.inner_folds, shuffle=True, random_state=self.seed).split(X):
            pre = preprocessor(self.rfi).fit(X[tr])
            Xt, Xv = pre.transform(X[tr]), pre.transform(X[va])
            m, s = y[tr].mean(), y[tr].std()
            for g, (l1, alphas) in enumerate(grid):
                B = self._path(Xt, (y[tr] - m) / s, l1, alphas)
                pred = m + s * (Xv @ B)
                mse[g] += ((pred - y[va][:, None]) ** 2).sum(0)
        g, j = min(((g, int(np.argmin(v))) for g, v in enumerate(mse)), key=lambda t: mse[t[0]][t[1]])
        self.l1_ratio_, self.alpha_ = grid[g][0], grid[g][1][j]
        self.coef_ = self._path(Xs, (y - ym) / ys, self.l1_ratio_, np.array([self.alpha_]))[:, 0]
        self.ym_, self.ys_ = ym, ys
        return self

    def predict(self, X):
        return self.ym_ + self.ys_ * (self.pre_.transform(np.asarray(X, float)) @ self.coef_)


class MeanOnly(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


def gpr(rfi):
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(3.0, (0.05, 100)) + WhiteKernel(0.5, (1e-4, 2))
    pipe = Pipeline([('pre', preprocessor(rfi)),
                     ('m', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=1))])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


class BalanceModel(BaseEstimator, RegressorMixin):
    """Predicts the excreta concentration(s) that were not measured and computes the coefficient exactly.

    X = [predictors | DMI, GEI, CPI, DEW, EX65, GEE, CPE]; only quantities measured in the scenario are used
    for prediction. Training birds are fully analysed, so their concentrations are known.
    Concentrations: residual moisture fraction of 65 degC-dried excreta (cMDM), GE or CP per g of excreta
    DM (when DM is determined) or per g of 65 degC-dried excreta (otherwise)."""

    def __init__(self, target='cAMEn', assays='DM', n_feat=0, rfi=True, learner='ridge'):
        self.target, self.assays, self.n_feat, self.rfi, self.learner = target, assays, n_feat, rfi, learner

    def _split(self, X):
        X = np.asarray(X, float)
        return X[:, :self.n_feat], {k: X[:, self.n_feat + i] for i, k in enumerate(AUX)}

    def _base(self, A):
        return A['DEW'] if self.assays.startswith('DM') else A['EX65']

    def _unknowns(self):
        t, a = self.target, self.assays
        if t == 'cMDM':
            return ['moist']
        if t == 'cMCP':
            return ['cp']
        if a == 'DM+CP':
            return ['ge']
        if a in ('DM+GE', 'GE'):
            return ['cp']
        return ['ge', 'cp']                      # EX65 or DM only

    def _learner(self):
        return MeanOnly() if self.learner == 'mean' else NestedLinear('ridge', rfi=self.rfi)

    def fit(self, X, y):
        F, A = self._split(X)
        base = self._base(A)
        z = {'moist': 1 - A['DEW'] / A['EX65'], 'ge': A['GEE'] / base, 'cp': A['CPE'] / base}
        self.m_ = {u: self._learner().fit(F, z[u]) for u in self._unknowns()}
        return self

    def predict(self, X):
        F, A = self._split(X)
        base = self._base(A)
        z = {u: m.predict(F) for u, m in self.m_.items()}
        if self.target == 'cMDM':
            dew = A['EX65'] * (1 - z['moist'])
            return 100 * (A['DMI'] - dew) / A['DMI']
        cpe = z['cp'] * base if 'cp' in z else A['CPE']
        if self.target == 'cMCP':
            return 100 * (A['CPI'] - cpe) / A['CPI']
        gee = z['ge'] * base if 'ge' in z else A['GEE']
        return 100 * (A['GEI'] - gee - K_N * (A['CPI'] - cpe)) / A['GEI']


MODELS = ('EN', 'GPR', 'BAL', 'BAL0')


def make_model(alg, scen, target, n_feat):
    rfi = scen['rfi']
    if alg == 'EN':
        return NestedLinear('enet', rfi=rfi)
    if alg == 'GPR':
        return gpr(rfi)
    return BalanceModel(target=target, assays=scen['assays'], n_feat=n_feat, rfi=rfi,
                        learner='mean' if alg == 'BAL0' else 'ridge')


# ----------------------------------------------------------------------------------------------- evaluation
def metrics(y, pred):
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    k = int(round(0.2 * len(y)))
    top = len(set(np.argsort(-y)[:k]) & set(np.argsort(-pred)[:k])) / k
    slope = np.polyfit(pred, y, 1)[0] if np.std(pred) > 0 else np.nan
    return dict(R2=1 - sse / sst, r=np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 else np.nan,
                RMSE=np.sqrt(sse / len(y)), MAE=np.mean(np.abs(y - pred)), bias=np.mean(pred - y),
                slope=slope, top20=100 * top)


def run_one(D, scope, scen, target, alg, rep, folds=10):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    if alg in ('BAL', 'BAL0') and scen['assays'] == 'none':
        return None
    rfi_src = [f'ADFI_{scope}', f'MBW_{scope}', f'ADG_{scope}'] if scen['rfi'] else []
    feat = list(scen['x']) + rfi_src
    cols = feat + ([f'{a}_{scope}' for a in AUX] if alg in ('BAL', 'BAL0') else [])
    sub = D[D[f'{target}_{scope}'].notna()].reset_index(drop=True)
    X = sub[cols].to_numpy(float)
    y = sub[f'{target}_{scope}'].to_numpy(float)
    pred = np.full(len(y), np.nan)
    t0 = time.time()
    for tr, te in KFold(folds, shuffle=True, random_state=OUTER_SEED0 + rep).split(X):
        m = make_model(alg, scen, target, len(feat)).fit(X[tr], y[tr])
        pred[te] = np.ravel(m.predict(X[te]))
    return dict(scope=scope, scenario=scen['name'], assays=scen['assays'], target=target, alg=alg, rep=rep,
                n=len(y), sec=time.time() - t0, **metrics(y, pred), ids=sub.id.tolist(), oof=pred.tolist())


def run_scope(D, scope, models=MODELS, reps=10, folds=10, jobs=1, only=None):
    from joblib import Parallel, delayed
    tasks = [(s, t, a, r) for s in scenarios(scope) if (only is None or s['name'] in only)
             for t in s['targets'] for a in models for r in range(reps)]
    out = Parallel(n_jobs=jobs)(delayed(run_one)(D, scope, s, t, a, r, folds) for s, t, a, r in tasks)
    out = [o for o in out if o is not None]
    reps_df = pd.DataFrame([{k: v for k, v in o.items() if k not in ('ids', 'oof')} for o in out])
    oof = pd.DataFrame([(o['scope'], o['scenario'], o['target'], o['alg'], o['rep'], i, v)
                        for o in out if o['rep'] == 0 for i, v in zip(o['ids'], o['oof'])],
                       columns=['scope', 'scenario', 'target', 'alg', 'rep', 'id', 'pred'])
    summ = (reps_df.groupby(['scope', 'scenario', 'assays', 'target', 'alg'], sort=False)
            .agg(n=('n', 'first'), R2=('R2', 'mean'), R2_sd=('R2', 'std'), r=('r', 'mean'), RMSE=('RMSE', 'mean'),
                 MAE=('MAE', 'mean'), bias=('bias', 'mean'), slope=('slope', 'mean'), top20=('top20', 'mean'),
                 sec=('sec', 'sum')).reset_index())
    return summ, reps_df, oof
