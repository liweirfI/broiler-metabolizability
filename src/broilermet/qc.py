"""Phenotypic quality control.

Step 1: birds whose 14-d RFI, FCR, cMDM, cAMEn, or cMCP deviate by more than 3 SD from the mean are excluded.
Step 2: in the retained birds, weekly values of the same traits deviating by more than 3 SD from the weekly
        mean are set to missing; all other records of these birds are kept.
With apply=False the input is returned unchanged except for derived traits (sensitivity analysis).
"""
import numpy as np
import pandas as pd

from .traits import derive

QC_TRAITS = ['RFI', 'FCR', 'cMDM', 'cAMEn', 'cMCP']
LIMIT_SD = 3.0


def run_qc(base: pd.DataFrame, apply: bool = True):
    d = derive(base)
    if not apply:
        return d, pd.DataFrame(), dict(n_input=len(base), n_excluded=0, n_retained=len(d), n_weekly_na=0)
    cols14 = [f'{t}_T' for t in QC_TRAITS]
    z = (d[cols14] - d[cols14].mean()) / d[cols14].std()
    excl = (z.abs() > LIMIT_SD).any(axis=1)
    excluded = d.loc[excl, ['id'] + cols14].reset_index(drop=True)
    keep = base.loc[~excl.values].reset_index(drop=True)
    d = derive(keep)
    rows = []
    for p, week in (('P1', 1), ('P2', 2)):
        for t in QC_TRAITS:
            c = f'{t}_{p}'
            m, s = d[c].mean(), d[c].std()
            zz = (d[c] - m) / s
            for i in np.where(zz.abs() > LIMIT_SD)[0]:
                rows.append(dict(id=d.id[i], week=week, trait=t, value=d[c][i], z=zz[i]))
    weekly = pd.DataFrame(rows, columns=['id', 'week', 'trait', 'value', 'z'])
    for _, q in weekly.iterrows():
        d.loc[d.id == q.id, f"{q.trait}_{'P1' if q.week == 1 else 'P2'}"] = np.nan
    summary = dict(n_input=len(base), n_excluded=int(excl.sum()), excluded_ids=excluded.id.tolist(),
                   n_retained=len(d), n_weekly_na=len(weekly), n_birds_weekly_na=int(weekly.id.nunique()))
    return d, weekly, summary
