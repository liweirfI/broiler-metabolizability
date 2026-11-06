"""Weekly and 14-d traits derived from the base weekly records.

Period codes: P1 = 28 to 35 d (week 1), P2 = 35 to 42 d (week 2), T = 28 to 42 d (14-d record).
GE and CP of excreta are expressed per gram of 65 degC-dried excreta, so GE and CP excretion do not
depend on the residual moisture determination; dry excreta weight (DEW) does.
"""
import numpy as np
import pandas as pd

N_KCAL_PER_G = 8.22      # kcal per g of retained nitrogen (Hill and Anderson, 1958)
CP_TO_N = 6.25


def derive(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    fdm = 1 - d.feed_moist_pct / 100
    fge = d.feed_GE_kcal_g
    fcp = d.feed_CP_pct / 100
    for p, (b0, b1) in {'P1': ('BW28', 'BW35'), 'P2': ('BW35', 'BW42')}.items():
        d[f'ADFI_{p}'] = d[f'feed_total_{p}'] / 7
        d[f'ADG_{p}'] = (d[b1] - d[b0]) / 7
        d[f'FCR_{p}'] = d[f'ADFI_{p}'] / d[f'ADG_{p}']
        d[f'MBW_{p}'] = (((d[b0] + d[b1]) / 2) / 1000) ** 0.75
        d[f'EX65_{p}'] = d[f'excr_fw_{p}'] / 7
        d[f'DEW_{p}'] = d[f'EX65_{p}'] * (1 - d[f'excr_moist_pct_{p}'] / 100)
        d[f'GEE_{p}'] = d[f'EX65_{p}'] * d[f'excr_GE_kcal_g_{p}']
        d[f'CPE_{p}'] = d[f'EX65_{p}'] * d[f'excr_CP_pct_{p}'] / 100
        d[f'DMI_{p}'] = d[f'ADFI_{p}'] * fdm
        d[f'GEI_{p}'] = d[f'ADFI_{p}'] * fge
        d[f'CPI_{p}'] = d[f'ADFI_{p}'] * fcp
        d[f'MDM_{p}'] = d[f'DMI_{p}'] - d[f'DEW_{p}']
        n_ret = (d[f'CPI_{p}'] - d[f'CPE_{p}']) / CP_TO_N
        d[f'AMEn_{p}'] = d[f'GEI_{p}'] - d[f'GEE_{p}'] - N_KCAL_PER_G * n_ret
        d[f'MCP_{p}'] = d[f'CPI_{p}'] - d[f'CPE_{p}']
        d[f'cMDM_{p}'] = 100 * d[f'MDM_{p}'] / d[f'DMI_{p}']
        d[f'cAMEn_{p}'] = 100 * d[f'AMEn_{p}'] / d[f'GEI_{p}']
        d[f'cAME_{p}'] = 100 * (d[f'GEI_{p}'] - d[f'GEE_{p}']) / d[f'GEI_{p}']
        d[f'cMCP_{p}'] = 100 * d[f'MCP_{p}'] / d[f'CPI_{p}']
        d[f'DMout_per_kg_{p}'] = d[f'DEW_{p}'] / d[f'ADG_{p}'] * 1000
        d[f'Nout_per_kg_{p}'] = d[f'CPE_{p}'] / CP_TO_N / d[f'ADG_{p}'] * 1000
    # 14-d record: daily means over both weeks (intake and excretion summed over 14 d)
    for v in ['ADFI', 'EX65', 'DEW', 'GEE', 'CPE', 'DMI', 'GEI', 'CPI', 'MDM', 'AMEn', 'MCP']:
        d[f'{v}_T'] = (d[f'{v}_P1'] + d[f'{v}_P2']) / 2
    d['ADG_T'] = (d.BW42 - d.BW28) / 14
    d['FCR_T'] = d.ADFI_T / d.ADG_T
    d['MBW_T'] = (((d.BW28 + d.BW42) / 2) / 1000) ** 0.75
    d['cMDM_T'] = 100 * d.MDM_T / d.DMI_T
    d['cAMEn_T'] = 100 * d.AMEn_T / d.GEI_T
    d['cAME_T'] = 100 * (d.GEI_T - d.GEE_T) / d.GEI_T
    d['cMCP_T'] = 100 * d.MCP_T / d.CPI_T
    d['DMout_per_kg_T'] = d.DEW_T / d.ADG_T * 1000
    d['Nout_per_kg_T'] = d.CPE_T / CP_TO_N / d.ADG_T * 1000
    # RFI estimated on the full sample (descriptive analyses); prediction models re-estimate RFI in training data
    for p in ['P1', 'P2', 'T']:
        d[f'RFI_{p}'] = rfi(d[f'ADFI_{p}'], d[f'MBW_{p}'], d[f'ADG_{p}'])
    return d


def rfi(adfi, mbw, adg):
    """Residual of ADFI = b0 + b1 * MBW + b2 * ADG (ordinary least squares)."""
    adfi, mbw, adg = (pd.Series(np.asarray(v, float)) for v in (adfi, mbw, adg))
    ok = adfi.notna() & mbw.notna() & adg.notna()
    Z = np.column_stack([np.ones(ok.sum()), mbw[ok], adg[ok]])
    b = np.linalg.lstsq(Z, adfi[ok], rcond=None)[0]
    out = np.full(len(adfi), np.nan)
    out[ok.values] = adfi[ok] - Z @ b
    return out
