# broiler-metabolizability

Data and analysis pipeline for the manuscript

> Li W, Li W, Liu R, Wen J, Guo L, Zhao G. **Repeatability of weekly records and prediction of nutrient metabolizability in broiler chickens.** (submitted to *Poultry Science*)

The pipeline reproduces every number, table, and figure of the manuscript and its Supplementary Tables from one input file of weekly records.

## Contents

| Path | Content |
|---|---|
| `data/base_records.tsv` | Weekly base records of 189 male broilers: BW at 28, 35, and 42 d; feed intake per week; weight, residual moisture, GE, and CP of the 65°C-dried excreta per week; diet composition; carcass traits |
| `src/broilermet/traits.py` | Weekly and 14-d traits (ADFI, ADG, FCR, RFI, DEW, GEE, CPE, MDM, AMEn intake, MCP, cMDM, cAME, cAMEn, cMCP, nitrogen output) |
| `src/broilermet/qc.py` | Quality control (3-SD rule for the 14-d record, then for weekly values) |
| `src/broilermet/stats_analysis.py` | Week comparison, REML repeatability with bootstrap CI, between-week consistency, low- vs. high-RFI groups, partial correlations, correlation matrix |
| `src/broilermet/prediction.py` | Prediction scenarios; elastic net, Gaussian process regression, balance model, and identity baseline; nested cross-validation |
| `src/broilermet/figures.py` | Figures 1 to 5 (vector PDF, 600-dpi TIFF, PNG preview) |
| `run_pipeline.py` | Runs all steps |
| `tests/test_pipeline.py` | Checks of the trait identities, quality control, repeatability, and balance model |

## Requirements

Python ≥ 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The published results were obtained with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9, joblib 1.6, and Matplotlib 3.11.

## Usage

```
python run_pipeline.py                 # full analysis (about 40 min on 2 cores)
python run_pipeline.py --quick         # fast check: 1 cross-validation repetition, 50 bootstrap samples
python run_pipeline.py --steps 1,2     # quality control and statistics only
python -m pytest tests                 # tests
```

Options: `--reps` (cross-validation repetitions, default 10), `--folds` (default 10), `--boot` (bootstrap samples, default 2000), `--jobs` (parallel jobs), `--out` (output folder, default `results`).

## Outputs (`results/`)

| File | Manuscript |
|---|---|
| `qc_summary.json`, `qc_weekly_missing.tsv` | Quality control |
| `stats/week_comparison.tsv` | Table 1 |
| `stats/repeatability.tsv` | Table 2, Figure 3C |
| `stats/rfi_groups.tsv`, `stats/rfi_groups_cAME_same_birds.tsv` | Table 3, Figure 3A and B |
| `stats/rfi_classes.tsv`, `stats/summary.json` | Supplementary Table S4, between-week consistency, excreta composition |
| `stats/partial_correlations.tsv` | Supplementary Table S5 |
| `stats/correlations.tsv` | Supplementary Table S8 |
| `prediction/summary_{T,P1,P2}.tsv` | Table 4, Figure 4, Supplementary Table S7 (T = 28 to 42 d, P1 = 28 to 35 d, P2 = 35 to 42 d) |
| `prediction/oof_T.tsv.gz` | Out-of-fold predictions, Figure 5 |
| `sensitivity/` | Supplementary Table S3 (all 189 birds, no quality control) |
| `figures/` | Figures 1 to 5 |

## Methods in brief

**Traits.** GE and CP of excreta are determined on the 65°C-dried samples, so GE and CP excretion do not require the residual moisture determination; dry excreta weight does. AMEn is corrected to zero nitrogen retention with 8.22 kcal/g of retained nitrogen.

**Balance model.** The coefficient is not predicted directly. The model predicts the excreta concentration that was not analyzed (residual moisture of the 65°C-dried excreta; GE or CP per gram of excreta DM, or per gram of 65°C-dried excreta when DM is not determined) by ridge regression and then calculates the coefficient from the intake–excretion identities, for example cAMEn = cAME − k × cMCP with k = 8.22 × CP_diet / (6.25 × GE_diet). The identity baseline uses the same identities with the training-set mean concentration.

**Validation.** 10-fold cross-validation over birds repeated 10 times (partition seeds 100 to 109). Elastic-net and ridge penalties are tuned by inner 5-fold cross-validation (seed 7). RFI estimation, median imputation, and standardization are refitted in every inner and outer training set. R²_CV = 1 − SSE/SST, with SST around the mean of all observed values of a repetition.

## License

Code: MIT License. Data: CC BY 4.0. Please cite the article when using the data or code.
