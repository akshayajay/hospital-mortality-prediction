# In-Hospital Mortality Prediction

[![Tests](https://github.com/akshayajay/hospital-mortality-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/hospital-mortality-prediction/actions/workflows/tests.yml)

A retrospective machine learning study of in-hospital mortality in **9,105 critically ill patients** from SUPPORT2. The [analysis notebook](mortality_prediction.ipynb) compares five classifiers using 31 input features, XGBoost-based iterative imputation, and stratified cross-validation.

The saved notebook reports **0.9160 test ROC-AUC for Gradient Boosting** and **0.916 ± 0.007 ROC-AUC across 10 repeated stratified splits**. These are historical notebook outputs, not results from a fresh end-to-end run. The retained features include information collected after admission; these scores do not establish performance for early clinical prediction.

## Dataset and target

- **Source:** [UCI SUPPORT2, dataset 880](https://archive.ics.uci.edu/dataset/880/support2), fetched in the notebook with `fetch_ucirepo(id=880)`.
- **Patients:** 9,105; the saved target distribution is 25.92% in-hospital deaths.
- **Target:** `hospdead` (`1` = death in hospital, `0` = no death in hospital). `death` is a different outcome and is excluded from the predictors.
- **Column counts:** UCI lists 42 features. The notebook loads `support2.data.original`, which has 48 columns including identifiers, targets, and other variables, then removes 17 columns to retain **31 input features: 25 numeric and 6 categorical**, before one-hot encoding.
- **Missingness:** 86.82% of rows have at least one missing value among the retained features. The most incomplete are `urine` (53.40%), `glucose` (49.42%), and `bun` (47.80%).

### Feature exclusions

The notebook removes the following 17 columns:

```text
id, death, hospdead, surv2m, surv6m, prg2m, prg6m, d.time, sfdm2,
slos, charges, totcst, totmcst, adlp, adls, dnr, adlsc
```

This list combines identifiers, the target, other outcomes, survival estimates, cost fields, and additional excluded variables. It is more precise to describe it as **17 leakage-related or non-feature exclusions** than as 17 proven leakage columns. The survival fields include model and physician estimates, not simply observed future outcomes; see the [SUPPORT2 variable descriptions](https://hbiostat.org/data/repo/supportdesc).

The retained features are:

| Type | Features |
| --- | --- |
| Numeric (25) | `age`, `num.co`, `edu`, `scoma`, `avtisst`, `sps`, `aps`, `hday`, `diabetes`, `dementia`, `dnrday`, `meanbp`, `wblc`, `hrt`, `resp`, `temp`, `pafi`, `alb`, `bili`, `crea`, `sod`, `ph`, `glucose`, `bun`, `urine` |
| Categorical (6) | `sex`, `dzgroup`, `dzclass`, `income`, `race`, `ca` |

## Modeling workflow

1. **Stratified split:** 60% training (5,463 patients), 20% validation (1,821), and 20% test (1,821), with `random_state=42`.
2. **Numeric preprocessing:** `IterativeImputer` with an `XGBRegressor` estimator, median initialization, and up to 10 imputation iterations, followed by `StandardScaler`. The regressor uses 200 trees, maximum depth 3, and learning rate 0.1.
3. **Categorical preprocessing:** most-frequent imputation and `OneHotEncoder(handle_unknown='ignore')`.
4. **Hyperparameter tuning:** `GridSearchCV` optimizes ROC-AUC using shuffled 5-fold `StratifiedKFold` on the training set. Preprocessing is inside each model pipeline, so it is fitted within the training folds.
5. **Class imbalance handling:** balanced sample weights are passed to all five classifiers; applicable model grids also include class-weight settings.
6. **Evaluation:** validation/test ROC-AUC, test accuracy, precision, recall, F1, ROC curve, and confusion matrix. Classification metrics use a 0.5 probability threshold.

The five classifiers are **Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and RBF Support Vector Machine**. XGBoost supplies the numeric imputation estimator; it is not a sixth classifier or the winning mortality classifier.

## Saved results

These values are transcribed from the notebook's saved tuning and full-metrics outputs.

| Model | Validation ROC-AUC | Test ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9124 | 0.9092 | 0.8413 | 0.6488 | 0.8453 | 0.7341 |
| Decision Tree | 0.8893 | 0.8897 | 0.8331 | 0.6511 | 0.7669 | 0.7043 |
| Random Forest | 0.9121 | 0.9139 | 0.8364 | 0.6394 | 0.8453 | 0.7281 |
| **Gradient Boosting** | **0.9181** | **0.9160** | 0.8440 | 0.6626 | 0.8114 | 0.7295 |
| RBF SVM | 0.9128 | 0.9105 | 0.8710 | 0.7926 | 0.6801 | 0.7320 |

Gradient Boosting has the highest saved validation and test ROC-AUC, with learning rate 0.1, 100 estimators, and maximum depth 3. The majority-class baseline has 0.741 accuracy and zero positive-class F1.

### Repeated-split analysis

The notebook reports **0.916 ± 0.007**, where `±` is the standard deviation across 10 stratified 80/20 train/test splits (`np.std`, `ddof=0`), not a confidence interval. Each split refits preprocessing and the selected classifier with fixed hyperparameters. The repeated-split cell does not pass the sample weights used during the original tuning run.

This is a sensitivity check on the same dataset with overlapping splits, not nested cross-validation or external validation. It uses 80% training data rather than the original 60%, and does not repeat hyperparameter selection within each split. Stability across these splits does not resolve feature-timing or leakage concerns.

## Model interpretation

The saved notebook includes tree feature importances, a SHAP summary/beeswarm plot, a global mean absolute SHAP table, an age dependence plot, a local force plot, and validation-set permutation importance with 10 repeats and ROC-AUC scoring.

**`avtisst` ranks first in both saved SHAP and permutation-importance tables**:

- Mean absolute SHAP value: **1.225489** on transformed test features.
- Mean validation ROC-AUC decrease under permutation: **0.171465** (standard deviation **0.012008**) on the original input features.

These quantities measure different forms of model importance and are not directly comparable or causal effects. `dnrday` ranks second in both tables.

**Feature definition matters:** `avtisst` is average TISS over days 3–25, as labeled in the [SUPPORT data dictionary](https://hbiostat.org/data/repo/csupport). It should not be described as acute physiology severity; `aps` is the APACHE III day-3 physiology score.

## Interpretation and reproducibility limits

- **Prediction timing:** `avtisst` aggregates days 3–25, and retained `dnrday` records the day of a DNR order. These can contain information unavailable at an admission or day-3 prediction time. Most physiologic assessments are from study day 3, and `aps`/`sps` are existing physiology scores. An early-prediction claim needs a defined prediction time, a feature-availability audit, and evaluation after excluding unavailable information. See the [dataset author's guidance](https://hbiostat.org/data/repo/supportdesc).
- **Missing model-selection assignment:** the committed notebook references `best_model` and `best_model_name` before defining them. Its saved outputs exist, but a clean top-to-bottom run will stop at that point. Selecting from `results` by validation ROC-AUC and retrieving the corresponding entry from `best_models` is required before the ROC/interpretation cells.
- **Utility/notebook mismatch:** `src/utils.py` defaults to removing 8 columns, whereas the notebook removes 17. The notebook defines its own exclusion list and does not use that helper. Utility tests do not validate the notebook's complete feature-selection or training workflow.
- **Verification scope:** saved outputs support the reported historical metrics and interpretation tables. This documentation update does not rerun training, repair notebook state, or establish clinical deployment, calibration, or performance on a new population.

## Open the analysis

Use Python 3.11, matching the utility-test CI configuration:

```bash
git clone https://github.com/akshayajay/hospital-mortality-prediction.git
cd hospital-mortality-prediction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install ucimlrepo shap
jupyter notebook mortality_prediction.ipynb
```

`ucimlrepo` and `shap` are imported by the notebook but are currently missing from `requirements.txt`, so they are installed explicitly above. The notebook downloads SUPPORT2 directly from UCI; no manually placed `data/support2.csv` is read. Internet access is required for the download.

You can inspect the saved outputs immediately. Before rerunning all cells, address the missing model-selection assignment above. The full grid search repeatedly fits XGBoost-based imputers and can be computationally expensive; the commands above are not a verified fresh reproduction of the saved scores.

To run the existing utility tests without downloading data or training models:

```bash
python -m pip install pytest
python -m pytest tests/ -v
```

## Project structure

```text
hospital-mortality-prediction/
├── mortality_prediction.ipynb   # Analysis, training code, and saved outputs
├── src/
│   └── utils.py                 # Standalone helpers; exclusions differ from notebook
├── tests/
│   └── test_utils.py            # Utility tests, not end-to-end model validation
├── .github/workflows/tests.yml  # Utility-test CI
├── requirements.txt
├── renovate.json
├── LICENSE
└── README.md
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost (imputation) · SHAP · ucimlrepo · matplotlib · seaborn · Jupyter

## Context and author

Built as the final project for DATA 1030 (Hands-on Data Science) at Brown University, School of Professional Studies, 2025.

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
