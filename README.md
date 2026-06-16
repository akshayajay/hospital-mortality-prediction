# In-Hospital Mortality Prediction

[![Tests](https://github.com/akshayajay/hospital-mortality-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/hospital-mortality-prediction/actions/workflows/tests.yml)

Predicting mortality risk among critically ill patients using the SUPPORT2 clinical dataset — built to support early clinical decision-making and ICU resource allocation.

---

## Overview

This project builds and compares machine learning models to classify in-hospital mortality using physiological measurements, demographic data, and clinical indicators from 9,105 critically ill patients.

The goal is to give clinicians a data-driven risk signal early in a patient's admission so that interventions can be initiated sooner and resources allocated more effectively.

---

## Dataset

**SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments)**

- 9,105 patients, 42 features
- Source: [Vanderbilt Biostatistics](https://hbiostat.org/data/) — available via the `Hmisc` R package or direct download
- Target variable: `death` (in-hospital mortality, binary)

Key features include age, sex, disease category, APACHE III score, Glasgow Coma Scale, vital signs, and lab values (serum creatinine, bilirubin, albumin, etc.).

---

## Methodology

**Preprocessing**
- Missing value imputation (median for numeric, mode for categorical)
- One-hot encoding of categorical variables
- StandardScaler normalisation
- Train / validation / test split (60 / 20 / 20)

**Models evaluated**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Support Vector Machine

**Evaluation metrics**
- ROC-AUC (primary)
- Precision, Recall, F1
- Confusion matrix

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/akshayajay/hospital-mortality-prediction.git
cd hospital-mortality-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `support2.csv` from [hbiostat.org/data](https://hbiostat.org/data/) and place it in a `data/` folder:

```
hospital-mortality-prediction/
└── data/
    └── support2.csv
```

### 5. Run the notebook

```bash
jupyter notebook mortality_prediction.ipynb
```

---

## Project Structure

```
hospital-mortality-prediction/
├── mortality_prediction.ipynb   # Full analysis and modelling pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn`

---

## Context

Built as the final project for DATA 1030 (Hands-on Data Science) at Brown University, School of Professional Studies, 2025.

---

## Author

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
