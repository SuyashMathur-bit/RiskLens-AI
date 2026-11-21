# RiskLens-AI

### Machine Learning Credit Risk Assessment, XGBoost Default Prediction & Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-84%25%20Accuracy-red.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## 📌 Project Overview

**RiskLens-AI** is an end-to-end Machine Learning credit-risk intelligence pipeline built on **32,000+ loan applications**. It combines unsupervised customer segmentation and supervised gradient boosting to analyze borrower risk profiles, forecast loan default, address severe class imbalance, and automate borrower advisory.

---

## 🎯 Key Highlights

* **32,000+ Loan Applications Pipeline:** Data cleaning, missing-value imputation, feature scaling, and feature engineering.
* **Borrower Profile Segmentation (KMeans):** Segmented borrowers into **3 distinct behavioral clusters** based on income, credit history, and debt load.
* **XGBoost Default Prediction:** Trained an XGBoost Classifier achieving **84% overall accuracy** and **0.75 precision on defaulters**.
* **Class Imbalance Analysis:** Analyzed the natural lending skew (~22% defaulters vs. ~78% non-defaulters; 3.58:1 ratio) to ensure robust precision-recall evaluation.
* **Feature Importance Analysis:** Discovered that **loan-to-income ratio** is the overwhelming primary driver of default (~61% feature importance), followed by person income (~22%).
* **Rule-Based Recommendation Engine:** Automated credit advisory that flags loans that are disproportionately high relative to borrower income and suggests corrective actions.

---

## 📊 Dataset

* **Source:** Credit Risk Dataset
* **Records:** 32,581 loan applications
* **File:** `credit_risk_dataset.csv.zip` (included directly in this repository)
* **Target:** `loan_status` (0 = Non-defaulter, 1 = Defaulter)
* **Key Features:**
  - `person_income`: Annual borrower income
  - `loan_amnt`: Total requested loan amount
  - `loan_percent_income`: Loan-to-income ratio (primary risk driver)
  - `cb_person_cred_hist_length`: Credit history duration in years

---

## 📈 Model Performance & Metrics

### XGBoost Classifier Performance

| Metric | Non-Defaulters (Class 0) | Defaulters (Class 1) | Overall / Macro |
|---|---|---|---|
| **Precision** | 0.86 | **0.75** | 0.80 |
| **Recall** | 0.96 | 0.43 | 0.70 |
| **F1-Score** | 0.91 | 0.55 | 0.73 |
| **Accuracy** | — | — | **84.2%** |

### Top Feature Drivers (Feature Importance)

1. **`loan_percent_income`** (Loan-to-Income Ratio): **~61.2%**
2. **`person_income`** (Annual Income): **~21.8%**
3. **`loan_amnt`** (Loan Principal): **~9.9%**
4. **`cb_person_cred_hist_length`** (Credit History Length): **~7.1%**

---

## 💡 Recommendation Engine Logic

The pipeline includes an automated recommendation engine that inspects applicant profiles:
* **High Debt Burden:** Flags when loan amount exceeds 60% of annual income (`loan_amnt > 0.6 * person_income`).
* **Credit Maturity:** Recommends credit building if credit history length is under 2 years.
* **Risk Tiering:** Automatically stratifies predicted default probabilities into Low Risk ($p < 0.3$), Medium Risk ($0.3 \le p < 0.7$), and High Risk ($p \ge 0.7$).

---

## 🛠 Tech Stack

* **Language:** Python 3
* **Libraries:** XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib
* **Environment:** Jupyter Notebook

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/SuyashMathur-bit/RiskLens-AI.git
cd RiskLens-AI

# 2. Install dependencies
pip install xgboost scikit-learn pandas numpy matplotlib

# 3. Launch notebook
jupyter notebook "RiskLens AI.ipynb"
```
The notebook automatically loads `credit_risk_dataset.csv.zip` from the repository root via relative path.

---

## 👤 Author

* **Suyash Mathur** - [SuyashMathur-bit](https://github.com/SuyashMathur-bit)
