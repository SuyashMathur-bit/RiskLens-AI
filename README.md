# RiskLens-AI

### Machine Learning-Based Credit Risk Assessment and Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classification-red.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## Project Overview

**RiskLens-AI** is an end-to-end Machine Learning project designed to analyze borrower profiles, identify customer segments, predict credit risk, and generate personalized financial recommendations.

The project combines both **Unsupervised Learning** (K-Means Clustering) and **Supervised Learning** (Random Forest baseline and XGBoost classifier) to evaluate customer behavior and assess the likelihood of loan default.

The primary objective is to help financial institutions make data-driven lending decisions while identifying high-risk borrowers and improving risk management strategies.

---

## Dataset

* **Source:** Credit Risk Dataset
* **Records:** 32,000+ loan applications
* **File:** `credit_risk_dataset.csv.zip` (included directly in this repository)
* **Data Type:** Structured Financial Data

### Key Features
* Person Age
* Person Income
* Home Ownership
* Employment Length
* Loan Amount
* Loan Interest Rate
* Loan Intent
* Credit History Length
* Credit Score
* **Loan Status (Target Variable)**

---

## Tools & Technologies

### Python Libraries
* Pandas, NumPy
* Matplotlib
* Scikit-Learn
* XGBoost

### Machine Learning Techniques
* Data Cleaning & Imputation
* Feature Scaling (StandardScaler)
* Train-Test Split (80% Train / 20% Test)
* Customer Segmentation (K-Means Clustering)
* Baseline Modeling (Random Forest Classifier)
* Advanced Modeling (XGBoost Classifier with class weighting)
* Feature Importance Analysis
* Risk Categorization & Automated Advisory

---

## Project Workflow

1. **Data Preprocessing:** Loaded dataset, handled missing values, and performed feature scaling.
2. **Train-Test Split:** Configured 80% training and 20% testing split (`test_size=0.2`).
3. **Customer Segmentation (K-Means):** Clustered borrowers into financial profiles to understand behavior.
4. **Credit Risk Modeling:**
   * **Baseline:** Random Forest Classifier.
   * **Advanced Model:** XGBoost Classifier with `scale_pos_weight` to address the 3.58:1 safe-to-risky class imbalance.
5. **Evaluation & Diagnosis:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. **Rule-Based Recommendations:** Generates actionable guidance (e.g. debt-to-income limits, building credit history).

---

## Model Evaluation

### Evaluation Metrics
* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-Score)

### Performance Summary
* **XGBoost Accuracy:** ~84%
* **Precision (Risky Borrowers):** ~75%
* **Recall (Risky Borrowers):** ~43%
* **F1-Score:** ~55%

---

## Feature Importance Analysis

Feature importance analysis identifies the strongest drivers of borrower risk:
* **Loan-to-Income Ratio** (~61% importance)
* **Person Income** (~22% importance)
* **Loan Amount** (~10% importance)
* **Credit History Length** (~7% importance)

---

## How to Run Locally

```bash
git clone https://github.com/SuyashMathur-bit/RiskLens-AI.git
cd RiskLens-AI
pip install pandas numpy scikit-learn xgboost matplotlib
jupyter notebook "RiskLens AI.ipynb"
```
The notebook automatically loads `credit_risk_dataset.csv.zip` from the repository root.

---

## Repository Structure

```
RiskLens-AI/
├── credit_risk_dataset.csv.zip
├── README.md
└── RiskLens AI.ipynb
```

---

## Author

* **Suyash Mathur** - [SuyashMathur-bit](https://github.com/SuyashMathur-bit)
