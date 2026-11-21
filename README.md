# RiskLens-AI

### Credit Risk Intelligence Pipeline: XGBoost Default Prediction, Threshold Tuning, Statistical Hypothesis Testing & Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.835-brightgreen.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## 📌 Project Overview

**RiskLens-AI** is an end-to-end Machine Learning credit-risk intelligence pipeline built on **32,000+ loan applications**. It combines inferential statistics, unsupervised borrower segmentation, supervised gradient boosting with class-imbalance weighting, ROC-AUC diagnostic curves, and business-focused decision threshold tuning to identify high-risk borrowers and automate credit advisory.

---

## 🎯 Key Highlights & Methodologies

* **32,000+ Loan Records Pipeline:** Full data preprocessing, feature normalization, and an 80/20 train-test split (`test_size=0.2`).
* **Correlation Matrix Analysis:** Evaluated feature associations using Pearson correlation matrices and heatmap visualizations, identifying `loan_percent_income` as the highest positive correlate with default.
* **Inferential Hypothesis Testing (Welch's t-test):** Conducted two-sample statistical hypothesis testing (**$t = 59.1, p < 0.0001$**), proving that borrowers who default carry statistically significant higher loan-to-income ratios (mean 0.247 vs. 0.149).
* **Borrower Profiling (K-Means Clustering):** Segmented borrowers into **3 distinct behavioral clusters** based on income, credit history length, and requested principal.
* **XGBoost Classifier with Class-Imbalance Weighting:** Handled the 3.58:1 non-defaulter to defaulter skew (~22% default rate) using dynamic `scale_pos_weight`.
* **ROC-AUC Evaluation (AUC = 0.835):** Evaluated overall discriminatory performance using ROC curves and AUC metrics.
* **Decision Threshold Tuning (Recall Boost to 84.2%):** Tuned the decision threshold to **0.35**, prioritizing loan default capture and boosting recall from standard levels up to **84.2%** (with precision ~0.40) to minimize unhedged default risk.
* **Feature Importance Ranking:** Verified that **loan-to-income ratio** is the top driver of risk (**~61% importance**), followed by annual income (**~22%**).
* **Automated Credit Recommendation Engine:** Flags loans exceeding 60% of borrower income and generates personalized advice on loan reduction and credit history building.

---

## 📊 Dataset

* **Source:** Credit Risk Dataset
* **Records:** 32,581 loan applications
* **File:** `credit_risk_dataset.csv.zip` (included directly in this repository)
* **Target:** `loan_status` (0 = Non-defaulter, 1 = Defaulter)
* **Features:**
  - `person_income`: Annual borrower income
  - `loan_amnt`: Requested loan principal
  - `loan_percent_income`: Loan-to-income ratio (primary risk driver)
  - `cb_person_cred_hist_length`: Credit history duration in years

---

## 🔬 Statistical Analysis & Hypothesis Testing

* **Null Hypothesis ($H_0$):** Mean loan-to-income ratio between defaulters and non-defaulters is identical.
* **Alternative Hypothesis ($H_1$):** Defaulters carry a significantly higher loan-to-income ratio.
* **Statistical Test:** Two-sample Welch's t-test:
  - **Non-Defaulters Mean Ratio:** `0.1488`
  - **Defaulters Mean Ratio:** `0.2469`
  - **T-statistic:** `59.09`
  - **P-value:** `< 0.0001`
* **Conclusion:** Null hypothesis is decisively rejected; loan-to-income ratio serves as an empirically verified risk factor.

---

## 📈 Model Performance & Threshold Tuning

In credit risk assessment, a **False Negative** (approving a loan that defaults) costs significantly more than a **False Positive** (flagging an applicant for secondary review). The pipeline tunes the probability threshold to balance the business trade-off:

| Decision Threshold | Defaulter Recall | Defaulter Precision | F1-Score | Overall Accuracy | Business Impact |
|---|---|---|---|---|---|
| **0.55** | 0.638 | 0.557 | 0.595 | 80.7% | High precision, lower risk coverage |
| **0.50 (Default)** | 0.673 | 0.526 | 0.590 | 79.3% | Standard baseline |
| **0.45** | 0.709 | 0.501 | 0.587 | 77.9% | Balanced risk capture |
| **0.40** | 0.768 | 0.453 | 0.570 | 74.3% | High sensitivity |
| **0.35 (Tuned)** | **0.842 (84.2%)** | **0.387 (~40%)** | **0.530** | **66.9%** | **Maximizes default detection, capturing 84.2% of risky loans** |
| **0.30** | 0.890 | 0.348 | 0.500 | 60.5% | Conservative, high-caution screening |

* **ROC-AUC Score:** **0.835**

---

## 💡 Recommendation Engine Rules

The rule-based advisory component inspects applicants to generate actionable financial feedback:
1. **Excessive Debt Burden:** Flags when loan amount exceeds 60% of annual income (`loan_amnt > 0.6 * person_income`).
2. **Limited Credit Duration:** Prompts credit history building when history length is less than 2 years.
3. **Probability-Based Risk Tiering:**
   - **Low Risk:** Probability $< 0.30$
   - **Medium Risk:** $0.30 \le \text{Probability} < 0.70$
   - **High Risk:** Probability $\ge 0.70$

---

## 🛠 Tech Stack

* **Language:** Python 3
* **Libraries:** XGBoost, Scikit-Learn, SciPy (Stats), Pandas, NumPy, Matplotlib
* **Environment:** Jupyter Notebook

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/SuyashMathur-bit/RiskLens-AI.git
cd RiskLens-AI

# 2. Install dependencies
pip install xgboost scikit-learn scipy pandas numpy matplotlib

# 3. Launch notebook
jupyter notebook "RiskLens AI.ipynb"
```
The notebook automatically loads `credit_risk_dataset.csv.zip` from the repository root via relative path.

---

## 👤 Author

* **Suyash Mathur** - [SuyashMathur-bit](https://github.com/SuyashMathur-bit)
