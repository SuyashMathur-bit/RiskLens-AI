# RiskLens-AI

### Credit Risk Intelligence Pipeline: XGBoost Default Prediction, Threshold Tuning, Statistical Hypothesis Testing & Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-~0.87-brightgreen.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

---

## 📌 Project Overview

**RiskLens-AI** is an end-to-end Machine Learning credit-risk intelligence pipeline built on **32,000+ loan applications**. It combines inferential statistics, unsupervised borrower segmentation, supervised gradient boosting with class-imbalance weighting, ROC-AUC diagnostic curves, and business-focused decision threshold tuning to identify high-risk borrowers and automate credit advisory.

---

## 🎯 Key Highlights & Methodologies

* **32,000+ Loan Records Pipeline:** Full data preprocessing, feature normalization, and an 80/20 train-test split (`test_size=0.2`).
* **Correlation Matrix Analysis:** Evaluated feature associations using Pearson correlation matrices and heatmap visualizations, identifying `loan_percent_income` as the highest positive correlate with default.
* **Inferential Hypothesis Testing (Welch's t-test):** Conducted two-sample statistical hypothesis testing ($p < 0.001$), rigorously proving that borrowers who default carry statistically significant higher loan-to-income ratios than non-defaulters.
* **Borrower Profiling (K-Means Clustering):** Segmented borrowers into **3 distinct behavioral clusters** based on income, credit history length, and requested principal.
* **XGBoost Classifier with Class-Imbalance Weighting:** Handled the 3.58:1 non-defaulter to defaulter skew (~22% default rate) using dynamic `scale_pos_weight`.
* **ROC-AUC Evaluation (~0.87):** Evaluated overall discriminatory performance using ROC curves and AUC metrics.
* **Decision Threshold Tuning (Recall Boost: 43% ➔ 72%+):** Lowered the default 0.50 threshold to **0.35**, prioritizing loan default capture and boosting recall from **0.43 to >0.72** while keeping precision balanced.
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
* **Statistical Test:** Two-sample Welch's t-test ($t \approx 72.4, p < 0.0001$).
* **Conclusion:** Null hypothesis is decisively rejected; loan-to-income ratio serves as an empirically verified risk factor.

---

## 📈 Model Performance & Threshold Tuning

In credit risk assessment, a **False Negative** (approving a loan that defaults) is substantially more expensive than a **False Positive** (flagging an applicant for secondary review). The pipeline tunes the probability threshold to balance the business trade-off:

| Decision Threshold | Defaulter Recall | Defaulter Precision | F1-Score | Business Impact |
|---|---|---|---|---|
| **0.50 (Default)** | 0.43 | 0.75 | 0.55 | High false negatives, miss 57% of defaults |
| **0.40** | 0.61 | 0.69 | 0.65 | Balanced risk capture |
| **0.35 (Tuned)** | **0.72 - 0.75** | **0.65** | **0.68 - 0.70** | **Captures ~75% of default risk with high precision** |
| **0.30** | 0.81 | 0.59 | 0.68 | Conservative, high-caution lending |

* **ROC-AUC Score:** **~0.87**

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
