# RiskLens-AI

### Machine Learning-Based Credit Risk Assessment and Customer Segmentation

## Project Overview

RiskLens-AI is an end-to-end Machine Learning project designed to analyze borrower profiles, identify customer segments, predict credit risk, and generate personalized financial recommendations.

The project combines both Unsupervised Learning and Supervised Learning techniques to understand customer behavior and assess the likelihood of loan default.

The primary objective is to help financial institutions make data-driven lending decisions while identifying high-risk borrowers and improving risk management strategies.

---

## Dataset

**Source:** Credit Risk Dataset

**Records:** 32,000+ loan applications

**Data Type:** Structured Financial Data

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
* Loan Status (Target Variable)

---

## Tools & Technologies

### Python Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost

### Machine Learning Techniques

* Data Cleaning
* Feature Engineering
* Data Normalization
* KMeans Clustering
* Random Forest Classification
* XGBoost Classification
* Feature Importance Analysis
* Risk Categorization
* Model Evaluation

---

## Project Workflow

### Data Preprocessing

* Loaded and explored dataset
* Checked missing values
* Removed inconsistencies
* Handled categorical features
* Performed feature scaling using StandardScaler

---

### Customer Segmentation (KMeans Clustering)

Applied KMeans Clustering to group borrowers into similar financial profiles.

#### Objectives

* Discover borrower segments
* Analyze financial behavior patterns
* Understand income and loan characteristics
* Support personalized recommendations

---

### Cluster Analysis

Generated cluster summaries using group statistics.

Analyzed:

* Average Income
* Average Loan Amount
* Average Credit Score
* Average Savings Behavior
* Spending Patterns

---

### Risk Recommendation Engine

Created a rule-based recommendation system to provide financial suggestions such as:

* Reducing loan burden
* Improving savings habits
* Increasing creditworthiness
* Managing debt-to-income ratio

---

### Credit Risk Prediction

Built supervised learning models to predict borrower risk.

#### Models Implemented

##### Random Forest Classifier

Used as a baseline ensemble model.

##### XGBoost Classifier

Used for advanced risk prediction and performance comparison.

---

## Model Evaluation

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

### XGBoost Performance

* Accuracy: ~84%
* Precision (Risky Borrowers): 75%
* Recall (Risky Borrowers): 43%
* F1 Score: 55%

The dataset contains approximately:

* Safe Borrowers: 25,473
* Risky Borrowers: 7,108

This class imbalance was considered during model evaluation and interpretation.

---

## Feature Importance Analysis

Feature importance was used to identify the strongest drivers of borrower risk.

Examples:

* Credit Score
* Loan-to-Income Ratio
* Income Level
* Loan Amount
* Credit History Length

This helps explain model decisions and improves interpretability.

---

## Visualizations

The project includes:

### Exploratory Analysis

* Income Distribution
* Loan Amount Distribution
* Credit Score Analysis

### Clustering Visualizations

* Customer Segment Distribution
* Cluster Comparison Charts

### Model Evaluation Visualizations

* Actual vs Predicted Results
* Feature Importance Bar Chart
* Confusion Matrix

---

## Key Insights

* Credit Score is a major indicator of borrower risk.
* Loan-to-Income Ratio significantly affects repayment behavior.
* Customer segmentation reveals distinct borrower profiles.
* XGBoost outperformed the baseline model.
* Class imbalance affects recall for risky borrowers.

---

## Business Impact

This project can help financial institutions:

* Identify high-risk applicants
* Improve lending decisions
* Reduce default risk
* Understand customer segments
* Support personalized financial recommendations
* Improve portfolio risk management

---

## Future Improvements

* Hyperparameter Optimization
* SMOTE for Class Balancing
* ROC-AUC Analysis
* Model Deployment with Flask/FastAPI
* Real-Time Prediction Dashboard
* Deep Learning-Based Risk Prediction

---

## Repository Structure

├── README.md

├── RiskLens_AI.ipynb

└── credit_risk_dataset.csv
