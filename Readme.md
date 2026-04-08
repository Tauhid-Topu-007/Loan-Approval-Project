# Loan Approval Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project implements a comprehensive machine learning pipeline for predicting loan approval decisions based on applicant information. The model analyzes various factors including income, credit score, employment status, and other financial indicators to determine loan eligibility with **92.5% accuracy** using XGBoost.

## 📊 Dataset

The dataset (`loan approval data.csv`) contains 1000 entries with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Applicant_ID | Numeric | Unique identifier for each applicant |
| Applicant_Income | Numeric | Income of the primary applicant ($2,009 - $19,988) |
| Coapplicant_Income | Numeric | Income of the co-applicant ($1 - $9,996) |
| Employment_Status | Categorical | Employment type (Salaried/Self-employed/Unemployed) |
| Age | Numeric | Applicant's age (21 - 59) |
| Marital_Status | Categorical | Marital status (Married/Single) |
| Dependents | Numeric | Number of dependents (0 - 3) |
| Credit_Score | Numeric | Applicant's credit score (550 - 799) |
| Existing_Loans | Numeric | Number of existing loans (0 - 4) |
| DTI_Ratio | Numeric | Debt-to-Income ratio (0.10 - 0.60) |
| Savings | Numeric | Total savings amount ($65 - $19,996) |
| Collateral_Value | Numeric | Value of collateral provided ($36 - $49,954) |
| Loan_Amount | Numeric | Requested loan amount ($1,015 - $39,995) |
| Loan_Term | Numeric | Loan term in months (12 - 84) |
| Loan_Purpose | Categorical | Purpose of the loan |
| Property_Area | Categorical | Area type (Urban/Semiurban/Rural) |
| Education_Level | Categorical | Education qualification |
| Gender | Categorical | Applicant's gender |
| Employer_Category | Categorical | Type of employer |
| **Loan_Approved** | **Categorical** | **Target variable (Yes/No)** |

## 🔧 Project Pipeline

### 1. Data Preprocessing
- **Handling Missing Values**: 
  - Numerical columns: Mean imputation
  - Categorical columns: Mode (most frequent) imputation
- **Feature Engineering**:
  - Removed `Applicant_ID` (non-predictive feature)
  - Created polynomial features: `DTI_Ratio²`, `Credit_Score²`

### 2. Exploratory Data Analysis (EDA)
- Class distribution visualization (Pie Chart)
- Categorical feature analysis (Bar plots)
- Numerical feature distributions (Histograms)
- Outlier detection (Box plots)
- Correlation analysis (Heatmap)

**Key Findings:**
- Strong positive correlation between `Credit_Score` and `Loan_Approved` (0.45)
- Strong negative correlation between `DTI_Ratio` and `Loan_Approved` (-0.44)
- Higher income applicants show slightly better approval rates

### 3. Feature Encoding
- **Label Encoding**: Binary categorical variables (`Education_Level`, `Loan_Approved`)
- **One-Hot Encoding**: Multi-class categorical features (Employment status, Marital status, etc.)

### 4. Feature Scaling
- **StandardScaler**: Applied to normalize all numerical features

### 5. Model Training & Evaluation

The following models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **92.5%** | **0.848** | **0.918** | **0.882** |
| **Gradient Boosting** | **91.5%** | **0.833** | **0.902** | **0.866** |
| AdaBoost | 90.5% | 0.809 | 0.902 | 0.853 |
| Random Forest | 90.5% | 0.828 | 0.869 | 0.848 |
| Decision Tree | 89.0% | 0.820 | 0.820 | 0.820 |
| Logistic Regression | 87.5% | 0.790 | 0.803 | 0.797 |
| Naive Bayes | 86.5% | 0.783 | 0.770 | 0.777 |
| SVM | 84.0% | 0.774 | 0.672 | 0.719 |
| KNN | 75.5% | 0.620 | 0.508 | 0.559 |

### 6. Model Comparison

#### Before Feature Engineering:
- XGBoost: 92.5% accuracy
- AdaBoost: 90.5% accuracy
- Random Forest: 89.5% accuracy

#### After Feature Engineering:
- XGBoost: 92.5% accuracy (maintained)
- Random Forest: 90.5% accuracy (improved)
- Gradient Boosting: 91.5% accuracy

## 📈 Key Insights

1. **Credit Score** is the strongest predictor of loan approval
2. **DTI Ratio** (Debt-to-Income) shows strong negative correlation with approval
3. **XGBoost** consistently outperforms other models
4. Feature engineering improved ensemble model performance

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
