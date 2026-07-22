# Pima Indians Diabetes Classification

A machine learning project to predict diabetes using the Pima Indians Diabetes dataset from UCI / Kaggle.

## Project Overview

This project follows a complete data science pipeline — from raw data exploration to model evaluation — built and understood step by step.

**Goal:** Predict whether a patient has diabetes (binary classification: 0 = No, 1 = Yes)

**Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
768 patients, 8 features, 1 target variable

---

## Pipeline

| Step | Task | Key finding |
|------|------|-------------|
| 1 | Load & explore | 768 rows, 9 columns, no visible NaN |
| 2 | Detect missing values | Zeros in 5 columns are hidden missing values |
| 3 | Imputation | dropna for <5% missing, fillna(median) for the rest |
| 4 | EDA & Visualization | Glucose has highest correlation with Outcome (0.49) |
| 5 | Feature engineering | StandardScaler applied to all features |
| 6 | Train/test split | 80/20 split, random_state=42 |
| 7 | Modeling | Compared Logistic Regression, Decision Tree, Random Forest |
| 8 | Evaluation | Selected Random Forest based on accuracy + recall balance |

---

## Results

| Model | Accuracy | Recall (Diabetes) | F1 (Diabetes) |
|-------|----------|-------------------|---------------|
| Logistic Regression | 0.79 | 0.63 | 0.64 |
| Decision Tree | 0.77 | 0.77 | 0.67 |
| **Random Forest** | **0.80** | **0.70** | **0.67** |

**Selected model: Random Forest**  
Chosen for the best balance between overall accuracy (0.80) and recall for the Diabetes class (0.70).

> For medical screening tasks, recall matters most — missing a diabetic patient (false negative) is more dangerous than a false alarm.

---

## Key Learnings

- Missing values are not always `NaN` — domain knowledge is required to spot hidden zeros
- Correlation between individual features and target is low (~0.2–0.5), which is normal for health data; ML combines all features together
- Accuracy alone can be misleading — always check precision, recall, and F1, especially on imbalanced datasets
- Model selection depends on the use case, not just the highest number

---

## Tech Stack

- Python 3.14
- pandas, numpy
- scikit-learn
- matplotlib

---

## How to Run

```bash
git clone https://github.com/<your-username>/pima-diabetes-classification
cd pima-diabetes-classification
pip install pandas numpy scikit-learn matplotlib
jupyter notebook pima_v2_practice.ipynb
```

---

## File Structure

```
pima-diabetes-classification/
├── pima_v2_practice.ipynb   # main notebook
├── diabetes.csv             # dataset
└── README.md                # this file
```
