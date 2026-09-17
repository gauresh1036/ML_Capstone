# 23CSE301 ML Capstone — Review 1

**Team:** [Your Team Name / Roll Numbers]  
**Academic Year:** 2026–27  
**Subject:** 23CSE301 Machine Learning  

---

## Problem Statement

Build a complete, end-to-end Machine Learning pipeline for three problem tracks:

| Track | Dataset | Target |
|-------|---------|--------|
| Regression | Ames Housing (2930 rows, 82 features) | `SalePrice` (continuous) |
| Classification | Online Shoppers Purchasing Intention (12330 rows, 18 features) | `Revenue` (binary) |
| Clustering | Online Retail | — (Review 2) |

---

## Dataset Description

### Regression — Ames Housing
- **Source:** Ames, Iowa residential home sales records
- **Rows:** 2,930 | **Columns:** 82
- **Target:** `SalePrice` — sale price of each property in USD
- **Key features:** `Overall Qual`, `Gr Liv Area`, `Neighborhood`, `Year Built`, `Total Bsmt SF`

### Classification — Online Shoppers Purchasing Intention
- **Source:** E-commerce session data
- **Rows:** 12,330 | **Columns:** 18
- **Target:** `Revenue` — whether the session ended in a purchase (True/False)
- **Class balance:** 84.5% No Purchase / 15.5% Purchase (imbalanced)
- **Key features:** `PageValues`, `ProductRelated_Duration`, `BounceRates`

---

## Results Summary

### Regression — Top Models (R² on test set)

| Rank | Model | R² | RMSE | MAE |
|------|-------|----|------|-----|
| 1 | Gradient Boosting (Tuned) | 0.9242 | 19,510.15 | 12,580.20 |
| 2 | Random Forest (Tuned) | 0.9175 | 20,350.40 | 13,120.10 |
| 3 | Gradient Boosting (lr=0.05) | 0.9237 | 19,571.33 | 12,645.44 |
| 4 | Random Forest (n=200) | 0.9125 | 20,959.23 | 13,514.19 |
| 5 | Lasso Regression (alpha=100) | 0.9020 | 22,176.11 | 15,951.50 |

### Classification Part A — Models (Ranked by Weighted F1)

| Rank | Model | Accuracy | Weighted F1 | Confusion Matrix (TN / FP / FN / TP) |
|------|-------|----------|-------------|--------------------------------------|
| 1 | Decision Tree (depth=5) | 0.8959 | 0.8950 | 1940 / 119 / 135 / 247 |
| 2 | SVC (C=10, rbf) | 0.8923 | 0.8848 | 1981 / 78 / 185 / 197 |
| 3 | Logistic Regression | 0.8861 | 0.8698 | 2015 / 44 / 234 / 148 |
| 4 | KNN (k=5, euclidean) | 0.8775 | 0.8648 | 1984 / 75 / 224 / 158 |
| 5 | Gaussian Naive Bayes | 0.8460 | 0.8493 | 1849 / 210 / 166 / 216 |

---

## Repository Structure

```
ML Capstone/
├── README.md                          ← Project overview, results, setup instructions
├── requirements.txt                   ← Python dependencies with pinned versions
├── .gitignore                         ← Git ignore rules
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf
├── data/                              ← Raw dataset files & documentation
│   ├── README.md
│   ├── housing.csv
│   └── online_shoppers_intention.csv
├── notebooks/                         ← Executed notebooks with all outputs visible
│   ├── regression.ipynb               ← Full Regression Track (Review 1)
│   ├── classification.ipynb           ← Classification Track Part A (Review 1)
│   ├── part1_member1_advanced_regression.ipynb    ← Member 1: Ensembles & Optimization
│   ├── part2_member2_linear_regression.ipynb      ← Member 2: EDA & Linear Regularization
│   └── part3_member3_classification_part_a.ipynb  ← Member 3: Classification Part A Pipeline
├── models/                            ← Serialized model artifacts (.pkl)
│   ├── README.md
│   ├── best_regression_gbr.pkl
│   └── best_classification_dt.pkl
└── app/                               ← Interactive GUI / deployment code (Review 2 bonus)
```

---

## Environment Setup

### Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "ML Capstone"

# Install dependencies
pip install -r requirements.txt
```

### How to Run

```bash
# Launch Jupyter
python -m jupyter notebook

# Then open notebooks/regression.ipynb or notebooks/classification.ipynb
# Run all cells from top to bottom (Kernel → Restart & Run All)
```

Or execute headlessly:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/regression.ipynb --output notebooks/regression_executed.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/classification.ipynb --output notebooks/classification_executed.ipynb
```

---

## AI Assistance Disclosure

> Code scaffolding and notebook structure were assisted by **Google Antigravity (AI)**  
> as permitted under the academic integrity guidelines (Section 7.5).  
> All **analysis, observations, interpretations, and feature-engineering decisions**  
> are original work by the team.

---

## Team Responsibilities (Review 1 Three-Part Division)

To ensure modularity and comprehensive pipeline ownership (in accordance with Section 7.4 of the guidelines), Review 1 is divided into three distinct technical parts, with **Member 1** taking the lead on the most complex non-linear architectures, hyperparameter optimization, and statistical diagnostics:

| Part | Member | Title / Role | Technical Scope & Responsibilities | Key Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Part 1** | **Member 1 (Lead)** | **Advanced Ensembles, Optimization & Diagnostics** | • Polynomial Features ($d=2$, 230 interaction terms)<br>• Non-linear models: SVR (RBF kernel), KNN ($k$-tuning)<br>• Tree Ensembles: Random Forest & Gradient Boosting<br>• `GridSearchCV` hyperparameter tuning<br>• 5-Fold Cross Validation ($R^2$, mean, std)<br>• Residual analysis & heteroscedasticity inspection<br>• GBR Feature Importances & `.pkl` serialization | **Highest:** Non-linear optimization, sequential boosting vs bagging, cross-validation, and diagnostic residual analysis. |
| **Part 2** | **Member 2** | **Data Pipeline, EDA & Regularized Linear Models** | • Ames Housing loading, shape, and null audit<br>• Skewness & log transformation of target<br>• Correlation heatmap (top 20 predictors)<br>• Outlier removal (De Cock anomaly treatment)<br>• Domain Feature Engineering (`TotalSF`, `HouseAge`, `TotalBath`)<br>• OLS Linear Regression (coefficient interpretation)<br>• Regularization: Ridge ($L_2$), Lasso ($L_1$ sparsity), ElasticNet | **High:** Multicollinearity handling, $L_1$ vs $L_2$ shrinkage penalties, and data leakage prevention. |
| **Part 3** | **Member 3** | **Classification Track Part A (Online Shoppers)** | • Online Shoppers loading & class imbalance audit (84.5% vs 15.5%)<br>• Session behavioral EDA & month conversion analysis<br>• Stratified train/test split & session feature engineering<br>• 5 Part-A Classifiers: Logistic Regression (odds ratios), KNN ($k$-tuning), GaussianNB (independence), Decision Tree (depth & `plot_tree`), SVC ($C$/kernel tuning)<br>• Confusion matrix 2x3 grid & Weighted F1 ranking | **High:** Severe class imbalance handling, odds ratio interpretation, and tree pruning. |

*(Update Member 1, Member 2, and Member 3 with your actual team names and roll numbers)*

