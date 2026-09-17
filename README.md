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
│   └── classification.ipynb           ← Classification Track Part A (Review 1)
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


