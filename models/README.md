# Saved models

Each notebook saves its final model here with `joblib`. Each file is a complete scikit-learn object: preprocessing
is fitted on the training split only and bundled with the estimator, so a model can be applied to raw rows directly.

| File | Created by | Contents | Input |
|------|-----------|----------|-------|
| `regression_best_model.pkl` | `notebooks/regression.ipynb` §8 | `Pipeline(ColumnTransformer → TransformedTargetRegressor(best regressor, log1p/expm1))` | raw Ames columns + the 6 engineered features (`app/features.py`) → price in USD |
| `classification_best_model.pkl` | `notebooks/classification.ipynb` §9 | `Pipeline(ColumnTransformer → tuned GradientBoostingClassifier)` | raw session columns + the 5 engineered features (`app/features.py`) → `predict_proba` |
| `regression_all_models.pkl` | `notebooks/regression.ipynb` §8 | dict of all 10 tuned regressors as raw-input pipelines (used by the GUI model lab) | same as above |
| `classification_all_models.pkl` | `notebooks/classification.ipynb` §9 | dict of all 10 tuned classifiers as raw-input pipelines (used by the GUI model lab) | same as above |
| `clustering_kmeans.pkl` | `notebooks/clustering.ipynb` §9 | dict: `pipeline` (log1p → StandardScaler → KMeans), `pca`, `segment_names`, `profile`, `sample_points` | `Recency`, `Frequency`, `Monetary` → cluster id |

Test-set metrics for each model are stored next to them in `reports/*_results.json`.

```python
import joblib, pandas as pd
model = joblib.load('models/classification_best_model.pkl')
```

The pickles require **scikit-learn 1.8.0** (pinned in `requirements.txt`).
