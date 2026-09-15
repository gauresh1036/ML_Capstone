# Saved Model Artifacts (Review 1)

This directory stores serialized models (via `joblib`) for the best-performing algorithms as outlined in the Capstone Guidelines (Section 8).

## Models Included:
1. `best_regression_gbr.pkl`:
   - Algorithm: Gradient Boosting Regressor (Hyperparameter-tuned)
   - Parameters: `learning_rate=0.03, max_depth=4, n_estimators=400, subsample=0.8`
   - Evaluation Metric: $R^2 = 0.9242$ on held-out test split
2. `best_classification_dt.pkl`:
   - Algorithm: Decision Tree Classifier (Tuned `max_depth=5`)
   - Evaluation Metric: Weighted F1 = 0.8950, Accuracy = 0.8959
