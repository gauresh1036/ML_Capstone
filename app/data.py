"""Cached loaders for datasets, saved models, notebook results and the common test splits.

The test splits are rebuilt with exactly the same cleaning steps, feature engineering and
`train_test_split(..., random_state=42)` call as the notebooks, so every metric computed
live in the GUI equals the one reported in the notebooks.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from features import engineer_housing_features, engineer_session_features

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / 'data', ROOT / 'models', ROOT / 'reports'
FIGURES = REPORTS / 'figures'
RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def results(track):
    return json.loads((REPORTS / f'{track}_results.json').read_text())


@st.cache_data(show_spinner=False)
def clustering_eda():
    return json.loads((REPORTS / 'clustering_eda.json').read_text())


@st.cache_resource(show_spinner='Loading models …')
def model(filename):
    return joblib.load(MODELS / filename)


# ----------------------------------------------------------------------------- raw data (for EDA)
@st.cache_data(show_spinner=False)
def housing_raw():
    return pd.read_csv(DATA / 'housing.csv')


@st.cache_data(show_spinner=False)
def shoppers_raw():
    return pd.read_csv(DATA / 'online_shoppers_intention.csv')


@st.cache_data(show_spinner=False)
def customers():
    """One row per customer: RFM, both cluster labels, segment name, PCA and t-SNE coordinates."""
    return pd.DataFrame(model('clustering_kmeans.pkl')['customers'])


# ----------------------------------------------------------------------------- test splits
@st.cache_data(show_spinner=False)
def housing_split():
    """Same cleaning + engineering + 80:20 split as notebooks/regression.ipynb."""
    df = housing_raw().drop(columns=['Order', 'PID'])
    q1, q3 = df['SalePrice'].quantile([0.25, 0.75])
    df = df[df['SalePrice'].between(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))]
    df = engineer_housing_features(df[df['Gr Liv Area'] < 4500].reset_index(drop=True))
    X, y = df.drop(columns='SalePrice'), df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


@st.cache_data(show_spinner=False)
def shoppers_split():
    """Same de-duplication + engineering + stratified 80:20 split as notebooks/classification.ipynb."""
    df = engineer_session_features(shoppers_raw().drop_duplicates().reset_index(drop=True))
    X, y = df.drop(columns='Revenue'), df['Revenue'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


@st.cache_data(show_spinner='Scoring all 10 classifiers on the test split …')
def classification_test_scores():
    """Purchase probability (or SVM decision score) of every model on the common test split."""
    _, X_test, _, y_test = shoppers_split()
    out = {}
    for name, pipe in model('classification_all_models.pkl').items():
        est = pipe.named_steps['model']
        out[name] = (pipe.predict_proba(X_test)[:, 1] if hasattr(est, 'predict_proba')
                     else pipe.decision_function(X_test))
    return out, y_test.to_numpy()


@st.cache_data(show_spinner='Scoring all 10 regressors on the test split …')
def regression_test_predictions():
    _, X_test, _, y_test = housing_split()
    preds = {name: pipe.predict(X_test) for name, pipe in model('regression_all_models.pkl').items()}
    return preds, y_test.to_numpy()


def has_proba(name):
    return hasattr(model('classification_all_models.pkl')[name].named_steps['model'], 'predict_proba')


def figure_path(name):
    p = FIGURES / name
    return str(p) if p.exists() else None


def numeric_summary(df):
    s = df.describe().T
    s['missing'] = df.isna().sum()
    s['skew'] = df.skew(numeric_only=True)
    return s.round(3)


MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
NOMINAL_CODES = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']


def safe_float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0.0
