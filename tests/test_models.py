"""Consistency checks between the notebooks, the saved models and the GUI.

Run from the repository root after executing the notebooks:
    python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
from features import engineer_housing_features, engineer_session_features  # noqa: E402


def results(track):
    return json.loads((ROOT / 'reports' / f'{track}_results.json').read_text())


def test_classification_model_reproduces_notebook_score():
    """GUI feature code + saved pipeline must give the notebook's test weighted F1."""
    df = pd.read_csv(ROOT / 'data' / 'online_shoppers_intention.csv').drop_duplicates().reset_index(drop=True)
    data = engineer_session_features(df)
    X, y = data.drop(columns='Revenue'), data['Revenue'].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = joblib.load(ROOT / 'models' / 'classification_best_model.pkl')
    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    assert f1 == pytest.approx(results('classification')['test_metrics']['F1 (w)'], abs=1e-4)


def test_regression_model_reproduces_notebook_score():
    """GUI feature code + saved pipeline must give the notebook's test R²."""
    from sklearn.metrics import r2_score
    df = pd.read_csv(ROOT / 'data' / 'housing.csv').drop(columns=['Order', 'PID'])
    q1, q3 = df['SalePrice'].quantile([0.25, 0.75])
    df = df[df['SalePrice'].between(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))]
    df = engineer_housing_features(df[df['Gr Liv Area'] < 4500].reset_index(drop=True))
    X, y = df.drop(columns='SalePrice'), df['SalePrice']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(ROOT / 'models' / 'regression_best_model.pkl')
    assert r2_score(y_test, model.predict(X_test)) == pytest.approx(results('regression')['test_metrics']['R2'], abs=1e-4)


def test_clustering_artifact_is_consistent():
    art = joblib.load(ROOT / 'models' / 'clustering_kmeans.pkl')
    sample = pd.DataFrame(art['sample_points'])
    pred = art['pipeline'].predict(sample[art['features']])
    assert (pred == sample['Cluster'].to_numpy()).mean() > 0.99
    assert len(set(art['segment_names'].values())) == results('clustering')['k']


def test_feature_functions_match_notebooks():
    """The GUI copies of the feature functions must be identical to the notebook definitions."""
    import inspect
    for nb, fn, gui in [('classification.ipynb', 'def engineer_features', engineer_session_features),
                        ('regression.ipynb', 'def engineer_housing_features', engineer_housing_features)]:
        cells = json.loads((ROOT / 'notebooks' / nb).read_text(encoding='utf-8'))['cells']
        src = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code' and fn in ''.join(c['source']))
        def body(lines):
            """Indented lines after the def + docstring line, stopping at the first unindented line."""
            out = []
            for line in lines[2:]:
                if line.strip() and not line.startswith(' '):
                    break
                if line.strip():
                    out.append(line.strip())
            return out

        nb_body = body((fn + src.split(fn)[1]).splitlines())
        gui_body = body(inspect.getsource(gui).splitlines())
        assert nb_body == gui_body, f'{gui.__name__} differs from {nb}'
