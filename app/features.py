"""Feature engineering used by the GUI.

These functions mirror, line for line, the ones defined in the notebooks
(`engineer_features` in notebooks/classification.ipynb and
`engineer_housing_features` in notebooks/regression.ipynb). The saved model
pipelines expect these engineered columns to be present, so the GUI must apply
exactly the same transformations before calling `predict`.
tests/test_models.py checks that the two copies stay in sync.
"""
import numpy as np
import pandas as pd


def engineer_session_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Classification track: add the five engineered session features."""
    out = frame.copy()
    out['TotalPageVisits'] = out['Administrative'] + out['Informational'] + out['ProductRelated']
    out['TotalDuration'] = (out['Administrative_Duration'] + out['Informational_Duration']
                            + out['ProductRelated_Duration'])
    out['ProductEngagementRatio'] = np.where(out['TotalDuration'] > 0,
                                             out['ProductRelated_Duration'] / out['TotalDuration'], 0.0)
    out['AvgProductPageTime'] = np.where(out['ProductRelated'] > 0,
                                         out['ProductRelated_Duration'] / out['ProductRelated'], 0.0)
    out['HasPageValue'] = (out['PageValues'] > 0).astype(int)
    out['Weekend'] = out['Weekend'].astype(int)
    return out


def engineer_housing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Regression track: add the six engineered housing features."""
    out = frame.copy()
    out['TotalSF'] = out['Total Bsmt SF'].fillna(0) + out['1st Flr SF'] + out['2nd Flr SF']
    out['HouseAge'] = out['Yr Sold'] - out['Year Built']
    out['RemodAge'] = out['Yr Sold'] - out['Year Remod/Add']
    out['TotalBath'] = (out['Full Bath'] + 0.5 * out['Half Bath']
                        + out['Bsmt Full Bath'].fillna(0) + 0.5 * out['Bsmt Half Bath'].fillna(0))
    out['HasPool'] = (out['Pool Area'] > 0).astype(int)
    out['HasGarage'] = (out['Garage Area'].fillna(0) > 0).astype(int)
    return out
