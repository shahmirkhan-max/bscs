"""
Regression modelling utilities using statsmodels.

Models can be run at LA level, school level, or any other aggregation
as long as the dataframe has the required columns.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def prepare_modelling_data(
    df: pd.DataFrame,
    dependent_var: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for OLS regression.

    - Creates dummy variables for categorical predictors.
    - Adds a constant term.
    """
    cols_needed = [dependent_var] + continuous_vars + categorical_vars
    df_model = df.copy()[cols_needed]

    if drop_na:
        df_model = df_model.dropna(subset=cols_needed)

    X_cont = df_model[continuous_vars].astype(float)
    X_cat = pd.get_dummies(df_model[categorical_vars], drop_first=True, dummy_na=False)

    X = pd.concat([X_cont, X_cat], axis=1)
    X = sm.add_constant(X)

    y = df_model[dependent_var].astype(float)

    return X, y


def run_ols_regression(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y, X, missing="drop").fit()


def summarise_model(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues
    conf_int = model.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]

    out = pd.concat([params, bse, tvalues, pvalues, conf_int], axis=1)
    out.columns = ["coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
    return out


def model_diagnostics(
    model: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, pd.Series]:
    resid = model.resid
    fitted = model.fittedvalues
    influence = model.get_influence()
    cooks_d, _ = influence.cooks_distance
    return {
        "residuals": resid,
        "fitted": fitted,
        "cooks_distance": pd.Series(cooks_d, index=model.model.data.row_labels),
    }


def classify_outliers_by_residual(
    df: pd.DataFrame,
    residuals: pd.Series,
    high_quantile: float = 0.9,
    low_quantile: float = 0.1,
    id_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if id_cols is None:
        id_cols = []

    diag_df = df.copy()
    diag_df = diag_df.assign(residual=residuals)

    high_thr = diag_df["residual"].quantile(high_quantile)
    low_thr = diag_df["residual"].quantile(low_quantile)

    conditions = [
        diag_df["residual"] >= high_thr,
        diag_df["residual"] <= low_thr
    ]
    choices = ["over-performing", "under-performing"]
    diag_df["performance_flag"] = np.select(conditions, choices, default="as-expected")

    cols_to_show = id_cols + ["residual", "performance_flag"]
    cols_to_show = [c for c in cols_to_show if c in diag_df.columns]

    return diag_df[cols_to_show].sort_values("residual", ascending=False)
