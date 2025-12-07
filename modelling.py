"""
modelling.py

OLS regression utilities WITHOUT statsmodels.

Implements:

- prepare_modelling_data:
    Build design matrix X (with intercept) and response y.
- run_ols_regression:
    Run OLS using NumPy linear algebra.
- summarise_model:
    Return coefficient table (coef, std_err, t_stat, p_value).
- model_diagnostics:
    Return residuals, fitted values, Cook's-like influence measure.
- classify_outliers_by_residual:
    Label over-/under-performing units based on residual quantiles.
"""

from typing import List, Tuple, Optional, Dict
from math import erf, sqrt

import numpy as np
import pandas as pd


# ---------- Helpers ---------- #

def normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------- Data preparation ---------- #

def prepare_modelling_data(
    df: pd.DataFrame,
    dependent_var: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for OLS regression.

    - continuous_vars: numeric predictors (will be cast to float)
    - categorical_vars: converted to dummies (drop_first=True)
    - Adds a constant term named 'const' as the first column.

    Returns
    -------
    X : DataFrame (n x k)
    y : Series (n,)
    """
    cols_needed = [dependent_var] + continuous_vars + categorical_vars
    df_model = df[cols_needed].copy()

    if drop_na:
        df_model = df_model.dropna(subset=cols_needed)

    # Continuous
    X_cont = df_model[continuous_vars].astype(float) if continuous_vars else pd.DataFrame(index=df_model.index)

    # Categorical → dummies
    if categorical_vars:
        X_cat = pd.get_dummies(df_model[categorical_vars], drop_first=True, dummy_na=False)
    else:
        X_cat = pd.DataFrame(index=df_model.index)

    # Intercept
    X_const = pd.Series(1.0, index=df_model.index, name="const")

    # Combine
    X = pd.concat([X_const, X_cont, X_cat], axis=1)
    y = df_model[dependent_var].astype(float)

    return X, y


# ---------- OLS using NumPy ---------- #

def run_ols_regression(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, object]:
    """
    Run OLS regression using NumPy (no statsmodels).

    Returns a dict with:
    - beta: Series of coefficients
    - se: Series of standard errors
    - t: Series of t-statistics
    - p: Series of ~p-values (normal approximation)
    - y_hat: Series of fitted values
    - residuals: Series of residuals
    - r2: float R-squared
    - adj_r2: float adjusted R-squared
    - n: int number of observations
    - k: int number of parameters
    - sigma2: float residual variance (SSR / (n - k))
    """
    # Convert to matrices
    X_mat = X.values
    y_vec = y.values.reshape(-1, 1)

    n, k = X_mat.shape  # n observations, k parameters

    # (X'X)^-1 X'y
    XtX = X_mat.T @ X_mat
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_mat.T @ y_vec
    beta = XtX_inv @ XtY  # (k, 1)

    # Predictions & residuals
    y_hat = X_mat @ beta
    residuals = y_vec - y_hat

    # Sum of squares
    ssr = float((residuals.T @ residuals))  # residual sum of squares
    sst = float(((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean())))  # total sum of squares

    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan

    # Residual variance
    sigma2 = ssr / (n - k) if n > k else np.nan

    # Var(beta) = sigma^2 (X'X)^-1
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)

    # t-stats & approximate p-values (normal approx)
    t_stats = beta / se
    t_flat = t_stats.flatten()
    p_vals = np.array([2 * (1 - normal_cdf(abs(t))) for t in t_flat]).reshape(-1, 1)

    # Wrap into Series
    beta_s = pd.Series(beta.flatten(), index=X.columns, name="coef")
    se_s = pd.Series(se.flatten(), index=X.columns, name="std_err")
    t_s = pd.Series(t_flat, index=X.columns, name="t_stat")
    p_s = pd.Series(p_vals.flatten(), index=X.columns, name="p_value")

    # Fitted & residuals as Series
    y_hat_s = pd.Series(y_hat.flatten(), index=X.index, name="fitted")
    resid_s = pd.Series(residuals.flatten(), index=X.index, name="residual")

    results = {
        "beta": beta_s,
        "se": se_s,
        "t": t_s,
        "p": p_s,
        "y_hat": y_hat_s,
        "residuals": resid_s,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "sigma2": sigma2,
        "XtX_inv": XtX_inv,  # keep for diagnostics if needed
        "X": X,
    }
    return results


# ---------- Summaries & diagnostics ---------- #

def summarise_model(results: Dict[str, object]) -> pd.DataFrame:
    """
    Build a coefficient table from the results dict.
    Columns: coef, std_err, t_stat, p_value
    """
    coef_df = pd.concat(
        [results["beta"], results["se"], results["t"], results["p"]],
        axis=1
    )
    coef_df.columns = ["coef", "std_err", "t_stat", "p_value"]
    return coef_df


def model_diagnostics(results: Dict[str, object]) -> Dict[str, pd.Series]:
    """
    Return residuals, fitted values, and a Cook's-like influence measure.
    """
    X = results["X"]
    residuals = results["residuals"]
    fitted = results["y_hat"]
    sigma2 = results["sigma2"]
    n = results["n"]
    k = results["k"]

    # Leverage h_ii from hat matrix H = X (X'X)^-1 X'
    X_mat = X.values
    XtX_inv = results["XtX_inv"]
    hat_matrix = X_mat @ XtX_inv @ X_mat.T
    h_ii = np.diag(hat_matrix)

    # Cook's distance-like measure:
    # D_i ≈ (resid_i^2 / (k * sigma2)) * (h_ii / (1 - h_ii)^2)
    resid_vals = residuals.values
    cooks = (resid_vals**2 / (k * sigma2)) * (h_ii / (1 - h_ii)**2)
    cooks_s = pd.Series(cooks, index=X.index, name="cooks_distance")

    return {
        "residuals": residuals,
        "fitted": fitted,
        "cooks_distance": cooks_s,
    }


def classify_outliers_by_residual(
    df: pd.DataFrame,
    residuals: pd.Series,
    high_quantile: float = 0.9,
    low_quantile: float = 0.1,
    id_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Label observations as 'over-performing' or 'under-performing' based on
    residual quantiles.

    Parameters
    ----------
    df : DataFrame
        Original data used in modelling (matching residual index).
    residuals : Series
        Residuals from the regression (same index as df).
    high_quantile, low_quantile : float
        Thresholds to classify residuals.
    id_cols : list of str, optional
        Columns to show as identifiers (e.g. ['la_name', 'region_name']).

    Returns
    -------
    DataFrame with id_cols, residual, performance_flag.
    """
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
