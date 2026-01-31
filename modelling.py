"""
modelling.py

Custom "White Box" Statistical Engine.
Implements OLS regression from first principles (Linear Algebra) 
rather than using 'Black Box' libraries like sklearn.

Used to:
1. Calculate the 'Deprivation Gradient' (Regression Line).
2. Identify Contextual Outliers (Residual Analysis).
3. Quantify Feature Importance (Coefficients).
"""

from typing import List, Tuple, Optional, Dict
from math import erf, sqrt

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# =========================================================
# 1. MATH HELPERS (The "White Box" Engine)
# =========================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using error function (erf)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def prepare_modelling_data(
    df: pd.DataFrame,
    dependent_var: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Design Matrix X and Target Vector y.
    Adds intercept term automatically.
    """
    cols_needed = [dependent_var] + continuous_vars + categorical_vars
    df_model = df[cols_needed].copy()

    if drop_na:
        df_model = df_model.dropna(subset=cols_needed)

    # Continuous columns
    if continuous_vars:
        X_cont = df_model[continuous_vars].astype(float)
    else:
        X_cont = pd.DataFrame(index=df_model.index)

    # Categorical columns (One-Hot Encoding)
    if categorical_vars:
        X_cat = pd.get_dummies(df_model[categorical_vars], drop_first=True, dummy_na=False)
        # Clean column names (replace spaces with underscores)
        X_cat.columns = X_cat.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
    else:
        X_cat = pd.DataFrame(index=df_model.index)

    # Intercept (The Baseline)
    X_const = pd.Series(1.0, index=df_model.index, name="const")

    # Assemble X and y
    X = pd.concat([X_const, X_cont, X_cat], axis=1)
    y = df_model[dependent_var].astype(float)

    return X, y

def run_ols_regression(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """
    Performs OLS Regression using Matrix Algebra: beta = (X'X)^-1 X'y
    """
    # Convert to NumPy arrays
    X_mat = X.values
    y_vec = y.values.reshape(-1, 1)
    n, k = X_mat.shape

    # 1. Calculate Coefficients (Beta)
    XtX = X_mat.T @ X_mat
    
    # Handle singular matrix (collinearity protection)
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix - variables are perfectly correlated."}
        
    XtY = X_mat.T @ y_vec
    beta = XtX_inv @ XtY

    # 2. Predictions & Residuals
    y_hat = X_mat @ beta
    residuals = y_vec - y_hat

    # 3. Statistics
    ssr = float((residuals.T @ residuals)) # Sum Squared Residuals
    sst = float(((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean()))) # Total Sum Squares
    
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0
    
    sigma2 = ssr / (n - k) if n > k else 0.0 # Variance
    
    # Standard Errors
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)
    
    # t-stats and p-values
    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se!=0)
    p_vals = np.array([2 * (1 - normal_cdf(abs(t))) for t in t_stats.flatten()]).reshape(-1, 1)

    # Pack results
    results = {
        "beta": pd.Series(beta.flatten(), index=X.columns, name="Coefficient"),
        "se": pd.Series(se.flatten(), index=X.columns, name="Std Err"),
        "t": pd.Series(t_stats.flatten(), index=X.columns, name="t-stat"),
        "p": pd.Series(p_vals.flatten(), index=X.columns, name="P-value"),
        "y_hat": pd.Series(y_hat.flatten(), index=X.index, name="Predicted"),
        "residuals": pd.Series(residuals.flatten(), index=X.index, name="Residual"),
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "X": X
    }
    return results

# =========================================================
# 2. DIAGNOSTICS & OUTLIER DETECTION
# =========================================================

def classify_outliers(df: pd.DataFrame, residuals: pd.Series) -> pd.DataFrame:
    """
    Flags schools as Over/Under performing based on residual Standard Deviation.
    """
    std_dev = residuals.std()
    
    # Define thresholds (1.5 Sigma rule)
    high_thr = 1.5 * std_dev
    low_thr = -1.5 * std_dev
    
    # Create output dataframe
    out = df.copy()
    out["residual"] = residuals
    out["predicted"] = out["avg_att8"] - out["residual"]
    
    conditions = [
        out["residual"] >= high_thr,
        out["residual"] <= low_thr
    ]
    choices = ["🟢 Over-performing", "🔴 Under-performing"]
    
    out["outlier_status"] = np.select(conditions, choices, default="⚪ As Expected")
    
    return out

# =========================================================
# 3. MAIN DASHBOARD WRAPPER (This is what app.py calls)
# =========================================================

def run_school_outlier_analysis(df):
    """
    Runs a Linear Regression to predict Attainment 8 based on Deprivation.
    Returns the annotated dataframe and a summary dictionary.
    """
    # 1. Prepare Data
    # We need 'avg_att8' and 'idaci_decile'. 
    model_df = df.copy()
    
    # Check columns exist
    if "avg_att8" not in model_df.columns or "idaci_decile" not in model_df.columns:
        return pd.DataFrame(), {}

    # 2. Clean Data (Drop missing values)
    model_df = model_df.dropna(subset=["avg_att8", "idaci_decile"])
    
    if model_df.empty:
        return pd.DataFrame(), {}

    # 3. Feature Engineering: Convert IDACI Bands (Text) to Numeric (1-10)
    # This fixes the "Outlier code not working" issue
    idaci_map = {
        "0-10%": 1, "10-20%": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
        "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10
    }
    
    # Use map, but handle cases where it might already be numeric or 'Quintile'
    if model_df["idaci_decile"].dtype == 'object':
        model_df["poverty_index"] = model_df["idaci_decile"].map(idaci_map)
    else:
        # If it's already numeric (e.g. from Ofsted Quintiles 1-5), just ensure it's float
        model_df["poverty_index"] = pd.to_numeric(model_df["idaci_decile"], errors='coerce')

    # Drop rows where mapping failed (e.g. "Unknown" or "Total")
    model_df = model_df.dropna(subset=["poverty_index"])

    # 4. Run Regression (The "Data Mining" Part)
    X = model_df[["poverty_index"]]
    y = model_df["avg_att8"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 5. Calculate Outputs
    model_df["predicted_att8"] = model.predict(X)
    model_df["residual"] = model_df["avg_att8"] - model_df["predicted_att8"]
    
    # 6. Define Outliers (Standard Deviation Method)
    # Any school performing > 1.5 Std Devs away from expectation is an outlier
    std_dev = model_df["residual"].std()
    
    def classify_school(r):
        if r > (1.5 * std_dev): return "🟢 Over-performing"
        if r < (-1.5 * std_dev): return "🔴 Under-performing"
        return "⚪ As Expected"
        
    model_df["outlier_status"] = model_df["residual"].apply(classify_school)
    
    # 7. Summary Stats for Dashboard
    summary = {
        "r2": r2_score(y, model_df["predicted_att8"]),
        "intercept": model.intercept_, # The base score
        "poverty_coef": model.coef_[0], # How much score changes per poverty step
        "std_error": std_dev
    }
    
    return model_df, summary