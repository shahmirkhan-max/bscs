import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple
from math import erf, sqrt


# ============================================================
# Helper functions – general cleaning
# ============================================================

def extract_closing_year(year_value) -> Optional[int]:
    """
    Convert codes like 201516 or 202122 into closing year (2016, 2022).
    """
    if pd.isna(year_value):
        return None
    s = str(year_value).strip()

    # Already a 4-digit year
    if len(s) == 4 and s.isdigit():
        return int(s)

    # Academic code 'YYYYYY'
    if len(s) == 6 and s.isdigit():
        try:
            start_year = int(s[:4])
            return start_year + 1
        except ValueError:
            return None

    try:
        val = int(float(s))
        if 1900 <= val <= 2100:
            return val
    except ValueError:
        pass

    return None


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase + snake_case column names.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def fix_year_column(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """
    Add 'year' column from time_period / academic_year.
    """
    df = df.copy()
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return df
    col = existing[0]
    df["year"] = df[col].apply(extract_closing_year)
    return df


def filter_local_authority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep geographic_level == 'Local authority'.
    """
    df = df.copy()
    if "geographic_level" in df.columns:
        mask = df["geographic_level"].str.lower().str.replace(" ", "_") == "local_authority"
        df = df[mask]
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_spc_from_upload(upload) -> pd.DataFrame:
    """
    Read a CSV from Streamlit file_uploader and apply generic SPC cleaning.
    """
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period"])
    df = filter_local_authority(df)
    return df


def common_key_cols(df: pd.DataFrame) -> List[str]:
    """
    Common LA-level keys.
    """
    candidates = [
        "year",
        "country_code", "country_name",
        "region_code", "region_name",
        "la_code", "la_name",
        "school_type",
    ]
    return [c for c in candidates if c in df.columns]


def safe_left_merge(left: pd.DataFrame, right: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Left join using intersection of common key columns.
    """
    if right is None or right.empty:
        return left

    left_keys = common_key_cols(left)
    right_keys = common_key_cols(right)
    keys = [k for k in left_keys if k in right_keys]
    if not keys:
        return left
    return left.merge(right, on=keys, how="left")


# ============================================================
# SPC cleaning functions – expect uploads / DataFrames
# ============================================================

def clean_spc_school_characteristics(upload) -> pd.DataFrame:
    """
    Build LA × school_type cohorts.
    """
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period"])
    df = filter_local_authority(df)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    cols_keep = common_key_cols(df) + ["headcount_of_pupils"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep]

    if "headcount_of_pupils" in df.columns:
        group_cols = [c for c in common_key_cols(df) if c != "country_code"]
        df = (
            df.groupby(group_cols, as_index=False)["headcount_of_pupils"]
            .sum()
            .rename(columns={"headcount_of_pupils": "cohort_size"})
        )

    return df


def clean_spc_pupils_fsm(upload) -> pd.DataFrame:
    """
    FSM% at LA × school_type.
    """
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm" in df.columns:
        eligible = df["fsm"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm_percent"})
    )
    return out


def clean_spc_fsm6(upload) -> pd.DataFrame:
    """
    FSM6% at LA × school_type.
    """
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm6_eligibility" in df.columns:
        eligible = df["fsm6_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm6_percent"})
    )
    return out


def clean_spc_pupils_fsm_ethnicity_yrgp(upload) -> pd.DataFrame:
    """
    Build disadvantaged_percent.
    """
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm_eligibility" in df.columns:
        eligible = df["fsm_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    if "characteristic" in df.columns:
        total = df["characteristic"].astype(str).str.lower() == "total"
        if total.any():
            df = df[total]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "disadvantaged_percent"})
    )
    return out


def clean_spc_pupils_ethnicity_and_language(upload) -> pd.DataFrame:
    """
    Ethnicity proportions (eth_*) + eal_percent.
    """
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    keys = common_key_cols(df)

    # Ethnicity wide
    if "ethnicity_minor" in df.columns:
        eth = (
            df.groupby(keys + ["ethnicity_minor"], as_index=False)["percent_of_pupils"]
            .mean()
        )
        eth_wide = eth.pivot_table(
            index=keys,
            columns="ethnicity_minor",
            values="percent_of_pupils",
            aggfunc="first",
        )
        eth_wide.columns = [
            "eth_" + str(c).lower().replace(" ", "_").replace("/", "_")
            for c in eth_wide.columns
        ]
        eth_wide = eth_wide.reset_index()
    else:
        eth_wide = pd.DataFrame(columns=keys)

    # EAL
    if "language" in df.columns:
        lang = (
            df.groupby(keys + ["language"], as_index=False)["percent_of_pupils"]
            .mean()
        )
        mask_eal = lang["language"].astype(str).str.contains("other than english", case=False, na=False)
        eal = (
            lang[mask_eal]
            .groupby(keys, as_index=False)["percent_of_pupils"]
            .sum()
            .rename(columns={"percent_of_pupils": "eal_percent"})
        )
    else:
        eal = pd.DataFrame(columns=keys + ["eal_percent"])

    merged = pd.merge(eth_wide, eal, on=keys, how="outer")
    return merged


def clean_spc_pupils_age_and_sex(upload) -> pd.DataFrame:
    """
    pct_male / pct_female at LA × school_type.
    """
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["headcount"])

    keys = common_key_cols(df)
    if "sex" not in df.columns or "headcount" not in df.columns:
        return df.iloc[0:0][keys]

    g = (
        df.groupby(keys + ["sex"], as_index=False)["headcount"]
        .sum()
    )
    pivot = g.pivot_table(
        index=keys,
        columns="sex",
        values="headcount",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    cols = list(pivot.columns)
    male_col = next((c for c in cols if str(c).lower().startswith("male")), None)
    female_col = next((c for c in cols if str(c).lower().startswith("female")), None)

    if male_col is None or female_col is None:
        pivot["pct_male"] = np.nan
        pivot["pct_female"] = np.nan
    else:
        total = pivot[male_col] + pivot[female_col]
        pivot["pct_male"] = np.where(total > 0, 100 * pivot[male_col] / total, np.nan)
        pivot["pct_female"] = np.where(total > 0, 100 * pivot[female_col] / total, np.nan)

    return pivot[keys + ["pct_male", "pct_female"]]


def clean_spc_uifsm(upload) -> pd.DataFrame:
    """
    UIFSM % at LA × school_type.
    """
    if upload is None:
        return None

    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percentage"])

    if "characteristic" in df.columns:
        mask_total = df["characteristic"].astype(str).str.lower() == "total"
        if mask_total.any():
            df = df[mask_total]

    keys = common_key_cols(df)
    if "percentage" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percentage"]
        .mean()
        .rename(columns={"percentage": "uifsm_percent_total"})
    )
    return out


def clean_performance_data(upload) -> pd.DataFrame:
    """
    Clean KS4/KS5 performance data from upload.
    """
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period", "academic_year"])

    perf_cols = ["attainment8", "progress8", "performance_score"]
    df = ensure_numeric(df, perf_cols)
    return df


def clean_uk_hpi(upload) -> pd.DataFrame:
    """
    Clean UK-HPI-cleaned.csv from upload to region-year avg house price.
    """
    if upload is None:
        return None

    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    df = df.rename(
        columns={
            "regionname": "region_name",
            "areacode": "region_code",
            "averageprice": "avg_house_price",
        }
    )

    group_cols = [c for c in ["year", "region_code", "region_name"] if c in df.columns]
    out = (
        df.groupby(group_cols, as_index=False)["avg_house_price"]
        .mean()
    )
    return out


# ============================================================
# Modelling utilities (NumPy OLS)
# ============================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def prepare_modelling_data(
    df: pd.DataFrame,
    dependent_var: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X (with intercept) and y for OLS.
    """
    cols_needed = [dependent_var] + continuous_vars + categorical_vars
    df_model = df[cols_needed].dropna(subset=cols_needed).copy()

    X_cont = df_model[continuous_vars].astype(float)
    X_cat = pd.get_dummies(df_model[categorical_vars], drop_first=True, dummy_na=False)

    # Add intercept
    X = pd.concat(
        [pd.Series(1.0, index=df_model.index, name="const"), X_cont, X_cat],
        axis=1
    )
    y = df_model[dependent_var].astype(float)

    return X, y


def run_ols_numpy(X: pd.DataFrame, y: pd.Series):
    """
    Run OLS using NumPy (no statsmodels).
    Returns dict with:
        beta, se, t, p, y_hat, residuals, r2, adj_r2, n, k
    """
    X_mat = X.values
    y_vec = y.values.reshape(-1, 1)

    n, k = X_mat.shape  # n obs, k params

    # (X'X)^-1 X'y
    XtX = X_mat.T @ X_mat
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_mat.T @ y_vec
    beta = XtX_inv @ XtY  # (k, 1)

    y_hat = X_mat @ beta
    residuals = y_vec - y_hat

    # Sum of squares
    ssr = float((residuals.T @ residuals))
    sst = float(((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean())))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan

    # Variance of residuals (sigma^2)
    sigma2 = ssr / (n - k)

    # Var(beta)
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)

    # t-stats and approx p-values using normal approximation
    t_stats = beta / se
    t_stats_flat = t_stats.flatten()
    p_vals = np.array([2 * (1 - normal_cdf(abs(t))) for t in t_stats_flat]).reshape(-1, 1)

    # Wrap as pandas Series
    beta_s = pd.Series(beta.flatten(), index=X.columns, name="coef")
    se_s = pd.Series(se.flatten(), index=X.columns, name="std_err")
    t_s = pd.Series(t_stats_flat, index=X.columns, name="t_stat")
    p_s = pd.Series(p_vals.flatten(), index=X.columns, name="p_value")

    results = {
        "beta": beta_s,
        "se": se_s,
        "t": t_s,
        "p": p_s,
        "y_hat": pd.Series(y_hat.flatten(), index=X.index),
        "residuals": pd.Series(residuals.flatten(), index=X.index),
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
    }
    return results


def cooks_like_influence(X: pd.DataFrame, residuals: pd.Series, sigma2: float):
    """
    A simple influence measure using leverage (diagonal of hat matrix) + residuals.
    h_ii = x_i' (X'X)^-1 x_i
    Approx Cook's: (resid_i^2 / (k * sigma2)) * (h_ii / (1 - h_ii)^2)
    """
    X_mat = X.values
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    hat_matrix = X_mat @ XtX_inv @ X_mat.T
    h = np.diag(hat_matrix)
    k = X_mat.shape[1]
    resid = residuals.values

    cooks = (resid**2 / (k * sigma2)) * (h / (1 - h)**2)
    return pd.Series(cooks, index=X.index, name="cooks_distance")


# ============================================================
# Visualisation helpers
# ============================================================

def scatter_with_trend(df, x, y, color=None, hover=None, title="", xlab=None, ylab=None):
    """
    Scatter plot with a manually computed regression line (overall).
    """
    df_plot = df.dropna(subset=[x, y]).copy()
    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        color=color,
        hover_data=hover,
        title=title,
    )

    # Fit simple linear regression line globally (not by group)
    x_vals = df_plot[x].values.astype(float)
    y_vals = df_plot[y].values.astype(float)
    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="OLS line",
                line=dict(dash="dash")
            )
        )

    fig.update_layout(
        xaxis_title=xlab or x,
        yaxis_title=ylab or y,
        legend_title_text=color if color else "",
    )
    return fig


def heatmap_region_fsm_perf(df, region_col, fsm_col, perf_col):
    df = df.dropna(subset=[region_col, fsm_col, perf_col]).copy()
    if df.empty:
        return go.Figure()

    df["fsm_bin"] = pd.qcut(df[fsm_col], q=4, duplicates="drop")
    df["fsm_bin"] = df["fsm_bin"].astype(str)

    pivot = (
        df.groupby([region_col, "fsm_bin"], as_index=False)[perf_col]
        .mean()
        .pivot(index=region_col, columns="fsm_bin", values=perf_col)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorbar=dict(title=perf_col),
        )
    )
    fig.update_layout(
        title="Region × FSM × Performance",
        xaxis_title="FSM quantile group",
        yaxis_title="Region",
    )
    return fig


def boxplot_perf_by_school_type(df, perf_col, school_type_col):
    df = df.dropna(subset=[perf_col, school_type_col]).copy()
    if df.empty:
        return go.Figure()

    fig = px.box(
        df,
        x=school_type_col,
        y=perf_col,
        points="outliers",
        title="Performance by school type",
    )
    fig.update_layout(
        xaxis_title="School type / phase",
        yaxis_title=perf_col,
        xaxis=dict(tickangle=45),
    )
    return fig


def residual_plot(fitted, residuals):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(opacity=0.6),
        )
    )
    fig.add_hline(y=0, line=dict(dash="dash"), annotation_text="0")
    fig.update_layout(
        title="Residuals vs fitted values",
        xaxis_title="Fitted values",
        yaxis_title="Residuals",
    )
    return fig


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="SPC Regression Explorer (No statsmodels)", layout="wide")
st.title("Data Mining in School Performance – SPC Upload & Regression (NumPy OLS)")

st.markdown(
    """
Upload the required SPC files, performance file, and (optionally) UK-HPI data.

**Required SPC files (CSV):**

- `spc_pupils_fsm`
- `spc_fsm6`
- `spc_pupils_fsm_ethnicity_yrgp`
- `spc_pupils_age_and_sex`
- `spc_pupils_ethnicity_and_language`
- `spc_school_characteristics`

**Required for regression:**

- Performance file (KS4/KS5 – contains Attainment8/Progress8 etc.)

**Optional:**

- `spc_uifsm`
- UK-HPI cleaned file (region house prices)
"""
)

# --- Upload widgets ---

col1, col2 = st.columns(2)

with col1:
    fsm_file = st.file_uploader("spc_pupils_fsm.csv", type="csv")
    fsm6_file = st.file_uploader("spc_fsm6.csv", type="csv")
    fsm_eth_file = st.file_uploader("spc_pupils_fsm_ethnicity_yrgp.csv", type="csv")
    age_sex_file = st.file_uploader("spc_pupils_age_and_sex.csv", type="csv")

with col2:
    eth_lang_file = st.file_uploader("spc_pupils_ethnicity_and_language.csv", type="csv")
    school_char_file = st.file_uploader("spc_school_characteristics.csv", type="csv")
    uifsm_file = st.file_uploader("spc_uifsm.csv (optional)", type="csv")
    hpi_file = st.file_uploader("UK-HPI-cleaned.csv (optional)", type="csv")

perf_file = st.file_uploader("Performance file (KS4/KS5) – required for regression", type="csv")

required = [fsm_file, fsm6_file, fsm_eth_file, age_sex_file, eth_lang_file, school_char_file, perf_file]

if not all(required):
    st.warning("Please upload all required SPC files and the performance file to run the analysis.")
    st.stop()

if st.button("Run cleaning & regression"):
    # ---------------- Build analysis dataset ----------------
    with st.spinner("Cleaning and merging uploaded files..."):

        school_char_df = clean_spc_school_characteristics(school_char_file)
        fsm_df = clean_spc_pupils_fsm(fsm_file)
        fsm6_df = clean_spc_fsm6(fsm6_file)
        fsm_eth_df = clean_spc_pupils_fsm_ethnicity_yrgp(fsm_eth_file)
        age_sex_df = clean_spc_pupils_age_and_sex(age_sex_file)
        eth_lang_df = clean_spc_pupils_ethnicity_and_language(eth_lang_file)
        uifsm_df = clean_spc_uifsm(uifsm_file)
        perf_df = clean_performance_data(perf_file)
        hpi_df = clean_uk_hpi(hpi_file)

        merged = school_char_df.copy()
        merged = safe_left_merge(merged, fsm_df)
        merged = safe_left_merge(merged, fsm6_df)
        merged = safe_left_merge(merged, fsm_eth_df)
        merged = safe_left_merge(merged, age_sex_df)
        merged = safe_left_merge(merged, eth_lang_df)
        merged = safe_left_merge(merged, uifsm_df)
        merged = safe_left_merge(merged, perf_df)
        merged = safe_left_merge(merged, hpi_df)

        # Fallback for disadvantaged_percent
        if "disadvantaged_percent" not in merged.columns:
            if "fsm6_percent" in merged.columns:
                merged["disadvantaged_percent"] = merged["fsm6_percent"]
            elif "fsm_percent" in merged.columns:
                merged["disadvantaged_percent"] = merged["fsm_percent"]

    st.success(f"Dataset built – {merged.shape[0]} rows, {merged.shape[1]} columns")

    # ----------------- Overview tab -----------------
    tab_overview, tab_viz, tab_reg = st.tabs(["Data overview", "Visualisations", "Regression & diagnostics"])

    with tab_overview:
        st.subheader("Preview of merged dataset")
        st.dataframe(merged.head(200))

        st.subheader("Summary statistics (numeric)")
        st.write(merged.describe().T)

    # ----------------- Visualisations tab -----------------
    with tab_viz:
        st.subheader("Exploratory charts")

        region_col = "region_name" if "region_name" in merged.columns else None
        la_col = "la_name" if "la_name" in merged.columns else None
        school_type_col = "school_type" if "school_type" in merged.columns else None

        perf_candidates = ["attainment8", "progress8", "performance_score"]
        perf_col = next((c for c in perf_candidates if c in merged.columns), None)

        fsm_col = "fsm_percent" if "fsm_percent" in merged.columns else None
        eal_col = "eal_percent" if "eal_percent" in merged.columns else None

        # FSM vs performance
        if fsm_col and perf_col:
            st.markdown("**FSM% vs performance**")
            df_plot = merged.dropna(subset=[fsm_col, perf_col])
            fig = scatter_with_trend(
                df_plot,
                x=fsm_col,
                y=perf_col,
                color=region_col,
                hover=[la_col, school_type_col] if la_col and school_type_col else None,
                title=f"{fsm_col} vs {perf_col}",
                xlab="FSM (%)",
                ylab=perf_col,
            )
            st.plotly_chart(fig, use_container_width=True)

        # EAL vs performance
        if eal_col and perf_col:
            st.markdown("**EAL% vs performance**")
            df_plot = merged.dropna(subset=[eal_col, perf_col])
            fig = scatter_with_trend(
                df_plot,
                x=eal_col,
                y=perf_col,
                color=region_col,
                hover=[la_col, school_type_col] if la_col and school_type_col else None,
                title=f"{eal_col} vs {perf_col}",
                xlab="EAL (%)",
                ylab=perf_col,
            )
            st.plotly_chart(fig, use_container_width=True)

        # House price vs performance
        if "avg_house_price" in merged.columns and perf_col:
            st.markdown("**Average house price vs performance**")
            df_plot = merged.dropna(subset=["avg_house_price", perf_col])
            fig = scatter_with_trend(
                df_plot,
                x="avg_house_price",
                y=perf_col,
                color=region_col,
                hover=[la_col, school_type_col] if la_col and school_type_col else None,
                title=f"Average house price vs {perf_col}",
                xlab="Average house price (£)",
                ylab=perf_col,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        if region_col and fsm_col and perf_col:
            st.markdown("**Region × FSM × performance heatmap**")
            fig = heatmap_region_fsm_perf(merged, region_col, fsm_col, perf_col)
            st.plotly_chart(fig, use_container_width=True)

        # Boxplot by school type
        if perf_col and school_type_col:
            st.markdown("**Performance by school type**")
            fig = boxplot_perf_by_school_type(merged, perf_col, school_type_col)
            st.plotly_chart(fig, use_container_width=True)

    # ----------------- Regression tab -----------------
    with tab_reg:
        st.subheader("Regression setup")

        perf_candidates = ["attainment8", "progress8", "performance_score"]
        dep_var = st.selectbox(
            "Dependent variable (performance metric)",
            [c for c in perf_candidates if c in merged.columns],
        )

        default_cont = [
            col for col in [
                "fsm_percent",
                "fsm6_percent",
                "disadvantaged_percent",
                "eal_percent",
                "pct_female",
                "cohort_size",
                "avg_house_price",
            ] if col in merged.columns
        ]

        continuous_vars = st.multiselect(
            "Continuous predictors",
            [c for c in merged.columns if merged[c].dtype != "object" and c != dep_var],
            default=default_cont,
        )

        cat_candidates = [c for c in ["school_type", "region_name"] if c in merged.columns]
        categorical_vars = st.multiselect(
            "Categorical predictors (fixed effects)",
            [c for c in merged.columns if merged[c].dtype == "object"],
            default=cat_candidates,
        )

        if st.button("Run OLS (NumPy)"):
            with st.spinner("Fitting OLS model..."):
                X, y = prepare_modelling_data(merged, dep_var, continuous_vars, categorical_vars)
                results = run_ols_numpy(X, y)

            st.success("Model fitted.")

            st.subheader("Model fit statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R-squared", f"{results['r2']:.3f}")
            with col_b:
                st.metric("Adjusted R-squared", f"{results['adj_r2']:.3f}")

            coef_df = pd.concat(
                [results["beta"], results["se"], results["t"], results["p"]],
                axis=1
            )
            coef_df.columns = ["coef", "std_err", "t_stat", "p_value"]

            st.subheader("Coefficient table")
            st.dataframe(
                coef_df.style.format(
                    {"coef": "{:.3f}", "std_err": "{:.3f}", "t_stat": "{:.2f}", "p_value": "{:.3f}"}
                )
            )

            # Diagnostics
            resid = results["residuals"]
            fitted = results["y_hat"]
            sigma2 = float((resid ** 2).sum() / (results["n"] - results["k"]))
            cooks_series = cooks_like_influence(X, resid, sigma2)

            st.subheader("Residual diagnostics")
            fig_res = residual_plot(fitted, resid)
            st.plotly_chart(fig_res, use_container_width=True)

            st.subheader("Top 15 observations by influence (Cook's-like distance)")
            diag_df = pd.DataFrame({
                "fitted": fitted,
                "residual": resid,
                "cooks_distance": cooks_series,
            })
            st.dataframe(diag_df.sort_values("cooks_distance", ascending=False).head(15))
