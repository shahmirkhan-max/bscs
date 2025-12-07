"""
Streamlit dashboard for SPC-based analysis at LA level.

To run locally:
    pip install pandas numpy statsmodels plotly streamlit
    streamlit run app.py
"""

import os
from typing import Optional

import pandas as pd
import streamlit as st

from data_processing import build_analysis_dataset
from modelling import (
    prepare_modelling_data,
    run_ols_regression,
    summarise_model,
    model_diagnostics,
    classify_outliers_by_residual,
)
from visualisations import (
    scatter_with_trend,
    heatmap_region_fsm_attainment,
    bar_ethnicity_composition_by_region,
    boxplot_performance_by_school_type,
    residual_plot,
)


# ---------- Helpers ---------- #

def load_data_from_paths(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load from /data using the SPC filenames plus UK-HPI-cleaned.csv.
    """
    try:
        fsm_path = os.path.join(data_dir, "spc_pupils_fsm.csv")
        fsm6_path = os.path.join(data_dir, "spc_fsm6.csv")
        fsm_eth_path = os.path.join(data_dir, "spc_pupils_fsm_ethnicity_yrgp.csv")
        age_sex_path = os.path.join(data_dir, "spc_pupils_age_and_sex.csv")
        eth_lang_path = os.path.join(data_dir, "spc_pupils_ethnicity_and_language.csv")
        school_char_path = os.path.join(data_dir, "spc_school_characteristics.csv")
        uifsm_path = os.path.join(data_dir, "spc_uifsm.csv")
        cbm_path = os.path.join(data_dir, "spc_cbm.csv")

        hpi_path = os.path.join(data_dir, "UK-HPI-cleaned.csv")
        if not os.path.exists(hpi_path):
            hpi_path = None  # run fine without it

        df = build_analysis_dataset(
            fsm_path=fsm_path,
            fsm6_path=fsm6_path,
            fsm_ethnicity_path=fsm_eth_path,
            age_sex_path=age_sex_path,
            eth_lang_path=eth_lang_path,
            school_char_path=school_char_path,
            uifsm_path=uifsm_path,
            cbm_path=cbm_path,
            performance_path=None,     # hook in KS4/KS5 later
            hpi_path=hpi_path,
        )
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data from disk: {e}")
        return None



def build_data_from_uploads(
    fsm_file,
    fsm6_file,
    fsm_eth_file,
    age_sex_file,
    eth_lang_file,
    school_char_file,
    uifsm_file=None,
    cbm_file=None,
    performance_file=None,
) -> pd.DataFrame:
    """
    Build dataset from uploaded CSVs.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        def save(file, name):
            path = os.path.join(tmpdir, name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            return path

        fsm_path = save(fsm_file, "spc_pupils_fsm.csv")
        fsm6_path = save(fsm6_file, "spc_fsm6.csv")
        fsm_eth_path = save(fsm_eth_file, "spc_pupils_fsm_ethnicity_yrgp.csv")
        age_sex_path = save(age_sex_file, "spc_pupils_age_and_sex.csv")
        eth_lang_path = save(eth_lang_file, "spc_pupils_ethnicity_and_language.csv")
        school_char_path = save(school_char_file, "spc_school_characteristics.csv")
        uifsm_path = save(uifsm_file, "spc_uifsm.csv") if uifsm_file else None
        cbm_path = save(cbm_file, "spc_cbm.csv") if cbm_file else None
        performance_path = save(performance_file, "performance.csv") if performance_file else None

        df = build_analysis_dataset(
            fsm_path=fsm_path,
            fsm6_path=fsm6_path,
            fsm_ethnicity_path=fsm_eth_path,
            age_sex_path=age_sex_path,
            eth_lang_path=eth_lang_path,
            school_char_path=school_char_path,
            uifsm_path=uifsm_path,
            cbm_path=cbm_path,
            performance_path=performance_path,
        )

    return df


# ---------- Streamlit layout ---------- #

st.set_page_config(
    page_title="SPC – Socioeconomic Factors & School Performance",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Page 1: Data Explorer",
        "Page 2: Visualisations",
        "Page 3: Regression Results",
        "Page 4: Outlier Areas",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("Data source")

source = st.sidebar.radio("Use data from:", ["Local /data folder", "Upload CSVs"])

df_all = None

if source == "Local /data folder":
    df_all = load_data_from_paths("data")
    if df_all is None:
        st.sidebar.warning("Expected SPC CSVs not found in /data. Try uploads instead.")
else:
    st.sidebar.write("Upload SPC CSVs (LA-level):")
    fsm_file = st.sidebar.file_uploader("spc_pupils_fsm.csv", type="csv")
    fsm6_file = st.sidebar.file_uploader("spc_fsm6.csv", type="csv")
    fsm_eth_file = st.sidebar.file_uploader("spc_pupils_fsm_ethnicity_yrgp.csv", type="csv")
    age_sex_file = st.sidebar.file_uploader("spc_pupils_age_and_sex.csv", type="csv")
    eth_lang_file = st.sidebar.file_uploader("spc_pupils_ethnicity_and_language.csv", type="csv")
    school_char_file = st.sidebar.file_uploader("spc_school_characteristics.csv", type="csv")
    uifsm_file = st.sidebar.file_uploader("spc_uifsm.csv", type="csv")
    cbm_file = st.sidebar.file_uploader("spc_cbm.csv", type="csv")
    perf_file = st.sidebar.file_uploader("Performance file (optional)", type="csv")

    required_files = [fsm_file, fsm6_file, fsm_eth_file, age_sex_file, eth_lang_file, school_char_file]

    if all(required_files) and st.sidebar.button("Build dataset"):
        df_all = build_data_from_uploads(
            fsm_file=fsm_file,
            fsm6_file=fsm6_file,
            fsm_eth_file=fsm_eth_file,
            age_sex_file=age_sex_file,
            eth_lang_file=eth_lang_file,
            school_char_file=school_char_file,
            uifsm_file=uifsm_file,
            cbm_file=cbm_file,
            performance_file=perf_file,
        )

if df_all is None:
    st.info("No data loaded yet. Configure /data or upload CSVs via the sidebar.")
    st.stop()

# ---------- Common filters ---------- #

st.sidebar.markdown("---")
st.sidebar.header("Filters")

region_col = "region_name" if "region_name" in df_all.columns else None
la_col = "la_name" if "la_name" in df_all.columns else None
school_type_col = "school_type" if "school_type" in df_all.columns else None

df_filtered = df_all.copy()

if region_col:
    regions = sorted(df_filtered[region_col].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    df_filtered = df_filtered[df_filtered[region_col].isin(selected_regions)]

if la_col:
    las = sorted(df_filtered[la_col].dropna().unique().tolist())
    selected_las = st.sidebar.multiselect("Local authority", las)
    if selected_las:
        df_filtered = df_filtered[df_filtered[la_col].isin(selected_las)]

if school_type_col:
    school_types = sorted(df_filtered[school_type_col].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("School type (phase)", school_types)
    if selected_types:
        df_filtered = df_filtered[df_filtered[school_type_col].isin(selected_types)]

# ---------- Page 1: Data Explorer ---------- #

if page == "Page 1: Data Explorer":
    st.title("Page 1 – Data Explorer")

    st.write("### Preview of merged SPC dataset (LA × school_type)")
    st.dataframe(df_filtered.head(200))

    st.write("### Summary statistics")
    st.write(df_filtered.describe(include="all").transpose())

    st.write("### Download filtered data")
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="spc_merged_filtered.csv",
        mime="text/csv",
    )

# ---------- Page 2: Visualisations ---------- #

elif page == "Page 2: Visualisations":
    st.title("Page 2 – Visualisations")

    fsm_col = "fsm_percent" if "fsm_percent" in df_filtered.columns else None
    eal_col = "eal_percent" if "eal_percent" in df_filtered.columns else None
    perf_candidates = ["attainment8", "progress8", "performance_score"]
    perf_col = next((c for c in perf_candidates if c in df_filtered.columns), None)

    # Scatter FSM vs performance
    if fsm_col and perf_col:
        st.subheader("FSM% vs Performance")
        fig1 = scatter_with_trend(
            df_filtered.dropna(subset=[fsm_col, perf_col]),
            x=fsm_col,
            y=perf_col,
            color=region_col,
            hover_data=[la_col, school_type_col] if la_col and school_type_col else None,
            title=f"{fsm_col} vs {perf_col}",
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("FSM% or performance column not found yet. Once you add a performance file, this will populate.")

    # Scatter EAL vs performance
    if eal_col and perf_col:
        st.subheader("EAL% vs Performance")
        fig2 = scatter_with_trend(
            df_filtered.dropna(subset=[eal_col, perf_col]),
            x=eal_col,
            y=perf_col,
            color=region_col,
            hover_data=[la_col, school_type_col] if la_col and school_type_col else None,
            title=f"{eal_col} vs {perf_col}",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    if region_col and fsm_col and perf_col:
        st.subheader("Region × FSM × Performance heatmap")
        fig3 = heatmap_region_fsm_attainment(
            df_filtered.dropna(subset=[region_col, fsm_col, perf_col]),
            region_col=region_col,
            fsm_col=fsm_col,
            performance_col=perf_col,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Ethnicity composition
    if region_col:
        st.subheader("Ethnicity composition by region")
        eth_cols = [c for c in df_filtered.columns if c.startswith("eth_")]
        if eth_cols:
            fig4 = bar_ethnicity_composition_by_region(
                df_filtered.dropna(subset=[region_col]),
                region_col=region_col,
                ethnicity_cols=eth_cols,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No ethnicity columns found yet – check that spc_pupils_ethnicity_and_language is loaded.")

    # Boxplot performance by school type
    if perf_col and school_type_col:
        st.subheader("Performance by school type")
        fig5 = boxplot_performance_by_school_type(
            df_filtered.dropna(subset=[perf_col, school_type_col]),
            performance_col=perf_col,
            school_type_col=school_type_col,
        )
        st.plotly_chart(fig5, use_container_width=True)

# ---------- Page 3: Regression Results ---------- #

elif page == "Page 3: Regression Results":
    st.title("Page 3 – Regression Results")

    perf_candidates = ["attainment8", "progress8", "performance_score"]
    dep_options = [c for c in perf_candidates if c in df_filtered.columns]

    if not dep_options:
        st.warning("No performance column in the dataset yet. Upload / add a KS4/KS5 performance file.")
        st.stop()

    dep_var = st.selectbox("Dependent variable", dep_options)

    # Suggest default continuous vars
    default_cont = [c for c in ["fsm_percent", "fsm6_percent", "disadvantaged_percent",
                                "eal_percent", "pct_female", "cohort_size"]
                    if c in df_filtered.columns]

    cont_vars = st.multiselect(
        "Continuous predictors",
        [c for c in df_filtered.columns if df_filtered[c].dtype != "object" and c != dep_var],
        default=default_cont,
    )

    cat_candidates = [c for c in [school_type_col, "region_name"] if c and c in df_filtered.columns]
    cat_vars = st.multiselect(
        "Categorical predictors (incl. region fixed effects)",
        [c for c in df_filtered.columns if df_filtered[c].dtype == "object"],
        default=cat_candidates,
    )

    if st.button("Run regression"):
        X, y = prepare_modelling_data(
            df=df_filtered,
            dependent_var=dep_var,
            continuous_vars=cont_vars,
            categorical_vars=cat_vars,
        )
        model = run_ols_regression(X, y)
        summary_df = summarise_model(model)
        diag = model_diagnostics(model)

        st.subheader("Model summary (text)")
        st.text(model.summary().as_text())

        st.subheader("Coefficients table")
        st.dataframe(
            summary_df.style.format(
                {
                    "coef": "{:.3f}", "std_err": "{:.3f}", "t": "{:.2f}",
                    "p_value": "{:.3f}", "ci_lower": "{:.3f}", "ci_upper": "{:.3f}",
                }
            )
        )

        st.subheader("Goodness of fit")
        st.metric("R-squared", f"{model.rsquared:.3f}")
        st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")

        st.subheader("Residual diagnostics")
        fig_resid = residual_plot(diag["fitted"], diag["residuals"])
        st.plotly_chart(fig_resid, use_container_width=True)

        st.subheader("Top 20 observations by Cook's distance")
        cooks_df = diag["cooks_distance"].to_frame("cooks_distance").sort_values("cooks_distance", ascending=False).head(20)
        st.dataframe(cooks_df)

        st.subheader("Interpretation notes")
        st.markdown(
            """
            - **Significance**: Low p-values suggest a predictor is statistically associated with performance.
            - **Direction**: Positive coefficients → higher performance; negative → lower, holding others constant.
            - **Region fixed effects**: Region dummies capture systematic regional differences after controls.
            - **Caution**: Associations, not causal effects.
            """
        )

# ---------- Page 4: Outlier Areas ---------- #

elif page == "Page 4: Outlier Areas":
    st.title("Page 4 – Outlier Local Authorities / Area Types")

    perf_candidates = ["attainment8", "progress8", "performance_score"]
    dep_options = [c for c in perf_candidates if c in df_filtered.columns]

    if not dep_options:
        st.warning("No performance column in the dataset yet.")
        st.stop()

    dep_var = st.selectbox("Dependent variable", dep_options)

    cont_vars = [c for c in ["fsm_percent", "fsm6_percent", "disadvantaged_percent",
                             "eal_percent", "pct_female", "cohort_size"]
                 if c in df_filtered.columns]
    cat_vars = [c for c in [school_type_col, "region_name"] if c and c in df_filtered.columns]

    if st.button("Identify outliers"):
        X, y = prepare_modelling_data(
            df=df_filtered,
            dependent_var=dep_var,
            continuous_vars=cont_vars,
            categorical_vars=cat_vars,
        )
        model = run_ols_regression(X, y)
        diag = model_diagnostics(model)

        id_cols = [c for c in ["la_name", "region_name", "school_type"] if c in df_filtered.columns]
        outliers = classify_outliers_by_residual(
            df=df_filtered.loc[X.index],
            residuals=diag["residuals"],
            id_cols=id_cols,
        )

        st.subheader("Residual-based outlier classification")
        st.write("Positive residuals → over-performing vs expected; negative → under-performing.")
        st.dataframe(outliers)

        csv_outliers = outliers.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download outlier list as CSV",
            data=csv_outliers,
            file_name="spc_outliers_la.csv",
            mime="text/csv",
        )
