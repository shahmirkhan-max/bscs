# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="School Performance – Entrant & Subject Visualisations",
    layout="wide",
)

# -------------------------------------------------------------------
# Paths & data loading
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

DATASETS = {
    "Count by Occupation": BASE_DIR / "Count by Occupation.xlsx",
    "Count by Level of Qualification": BASE_DIR / "Count by Level of Qualification.xlsx",
    "HECoS Data": BASE_DIR / "HECoS Data.xlsx",
}


@st.cache_data
def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


# -------------------------------------------------------------------
# Helper for safe filters
# -------------------------------------------------------------------
def multiselect_filter(df, col, selected):
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)]


# -------------------------------------------------------------------
# App layout
# -------------------------------------------------------------------
st.title("Entrant & Subject Visualisations Dashboard")

st.markdown(
    """
This dashboard reads the pre-cleaned analysis files and produces visualisations.
The tabs below cover:

- **Count by Occupation** (socio-economic classification, parental education, IMD)
- **Count by Level of Qualification** (time series of entrants by level)
- **HECoS Subject Data** (subject-level entrants by CAH grouping)
- **Custom visualisations file** – upload any CSV/Excel (e.g. `visualizations.csv`)
  and build your own charts.
"""
)

# Load data up front (with basic error handling)
dfs = {}
for name, path in DATASETS.items():
    try:
        dfs[name] = load_excel(path)
    except Exception as e:
        st.sidebar.error(f"Could not load {name}: {e}")

tab_occ, tab_level, tab_hecos, tab_custom = st.tabs(
    [
        "Count by Occupation",
        "Count by Level of Qualification",
        "HECoS Subject Data",
        "Custom visualisations file",
    ]
)

# -------------------------------------------------------------------
# TAB 1 – Count by Occupation
# -------------------------------------------------------------------
with tab_occ:
    st.header("Count by Occupation (socio-economic classification, IMD, etc.)")

    df = dfs.get("Count by Occupation")
    if df is None:
        st.error("Could not load **Count by Occupation.xlsx**.")
    else:
        # Sidebar-like filters within the tab
        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            cat_marker_vals = sorted(df["Category Marker"].dropna().unique())
            cat_marker = col1.multiselect(
                "Category marker", options=cat_marker_vals, default=cat_marker_vals
            )

            he_country_vals = sorted(df["Country of HE provider"].dropna().unique())
            he_country = col2.multiselect(
                "Country of HE provider",
                options=he_country_vals,
                default=["England"],
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col3.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["All"],
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col4.multiselect(
                "Level of study",
                options=level_vals,
                default=["All", "First degree"],
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Category Marker", cat_marker)
        df_f = multiselect_filter(df_f, "Country of HE provider", he_country)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_marker)
        df_f = multiselect_filter(df_f, "Level of study", level_of_study)
        if years:
            df_f = df_f[df_f["Academic Year"].isin(years)]

        metric = st.radio("Metric", ["Number", "Percentage"], horizontal=True)
        if metric == "Percentage":
            df_f = df_f.dropna(subset=["Percentage"])

        st.markdown("### Visualisations")

        view_type = st.radio(
            "View",
            ["By Category (latest year)", "Trend over time (by category)"],
            horizontal=True,
        )

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            # Aggregate
            metric_col = metric
            agg = (
                df_f.groupby(["Academic Year", "Category"], as_index=False)[metric_col]
                .sum()
            )

            if view_type == "By Category (latest year)":
                latest_year = sorted(agg["Academic Year"].unique())[-1]
                latest = agg[agg["Academic Year"] == latest_year]
                latest = latest.sort_values(metric_col, ascending=False)

                st.caption(f"Latest year in filtered data: **{latest_year}**")
                st.bar_chart(
                    data=latest.set_index("Category")[metric_col],
                    use_container_width=True,
                )

            else:  # Trend over time
                pivot = agg.pivot(
                    index="Academic Year", columns="Category", values=metric_col
                ).sort_index()
                st.line_chart(pivot, use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 2 – Count by Level of Qualification
# -------------------------------------------------------------------
with tab_level:
    st.header("Count by Level of Qualification")

    df = dfs.get("Count by Level of Qualification")
    if df is None:
        st.error("Could not load **Count by Level of Qualification.xlsx**.")
    else:
        with st.expander("Filters", expanded=True):
            col1, col2 = st.columns(2)

            level_vals = sorted(df["Level of qualification"].dropna().unique())
            level_sel = col1.multiselect(
                "Level of qualification",
                options=level_vals,
                default=[lv for lv in level_vals if lv != "Total"] or level_vals,
            )

            year_vals = sorted(df["Academic year"].dropna().unique())
            years = col2.multiselect(
                "Academic year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Level of qualification", level_sel)
        if years:
            df_f = df_f[df_f["Academic year"].isin(years)]

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            st.markdown("### Time series of entrants by level of qualification")

            # Pivot for time series: index = year, columns = level, values = Number
            pivot = (
                df_f.pivot(
                    index="Academic year",
                    columns="Level of qualification",
                    values="Number",
                )
                .sort_index()
            )

            st.line_chart(pivot, use_container_width=True)

            st.markdown("### Composition by level (latest year)")

            latest_year = sorted(df_f["Academic year"].unique())[-1]
            latest = df_f[df_f["Academic year"] == latest_year]
            latest = latest.groupby("Level of qualification", as_index=False)["Number"].sum()
            latest = latest.sort_values("Number", ascending=False)

            st.caption(f"Latest year in filtered data: **{latest_year}**")
            st.bar_chart(
                data=latest.set_index("Level of qualification")["Number"],
                use_container_width=True,
            )

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 3 – HECoS Subject Data
# -------------------------------------------------------------------
with tab_hecos:
    st.header("HECoS / CAH subject data")

    df = dfs.get("HECoS Data")
    if df is None:
        st.error("Could not load **HECoS Data.xlsx**.")
    else:
        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            cah_marker_vals = sorted(df["CAH level marker"].dropna().unique())
            cah_marker = col1.multiselect(
                "CAH level marker",
                options=cah_marker_vals,
                default=cah_marker_vals,
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col2.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["Entrant", "All"],
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col3.multiselect(
                "Level of study",
                options=level_vals,
                default=["All", "First degree"],
            )

            mode_vals = sorted(df["Mode of study"].dropna().unique())
            mode_of_study = col4.multiselect(
                "Mode of study", options=mode_vals, default=["All", "Full-time"]
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "CAH level marker", cah_marker)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_marker)
        df_f = multiselect_filter(df_f, "Level of study", level_of_study)
        df_f = multiselect_filter(df_f, "Mode of study", mode_of_study)
        if years:
            df_f = df_f[df_f["Academic Year"].isin(years)]

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            st.markdown("### Entrants by subject (latest year)")

            latest_year = sorted(df_f["Academic Year"].unique())[-1]
            latest = df_f[df_f["Academic Year"] == latest_year]
            latest = (
                latest.groupby("CAH level subject", as_index=False)["Number"]
                .sum()
                .sort_values("Number", ascending=False)
            )

            st.caption(f"Latest year in filtered data: **{latest_year}**")

            # Limit to top N for readability
            top_n = st.slider("Top N subjects", min_value=5, max_value=40, value=20)
            latest_top = latest.head(top_n)

            st.bar_chart(
                latest_top.set_index("CAH level subject")["Number"],
                use_container_width=True,
            )

            st.markdown("### Trend for a selected subject over time")
            subject_sel = st.selectbox(
                "Subject", options=sorted(df_f["CAH level subject"].unique())
            )

            subj_ts = (
                df_f[df_f["CAH level subject"] == subject_sel]
                .groupby("Academic Year", as_index=False)["Number"]
                .sum()
                .sort_values("Academic Year")
            )

            subj_ts = subj_ts.set_index("Academic Year")
            st.line_chart(subj_ts["Number"], use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 4 – Custom visualisations (e.g. visualizations.csv)
# -------------------------------------------------------------------
with tab_custom:
    st.header("Custom visualisations from another file")

    st.markdown(
        """
Upload any **CSV** or **Excel** file (for example, your `visualizations.csv`
that comes out of the regression script). Then choose the X/Y axes and chart type.
"""
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )

    if uploaded is not None:
        # Auto-detect format
        if uploaded.name.lower().endswith(".csv"):
            df_custom = pd.read_csv(uploaded)
        else:
            df_custom = pd.read_excel(uploaded)

        st.subheader("Data preview")
        st.dataframe(df_custom.head(), use_container_width=True, height=250)

        numeric_cols = df_custom.select_dtypes(include="number").columns.tolist()
        all_cols = df_custom.columns.tolist()
        non_numeric_cols = [c for c in all_cols if c not in numeric_cols]

        if not numeric_cols:
            st.warning("No numeric columns found – need at least one for Y-axis.")
        else:
            st.markdown("### Build a chart")

            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox(
                "X-axis",
                options=all_cols,
                index=0,
            )
            y_col = col2.selectbox(
                "Y-axis (numeric)",
                options=numeric_cols,
                index=0,
            )
            chart_type = col3.selectbox(
                "Chart type", options=["Bar", "Line", "Scatter"], index=0
            )

            # Optional group-by for aggregation
            group = st.checkbox(
                "Aggregate by X (sum of Y)", value=True, help="Useful for bar/line charts."
            )

            df_plot = df_custom.copy()

            if group:
                df_plot = (
                    df_plot.groupby(x_col, as_index=False)[y_col]
                    .sum()
                    .sort_values(x_col)
                )

            st.write("#### Chart")

            if chart_type == "Bar":
                st.bar_chart(
                    df_plot.set_index(x_col)[y_col],
                    use_container_width=True,
                )
            elif chart_type == "Line":
                st.line_chart(
                    df_plot.set_index(x_col)[y_col],
                    use_container_width=True,
                )
            else:  # Scatter
                # For scatter we’ll use st.pyplot via pandas plot
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.scatter(df_plot[x_col], df_plot[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig, use_container_width=True)

    else:
        st.info("Upload a CSV/Excel file to start building custom visualisations.")
