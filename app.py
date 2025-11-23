"""
UK HPI – Streamlit Explorer (no Plotly)

Usage:
- Put this file (app.py) in the same folder as: UK-HPI-full-file-2025-09.csv
- Install deps: pip install streamlit pandas
- Run: streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_FILE = "UK-HPI-full-file-2025-09.csv"


# ---------------------------
# Data loading / preprocessing
# ---------------------------

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standard UK HPI columns:
    # Date, RegionName, RegionCode, AveragePrice, Index, IndexSA,
    # SalesVolume, PropertyType, NewBuild, Tenure, RecordStatus

    # Date → datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        # Try to auto-detect a date column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.rename(columns={date_cols[0]: "Date"})
        else:
            st.warning("No 'Date' column found – time series plots will be limited.")

    # Clean standard numeric fields
    for col in ["AveragePrice", "Index", "IndexSA", "SalesVolume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make some columns categorical if present
    for col in ["RegionName", "PropertyType", "NewBuild", "Tenure", "RecordStatus"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Filter to 'A' (accepted) records if RecordStatus exists
    if "RecordStatus" in df.columns:
        df = df[df["RecordStatus"] == "A"].copy()

    return df


def get_latest_date(df: pd.DataFrame):
    if "Date" in df.columns:
        return df["Date"].max()
    return None


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="UK HPI Explorer",
        layout="wide",
    )

    st.title("UK House Price Index – Interactive Explorer")

    # Check file exists
    if not Path(DATA_FILE).exists():
        st.error(
            f"Data file '{DATA_FILE}' not found.\n\n"
            "Place 'UK-HPI-full-file-2025-09.csv' in the same folder as app.py "
            "or change DATA_FILE at the top of the script."
        )
        return

    df = load_data(DATA_FILE)

    if df.empty:
        st.error("Loaded UK HPI data is empty. Check the CSV file.")
        return

    st.markdown(
        "This app visualises the UK House Price Index (HPI) full file, "
        "with filters for region, property type, tenure and more."
    )

    # -----------------------
    # Sidebar filters
    # -----------------------
    st.sidebar.header("Filters")

    # Region filter
    if "RegionName" in df.columns:
        regions = sorted(df["RegionName"].dropna().unique().tolist())
        selected_regions = st.sidebar.multiselect(
            "Regions",
            options=regions,
            default=regions if len(regions) <= 5 else regions[:5],
        )
    else:
        selected_regions = None

    # Property type filter
    if "PropertyType" in df.columns:
        prop_types = sorted(df["PropertyType"].dropna().unique().tolist())
        selected_prop_types = st.sidebar.multiselect(
            "Property type",
            options=prop_types,
            default=prop_types,
        )
    else:
        selected_prop_types = None

    # New build filter
    if "NewBuild" in df.columns:
        newbuild_vals = sorted(df["NewBuild"].dropna().unique().tolist())
        selected_newbuild = st.sidebar.multiselect(
            "New build?",
            options=newbuild_vals,
            default=newbuild_vals,
        )
    else:
        selected_newbuild = None

    # Tenure filter
    if "Tenure" in df.columns:
        tenure_vals = sorted(df["Tenure"].dropna().unique().tolist())
        selected_tenure = st.sidebar.multiselect(
            "Tenure",
            options=tenure_vals,
            default=tenure_vals,
        )
    else:
        selected_tenure = None

    # Date range
    if "Date" in df.columns:
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        date_range = st.sidebar.slider(
            "Date range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        )
    else:
        date_range = None

    # Apply filters
    df_filt = df.copy()

    if "RegionName" in df_filt.columns and selected_regions:
        df_filt = df_filt[df_filt["RegionName"].isin(selected_regions)]

    if "PropertyType" in df_filt.columns and selected_prop_types:
        df_filt = df_filt[df_filt["PropertyType"].isin(selected_prop_types)]

    if "NewBuild" in df_filt.columns and selected_newbuild:
        df_filt = df_filt[df_filt["NewBuild"].isin(selected_newbuild)]

    if "Tenure" in df_filt.columns and selected_tenure:
        df_filt = df_filt[df_filt["Tenure"].isin(selected_tenure)]

    if "Date" in df_filt.columns and date_range:
        start, end = date_range
        df_filt = df_filt[(df_filt["Date"] >= start) & (df_filt["Date"] <= end)]

    if df_filt.empty:
        st.warning("No data left after applying filters.")
        return

    # -----------------------
    # Layout: tabs
    # -----------------------
    tab_overview, tab_time, tab_region, tab_property = st.tabs(
        ["Overview", "Time series", "Regional comparison", "Property type analysis"]
    )

    # -----------------------
    # Overview tab
    # -----------------------
    with tab_overview:
        st.subheader("Overview")

        latest = get_latest_date(df_filt)
        col1, col2, col3 = st.columns(3)

        if "AveragePrice" in df_filt.columns:
            with col1:
                st.metric(
                    "Overall average price (filtered)",
                    f"£{df_filt['AveragePrice'].mean():,.0f}",
                )

        if "SalesVolume" in df_filt.columns:
            with col2:
                st.metric(
                    "Average monthly sales volume (filtered)",
                    f"{df_filt['SalesVolume'].mean():,.0f}",
                )

        if latest is not None:
            with col3:
                st.metric("Latest date in filtered data", latest.strftime("%Y-%m"))

        st.markdown("### Latest month – average price by region")
        if latest is not None and "RegionName" in df_filt.columns and "AveragePrice" in df_filt.columns:
            latest_df = df_filt[df_filt["Date"] == latest]
            grp = (
                latest_df.dropna(subset=["RegionName", "AveragePrice"])
                .groupby("RegionName", as_index=False)["AveragePrice"]
                .mean()
            )
            if not grp.empty:
                chart_data = grp.set_index("RegionName")["AveragePrice"]
                st.bar_chart(chart_data)
            else:
                st.info("No data for latest month after filtering.")
        else:
            st.info("RegionName and AveragePrice needed for regional overview.")

    # -----------------------
    # Time series tab
    # -----------------------
    with tab_time:
        st.subheader("Time series – average price")

        if "Date" not in df_filt.columns or "AveragePrice" not in df_filt.columns:
            st.info("Date and AveragePrice columns are required for time series.")
        else:
            if "RegionName" in df_filt.columns:
                # Pivot to have regions as columns
                ts = (
                    df_filt.dropna(subset=["RegionName", "AveragePrice"])
                    .pivot_table(
                        index="Date",
                        columns="RegionName",
                        values="AveragePrice",
                        aggfunc="mean",
                    )
                    .sort_index()
                )
                st.line_chart(ts)
            else:
                ts = (
                    df_filt[["Date", "AveragePrice"]]
                    .dropna()
                    .set_index("Date")
                    .sort_index()
                )
                st.line_chart(ts)

            # Optional: Index time series
            if "Index" in df_filt.columns:
                st.markdown("### House price index over time")
                if "RegionName" in df_filt.columns:
                    ts_idx = (
                        df_filt.dropna(subset=["RegionName", "Index"])
                        .pivot_table(
                            index="Date",
                            columns="RegionName",
                            values="Index",
                            aggfunc="mean",
                        )
                        .sort_index()
                    )
                    st.line_chart(ts_idx)
                else:
                    ts_idx = (
                        df_filt[["Date", "Index"]]
                        .dropna()
                        .set_index("Date")
                        .sort_index()
                    )
                    st.line_chart(ts_idx)

    # -----------------------
    # Regional comparison tab
    # -----------------------
    with tab_region:
        st.subheader("Regional comparison")

        if "RegionName" not in df_filt.columns or "AveragePrice" not in df_filt.columns:
            st.info("RegionName and AveragePrice are required for regional comparison.")
        else:
            st.markdown("### Average price over selected period (by region)")
            grp = (
                df_filt.dropna(subset=["RegionName", "AveragePrice"])
                .groupby("RegionName", as_index=False)["AveragePrice"]
                .mean()
            )
            if not grp.empty:
                chart_data = grp.set_index("RegionName")["AveragePrice"]
                st.bar_chart(chart_data)

            if "SalesVolume" in df_filt.columns:
                st.markdown("### Average sales volume over selected period (by region)")
                grp_vol = (
                    df_filt.dropna(subset=["RegionName", "SalesVolume"])
                    .groupby("RegionName", as_index=False)["SalesVolume"]
                    .mean()
                )
                if not grp_vol.empty:
                    vol_data = grp_vol.set_index("RegionName")["SalesVolume"]
                    st.bar_chart(vol_data)

    # -----------------------
    # Property type tab
    # -----------------------
    with tab_property:
        st.subheader("Property type analysis")

        if "PropertyType" not in df_filt.columns or "AveragePrice" not in df_filt.columns:
            st.info("PropertyType and AveragePrice are required for this analysis.")
        else:
            st.markdown("### Average price by property type")

            grp = (
                df_filt.dropna(subset=["PropertyType", "AveragePrice"])
                .groupby("PropertyType", as_index=False)["AveragePrice"]
                .mean()
            )
            if not grp.empty:
                chart_data = grp.set_index("PropertyType")["AveragePrice"]
                st.bar_chart(chart_data)

            if "Date" in df_filt.columns:
                st.markdown("### Time series by property type")
                ts_prop = (
                    df_filt.dropna(subset=["PropertyType", "AveragePrice"])
                    .pivot_table(
                        index="Date",
                        columns="PropertyType",
                        values="AveragePrice",
                        aggfunc="mean",
                    )
                    .sort_index()
                )
                st.line_chart(ts_prop)


if __name__ == "__main__":
    main()
