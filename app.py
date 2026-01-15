import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="School Performance Explorer – England",
    layout="wide",
)

# =========================================================
# GOOGLE DRIVE HELPERS
# =========================================================

def load_csv_from_gdrive(file_id: str) -> pd.DataFrame:
    """
    Download a CSV directly from Google Drive using its file ID.
    Tries UTF-8 first, then falls back to Latin-1 (common for DfE / Ofsted files).
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    content = resp.content

    # Try UTF-8, fall back to Latin-1 / Windows-1252
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin1")

    return pd.read_csv(StringIO(text), low_memory=False)


def load_excel_from_gdrive(file_id: str, engine: str = "openpyxl") -> pd.DataFrame:
    """
    Download an Excel file directly from Google Drive using its file ID.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content), engine=engine)


# =========================================================
# FILE IDS – REPLACE THE PLACEHOLDERS WITH YOUR REAL IDs
# =========================================================
# For each file in the shared folder:
# 1. Open in browser
# 2. Copy the ID between '/d/' and '/view?...'
# 3. Paste into the corresponding entry below.

FILEIDS = {
    # CSV: 202425_rurality_and_idaci_of_region_provisional.csv
    "rurality_idaci": {
        "id": "1lSNdDytkgLKvgTsuy0hfRwuhHt22BFVE",
        "type": "csv",
    },
    # CSV: 202425_geography_by_school_types_provisional.csv
    "geo_school_types": {
        "id": "1R703u8VB2TDSPZaOXhRN2M5dDX4YdPoR",
        "type": "csv",
    },
    # CSV: spc_pupils_fsm.csv
    "fsm": {
        "id": "1DPrbSj5mb3DcM2NyVMkeUvJB51Uyp4aj",
        "type": "csv",
    },
    # CSV: 202324_performance_tables_schools_final.csv
    "ks4": {
        "id": "1m5Rm4vHFevaYZa_kyImoagOtF5rUcJUE",
        "type": "csv",
    },
    # CSV: Management_information_-_state-funded_schools_-_all_inspections_-_year_to_date_published_by_31_Dec_2024.csv
    "ofsted_mi": {
        "id": "1mKMIf_uF8zxqjK_JBBiQQvQ6zTB8IA7c",
        "type": "csv",
    },
}


def load_dataset(key: str) -> pd.DataFrame:
    """
    Wrapper that uses FILEIDS to load the right file from Drive.
    """
    meta = FILEIDS[key]
    file_id = meta["id"]
    ftype = meta["type"]

    if "PUT_" in file_id:
        raise ValueError(
            f"File ID for '{key}' has not been set. "
            f"Edit FILEIDS in app.py and paste the correct Google Drive ID."
        )

    if ftype == "csv":
        return load_csv_from_gdrive(file_id)
    else:
        return load_excel_from_gdrive(file_id)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def closing_year(time_period: int) -> int:
    """
    Convert a DfE-style time_period (e.g. 202122, 202324) into the closing year (e.g. 2022, 2024).
    Assumes the last two digits correspond to the final calendar year of the academic year.
    """
    s = str(int(time_period))
    return 2000 + int(s[-2:])


def format_percent(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.0f}%"


def format_1dp(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.1f}"


def format_2dp(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.2f}"


# =========================================================
# LOAD DATA FROM GOOGLE DRIVE
# =========================================================

@st.cache_data
def load_rurality():
    df = load_dataset("rurality_idaci")
    for col in [
        "attainment8_sum",
        "pupil_count",
        "progress8_pupil_count",
        "progress8_sum",
        "school_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["time_period"].apply(closing_year)
    df["idaci_band"] = df["idaci_decile"].fillna("Unknown")
    return df


@st.cache_data
def load_geo():
    df = load_dataset("geo_school_types")
    for col in [
        "attainment8_sum",
        "pupil_count",
        "progress8_pupil_count",
        "progress8_sum",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["time_period"].apply(closing_year)
    return df


@st.cache_data
def load_fsm_secondary():
    df = load_dataset("fsm")
    for col in ["percent_of_pupils", "headcount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["time_period"].apply(closing_year)

    # Restrict to state-funded secondary & FSM indicator used in performance tables
    mask = (
        (df["phase_type_grouping"] == "State-funded secondary")
        & (
            df["fsm"]
            == "known to be eligible for free school meals (used for FSM in Performance Tables)"
        )
    )
    df = df.loc[mask].copy()
    return df


@st.cache_data
def load_ks4_with_ofsted():
    ks4 = load_dataset("ks4")
    ks4["year"] = ks4["time_period"].apply(closing_year)
    ks4 = ks4[ks4["breakdown"] == "Total"].copy()
    ks4["avg_att8"] = pd.to_numeric(ks4["avg_att8"], errors="coerce")
    ks4["avg_p8score"] = pd.to_numeric(ks4["avg_p8score"], errors="coerce")
    ks4["school_urn"] = pd.to_numeric(ks4["school_urn"], errors="coerce")

    mi = load_dataset("ofsted_mi")
    mi["URN"] = pd.to_numeric(mi["URN"], errors="coerce")

    rating_map = {
        1.0: "Outstanding",
        2.0: "Good",
        3.0: "Requires improvement",
        4.0: "Inadequate",
    }
    mi["ofsted_rating"] = mi["Previous overall effectiveness"].map(rating_map)

    merged = ks4.merge(
        mi[["URN", "ofsted_rating", "Region", "Local authority"]],
        left_on="school_urn",
        right_on="URN",
        how="left",
    )
    return merged


# =========================================================
# ACTUALLY LOAD THEM
# =========================================================

try:
    rurality_df = load_rurality()
    geo_df = load_geo()
    fsm_df = load_fsm_secondary()
    ks4_ofsted_df = load_ks4_with_ofsted()
except Exception as e:
    st.error(
        f"Error loading data from Google Drive: {e}\n\n"
        "Check that FILEIDS in app.py contain the correct file IDs, and that the files "
        "are shared as 'Anyone with the link – Viewer'."
    )
    st.stop()

# Restrict rurality to regional level for decile-based charts
rurality_reg = rurality_df[rurality_df["geographic_level"] == "Regional"].copy()

# =========================================================
# SIDEBAR FILTERS (mockup-aligned)
# =========================================================

st.sidebar.title("Filters")

years_available = sorted(rurality_reg["year"].dropna().unique())
default_years = [y for y in years_available if y >= 2019]

selected_years = st.sidebar.multiselect(
    "Year (closing year)",
    options=years_available,
    default=default_years or years_available,
)

regions = ["All regions"] + sorted(rurality_reg["region_name"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=regions,
    default=["All regions"],
)

school_types = (
    ["All types"]
    + sorted(
        geo_df.loc[geo_df["geographic_level"] == "Regional", "establishment_type_group"]
        .dropna()
        .unique()
        .tolist()
    )
)
selected_school_types = st.sidebar.multiselect(
    "School type (establishment group)",
    options=school_types,
    default=["All types"],
)

rating_order = ["All ratings", "Outstanding", "Good", "Requires improvement", "Inadequate"]
present_ratings = sorted(
    ks4_ofsted_df["ofsted_rating"].dropna().unique().tolist(),
    key=lambda x: rating_order.index(x) if x in rating_order else 99,
)
ofsted_options = ["All ratings"] + present_ratings
selected_ofsted = st.sidebar.multiselect(
    "Ofsted rating",
    options=ofsted_options,
    default=["All ratings"],
)

idaci_bands_all = [
    "All bands",
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]
selected_idaci_bands = st.sidebar.multiselect(
    "IMD / IDACI band",
    options=idaci_bands_all,
    default=["All bands"],
)

st.sidebar.button("Run analysis")

st.sidebar.markdown("---")
st.sidebar.subheader("View options")
primary_outcome = st.sidebar.selectbox(
    "Primary outcome",
    options=["Attainment 8", "Progress 8", "A-level average points (placeholder)"],
    index=0,
)
split_by = st.sidebar.selectbox(
    "Split by",
    options=["IMD / IDACI band", "Region", "School type"],
    index=0,
)

# =========================================================
# FILTER HELPERS
# =========================================================

def filter_rurality():
    df = rurality_reg.copy()
    if selected_years:
        df = df[df["year"].isin(selected_years)]
    if selected_regions and "All regions" not in selected_regions:
        df = df[df["region_name"].isin(selected_regions)]
    if selected_idaci_bands and "All bands" not in selected_idaci_bands:
        df = df[df["idaci_band"].isin(selected_idaci_bands)]
    return df


def filter_geo_regional():
    df = geo_df[geo_df["geographic_level"] == "Regional"].copy()
    if selected_years:
        df = df[df["year"].isin(selected_years)]
    if selected_regions and "All regions" not in selected_regions:
        df = df[df["region_name"].isin(selected_regions)]
    if selected_school_types and "All types" not in selected_school_types:
        df = df[df["establishment_type_group"].isin(selected_school_types)]
    return df


def filter_fsm():
    df = fsm_df.copy()
    if selected_years:
        df = df[df["year"].isin(selected_years)]
    if selected_regions and "All regions" not in selected_regions:
        df = df[df["region_name"].isin(selected_regions)]
    return df


def filter_ks4_ofsted():
    df = ks4_ofsted_df.copy()
    if selected_years:
        df = df[df["year"].isin(selected_years)]
    if selected_regions and "All regions" not in selected_regions:
        df = df[df["Region"].isin(selected_regions)]
    if selected_ofsted and "All ratings" not in selected_ofsted:
        df = df[df["ofsted_rating"].isin(selected_ofsted)]
    return df


# =========================================================
# KPI CALCULATIONS
# =========================================================

rur_f = filter_rurality()
geo_f = filter_geo_regional()
fsm_f = filter_fsm()

if rur_f["pupil_count"].sum() > 0:
    kpi_att8 = rur_f["attainment8_sum"].sum() / rur_f["pupil_count"].sum()
else:
    kpi_att8 = float("nan")

if rur_f["progress8_pupil_count"].sum() > 0:
    kpi_p8 = rur_f["progress8_sum"].sum() / rur_f["progress8_pupil_count"].sum()
else:
    kpi_p8 = float("nan")

if not fsm_f.empty:
    kpi_fsm = fsm_f["percent_of_pupils"].median()
else:
    kpi_fsm = float("nan")

rur_hd = rur_f[rur_f["idaci_decile"] != "Total"].copy()
if not rur_hd.empty and rur_hd["school_count"].notna().any():
    rur_hd["school_count"] = pd.to_numeric(rur_hd["school_count"], errors="coerce")
    high_bands = ["0-10%", "10-20%", "20-30%"]
    num_hd = rur_hd[rur_hd["idaci_decile"].isin(high_bands)]["school_count"].sum()
    den_hd = rur_hd["school_count"].sum()
    kpi_high_depr = (num_hd / den_hd * 100) if den_hd else float("nan")
else:
    kpi_high_depr = float("nan")

# =========================================================
# HEADER (title + chips)
# =========================================================

header_col, chips_col = st.columns([3, 2])

with header_col:
    st.title("School Performance Explorer - England")
    st.caption(
        "Interactive concept for analysing GCSE and A-level outcomes in England by "
        "socioeconomic context, school type and inspection outcomes. "
        "Emphasis is on understanding patterns rather than ranking individual schools."
    )

with chips_col:
    st.markdown(
        """
<div style="display:flex; justify-content:flex-end; gap:0.5rem; flex-wrap:wrap;">
  <span style="background:#E6F4EA; color:#137333; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">
    Public, anonymised data
  </span>
  <span style="background:#FCE8E6; color:#C5221F; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">
    GCSE &amp; A-level
  </span>
  <span style="background:#FFF7DB; color:#B06000; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">
    IMD &amp; FSM context
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab_overview, tab_socio, tab_ofsted, tab_gap, tab_outliers = st.tabs(
    [
        "Overview",
        "Socioeconomic patterns",
        "Ofsted & governance",
        "Disadvantage gap",
        "Contextual outliers",
    ]
)

# =========================================================
# OVERVIEW TAB
# =========================================================

with tab_overview:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("**Average Attainment 8**")
        st.markdown(f"<h2 style='margin:0'>{format_1dp(kpi_att8)}</h2>", unsafe_allow_html=True)
        st.caption("Mean GCSE Attainment 8 score across selected filters.")

    with kpi2:
        st.markdown("**Progress 8**")
        st.markdown(f"<h2 style='margin:0'>{format_2dp(kpi_p8)}</h2>", unsafe_allow_html=True)
        st.caption("Average Progress 8 relative to expected pupil trajectories.")

    with kpi3:
        st.markdown("**FSM share (median)**")
        st.markdown(f"<h2 style='margin:0'>{format_percent(kpi_fsm)}</h2>", unsafe_allow_html=True)
        st.caption("Median proportion of pupils eligible for Free School Meals.")

    with kpi4:
        st.markdown("**High-deprivation schools**")
        st.markdown(f"<h2 style='margin:0'>{format_percent(kpi_high_depr)}</h2>", unsafe_allow_html=True)
        st.caption("Share of schools located in the most deprived IDACI bands (0–30%).")

    st.markdown("")
st.markdown("##### Driver chart: Primary outcome split by selected dimension")
st.caption(
    "This chart is controlled by **Chart controls** in the sidebar: "
    "**Primary outcome** determines the metric; **Split by** determines the grouping."
)

# Choose dataset + grouping field based on split_by
if split_by == "IMD / IDACI band":
    df_driver = filter_rurality().copy()
    df_driver = df_driver[df_driver["idaci_decile"].isin(idaci_bands_all[1:])]

    group_col = "idaci_decile"
    num_col = "attainment8_sum" if primary_outcome == "Attainment 8" else "progress8_sum"
    den_col = "pupil_count" if primary_outcome == "Attainment 8" else "progress8_pupil_count"

elif split_by == "Region":
    df_driver = geo_df[geo_df["geographic_level"] == "Regional"].copy()
    if selected_years:
        df_driver = df_driver[df_driver["year"].isin(selected_years)]
    if selected_regions and "All regions" not in selected_regions:
        df_driver = df_driver[df_driver["region_name"].isin(selected_regions)]

    group_col = "region_name"
    num_col = "attainment8_sum" if primary_outcome == "Attainment 8" else "progress8_sum"
    den_col = "pupil_count" if primary_outcome == "Attainment 8" else "progress8_pupil_count"

else:  # School type
    df_driver = filter_geo_regional().copy()
    group_col = "establishment_type_group"
    num_col = "attainment8_sum" if primary_outcome == "Attainment 8" else "progress8_sum"
    den_col = "pupil_count" if primary_outcome == "Attainment 8" else "progress8_pupil_count"

if df_driver.empty:
    st.info("No data available for the selected filters.")
else:
    agg = df_driver.groupby(group_col, as_index=False)[[num_col, den_col]].sum()
    agg["value"] = agg[num_col] / agg[den_col]
    agg = agg.sort_values("value", ascending=False).set_index(group_col)
    st.bar_chart(agg["value"], use_container_width=True)

    top_left, top_right = st.columns(2)

    with top_left:
        st.markdown("##### Attainment 8 by IMD / IDACI band")
        st.caption(
            "Mean Attainment 8 by IDACI band (0–10% most deprived to 90–100% least deprived), "
            "with separate lines for each year."
        )

        rur_chart = rur_f[
            rur_f["idaci_decile"].isin(
                [
                    "0-10%",
                    "10-20%",
                    "20-30%",
                    "30-40%",
                    "40-50%",
                    "50-60%",
                    "60-70%",
                    "70-80%",
                    "80-90%",
                    "90-100%",
                ]
            )
        ].copy()

        if rur_chart.empty:
            st.info("No data for the selected filters.")
        else:
            grouped = (
                rur_chart.groupby(["year", "idaci_decile"], as_index=False)[
                    ["attainment8_sum", "pupil_count"]
                ].sum()
            )
            grouped["att8"] = grouped["attainment8_sum"] / grouped["pupil_count"]
            pivot = grouped.pivot(index="idaci_decile", columns="year", values="att8")
            order = [
                "0-10%",
                "10-20%",
                "20-30%",
                "30-40%",
                "40-50%",
                "50-60%",
                "60-70%",
                "70-80%",
                "80-90%",
                "90-100%",
            ]
            pivot = pivot.reindex(order)
            st.line_chart(pivot, use_container_width=True)
            st.caption("Aggregated at IDACI-band level; no individual schools are shown.")

    with top_right:
        st.markdown("##### Attainment 8 by Ofsted rating")
        st.caption(
            "Attainment 8 summary by Ofsted rating. Ratings are proxied using the "
            "‘Previous overall effectiveness’ field from the management information file."
        )

        ofsted_f = filter_ks4_ofsted()
        ofsted_f = ofsted_f.dropna(subset=["ofsted_rating", "avg_att8"]).copy()
        ofsted_f["avg_att8"] = pd.to_numeric(ofsted_f["avg_att8"], errors="coerce")

        if ofsted_f.empty:
            st.info("No matched KS4–Ofsted data for the selected filters.")
        else:
            rating_means = (
                ofsted_f.groupby("ofsted_rating", as_index=False)["avg_att8"]
                .mean()
                .sort_values("avg_att8", ascending=False)
            )
            rating_means = rating_means.set_index("ofsted_rating")
            st.bar_chart(rating_means["avg_att8"], use_container_width=True)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("##### Regional and school-type differences")
        st.caption(
            "Average performance by region and establishment type "
            "(for example, academies vs LA-maintained schools)."
        )
        geo_chart = geo_f.copy()
        if geo_chart.empty:
            st.info("No data for the selected filters.")
        else:
            if primary_outcome == "Attainment 8":
                num_col = "attainment8_sum"
            else:
                num_col = "progress8_sum"

            grouped = (
                geo_chart.groupby(["region_name", "establishment_type_group"], as_index=False)[
                    [num_col, "pupil_count"]
                ].sum()
            )
            grouped["value"] = grouped[num_col] / grouped["pupil_count"]
            pivot = grouped.pivot(
                index="region_name", columns="establishment_type_group", values="value"
            ).sort_index()
            st.bar_chart(pivot, use_container_width=True)
            st.caption(
                "Designed to highlight broad patterns by region and school type, rather than produce league tables."
            )

    with bottom_right:
        st.markdown("##### Contextual outliers (aggregated)")
        st.caption(
            "Illustrative summary of regions by Progress 8 relative to the England mean."
        )

        geo_all_state = geo_df[
            (geo_df["geographic_level"] == "Regional")
            & (geo_df["establishment_type_group"] == "All state-funded")
        ].copy()
        geo_all_state["year"] = geo_all_state["time_period"].apply(closing_year)
        if selected_years:
            geo_all_state = geo_all_state[geo_all_state["year"].isin(selected_years)]
        if selected_regions and "All regions" not in selected_regions:
            geo_all_state = geo_all_state[geo_all_state["region_name"].isin(selected_regions)]

        if geo_all_state.empty:
            st.info("No data available to calculate contextual summaries.")
        else:
            geo_all_state["progress8_sum"] = pd.to_numeric(
                geo_all_state["progress8_sum"], errors="coerce"
            )
            geo_all_state["progress8_pupil_count"] = pd.to_numeric(
                geo_all_state["progress8_pupil_count"], errors="coerce"
            )

            region_stats = (
                geo_all_state.groupby("region_name", as_index=False)[
                    ["progress8_sum", "progress8_pupil_count"]
                ]
                .sum()
            )
            region_stats["p8"] = (
                region_stats["progress8_sum"] / region_stats["progress8_pupil_count"]
            )
            england_mean = (
                region_stats["progress8_sum"].sum()
                / region_stats["progress8_pupil_count"].sum()
            )

            def band(p8):
                if pd.isna(p8):
                    return "No data"
                diff = p8 - england_mean
                if diff > 0.1:
                    return "Above expected"
                elif diff < -0.1:
                    return "Below expected"
                else:
                    return "Around expected"

            region_stats["Residual band"] = region_stats["p8"].apply(band)
            region_stats["Region"] = region_stats["region_name"]
            region_stats["LA group"] = "All local authorities"
            region_stats["School type mix"] = "All state-funded"

            table = region_stats[["Region", "LA group", "School type mix", "Residual band"]]
            st.dataframe(table, use_container_width=True, height=260)
            st.caption(
                "In a full regression-based implementation, residual bands would be derived from "
                "school-level value-added models and aggregated at LA/region level to avoid naming "
                "or shaming individual schools."
            )

# =========================================================
# SOCIOECONOMIC PATTERNS TAB
# =========================================================

with tab_socio:
    st.subheader("Socioeconomic patterns")

    st.markdown(
        "This tab explores how attainment and progress vary with measures of "
        "socioeconomic context, such as IDACI bands and FSM rates."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Attainment 8 vs IDACI band")
        rur_chart = filter_rurality()
        rur_chart = rur_chart[
            rur_chart["idaci_decile"].isin(
                [
                    "0-10%",
                    "10-20%",
                    "20-30%",
                    "30-40%",
                    "40-50%",
                    "50-60%",
                    "60-70%",
                    "70-80%",
                    "80-90%",
                    "90-100%",
                ]
            )
        ]
        if rur_chart.empty:
            st.info("No data for the selected filters.")
        else:
            grouped = (
                rur_chart.groupby(["idaci_decile"], as_index=False)[
                    ["attainment8_sum", "pupil_count"]
                ].sum()
            )
            grouped["att8"] = grouped["attainment8_sum"] / grouped["pupil_count"]
            grouped = grouped.sort_values("att8")
            grouped = grouped.set_index("idaci_decile")
            st.bar_chart(grouped["att8"], use_container_width=True)
            st.caption(
                "Lower IDACI bands (0–10%, 10–20%) correspond to more deprived areas; "
                "this illustrates the deprivation gradient in Attainment 8."
            )

    with col_b:
        st.markdown("##### Progress 8 vs IDACI band")
        if rur_chart.empty:
            st.info("No data for the selected filters.")
        else:
            grouped = (
                rur_chart.groupby(["idaci_decile"], as_index=False)[
                    ["progress8_sum", "progress8_pupil_count"]
                ].sum()
            )
            grouped["p8"] = grouped["progress8_sum"] / grouped["progress8_pupil_count"]
            grouped = grouped.set_index("idaci_decile")
            st.bar_chart(grouped["p8"], use_container_width=True)
            st.caption(
                "Progress 8 is already context-adjusted for prior attainment; "
                "this view checks whether substantial gradients remain by IDACI band."
            )

# =========================================================
# OFSTED & GOVERNANCE TAB
# =========================================================

with tab_ofsted:
    st.subheader("Ofsted & governance")

    st.markdown(
        "This tab focuses on how performance relates to inspection outcomes and, "
        "optionally, school type or governance arrangements."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Attainment 8 by Ofsted rating (mean)")
        ofsted_f = filter_ks4_ofsted()
        ofsted_f = ofsted_f.dropna(subset=["ofsted_rating", "avg_att8"]).copy()
        ofsted_f["avg_att8"] = pd.to_numeric(ofsted_f["avg_att8"], errors="coerce")

        if ofsted_f.empty:
            st.info("No matched KS4–Ofsted data for the selected filters.")
        else:
            rating_means = (
                ofsted_f.groupby("ofsted_rating", as_index=False)["avg_att8"]
                .mean()
                .sort_values("avg_att8", ascending=False)
            )
            rating_means = rating_means.set_index("ofsted_rating")
            st.bar_chart(rating_means["avg_att8"], use_container_width=True)

    with col2:
        st.markdown("##### Progress 8 by Ofsted rating (mean)")
        ofsted_f = filter_ks4_ofsted()
        ofsted_f = ofsted_f.dropna(subset=["ofsted_rating", "avg_p8score"]).copy()
        ofsted_f["avg_p8score"] = pd.to_numeric(ofsted_f["avg_p8score"], errors="coerce")

        if ofsted_f.empty:
            st.info("No matched KS4–Ofsted data for the selected filters.")
        else:
            rating_means = (
                ofsted_f.groupby("ofsted_rating", as_index=False)["avg_p8score"]
                .mean()
                .sort_values("avg_p8score", ascending=False)
            )
            rating_means = rating_means.set_index("ofsted_rating")
            st.bar_chart(rating_means["avg_p8score"], use_container_width=True)
            st.caption(
                "This aligns with the thesis question of whether inspection grades "
                "match context-adjusted performance (Progress 8)."
            )

# =========================================================
# DISADVANTAGE GAP TAB
# =========================================================

with tab_gap:
    st.subheader("Disadvantage gap")


    if fsm_f.empty:
        st.info("No FSM data for the selected filters.")
    else:
        gap_df = fsm_f.groupby("year", as_index=False)["percent_of_pupils"].median()
        gap_df = gap_df.set_index("year")
        st.line_chart(gap_df["percent_of_pupils"], use_container_width=True)
        st.caption(
            "Median FSM eligibility rate for state-funded secondary schools over time. "
            "In the final version, this would be complemented with attainment gaps between FSM and non-FSM pupils."
        )

# =========================================================
# CONTEXTUAL OUTLIERS TAB
# =========================================================

with tab_outliers:
    st.subheader("Contextual outliers")

    st.markdown(
        "This tab is intended to surface **regions or LA groupings that perform better or worse than expected** "
        "once socioeconomic context is taken into account. The proxy table on the **Overview** tab already "
        "summarises region-level Progress 8 relative to the England mean. "
    )
