import streamlit as st
import pandas as pd
import plotly.express as px
import modelling as md  # <--- ADD THIS
# Import custom modules
from utils import format_percent, format_1dp, format_2dp
import data_loader as dl

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="School Performance Explorer – England",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DATA LOADING (Via new data_loader.py)
# =========================================================
try:
    rurality_df = dl.load_rurality()
    geo_df = dl.load_geo()
    fsm_df = dl.load_fsm_secondary()
    ks4_ofsted_df = dl.load_ks4_with_ofsted()
except Exception as e:
    st.error(f"Data Loading Error: {e}")
    st.stop()

# Helper lists for dropdowns
all_years = sorted(rurality_df["year"].dropna().unique())
all_regions = sorted(rurality_df["region_name"].dropna().unique())
all_school_types = sorted(geo_df["establishment_type_group"].dropna().unique())
idaci_bands = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
]

available_ks4_years = sorted(ks4_ofsted_df["year"].dropna().unique())
latest_year = max(available_ks4_years) if available_ks4_years else 2024

# =========================================================
# SIDEBAR (Context & Metadata)
# =========================================================
with st.sidebar:
    st.title("ℹ️ About")
    st.info(
        "**School Performance Explorer**\n\n"
        "An interactive tool investigating the relationship between "
        "socioeconomic disadvantage, geography, and educational outcomes in England."
    )
    
    st.markdown("---")
    st.subheader("🗃️ Data Metadata")
    st.markdown(f"**Years Available:** {min(all_years)} – {max(all_years)}")
    st.markdown(f"**Total Regions:** {len(all_regions)}")
    st.markdown(f"**School Types:** {len(all_school_types)}")
    
    st.markdown("---")
    st.subheader("📌 User Guide")
    st.markdown(
        """
        1. **Overview:** See high-level trends and key performance drivers.
        2. **Socioeconomic:** Compare two regions side-by-side to see how poverty affects them differently.
        3. **Ofsted:** Analyze if 'Outstanding' ratings actually match exam results.
        """
    )
    st.markdown("---")
    st.caption("Dissertation Project • 2026")
        # =========================================================
    # SIDEBAR FILTERS (Global)
    # =========================================================
    st.sidebar.title("Global Filters")

    # 1. Year Filter
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=all_years,
        default=[latest_year],
        help="Filters the KPIs and Overview data."
    )

    # 2. Region Filter
    selected_regions = st.sidebar.multiselect(
        "Select Region(s)",
        options=["All Regions"] + all_regions,
        default=["All Regions"]
    )

    # 3. Apply Filters to Data (Create a filtered copy for KPIs)
    kpi_df = rurality_df.copy()
    fsm_kpi_df = fsm_df.copy()

    # Filter by Year
    if selected_years:
        kpi_df = kpi_df[kpi_df["year"].isin(selected_years)]
        if not fsm_kpi_df.empty and "year" in fsm_kpi_df.columns:
            fsm_kpi_df = fsm_kpi_df[fsm_kpi_df["year"].isin(selected_years)]

    # Filter by Region
    if selected_regions and "All Regions" not in selected_regions:
        kpi_df = kpi_df[kpi_df["region_name"].isin(selected_regions)]
        if not fsm_kpi_df.empty and "region_name" in fsm_kpi_df.columns:
            fsm_kpi_df = fsm_kpi_df[fsm_kpi_df["region_name"].isin(selected_regions)]

# =========================================================
# HEADER
# =========================================================
header_col, chips_col = st.columns([3, 2])

with header_col:
    st.title("School Performance Explorer - England")
    st.caption("A 'White Box' analysis of educational inequality.")

with chips_col:
    st.markdown(
        """
        <div style="display:flex; justify-content:flex-end; gap:0.5rem; flex-wrap:wrap;">
          <span style="background:#E6F4EA; color:#137333; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">
            Clean Data Loaded
          </span>
          <span style="background:#FCE8E6; color:#C5221F; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem;">
            Local Filtering Enabled
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab_overview, tab_socio, tab_ofsted, tab_gap, tab_outliers = st.tabs([
    "Overview", 
    "Socioeconomic Patterns", 
    "Ofsted & Governance", 
    "Disadvantage Gap", 
    "Contextual Outliers"
])

# =========================================================
# TAB 1: EXECUTIVE OVERVIEW (Enhanced)
# =========================================================
with tab_overview:
    st.subheader("National Performance Dashboard")
    st.markdown("High-level summary of educational outcomes across England.")

    # -----------------------------------------------------
    # 1. KPIs (Weighted Averages)
    # -----------------------------------------------------
    # Calculate weighted means based on user selection (kpi_df)
    total_pupils = kpi_df["pupil_count"].sum()
    p8_pupils = kpi_df["progress8_pupil_count"].sum()
    
    if total_pupils > 0:
        avg_att8 = kpi_df["attainment8_sum"].sum() / total_pupils
        # Subject specific averages (if columns exist in rurality_df)
        avg_eng = kpi_df["attainment8eng_sum"].sum() / total_pupils if "attainment8eng_sum" in kpi_df.columns else 0
        avg_mat = kpi_df["attainment8mat_sum"].sum() / total_pupils if "attainment8mat_sum" in kpi_df.columns else 0
    else:
        avg_att8, avg_eng, avg_mat = 0, 0, 0
        
    if p8_pupils > 0:
        avg_p8 = kpi_df["progress8_sum"].sum() / p8_pupils
    else:
        avg_p8 = 0

    # High Deprivation % (Schools in IDACI Deciles 1-3)
    valid_schools = kpi_df[~kpi_df["idaci_decile"].isin(["Total", "Unknown"])]
    deprived_count = valid_schools[valid_schools["idaci_decile"].isin(["0-10%", "10-20%", "20-30%"])]["school_count"].sum()
    total_school_count = valid_schools["school_count"].sum()
    deprivation_pct = (deprived_count / total_school_count * 100) if total_school_count > 0 else 0

    # Display Metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Attainment 8", format_1dp(avg_att8), help="Average grade across 8 subjects.")
    k2.metric("Progress 8", format_2dp(avg_p8), help="Value added (0 is average).")
    k3.metric("English Score", format_1dp(avg_eng * 2), help="English Element (Double Weighted).")
    k4.metric("Math Score", format_1dp(avg_mat * 2), help="Math Element (Double Weighted).")
    k5.metric("High Poverty Schools", f"{deprivation_pct:.0f}%", help="% of schools in bottom 30% deprivation deciles.")

    st.markdown("---")

    # -----------------------------------------------------
    # 2. ACADEMIC PERFORMANCE (Scatter & Subjects)
    # -----------------------------------------------------
    st.subheader("1. Academic Landscape")
    row1_1, row1_2 = st.columns([3, 2])

    with row1_1:
        st.markdown("##### 📈 Value Added vs. Raw Grades (by Local Authority)")
        st.caption("Do high grades always mean good teaching? (Top Right = High Performance & High Progress)")
        
        # Aggregate by Local Authority for the scatter
        # We need a dataframe that has LA names. geo_df is best for this.
        # Filter geo_df by the selected year/region filters
        df_la = geo_df[geo_df["year"].isin(selected_years)].copy()
        if selected_regions and "All Regions" not in selected_regions:
            df_la = df_la[df_la["region_name"].isin(selected_regions)]
            
        # Group by LA
        la_agg = df_la.groupby(["la_name", "region_name"], as_index=False).agg({
            "attainment8_sum": "sum", "pupil_count": "sum",
            "progress8_sum": "sum", "progress8_pupil_count": "sum"
        })
        
        la_agg["Attainment 8"] = la_agg["attainment8_sum"] / la_agg["pupil_count"]
        la_agg["Progress 8"] = la_agg["progress8_sum"] / la_agg["progress8_pupil_count"]
        
        fig_scatter = px.scatter(
            la_agg,
            x="Attainment 8",
            y="Progress 8",
            color="region_name",
            hover_name="la_name",
            size="pupil_count",
            size_max=30,
            title="LA Performance Matrix",
            labels={"region_name": "Region"},
            height=400
        )
        # Add Quadrant Lines
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="National Avg Progress")
        fig_scatter.add_vline(x=47, line_dash="dash", line_color="gray", annotation_text="Avg Attainment")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row1_2:
        st.markdown("##### 📚 Core Subjects: English vs Math")
        st.caption("Average component score by Region.")
        
        # Prepare Data
        # We use kpi_df (rurality) which has the breakdown columns
        if "attainment8eng_sum" in kpi_df.columns:
            reg_subj = kpi_df.groupby("region_name", as_index=False)[["attainment8eng_sum", "attainment8mat_sum", "pupil_count"]].sum()
            reg_subj["English"] = (reg_subj["attainment8eng_sum"] / reg_subj["pupil_count"]) * 2 # Double weighted in Att8
            reg_subj["Maths"] = (reg_subj["attainment8mat_sum"] / reg_subj["pupil_count"]) * 2
            
            # Melt for Grouped Bar
            reg_melt = reg_subj.melt(id_vars="region_name", value_vars=["English", "Maths"], var_name="Subject", value_name="Score")
            
            fig_subj = px.bar(
                reg_melt,
                x="region_name",
                y="Score",
                color="Subject",
                barmode="group",
                title="Core Subject Strength",
                color_discrete_map={"English": "#3498DB", "Maths": "#E67E22"},
                height=400
            )
            fig_subj.update_layout(xaxis_title=None)
            st.plotly_chart(fig_subj, use_container_width=True)
        else:
            st.warning("Subject breakdown data unavailable in current dataset.")

    st.markdown("---")

    # -----------------------------------------------------
    # 3. STRUCTURAL ANALYSIS (Geography & School Type)
    # -----------------------------------------------------
    st.subheader("2. Structural Drivers")
    row2_1, row2_2 = st.columns(2)

    with row2_1:
        st.markdown("##### 🏛️ The School Type Effect")
        st.caption("Which school models are achieving the highest grades?")
        
        # Use geo_df which has 'establishment_type_group'
        type_agg = df_la.groupby("establishment_type_group", as_index=False).agg({
            "attainment8_sum": "sum", "pupil_count": "sum"
        })
        type_agg["Score"] = type_agg["attainment8_sum"] / type_agg["pupil_count"]
        type_agg = type_agg.sort_values("Score", ascending=True) # Sort for bar chart
        
        fig_type = px.bar(
            type_agg,
            y="establishment_type_group",
            x="Score",
            orientation='h',
            title="Attainment 8 by School Type",
            text_auto='.1f',
            color="Score",
            color_continuous_scale="Blues"
        )
        fig_type.update_layout(yaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(fig_type, use_container_width=True)

    with row2_2:
        st.markdown("##### 🌍 The North-South Divide")
        st.caption("Aggregated performance by macro-region.")
        
        # Macro Region Mapping
        macro_map = {
            "North East": "North", "North West": "North", "Yorkshire and the Humber": "North",
            "East Midlands": "Midlands", "West Midlands": "Midlands",
            "South East": "South", "South West": "South", "East of England": "South",
            "London": "London"
        }
        
        # Create Macro Column
        df_la["Macro"] = df_la["region_name"].map(macro_map)
        macro_agg = df_la.groupby("Macro", as_index=False).agg({"attainment8_sum": "sum", "pupil_count": "sum"})
        macro_agg["Attainment 8"] = macro_agg["attainment8_sum"] / macro_agg["pupil_count"]
        
        fig_macro = px.bar(
            macro_agg,
            x="Macro",
            y="Attainment 8",
            color="Macro",
            title="Macro-Region Performance",
            text_auto='.1f',
            color_discrete_map={"North": "#E74C3C", "South": "#2ECC71", "London": "#9B59B6", "Midlands": "#F1C40F"}
        )
        st.plotly_chart(fig_macro, use_container_width=True)

    st.markdown("---")

# -----------------------------------------------------
    # 4. OFSTED SNAPSHOT (LOCKED TO 2024)
    # -----------------------------------------------------
    st.subheader("3. The Inspection View (2024 Only)")
    st.caption("Analyzing the correlation between Ofsted Ratings and Performance. *Data restricted to 2024.*")
    
    # Force filter to 2024 for this section
    ofsted_2024 = ks4_ofsted_df[ks4_ofsted_df["year"] == 2024].copy()
    ofsted_2024 = ofsted_2024.dropna(subset=["ofsted_rating", "avg_att8"])
    
    if not ofsted_2024.empty:
        o1, o2 = st.columns([1, 1])
        
        rating_order = ["Outstanding", "Good", "Requires Improvement", "Inadequate"]
        color_map_ofsted = {"Outstanding": "#00703c", "Good": "#84bd00", "Requires Improvement": "#ffb81c", "Inadequate": "#d4351c"}

        with o1:
            st.markdown("##### 'The Ofsted Premium'")
            st.caption("How many extra grade points does an 'Outstanding' school get?")
            
            # Calc mean by rating
            o_mean = ofsted_2024.groupby("ofsted_rating", as_index=False)["avg_att8"].mean()
            
            fig_o_bar = px.bar(
                o_mean,
                x="ofsted_rating",
                y="avg_att8",
                color="ofsted_rating",
                title="Average Attainment 8 by Rating",
                category_orders={"ofsted_rating": rating_order},
                color_discrete_map=color_map_ofsted,
                text_auto='.1f'
            )
            fig_o_bar.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig_o_bar, use_container_width=True)
            
        with o2:
            st.markdown("##### Inspection Outcomes 2024")
            st.caption("Distribution of school ratings.")
            
            o_dist = ofsted_2024["ofsted_rating"].value_counts().reset_index()
            o_dist.columns = ["Rating", "Count"]
            
            # ✅ FIX: Use px.pie with hole=0.4 instead of px.donut
            fig_pie = px.pie(
                o_dist,
                names="Rating",
                values="Count",
                color="Rating",
                title="School Rating Share",
                color_discrete_map=color_map_ofsted,
                hole=0.4 # This makes it a Donut chart
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No Ofsted/KS4 data available for 2024 to generate these charts.")

    st.markdown("---")

    # -----------------------------------------------------
    # 5. GEOGRAPHICAL MAP (Existing but Polished)
    # -----------------------------------------------------
    st.subheader("4. Geographical Performance Map")
    st.caption("Average Attainment 8 scores across England's regions.")

    # Define Coordinates for English Regions
    region_coords = {
        "North East":               {"lat": 55.00, "lon": -1.80},
        "North West":               {"lat": 53.50, "lon": -2.50},
        "Yorkshire and the Humber": {"lat": 53.80, "lon": -1.00},
        "East Midlands":            {"lat": 52.90, "lon": -0.80},
        "West Midlands":            {"lat": 52.50, "lon": -2.00},
        "East of England":          {"lat": 52.20, "lon": 0.50},
        "London":                   {"lat": 51.51, "lon": -0.13},
        "South East":               {"lat": 51.20, "lon": -0.60},
        "South West":               {"lat": 50.80, "lon": -3.50}
    }

    # Prepare Map Data
    map_df = kpi_df.groupby("region_name", as_index=False).agg({
        "attainment8_sum": "sum", "pupil_count": "sum", "school_count": "sum"
    })
    
    if map_df["pupil_count"].sum() > 0:
        map_df["Avg Attainment 8"] = map_df["attainment8_sum"] / map_df["pupil_count"]
        map_df["lat"] = map_df["region_name"].map(lambda x: region_coords.get(x, {}).get("lat"))
        map_df["lon"] = map_df["region_name"].map(lambda x: region_coords.get(x, {}).get("lon"))
        map_df = map_df.dropna(subset=["lat", "lon"])

        c_map, c_info = st.columns([2, 1])
        
        with c_map:
            map_title = f"Regional Performance ({', '.join(map(str, selected_years))})" if selected_years else "Regional Performance"

            fig_map = px.scatter_geo(
                map_df,
                lat="lat",
                lon="lon",
                size="pupil_count",      
                color="Avg Attainment 8", 
                hover_name="region_name",
                size_max=40,             
                color_continuous_scale="Viridis",
                scope="europe",          
                title=map_title,
                labels={"pupil_count": "Total Pupils"}
            )
            fig_map.update_geos(
                center=dict(lat=52.5, lon=-1.5),
                projection_scale=6,              
                visible=False,
                showcountries=True, 
                countrycolor="Black"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with c_info:
            st.markdown("#### Regional Leaderboard")
            leaderboard = map_df[["region_name", "Avg Attainment 8"]].sort_values("Avg Attainment 8", ascending=False)
            st.dataframe(
                leaderboard.style.format({"Avg Attainment 8": "{:.1f}"}).background_gradient(cmap="viridis"),
                use_container_width=True,
                height=400
            )
# -----------------------------------------------------
    # 6. TRENDS & INSIGHTS (New Advanced Section)
    # -----------------------------------------------------
    st.subheader("5. Trends & Strategic Insights")
    st.markdown("Long-term performance tracking and local authority leaderboards.")

    # A. Multi-Year Trend Analysis (Line Chart)
    # -----------------------------------------
    # We aggregate the 'rurality_df' (which has historical data) by Year
    trend_df = rurality_df.groupby("year", as_index=False).agg({
        "attainment8_sum": "sum", 
        "pupil_count": "sum"
    })
    trend_df["Average Score"] = trend_df["attainment8_sum"] / trend_df["pupil_count"]
    
    t1, t2 = st.columns([2, 1])
    
    with t1:
        st.markdown("##### 📈 National Performance Trend")
        fig_trend = px.line(
            trend_df, 
            x="year", 
            y="Average Score", 
            markers=True,
            title="National Average Attainment 8 (All Years)",
            color_discrete_sequence=["#2C3E50"]
        )
        fig_trend.update_yaxes(range=[40, 60]) # Zoom in to show variance
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with t2:
        st.markdown("##### 📊 Score Distribution (Current Selection)")
        # Histogram of School Scores for the selected year
        # Use 'df_la' or 'ks4_ofsted_df' filtered to selection
        hist_data = ks4_ofsted_df[ks4_ofsted_df["year"].isin(selected_years)]
        
        fig_hist = px.histogram(
            hist_data, 
            x="avg_att8", 
            nbins=30,
            title="School Score Spread",
            color_discrete_sequence=["#16A085"],
            labels={"avg_att8": "Attainment 8 Score"}
        )
        fig_hist.add_vline(x=avg_att8, line_dash="dash", annotation_text="Avg")
        st.plotly_chart(fig_hist, use_container_width=True)

    # B. Subject Leaderboard (English vs Maths Pass Rates)
    # ----------------------------------------------------
    st.markdown("##### 🏆 Top 10 Local Authorities (Pass Rates)")
    st.caption("Percentage of students achieving Grade 5+ (Strong Pass) in English & Math.")
    
    # Check if we have the Pass Rate columns in geo_df
    # Based on your file list: 'engmath_95_percent' might be in Rurality or Geo
    # If not, we calculate it from sums if available, or fall back to Attainment 8
    
    # We will use geo_df filtered by the selected year
    top_la = geo_df[geo_df["year"] == latest_year].copy()
    
    # We need to aggregate by LA Name (geo_df is usually Regional/LA level)
    # Let's check if we have a pass rate column. If not, use Attainment 8.
    metric_col = "Attainment 8"
    if "engmath_95_total" in top_la.columns:
        # Calculate calculated % manually to be safe
        top_la_agg = top_la.groupby("la_name", as_index=False)[["engmath_95_total", "pupil_count"]].sum()
        top_la_agg["Pass Rate"] = (top_la_agg["engmath_95_total"] / top_la_agg["pupil_count"]) * 100
        metric_col = "Pass Rate"
        y_label = "% Grade 5+ in Eng & Math"
    else:
        # Fallback to Att8
        top_la_agg = top_la.groupby("la_name", as_index=False)[["attainment8_sum", "pupil_count"]].sum()
        top_la_agg["Pass Rate"] = top_la_agg["attainment8_sum"] / top_la_agg["pupil_count"]
        y_label = "Average Attainment 8"

    # Get Top 10
    top_10 = top_la_agg.sort_values("Pass Rate", ascending=False).head(10)
    
    fig_top10 = px.bar(
        top_10,
        x="la_name",
        y="Pass Rate",
        title=f"Local Authority Leaderboard ({latest_year})",
        color="Pass Rate",
        color_continuous_scale="Tealgrn",
        text_auto='.1f',
        labels={"la_name": "Local Authority", "Pass Rate": y_label}
    )
    st.plotly_chart(fig_top10, use_container_width=True)
# =========================================================
# TAB 2: SOCIOECONOMIC PATTERNS (Enhanced Visuals)
# =========================================================
with tab_socio:
    st.subheader("Socioeconomic Patterns")
    st.markdown(
        "This tab explores how educational outcomes vary with deprivation (IDACI), "
        "revealing structural inequalities across the country."
    )

    # ---------------------------------------------------------
    # SECTION 1: THE NATIONAL GRADIENT (Beautiful Bar Charts)
    # ---------------------------------------------------------
    st.markdown("### 1. The Deprivation Gradient")
    st.caption("How strictly do exam results track with neighborhood poverty levels?")
    
    socio_year = st.selectbox("Select Year", all_years, index=len(all_years)-2, key="socio_yr_nat")
    
    # Data Prep
    df_socio = rurality_df[rurality_df["year"] == socio_year].copy()
    df_socio = df_socio[df_socio["idaci_decile"].isin(idaci_bands)]

    # Aggregation
    agg_socio = df_socio.groupby("idaci_decile", as_index=False)[
        ["attainment8_sum", "pupil_count", "progress8_sum", "progress8_pupil_count"]
    ].sum()

    # Calculate Scores
    agg_socio["Attainment 8"] = agg_socio["attainment8_sum"] / agg_socio["pupil_count"]
    agg_socio["Progress 8"] = agg_socio["progress8_sum"] / agg_socio["progress8_pupil_count"]

    c1, c2 = st.columns(2)

    with c1:
        # Custom Plotly Chart: Colored by Performance
        fig_att = px.bar(
            agg_socio, 
            x="idaci_decile", 
            y="Attainment 8",
            color="Attainment 8",  # Color bars by their score
            color_continuous_scale="Blues",
            title="Raw Grades (Attainment 8)",
            labels={"idaci_decile": "Deprivation (0-10% is Poorest)"}
        )
        fig_att.update_layout(coloraxis_showscale=False) # Hide legend to keep it clean
        st.plotly_chart(fig_att, use_container_width=True)

    with c2:
        # Diverging Color Scale for Progress (Positive vs Negative)
        fig_prog = px.bar(
            agg_socio, 
            x="idaci_decile", 
            y="Progress 8",
            color="Progress 8",
            color_continuous_scale="RdBu", # Red for negative, Blue for positive
            title="Value Added (Progress 8)",
            labels={"idaci_decile": "Deprivation (0-10% is Poorest)"}
        )
        fig_prog.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_prog, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2: THE "INEQUALITY HEATMAP" (New Analysis)
    # ---------------------------------------------------------
    st.markdown("### 2. The Geographic Inequality Heatmap")
    st.caption(
        "This chart answers a critical question: **Does poverty matter more than location?** "
        "Read horizontally to see the effect of poverty. Read vertically to see the effect of region."
    )

    # Prepare Heatmap Data (Region x IDACI)
    # We aggregate by BOTH Region and Decile
    heatmap_data = df_socio.groupby(["region_name", "idaci_decile"], as_index=False)[
        ["attainment8_sum", "pupil_count"]
    ].sum()
    
    heatmap_data["Score"] = heatmap_data["attainment8_sum"] / heatmap_data["pupil_count"]
    
    # Pivot for Matrix format
    heatmap_pivot = heatmap_data.pivot(index="region_name", columns="idaci_decile", values="Score")
    # Reorder columns
    heatmap_pivot = heatmap_pivot.reindex(columns=idaci_bands)

    # Render Heatmap
    fig_heat = px.imshow(
        heatmap_pivot,
        labels=dict(x="Deprivation Level", y="Region", color="Attainment 8"),
        x=idaci_bands,
        y=heatmap_pivot.index,
        color_continuous_scale="Viridis", # High contrast scale
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 3: DIRECT REGIONAL COMPARISON (Grouped Bar)
    # ---------------------------------------------------------
    st.markdown("### 3. Direct Comparison")
    st.caption("Compare the poverty gap between two specific regions.")

    col_cmp_1, col_cmp_2 = st.columns(2)
    with col_cmp_1:
        reg_a = st.selectbox("Region A", ["London"] + all_regions, key="comp_reg_a")
    with col_cmp_2:
        reg_b = st.selectbox("Region B", ["North East"] + all_regions, key="comp_reg_b")

    # Filter for both regions
    df_comp = df_socio[df_socio["region_name"].isin([reg_a, reg_b])].copy()
    
    # Aggregate
    agg_comp = df_comp.groupby(["region_name", "idaci_decile"], as_index=False)[
        ["attainment8_sum", "pupil_count"]
    ].sum()
    agg_comp["Attainment 8"] = agg_comp["attainment8_sum"] / agg_comp["pupil_count"]

    # Grouped Bar Chart (The best way to compare)
    fig_comp = px.bar(
        agg_comp,
        x="idaci_decile",
        y="Attainment 8",
        color="region_name", # Group by Region
        barmode="group",     # Place bars side-by-side
        title=f"Head-to-Head: {reg_a} vs {reg_b}",
        labels={"idaci_decile": "Deprivation Decile", "region_name": "Region"},
        color_discrete_sequence=["#3366CC", "#DC3912"] # Distinct colors
    )
    st.plotly_chart(fig_comp, use_container_width=True)
# =========================================================
# TAB 3: OFSTED & GOVERNANCE (Enhanced)
# =========================================================
with tab_ofsted:
    st.subheader("Inspection vs. Reality (2024)")
    st.markdown(
        "This tab investigates the relationship between **Ofsted Ratings** and actual exam performance. "
        "Data is restricted to **2024** to ensure consistency between inspection reports and exam results."
    )
    
    # 1. HARD FILTER FOR 2024 (The only year with KS4 + Ofsted data)
    # ---------------------------------------------------------
    target_year = 2024
    
    # Filter Data
    df_of = ks4_ofsted_df[ks4_ofsted_df["year"] == target_year].copy()
    
    # Ensure we have the critical columns (drop rows where ratings are missing)
    matched_df = df_of.dropna(subset=["ofsted_rating"])
    
    total_count = len(df_of)
    matched_count = len(matched_df)
    
    if total_count > 0:
        match_pct = (matched_count / total_count) * 100
    else:
        match_pct = 0
        
    # --- Top Level Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Schools (2024)", f"{total_count:,}")
    m2.metric("Matched with Ofsted", f"{matched_count:,}")
    m3.metric("Match Rate", f"{match_pct:.1f}%", help="Schools with a linked Ofsted rating.")

    if matched_count > 0:
        color_map = {
            "Outstanding": "#00703c", 
            "Good": "#84bd00", 
            "Requires Improvement": "#ffb81c", 
            "Inadequate": "#d4351c"
        }
        rating_order = ["Outstanding", "Good", "Requires Improvement", "Inadequate"]
        
        # ---------------------------------------------------------
        # ROW 1: Distribution & Subject Breakdown
        # ---------------------------------------------------------
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("#### 1. The Landscape")
            st.caption("Proportion of schools by rating.")
            sb_data = matched_df.groupby(["region_name", "ofsted_rating"], as_index=False).size()
            sb_data.columns = ["Region", "Rating", "Count"]
            
            fig_sun = px.sunburst(
                sb_data, 
                path=["Region", "Rating"], 
                values="Count", 
                color="Rating", 
                color_discrete_map=color_map, 
                height=400
            )
            st.plotly_chart(fig_sun, use_container_width=True)

        with c2:
            st.markdown("#### 2. Subject Specific Performance")
            st.caption("Does an 'Outstanding' badge mean better Math & Science scores? (Progress 8 Components)")
            
            # Define columns
            subject_cols = ["avg_p8eng", "avg_p8mat", "avg_p8ebac"]
            
            # ✅ FIX: Convert columns to numeric before averaging
            # This turns "c", "z", and "NE" into NaN so the crash stops
            for col in subject_cols:
                if col in matched_df.columns:
                    matched_df[col] = pd.to_numeric(matched_df[col], errors="coerce")
            
            # Rename for display
            nice_names = {"avg_p8eng": "English", "avg_p8mat": "Maths", "avg_p8ebac": "Science/Humanities"}
            
            # Now this aggregation will work safely
            subj_agg = matched_df.groupby("ofsted_rating", as_index=False)[subject_cols].mean()
            
            # Reshape for Plotly (Wide to Long format)
            subj_melt = subj_agg.melt(id_vars="ofsted_rating", var_name="Subject", value_name="Score")
            subj_melt["Subject"] = subj_melt["Subject"].map(nice_names)
            
            fig_subj = px.bar(
                subj_melt,
                x="ofsted_rating",
                y="Score",
                color="Subject",
                barmode="group",
                title="Progress 8 Score Breakdown",
                category_orders={"ofsted_rating": rating_order},
                labels={"ofsted_rating": "Ofsted Rating", "Score": "Progress 8 Score (Value Added)"}
            )
            st.plotly_chart(fig_subj, use_container_width=True)

        # ---------------------------------------------------------
        # ROW 2: The "Poverty Trap" (Scatter Plot)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 3. Contextual Analysis: The Poverty Trap")
        st.caption(
            "This chart reveals if schools in deprived areas are judged more harshly. "
            "Ideally, we should see 'Outstanding' dots (Green) across the whole chart. "
            "If they are clustered only on the right (Wealthy areas), it suggests a bias."
        )
        
        # Check if IDACI exists (it should now, thanks to the waterfall merge)
        if "idaci_decile" in matched_df.columns:
            # Drop NaN IDACI for the plot
            scatter_data = matched_df.dropna(subset=["idaci_decile", "avg_att8"])
            
            fig_scatter = px.scatter(
                scatter_data,
                x="idaci_decile", # Or 'The income deprivation...' column if raw
                y="avg_att8",
                color="ofsted_rating",
                color_discrete_map=color_map,
                category_orders={"ofsted_rating": rating_order},
                hover_name="School name", # Using new merged column
                hover_data=["la_name"],
                labels={
                    "idaci_decile": "Deprivation (1=Most Deprived, 5=Least Deprived)",
                    "avg_att8": "Attainment 8 Score"
                },
                height=500
            )
            # Add a trend line or reference
            fig_scatter.update_layout(legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Deprivation data (IDACI) missing for chart.")

        # ---------------------------------------------------------
        # ROW 3: Violin Plots (Spread)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 4. Performance Consistency")
        st.caption("How much variation is there within a single rating?")
        
        v1, v2 = st.columns(2)
        with v1:
            fig_v1 = px.violin(
                matched_df, 
                x="ofsted_rating", 
                y="avg_att8", 
                color="ofsted_rating", 
                color_discrete_map=color_map, 
                category_orders={"ofsted_rating": rating_order}, 
                box=True, 
                points=False,
                title="Raw Grades (Attainment 8)"
            )
            fig_v1.update_layout(showlegend=False)
            st.plotly_chart(fig_v1, use_container_width=True)
            
        with v2:
            fig_v2 = px.violin(
                matched_df, 
                x="ofsted_rating", 
                y="avg_p8score", 
                color="ofsted_rating", 
                color_discrete_map=color_map, 
                category_orders={"ofsted_rating": rating_order}, 
                box=True, 
                points=False,
                title="Value Added (Progress 8)"
            )
            fig_v2.add_hline(y=0, line_dash="dash", annotation_text="National Average")
            fig_v2.update_layout(showlegend=False)
            st.plotly_chart(fig_v2, use_container_width=True)

    else:
        st.error("No matches found between KS4 data and Ofsted data for 2024.")
        st.info("Tip: Ensure your `clean_ks4.csv` has 'school_urn' and `clean_ofsted.csv` has 'URN'.")

# =========================================================
# TAB 4: THE DISADVANTAGE GAP (Detailed & Comprehensive)
# =========================================================
with tab_gap:
    st.subheader("The Disadvantage Gap")
    st.markdown(
        "This module analyzes the structural divide in English education. It measures the 'Gap'—the difference "
        "in performance between the most affluent and most deprived communities—and how Geography and Ofsted interact with it."
    )

    # ---------------------------------------------------------
    # SECTION 1: THE NATIONAL TRENDS
    # ---------------------------------------------------------
    st.markdown("### 1. National Trends")
    
    # Prepare Trend Data (All Years)
    # We calculate the "Gap" for every year available in the data
    if not rurality_df.empty:
        # Filter for the extreme deciles to calculate the gap
        rich_trend = rurality_df[rurality_df["idaci_decile"] == "90-100%"].groupby("year")["attainment8_sum"].sum() / \
                     rurality_df[rurality_df["idaci_decile"] == "90-100%"].groupby("year")["pupil_count"].sum()
        
        poor_trend = rurality_df[rurality_df["idaci_decile"] == "0-10%"].groupby("year")["attainment8_sum"].sum() / \
                     rurality_df[rurality_df["idaci_decile"] == "0-10%"].groupby("year")["pupil_count"].sum()
        
        gap_trend_df = pd.DataFrame({
            "Least Deprived (Rich)": rich_trend,
            "Most Deprived (Poor)": poor_trend
        }).reset_index()
        
        gap_trend_df["The Gap"] = gap_trend_df["Least Deprived (Rich)"] - gap_trend_df["Most Deprived (Poor)"]

        # Layout: 2 Columns
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown("##### Is the Gap Closing?")
            st.caption("Difference in Attainment 8 points between top and bottom deciles over time.")
            
            fig_gap_line = px.line(
                gap_trend_df, 
                x="year", 
                y="The Gap", 
                markers=True,
                color_discrete_sequence=["#E74C3C"], # Red for danger/alert
                title="The Attainment Gap (Points)"
            )
            fig_gap_line.update_layout(yaxis_title="Gap Size (Points)", xaxis_title="Year")
            st.plotly_chart(fig_gap_line, use_container_width=True)

        with g2:
            st.markdown("##### Poverty Landscape")
            st.caption("Median % of FSM (Free School Meal) pupils per school.")
            if not fsm_df.empty and "year" in fsm_df.columns:
                trend_data = fsm_df.groupby("year")["percent_of_pupils"].median().reset_index()
                fig_fsm = px.area(
                    trend_data, 
                    x="year", 
                    y="percent_of_pupils",
                    color_discrete_sequence=["#3498DB"],
                    title="Rising Need: Median FSM %"
                )
                fig_fsm.update_layout(yaxis_title="Median FSM %")
                st.plotly_chart(fig_fsm, use_container_width=True)
            else:
                st.warning("FSM data missing.")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2: REGIONAL GAP ANALYSIS
    # ---------------------------------------------------------
    st.markdown("### 2. The Regional Lottery")
    st.caption("Where is the disadvantage gap the widest? Comparing the 'Penalty of Poverty' across regions.")

    # Controls
    gap_year = st.selectbox("Select Analysis Year", all_years, index=len(all_years)-2, key="gap_yr_2")
    
    # Prepare Data: Group by Region AND Decile
    df_reg_gap = rurality_df[rurality_df["year"] == gap_year].copy()
    
    # We only want the extremes to measure the gap width
    extremes = df_reg_gap[df_reg_gap["idaci_decile"].isin(["0-10%", "90-100%"])].copy()
    
    # Aggregate
    reg_agg = extremes.groupby(["region_name", "idaci_decile"], as_index=False)[["attainment8_sum", "pupil_count"]].sum()
    reg_agg["Score"] = reg_agg["attainment8_sum"] / reg_agg["pupil_count"]
    
    # Pivot to get Side-by-Side columns
    reg_pivot = reg_agg.pivot(index="region_name", columns="idaci_decile", values="Score").reset_index()
    
    # Rename for safety (handle missing columns if data is sparse)
    if "0-10%" in reg_pivot.columns and "90-100%" in reg_pivot.columns:
        reg_pivot["The Gap"] = reg_pivot["90-100%"] - reg_pivot["0-10%"]
        reg_pivot = reg_pivot.sort_values("The Gap", ascending=False)
        
        r1, r2 = st.columns([2, 1])
        
        with r1:
            # Bar Chart: The Gap by Region
            fig_reg_gap = px.bar(
                reg_pivot,
                x="region_name",
                y="The Gap",
                color="The Gap",
                color_continuous_scale="Reds",
                title=f"The Inequality Gap by Region ({gap_year})",
                text_auto='.1f'
            )
            fig_reg_gap.update_layout(xaxis_title=None, yaxis_title="Points Difference (Rich - Poor)")
            st.plotly_chart(fig_reg_gap, use_container_width=True)
            
        with r2:
            st.info(
                """
                **How to read this:**
                
                The taller the bar, the more unequal the education system is in that region.
                
                * **London** typically has a smaller gap (high bars).
                * **The North** often faces wider gaps due to deep-seated structural challenges.
                """
            )
            # Mini Leaderboard
            st.dataframe(
                reg_pivot[["region_name", "The Gap"]].style.background_gradient(cmap="Reds"),
                use_container_width=True,
                height=300
            )
    else:
        st.warning("Not enough data extremes (Rich vs Poor) to calculate regional gaps.")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 3: OFSTED & DISADVANTAGE ("The Fairness Check")
    # ---------------------------------------------------------
    st.markdown("### 3. Ofsted: The Fairness Check")
    st.caption("Do 'Outstanding' ratings just reflect wealthy intakes? Analyzing the correlation between Ratings and Deprivation.")

    # Filter to current year + matching data
    df_fairness = ks4_ofsted_df[ks4_ofsted_df["year"] == gap_year].copy()
    df_fairness = df_fairness.dropna(subset=["ofsted_rating", "idaci_decile"])

    if not df_fairness.empty:
        # Convert IDACI text (e.g., "0-10%") to a number (1-10) for sorting if necessary
        # Or if it's already "Quintile 1-5" from Ofsted file, we map it.
        # Assuming we have "idaci_decile" from the merge.
        
        o1, o2 = st.columns(2)
        
        with o1:
            st.markdown("##### Deprivation Profile by Rating")
            # We want to see the distribution of Poverty (IDACI) for each Rating bucket
            
            # Sort order
            rating_order = ["Outstanding", "Good", "Requires Improvement", "Inadequate"]
            
            # Simple Box Plot
            # Note: For this to work best, we need a numeric poverty score. 
            # If idaci_decile is text ("0-10%"), we map it.
            # Let's create a proxy "Affluence Score" (1=Poor, 10=Rich)
            idaci_map = {
                "0-10%": 1, "10-20%": 2, "20-30%": 3, "30-40%": 4, "40-50%": 5,
                "50-60%": 6, "60-70%": 7, "70-80%": 8, "80-90%": 9, "90-100%": 10
            }
            # Try mapping if it matches our standard format
            df_fairness["Affluence_Score"] = df_fairness["idaci_decile"].map(idaci_map)
            
            # If map failed (because data is Quintiles 1-5), just use the raw column if numeric
            if df_fairness["Affluence_Score"].isna().all():
                 df_fairness["Affluence_Score"] = pd.to_numeric(df_fairness["idaci_decile"], errors="coerce")

            fig_box = px.box(
                df_fairness,
                x="ofsted_rating",
                y="Affluence_Score",
                color="ofsted_rating",
                category_orders={"ofsted_rating": rating_order},
                title="Are Outstanding Schools Wealthier?",
                labels={"Affluence_Score": "Affluence (Higher is Richer)"}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            
        with o2:
            st.markdown("##### Beating the Odds (Scatter)")
            st.caption("Schools in Poor Areas (Left) with High Scores (Top). [Image of scatter plot]")
            
            fig_beat = px.scatter(
                df_fairness,
                x="Affluence_Score",
                y="avg_att8",
                color="ofsted_rating",
                category_orders={"ofsted_rating": rating_order},
                hover_name="School name", # From merged data
                title="Performance vs. Context",
                labels={"Affluence_Score": "Neighborhood Affluence (1=Poor, 10=Rich)", "avg_att8": "Attainment 8"}
            )
            # Add a vertical line for "High Poverty" cutoff
            fig_beat.add_vline(x=3.5, line_dash="dash", line_color="gray", annotation_text="High Poverty Zone")
            st.plotly_chart(fig_beat, use_container_width=True)
            
    else:
        st.warning("No linked Ofsted/Deprivation data available for this year.")

# =========================================================
# TAB 5: CONTEXTUAL OUTLIERS (Data Mining & AI)
# =========================================================
with tab_outliers:
    st.subheader("Contextual Outlier Detection (2024)")
    st.markdown(
        """
        This module performs **Predictive Modelling (OLS Regression)** to answer a complex question: 
        *Which schools are performing exceptionally well, once we remove the advantage of a wealthy catchment area?*
        
        **Data Mining Technique:** Residual Analysis. We predict an 'Expected Score' based on deprivation, 
        then identify schools that deviate significantly from this prediction.
        """
    )

    # 1. Setup & Data Filtering (LOCKED TO 2024)
    # -----------------------------------------------------
    target_year = 2024
    df_model_input = ks4_ofsted_df[ks4_ofsted_df["year"] == target_year].copy()
    
    # Check for Data Availability
    has_idaci = "idaci_decile" in df_model_input.columns
    has_att8 = "avg_att8" in df_model_input.columns
    
    if has_idaci and has_att8:
        # Run the "White Box" Regression from modelling.py
        annotated_df, summary = md.run_school_outlier_analysis(df_model_input)
        
        if not annotated_df.empty:
            # -----------------------------------------------------
            # A. MODEL EXPLAINABILITY & STATS
            # -----------------------------------------------------
            st.markdown("### 1. Model Diagnostics")
            st.caption("How much of a school's grade is determined solely by the poverty level of its neighborhood?")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R-Squared", f"{summary['r2']:.2f}", help="1.0 means Poverty explains 100% of the grade difference.")
            m2.metric("Base Performance", f"{summary['intercept']:.1f}", help="The theoretical starting grade for a school with high deprivation.")
            m3.metric("Poverty Coefficient", f"{summary['poverty_coef']:.2f}", 
                      help="For every step up the wealth ladder, grades increase by this amount.",
                      delta="Impact Factor")
            m4.metric("Schools Analyzed", f"{len(annotated_df):,}")

            st.markdown("---")

            # -----------------------------------------------------
            # B. VISUALIZATION 1: The Regression (Scatter)
            # -----------------------------------------------------
            c_scat, c_resid = st.columns([2, 1])
            
            with c_scat:
                st.markdown("#### 2. Expected vs. Actual Performance")
                st.caption("Schools above the trendline are 'beating the odds'.")
                
                fig_resid = px.scatter(
                    annotated_df,
                    x="poverty_index",
                    y="avg_att8",
                    color="outlier_status",
                    color_discrete_map={
                        "🟢 Over-performing": "#2E7D32", 
                        "⚪ As Expected": "#BDBDBD", 
                        "🔴 Under-performing": "#C62828"
                    },
                    hover_name="School name",
                    hover_data={"residual": ":.2f", "poverty_index": False, "avg_att8": True},
                    trendline="ols", 
                    labels={"poverty_index": "Affluence Index (1=Poor, 10=Rich)", "avg_att8": "Actual Attainment 8"},
                    title="The Regression Line",
                    height=450
                )
                st.plotly_chart(fig_resid, use_container_width=True)
            
            with c_resid:
                st.markdown("#### 3. Residual Distribution")
                st.caption("Are the errors normally distributed? (A bell curve indicates a healthy model).")
                
                fig_hist = px.histogram(
                    annotated_df,
                    x="residual",
                    nbins=30,
                    color="outlier_status",
                    color_discrete_map={
                        "🟢 Over-performing": "#2E7D32", 
                        "⚪ As Expected": "#BDBDBD", 
                        "🔴 Under-performing": "#C62828"
                    },
                    title="Distribution of Residuals"
                )
                fig_hist.add_vline(x=0, line_dash="dash", annotation_text="Exp")
                fig_hist.update_layout(showlegend=False, xaxis_title="Deviation from Expected Grade")
                st.plotly_chart(fig_hist, use_container_width=True)

            # -----------------------------------------------------
            # C. VISUALIZATION 2: Correlation & Validation
            # -----------------------------------------------------
            st.markdown("---")
            c_corr, c_box = st.columns(2)
            
            with c_corr:
                st.markdown("#### 4. Correlation Analysis")
                st.caption("How strong are the links between Poverty, Grades, and Progress?")
                
                # Prepare correlation matrix
                corr_cols = ["poverty_index", "avg_att8", "avg_p8score", "residual"]
                # Convert to numeric first
                for c in corr_cols:
                    if c in annotated_df.columns:
                        annotated_df[c] = pd.to_numeric(annotated_df[c], errors='coerce')
                
                corr_matrix = annotated_df[corr_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r", # Red = Negative Correlation
                    title="Feature Correlation Matrix",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            with c_box:
                st.markdown("#### 5. Validation: Progress 8")
                st.caption("Do our 'Statistical Outliers' actually have high Value Added scores?")
                
                fig_box = px.box(
                    annotated_df,
                    x="outlier_status",
                    y="avg_p8score", 
                    color="outlier_status",
                    color_discrete_map={
                        "🟢 Over-performing": "#2E7D32", 
                        "⚪ As Expected": "#BDBDBD", 
                        "🔴 Under-performing": "#C62828"
                    },
                    title="Progress 8 by Outlier Group",
                    labels={"avg_p8score": "Progress 8 Score"}
                )
                st.plotly_chart(fig_box, use_container_width=True)

            # -----------------------------------------------------
            # D. VISUALIZATION 3: Geographical Hotspots [Image of Map]

            # -----------------------------------------------------
            st.markdown("---")
            st.markdown("#### 6. Where are the Over-Performers?")
            st.caption("Regional breakdown of schools identified as 'Green' outliers.")
            
            # Filter for positive outliers only
            greens = annotated_df[annotated_df["outlier_status"] == "🟢 Over-performing"]
            
            if not greens.empty:
                geo_counts = greens["region_name"].value_counts().reset_index()
                geo_counts.columns = ["Region", "Count"]
                
                fig_geo_bar = px.bar(
                    geo_counts,
                    x="Region",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Greens",
                    title="Count of Over-Performing Schools by Region"
                )
                st.plotly_chart(fig_geo_bar, use_container_width=True)
            
            # -----------------------------------------------------
            # E. DRILL DOWN TABLE
            # -----------------------------------------------------
            st.markdown("### 7. The 'Beating the Odds' Leaderboard")
            st.caption("Top 20 schools with the highest positive residual (Actual Grade > Expected Grade).")
            
            # Sort by residual
            top_performers = greens.sort_values("residual", ascending=False).head(20)
            
            # Select display columns
            cols_to_show = ["school_urn", "School name", "region_name", "avg_att8", "poverty_index", "residual"]
            final_cols = [c for c in cols_to_show if c in top_performers.columns]
            
            st.dataframe(
                top_performers[final_cols].style.format({"residual": "{:.2f}", "avg_att8": "{:.1f}", "poverty_index": "{:.0f}"})
                .background_gradient(subset=["residual"], cmap="Greens"),
                use_container_width=True
            )
            
        else:
            st.warning("Model ran but returned no data. Check if IDACI/Att8 columns have valid numbers.")
    else:
        # Fallback if IDACI column is missing
        st.error("⚠️ Critical Data Missing: 'idaci_decile' or 'avg_att8'")
        st.markdown(
            "The model requires both an outcome (Attainment 8) and a predictor (IDACI). "
            "One of these is missing from your 2024 dataset."
        )