# app.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="School Performance Prototype", layout="wide")

st.title("School Performance Explorer – Prototype")
st.write(
    "This is an early technical prototype built in Streamlit. "
    "For now, it uses synthetic data to demonstrate the core layout and interactions."
)

# --- Fake data for now (will be replaced by real GCSE/A-level data) ---
np.random.seed(42)
years = np.arange(2018, 2023)
imd_deciles = np.arange(1, 11)

data = []
for y in years:
    for d in imd_deciles:
        # simple synthetic pattern: higher IMD decile → higher attainment
        base = 40 + d * 0.8
        noise = np.random.normal(0, 2)
        data.append({"Year": y, "IMD_decile": d, "Attainment8": base + noise})

df = pd.DataFrame(data)

# --- Sidebar filters ---
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
metric = st.sidebar.selectbox("Metric", ["Attainment8"])

view = df[df["Year"] == year]

st.subheader(f"{metric} by IMD decile ({year})")
st.line_chart(
    view.set_index("IMD_decile")[[metric]]
)

st.caption(
    "Synthetic data only – this prototype shows the structure of the system. "
    "In the final version, this view will be driven by merged school performance "
    "and deprivation datasets."
)
