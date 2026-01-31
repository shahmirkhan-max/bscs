import streamlit as st
import pandas as pd
import requests
import os
from io import StringIO
from utils import closing_year

# =========================================================
# DATA LOADER CLASS
# =========================================================

class SchoolDataLoader:
    """
    Handles loading data from either local clean CSVs (priority) 
    or Google Drive (fallback).
    """
    def __init__(self, use_local_files: bool = True):
        self.use_local = use_local_files
        
        # Map logical keys to local filenames
        self.local_files = {
            "rurality": "clean_rurality.csv",
            "geo": "clean_geo.csv",
            "fsm": "clean_fsm.csv",
            "ks4": "clean_ks4.csv",
            "ofsted": "clean_ofsted.csv"
        }
        
        # Map logical keys to Google Drive IDs
        self.drive_ids = {
            "rurality": "1lSNdDytkgLKvgTsuy0hfRwuhHt22BFVE",
            "geo": "1R703u8VB2TDSPZaOXhRN2M5dDX4YdPoR",
            "fsm": "1DPrbSj5mb3DcM2NyVMkeUvJB51Uyp4aj",
            "ks4": "1m5Rm4vHFevaYZa_kyImoagOtF5rUcJUE",
            "ofsted": "1mKMIf_uF8zxqjK_JBBiQQvQ6zTB8IA7c",
        }

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes BOM characters and whitespace from column names."""
        # Fix the specific BOM issue (ï»¿)
        df.columns = df.columns.str.replace('ï»¿', '', regex=False)
        df.columns = df.columns.str.strip()
        return df

    def _load_from_drive(self, file_id: str) -> pd.DataFrame:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            # Use utf-8-sig to handle BOM automatically
            return self._clean_columns(pd.read_csv(StringIO(resp.content.decode("utf-8-sig")), low_memory=False))
        except UnicodeDecodeError:
            return self._clean_columns(pd.read_csv(StringIO(resp.content.decode("latin1")), low_memory=False))
        except Exception as e:
            st.error(f"Failed to load from Drive: {e}")
            return pd.DataFrame()

    def get_data(self, key: str) -> pd.DataFrame:
        """Main entry point to get a dataset."""
        # 1. Try Local
        if self.use_local:
            filename = self.local_files.get(key)
            if filename and os.path.exists(filename):
                try:
                    # ✅ CRITICAL FIX: Use 'utf-8-sig' to strip the invisible BOM character
                    return self._clean_columns(pd.read_csv(filename, low_memory=False, encoding='utf-8-sig'))
                except UnicodeDecodeError:
                    return self._clean_columns(pd.read_csv(filename, low_memory=False, encoding='latin1'))
            else:
                st.warning(f"⚠️ Local file '{filename}' not found. Falling back to Cloud.")

        # 2. Fallback to Drive
        file_id = self.drive_ids.get(key)
        if file_id:
            return self._load_from_drive(file_id)
        
        return pd.DataFrame()

# Instantiate the loader once
loader = SchoolDataLoader(use_local_files=True)

# =========================================================
# PUBLIC FUNCTIONS (Called by app.py)
# =========================================================

def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to ensure 'year' column exists."""
    if "year" not in df.columns and "time_period" in df.columns:
        df["year"] = df["time_period"].apply(closing_year)
    return df

@st.cache_data
def load_rurality():
    df = loader.get_data("rurality")
    
    # Ensure numeric columns
    cols = ["attainment8_sum", "pupil_count", "progress8_sum", "progress8_pupil_count", "school_count"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = _ensure_year_column(df)
    
    if "idaci_decile" in df.columns:
        df["idaci_band"] = df["idaci_decile"].fillna("Unknown")
        
    return df

@st.cache_data
def load_geo():
    df = loader.get_data("geo")
    
    cols = ["attainment8_sum", "pupil_count", "progress8_sum", "progress8_pupil_count"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    df = _ensure_year_column(df)
    return df

@st.cache_data
def load_fsm_secondary():
    df = loader.get_data("fsm")
    df = _ensure_year_column(df)

    # Filter Logic
    if "phase_type_grouping" in df.columns:
        mask = (df["phase_type_grouping"] == "State-funded secondary")
        if "fsm" in df.columns:
             mask = mask & (df["fsm"] == "known to be eligible for free school meals (used for FSM in Performance Tables)")
        df = df.loc[mask].copy()
    
    # Cap percentages
    if "percent_of_pupils" in df.columns:
        df["percent_of_pupils"] = pd.to_numeric(df["percent_of_pupils"], errors="coerce")
        df = df[df["percent_of_pupils"] <= 100]
        
    return df

@st.cache_data
def load_ks4_with_ofsted():
    # ==========================================
    # 1. LOAD & CLEAN DATA
    # ==========================================
    ks4 = loader.get_data("ks4")
    ks4 = _ensure_year_column(ks4)
    if "breakdown" in ks4.columns:
        ks4 = ks4[ks4["breakdown"] == "Total"].copy()
    
    # Clean KS4 Numeric IDs
    for c in ["avg_att8", "avg_p8score", "school_urn"]:
        if c in ks4.columns:
            ks4[c] = pd.to_numeric(ks4[c], errors="coerce")

    # Load Ofsted
    mi = loader.get_data("ofsted")
    
    # Rename Ofsted columns for clarity and App compatibility
    # We rename 'URN' to 'ofsted_urn' to avoid collision with KS4 'school_urn'
    # We Rename 'Region' to 'region_name' for the graphs
    rename_map = {
        "Region": "region_name",
        "Local authority": "la_name_ofsted", # Rename to avoid collision with KS4 'la_name'
        "The income deprivation affecting children index (IDACI) quintile": "idaci_decile",
        "URN": "ofsted_urn",
        "URN at time of previous inspection": "previous_urn"
    }
    mi = mi.rename(columns=rename_map)

    # Ensure IDs are numeric for matching
    for c in ["ofsted_urn", "previous_urn"]:
        if c in mi.columns:
            mi[c] = pd.to_numeric(mi[c], errors="coerce")

    # Handle Ratings
    # If numeric (1-4), map it. If text, leave it.
    if "ofsted_rating" in mi.columns and pd.api.types.is_numeric_dtype(mi["ofsted_rating"]):
        rating_map = {1.0: "Outstanding", 2.0: "Good", 3.0: "Requires Improvement", 4.0: "Inadequate"}
        mi["ofsted_rating"] = mi["ofsted_rating"].map(rating_map)
    elif "Overall effectiveness" in mi.columns and "ofsted_rating" not in mi.columns:
         # Fallback creation
         mi["ofsted_rating"] = pd.to_numeric(mi["Overall effectiveness"], errors="coerce").map(
             {1: "Outstanding", 2: "Good", 3: "Requires Improvement", 4: "Inadequate"}
         )

    # ==========================================
    # 2. THE WATERFALL MERGE STRATEGY
    # ==========================================
    
    # --- ATTEMPT 1: Primary Match (Current URN) ---
    # We merge ALL columns from mi (no filtering)
    print("   ...Attempting Primary Merge on Current URN")
    merged_1 = ks4.merge(
        mi, 
        left_on="school_urn", 
        right_on="ofsted_urn", 
        how="left",
        suffixes=("", "_ofsted") # Keep KS4 names pure, add suffix to Ofsted collisions
    )

    # Identify which schools found a match and which didn't
    # We check 'ofsted_urn' because it comes from the Ofsted file. If it's NaN, the merge failed.
    mask_match = merged_1["ofsted_urn"].notna()
    
    found_df = merged_1[mask_match].copy()
    missing_df = merged_1[~mask_match].copy()

    # If we have missing rows, try to rescue them
    if not missing_df.empty and "previous_urn" in mi.columns:
        print(f"   ...Attempting Secondary Merge for {len(missing_df)} schools")
        
        # 1. Get the original KS4 data for these specific missing schools
        # We go back to the source 'ks4' dataframe to avoid columns polluted with NaNs/Suffixes
        missing_urns = missing_df["school_urn"].unique()
        ks4_missing_clean = ks4[ks4["school_urn"].isin(missing_urns)].copy()

        # --- ATTEMPT 2: Secondary Match (Previous URN) ---
        # We try to match KS4's 'school_urn' to Ofsted's 'previous_urn'
        merged_2 = ks4_missing_clean.merge(
            mi, 
            left_on="school_urn", 
            right_on="previous_urn", 
            how="left",
            suffixes=("", "_ofsted")
        )
        
        # Combine the successes from Round 1 and Round 2
        final_merged = pd.concat([found_df, merged_2], ignore_index=True)
        
    else:
        final_merged = merged_1

    # Final Cleanup: Fill 'la_name' if missing in KS4 but present in Ofsted
    if "la_name" in final_merged.columns and "la_name_ofsted" in final_merged.columns:
        final_merged["la_name"] = final_merged["la_name"].fillna(final_merged["la_name_ofsted"])

    return final_merged