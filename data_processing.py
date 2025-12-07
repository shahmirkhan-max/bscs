"""
Data processing for:
- Schools, pupils and their characteristics (SPC) files

This version:
- Works at Local Authority (LA) × school_type (phase_type_grouping) level
- Uses the actual SPC filenames you listed
- Does NOT require URN (the SPC files are not school-level)
"""

from typing import List, Optional, Dict

import numpy as np
import pandas as pd


# ---------- Generic helpers ---------- #

def extract_closing_year(year_value) -> Optional[int]:
    """
    Convert academic year codes like 201516 or 202122 into the closing year (e.g. 2016, 2022).

    Rules:
    - If value is already 4 digits → return as int.
    - If value is 6 digits 'YYYYYY' → take first 4 digits and add 1.
      e.g. 202122 → 2021 + 1 = 2022
    """
    if pd.isna(year_value):
        return None

    s = str(year_value).strip()

    if len(s) == 4 and s.isdigit():
        return int(s)

    if len(s) == 6 and s.isdigit():
        try:
            start = int(s[:4])
            return start + 1
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
    Standardise column names: lower_snake_case, strip whitespace.
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
    Add a 'year' column from time_period / academic_year.

    The SPC files use 'time_period' with codes like 201516, 202122.
    """
    df = df.copy()
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return df

    year_col = existing[0]
    df["year"] = df[year_col].apply(extract_closing_year)
    return df


def filter_local_authority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Local Authority rows (geographic_level == 'Local authority').
    """
    df = df.copy()
    if "geographic_level" in df.columns:
        mask = df["geographic_level"].str.lower().str.replace(" ", "_") == "local_authority"
        df = df[mask]
    return df


def load_spc_file(path: str) -> pd.DataFrame:
    """
    Load a generic SPC CSV and apply standard cleaning.
    """
    df = pd.read_csv(path)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period"])
    df = filter_local_authority(df)
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def common_key_cols(df: pd.DataFrame) -> List[str]:
    """
    Common LA-level key columns used across SPC files.
    """
    candidates = ["year", "country_code", "country_name",
                  "region_code", "region_name", "la_code", "la_name", "school_type"]
    return [c for c in candidates if c in df.columns]


# ---------- File-specific cleaners ---------- #

def clean_spc_school_characteristics(path: str) -> pd.DataFrame:
    """
    SPC: spc_school_characteristics.csv

    We keep LA-level pupil counts and basic structure:
    - school_type (phase_type_grouping)
    - cohort_size (headcount_of_pupils)
    """
    df = load_spc_file(path)

    # Rename phase_type_grouping → school_type
    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    # We mainly need counts at LA × school_type
    cols_keep = common_key_cols(df) + [
        c for c in ["headcount_of_pupils", "number_of_schools",
                    "denomination", "sex_of_school_description",
                    "admissions_policy", "urban_rural"] if c in df.columns
    ]
    df = df[cols_keep].drop_duplicates()

    # cohort_size = total headcount in that LA × school_type
    if "headcount_of_pupils" in df.columns:
        group_cols = [c for c in common_key_cols(df) if c != "country_code"]
        df = (
            df.groupby(group_cols, as_index=False)["headcount_of_pupils"]
            .sum()
        )
        df = df.rename(columns={"headcount_of_pupils": "cohort_size"})

    return df


def clean_spc_pupils_fsm(path: str) -> pd.DataFrame:
    """
    SPC: spc_pupils_fsm.csv

    Build FSM% at LA × school_type.
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    # Keep FSM-eligible rows
    if "fsm" in df.columns:
        eligible_mask = df["fsm"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible_mask]

    keys = common_key_cols(df)
    fsm = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm_percent"})
    )
    return fsm


def clean_spc_fsm6(path: str) -> pd.DataFrame:
    """
    SPC: spc_fsm6.csv

    FSM6% (eligible at any point in last 6 years) at LA × school_type.
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm6_eligibility" in df.columns:
        eligible_mask = df["fsm6_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible_mask]

    keys = common_key_cols(df)
    fsm6 = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm6_percent"})
    )
    return fsm6


def clean_spc_pupils_fsm_ethnicity_yrgp(path: str) -> pd.DataFrame:
    """
    SPC: spc_pupils_fsm_ethnicity_yrgp.csv

    Very detailed file. For now, we just construct a single 'disadvantaged_percent'
    using rows where:
    - characteristic == 'Total' (if present)
    - fsm_eligibility contains 'eligible'
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm_eligibility" in df.columns:
        eligible_mask = df["fsm_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible_mask]

    if "characteristic" in df.columns:
        total_mask = df["characteristic"].astype(str).str.lower() == "total"
        if total_mask.any():
            df = df[total_mask]

    keys = common_key_cols(df)
    disadv = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "disadvantaged_percent"})
    )
    return disadv


def clean_spc_pupils_ethnicity_and_language(path: str) -> pd.DataFrame:
    """
    SPC: spc_pupils_ethnicity_and_language.csv

    Build:
    - Ethnicity share columns: eth_<ethnicity_minor>
    - eal_percent: % of pupils whose first language is other than English
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    keys = common_key_cols(df)

    # --- Ethnicity shares ---
    if "ethnicity_minor" in df.columns:
        eth = (
            df.groupby(keys + ["ethnicity_minor"], as_index=False)["percent_of_pupils"]
            .mean()
        )
        eth_wide = eth.pivot_table(
            index=keys,
            columns="ethnicity_minor",
            values="percent_of_pupils",
            aggfunc="first"
        )
        eth_wide.columns = [
            "eth_" + str(c).lower().replace(" ", "_").replace("/", "_")
            for c in eth_wide.columns
        ]
        eth_wide = eth_wide.reset_index()
    else:
        eth_wide = pd.DataFrame(columns=keys)

    # --- EAL% from language ---
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

    # Merge ethnicity + eal
    merged = pd.merge(eth_wide, eal, on=keys, how="outer")
    return merged


def clean_spc_pupils_age_and_sex(path: str) -> pd.DataFrame:
    """
    SPC: spc_pupils_age_and_sex.csv

    Build gender proportions:
    - pct_female
    - pct_male
    at LA × school_type.
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["headcount"])

    keys = common_key_cols(df)
    if "sex" not in df.columns or "headcount" not in df.columns:
        return df.iloc[0:0][keys]  # empty but with keys

    g = (
        df.groupby(keys + ["sex"], as_index=False)["headcount"]
        .sum()
    )

    pivot = g.pivot_table(
        index=keys,
        columns="sex",
        values="headcount",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Standardise columns "Female", "Male" if they exist
    cols = list(pivot.columns)
    male_col = next((c for c in cols if str(c).lower().startswith("male")), None)
    female_col = next((c for c in cols if str(c).lower().startswith("female")), None)

    if male_col is None or female_col is None:
        pivot["pct_male"] = np.nan
        pivot["pct_female"] = np.nan
        return pivot

    total = pivot[male_col] + pivot[female_col]
    pivot["pct_male"] = np.where(total > 0, 100 * pivot[male_col] / total, np.nan)
    pivot["pct_female"] = np.where(total > 0, 100 * pivot[female_col] / total, np.nan)

    # Drop raw counts to keep it clean
    pivot = pivot[keys + ["pct_male", "pct_female"]]
    return pivot


def clean_spc_uifsm(path: str) -> pd.DataFrame:
    """
    SPC: spc_uifsm.csv

    Build:
    - uifsm_percent_total: % infant pupils taking a UIFSM meal (Total characteristic).
    """
    df = load_spc_file(path)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percentage"])

    if "characteristic" in df.columns:
        mask_total = df["characteristic"].astype(str).str.lower() == "total"
        if mask_total.any():
            df = df[mask_total]

    keys = common_key_cols(df)
    uifsm = (
        df.groupby(keys, as_index=False)["percentage"]
        .mean()
        .rename(columns={"percentage": "uifsm_percent_total"})
    )
    return uifsm


def clean_spc_cbm(path: str) -> pd.DataFrame:
    """
    SPC: spc_cbm.csv

    The cross-border movement file is complex (LAs as columns).
    For now we simply return it *untouched* so you have access
    in case you want to explore it separately in the dashboard.

    It is NOT merged into the main modelling dataset yet.
    """
    df = load_spc_file(path)
    return df


def clean_performance_data(path: str) -> pd.DataFrame:
    """
    Placeholder for KS4/KS5 performance data (NOT an SPC file).

    We assume it has:
      - time_period or year
      - la_code, la_name, region_code, region_name
      - a performance column: attainment8, progress8, or performance_score
    """
    df = pd.read_csv(path)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period", "academic_year"])

    perf_cols = ["attainment8", "progress8", "performance_score"]
    perf_cols = [c for c in perf_cols if c in df.columns]
    df = ensure_numeric(df, perf_cols)

    return df


# ---------- MERGE PIPELINE ---------- #

def _safe_left_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Left merge using the intersection of common key columns.
    If there are no common key columns, return left unchanged.
    """
    if right is None or right.empty:
        return left

    left_keys = common_key_cols(left)
    right_keys = common_key_cols(right)
    keys = [k for k in left_keys if k in right_keys]

    if not keys:
        return left

    return left.merge(right, on=keys, how="left")


def build_analysis_dataset(
    fsm_path: str,
    fsm6_path: str,
    fsm_ethnicity_path: str,
    age_sex_path: str,
    eth_lang_path: str,
    school_char_path: str,
    uifsm_path: Optional[str] = None,
    cbm_path: Optional[str] = None,
    performance_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    High-level pipeline.

    Returns an LA × school_type dataset with:
    - region_name, la_name, school_type
    - cohort_size
    - fsm_percent, fsm6_percent, disadvantaged_percent
    - eal_percent, ethnicity proportions
    - pct_male, pct_female
    - uifsm_percent_total
    - performance outcomes (if provided)
    """
    school_char = clean_spc_school_characteristics(school_char_path)
    fsm = clean_spc_pupils_fsm(fsm_path)
    fsm6 = clean_spc_fsm6(fsm6_path)
    fsm_eth = clean_spc_pupils_fsm_ethnicity_yrgp(fsm_ethnicity_path)
    age_sex = clean_spc_pupils_age_and_sex(age_sex_path)
    eth_lang = clean_spc_pupils_ethnicity_and_language(eth_lang_path)

    uifsm = clean_spc_uifsm(uifsm_path) if uifsm_path is not None else None
    _ = clean_spc_cbm(cbm_path) if cbm_path is not None else None  # loaded but not merged

    perf = clean_performance_data(performance_path) if performance_path is not None else None

    merged = school_char.copy()
    merged = _safe_left_merge(merged, fsm)
    merged = _safe_left_merge(merged, fsm6)
    merged = _safe_left_merge(merged, fsm_eth)
    merged = _safe_left_merge(merged, age_sex)
    merged = _safe_left_merge(merged, eth_lang)
    merged = _safe_left_merge(merged, uifsm)
    merged = _safe_left_merge(merged, perf)

    # If disadvantaged_percent not set, fall back to fsm6_percent or fsm_percent
    if "disadvantaged_percent" not in merged.columns:
        if "fsm6_percent" in merged.columns:
            merged["disadvantaged_percent"] = merged["fsm6_percent"]
        elif "fsm_percent" in merged.columns:
            merged["disadvantaged_percent"] = merged["fsm_percent"]

    return merged
