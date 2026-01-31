# utils.py
import pandas as pd

def closing_year(time_period: int) -> int:
    """Convert DfE time_period (202324) to year (2024)."""
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