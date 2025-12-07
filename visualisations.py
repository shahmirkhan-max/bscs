"""
visualisations.py

Plotly visualisations for:

- FSM% vs performance (with regression line)
- EAL% vs performance
- Average house price vs performance (UK HPI)
- Heatmap: Region × FSM × Performance
- Ethnicity composition by region
- Boxplot: Performance by school type
- Residual vs fitted plot
- Residual histogram
- Cook's distance bar chart

All functions return Plotly Figure objects that are Streamlit-compatible.
"""

from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Generic scatter with OLS trendline
# ---------------------------------------------------------------------

def scatter_with_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: str = "",
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> go.Figure:
    """
    Generic scatter with OLS trend line using Plotly Express.

    Parameters
    ----------
    df : DataFrame
        Data to plot (already filtered for non-null x/y).
    x, y : str
        Column names for x and y.
    color : str, optional
        Optional column for colour grouping (e.g. region_name).
    hover_data : list of str, optional
        Extra columns to show on hover.
    title : str
        Figure title.
    x_title, y_title : str, optional
        Axis labels (if None, use column names).

    Returns
    -------
    Figure
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        trendline="ols",
        title=title,
    )

    fig.update_layout(
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
        legend_title_text=color if color else "",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------
# Specific scatter plots used in the project
# ---------------------------------------------------------------------

def scatter_fsm_vs_performance(
    df: pd.DataFrame,
    fsm_col: str,
    performance_col: str,
    region_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
) -> go.Figure:
    """
    Scatter: FSM% vs performance with regression line.
    """
    title = f"{fsm_col} vs {performance_col}"
    return scatter_with_trend(
        df=df,
        x=fsm_col,
        y=performance_col,
        color=region_col,
        hover_data=hover_data,
        title=title,
        x_title="FSM (%)",
        y_title=performance_col,
    )


def scatter_eal_vs_performance(
    df: pd.DataFrame,
    eal_col: str,
    performance_col: str,
    region_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
) -> go.Figure:
    """
    Scatter: EAL% vs performance with regression line.
    """
    title = f"{eal_col} vs {performance_col}"
    return scatter_with_trend(
        df=df,
        x=eal_col,
        y=performance_col,
        color=region_col,
        hover_data=hover_data,
        title=title,
        x_title="EAL (%)",
        y_title=performance_col,
    )


def scatter_house_price_vs_performance(
    df: pd.DataFrame,
    house_price_col: str,
    performance_col: str,
    region_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
) -> go.Figure:
    """
    Scatter: Average house price vs performance with regression line.

    Typically used with:
    - house_price_col = 'avg_house_price'
    - performance_col = 'attainment8' / 'progress8' / 'performance_score'
    """
    title = f"Average house price vs {performance_col}"
    return scatter_with_trend(
        df=df,
        x=house_price_col,
        y=performance_col,
        color=region_col,
        hover_data=hover_data,
        title=title,
        x_title="Average house price (£)",
        y_title=performance_col,
    )


# ---------------------------------------------------------------------
# Heatmap: Region × FSM × Performance
# ---------------------------------------------------------------------

def heatmap_region_fsm_performance(
    df: pd.DataFrame,
    region_col: str,
    fsm_col: str,
    performance_col: str,
    title: str = "Region × FSM × Performance",
    n_bins: int = 4,
) -> go.Figure:
    """
    Heatmap of average performance by region and FSM% quantile.

    Parameters
    ----------
    df : DataFrame
        Data to summarise.
    region_col : str
        Region column (e.g. 'region_name').
    fsm_col : str
        FSM percentage column.
    performance_col : str
        Performance metric (e.g. 'attainment8').
    title : str
        Plot title.
    n_bins : int
        Number of FSM quantile bins to create.
    """
    df = df.dropna(subset=[region_col, fsm_col, performance_col]).copy()

    # Quantile bins for FSM%
    df["fsm_bin"] = pd.qcut(
        df[fsm_col],
        q=n_bins,
        duplicates="drop",
    )

    # Order labels nicely
    df["fsm_bin"] = df["fsm_bin"].astype(str)

    pivot = (
        df.groupby([region_col, "fsm_bin"], as_index=False)[performance_col]
        .mean()
        .pivot(index=region_col, columns="fsm_bin", values=performance_col)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorbar=dict(title=performance_col),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="FSM% quantile group",
        yaxis_title="Region",
        margin=dict(l=80, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------
# Ethnicity composition by region
# ---------------------------------------------------------------------

def bar_ethnicity_composition_by_region(
    df: pd.DataFrame,
    region_col: str,
    ethnicity_cols: List[str],
    title: str = "Ethnicity composition by region",
) -> go.Figure:
    """
    Stacked bar chart: average ethnicity composition by region.

    Parameters
    ----------
    df : DataFrame
        Data to aggregate.
    region_col : str
        Region column name.
    ethnicity_cols : list of str
        Columns containing ethnicity proportions (e.g. 'eth_white_british').
    """
    df = df.dropna(subset=[region_col]).copy()
    df_agg = df.groupby(region_col)[ethnicity_cols].mean().reset_index()

    fig = go.Figure()
    for col in ethnicity_cols:
        fig.add_trace(
            go.Bar(
                x=df_agg[region_col],
                y=df_agg[col],
                name=col,
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Region",
        yaxis_title="Average proportion (%)",
        legend_title_text="Ethnicity",
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis=dict(tickangle=45),
    )
    return fig


# ---------------------------------------------------------------------
# Boxplot: performance by school type
# ---------------------------------------------------------------------

def boxplot_performance_by_school_type(
    df: pd.DataFrame,
    performance_col: str,
    school_type_col: str,
    title: str = "Performance by school type",
) -> go.Figure:
    """
    Boxplot of performance by school type (phase or governance grouping).
    """
    df = df.dropna(subset=[performance_col, school_type_col]).copy()

    fig = px.box(
        df,
        x=school_type_col,
        y=performance_col,
        points="outliers",
        title=title,
    )
    fig.update_layout(
        xaxis_title="School type / phase",
        yaxis_title=performance_col,
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis=dict(tickangle=45),
    )
    return fig


# ---------------------------------------------------------------------
# Regression diagnostics
# ---------------------------------------------------------------------

def residual_plot(
    fitted: pd.Series,
    residuals: pd.Series,
    title: str = "Residuals vs fitted values",
) -> go.Figure:
    """
    Scatter of fitted values vs residuals with zero line.
    """
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

    fig.add_hline(
        y=0,
        line=dict(dash="dash"),
        annotation_text="0",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fitted values",
        yaxis_title="Residuals",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def residual_histogram(
    residuals: pd.Series,
    title: str = "Distribution of residuals",
) -> go.Figure:
    """
    Histogram of regression residuals.
    """
    fig = px.histogram(
        residuals,
        nbins=30,
        title=title,
    )
    fig.update_layout(
        xaxis_title="Residual",
        yaxis_title="Frequency",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def cooks_distance_bar(
    cooks_distance: pd.Series,
    top_n: int = 20,
    title: str = "Top observations by Cook's distance",
) -> go.Figure:
    """
    Bar chart of the top-N most influential observations (Cook's distance).

    Parameters
    ----------
    cooks_distance : Series
        Cook's distance values with an index that identifies rows (e.g. df.index).
    top_n : int
        Number of observations to display.
    """
    cd = cooks_distance.sort_values(ascending=False).head(top_n)
    fig = go.Figure(
        data=[
            go.Bar(
                x=cd.index.astype(str),
                y=cd.values,
                name="Cook's distance",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Observation index",
        yaxis_title="Cook's distance",
        margin=dict(l=40, r=20, t=60, b=80),
        xaxis=dict(tickangle=45),
    )
    return fig
