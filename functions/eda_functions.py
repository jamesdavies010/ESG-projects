import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def chart_visualisations(
    df: pd.DataFrame,
    columns_to_include: list,
    legend_column: str = None,
    n_cols: int = 3,
) -> go.Figure:
    """
    Creates a 3-column subplot visualization for discrete columns in the provided DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to visualize.

    columns_to_include : list
        A list of discrete columns to visualize.

    legend_column : str, optional
        The name of the column that differentiates the data groups. If None, no grouping is applied.

    n_cols : int, optional
        The number of columns in the subplot layout.

    Returns:
    -------
    go.Figure
        The final Plotly figure object.
    """
    # Identify relevant columns that exist in df
    relevant_columns = [col for col in columns_to_include if col in df.columns]

    # Identify integer columns (excluding "year" and "years_esg_data")
    int_columns = set(
        col
        for col in relevant_columns
        if df[col].dtype == "int64" and col not in {"year", "years_esg_data"}
    )

    n_rows = -(-len(relevant_columns) // n_cols)  # Ceiling division
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=relevant_columns)

    # Predefine colors for integer columns
    color_map = {0: "rgb(254,240,205)", 1: "rgb(6,212,124)"}

    for idx, col in enumerate(relevant_columns):
        row_num = (idx // n_cols) + 1
        col_num = (idx % n_cols) + 1

        grouped_data = df[col].value_counts().sort_index()

        if col in int_columns:
            colors = np.array(
                [color_map.get(x, "rgb(31, 119, 180)") for x in grouped_data.index]
            )
        else:
            colors = "rgb(31, 119, 180)"  # Default color

        fig.add_trace(
            go.Bar(
                x=grouped_data.index.astype(str),
                y=grouped_data.values,
                marker=dict(color=colors),
                name=f"{col} Count",
                showlegend=False,
            ),
            row=row_num,
            col=col_num,
        )

        fig.update_yaxes(title_text="Count", row=row_num, col=col_num, showgrid=False)

    fig.update_layout(
        showlegend=False,
        legend_title_text=legend_column,
        height=400 * n_rows,
        width=1600,
        template="plotly_white",
    )

    return fig


def summarise_boolean_values(
    df: pd.DataFrame, columns_to_include: list, groupby_columns: list = ["year"]
) -> pd.DataFrame:
    """
    Generates a summary table for specified boolean columns in a DataFrame,
    including overall statistics and breakdowns based on the specified grouping columns.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    columns_to_include : list
        A list of columns to analyze, expected to contain binary (0/1) values.

    groupby_columns : list
        The list of columns to group by (e.g., ["industry", "year"]). Defaults to just "year".

    Returns:
    -------
    pd.DataFrame
        A multi-index summary table showing the count of 1s, count of 0s, and percentage of 1s
        for each column, both overall and per grouping.
    """
    summary_data = []

    # Overall summary
    for col in columns_to_include:
        if col in df.columns:
            count_1s = (df[col] == 1).sum()
            count_0s = (df[col] == 0).sum()
            perc_1s = (
                count_1s / (count_1s + count_0s) if (count_1s + count_0s) > 0 else 0
            )

            summary_data.append(
                {
                    "Column": col,
                    **{
                        col_name: "Overall" for col_name in groupby_columns
                    },  # Assign "Overall" for all grouping levels
                    "Mean": round(perc_1s, 2),
                }
            )

    # Grouped summary (based on year, industry, or any other column)
    grouped_summary = df.groupby(groupby_columns)[columns_to_include].agg(
        ["sum", "count"]
    )

    for group_values, data in grouped_summary.iterrows():
        group_dict = {
            col_name: val for col_name, val in zip(groupby_columns, group_values)
        }

        for col in columns_to_include:
            count_1s = data[(col, "sum")]
            count_total = data[(col, "count")]
            count_0s = count_total - count_1s
            perc_1s = count_1s / count_total if count_total > 0 else 0

            summary_data.append(
                {
                    "Column": col,
                    **group_dict,
                    "Mean": round(perc_1s, 2),
                }
            )

    # Convert to DataFrame and pivot based on the specified grouping columns
    summary_df = pd.DataFrame(summary_data)
    summary_pivot = summary_df.pivot_table(
        index="Column", columns=groupby_columns, values="Mean"
    )

    # Swap levels so that the first grouping column (e.g., year) is at the top
    summary_pivot = summary_pivot.swaplevel(0, len(groupby_columns) - 1, axis=1)

    # Sorting rows by "Overall" Mean, if available
    if tuple(["Overall"] * len(groupby_columns)) in summary_pivot.columns:
        sorted_order = (
            summary_pivot[tuple(["Overall"] * len(groupby_columns))]
            .sort_values(ascending=False)
            .index
        )
        summary_pivot = summary_pivot.loc[sorted_order, :]

    return summary_pivot


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parameters_by_year(
    df: pd.DataFrame, columns_to_include: list, display_mode: str = "percentage"
) -> go.Figure:
    """
    Creates stacked bar charts for selected parameters by year, with an option to display either percentages or counts.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to visualize.

    columns_to_include : list
        A list of discrete columns to visualize.

    display_mode : str, optional (default="percentage")
        Determines whether to display data as "percentage" or "count".

    Returns:
    --------
    go.Figure
        A Plotly figure displaying stacked bar charts for each parameter over time.
    """

    # Exclude columns that should not be visualized
    exclude_columns = {}
    
    relevant_columns = [
        col
        for col in columns_to_include
        if col in df.columns and col not in exclude_columns
    ]

    # Modify hq_country to group non-Nordic countries into "Other"
    if "hq_country" in df.columns:
        df["hq_country"] = df["hq_country"].apply(
            lambda x: x if x in {"Sweden", "Norway", "Denmark", "Finland"} else "Other"
        )

    # Identify integer columns that are binary (0/1)
    binary_columns = [
        col
        for col in relevant_columns
        if df[col].nunique() == 2 and sorted(df[col].unique()) == [0, 1]
    ]

    # Create subplot layout
    n_cols = 2
    n_rows = -(-len(relevant_columns) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=relevant_columns)

    for idx, col in enumerate(relevant_columns):
        row_num = (idx // n_cols) + 1
        col_num = (idx % n_cols) + 1

        # Aggregate counts per year per category
        grouped_data = df.groupby(["year", col]).size().unstack(fill_value=0)

        if display_mode == "percentage":
            grouped_data = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

        # Assign colors (binary columns get green/beige, others get default)
        if col in binary_columns:
            colors = ["rgb(6,212,124)", "rgb(254,240,205)"]  # Green for 1, Beige for 0
        else:
            colors = None  # Default Plotly colors

        # Add traces for each unique category
        for i, category in enumerate(grouped_data.columns):
            fig.add_trace(
                go.Bar(
                    x=grouped_data.index.astype(str),
                    y=grouped_data[category],
                    name=f"{category}",
                    marker=dict(color=colors[i] if colors else None),
                    showlegend=False,
                ),
                row=row_num,
                col=col_num,
            )

    # Adjust layout based on display mode
    yaxis_title = "Percentage" if display_mode == "percentage" else "Count"
    tick_format = ".1f%" if display_mode == "percentage" else None

    fig.update_layout(
        barmode="stack",
        height=400 * n_rows,
        width=1200,
        template="plotly_white",
        showlegend=False,
        yaxis=dict(title=yaxis_title, tickformat=tick_format),
    )

    return fig
