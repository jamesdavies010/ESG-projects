import numpy as np
import pandas as pd
from typing import List, Union
from pandas.api.types import CategoricalDtype


def calculate_industry_weights(df, scoring_cols):
    """
    Calculate industry-level materiality weights for ESG scoring.

    Parameters:
    - reporting_df (pd.DataFrame): The ESG dataset containing company data.
    - scoring_cols (list): List of columns to be used for scoring.

    Returns:
    - pd.DataFrame: A DataFrame with industry-level materiality weights.
    """
    # Step 1: Compute industry-level means across all years
    industry_means = df.groupby("industry")[scoring_cols].mean()

    # Step 2: Compute industry materiality score (sum of all means per industry)
    industry_means["industry_materiality_score"] = industry_means.sum(axis=1)

    # Step 3: Compute materiality weight for each variable
    materiality_weights = {}
    for col in scoring_cols:
        materiality_weights[f"{col}_materiality_weight"] = (
            industry_means[col] / industry_means["industry_materiality_score"]
        )

    # Create a new DataFrame with interleaved original and materiality weight columns
    interleaved_columns = []
    for col in scoring_cols:
        interleaved_columns.append(col)
        interleaved_columns.append(f"{col}_materiality_weight")

    industry_weights_df = industry_means.assign(**materiality_weights)[
        interleaved_columns + ["industry_materiality_score"]
    ]

    return industry_weights_df


def calculate_company_percentiles(
    df: pd.DataFrame, scoring_cols: List[str], by_industry: bool = True
) -> pd.DataFrame:
    """
    Calculate percentile scores for each company, grouped by industry and year.
    - If a company has a value of 1 for a metric, its percentile is calculated as:
        ((# of values less than 1) + (# of values equal to 1)/2) / (total companies in group)
    - If a company has a value of 0 for a metric, its percentile is set to 0.

    Parameters:
      df (pd.DataFrame): The ESG dataset containing company data.
      scoring_cols (List[str]): List of columns to be used for scoring.
      by_industry (bool): Whether to group by industry and year (True) or only by year (False). Default is True.

    Returns:
      pd.DataFrame: A DataFrame with percentile scores for each company.
    """
    result_dfs = []

    group_by_cols = ["industry", "year"] if by_industry else ["year"]

    # Iterate over each group defined by industry and year
    for group_keys, group in df.groupby(group_by_cols):
        group_copy = group.copy()
        total_count = len(group_copy)

        # Process each scoring column for this group
        for col in scoring_cols:
            ones_count = (group_copy[col] == 1).sum()
            if ones_count == 0:
                group_copy[f"{col}_percentile"] = 0
            else:
                group_copy[f"{col}_percentile"] = group_copy[col].apply(
                    lambda x: (
                        0
                        if x == 0
                        else (
                            (
                                (group_copy[col] < x).sum()
                                + (group_copy[col] == x).sum() / 2
                            )
                            / total_count
                        )
                    )
                )
        result_dfs.append(group_copy)

    # combine all the results into a single DataFrame
    company_percentiles = pd.concat(result_dfs, ignore_index=True)

    # Select and order the desired columns
    cols = ["company", "ticker", "year", "industry"] + [
        f"{col}_percentile" for col in scoring_cols
    ]
    return company_percentiles[cols]


import numpy as np
import pandas as pd
from typing import List


def calculate_raw_score(
    company_percentiles_df: pd.DataFrame,
    materiality_weights_df: pd.DataFrame,
    scoring_cols: List[str],
    by_industry: bool = True,
) -> pd.DataFrame:
    """
    Calculate a company's weighted raw score based on percentile scores and materiality weights.

    Parameters:
    - company_percentiles_df (pd.DataFrame): DataFrame with company percentile scores per metric.
    - materiality_weights_df (pd.DataFrame): DataFrame with materiality weights per metric.
    - scoring_cols (List[str]): List of columns to be used for scoring.
    - by_industry (bool, optional): Whether to apply industry-specific weights (default=True).
                                    If False, a single weight per metric is used.

    Returns:
    - pd.DataFrame: A DataFrame with weighted scores and an overall raw score for each company.
    """
    weighted_scores_df = company_percentiles_df.copy()
    score_cols = []

    for col in scoring_cols:
        weight_col = f"{col}_materiality_weight"
        score_col = f"{col}_score"

        if by_industry:
            # Map industry-specific materiality weights
            weighted_scores_df[score_col] = weighted_scores_df.apply(
                lambda row: (
                    row[f"{col}_percentile"]
                    * materiality_weights_df.loc[row["industry"], weight_col]
                    if row["industry"] in materiality_weights_df.index
                    else np.nan
                ),
                axis=1,
            )
        else:
            # Ensure the weights DataFrame is transposed correctly
            if "metric_materiality" in materiality_weights_df.index:
                if col in materiality_weights_df.columns:
                    global_weight = materiality_weights_df.loc[
                        "metric_materiality", col
                    ]
                    weighted_scores_df[score_col] = (
                        weighted_scores_df[f"{col}_percentile"] * global_weight
                    )
                else:
                    weighted_scores_df[score_col] = np.nan  # Handle missing metric
            else:
                weighted_scores_df[score_col] = (
                    np.nan
                )  # Handle case where row is missing

        score_cols.append(score_col)

    # Define the correct output column name
    score_col_name = "industry_score_raw" if by_industry else "overall_score_raw"

    # Compute overall raw score as the sum of all individual scores
    weighted_scores_df[score_col_name] = weighted_scores_df[score_cols].sum(axis=1)

    return weighted_scores_df[
        ["company", "ticker", "year", "industry", score_col_name] + score_cols
    ]


def calculate_adjusted_score(df, raw_column, by_industry=True):
    """
    Calculates the percentile ranking of each company's environmental score relative to
    other companies in the same industry and year.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing company data.
        raw_column (str): The column name for the raw environmental score.
        by_industry (bool): If True, percentiles are calculated within industry-year groups.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for percentiles.
    """
    result_dfs = []

    # Group by industry and year if by_industry is True; otherwise, group only by year
    groupby_cols = ["industry", "year"] if by_industry else ["year"]

    adjusted_score_col = (
        "industry_score_adjusted" if by_industry else "overall_score_adjusted"
    )

    for _, group in df.groupby(groupby_cols):
        group_copy = group.copy()
        total_count = len(group_copy)

        if total_count > 0:
            group_copy[adjusted_score_col] = group_copy[raw_column].apply(
                lambda x: (
                    0
                    if x == 0
                    else (
                        (
                            (group_copy[raw_column] < x).sum()
                            + (group_copy[raw_column] == x).sum() / 2
                        )
                        / total_count
                    )
                )
            )

        result_dfs.append(group_copy)

    # Concatenate results and return only relevant columns
    final_df = pd.concat(result_dfs, ignore_index=True)

    # selected_cols = [
    #     "company",
    #     "ticker",
    #     "year",
    #     "industry",
    #     "adjusted_industry_score",
    # ]

    return final_df


import pandas as pd
from pandas.api.types import CategoricalDtype


def assign_rating(percentile: float) -> str:
    """
    Assigns a rating (A+ to D-) based on the given percentile.
    Returns a plain string. We will convert it to a categorical later.
    """
    rating_order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"]

    # Assign rating based on percentile
    if 0.92 < percentile <= 1:
        return "A+"
    elif 0.83 < percentile <= 0.92:
        return "A"
    elif 0.75 < percentile <= 0.83:
        return "A-"
    elif 0.67 < percentile <= 0.75:
        return "B+"
    elif 0.58 < percentile <= 0.67:
        return "B"
    elif 0.50 < percentile <= 0.58:
        return "B-"
    elif 0.42 < percentile <= 0.50:
        return "C+"
    elif 0.33 < percentile <= 0.42:
        return "C"
    elif 0.25 < percentile <= 0.33:
        return "C-"
    elif 0.17 < percentile <= 0.25:
        return "D+"
    elif 0.08 < percentile <= 0.17:
        return "D"
    elif 0.0 <= percentile <= 0.08:
        return "D-"
    else:
        return "Invalid percentile"


def cast_to_rating_category(series: pd.Series) -> pd.Series:
    """
    Converts a string-based rating Series into an ordered categorical
    with the same rating order used above, so we only define it once.
    """
    rating_order = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"]
    dtype = CategoricalDtype(categories=rating_order, ordered=True)
    return series.astype(dtype)