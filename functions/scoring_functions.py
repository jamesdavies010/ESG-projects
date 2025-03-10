import numpy as np
import pandas as pd
from typing import List


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
