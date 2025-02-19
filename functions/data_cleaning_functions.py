import pandas as pd
import numpy as np
from typing import List

def display_unique_counts(df: pd.DataFrame) -> None:
    """
    Display the unique counts of companies and tickers in a DataFrame.

    This function prints the number of unique companies and tickers present in the given DataFrame.
    It checks for the existence of the 'comp_name' and 'ticker' columns before attempting to access them.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame that should contain the columns 'comp_name' and/or 'ticker'.

    Returns:
        None
    """
    if "comp_name" in df.columns:
        print("Unique companies in the database: ", df["comp_name"].nunique())
    else:
        print("'comp_name' column not found in the DataFrame.")

    if "ticker" in df.columns:
        print("Unique tickers in the database: ", df["ticker"].nunique())
    else:
        print("'ticker' column not found in the DataFrame.")


def test_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Filter a DataFrame to return all rows associated with a specific ticker.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'ticker' column.
        ticker (str): The ticker value to filter the DataFrame by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the 'ticker' column matches the specified value.
    """
    return df[df["ticker"] == ticker]


import pandas as pd


def tickers_with_multiple_companies(df: pd.DataFrame) -> list:
    """
    Identifies tickers associated with multiple companies in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'ticker' and 'comp_name' columns.

    Returns:
    list: A list of strings, where each string contains a ticker and the associated companies.
    """

    # Group by ticker and count unique company names for each ticker
    companies_per_ticker_df = (
        df.groupby("ticker")["comp_name"].nunique().sort_values(ascending=False)
    )

    tickers_with_multiple_companies_index = companies_per_ticker_df[
        companies_per_ticker_df > 1
    ].index

    # Get DataFrame of tickers with multiple companies
    tickers_multi_comp_df = df[df["ticker"].isin(tickers_with_multiple_companies_index)]

    # Build the list of tickers and associated companies
    tickers_multi_comp_list = []
    for ticker, other_cols in tickers_multi_comp_df.groupby("ticker"):
        companies = ", ".join(sorted(other_cols["comp_name"].unique()))
        tickers_multi_comp_list.append(f"{ticker}: {companies}")

    print(
        "Tickers associated with multiple companies: ",
        len(tickers_with_multiple_companies_index),
        end="\n\n",
    )

    return tickers_multi_comp_list


def apply_most_recent_company_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the 'comp_name' column in the DataFrame to use the most recent company name for each ticker.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'ticker', 'year', and 'comp_name' columns.

    Returns:
    pd.DataFrame: A DataFrame with the 'comp_name' column updated based on the latest year for each ticker.
    """

    # Sort by ticker (alphabetically) and year (descending) to get the latest company name per ticker
    ticker_latest_company_df = df.sort_values(
        by=["ticker", "year"], ascending=[True, False]
    ).drop_duplicates(subset=["ticker"], keep="first")

    # Create a mapping dictionary: {ticker: latest company name}
    comp_name_mapped_dict = dict(
        zip(ticker_latest_company_df["ticker"], ticker_latest_company_df["comp_name"])
    )

    # Update 'comp_name' in df using the mapping dictionary
    df.loc[:, "comp_name"] = df["ticker"].map(comp_name_mapped_dict)


def companies_with_multiple_tickers(df: pd.DataFrame) -> list:
    """
    Identifies companies associated with multiple tickers in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'comp_name' and 'ticker' columns.

    Returns:
    list: A list of strings, where each string contains a company name and its associated tickers.
    """

    # Group by company name and count unique tickers
    tickers_per_company_df = (
        df.groupby("comp_name")["ticker"].nunique().sort_values(ascending=False)
    )

    # Filter to keep only companies associated with multiple tickers
    companies_with_multiple_tickers_index = tickers_per_company_df[
        tickers_per_company_df > 1
    ].index

    # Get DataFrame of companies with multiple tickers
    companies_multi_ticker_df = df[
        df["comp_name"].isin(companies_with_multiple_tickers_index)
    ]

    # Build the list of companies and associated tickers
    companies_multi_ticker_list = []
    for company, other_cols in companies_multi_ticker_df.groupby("comp_name"):
        tickers = ", ".join(sorted(other_cols["ticker"].unique()))
        companies_multi_ticker_list.append(f"{company}: {tickers}")

    print(
        "Companies associated with multiple tickers: ",
        len(companies_with_multiple_tickers_index),
        end="\n\n",
    )

    return companies_multi_ticker_list


def test_company(df: pd.DataFrame, company: str) -> pd.DataFrame:
    """
    Filter a DataFrame to return all rows associated with a specific company.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'comp_name' column.
        company (str): The company name to filter the DataFrame by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where the 'comp_name' column matches the specified value.
    """
    return df[df["comp_name"] == company]


import pandas as pd


def apply_most_recent_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the most recent ticker to each company based on the latest available year.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'comp_name', 'year', and 'ticker' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with the most recent ticker applied.
    """
    # Ensure necessary columns exist
    required_columns = {"comp_name", "year", "ticker"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Sort by company name and year (descending to get the latest first)
    company_latest_ticker_df = df.sort_values(
        by=["comp_name", "year"], ascending=[True, False]
    ).drop_duplicates(subset=["comp_name"], keep="first")

    # Create a mapping of company name to the latest ticker
    ticker_mapping = dict(
        zip(company_latest_ticker_df["comp_name"], company_latest_ticker_df["ticker"])
    )

    # Map the most recent ticker back to the original dataframe
    df.loc[:, "ticker"] = df["comp_name"].map(ticker_mapping)

    # return df


from typing import Optional, Dict, List, Any


def test_filter(
    df: pd.DataFrame,
    columns_to_filter: Optional[Dict[str, Any]] = None,
    columns_to_show: Optional[List[str]] = None,
    rows_to_show: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filters a DataFrame based on specified columns and their corresponding values,
    optionally selects specific columns, and limits the number of rows displayed.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter.
    - columns_to_filter (dict, optional): A dictionary where keys are column names and values are lists of filter values.
    - columns_to_show (list, optional): A list of columns to display (default is all columns).
    - rows_to_show (int, optional): The number of rows to display. If None, displays all rows.

    Returns:
    - pd.DataFrame: A filtered DataFrame with the specified columns and limited rows.
    """
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    if columns_to_filter:
        for column, values in columns_to_filter.items():
            # Ensure values is a list for consistent filtering
            if not isinstance(values, list):
                values = [values]
            df = df[df[column].isin(values)]

    df if columns_to_show is None else df[columns_to_show]

    # figure out a way to make this work as you intend it to
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")

    return df if rows_to_show is None else df.head(rows_to_show)


import pandas as pd


def show_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display missing values and their percentages for each column in a DataFrame.

    This function calculates the number and percentage of missing (NaN) values in each column.
    Additionally, it counts occurrences of the string 'ND' (Not Disclosed) and computes its percentage.

    Parameters:
        df (pd.DataFrame): The input DataFrame to analyze for missing values.

    Returns:
        pd.DataFrame: A summary DataFrame containing:
            - 'Missing Values': Count of NaN values in each column.
            - 'Missing Percentage': Percentage of NaN values relative to the total rows.
            - "'ND' Values": Count of occurrences of 'ND' in each column.
            - "'ND' Percentage": Percentage of 'ND' occurrences relative to the total rows.
    """
    # Calculate the count and percentage of missing values
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df)) * 100

    # Count occurrences of 'ND' and calculate its percentage
    nd_values = (df == "ND").sum()
    nd_percentage = (nd_values / len(df)) * 100

    missing_summary = pd.DataFrame(
        {
            "Missing Values": missing_values,
            "Missing Percentage": missing_percentage.round(2),
            "'ND' Values": nd_values,
            "'ND' Percentage": nd_percentage.round(2),
        },
        index=pd.Index(missing_values.index, name="cols"),
    )

    return missing_summary


import pandas as pd
from typing import Optional
from rapidfuzz import fuzz, process


def find_similar_entries(df: pd.DataFrame, threshold: int = 75) -> pd.DataFrame:
    """
    Identifies pairs of similar entries based on a combination of 'comp_name' and 'ticker',
    ensuring that the first two characters of 'comp_name' are the same. The function also
    includes the corresponding year for each entry.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'comp_name', 'ticker', and 'year' columns.
        threshold (int, optional): The similarity score threshold (default is 75). Pairs must meet or exceed
            this score to be included.

    Returns:
        pd.DataFrame: A DataFrame containing similar entry pairs with:
            - 'entry1': First matched entry (comp_name + ticker).
            - 'year1': Corresponding year for entry1.
            - 'entry2': Second matched entry (comp_name + ticker).
            - 'year2': Corresponding year for entry2.
            - 'similarity': Similarity score between the two entries.
    """
    input_df = df.copy()

    # Combine columns for a more robust measure of similarity
    input_df["combined"] = input_df["comp_name"] + " (" + input_df["ticker"] + ")"

    # If two similar entries are the same company, keep the one with the most recent year
    year_lookup = input_df.groupby("combined")["year"].max().to_dict()

    # Get unique combined values
    unique_values = input_df["combined"].drop_duplicates().tolist()
    similar_pairs = []

    # Compare each unique value to others
    for i, value in enumerate(unique_values):
        matches = process.extract(
            # [i + 1 :] to avoid comparing to itself
            value,
            unique_values[i + 1 :],
            scorer=fuzz.ratio,
            limit=None,
        )
        for match_value, score, _ in matches:
            # Lookup precomputed years for both entries; return None if not found
            year1 = year_lookup.get(value, None)
            year2 = year_lookup.get(match_value, None)

            # Check similarity threshold and ensure first two characters of comp_name match
            if score >= threshold and value[:2] == match_value[:2]:
                # Ensure entry1 has the more recent year
                if year2 > year1:
                    value, match_value = match_value, value  # Swap values
                    year1, year2 = year2, year1  # Swap years

                similar_pairs.append((value, year1, match_value, year2, score))

    # Drop the temporary 'combined' column
    input_df.drop(columns=["combined"], inplace=True)

    output_df = (
        pd.DataFrame(
            similar_pairs, columns=["entry1", "year1", "entry2", "year2", "similarity"]
        )
        .sort_values(by="similarity", ascending=False)
        .reset_index(drop=True)
    )

    return output_df


def map_similar_pairs(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    indices_to_keep: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Merges similar company names in `output_df` using mappings from `input_df`.

    Parameters:
    -----------
    input_df : pd.DataFrame
        A DataFrame containing columns `entry1` and `entry2`, where `entry1` is the preferred name/ticker.
    output_df : pd.DataFrame
        The target DataFrame where name and ticker replacements will be applied.
    indices_to_keep : Optional[List[int]], default=None
        A list of row indices from `input_df` to use for mapping. If None, all rows are used.

    Returns:
    --------
    pd.DataFrame
        A modified version of `output_df` with corrected `comp_name` and `ticker` values.
    """

    # If indices_to_keep is specified, filter the input_df
    if indices_to_keep is not None:
        input_df = input_df.loc[indices_to_keep]

    # Extract company names and tickers from 'entry1' and 'entry2' columns
    input_df[["comp_name1", "ticker1"]] = input_df["entry1"].str.extract(
        r"^(.*) \((.*)\)$"
    )
    input_df[["comp_name2", "ticker2"]] = input_df["entry2"].str.extract(
        r"^(.*) \((.*)\)$"
    )

    # Create mappings for replacement
    comp_name_mapping = dict(zip(input_df["comp_name2"], input_df["comp_name1"]))
    ticker_mapping = dict(zip(input_df["ticker2"], input_df["ticker1"]))

    # Apply replacements to the output DataFrame
    output_df["comp_name"] = output_df["comp_name"].map(lambda row: comp_name_mapping.get(row, row))
    output_df["ticker"] = output_df["ticker"].map(lambda row: ticker_mapping.get(row, row))


def update_segments_remove_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the 'segment' column in the given DataFrame based on the most recent valid segment
    for companies with missing segment information.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'comp_name', 'segment', and 'year' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with missing segment values replaced.
    """

    valid_segments: list[str] = ["Mid", "Large", "Small"]

    # Identify companies with missing segment information
    companies_missing_segments: np.ndarray = df[
        df["segment"].isin(["ND", "0", np.nan])
    ]["comp_name"].unique()

    most_recent_segments: Dict[str, str] = (
        df[
            df["comp_name"].isin(companies_missing_segments)
            & df["segment"].isin(valid_segments)
        ]
        .dropna(subset=["segment", "year"])
        .sort_values(by="year", ascending=False)
        .groupby("comp_name")
        .first()["segment"]
        .to_dict()
    )

    # Update the 'segment' column using the most recent valid segment if applicable
    df["segment"] = df.apply(
        lambda row: (
            most_recent_segments.get(row["comp_name"], row["segment"])
            if row["segment"] in ["ND", "0", np.nan]
            else row["segment"]
        ),
        axis=1,
    )


def get_most_recent_values(
    df: pd.DataFrame,
    invalid_values: List[str] = ["ND"],
    columns_to_update: List[str] = ["segment", "industry", "hq_country"],
    sort_column: str = "year",
) -> pd.DataFrame:
    """
    Updates a DataFrame by replacing specified columns with the most recent valid value per company.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing historical data.
        invalid_values (List[str], optional): A list of invalid values to be ignored. Defaults to ["ND"].
        columns_to_update (List[str], optional): Columns to update with the most recent valid values.
        sort_column (str, optional): The column used for sorting records (e.g., "year"). Defaults to "year".

    Returns:
        pd.DataFrame: A DataFrame with updated columns containing the most recent valid values.
    """

    def get_most_recent_valid_data(group: pd.DataFrame, column: str) -> str:
        valid_data = group[~(group[column].isin(invalid_values) | group[column].isna())]
        if not valid_data.empty:
            return valid_data.iloc[0][column]  # Most recent valid value
        return "Unknown"  # Default fallback

    most_recent_data = {
        comp_name: {
            col: get_most_recent_valid_data(group, col) for col in columns_to_update
        }
        for comp_name, group in df.sort_values(by=sort_column, ascending=False).groupby(
            "comp_name"
        )
    }

    for col in columns_to_update:
        df[col] = df["comp_name"].map(lambda x: most_recent_data[x][col])


def generate_binary_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary DataFrame with column names, data types, and counts of 1s and 0s.

    Args:
        df (pd.DataFrame): The input DataFrame to summarize.

    Returns:
        pd.DataFrame: A DataFrame summarizing the column names, data types, and counts of 1s and 0s.
    """
    summary_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": [df[col].dtype for col in df.columns],
            "1s": [(df[col] == 1).sum() for col in df.columns],
            "0s": [(df[col] == 0).sum() for col in df.columns],
        }
    )
    return summary_df
