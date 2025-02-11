# include type hints
# create a function that returns the specified column to a list of unique values with their value counts

import pandas as pd


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


def find_similar_entries(input_df: pd.DataFrame, threshold: int = 75) -> pd.DataFrame:
    """
    Identifies pairs of similar entries based on a combination of 'comp_name' and 'ticker',
    ensuring that the first two characters of 'comp_name' are the same. The function also
    includes the corresponding year for each entry.

    Parameters:
        input_df (pd.DataFrame): The input DataFrame containing 'comp_name', 'ticker', and 'year' columns.
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
    input_df = input_df.copy()

    # Create a combined column of company name and ticker
    input_df["combined"] = input_df["comp_name"] + " (" + input_df["ticker"] + ")"

    # Store the most recent year for each combined entry
    year_lookup = input_df.groupby("combined")["year"].max().to_dict()

    # Get unique combined values
    unique_values = input_df["combined"].drop_duplicates().tolist()
    similar_pairs = []

    # Compare each unique value to others
    for i, value in enumerate(unique_values):
        matches = process.extract(
            value, unique_values[i + 1 :], scorer=fuzz.ratio, limit=None
        )
        for match_value, score, _ in matches:
            # Lookup precomputed years for both entries
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

    return (
        pd.DataFrame(
            similar_pairs, columns=["entry1", "year1", "entry2", "year2", "similarity"]
        )
        .sort_values(by="similarity", ascending=False)
        .reset_index(drop=True)
    )
