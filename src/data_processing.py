"""This module contains functions for processing financial transaction data."""
import json
import pandas as pd
import config.config as config


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

def load_category_mapping() -> dict:
    """Loads the category mapping from JSON config file."""
    # Open and read the JSON mapping file
    with open(config.CATEGORY_MAPPING, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Normalize all keys to uppercase
    return {key.upper(): value for key, value in mapping.items()}


def load_alias_mapping() -> dict:
    """Loads the alias mapping from JSON config file."""
    # Open and read the JSON mapping file
    with open(config.ALIAS_MAPPING, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Normalize all keys to uppercase
    return {key.upper(): value for key, value in mapping.items()}


# Load mapping at module load time
CP2CAT = load_category_mapping()
ALIASES = load_alias_mapping()


# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Reads and processes the financial transaction data from a CSV file."""
    # Define column names since the CSV does not have a header, read into DataFrame
    column_names = ["booking_date", "subject", "execution_date", "amount", "currency", "timestamp"]
    df = pd.read_csv(config.INPUT_FILE, delimiter=";", names=column_names)

    # Convert amount from German format (comma as decimal) to float
    df["amount"] = df["amount"].str.replace('.', '', regex=False)  # Remove thousands separator
    df["amount"] = df["amount"].str.replace(',', '.', regex=False)  # Replace comma with period
    df["amount"] = pd.to_numeric(df["amount"])

    # Convert dates from German format (DD.MM.YYYY)
    df["booking_date"] = pd.to_datetime(df["booking_date"], dayfirst=True)
    df["execution_date"] = pd.to_datetime(df["execution_date"], dayfirst=True)

    return df


# ============================================================================
# DATA EXTRACTION AND CATEGORY ASSIGNMENT
# ============================================================================

def extract_counterparty(subject: str) -> str:
    """Extract counterparty by checking against loaded mapping.
    
    :param subject: Transaction subject string
    :return: Extracted counterparty name
    """
    subject_upper = subject.upper()

    # Check aliases first, longest first (sorted by key length)
    for alias in sorted(ALIASES.keys(), key=len, reverse=True):
        if alias in subject_upper:
            return ALIASES[alias]   # Return mapped counterparty name

    # Check mapping keys, longest first
    for counterparty in sorted(CP2CAT.keys(), key=len, reverse=True):
        if counterparty in subject_upper:
            return counterparty  # Return as is since keys are already uppercase

    # If no match found, return a default value
    return "UNCATEGORIZED"


def assign_category(counterparty: str) -> str:
    """Assign category based on counterparty using the loaded mapping.
    
    :param counterparty: Extracted counterparty name
    :return: Assigned category name
    """
    return CP2CAT.get(counterparty, "uncategorized")


def extract_purpose(subject: str) -> str:
    """Extracts the purpose from the transaction subject.
    
    :param subject: Transaction subject string
    :return: Extracted purpose string
    """
    marker = "Verwendungszweck: "
    index = subject.find(marker)  # Returns index of the marker or -1 if not found
    start = index + len(marker)
    end = subject.find(" ", start)  # Find the next space after the marker, starting from 'start'

    return subject[start:end] if index != -1 else ""


# ============================================================================
# PERFORM CALCULATIONS
# ============================================================================

def category_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sum of amounts per category in total.
    
    :param df: DataFrame with processed transaction data
    :return: DataFrame with sum of amounts per category
    """
    # Load data if not provided
    category_sum = df.groupby("category")["amount"].sum().reset_index()
    return category_sum


def period_totals(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Calculates sum of amounts per category for a given period (e.g., monthly).
    
    :param df: DataFrame with processed transaction data
    :param period: Period string for grouping (e.g., 'M' for month)
    :return: DataFrame with sum of amounts per category per period
    """
    df['period'] = df["booking_date"].dt.to_period(period)
    period_sum = df.groupby(["period", "category"])["amount"].sum().reset_index()
    return period_sum


# ============================================================================
# DATA TRANSFORMATION PIPELINE
# ============================================================================

def transform_file() -> pd.DataFrame:
    """Transforms the input CSV file and saves the processed data."""
    # Load mapping and data
    df = load_data()

    # Select needed columns and add new columns to the DataFrame
    df_processed = df[["booking_date", "subject", "amount"]].copy()
    df_processed["counterparty"] = df_processed["subject"].apply(extract_counterparty)
    df_processed["category"] = df_processed["counterparty"].apply(assign_category)
    df_processed["purpose"] = df_processed["subject"].apply(extract_purpose)

    # Save processed data to CSV
    df_processed.to_csv(config.OUTPUT_FILE, index=False, decimal=",", sep=";")
    return df_processed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for demonstrating the data loading."""
    df_transformed = transform_file()
    print(df_transformed.head(50))
    print("\nCategory Totals:")
    print("===================================")
    category = category_totals(df_transformed)
    print(category)
    print("\nCategory Totals (Monthly):")
    print("===================================")
    monthly = period_totals(df_transformed, 'M')
    print(monthly)

if __name__ == "__main__":
    main()
