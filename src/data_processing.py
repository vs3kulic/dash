"""This module contains functions for processing financial transaction data."""
from pathlib import Path
import json
import pandas as pd

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent

# Data files
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = RAW_DATA_DIR / "Umsatzliste_raw.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "Umsatzliste_processed.csv"

# Mapping files
ALIAS_MAPPING = PROJECT_ROOT / "config" / "alias_mapping.json"
CATEGORY_MAPPING = PROJECT_ROOT / "config" / "category_mapping.json"


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

def load_category_mapping() -> dict:
    """Loads the category mapping from JSON config file."""
    # Open and read the JSON mapping file
    with open(CATEGORY_MAPPING, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Normalize all keys to uppercase
    return {key.upper(): value for key, value in mapping.items()}


def load_alias_mapping() -> dict:
    """Loads the alias mapping from JSON config file."""
    # Open and read the JSON mapping file
    with open(ALIAS_MAPPING, 'r', encoding='utf-8') as f:
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
    df = pd.read_csv(INPUT_FILE, delimiter=";", names=column_names)

    # Convert amount from German format (comma as decimal) to float
    df["amount"] = df["amount"].str.replace('.', '', regex=False)  # Remove thousands separator
    df["amount"] = df["amount"].str.replace(',', '.', regex=False)  # Replace comma with period
    df["amount"] = pd.to_numeric(df["amount"])

    # Convert dates from German format (DD.MM.YYYY)
    df["booking_date"] = pd.to_datetime(df["booking_date"], dayfirst=True)
    df["execution_date"] = pd.to_datetime(df["execution_date"], dayfirst=True)

    return df


# ============================================================================
# COUNTERPARTY EXTRACTION AND CATEGORY ASSIGNMENT
# ============================================================================

def extract_counterparty(subject: str) -> str:
    """Extract counterparty by checking against loaded mapping.
    
    :param subject: Transaction subject string
    :return: Extracted counterparty name
    """
    subject_upper = subject.upper()
    
    # Check aliases first, longest first
    for alias, canonical in sorted(ALIASES.items(), reverse=True):
        if alias.upper() in subject_upper:
            return canonical.upper()

    # Check mapping keys, longest first
    for counterparty in sorted(CP2CAT.keys(), key=len, reverse=True):
        if counterparty in subject_upper:
            return counterparty

    return "UNCATEGORIZED"


def assign_category(counterparty: str) -> str:
    """Assign category based on counterparty using the loaded mapping.
    
    :param counterparty: Extracted counterparty name
    :return: Assigned category name
    """
    return CP2CAT.get(counterparty, "uncategorized")


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

    # Save processed data to CSV
    df_processed.to_csv(OUTPUT_FILE, index=False, decimal=",", sep=";")
    return df_processed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for demonstrating the data loading."""
    df_transformed = transform_file()
    print(df_transformed.head(50))

if __name__ == "__main__":
    main()
