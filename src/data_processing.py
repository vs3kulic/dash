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
INPUT_FILE = RAW_DATA_DIR / "Umsatzliste_AT943200000014664403.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "Umsatzliste_processed.csv"

# Mapping file
MAPPING_FILE = PROJECT_ROOT / "config" / "counterparty_mapping.json"


# ============================================================================
# MAPPING LOADING
# ============================================================================

def load_mapping() -> dict:
    """Loads the counterparty to category mapping from JSON config file."""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

CP2CAT = load_mapping()


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
# COUNTERPARTY AND CATEGORY EXTRACTION
# ============================================================================

def extract_counterparty(subject: str) -> str:
    """Extract counterparty by checking against loaded mapping."""
    for key in CP2CAT.keys():
        if key.upper() in subject.upper():
            return key
    return "Uncategorized"


def assign_category(counterparty: str) -> str:
    """Assigns category based on the counterparty.
    
    :param counterparty: Extracted counterparty name
    :return: Category name or "N/A" if not found
    """
    return CP2CAT.get(counterparty, "Uncategorized")


# ============================================================================
# DATA TRANSFORMATION PIPELINE
# ============================================================================

def transform_file() -> pd.DataFrame:
    """Transforms the input CSV file and saves the processed data."""
    # Load mapping and data
    df = load_data()

    # Select needed columns and add new columns to the DataFrame
    df_processed = df[["booking_date", "subject", "amount"]].copy()

    # Extract counterparty and assign category
    df_processed["counterparty"] = df_processed['subject'].apply(extract_counterparty)
    df_processed["category"] = df_processed["counterparty"].apply(assign_category)

    # Save the processed DataFrame to a new CSV file
    df_processed.to_csv(OUTPUT_FILE, index=False, decimal=',', sep=';')
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
