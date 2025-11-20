"""This module contains functions for processing financial transaction data."""
from pathlib import Path
import json
import pandas as pd


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MAPPING_FILE = PROJECT_ROOT / "config" / "counterparty_mapping.json"
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Umsatzliste_AT943200000014664403.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "Umsatzliste_processed.csv"
COLUMN_NAMES = ["booking_date", "subject", "execution_date", "amount", "currency", "timestamp"]


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

# Load counterparty to category mapping from config file
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
    df = pd.read_csv(INPUT_FILE, delimiter=";", names=COLUMN_NAMES)

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
    """Extracts sender or recipient information from the subject field."""
    for key in CP2CAT.keys():
        if key.upper() in subject.upper():
            return key
    return "N/A"


def assign_category(counterparty: str) -> str:
    """Assigns category based on the counterparty."""
    return CP2CAT.get(counterparty, "N/A")


# ============================================================================
# DATA TRANSFORMATION PIPELINE
# ============================================================================

def transform_file() -> pd.DataFrame:
    """Transforms the input CSV file and saves the processed data."""
    # Load the data
    df = load_data()

    # Select needed columns and add new columns to the DataFrame
    df_processed = df[["booking_date", "subject", "amount"]].copy()

    # First extract counterparty, then assign category based on it
    df_processed["counterparty"] = df_processed['subject'].apply(extract_counterparty)
    df_processed["category"] = df_processed["counterparty"].apply(assign_category)

    # Save the processed DataFrame to a new CSV file
    df_processed.to_csv(OUTPUT_FILE, index=False)  # Save processed data without index
    return df_processed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for demonstrating the data loading."""
    # df = load_data()
    # print(df.head())
    df_transformed = transform_file()
    print(df_transformed.head(50))

if __name__ == "__main__":
    main()
