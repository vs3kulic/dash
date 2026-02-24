"""This module contains functions for processing financial transaction data."""
import sys
from pathlib import Path

# Add project root to Python path to enable config imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

def load_data(input_file: Path = None) -> pd.DataFrame:
    """Reads and processes the financial transaction data from a CSV file.
    
    :param input_file: Path to the input CSV file. Must be provided.
    :return: DataFrame with processed transaction data
    """
    if input_file is None:
        raise ValueError("input_file parameter must be provided")

    # Define column names since the CSV does not have a header, read into DataFrame
    column_names = ["booking_date", "subject", "execution_date", "amount", "currency", "timestamp"]
    df = pd.read_csv(input_file, delimiter=";", names=column_names)

    # Convert amount from German format (comma as decimal) to float
    df["amount"] = df["amount"].str.replace('.', '', regex=False)  # Remove thousands separator
    df["amount"] = df["amount"].str.replace(',', '.', regex=False)  # Replace comma with period
    df["amount"] = pd.to_numeric(df["amount"])

    # Convert dates from German format (DD.MM.YYYY or DD.MM.YY)
    df["booking_date"] = pd.to_datetime(df["booking_date"], format='mixed', dayfirst=True)
    df["execution_date"] = pd.to_datetime(df["execution_date"], format='mixed', dayfirst=True)

    # Clean subject by stripping whitespace and converting to lowercase
    df["subject"] = df["subject"].str.strip()

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
    if counterparty not in CP2CAT:
        return "uncategorized"
    return CP2CAT[counterparty]


# ============================================================================
# DATA TRANSFORMATION PIPELINE
# ============================================================================

def transform_file(input_file: Path = None, output_file: Path = None) -> pd.DataFrame:
    """Transforms the input CSV file and saves the processed data.
    
    :param input_file: Path to the input CSV file. Must be provided.
    :param output_file: Path to save the processed CSV. Must be provided.
    :return: DataFrame with processed transaction data
    """
    if input_file is None:
        raise ValueError("input_file parameter must be provided")
    if output_file is None:
        raise ValueError("output_file parameter must be provided")

    # Load source data, select relevant columns and create a copy
    src = load_data(input_file)
    df_processed = src[["booking_date", "subject", "amount"]].copy()

    # Extract counterparty, category, and purpose
    df_processed["counterparty"] = df_processed["subject"].apply(extract_counterparty)
    df_processed["category"] = df_processed["counterparty"].apply(assign_category)

    # Save processed data to CSV
    df_processed.to_csv(output_file, index=False, decimal=",", sep=";")
    return df_processed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function for demonstrating the data processing pipeline."""
    # Process training data
    print("\n" + "="*60)
    print("PROCESSING TRAINING DATA")
    print("="*60)
    df_train = transform_file(config.TRAIN_INPUT_FILE, config.TRAIN_OUTPUT_FILE)
    print(f"Training data processed: {config.TRAIN_INPUT_FILE}")
    print(f"Saved to: {config.TRAIN_OUTPUT_FILE}")
    print(f"Training samples: {len(df_train)}")

    # Process validation data
    print("\n" + "="*60)
    print("PROCESSING VALIDATION DATA")
    print("="*60)
    df_val = transform_file(config.VAL_INPUT_FILE, config.VAL_OUTPUT_FILE)
    print(f"Validation data processed: {config.VAL_INPUT_FILE}")
    print(f"Saved to: {config.VAL_OUTPUT_FILE}")
    print(f"Validation samples: {len(df_val)}")

if __name__ == "__main__":
    main()
