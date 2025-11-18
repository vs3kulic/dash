"""This module contains functions for processing financial transaction data."""
from pathlib import Path
import pandas as pd

# Get absolute path to the CSV file, relative to THIS file's location
PROJECT_ROOT = Path(__file__).parent.parent
input_file = PROJECT_ROOT/"data"/"raw"/"Umsatzliste_AT943200000014664403.csv"

def load_data():
    """Reads and processes the financial transaction data from a CSV file."""
    # Define column names since the CSV has no header
    column_names = ['booking_date', 'subject', 'execution_date', 'amount', 'currency', 'timestamp']

    # Read the CSV file
    df = pd.read_csv(input_file, delimiter=";", names=column_names)

    # Convert amount from German format (comma as decimal) to float
    df['amount'] = df['amount'].str.replace('.', '', regex=False)  # Remove thousands separator
    df['amount'] = df['amount'].str.replace(',', '.', regex=False)  # Replace comma with period
    df['amount'] = pd.to_numeric(df['amount'])

    # Convert dates from German format (DD.MM.YYYY)
    df['booking_date'] = pd.to_datetime(df['booking_date'], dayfirst=True)
    df['execution_date'] = pd.to_datetime(df['execution_date'], dayfirst=True)

    return df

def main():
    """Main function for demonstrating the data loading."""
    df = load_data()
    print(df.head())

if __name__ == "__main__":
    main()
