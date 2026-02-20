import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import config.config as config


def compute_location_measures():
    """Compute location measures (central tendency) for training data."""

    # Load training data
    df = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",")

    print("="*60)
    print("LOCATION MEASURES - TRAINING DATA")
    print("="*60)

    # Amount statistics
    print("\n📊 Transaction Amount Statistics:")
    print("-"*60)
    print(f"Mean (Average):        €{df['amount'].mean():>12,.2f}")
    print(f"Median (Middle):       €{df['amount'].median():>12,.2f}")
    print(f"Mode (Most common):    €{df['amount'].mode().values[0]:>12,.2f}")
    print(f"\nMinimum:               €{df['amount'].min():>12,.2f}")
    print(f"Maximum:               €{df['amount'].max():>12,.2f}")
    print(f"Range:                 €{df['amount'].max() - df['amount'].min():>12,.2f}")

    # Spread measures
    print("\n📈 Spread Measures:")
    print("-"*60)
    print(f"Standard Deviation:    €{df['amount'].std():>12,.2f}")
    print(f"Variance:              €{df['amount'].var():>12,.2f}")
    print(f"IQR (Q3 - Q1):         €{df['amount'].quantile(0.75) - df['amount'].quantile(0.25):>12,.2f}")

    # Quartiles
    print("\n📉 Quartiles:")
    print("-"*60)
    print(f"25th percentile (Q1):  €{df['amount'].quantile(0.25):>12,.2f}")
    print(f"50th percentile (Q2):  €{df['amount'].quantile(0.50):>12,.2f}")
    print(f"75th percentile (Q3):  €{df['amount'].quantile(0.75):>12,.2f}")

    # Check for outliers using IQR method
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
    
    print("\n🔍 Outlier Detection (IQR method):")
    print("-"*60)
    print(f"Lower bound:           €{lower_bound:>12,.2f}")
    print(f"Upper bound:           €{upper_bound:>12,.2f}")
    print(f"Number of outliers:    {len(outliers):>12}")
    
    if len(outliers) > 0:
        print("\nOutlier transactions:")
        print(outliers[['booking_date', 'counterparty', 'amount', 'category']].to_string(index=False))
    
    # Amount by category
    print("\n\n💶 Amount Statistics by Category:")
    print("="*60)
    category_stats = df.groupby('category')['amount'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('total', 'sum')
    ]).sort_values('total', ascending=False)

    print(category_stats.to_string())

    # Missing values check
    print("\n\n❓ Missing Values Check:")
    print("-"*60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(missing[missing > 0])

    # Data types
    print("\n\n📋 Data Types:")
    print("-"*60)
    print(df.dtypes)


if __name__ == "__main__":
    compute_location_measures()
