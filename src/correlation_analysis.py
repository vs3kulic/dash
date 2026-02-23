import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import config.config as config


def compute_correlations():
    """Compute and visualize correlations between amount and transaction categories."""

    # Load training data
    df = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",")

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # One-hot encode the category column
    df_encoded = pd.get_dummies(df[['amount', 'category']], columns=['category'], prefix='cat')

    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()

    # Print correlation of amount with each category
    print("\nCorrelation of 'amount' with each category:")
    print("-" * 60)
    amount_corr = correlation_matrix['amount'].drop('amount').sort_values(key=abs, ascending=False)
    for cat, val in amount_corr.items():
        print(f"  {cat:35} {val:+.3f}")

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix: Amount vs Categories')
    plt.tight_layout()

    # Save plot
    plot_path = config.PROJECT_ROOT / "docs" / "correlation_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    plt.close()

    return correlation_matrix


if __name__ == "__main__":
    compute_correlations()
