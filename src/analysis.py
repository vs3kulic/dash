"""
analysis.py
===========
Unified exploratory data analysis pipeline.

Sections
--------
1. Class Distribution
2. Correlation Analysis
3. Location Measures
4. Text Pattern Analysis
"""

import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path so config is importable regardless of working directory
# ---------------------------------------------------------------------------
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import config.config as config


# ===========================================================================
# SECTION 1 — CLASS DISTRIBUTION
# ===========================================================================

def analyze_class_distribution(df: pd.DataFrame) -> None:
    """Analyze and visualize class distribution in training data."""

    print("=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    class_counts = df["category"].value_counts().sort_values(ascending=False)

    print("\nSamples per category:")
    print("-" * 40)
    for category, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category:25} {count:4} ({percentage:5.1f}%)")

    print(f"\n  Total samples:      {len(df)}")
    print(f"  Number of classes:  {len(class_counts)}")

    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class

    print(f"\n  Largest class:   {class_counts.idxmax()} ({max_class} samples)")
    print(f"  Smallest class:  {class_counts.idxmin()} ({min_class} samples)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio > 10:
        print("\n  ⚠️  SEVERE CLASS IMBALANCE — consider class_weight='balanced'")
    elif imbalance_ratio > 3:
        print("\n  ⚠️  Moderate class imbalance — consider class weights or resampling")
    else:
        print("\n  ✓  Classes are relatively balanced")

    plt.figure(figsize=(12, 6))
    class_counts.plot(kind="barh")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Category")
    plt.title("Class Distribution in Training Data")
    plt.tight_layout()

    plot_path = config.PROJECT_ROOT / "docs" / "class_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved → {plot_path}")
    plt.close()


# ===========================================================================
# SECTION 2 — CORRELATION ANALYSIS
# ===========================================================================

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and visualize correlations between amount and transaction categories."""

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    df_encoded = pd.get_dummies(df[["amount", "category"]], columns=["category"], prefix="cat")
    correlation_matrix = df_encoded.corr()

    print("\nCorrelation of 'amount' with each category:")
    print("-" * 60)
    amount_corr = (
        correlation_matrix["amount"]
        .drop("amount")
        .sort_values(key=abs, ascending=False)
    )
    for cat, val in amount_corr.items():
        print(f"  {cat:35} {val:+.3f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix: Amount vs Categories")
    plt.tight_layout()

    plot_path = config.PROJECT_ROOT / "docs" / "correlation_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved → {plot_path}")
    plt.close()

    return correlation_matrix


# ===========================================================================
# SECTION 3 — LOCATION MEASURES
# ===========================================================================

def compute_location_measures(df: pd.DataFrame) -> None:
    """Compute central tendency, spread, outlier, and per-category statistics."""

    print("=" * 60)
    print("LOCATION MEASURES")
    print("=" * 60)

    # --- Central tendency ------------------------------------------------
    print("\n  Transaction Amount Statistics:")
    print("-" * 60)
    print(f"  Mean (Average):        €{df['amount'].mean():>12,.2f}")
    print(f"  Median (Middle):       €{df['amount'].median():>12,.2f}")
    print(f"  Mode (Most common):    €{df['amount'].mode().values[0]:>12,.2f}")
    print(f"\n  Minimum:               €{df['amount'].min():>12,.2f}")
    print(f"  Maximum:               €{df['amount'].max():>12,.2f}")
    print(f"  Range:                 €{df['amount'].max() - df['amount'].min():>12,.2f}")

    # --- Spread -----------------------------------------------------------
    print("\n  Spread Measures:")
    print("-" * 60)
    print(f"  Standard Deviation:    €{df['amount'].std():>12,.2f}")
    print(f"  Variance:              €{df['amount'].var():>12,.2f}")
    iqr = df["amount"].quantile(0.75) - df["amount"].quantile(0.25)
    print(f"  IQR (Q3 - Q1):         €{iqr:>12,.2f}")

    # --- Quartiles --------------------------------------------------------
    print("\n  Quartiles:")
    print("-" * 60)
    for label, q in [("25th (Q1)", 0.25), ("50th (Q2)", 0.50), ("75th (Q3)", 0.75)]:
        print(f"  {label} percentile:  €{df['amount'].quantile(q):>12,.2f}")

    # --- Outliers (IQR method) --------------------------------------------
    Q1, Q3 = df["amount"].quantile(0.25), df["amount"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df["amount"] < lower) | (df["amount"] > upper)]

    print("\n  Outlier Detection (IQR method):")
    print("-" * 60)
    print(f"  Lower bound:           €{lower:>12,.2f}")
    print(f"  Upper bound:           €{upper:>12,.2f}")
    print(f"  Number of outliers:    {len(outliers):>12}")

    if len(outliers) > 0:
        print("\n  Outlier transactions:")
        print(
            outliers[["booking_date", "counterparty", "amount", "category"]]
            .to_string(index=False)
        )

    # --- Per-category stats -----------------------------------------------
    print("\n  Amount Statistics by Category:")
    print("=" * 60)
    category_stats = (
        df.groupby("category")["amount"]
        .agg(count="count", mean="mean", median="median", total="sum")
        .sort_values("total", ascending=False)
    )
    print(category_stats.to_string())

    # --- Data quality -----------------------------------------------------
    print("\n  Missing Values:")
    print("-" * 60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓  No missing values found")
    else:
        print(missing[missing > 0])

    print("\n  Data Types:")
    print("-" * 60)
    print(df.dtypes.to_string())


# ===========================================================================
# SECTION 4 — TEXT PATTERN ANALYSIS
# ===========================================================================

def analyze_text_patterns(df: pd.DataFrame) -> None:
    """Analyze text patterns and keywords in the subject field."""

    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("TEXT PATTERN ANALYSIS")
    lines.append("=" * 60)

    for category in sorted(df["category"].unique()):
        subset = df[df["category"] == category]
        all_text = " ".join(subset["subject"].astype(str))
        words = re.findall(r"\b\w+\b", all_text.lower())

        word_freq: dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"{w}({c})" for w, c in top_words)

        lines.append(f"\n  [{category}]  ({len(subset)} samples)")
        lines.append(f"  Top keywords: {top_str}")

    lines.append("\n" + "=" * 60)
    lines.append("OVERALL WORD FREQUENCY")
    lines.append("=" * 60)

    all_text = " ".join(df["subject"].astype(str))
    words = re.findall(r"\b\w+\b", all_text.lower())

    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    lines.append("\n  Top 20 words across all categories:")
    lines.append("  " + ", ".join(f"{w}({c})" for w, c in top_words))

    output = "\n".join(lines)
    print(output)

    txt_path = config.PROJECT_ROOT / "docs" / "text_analysis.txt"
    txt_path.write_text(output, encoding="utf-8")
    print(f"\n  Results saved → {txt_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run the full EDA pipeline in sequence."""

    print("\n" + "#" * 60)
    print("#  EXPLORATORY DATA ANALYSIS PIPELINE")
    print(f"#  Source: {config.TRAIN_OUTPUT_FILE}")
    print("#" * 60 + "\n")

    df = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",")

    analyze_class_distribution(df)
    print()
    compute_correlations(df)
    print()
    compute_location_measures(df)
    print()
    analyze_text_patterns(df)

    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
