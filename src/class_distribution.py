import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import config.config as config


def analyze_class_distribution():
    """Analyze and visualize class distribution in training data."""
    
    # Load training data
    df = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",")
    
    print("="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count samples per category
    class_counts = df["category"].value_counts().sort_values(ascending=False)
    
    print("\nSamples per category:")
    print("-"*40)
    for category, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category:25} {count:4} ({percentage:5.1f}%)")
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Number of categories: {len(class_counts)}")
    
    # Calculate imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nImbalance metrics:")
    print(f"  Largest class:  {class_counts.idxmax()} ({max_class} samples)")
    print(f"  Smallest class: {class_counts.idxmin()} ({min_class} samples)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("\n⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print("   Consider: class_weight='balanced' in LogisticRegression")
    elif imbalance_ratio > 3:
        print("\n⚠️  Moderate class imbalance detected")
        print("   May need class weights or resampling")
    else:
        print("\n✓  Classes are relatively balanced")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='barh')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Category')
    plt.title('Class Distribution in Training Data')
    plt.tight_layout()
    
    # Save plot
    plot_path = config.PROJECT_ROOT / "docs" / "class_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    analyze_class_distribution()