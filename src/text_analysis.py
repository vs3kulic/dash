import pandas as pd
import re
from pathlib import Path
import sys

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import config.config as config


def analyze_text_patterns():
    """Analyze text patterns and keywords in the subject field."""

    # Load training data
    df = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",")

    print("=" * 60)
    print("TEXT PATTERN ANALYSIS")
    print("=" * 60)

    # Common keywords per category
    for category in sorted(df['category'].unique()):
        category_data = df[df['category'] == category]

        # Extract all words from the subject field
        all_text = ' '.join(category_data['subject'].astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())

        # Count word frequencies using a dictionary
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top 5 most common words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\nCategory: {category} ({len(category_data)} samples)")
        print(f"Top keywords: {', '.join([f'{word}({count})' for word, count in top_words])}")

    # Overall word frequency
    print("\n" + "=" * 60)
    print("OVERALL WORD FREQUENCY")
    print("=" * 60)

    all_text = ' '.join(df['subject'].astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Count word frequencies using a dictionary
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Get top 20 most common words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    print("\nTop 20 most common words across all categories:")
    print(", ".join([f"{word}({count})" for word, count in top_words]))


if __name__ == "__main__":
    analyze_text_patterns()