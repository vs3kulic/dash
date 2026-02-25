"""
feature_engineering.py
=======================
Transforms processed transaction data into ML-ready feature matrices.

Sections
--------
1. Text Cleaning
2. Temporal Feature Extraction
3. TF-IDF Vectorization
4. Numerical Feature Scaling
5. Full Feature Pipeline
6. Build Feature Matrix
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Bootstrap path so config is importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config.config as config

# ---------------------------------------------------------------------------
# TF-IDF configuration defaults
# ---------------------------------------------------------------------------
TFIDF_NGRAM_RANGE = (1, 2)   # Unigrams + bigrams
TFIDF_MAX_FEATURES = 500     # Vocabulary cap — revisit after first eval

# German banking boilerplate patterns to strip from subject field before
# vectorization so the classifier focuses on meaningful tokens.
_BOILERPLATE_PATTERNS: list[str] = [
    r"ONLINE BANKING VOM \d{2}\.\d{2}(?: UM \d{2}:\d{2})?",
    r"Auftraggeber:\s*",
    r"Empf[äÄ]nger:\s*",
    r"Zahlungsreferenz:\s*",
    r"IBAN\s+(?:Auftraggeber|Empf[äÄ]nger):\s*[A-Z0-9]+",
    r"BIC\s+(?:Auftraggeber|Empf[äÄ]nger):\s*[A-Z0-9]+",
    r"AT\d{18,20}",          # Austrian IBAN numbers
    r"[A-Z]{4}AT[A-Z0-9]+",  # BIC codes
    r"\d{2}\.\d{2}\b",       # Bare date fragments (DD.MM)
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), flags=re.IGNORECASE)


# ===========================================================================
# SECTION 1 — TEXT CLEANING
# ===========================================================================

def clean_subject(text: str) -> str:
    """Strip banking boilerplate from a transaction subject string.

    Removes IBAN/BIC numbers, standard German banking prefixes, and
    normalizes whitespace so TF-IDF can focus on meaningful tokens.

    :param text: Raw subject string from the processed CSV.
    :return: Cleaned, lowercased string.
    """
    if not isinstance(text, str):
        return ""

    # Remove boilerplate patterns
    cleaned = _BOILERPLATE_RE.sub(" ", text)

    # Collapse multiple whitespace and strip edges
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Lowercase for consistent vocabulary
    return cleaned.lower()


# ===========================================================================
# SECTION 2 — TEMPORAL FEATURE EXTRACTION
# ===========================================================================

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived columns to the DataFrame in-place.

    New columns added:
        - ``day``         : Day of month (1–31)
        - ``month``       : Month number (1–12)
        - ``day_of_week`` : Day of week (0 = Monday, 6 = Sunday)

    :param df: DataFrame containing a ``booking_date`` column (datetime).
    :return: DataFrame with new temporal columns appended.
    """
    df = df.copy()
    dates = pd.to_datetime(df["booking_date"])
    df["day"] = dates.dt.day
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    return df


# ===========================================================================
# SECTION 3 — TF-IDF VECTORIZATION
# ===========================================================================

def build_tfidf_vectorizer(
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> TfidfVectorizer:
    """Create a configured TF-IDF vectorizer.

    :param ngram_range: Tuple (min_n, max_n) for the n-gram range.
    :param max_features: Maximum number of features (vocabulary size).
    :return: Unfitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,        # Replace TF with 1+log(TF) to dampen frequency
        strip_accents="unicode",  # Normalize German umlauts (ä→a, ö→o, ü→u)
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-ZäöüÄÖÜß]{2,}\b",  # Skip pure numbers / 1-char tokens
        min_df=1,                 # Keep rare terms (small dataset)
    )


def fit_tfidf(
    texts: list[str],
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer on *texts* and return the vectorizer + matrix.

    :param texts: Cleaned subject strings for the training corpus.
    :param ngram_range: Passed to :func:`build_tfidf_vectorizer`.
    :param max_features: Passed to :func:`build_tfidf_vectorizer`.
    :return: ``(fitted_vectorizer, sparse_tfidf_matrix)``
    """
    vectorizer = build_tfidf_vectorizer(ngram_range, max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    return vectorizer, X_tfidf


# ===========================================================================
# SECTION 4 — NUMERICAL FEATURE SCALING
# ===========================================================================

def build_numerical_features(df: pd.DataFrame) -> np.ndarray:
    """Construct the raw numerical feature array from a processed DataFrame.

    Features:
        - ``amount``      : Raw transaction amount (negative = expense).
        - ``amount_sign`` : +1 for income, -1 for expense (binary direction).
        - ``day``         : Day of month (1–31).
        - ``month``       : Calendar month (1–12).
        - ``day_of_week`` : Day of week (0 = Monday, 6 = Sunday).

    Assumes :func:`extract_temporal_features` has already been applied.

    :param df: DataFrame with ``amount``, ``day``, ``month``, ``day_of_week``.
    :return: 2-D float array of shape ``(n_samples, 5)``.
    """
    amount = df["amount"].values.astype(float)
    amount_sign = np.sign(amount)  # +1.0 / -1.0 / 0.0

    numerical = np.column_stack([
        amount,
        amount_sign,
        df["day"].values,
        df["month"].values,
        df["day_of_week"].values,
    ])
    return numerical


def fit_scaler(numerical: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit a StandardScaler on *numerical* and return the scaler + scaled array.

    :param numerical: Raw numerical feature array from :func:`build_numerical_features`.
    :return: ``(fitted_scaler, scaled_array)``
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical)
    return scaler, X_scaled


# ===========================================================================
# SECTION 5 — FULL FEATURE PIPELINE
# ===========================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> tuple[csr_matrix, np.ndarray, TfidfVectorizer, StandardScaler]:
    """Build the full feature matrix X and label vector y from a processed DataFrame.

    Pipeline steps:
        1. Clean subject text.
        2. Extract temporal features (day, month, day_of_week).
        3. Fit TF-IDF on cleaned text.
        4. Build and scale numerical features.
        5. Horizontally stack sparse TF-IDF and dense numerical matrices.

    :param df: Processed DataFrame with columns
               ``booking_date``, ``subject``, ``amount``, ``category``.
    :param ngram_range: N-gram range for TF-IDF.
    :param max_features: Maximum TF-IDF vocabulary size.
    :return: ``(X, y, fitted_vectorizer, fitted_scaler)``

        - ``X``  : Sparse feature matrix of shape ``(n, tfidf_features + 5)``.
        - ``y``  : Label array of shape ``(n,)`` with category strings.
        - ``fitted_vectorizer`` : Can be used to transform new data.
        - ``fitted_scaler``     : Can be used to transform new data.
    """
    # Step 1 — clean text
    cleaned_texts = df["subject"].apply(clean_subject).tolist()

    # Step 2 — temporal features
    df_temporal = extract_temporal_features(df)

    # Step 3 — TF-IDF
    vectorizer, X_tfidf = fit_tfidf(cleaned_texts, ngram_range, max_features)

    # Step 4 — numerical features + scaling
    X_numerical_raw = build_numerical_features(df_temporal)
    scaler, X_numerical = fit_scaler(X_numerical_raw)

    # Step 5 — combine sparse + dense
    X_numerical_sparse = csr_matrix(X_numerical)
    X = hstack([X_tfidf, X_numerical_sparse])

    # Labels
    y = df["category"].values

    return X, y, vectorizer, scaler


def transform_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
) -> csr_matrix:
    """Transform a new dataset using *already fitted* vectorizer and scaler.

    Use this to prepare the validation set after fitting on training data.

    :param df: Processed DataFrame (same schema as training data).
    :param vectorizer: Fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    :param scaler: Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    :return: Sparse feature matrix matching the training feature space.
    """
    cleaned_texts = df["subject"].apply(clean_subject).tolist()
    df_temporal = extract_temporal_features(df)

    X_tfidf = vectorizer.transform(cleaned_texts)

    X_numerical_raw = build_numerical_features(df_temporal)
    X_numerical = scaler.transform(X_numerical_raw)

    X_numerical_sparse = csr_matrix(X_numerical)
    return hstack([X_tfidf, X_numerical_sparse])


# ===========================================================================
# SECTION 6 — MAIN (demo / smoke test)
# ===========================================================================

def main() -> None:
    """Load processed train/val data, build features, and print a summary."""

    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # ── Load training data ──────────────────────────────────────────────────
    df_train = pd.read_csv(
        config.TRAIN_OUTPUT_FILE,
        sep=";",
        decimal=",",
        parse_dates=["booking_date"],
    )
    df_val = pd.read_csv(
        config.VAL_OUTPUT_FILE,
        sep=";",
        decimal=",",
        parse_dates=["booking_date"],
    )

    print(f"\nTraining samples : {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    # ── Build training features ─────────────────────────────────────────────
    print(f"\nTF-IDF config  : ngram_range={TFIDF_NGRAM_RANGE}, max_features={TFIDF_MAX_FEATURES}")
    X_train, y_train, vectorizer, scaler = build_feature_matrix(df_train)

    print(f"\nTraining feature matrix shape : {X_train.shape}")
    print(f"  TF-IDF features  : {vectorizer.get_feature_names_out().shape[0]}")
    print(f"  Numerical features: 5  (amount, amount_sign, day, month, day_of_week)")
    print(f"  Total features   : {X_train.shape[1]}")

    # ── Sample token vocabulary ─────────────────────────────────────────────
    vocab = vectorizer.get_feature_names_out()
    print(f"\nSample vocabulary (first 20 tokens): {list(vocab[:20])}")

    # ── Transform validation set ────────────────────────────────────────────
    X_val = transform_features(df_val, vectorizer, scaler)
    y_val = df_val["category"].values

    print(f"\nValidation feature matrix shape: {X_val.shape}")

    # ── Class distribution in y ─────────────────────────────────────────────
    print("\nLabel distribution (training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {label:30} {count}")

    print("\nFeature engineering complete.")
    return X_train, y_train, X_val, y_val, vectorizer, scaler


if __name__ == "__main__":
    main()
