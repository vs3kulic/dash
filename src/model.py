"""
model.py
========
Train and evaluate a Logistic Regression classifier on transaction data.

Sections
--------
1. Model Training
2. Model Evaluation
3. Main
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Bootstrap path so config and feature_engineering are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config.config as config
from src.feature_engineering import build_feature_matrix, transform_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"


# ===========================================================================
# SECTION 1 — MODEL TRAINING
# ===========================================================================

def train_logistic_regression(
    X_train: csr_matrix,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Fit a Logistic Regression classifier on the training data.

    Uses ``class_weight='balanced'`` to compensate for class imbalance,
    and the ``lbfgs`` solver which handles multiclass natively via
    one-vs-rest internally.

    :param X_train: Sparse feature matrix from :func:`build_feature_matrix`.
    :param y_train: Label array of category strings.
    :param C: Inverse regularization strength. Smaller = stronger regularization.
    :param max_iter: Maximum iterations for the solver to converge.
    :return: Fitted :class:`~sklearn.linear_model.LogisticRegression`.
    """
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ===========================================================================
# SECTION 2 — MODEL EVALUATION
# ===========================================================================

def evaluate(
    model: LogisticRegression,
    X: csr_matrix,
    y: np.ndarray,
    split_name: str = "Validation",
) -> float:
    """Print a classification report and return the macro-F1 score.

    :param model: Fitted classifier.
    :param X: Feature matrix for the split to evaluate.
    :param y: True labels.
    :param split_name: Label printed in the output header.
    :return: Macro-F1 score.
    """
    y_pred = model.predict(X)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"{split_name} Results")
    print(f"{'=' * 60}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print()
    print(classification_report(y, y_pred, zero_division=0))

    return macro_f1


# ===========================================================================
# SECTION 3 — MAIN
# ===========================================================================

def main() -> None:
    """Load data, build features, train, evaluate, and save the model."""

    # ── Load data ────────────────────────────────────────────────────────────
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

    print(f"Training samples  : {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")

    # ── Build features ───────────────────────────────────────────────────────
    X_train, y_train, vectorizer, scaler = build_feature_matrix(df_train)
    X_val = transform_features(df_val, vectorizer, scaler)
    y_val = df_val["category"].values

    print(f"Feature matrix shape: {X_train.shape}")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining Logistic Regression (C=1.0, class_weight='balanced') ...")
    model = train_logistic_regression(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    evaluate(model, X_train, y_train, split_name="Training")
    evaluate(model, X_val, y_val, split_name="Validation")

    # ── Misclassified validation samples ─────────────────────────────────────
    y_pred_val = model.predict(X_val)
    misclassified = df_val[y_pred_val != y_val].copy()
    misclassified["predicted"] = y_pred_val[y_pred_val != y_val]
    misclassified = misclassified[["subject", "amount", "category", "predicted"]]
    misclassified.columns = ["subject", "amount", "true", "predicted"]

    print(f"\n{'=' * 60}")
    print(f"Misclassified Validation Samples ({len(misclassified)} of {len(df_val)})")
    print(f"{'=' * 60}")
    for _, row in misclassified.iterrows():
        print(f"  true={row['true']:20}  pred={row['predicted']:20}  amount={row['amount']:>10.2f}  subject={row['subject'][:80]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer, "scaler": scaler}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
