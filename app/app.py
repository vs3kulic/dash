"""
app.py
======
Shiny app for the transaction categorization project.

Panels
------
1. Data Explorer  — class distribution, amount by category, top tokens
2. Model Results  — training vs validation metrics, confusion matrix
3. Prediction     — type a subject + amount, get a predicted category
"""

import sys
from pathlib import Path
import argparse
import pickle
import pandas as pd
import uvicorn
from datetime import date
from shiny import App, ui, render, reactive
from sklearn.metrics import f1_score

# Resolve project root before any local imports
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import transform_features, build_feature_matrix
import config.config as config

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

MODEL = _bundle["model"]
VECTORIZER = _bundle["vectorizer"]
SCALER = _bundle["scaler"]

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------
df_train = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",", parse_dates=["booking_date"])
df_val   = pd.read_csv(config.VAL_OUTPUT_FILE,   sep=";", decimal=",", parse_dates=["booking_date"])

# ===========================================================================
# UI
# ===========================================================================
app_ui = ui.page_navbar(
    # ── Panel 1: Data Explorer ───────────────────────────────────────────────
    ui.nav_panel(
        "Data Explorer",
        ui.h3("Data Explorer"),
        ui.p("Visualize the raw data before touching the model."),
        ui.output_text_verbatim("data_summary"),
    ),

    # ── Panel 2: Model Results ───────────────────────────────────────────────
    ui.nav_panel(
        "Model Results",
        ui.h3("Model Results"),
        ui.p("Training vs validation performance."),
        ui.div(
            ui.h5("What is Macro F1?"),
            ui.p(
                "F1 is the harmonic mean of precision and recall for a single class. "
                "Macro F1 averages the per-class F1 scores without weighting by class size, "
                "so a rare class with F1=0.00 hurts the score just as much as a common one. "
                "This makes it a strict measure: the model must perform well across ",
                ui.strong("all"), " categories, not just the frequent ones."
            ),
            ui.p(
                ui.em("Rule of thumb: "), "Training F1 ≫ Validation F1 signals overfitting. "
                "Here, the gap is driven mainly by ", ui.code("business_equipment"),
                ", which scores F1=0.96 on training data but F1=0.00 on validation — "
                "likely because it acts as a catch-all category with too few and too diverse samples to generalize."
            ),
            class_="alert alert-secondary",
            style="font-size: 0.9rem;",
        ),
        ui.output_text_verbatim("model_summary"),
    ),

    # ── Panel 3: Hyperparameter Tuning ────────────────────────────────────────
    ui.nav_panel(
        "Hyperparameter Tuning",
        ui.h3("Hyperparameter Tuning"),
        ui.p(
            "Retrain the model with different settings and see the effect on macro-F1 immediately. "
            "Compare against the baseline (C=1.0, max_features=500, ngrams 1–2)."
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Logistic Regression"),
                ui.input_numeric(
                    "tune_C", "C (regularization)",
                    value=1.0, min=0.01, max=100.0, step=0.1,
                ),
                ui.p(
                    ui.em("Smaller C = stronger regularization = simpler model. "
                          "Try lowering it if the model overfits."),
                    style="font-size:0.8rem; color:#666;",
                ),
                ui.hr(),
                ui.h5("TF-IDF"),
                ui.input_numeric(
                    "tune_max_features", "max_features (vocabulary size)",
                    value=500, min=50, max=5000, step=50,
                ),
                ui.input_numeric(
                    "tune_ngram_min", "n-gram min",
                    value=1, min=1, max=3, step=1,
                ),
                ui.input_numeric(
                    "tune_ngram_max", "n-gram max",
                    value=2, min=1, max=3, step=1,
                ),
                ui.p(
                    ui.em("(1,1) = unigrams only. (1,2) adds bigrams. "
                          "Bigrams capture phrases like 'google ads' as one token."),
                    style="font-size:0.8rem; color:#666;",
                ),
                ui.hr(),
                ui.input_action_button("tune_btn", "Retrain & Evaluate", class_="btn-primary w-100"),
            ),
            ui.output_text_verbatim("tune_result"),
        ),
    ),

    # ── Panel 4: Prediction Tool ─────────────────────────────────────────────
    ui.nav_panel(
        "Prediction",
        ui.h3("Prediction Tool"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text("subject", "Subject text", placeholder="e.g. Google Ads Payment"),
                ui.input_numeric("amount", "Amount (€)", value=-50.0, step=0.01),
                ui.input_action_button("predict_btn", "Predict", class_="btn-primary"),
            ),
            ui.output_text_verbatim("prediction_result"),
        ),
    ),

    title="Transaction Categorizer",
    id="navbar",
)

# ===========================================================================
# Server
# ===========================================================================
def server(input, output, session):

    # ── Panel 1 ──────────────────────────────────────────────────────────────
    @output
    @render.text
    def data_summary():
        lines = [
            f"Training samples  : {len(df_train)}",
            f"Validation samples: {len(df_val)}",
            "",
            "Class distribution (train):",
        ]
        counts = df_train["category"].value_counts()
        for cat, n in counts.items():
            lines.append(f"  {cat:<25} {n:>3}")
        return "\n".join(lines)

    # ── Panel 2 ──────────────────────────────────────────────────────────────
    @output
    @render.text
    def model_summary():
        X_train, y_train, _, _ = build_feature_matrix(df_train)
        X_val = transform_features(df_val, VECTORIZER, SCALER)
        y_val = df_val["category"].values

        f1_train = f1_score(y_train, MODEL.predict(X_train), average="macro", zero_division=0)
        f1_val   = f1_score(y_val,   MODEL.predict(X_val),   average="macro", zero_division=0)

        return (
            f"Training macro-F1  : {f1_train:.4f}\n"
            f"Validation macro-F1: {f1_val:.4f}\n"
        )

    # ── Panel 3: Hyperparameter Tuning ──────────────────────────────────────
    @reactive.Calc
    @reactive.event(input.tune_btn)
    def _tune_result():
        from sklearn.linear_model import LogisticRegression

        C            = float(input.tune_C())
        max_features = int(input.tune_max_features())
        ngram_min    = int(input.tune_ngram_min())
        ngram_max    = int(input.tune_ngram_max())

        if ngram_min > ngram_max:
            return "n-gram min must be ≤ n-gram max."

        ngram_range = (ngram_min, ngram_max)

        # Retrain from scratch with new hyperparameters
        X_train, y_train, vectorizer, scaler = build_feature_matrix(
            df_train, ngram_range=ngram_range, max_features=max_features
        )
        X_val = transform_features(df_val, vectorizer, scaler)
        y_val = df_val["category"].values

        model = LogisticRegression(
            C=C, class_weight="balanced", solver="lbfgs",
            max_iter=1000, random_state=42,
        )
        model.fit(X_train, y_train)

        f1_train = f1_score(y_train, model.predict(X_train), average="macro", zero_division=0)
        f1_val   = f1_score(y_val,   model.predict(X_val),   average="macro", zero_division=0)

        # Baseline (model loaded at startup)
        f1_base_train = f1_score(y_train, MODEL.predict(X_train), average="macro", zero_division=0)
        f1_base_val   = f1_score(y_val,   MODEL.predict(X_val),   average="macro", zero_division=0)

        # Per-class F1
        from sklearn.metrics import f1_score as f1
        classes = sorted(set(y_train))
        lines = [
            f"Config : C={C}, max_features={max_features}, ngrams={ngram_range}",
            "",
            f"{'':26} {'baseline':>10} {'tuned':>10} {'delta':>10}",
            f"{'Train macro-F1':<26} {f1_base_train:>10.4f} {f1_train:>10.4f} {f1_train - f1_base_train:>+10.4f}",
            f"{'Val macro-F1':<26} {f1_base_val:>10.4f} {f1_val:>10.4f} {f1_val - f1_base_val:>+10.4f}",
            "",
            f"{'Category':<26} {'val F1 (base)':>14} {'val F1 (tuned)':>14}",
            "-" * 56,
        ]
        y_pred_base  = MODEL.predict(X_val)
        y_pred_tuned = model.predict(X_val)
        for cls in classes:
            mask = y_val == cls
            if not mask.any():
                continue
            f1_b = f1(y_val, y_pred_base,  labels=[cls], average="macro", zero_division=0)
            f1_t = f1(y_val, y_pred_tuned, labels=[cls], average="macro", zero_division=0)
            marker = " ↑" if f1_t > f1_b else (" ↓" if f1_t < f1_b else "")
            lines.append(f"  {cls:<24} {f1_b:>14.4f} {f1_t:>14.4f}{marker}")

        return "\n".join(lines)

    @output
    @render.text
    def tune_result():
        return _tune_result()

    # ── Panel 4: Prediction ───────────────────────────────────────────────────
    @reactive.Calc
    @reactive.event(input.predict_btn)
    def _prediction():
        subject = input.subject().strip()
        amount  = float(input.amount())

        if not subject:
            return None, None

        # Build a single-row DataFrame matching the processed CSV schema
        today = pd.Timestamp(date.today())
        row = pd.DataFrame([{
            "booking_date": today,
            "subject": subject,
            "amount": amount,
            "counterparty": "",
            "category": "unknown",
        }])

        X = transform_features(row, VECTORIZER, SCALER)
        predicted = MODEL.predict(X)[0]
        proba = MODEL.predict_proba(X)[0]
        classes = MODEL.classes_

        return predicted, sorted(zip(classes, proba), key=lambda x: -x[1])

    @output
    @render.text
    def prediction_result():
        predicted, scores = _prediction()
        if predicted is None:
            return "Enter a subject and click Predict."

        lines = [f"Predicted category: {predicted}", "", "Confidence scores:"]
        for cat, p in scores:
            bar = "█" * int(p * 30)
            lines.append(f"  {cat:<25} {p:.2%}  {bar}")
        return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================
app = App(app_ui, server)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
