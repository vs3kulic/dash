"""
app.py
======
Shiny app for the transaction categorization project.

Panels
------
1. Data Loading          — upload raw bank CSVs, run processing pipeline
2. Data Explorer         — class distribution, sample counts
3. Hyperparameter Tuning — retrain LR / RF with different settings
4. Model Results         — training vs validation metrics for the last-trained model
"""

import sys
from pathlib import Path
import argparse
import pickle
import pandas as pd
import uvicorn
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Resolve project root before any local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import transform_features, build_feature_matrix
from src.processing import load_data, extract_counterparty, assign_category
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
# Load data once at startup (defaults — can be replaced via Data Loading tab)
# ---------------------------------------------------------------------------
_df_train_default = pd.read_csv(config.TRAIN_OUTPUT_FILE, sep=";", decimal=",", parse_dates=["booking_date"])
_df_val_default   = pd.read_csv(config.VAL_OUTPUT_FILE,   sep=";", decimal=",", parse_dates=["booking_date"])

# ===========================================================================
# UI
# ===========================================================================
app_ui = ui.page_navbar(
    # ── Panel 1: Data Loading ───────────────────────────────────────────────
    ui.nav_panel(
        "Data Loading",
        ui.h3("Data Loading"),
        ui.p(
            "Upload raw bank CSV exports. The processing pipeline will clean the data, "
            "extract counterparties, and assign categories automatically."
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Training data"),
                ui.input_file(
                    "upload_train", "Upload training CSV",
                    accept=[".csv"], multiple=False,
                ),
                ui.output_text_verbatim("upload_train_status"),
            ),
            ui.card(
                ui.card_header("Validation data"),
                ui.input_file(
                    "upload_val", "Upload validation CSV",
                    accept=[".csv"], multiple=False,
                ),
                ui.output_text_verbatim("upload_val_status"),
            ),
            col_widths=(6, 6),
        ),
        ui.card(
            ui.card_header("Data preview (training)"),
            ui.output_text_verbatim("upload_preview"),
        ),
    ),

    # ── Panel 2: Data Explorer ──────────────────────────────────────────────
    ui.nav_panel(
        "Data Explorer",
        ui.h3("Data Explorer"),
        ui.p("Visualize the raw data before touching the model."),
        ui.layout_columns(
            ui.card(
                ui.card_header("Class distribution"),
                ui.output_text_verbatim("data_summary"),
            ),
            ui.card(
                ui.card_header("Sample count per category"),
                ui.output_plot("class_distribution", height="350px"),
            ),
            col_widths=(4, 8),
        ),
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
                ui.h5("Algorithm"),
                ui.input_select(
                    "tune_algo", "Model",
                    choices=["Logistic Regression", "Random Forest"],
                    selected="Logistic Regression",
                ),
                ui.hr(),
                ui.panel_conditional(
                    "input.tune_algo === 'Logistic Regression'",
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
                ),
                ui.panel_conditional(
                    "input.tune_algo === 'Random Forest'",
                    ui.h5("Random Forest"),
                    ui.input_numeric(
                        "tune_n_estimators", "Number of trees",
                        value=200, min=10, max=2000, step=10,
                    ),
                    ui.input_numeric(
                        "tune_max_depth", "Max depth (0 = unlimited)",
                        value=0, min=0, max=100, step=1,
                    ),
                    ui.p(
                        ui.em("More trees = more stable predictions but slower. "
                              "Limiting depth can reduce overfitting."),
                        style="font-size:0.8rem; color:#666;",
                    ),
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

    # ── Panel 4: Model Results ───────────────────────────────────────────────
    ui.nav_panel(
        "Model Results",
        ui.h3("Model Results"),
        ui.p("Training vs validation performance."),
        ui.layout_columns(
            ui.card(
                ui.card_header("Model Details"),
                ui.card_body(
                    ui.output_text_verbatim("model_details"),
                ),
            ),
            ui.card(
                ui.card_header("Model Summary"),
                ui.output_text_verbatim("model_summary"),
            ),
            col_widths=(4, 8),
        ),
    ),

    title="Transaction Categorizer",
    id="navbar",
)

# ===========================================================================
# Server
# ===========================================================================
def server(input, output, session):

    # ── Reactive data: train/val DataFrames ───────────────────────────────
    r_df_train = reactive.Value(_df_train_default)
    r_df_val   = reactive.Value(_df_val_default)

    # ── Reactive values for last-tuned model ────────────────────────────────
    _last_tuned_model      = reactive.Value(None)
    _last_tuned_vectorizer = reactive.Value(None)
    _last_tuned_scaler     = reactive.Value(None)
    _last_tuned_details    = reactive.Value(None)

    # ── Helper: process a raw uploaded CSV ─────────────────────────────────
    def _process_upload(file_info: list | None) -> pd.DataFrame | None:
        """Run the processing pipeline on an uploaded raw bank CSV."""
        if file_info is None or len(file_info) == 0:
            return None
        path = Path(file_info[0]["datapath"])
        src = load_data(path)
        df = src[["booking_date", "subject", "amount"]].copy()
        df["counterparty"] = df["subject"].apply(extract_counterparty)
        df["category"] = df["counterparty"].apply(assign_category)
        return df

    # ── Panel 1: Data Loading ───────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.upload_train)
    def _on_upload_train():
        df = _process_upload(input.upload_train())
        if df is not None:
            r_df_train.set(df)
            # Reset tuned model since data changed
            _last_tuned_model.set(None)
            _last_tuned_details.set(None)

    @reactive.Effect
    @reactive.event(input.upload_val)
    def _on_upload_val():
        df = _process_upload(input.upload_val())
        if df is not None:
            r_df_val.set(df)
            _last_tuned_model.set(None)
            _last_tuned_details.set(None)

    @output
    @render.text
    def upload_train_status():
        df = r_df_train.get()
        n_cats = df["category"].nunique()
        n_uncat = (df["category"] == "uncategorized").sum()
        return (
            f"Loaded: {len(df)} transactions, {n_cats} categories\n"
            f"Uncategorized: {n_uncat}"
        )

    @output
    @render.text
    def upload_val_status():
        df = r_df_val.get()
        n_cats = df["category"].nunique()
        n_uncat = (df["category"] == "uncategorized").sum()
        return (
            f"Loaded: {len(df)} transactions, {n_cats} categories\n"
            f"Uncategorized: {n_uncat}"
        )

    @output
    @render.text
    def upload_preview():
        df = r_df_train.get()
        lines = []
        for _, row in df.head(10).iterrows():
            lines.append(
                f"  {str(row['booking_date'])[:10]}  "
                f"{row['amount']:>10.2f}  "
                f"{row['category']:<22} "
                f"{row['subject'][:60]}"
            )
        header = f"{'  Date':<12} {'Amount':>10}  {'Category':<22} Subject"
        return header + "\n" + "-" * 100 + "\n" + "\n".join(lines) + f"\n\n... ({len(df)} rows total)"

    # ── Panel 2 ──────────────────────────────────────────────────────────────
    @output
    @render.text
    def data_summary():
        df_train = r_df_train.get()
        df_val = r_df_val.get()
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

    @output
    @render.plot
    def class_distribution():
        df_train = r_df_train.get()
        counts = df_train["category"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax, rot=45)
        ax.set_ylabel("Count")
        return fig

    # ── Panel: Model Results (reactive — updates after tuning) ───────────────
    @output
    @render.text
    def model_summary():
        df_train = r_df_train.get()
        df_val = r_df_val.get()
        mdl = _last_tuned_model.get()
        vec = _last_tuned_vectorizer.get()
        sc  = _last_tuned_scaler.get()

        if mdl is None:
            mdl, vec, sc = MODEL, VECTORIZER, SCALER

        y_train = df_train["category"].values
        X_train_t = transform_features(df_train, vec, sc)
        X_val = transform_features(df_val, vec, sc)
        y_val = df_val["category"].values

        f1_train = f1_score(y_train, mdl.predict(X_train_t), average="macro", zero_division=0)
        f1_val   = f1_score(y_val,   mdl.predict(X_val),   average="macro", zero_division=0)

        report = classification_report(
            y_val, mdl.predict(X_val), zero_division=0, digits=2
        )

        return (
            f"Training macro-F1  : {f1_train:.4f}\n"
            f"Validation macro-F1: {f1_val:.4f}\n"
            f"\nPer-class breakdown (validation):\n"
            f"{report}"
        )

    @output
    @render.text
    def model_details():
        details = _last_tuned_details.get()
        if details is not None:
            return details

        return (
            "Algorithm: Multinomial Logistic Regression\n"
            "C: 1.0 (regularization strength)\n"
            "Solver: lbfgs (good for small datasets)\n"
            "Class weight: balanced\n"
            "Feature space: 491 features (TF-IDF + numerical)\n"
            "n-gram range: (1, 2), max_features: 500\n"
        )

    # ── Panel 3: Hyperparameter Tuning ──────────────────────────────────────
    @reactive.Calc
    @reactive.event(input.tune_btn)
    def _tune_result():
        df_train = r_df_train.get()
        df_val = r_df_val.get()
        algo         = input.tune_algo()
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
        X_val_tuned = transform_features(df_val, vectorizer, scaler)
        y_val = df_val["category"].values

        if algo == "Random Forest":
            n_estimators = int(input.tune_n_estimators())
            max_depth_val = int(input.tune_max_depth())
            max_depth = None if max_depth_val == 0 else max_depth_val
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            config_str = f"Algorithm: Random Forest, n_estimators={n_estimators}, max_depth={max_depth}"
        else:
            C = float(input.tune_C())
            model = LogisticRegression(
                C=C, class_weight="balanced", solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            config_str = f"Algorithm: Logistic Regression, C={C}"

        model.fit(X_train, y_train)

        # Store so Model Results tab reflects this model
        _last_tuned_model.set(model)
        _last_tuned_vectorizer.set(vectorizer)
        _last_tuned_scaler.set(scaler)

        # Build details string for the Model Results tab
        detail_lines = [config_str]
        if algo == "Random Forest":
            detail_lines += [
                f"n_estimators: {n_estimators}",
                f"max_depth: {max_depth}",
            ]
        else:
            detail_lines += [
                f"C: {C} (regularization strength)",
                "Solver: lbfgs",
            ]
        detail_lines += [
            "Class weight: balanced",
            f"Feature space: {X_train.shape[1]} features (TF-IDF + numerical)",
            f"n-gram range: {ngram_range}, max_features: {max_features}",
        ]
        _last_tuned_details.set("\n".join(detail_lines))

        f1_train = f1_score(y_train, model.predict(X_train), average="macro", zero_division=0)
        f1_val   = f1_score(y_val,   model.predict(X_val_tuned), average="macro", zero_division=0)

        # Baseline — must use original VECTORIZER/SCALER (different feature space)
        X_train_base, y_train_base, _, _ = build_feature_matrix(df_train)
        X_val_base = transform_features(df_val, VECTORIZER, SCALER)
        f1_base_train = f1_score(y_train_base, MODEL.predict(X_train_base), average="macro", zero_division=0)
        f1_base_val   = f1_score(y_val,        MODEL.predict(X_val_base),   average="macro", zero_division=0)

        # Per-class F1
        classes = sorted(set(y_train))
        lines = [
            f"{config_str}, max_features={max_features}, ngrams={ngram_range}",
            "",
            f"{'':26} {'baseline':>10} {'tuned':>10} {'delta':>10}",
            f"{'Train macro-F1':<26} {f1_base_train:>10.4f} {f1_train:>10.4f} {f1_train - f1_base_train:>+10.4f}",
            f"{'Val macro-F1':<26} {f1_base_val:>10.4f} {f1_val:>10.4f} {f1_val - f1_base_val:>+10.4f}",
            "",
            f"{'Category':<26} {'val F1 (base)':>14} {'val F1 (tuned)':>14}",
            "-" * 56,
        ]
        y_pred_base  = MODEL.predict(X_val_base)
        y_pred_tuned = model.predict(X_val_tuned)
        for cls in classes:
            mask = y_val == cls
            if not mask.any():
                continue
            f1_b = f1_score(y_val, y_pred_base,  labels=[cls], average="macro", zero_division=0)
            f1_t = f1_score(y_val, y_pred_tuned, labels=[cls], average="macro", zero_division=0)
            marker = " ↑" if f1_t > f1_b else (" ↓" if f1_t < f1_b else "")
            lines.append(f"  {cls:<24} {f1_b:>14.4f} {f1_t:>14.4f}{marker}")

        return "\n".join(lines)

    @output
    @render.text
    def tune_result():
        return _tune_result()


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
