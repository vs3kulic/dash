"""
categorize.py
=============
Production app for monthly bank transaction categorization.

Flow
----
1. Upload a raw bank CSV export
2. Processing pipeline auto-labels known counterparties via category_mapping.json
3. Trained RF model predicts the remaining uncategorized transactions
4. User reviews all transactions, can override any label
5. Download the fully labeled CSV
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
import uvicorn
from shiny import App, ui, render, reactive

# ---------------------------------------------------------------------------
# Resolve project root before any local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import transform_features
from src.processing import load_data, extract_counterparty, assign_category

# ---------------------------------------------------------------------------
# Load the trained Random Forest model once at startup
# ---------------------------------------------------------------------------
RF_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

with open(RF_MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

MODEL = _bundle["model"]
VECTORIZER = _bundle["vectorizer"]
SCALER = _bundle["scaler"]

CATEGORIES = sorted(MODEL.classes_.tolist())

# ===========================================================================
# UI
# ===========================================================================
app_ui = ui.page_navbar(
    # ── Tab 1: Upload & Categorize ──────────────────────────────────────────
    ui.nav_panel(
        "Categorize",
        ui.h3("Monthly Transaction Categorization"),
        ui.p(
            "Upload this month's raw bank CSV export. "
            "Known counterparties are labeled by rules; the rest are predicted by the model."
        ),
        ui.card(
            ui.card_header("Upload"),
            ui.input_file(
                "upload_csv", "Bank CSV export",
                accept=[".csv"], multiple=False,
            ),
            ui.output_text_verbatim("upload_status"),
        ),
        ui.card(
            ui.card_header("All Transactions"),
            ui.output_data_frame("tx_table"),
        ),
    ),

    # ── Tab 2: Review & Export ──────────────────────────────────────────────
    ui.nav_panel(
        "Review & Export",
        ui.h3("Review & Export"),
        ui.p("Override any label, then download the final CSV."),
        ui.layout_columns(
            ui.card(
                ui.card_header("Fix a transaction"),
                ui.input_numeric(
                    "fix_row", "Row number (from table)",
                    value=1, min=1, step=1,
                ),
                ui.input_select(
                    "fix_category", "Correct category",
                    choices=CATEGORIES,
                ),
                ui.input_action_button(
                    "fix_btn", "Apply",
                    class_="btn-warning w-100",
                ),
                ui.output_text_verbatim("fix_status"),
            ),
            ui.card(
                ui.card_header("Summary"),
                ui.output_text_verbatim("summary"),
            ),
            col_widths=(4, 8),
        ),
        ui.download_button(
            "download_csv", "Download Labeled CSV",
            class_="btn-success mt-2",
        ),
    ),

    title="Transaction Categorizer",
    id="navbar",
)


# ===========================================================================
# Server
# ===========================================================================
def server(input, output, session):

    # Reactive DataFrame holding the current transactions + labels
    r_df = reactive.Value(pd.DataFrame())

    # ── Upload & process ────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.upload_csv)
    def _on_upload():
        file_info = input.upload_csv()
        if file_info is None or len(file_info) == 0:
            return

        path = Path(file_info[0]["datapath"])

        # Step 1: Processing pipeline (clean, extract counterparty, rules)
        src = load_data(path)
        df = src[["booking_date", "subject", "amount"]].copy()
        df["counterparty"] = df["subject"].apply(extract_counterparty)
        df["category"] = df["counterparty"].apply(assign_category)

        # Step 2: ML prediction for uncategorized rows
        uncat_mask = df["category"] == "uncategorized"
        n_uncat = uncat_mask.sum()

        # Track how each label was assigned
        df["source"] = "rule"

        if n_uncat > 0:
            X_uncat = transform_features(df[uncat_mask], VECTORIZER, SCALER)
            predictions = MODEL.predict(X_uncat)
            probabilities = MODEL.predict_proba(X_uncat).max(axis=1)

            df.loc[uncat_mask, "category"] = predictions
            df.loc[uncat_mask, "source"] = [
                f"model ({p:.0%})" for p in probabilities
            ]

        r_df.set(df)

    # ── Upload status ───────────────────────────────────────────────────────
    @output
    @render.text
    def upload_status():
        df = r_df.get()
        if df.empty:
            return "No file uploaded yet."
        n_rule = (df["source"] == "rule").sum()
        n_model = (~df["source"].isin(["rule", "manual"])).sum()
        n_manual = (df["source"] == "manual").sum()
        return (
            f"{len(df)} transactions loaded\n"
            f"  Rules:  {n_rule}\n"
            f"  Model:  {n_model}\n"
            f"  Manual: {n_manual}"
        )

    # ── Transaction table ───────────────────────────────────────────────────
    @render.data_frame
    def tx_table():
        df = r_df.get()
        if df.empty:
            return render.DataGrid(pd.DataFrame())
        display = df[["booking_date", "subject", "amount", "category", "source"]].copy()
        display["booking_date"] = display["booking_date"].astype(str).str[:10]
        display.insert(0, "row", range(1, len(display) + 1))
        return render.DataGrid(display, filters=True)

    # ── Manual override ─────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.fix_btn)
    def _apply_fix():
        df = r_df.get()
        if df.empty:
            return
        row_num = int(input.fix_row()) - 1  # 0-indexed
        if row_num < 0 or row_num >= len(df):
            return
        new_cat = input.fix_category()
        df_copy = df.copy()
        df_copy.iloc[row_num, df_copy.columns.get_loc("category")] = new_cat
        df_copy.iloc[row_num, df_copy.columns.get_loc("source")] = "manual"
        r_df.set(df_copy)

    @output
    @render.text
    def fix_status():
        df = r_df.get()
        if df.empty:
            return ""
        n_manual = (df["source"] == "manual").sum()
        if n_manual == 0:
            return "No manual overrides yet."
        return f"{n_manual} manual override(s) applied."

    # ── Summary ─────────────────────────────────────────────────────────────
    @output
    @render.text
    def summary():
        df = r_df.get()
        if df.empty:
            return "Upload a file to see the summary."
        lines = [
            f"Total transactions: {len(df)}",
            "",
            "Category breakdown:",
        ]
        counts = df["category"].value_counts()
        for cat, n in counts.items():
            lines.append(f"  {cat:<25} {n:>3}")
        lines += [
            "",
            "Label sources:",
            f"  Rule-based:   {(df['source'] == 'rule').sum()}",
            f"  Model:        {df['source'].str.startswith('model').sum()}",
            f"  Manual:       {(df['source'] == 'manual').sum()}",
        ]
        return "\n".join(lines)

    # ── CSV download ────────────────────────────────────────────────────────
    @render.download(filename="categorized_transactions.csv")
    async def download_csv():
        df = r_df.get()
        export = df[["booking_date", "subject", "amount", "counterparty", "category"]].copy()
        yield export.to_csv(index=False, sep=";", decimal=",")


# ===========================================================================
# Entry point
# ===========================================================================
app = App(app_ui, server)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
