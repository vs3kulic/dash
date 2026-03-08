# Bank Transaction Categorization

A Python project for processing, analyzing, and automatically categorizing bank transactions using supervised machine learning, with an interactive Shiny dashboard for exploration and hyperparameter tuning.

## Goal

Automatically categorize bank transactions based on subject text and amount, using ~250 labeled transactions across 13 categories.

## Project Structure

```
Dash/
├── app/
│   ├── dev.py                   # Model development dashboard
│   └── categorize.py            # Monthly production categorization
├── models/                      # Pickled model artifacts (.pkl)
├── src/
│   ├── processing.py            # ETL: load, clean, extract, categorize
│   ├── analysis.py              # EDA: distributions, correlations, text
│   ├── feature_engineering.py   # TF-IDF + numerical feature pipeline
│   └── model.py                 # Train & evaluate LR and RF classifiers
├── requirements.txt
└── README.md
```

## Pipeline — Step by Step

### 1. Data Processing (`src/processing.py`)

- Load raw bank CSV exports (semicolon-separated, German decimal format)
- Clean and normalize text fields (subject, counterparty)
- Apply rule-based category labels via `category_mapping.json`
- Split into training and validation CSVs saved to `data/processed/`

### 2. Feature Engineering (`src/feature_engineering.py`)

- **Text cleaning**: Strip German banking boilerplate (IBAN, BIC, "ONLINE BANKING VOM …", etc.) from subject strings
- **TF-IDF vectorization**: Convert cleaned text into a sparse matrix (configurable n-gram range and vocabulary size)
- **Temporal features**: Extract day, month, and day-of-week from booking dates
- **Numerical features**: Amount, amount sign, plus the temporal columns — all StandardScaler-normalized
- **Combine**: Horizontally stack TF-IDF (sparse) and numerical (dense→sparse) into a single feature matrix

### 3. Model Training (`src/model.py`)

- **Logistic Regression**: `C=1.0`, `class_weight='balanced'`, `lbfgs` solver
- **Random Forest**: `n_estimators=200`, `class_weight='balanced'`, unlimited depth
- Both models are evaluated on train and validation splits (macro-F1, per-class classification report)
- Misclassified validation samples are printed for error analysis
- Each model is pickled to `models/` along with its vectorizer and scaler

### 4. Interactive Dashboard (`app/dev.py`)

A Shiny for Python app with four tabs:

| Tab | Purpose |
|-----|---------|
| **Data Loading** | Upload raw bank CSVs — the processing pipeline runs automatically (clean, extract counterparties, assign categories). Shows upload status and a data preview. All other tabs update reactively. |
| **Data Explorer** | Class distribution chart, sample counts per category |
| **Hyperparameter Tuning** | Pick LR or RF, adjust algorithm + TF-IDF settings, retrain live, compare against baseline (delta table with ↑/↓ markers) |
| **Model Results** | Full classification report and macro-F1 for the last-trained model (updates automatically after tuning) |

Run the app:

```bash
python app/dev.py              # model development — http://127.0.0.1:8008
python app/categorize.py       # monthly categorization — http://127.0.0.1:8009
```

## Setup

```bash
conda create -n <env> python=3.11
conda activate <env>
pip install -r requirements.txt
pip install -e .
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built transaction by transaction. One day the model will know where the money went before we do.*

