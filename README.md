# Bank Transaction Analysis

A Python-based tool for processing and analyzing bank transaction data. This project incorporates supervised machine learning for transaction categorization.

## Features

- Automated data processing from CSV exports
- Configurable mappings and aliases via JSON mappings
- Supervised machine learning for transaction categorization
- Detailed analysis of income, expenses, and category breakdowns
- European format support (DD.MM.YYYY, comma decimal)

## Technology Stack

- Pandas (data processing)
- Scikit-learn (supervised ML)

## Project Structure

```
Dashboard/
├── src/
│   ├── processing.py            # ETL pipeline
│   ├── class_distribution.py    # Distribution of categories
│   ├── location_measures.py     # Statistics for dataset
│   ├── text_analysis.py         # Subject text analysis 
│   ├── categorization.py        # ML model training and prediction
├── config/
│   ├── category_mapping.json    # Transaction categories
│   └── alias_mapping.json       # Counterparty aliases
├── data/
│   ├── raw/                     # Input CSVs
│   └── processed/               # Output data
├── docs/
│   └── notes.md                 # Dev notes
└── requirements.txt             # Dependencies
```

## Setup

### Conda (Recommended)

```bash
conda create -n dashboard python=3.11
conda activate dashboard
conda install pip
pip install -r requirements.txt
pip install -e .
```

## Usage

Process transaction data:

```bash
python src/data_processing.py
```

Train categorization model:

```bash
python src/categorization.py --train
```

Predict categories:

```bash
python src/categorization.py --predict
```

## Data Processing Pipeline

1. Load: Read CSV with European formats
2. Extract: Identify counterparties using pattern matching
3. Normalize: Apply aliases (e.g., "AMZN" → "AMAZON")
4. Categorize: Map transactions to categories using supervised ML
5. Transform: Structure data for analysis
6. Export: Save processed data

## Configuration

- Transaction categories and aliases are configured via JSON files in `config/`
- Paths to I/O files and mappings are configured via `config.py` in `config/`
- ML model parameters and training data paths are configured in `categorization.py`

## Dependencies

See [`requirements.txt`](requirements.txt) for details.

## Notes

This project is for learning and experimentation. For setup and troubleshooting, see [`docs/notes.md`](docs/notes.md).

---

Happy coding!
