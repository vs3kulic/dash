# Bank Transaction Dashboard

A Python-based dashboard for processing and visualizing bank transaction data. Built with Streamlit and Pandas.

## Features

- Automated data processing from CSV exports
- Configurable mappings and aliases via JSON mappings
- Interactive charts for income, expenses, and category breakdowns
- European format support (DD.MM.YYYY, comma decimal)

## Technology Stack

- Streamlit (dashboard UI)
- Pandas (data processing)

## Project Structure

```
Dashboard/
├── src/
│   ├── data_processing.py    # ETL pipeline
│   └── dashboard.py         # Streamlit UI
├── config/
│   ├── category_mapping.json # Transaction categories
│   └── alias_mapping.json    # Counterparty aliases
├── data/
│   ├── raw/                 # Input CSVs
│   └── processed/           # Output data
├── docs/
│   └── notes.md             # Dev notes
├── pyproject.toml           # Project config
└── requirements.txt         # Dependencies
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

Launch dashboard:

```bash
streamlit run src/dashboard.py
```

## Data Processing Pipeline

1. Load: Read CSV with European formats
2. Extract: Identify counterparties using pattern matching
3. Normalize: Apply aliases (e.g., "AMZN" → "AMAZON")
4. Categorize: Map transactions to categories
5. Transform: Structure data for analysis
6. Export: Save processed data


## Configuration

Transaction categories and aliases are configured via JSON files in `config/`. 

## Dependencies

See [`requirements.txt`](requirements.txt) for details.

## Notes

This project is for learning and experimentation. For setup and troubleshooting, see [`docs/notes.md`](docs/notes.md).

---

Happy coding!
