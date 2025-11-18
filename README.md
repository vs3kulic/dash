# Bank Transaction Dashboard

A Python-based dashboard app for processing and visualizing bank transaction data. Built with Streamlit, Pandas, and Plotly as a learning project for data engineering and visualization fundamentals.

## Overview

This application ingests CSV exports from bank accounts, cleans and transforms the data, and provides an interactive web interface for analyzing transaction patterns.

**Features:**
- CSV data ingestion and parsing
- Data cleaning (handles German date/number formats)
- Processed data storage
- Transaction categorization (in progress)
- Time-series visualization

## Technology Stack

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualization

## Setup

**Prerequisites:** Python 3.8+

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run application:
   ```bash
   streamlit run src/dashboard.py
   ```

4. Access dashboard at `http://localhost:8501`

## Implementation Notes

- Uses `pathlib` for cross-platform file path handling
- Implements data cleaning pipeline separate from visualization layer
- Handles European numeric formats (comma decimal separator, period thousands separator)
- Date parsing configured for DD.MM.YYYY format
