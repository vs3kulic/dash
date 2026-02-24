# Bank Transaction Categorization

A Python project for processing, analyzing, and automatically categorizing bank transactions using supervised machine learning.

## Goal

This project explores if we can automatically categorize transactions based on subject text and amount, using ~250 labeled transactions across 13 categories.

## Project Structure

```
Dash/
├── src/
│   ├── processing.py            # ETL pipeline: load, clean, extract, categorize
│   └── analysis.py              # EDA: location measures, class distribution, correlation, text patterns
├── README.md
└── requirements.txt
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

