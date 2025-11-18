# Python Path Navigation with pathlib

## Why pathlib.Path > old os.path

1. **Readable:** `Path(__file__).parent / "data" / "file.csv"` vs `os.path.join(os.path.dirname(__file__), "data", "file.csv")`
2. **Cross-platform:** `/` works on Windows, Mac, Linux automatically
3. **Object-oriented:** You can chain methods like `.exists()`, `.read_text()`, `.stem`, etc.
4. **No string manipulation:** It's an actual Path object, not just strings

## The Rule (Learn This Once, Use Forever)

```python
from pathlib import Path

# Path to THIS file
Path(__file__)

# Directory containing THIS file
Path(__file__).parent

# Go up one directory
Path(__file__).parent.parent

# Build paths with /
Path(__file__).parent / "data" / "file.csv"
```

**Why it works:** `__file__` is the absolute path to the current Python file, so you always know where you are. No guessing about "current directory".

## Useful Path Tricks

```python
from pathlib import Path

# Example: working with input file
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT/"data"/"raw"/
input_file = RAW_DATA/"Umsatzliste_AT943200000014664403.csv"

# Check if file exists
if input_file.exists():
    print("File found!")

# Get filename without extension
input_file.stem  # "Umsatzliste_AT943200000014664403"

# Get just the filename
input_file.name  # "Umsatzliste_AT943200000014664403.csv"

# Get file extension
input_file.suffix  # ".csv"

# Create directories if they don't exist
output_dir = Path(__file__).parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Read file content directly
text_content = input_file.read_text()

# Write to file
output_file = output_dir / "cleaned_data.csv"
output_file.write_text("data here")
```

## Common Pattern for Data Projects

```python
from pathlib import Path

# Define project structure relative to current file
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Use them
input_file = RAW_DATA_DIR / "transactions.csv"
output_file = PROCESSED_DATA_DIR / "cleaned_transactions.csv"
```

