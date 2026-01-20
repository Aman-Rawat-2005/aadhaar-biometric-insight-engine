from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

# Date column
DATE_COL = "date"

# Geography columns
STATE_COL = "state"
DISTRICT_COL = "district"
PINCODE_COL = "pincode"

# Age-based biometric columns (adjust if more exist)
AGE_COLUMNS = [
    "bio_age_5_",
    "bio_age_17_"
]

# Time aggregation
TIME_FREQ = "M"  # Monthly

# Random seed
RANDOM_STATE = 42
