import pandas as pd
from pathlib import Path

# Always resolve project root safely
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def load_biometric_data():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Accept xls, xlsx, csv (safe)
    files = list(DATA_DIR.glob("*.xlsx")) + \
            list(DATA_DIR.glob("*.xls")) + \
            list(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Aadhaar biometric files found in {DATA_DIR}"
        )

    df_list = []
    for file in files:
        if file.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df["source_file"] = file.name
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
