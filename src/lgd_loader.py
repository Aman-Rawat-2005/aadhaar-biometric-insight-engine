# src/lgd_loader.py

import pandas as pd

def load_lgd_district_master(file_path: str) -> pd.DataFrame:
    """
    Loads LGD district master CSV (FAST + SAFE)
    - Reads only required columns
    - No data loss
    - No logic change
    """

    # ------------------------------
    # READ ONLY REQUIRED COLUMNS
    # ------------------------------
    usecols = [
        "State Name",
        "District Name (In English)"
    ]

    df = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype="string",          # faster + memory efficient
        engine="c",              # fastest CSV engine
        low_memory=False
    )

    # ------------------------------
    # STANDARDIZE COLUMN NAMES
    # ------------------------------
    df = df.rename(columns={
        "State Name": "state",
        "District Name (In English)": "district"
    })

    # ------------------------------
    # NORMALIZATION (VECTOR, ONCE)
    # ------------------------------
    df["state"] = (
        df["state"]
        .str.strip()
        .str.title()
    )

    df["district"] = (
        df["district"]
        .str.strip()
        .str.title()
    )

    return df
