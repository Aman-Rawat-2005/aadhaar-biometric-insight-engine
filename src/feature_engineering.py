import pandas as pd
from config import AGE_COLUMNS

def create_features(df):
    df = df.copy()

    df["total_bio_updates"] = df[AGE_COLUMNS].sum(axis=1)

    df["child_ratio"] = df["bio_age_5_"] / (df["total_bio_updates"] + 1)
    df["adolescent_ratio"] = df["bio_age_17_"] / (df["total_bio_updates"] + 1)

    df["growth_rate"] = (
        df.groupby(["state", "district"])["total_bio_updates"]
        .pct_change()
        .fillna(0)
    )

    return df
