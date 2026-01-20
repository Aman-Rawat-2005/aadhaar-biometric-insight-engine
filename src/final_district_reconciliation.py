import pandas as pd

def reconcile_active_districts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps ONLY districts which are:
    - Present in Aadhaar data
    - Have real biometric activity
    """

    df = df.copy()

    # Rule 1: Must have activity
    df = df[df["total_updates"] > 0]

    # Rule 2: Collapse duplicates AFTER LGD cleaning
    df = (
        df.groupby(["state", "district"], as_index=False)
        .agg(
            total_updates=("total_updates", "sum"),
            bio_age_5=("bio_age_5", "sum"),
            bio_age_17=("bio_age_17", "sum"),
            date=("date", "max"),
        )
    )

    return df
