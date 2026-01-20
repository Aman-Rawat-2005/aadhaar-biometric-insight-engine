def calculate_brs(df):
    df["BRS"] = (
        0.4 * (1 - df["growth_rate"].abs()) +
        0.3 * (1 - df["child_ratio"]) +
        0.3 * (1 - df["adolescent_ratio"])
    )
    return df
