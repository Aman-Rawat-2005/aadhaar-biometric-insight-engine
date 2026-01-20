import pandas as pd

# ======================================================
# STATE LEVEL ANALYTICS
# ======================================================

def state_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns state-wise aggregated intelligence metrics
    """
    state_df = (
        df.groupby("state", as_index=False)
        .agg(
            total_updates=("total_updates", "sum"),
            total_districts=("district", "nunique"),
            avg_updates_per_district=("total_updates", "mean")
        )
        .sort_values("total_updates", ascending=False)
    )

    # National share (% contribution)
    national_total = state_df["total_updates"].sum()
    state_df["national_share_pct"] = (
        state_df["total_updates"] / national_total * 100
    ).round(2)

    return state_df


# ======================================================
# DISTRICT LEVEL ANALYTICS (STATE-SPECIFIC)
# ======================================================

def district_level_metrics(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Returns district-wise metrics for a given state
    """
    district_df = (
        df[df["state"] == state]
        .groupby("district", as_index=False)
        .agg(
            total_updates=("total_updates", "sum")
        )
        .sort_values("total_updates", ascending=False)
    )

    return district_df


# ======================================================
# DISTRICT INEQUALITY ANALYSIS (IMPORTANT)
# ======================================================

def district_inequality_score(df: pd.DataFrame, state: str) -> dict:
    """
    Measures how uneven biometric activity is within a state
    """
    district_df = district_level_metrics(df, state)

    if district_df.empty or len(district_df) == 1:
        return {
            "state": state,
            "inequality_ratio": 0,
            "top_district": None,
            "bottom_district": None
        }

    top_value = district_df.iloc[0]["total_updates"]
    bottom_value = district_df.iloc[-1]["total_updates"]

    ratio = round(top_value / max(bottom_value, 1), 2)

    return {
        "state": state,
        "inequality_ratio": ratio,
        "top_district": district_df.iloc[0]["district"],
        "bottom_district": district_df.iloc[-1]["district"]
    }


# ======================================================
# NATIONAL LEVEL KPIs
# ======================================================

def national_kpis(df: pd.DataFrame) -> dict:
    """
    High-level KPIs for India-wide biometric behavior
    """
    return {
        "national_total_updates": int(df["total_updates"].sum()),
        "total_states": df["state"].nunique(),
        "total_districts": df["district"].nunique(),
        "avg_updates_per_district": round(df["total_updates"].mean(), 2),
        "median_updates_per_district": round(df["total_updates"].median(), 2)
    }
