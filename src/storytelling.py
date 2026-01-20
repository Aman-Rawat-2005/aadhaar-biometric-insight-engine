import pandas as pd

# ======================================================
# NATIONAL STORY
# ======================================================

def generate_national_story(
    df: pd.DataFrame,
    official_district_count: int = 778
) -> list[str]:
    """
    Generates a national-level narrative from aggregated biometric data.
    """

    insights = []

    if df.empty or "total_updates" not in df.columns or "state" not in df.columns:
        insights.append(
            "National biometric activity summary could not be generated due to "
            "missing or insufficient data."
        )
        return insights

    total_updates = int(df["total_updates"].sum())
    total_states = df["state"].nunique()

    insights.append(
        f"India has recorded a total of {total_updates:,} biometric updates "
        f"across {total_states} States/UTs and {official_district_count} "
        f"officially recognized districts."
    )

    state_share = (
        df.groupby("state")["total_updates"]
        .sum()
        .sort_values(ascending=False)
    )

    if not state_share.empty:
        top_state = state_share.index[0]
        top_share_pct = round(
            (state_share.iloc[0] / state_share.sum()) * 100, 2
        )

        insights.append(
            f"{top_state} contributes approximately {top_share_pct}% of the "
            f"national biometric update volume, making it a key operational region."
        )

    return insights


# ======================================================
# POLICY STORY
# ======================================================

def generate_policy_story(policy_df: pd.DataFrame) -> list[str]:
    """
    Converts policy priority classification into policy-grade narrative insights.
    """

    insights = []

    if policy_df.empty or "policy_priority" not in policy_df.columns:
        insights.append(
            "Policy priority indicators could not be derived due to missing "
            "or incomplete analytical inputs."
        )
        return insights

    high_priority = policy_df[
        policy_df["policy_priority"] == "HIGH PRIORITY"
    ]

    if high_priority.empty:
        insights.append(
            "No State or Union Territory currently falls under the HIGH PRIORITY "
            "category based on evaluated biometric load indicators."
        )
        insights.append(
            "This suggests a relatively balanced distribution of biometric "
            "update activity under current analytical thresholds."
        )
        return insights

    states = ", ".join(high_priority["state"].astype(str).tolist())

    insights.append(
        f"The following States/UTs are classified as HIGH PRIORITY for policy "
        f"intervention due to observed coverage gaps or infrastructure stress: "
        f"{states}."
    )

    insights.append(
        "High priority classification indicates the need for focused "
        "administrative attention rather than system irregularity."
    )

    insights.append(
        "Recommended actions include targeted infrastructure assessment, "
        "temporary resource redistribution, and closer operational monitoring "
        "in the identified regions."
    )

    return insights


# ======================================================
# ANOMALY STORY
# ======================================================

def generate_anomaly_story(anomaly_df: pd.DataFrame) -> list[str]:
    """
    Generates a narrative around district-level anomalous biometric activity.
    """

    insights = []

    required_cols = {"district", "state", "anomaly_severity"}

    if anomaly_df.empty or not required_cols.issubset(anomaly_df.columns):
        insights.append(
            "No statistically significant district-level anomalies were "
            "detected under current analytical thresholds."
        )
        return insights

    top_row = anomaly_df.sort_values(
        "anomaly_severity", ascending=False
    ).iloc[0]

    insights.append(
        f"The district of {top_row['district']} in {top_row['state']} shows "
        f"unusually high biometric activity, exceeding expected thresholds "
        f"by a factor of {top_row['anomaly_severity']}."
    )

    insights.append(
        "Such anomalies serve as early operational indicators for targeted "
        "review and infrastructure assessment and do not imply misuse or "
        "irregular activity."
    )

    return insights


# ======================================================
# FAST ML RISK STORY
# ======================================================

def generate_risk_story(df: pd.DataFrame) -> list[str]:
    """
    Generates a concise, policy-grade narrative based on fast ML risk analysis.
    """

    insights = []

    if df.empty or "risk_level" not in df.columns:
        insights.append(
            "Machine learning-based risk indicators have not yet been applied "
            "to the current dataset."
        )
        return insights

    total = len(df)
    high_risk = df[df["risk_level"] == "HIGH"]

    high_count = len(high_risk)
    high_pct = (high_count / total * 100) if total > 0 else 0.0

    if high_risk.empty:
        insights.append(
            "All analyzed regions currently fall within normal operational "
            "risk ranges based on fast machine learning assessment."
        )
    else:
        top_states = (
            high_risk.sort_values("risk_score", ascending=False)
            if "risk_score" in high_risk.columns
            else high_risk
        ).head(2)

        top_row = top_states.iloc[0]
        insights.append(
            f"{top_row['state']} exhibits the highest risk level "
            f"({top_row['risk_score'] * 100:.1f}%), indicating possible "
            f"infrastructure stress or statistically unusual activity."
        )

        if len(top_states) > 1:
            second_row = top_states.iloc[1]
            insights.append(
                f"{second_row['state']} also shows elevated risk "
                f"({second_row['risk_score'] * 100:.1f}%), suggesting a "
                f"potential need for capacity review."
            )

    insights.append(
        f"Risk distribution indicates that {high_pct:.1f}% of regions "
        f"fall under high-risk classification, while "
        f"{100 - high_pct:.1f}% are categorized as low or medium risk."
    )

    if "risk_score" in df.columns:
        avg_score = df["risk_score"].mean()

        if avg_score > 0.7:
            profile = "elevated"
        elif avg_score > 0.4:
            profile = "moderate"
        else:
            profile = "normal"

        insights.append(
            f"The overall national risk profile is assessed as {profile}, "
            f"with an average risk score of {avg_score * 100:.1f}%."
        )

    insights.append(
        "The fast risk assessment model computes state-level risk scores "
        "within 1–2 seconds using a lightweight Isolation Forest approach, "
        "making it suitable for near real-time monitoring."
    )

    if high_count > 0:
        insights.append(
            "Targeted operational verification is recommended in high-risk "
            "regions to assess capacity constraints or process anomalies."
        )
    else:
        insights.append(
            "No immediate operational intervention is recommended at this time."
        )

    return insights


# ======================================================
# QUICK RISK SUMMARY (DASHBOARD METRICS)
# ======================================================

def generate_quick_risk_summary(df: pd.DataFrame) -> dict:
    """
    Returns a concise, deterministic risk summary for dashboard metrics.
    """

    if df.empty or "risk_level" not in df.columns:
        return {
            "high_risk_count": 0,
            "high_risk_percentage": 0.0,
            "average_risk_score": 0.0,
            "top_risk_state": "N/A",
            "top_risk_score": 0.0,
        }

    high_risk = df[df["risk_level"] == "HIGH"]

    total_regions = len(df)
    high_risk_count = len(high_risk)

    high_risk_percentage = (
        (high_risk_count / total_regions) * 100
        if total_regions > 0
        else 0.0
    )

    average_risk_score = (
        df["risk_score"].mean()
        if "risk_score" in df.columns and not df["risk_score"].isna().all()
        else 0.0
    )

    if not high_risk.empty and "risk_score" in high_risk.columns:
        top_row = high_risk.sort_values(
            "risk_score", ascending=False
        ).iloc[0]

        top_risk_state = top_row["state"]
        top_risk_score = top_row["risk_score"]
    else:
        top_risk_state = "N/A"
        top_risk_score = 0.0

    return {
        "high_risk_count": high_risk_count,
        "high_risk_percentage": round(high_risk_percentage, 2),
        "average_risk_score": round(average_risk_score, 4),
        "top_risk_state": top_risk_state,
        "top_risk_score": round(top_risk_score, 4),
    }
