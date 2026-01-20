import pandas as pd
import numpy as np

# ======================================================
# ENHANCED DISTRICT-LEVEL ANOMALY DETECTION
# ======================================================

def detect_district_anomalies(df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
    """
    Enhanced anomaly detection with multiple methods and better metrics.
    """
    
    # Aggregate at district level
    district_df = (
        df.groupby(["state", "district"], as_index=False)
        .agg(total_updates=("total_updates", "sum"))
    )
    
    # Add derived metrics
    district_df['log_updates'] = np.log1p(district_df['total_updates'])
    district_df['updates_percentile'] = district_df['total_updates'].rank(pct=True)
    
    if method == "iqr":
        # Robust IQR method (default)
        q1 = district_df["total_updates"].quantile(0.25)
        q3 = district_df["total_updates"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        anomaly_threshold = upper_bound
    
    elif method == "zscore":
        # Z-score method for normally distributed data
        mean_val = district_df["total_updates"].mean()
        std_val = district_df["total_updates"].std()
        anomaly_threshold = mean_val + 3 * std_val
    
    elif method == "percentile":
        # Percentile-based (top 5%)
        anomaly_threshold = district_df["total_updates"].quantile(0.95)
    
    # Flag anomalies
    district_df["anomaly_flag"] = district_df["total_updates"] > anomaly_threshold
    district_df["anomaly_severity"] = (
        district_df["total_updates"] / anomaly_threshold
    ).round(2)
    
    # Add severity category
    def categorize_severity(severity):
        if severity >= 3.0:
            return "CRITICAL"
        elif severity >= 2.0:
            return "HIGH"
        elif severity >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    district_df["severity_category"] = district_df["anomaly_severity"].apply(categorize_severity)
    
    # Add anomaly type based on characteristics
    district_df["anomaly_type"] = district_df.apply(
        lambda row: "VOLUME_SPIKE" if row["total_updates"] > anomaly_threshold * 2 
        else "SUSPICIOUS_ACTIVITY",
        axis=1
    )
    
    # Keep only suspicious records
    anomalies = district_df[district_df["anomaly_flag"]].copy()
    
    # Add state-level context
    if not anomalies.empty:
        state_totals = df.groupby("state")["total_updates"].sum().to_dict()
        anomalies["state_total_updates"] = anomalies["state"].map(state_totals)
        anomalies["pct_of_state"] = (anomalies["total_updates"] / anomalies["state_total_updates"] * 100).round(2)
    
    return (
        anomalies
        .sort_values(["anomaly_severity", "total_updates"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ======================================================
# ENHANCED STATE-LEVEL ANOMALY SUMMARY
# ======================================================

def anomaly_summary_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates comprehensive state-level anomaly summary for heatmap visualization.
    """
    
    anomalies = detect_district_anomalies(df)
    
    if anomalies.empty:
        # Create empty dataframe with proper columns
        return pd.DataFrame(columns=[
            "state", "anomalous_districts", "max_severity", 
            "avg_severity", "total_anomaly_updates", "severity_score"
        ])
    
    # Create comprehensive summary
    summary = (
        anomalies
        .groupby("state", as_index=False)
        .agg(
            anomalous_districts=("district", "nunique"),
            max_severity=("anomaly_severity", "max"),
            avg_severity=("anomaly_severity", "mean"),
            total_anomaly_updates=("total_updates", "sum"),
            critical_count=("severity_category", lambda x: (x == "CRITICAL").sum()),
            high_count=("severity_category", lambda x: (x == "HIGH").sum()),
        )
        .round(2)
    )
    
    # Calculate derived metrics
    state_totals = df.groupby("state")["total_updates"].sum().reset_index()
    state_totals.columns = ["state", "state_total_updates"]
    
    summary = summary.merge(state_totals, on="state", how="left")
    summary["pct_state_anomaly"] = (
        summary["total_anomaly_updates"] / summary["state_total_updates"] * 100
    ).round(2)
    
    # Calculate composite severity score (0-100)
    summary["severity_score"] = (
        summary["max_severity"] * 30 +
        summary["avg_severity"] * 20 +
        (summary["critical_count"] * 5) +
        (summary["high_count"] * 3) +
        summary["pct_state_anomaly"]
    ).round(2)
    
    # Add risk level
    def get_risk_level(score):
        if score >= 60:
            return "HIGH RISK"
        elif score >= 30:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    summary["risk_level"] = summary["severity_score"].apply(get_risk_level)
    
    # Sort by severity
    summary = summary.sort_values(["severity_score", "anomalous_districts"], ascending=[False, False])
    
    return summary.reset_index(drop=True)


# ======================================================
# HEATMAP DATA PREPARATION FUNCTION
# ======================================================

def prepare_anomaly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data specifically for anomaly heatmap visualization.
    Returns a pivot table ready for Plotly heatmap.
    """
    
    summary = anomaly_summary_by_state(df)
    
    if summary.empty:
        return pd.DataFrame()
    
    # Get top states for heatmap
    top_states = summary.head(10)["state"].tolist()
    
    # Prepare metrics for heatmap
    metrics_data = []
    
    for state in top_states:
        state_data = summary[summary["state"] == state].iloc[0]
        
        metrics_data.append({
            "state": state,
            "anomalous_districts": state_data["anomalous_districts"],
            "max_severity": state_data["max_severity"],
            "avg_severity": state_data.get("avg_severity", 0),
            "severity_score": state_data["severity_score"],
            "pct_state_anomaly": state_data.get("pct_state_anomaly", 0),
            "critical_count": state_data.get("critical_count", 0),
            "high_count": state_data.get("high_count", 0)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create pivot table for heatmap
    heatmap_df = metrics_df.set_index("state")
    
    # Select and scale metrics for heatmap
    heatmap_metrics = ["anomalous_districts", "max_severity", "avg_severity", "severity_score"]
    heatmap_df = heatmap_df[heatmap_metrics]
    
    # Normalize each column for better heatmap visualization
    for col in heatmap_df.columns:
        min_val = heatmap_df[col].min()
        max_val = heatmap_df[col].max()
        if max_val > min_val:
            heatmap_df[f"{col}_scaled"] = (heatmap_df[col] - min_val) / (max_val - min_val)
        else:
            heatmap_df[f"{col}_scaled"] = 0.5
    
    return heatmap_df