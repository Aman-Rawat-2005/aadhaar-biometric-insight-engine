"""
Aadhaar Biometric Risk Model - Enhanced for Better Visualization
Creates meaningful risk scores (0-1) with proper distribution
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced ML-based operational risk scoring with better distribution
    """
    
    # Create safe copy
    data = df.copy()
    
    # ============================================
    # ENHANCED FEATURE ENGINEERING
    # ============================================
    
    # Ensure we have state-level aggregation if multiple rows exist
    if data['state'].nunique() < len(data):
        # Aggregate to state level for risk scoring
        state_agg = data.groupby('state', as_index=False).agg({
            'total_updates': 'sum'
        })
        data = state_agg.copy()
    
    # 1. Multiple meaningful features
    # Calculate relative metrics
    total_national = data['total_updates'].sum()
    data['pct_of_national'] = (data['total_updates'] / total_national * 100) if total_national > 0 else 0
    
    # 2. Statistical features
    data['log_updates'] = np.log1p(data['total_updates'])
    
    # 3. Deviation from average (z-score)
    avg_updates = data['total_updates'].mean()
    std_updates = data['total_updates'].std()
    data['z_score'] = (data['total_updates'] - avg_updates) / std_updates if std_updates > 0 else 0
    
    # 4. Percentile ranking
    data['percentile'] = data['total_updates'].rank(pct=True)
    
    # 5. Volume-to-rank ratio (high volume + low rank = anomaly)
    data['volume_rank_ratio'] = data['total_updates'] / data['total_updates'].rank()
    
    # ============================================
    # FEATURE SELECTION & SCALING
    # ============================================
    feature_cols = ['log_updates', 'z_score', 'percentile', 'volume_rank_ratio', 'pct_of_national']
    features = data[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # ============================================
    # ENHANCED RISK MODELING
    # ============================================
    
    # Method 1: Isolation Forest with dynamic contamination
    n_samples = len(features_scaled)
    contamination = min(0.2, max(0.05, 10 / n_samples))  # Dynamic: 5-20%
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples='auto'
    )
    
    iso_forest.fit(features_scaled)
    iso_scores = -iso_forest.decision_function(features_scaled)
    
    # Method 2: Z-score based risk
    z_risk = np.abs(data['z_score'].fillna(0))
    
    # Method 3: Percentile inversion (low percentile = high risk for fraud detection)
    pct_risk = 1 - data['percentile']
    
    # ============================================
    # COMBINE MULTIPLE RISK INDICATORS
    # ============================================
    
    # Normalize each risk component to 0-1
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min()) if iso_scores.max() > iso_scores.min() else iso_scores
    z_norm = (z_risk - z_risk.min()) / (z_risk.max() - z_risk.min()) if z_risk.max() > z_risk.min() else z_risk
    pct_norm = (pct_risk - pct_risk.min()) / (pct_risk.max() - pct_risk.min()) if pct_risk.max() > pct_risk.min() else pct_risk
    
    # Weighted combination (tune weights based on importance)
    data['risk_raw'] = (
        0.4 * iso_norm +    # Isolation Forest anomaly detection
        0.3 * z_norm +      # Statistical outliers
        0.3 * pct_norm      # Rank-based risk
    )
    
    # ============================================
    # PROPER RISK SCORE NORMALIZATION (0-1)
    # ============================================
    
    # Apply softmax-like transformation for better distribution
    risk_min = data['risk_raw'].min()
    risk_max = data['risk_raw'].max()
    
    if risk_max > risk_min:
        # Linear normalization
        data['risk_score'] = (data['risk_raw'] - risk_min) / (risk_max - risk_min)
        
        # Apply sigmoid to spread out middle values
        data['risk_score'] = 1 / (1 + np.exp(-5 * (data['risk_score'] - 0.5)))
        
        # Re-normalize to 0-1
        data['risk_score'] = (data['risk_score'] - data['risk_score'].min()) / \
                            (data['risk_score'].max() - data['risk_score'].min())
    else:
        data['risk_score'] = 0.5  # Default neutral score
    
    # ============================================
    # ENHANCED RISK CATEGORIZATION
    # ============================================
    
    # Use statistical quartiles for better distribution
    q1 = data['risk_score'].quantile(0.25)
    q2 = data['risk_score'].quantile(0.50)  # Median
    q3 = data['risk_score'].quantile(0.75)
    
    def categorize_risk(score):
        if score >= q3:  # Top 25% = HIGH
            return "HIGH"
        elif score >= q2:  # 25-50% = MEDIUM
            return "MEDIUM"
        else:  # Bottom 50% = LOW
            return "LOW"
    
    data['risk_level'] = data['risk_score'].apply(categorize_risk)
    
    # Add color codes for visualization
    color_map = {
        'HIGH': '#EF4444',    # Red
        'MEDIUM': '#F59E0B',  # Orange
        'LOW': '#10B981'      # Green
    }
    data['risk_color'] = data['risk_level'].map(color_map)
    
    # Confidence score (0-100)
    data['risk_confidence'] = (data['risk_score'] * 100).round(1)
    
    # Add descriptive labels
    data['risk_description'] = data.apply(
        lambda row: f"{row['state']}: Risk Score {row['risk_score']:.3f} ({row['risk_level']})",
        axis=1
    )
    
    # Sort by risk score for better display
    data = data.sort_values('risk_score', ascending=False)
    
    # Reset index
    data = data.reset_index(drop=True)
    
    return data


def get_high_risk_regions(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Enhanced high-risk region extraction with visualization support
    """
    if "risk_level" not in df.columns:
        df = compute_risk_scores(df)
    
    # Get high risk states
    high_risk = df[df["risk_level"] == "HIGH"].copy()
    
    if high_risk.empty:
        return pd.DataFrame()
    
    # Sort and limit
    high_risk = high_risk.sort_values(["risk_score", "total_updates"], ascending=[False, False]).head(top_n)
    
    # Add visualization-friendly columns
    high_risk['size'] = np.log1p(high_risk['total_updates'])  # For bubble/treemap sizing
    high_risk['opacity'] = high_risk['risk_score']  # For transparency based on risk
    
    # Format for display
    display_cols = ["state", "total_updates", "risk_score", "risk_level", "risk_color", "size", "opacity"]
    available_cols = [col for col in display_cols if col in high_risk.columns]
    
    result = high_risk[available_cols].copy()
    result['risk_score'] = result['risk_score'].round(3)
    result['total_updates'] = result['total_updates'].astype(int)
    
    return result.reset_index(drop=True)


def get_risk_summary(df: pd.DataFrame) -> dict:
    """
    Fast risk summary statistics
    """
    if "risk_level" not in df.columns:
        df = compute_risk_scores(df)
    
    # Calculate safely
    high_count = (df["risk_level"] == "HIGH").sum()
    total_count = len(df)
    
    summary = {
        "total_records": total_count,
        "high_risk_count": int(high_count),
        "medium_risk_count": int((df["risk_level"] == "MEDIUM").sum()),
        "low_risk_count": int((df["risk_level"] == "LOW").sum()),
        "high_risk_percentage": round(float(high_count) / total_count * 100, 2) if total_count > 0 else 0.0,
        "average_risk_score": round(float(df["risk_score"].mean()), 3),
        "median_risk_score": round(float(df["risk_score"].median()), 3),  # FIXED!
        "max_risk_score": round(float(df["risk_score"].max()), 3),
        "min_risk_score": round(float(df["risk_score"].min()), 3),
        "risk_std": round(float(df["risk_score"].std()), 3) if total_count > 1 else 0.0
    }
    
    return summary


def get_risk_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data specifically for visualization
    Returns data with proper scaling and coloring
    """
    if "risk_score" not in df.columns:
        df = compute_risk_scores(df)
    
    viz_df = df.copy()
    
    # Scale values for visualization
    viz_df['viz_size'] = np.interp(
        np.log1p(viz_df['total_updates']),
        (np.log1p(viz_df['total_updates'].min()), np.log1p(viz_df['total_updates'].max())),
        (10, 100)
    )
    
    # Create gradient color based on risk score
    def get_gradient_color(score):
        # Red (high) -> Orange (medium) -> Green (low) gradient
        if score >= 0.7:
            return '#EF4444'  # Red
        elif score >= 0.4:
            return '#F59E0B'  # Orange
        else:
            return '#10B981'  # Green
    
    viz_df['gradient_color'] = viz_df['risk_score'].apply(get_gradient_color)
    
    # Add hover text
    viz_df['hover_text'] = viz_df.apply(
        lambda row: f"<b>{row['state']}</b><br>"
                   f"Risk: {row['risk_score']:.3f} ({row['risk_level']})<br>"
                   f"Updates: {row['total_updates']:,}<br>"
                   f"Confidence: {row.get('risk_confidence', 0)}%",
        axis=1
    )
    
    return viz_df