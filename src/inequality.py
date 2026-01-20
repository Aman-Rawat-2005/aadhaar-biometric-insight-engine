import pandas as pd
import numpy as np
from scipy import stats

# ======================================================
# ENHANCED STATE-LEVEL INEQUALITY INDEX
# ======================================================
def state_inequality_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive inequality metrics per state with:
    - Multiple inequality indices (Gini, Theil, Atkinson)
    - Statistical measures (CV, Skewness, Kurtosis)
    - Visual-friendly categories
    """
    
    # -----------------------------
    # District-level aggregation
    # -----------------------------
    if df['district'].isnull().all():
        # If no district data, create synthetic district-level data
        district_agg = df.copy()
        if 'district' not in district_agg.columns:
            district_agg['district'] = 'District_' + (district_agg.groupby('state').cumcount() + 1).astype(str)
    else:
        district_agg = (
            df.groupby(["state", "district"], as_index=False)["total_updates"]
            .sum()
        )
    
    results = []
    
    # -----------------------------
    # Helper function: Gini Coefficient
    # -----------------------------
    def gini_coefficient(arr):
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
        arr = np.array(arr)
        if len(arr) == 0 or np.sum(arr) == 0:
            return 0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr))
    
    # -----------------------------
    # Helper function: Theil Index
    # -----------------------------
    def theil_index(arr):
        """Calculate Theil index (0 = perfect equality)"""
        arr = np.array(arr)
        if len(arr) == 0 or np.mean(arr) == 0:
            return 0
        normalized = arr / np.mean(arr)
        return np.mean(normalized * np.log(normalized))
    
    # -----------------------------
    # Helper function: Atkinson Index
    # -----------------------------
    def atkinson_index(arr, epsilon=0.5):
        """Calculate Atkinson index (0 = perfect equality, 1 = perfect inequality)"""
        arr = np.array(arr)
        if len(arr) == 0 or np.mean(arr) == 0:
            return 0
        if epsilon == 1:
            # Special case for epsilon=1
            geometric_mean = np.exp(np.mean(np.log(arr[arr > 0])))
            return 1 - geometric_mean / np.mean(arr)
        else:
            mean_val = np.mean(arr)
            if epsilon > 0:
                return 1 - (np.mean(arr**(1-epsilon))**(1/(1-epsilon))) / mean_val
            else:
                return 0
    
    # -----------------------------
    # Per-state comprehensive computation
    # -----------------------------
    for state, g in district_agg.groupby("state"):
        g = g.sort_values("total_updates", ascending=False)
        
        # Extract values array for calculations
        values = g["total_updates"].values
        
        # Basic metrics
        total_state_updates = np.sum(values)
        num_districts = len(values)
        avg_district_updates = np.mean(values) if len(values) > 0 else 0
        max_district_updates = np.max(values) if len(values) > 0 else 0
        min_district_updates = np.min(values) if len(values) > 0 else 0
        
        # ---- Multiple Inequality Indices ----
        # 1. Original ratio index
        inequality_index = round(
            max_district_updates / avg_district_updates, 2
        ) if avg_district_updates > 0 else 0
        
        # 2. Gini coefficient (more robust)
        gini = gini_coefficient(values)
        
        # 3. Theil index
        theil = theil_index(values)
        
        # 4. Atkinson index
        atkinson = atkinson_index(values)
        
        # ---- Top concentration metrics ----
        # Top 20% districts share
        top_n_20 = max(1, int(len(g) * 0.2))
        top_20_share = round(
            (np.sum(values[:top_n_20]) / total_state_updates) * 100, 2
        ) if total_state_updates > 0 else 0
        
        # Top 10% districts share
        top_n_10 = max(1, int(len(g) * 0.1))
        top_10_share = round(
            (np.sum(values[:top_n_10]) / total_state_updates) * 100, 2
        ) if total_state_updates > 0 else 0
        
        # ---- Statistical measures ----
        # Coefficient of Variation
        cv = round(np.std(values) / avg_district_updates * 100, 2) if avg_district_updates > 0 else 0
        
        # Skewness and Kurtosis (if enough data)
        if len(values) > 2 and np.std(values) > 0:
            skewness = round(stats.skew(values), 3)
            kurtosis = round(stats.kurtosis(values), 3)
        else:
            skewness = 0
            kurtosis = 0
        
        # ---- Dominance ratio (max/min) ----
        dominance_ratio = round(max_district_updates / min_district_updates, 2) if min_district_updates > 0 else 0
        
        # ---- Categorization for visualization ----
        # Inequality category based on Gini
        if gini < 0.3:
            inequality_category = "LOW"
            inequality_color = "#10B981"  # Green
        elif gini < 0.6:
            inequality_category = "MODERATE"
            inequality_color = "#F59E0B"  # Orange
        else:
            inequality_category = "HIGH"
            inequality_color = "#EF4444"  # Red
        
        # Concentration category
        if top_20_share < 50:
            concentration_category = "BALANCED"
        elif top_20_share < 70:
            concentration_category = "MODERATE"
        else:
            concentration_category = "CONCENTRATED"
        
        results.append({
            # Basic identifiers
            "state": state,
            "num_districts": num_districts,
            "total_updates": total_state_updates,
            
            # Distribution statistics
            "avg_district_updates": round(avg_district_updates, 2),
            "max_district_updates": max_district_updates,
            "min_district_updates": min_district_updates,
            
            # Inequality indices (multiple measures)
            "inequality_index": inequality_index,
            "gini_coefficient": round(gini, 3),
            "theil_index": round(theil, 3),
            "atkinson_index": round(atkinson, 3),
            
            # Concentration metrics
            "top_10pct_share_%": top_10_share,
            "top_20pct_share_%": top_20_share,
            "dominance_ratio": dominance_ratio,
            
            # Statistical measures
            "coefficient_of_variation_%": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            
            # Visualization categories
            "inequality_category": inequality_category,
            "inequality_color": inequality_color,
            "concentration_category": concentration_category,
            
            # Additional useful metrics
            "median_district_updates": round(np.median(values), 2) if len(values) > 0 else 0,
            "std_district_updates": round(np.std(values), 2) if len(values) > 0 else 0,
            "iqr_updates": round(
                np.percentile(values, 75) - np.percentile(values, 25), 2
            ) if len(values) >= 4 else 0
        })
    
    # -----------------------------
    # Create DataFrame and sort
    # -----------------------------
    state_ineq_df = pd.DataFrame(results)
    
    # Add percentile ranks
    state_ineq_df["inequality_rank"] = state_ineq_df["gini_coefficient"].rank(
        method="dense", ascending=False
    ).astype(int)
    
    state_ineq_df["concentration_rank"] = state_ineq_df["top_20pct_share_%"].rank(
        method="dense", ascending=False
    ).astype(int)
    
    # -----------------------------
    # Final sorting and formatting
    # -----------------------------
    state_ineq_df = state_ineq_df.sort_values(
        "gini_coefficient",
        ascending=False
    )
    
    # Reset index for clean output
    state_ineq_df = state_ineq_df.reset_index(drop=True)
    
    return state_ineq_df


# ======================================================
# ADDITIONAL HELPER FUNCTION: Get inequality insights
# ======================================================
def get_inequality_insights(ineq_df: pd.DataFrame) -> dict:
    """
    Generate actionable insights from inequality analysis
    """
    insights = {
        "most_unequal_state": ineq_df.iloc[0]["state"] if not ineq_df.empty else "N/A",
        "most_equal_state": ineq_df.iloc[-1]["state"] if not ineq_df.empty else "N/A",
        "avg_gini": round(ineq_df["gini_coefficient"].mean(), 3),
        "high_inequality_count": (ineq_df["inequality_category"] == "HIGH").sum(),
        "top_concentrated_state": ineq_df.loc[ineq_df["top_20pct_share_%"].idxmax()]["state"] if not ineq_df.empty else "N/A",
        
        # Recommendations based on thresholds
        "recommendations": []
    }
    
    # Generate recommendations
    for _, row in ineq_df.head(5).iterrows():
        if row["gini_coefficient"] > 0.6:
            insights["recommendations"].append(
                f"{row['state']}: High inequality (Gini={row['gini_coefficient']}) - Consider resource redistribution"
            )
        elif row["top_20pct_share_%"] > 70:
            insights["recommendations"].append(
                f"{row['state']}: Top 20% districts control {row['top_20pct_share_%']}% of updates - Focus on underserved districts"
            )
    
    return insights