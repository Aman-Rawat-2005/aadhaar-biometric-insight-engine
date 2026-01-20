import pandas as pd
import numpy as np
import re

# =======================================================
# 1. CONSTANTS & MAPPING
# =======================================================
STATE_UT_CANONICAL_MAP = {
    # STATES
    "andhrapradesh": "Andhra Pradesh",
    "arunachalpradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "chatisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachalpradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhyapradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamilnadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttarpradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "westbengal": "West Bengal",

    # UNION TERRITORIES
    "andamanandnicobarislands": "Andaman and Nicobar Islands",
    "chandigarh": "Chandigarh",
    "dadraandnagarhaveli": "Dadra and Nagar Haveli and Daman and Diu",
    "damananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadraandnagarhavelianddamananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "newdelhi": "Delhi",
    "jammuandkashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
}

# =======================================================
# 2. PREPROCESSING FUNCTION (UPDATED)
# =======================================================
def preprocess_biometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    PERFORMANCE-OPTIMIZED PREPROCESSING
    Includes logic for Bio-Updates, Demographic Updates, and Aadhaar Generated.
    """
    df = df.copy()

    # --- STEP 1: Column Standardization ---
    df.columns = df.columns.str.strip().str.lower()

    # --- STEP 2: Handle Missing/Renamed Columns (Robustness) ---
    # 2a. Biometric Age Specifics
    if "bio_age_5" not in df.columns and "bio_age_5_17" in df.columns:
        df["bio_age_5"] = df["bio_age_5_17"]
    if "bio_age_17" not in df.columns and "bio_age_17_" in df.columns:
        df["bio_age_17"] = df["bio_age_17_"]

    # 2b. Aadhaar Generated (New Logic)
    if "aadhaar_generated" not in df.columns:
        # Check for common aliases
        if "generated" in df.columns:
            df["aadhaar_generated"] = df["generated"]
        else:
            df["aadhaar_generated"] = 0 # Default to 0 if missing

    # 2c. Demographic Updates (New Logic)
    if "demographic_updates" not in df.columns:
        df["demographic_updates"] = 0

    # --- STEP 3: Convert Numeric Columns Safely ---
    cols_to_numeric = ["bio_age_5", "bio_age_17", "aadhaar_generated", "demographic_updates"]
    
    for col in cols_to_numeric:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- STEP 4: Calculate Totals ---
    # Mandatory Biometric Updates (MBU)
    df["total_biometric_updates"] = df["bio_age_5"] + df["bio_age_17"]
    
    # Total Updates (Biometric + Demographic)
    df["total_updates"] = df["total_biometric_updates"] + df["demographic_updates"]
    
    # Total Activity (Includes New Generations)
    df["total_activity"] = df["total_updates"] + df["aadhaar_generated"]

    # --- STEP 5: State Canonicalization ---
    if "state" in df.columns:
        state_series = (
            df["state"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True)
        )
        df["state"] = state_series.map(STATE_UT_CANONICAL_MAP)
        # Drop unmapped states
        df = df[df["state"].notna()]

    # --- STEP 6: District Normalization ---
    if "district" in df.columns:
        df["district"] = (
            df["district"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

    # --- STEP 7: Date Parsing ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

# =======================================================
# 3. KPI CALCULATION FUNCTION (UPDATED)
# =======================================================
def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate mathematically consistent KPIs from preprocessed biometric data.
    Ensures: Total ≈ Avg × Number of States
    """
    
    # Validate input
    if df.empty:
        return {
            "error": "No data available",
            "coverage_percent": 0.0,
            "avg_activity_per_state_10k": 0.0,
            "total_updates_millions": 0.0,
            "total_generated_millions": 0.0,
            "top_performing_state": "N/A",
            "total_states_active": 0
        }
    
    # Required columns check
    required_cols = ['total_updates', 'aadhaar_generated', 'total_activity', 'state']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean[df_clean['state'].notna() & (df_clean['state'] != '')]
    
    if df_clean.empty:
        return {"error": "No valid state data after cleaning"}
    
    # Aggregate by state (CRITICAL: Don't filter out states for core KPIs!)
    state_agg = df_clean.groupby('state', as_index=False).agg({
        'total_updates': 'sum',
        'aadhaar_generated': 'sum',
        'total_activity': 'sum'
    })
    
    # Core calculations - MUST BE MATHEMATICALLY CONSISTENT
    # ------------------------------------------------------
    
    # 1. TOTAL UPDATES (in millions) - YOUR RAW DATA
    national_total_updates = df_clean['total_updates'].sum()
    total_updates_millions = national_total_updates / 1_000_000
    
    # 2. TOTAL GENERATED (in millions)
    national_total_generated = df_clean['aadhaar_generated'].sum()
    total_generated_millions = national_total_generated / 1_000_000
    
    # 3. TOTAL ACTIVITY - CRITICAL METRIC
    national_total_activity = df_clean['total_activity'].sum()
    
    # 4. AVERAGE ACTIVITY PER STATE (in 10k units)
    # Use ALL states for average calculation
    num_states = len(state_agg)
    if num_states > 0:
        avg_activity_per_state = national_total_activity / num_states
        avg_activity_per_state_10k = avg_activity_per_state / 10_000
    else:
        avg_activity_per_state_10k = 0.0
    
    # 5. MEDIAN ACTIVITY (for robustness)
    median_activity = state_agg['total_activity'].median() if not state_agg.empty else 0
    median_activity_per_state_10k = median_activity / 10_000
    
    # 6. COVERAGE % - SIMPLIFIED AND MEANINGFUL
    # Coverage = % of states performing above MEDIAN (not mean) - more stable
    if not state_agg.empty:
        median_activity_value = state_agg['total_activity'].median()
        above_median_states = state_agg[state_agg['total_activity'] > median_activity_value]
        
        # Calculate using ALL Indian states/UTs (36) for national perspective
        total_indian_states_uts = 36
        coverage_percent = (len(above_median_states) / total_indian_states_uts) * 100
        
        # Also calculate coverage of active states
        active_coverage_percent = (len(above_median_states) / num_states) * 100 if num_states > 0 else 0
    else:
        coverage_percent = 0.0
        active_coverage_percent = 0.0
    
    # 7. TOP PERFORMING STATE
    if not state_agg.empty:
        top_state_row = state_agg.loc[state_agg['total_activity'].idxmax()]
        top_state_name = top_state_row['state']
        top_state_activity = top_state_row['total_activity']
        
        # Dominance percentage
        top_state_dominance = (top_state_activity / national_total_activity) * 100 if national_total_activity > 0 else 0
        
        # Check if significantly above average
        is_significant = top_state_activity > (avg_activity_per_state * 1.5)
    else:
        top_state_name = "N/A"
        top_state_dominance = 0
        is_significant = False
    
    # 8. MATHEMATICAL CONSISTENCY CHECK
    # Verify: Total Activity ≈ Avg Activity × Number of States
    expected_total_from_avg = avg_activity_per_state * num_states
    consistency_ratio = (national_total_activity / expected_total_from_avg) if expected_total_from_avg > 0 else 1
    
    # 9. COMPLETENESS
    total_states_uts = 36
    completeness_percent = (num_states / total_states_uts) * 100
    
    # 10. INEQUALITY (Gini) - only if we have enough states
    def calculate_gini(activity_values):
        """Calculate Gini coefficient for inequality"""
        if len(activity_values) < 2:
            return 0.0
        
        values = np.array(activity_values)
        values = values.flatten()
        if np.sum(values) == 0:
            return 0.0
        
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
        return max(0.0, min(1.0, gini))
    
    gini_coefficient = calculate_gini(state_agg['total_activity'].values) if not state_agg.empty else 0.0
    
    # 11. PERFORMANCE DISTRIBUTION
    if len(state_agg) >= 4:
        q1 = state_agg['total_activity'].quantile(0.25)
        q2 = state_agg['total_activity'].quantile(0.50)
        q3 = state_agg['total_activity'].quantile(0.75)
    else:
        q1 = q2 = q3 = avg_activity_per_state
    
    # Return mathematically consistent KPIs
    return {
        # CORE KPIs (MUST BE CONSISTENT)
        "coverage_percent": round(coverage_percent, 1),  # Based on all 36 states/UTs
        "avg_activity_per_state_10k": round(avg_activity_per_state_10k, 1),
        "total_updates_millions": round(total_updates_millions, 2),
        "total_generated_millions": round(total_generated_millions, 2),
        
        # Supporting metrics
        "top_performing_state": top_state_name,
        "total_states_active": num_states,
        "raw_total_activity": int(national_total_activity),
        
        # Enhanced metrics
        "median_activity_per_state_10k": round(median_activity_per_state_10k, 1),
        "gini_coefficient": round(gini_coefficient, 3),
        "completeness_percent": round(completeness_percent, 1),
        "top_state_dominance_percent": round(top_state_dominance, 1),
        "active_coverage_percent": round(active_coverage_percent, 1),  # Coverage among active states
        
        # Mathematical consistency indicators
        "mathematical_consistency_ratio": round(consistency_ratio, 3),
        "expected_total_from_avg": round(expected_total_from_avg / 1_000_000, 2),
        
        # Performance distribution
        "performance_quartiles": {
            "q1_activity_10k": round(q1 / 10_000, 1),
            "median_activity_10k": round(q2 / 10_000, 1),
            "q3_activity_10k": round(q3 / 10_000, 1)
        },
        
        # Data quality
        "states_with_data": num_states,
        "total_states_expected": total_states_uts,
        "data_records_processed": len(df_clean),
        
        # Additional insights
        "updates_per_generated": round(national_total_updates / national_total_generated, 3) if national_total_generated > 0 else 0,
        "is_top_state_significant": is_significant
    }

# =======================================================
# 4. MAIN EXECUTION (FOR TESTING)
# =======================================================
if __name__ == "__main__":
    # Create Dummy Data to test the logic
    data = {
        "state": ["Uttar Pradesh", "Maharashtra", "Delhi", "Goa", "Bihar", "UnknownState"],
        "district": ["Lucknow", "Mumbai City", "New Delhi", "Panaji", "Patna", "X"],
        "bio_age_5_17": [5000, 4000, 2000, 500, 3000, 0],
        "bio_age_17_": [1000, 1200, 800, 100, 900, 0],
        "demographic_updates": [2000, 2500, 1500, 200, 1000, 0],
        "generated": [500, 400, 100, 50, 600, 0], # Column alias test
        "date": ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02", "2023-10-03", "2023-10-03"]
    }
    
    raw_df = pd.DataFrame(data)
    
    print("--- 1. Raw Data Columns ---")
    print(raw_df.columns.tolist())

    # Run Preprocessing
    clean_df = preprocess_biometric_data(raw_df)
    
    print("\n--- 2. Processed Data Head ---")
    print(clean_df[['state', 'total_updates', 'aadhaar_generated', 'total_activity']].head())

    # Run KPIs
    kpis = calculate_kpis(clean_df)
    
    print("\n--- 3. Calculated KPIs ---")
    for k, v in kpis.items():
        print(f"{k}: {v}")