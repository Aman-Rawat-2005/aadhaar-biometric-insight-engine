import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple

# ======================================================
# ADVANCED POLICY PRIORITY ENGINE
# ======================================================

def policy_priority_states(df: pd.DataFrame, 
                          weights: Dict[str, float] = None,
                          include_inequality: bool = True) -> pd.DataFrame:
    """
    Advanced policy priority scoring with multiple dimensions:
    
    1. Coverage Dimension (Accessibility)
    2. Efficiency Dimension (Performance)
    3. Inequality Dimension (Fairness)
    4. Risk Dimension (Vulnerability)
    
    Returns comprehensive policy recommendations with resource allocation guidance.
    """
    
    # -------------------------------
    # Default weights for dimensions
    # -------------------------------
    if weights is None:
        weights = {
            'coverage_weight': 0.25,      # How well covered is the state
            'efficiency_weight': 0.25,     # How efficient are operations
            'inequality_weight': 0.20,     # How unequal is distribution
            'risk_weight': 0.15,           # How vulnerable is the state
            'growth_weight': 0.15          # Recent performance trends
        }
    
    # -------------------------------
    # Multi-dimensional state aggregation
    # -------------------------------
    state_agg = (
        df.groupby("state", as_index=False)
        .agg(
            # Basic metrics
            total_updates=("total_updates", "sum"),
            total_districts=("district", "nunique"),
            
            # Statistical metrics
            avg_updates_per_district=("total_updates", "mean"),
            median_updates_per_district=("total_updates", "median"),
            std_updates_per_district=("total_updates", "std"),
            min_updates_per_district=("total_updates", "min"),
            max_updates_per_district=("total_updates", "max"),
            
            # Coverage metrics
            districts_with_data=("total_updates", lambda x: (x > 0).sum()),
            zero_update_districts=("total_updates", lambda x: (x == 0).sum()),
            
            # Efficiency metrics (if time data exists)
            # These would need timestamp column in your actual data
            # avg_processing_time=("processing_time", "mean"),
            # success_rate=("success_flag", "mean"),
        )
    )
    
    # Calculate derived metrics
    state_agg['coverage_rate'] = (state_agg['districts_with_data'] / 
                                  state_agg['total_districts'] * 100)
    state_agg['zero_update_pct'] = (state_agg['zero_update_districts'] / 
                                    state_agg['total_districts'] * 100)
    
    # Inequality metrics
    state_agg['inequality_ratio'] = np.where(
        state_agg['min_updates_per_district'] > 0,
        state_agg['max_updates_per_district'] / state_agg['min_updates_per_district'],
        0
    )
    
    state_agg['cv_percent'] = np.where(
        state_agg['avg_updates_per_district'] > 0,
        (state_agg['std_updates_per_district'] / state_agg['avg_updates_per_district']) * 100,
        0
    )
    
    # -------------------------------
    # National benchmarks (dynamic)
    # -------------------------------
    national_metrics = {
        'national_avg': state_agg['avg_updates_per_district'].mean(),
        'national_median': state_agg['avg_updates_per_district'].median(),
        'national_coverage': state_agg['coverage_rate'].mean(),
        'national_cv': state_agg['cv_percent'].mean(),
        'national_zero_pct': state_agg['zero_update_pct'].mean()
    }
    
    # -------------------------------
    # Multi-dimensional scoring system
    # -------------------------------
    
    # 1. COVERAGE DIMENSION (How well covered is the state)
    state_agg['coverage_score'] = 100 - state_agg['zero_update_pct']
    
    # 2. EFFICIENCY DIMENSION (Performance relative to national average)
    state_agg['efficiency_score'] = np.where(
        national_metrics['national_avg'] > 0,
        (state_agg['avg_updates_per_district'] / national_metrics['national_avg']) * 100,
        100
    )
    # Cap at 200% to prevent outliers from dominating
    state_agg['efficiency_score'] = np.minimum(state_agg['efficiency_score'], 200)
    
    # 3. INEQUALITY DIMENSION (Fairness of distribution)
    if include_inequality:
        # Lower CV = more equal = higher score
        state_agg['inequality_score'] = np.maximum(0, 100 - state_agg['cv_percent'])
    else:
        state_agg['inequality_score'] = 100  # Neutral if inequality not considered
    
    # 4. RISK DIMENSION (Vulnerability assessment)
    # States with low coverage AND high inequality are high risk
    state_agg['risk_score'] = np.where(
        (state_agg['coverage_score'] < 70) & (state_agg['inequality_score'] < 60),
        30,  # High risk = low score
        np.where(
            (state_agg['coverage_score'] < 85) | (state_agg['inequality_score'] < 75),
            60,  # Medium risk
            90   # Low risk
        )
    )
    
    # 5. GROWTH DIMENSION (Performance trend - would need temporal data)
    # Placeholder: using deviation from median as growth proxy
    state_agg['growth_score'] = np.where(
        national_metrics['national_median'] > 0,
        (state_agg['median_updates_per_district'] / national_metrics['national_median']) * 100,
        100
    )
    
    # -------------------------------
    # Normalize scores (0-100)
    # -------------------------------
    scaler = MinMaxScaler(feature_range=(0, 100))
    
    score_columns = ['coverage_score', 'efficiency_score', 
                     'inequality_score', 'risk_score', 'growth_score']
    
    for col in score_columns:
        if state_agg[col].std() > 0:
            state_agg[f'{col}_norm'] = scaler.fit_transform(
                state_agg[[col]]
            ).flatten()
        else:
            state_agg[f'{col}_norm'] = state_agg[col]
    
    # -------------------------------
    # Composite Policy Priority Score
    # -------------------------------
    state_agg['policy_priority_score'] = (
        weights['coverage_weight'] * state_agg['coverage_score_norm'] +
        weights['efficiency_weight'] * state_agg['efficiency_score_norm'] +
        weights['inequality_weight'] * state_agg['inequality_score_norm'] +
        weights['risk_weight'] * state_agg['risk_score_norm'] +
        weights['growth_weight'] * state_agg['growth_score_norm']
    )
    
    # -------------------------------
    # Priority classification (multi-tier)
    # -------------------------------
    def classify_priority(score, coverage, inequality):
        """Advanced classification with multiple conditions"""
        
        # CRITICAL: Very low coverage AND high inequality
        if coverage < 50 and inequality < 40:
            return "CRITICAL INTERVENTION", "#DC2626", 5
        
        # HIGH: Multiple risk factors
        elif score < 40 or (coverage < 70 and inequality < 60):
            return "HIGH PRIORITY", "#EF4444", 4
        
        # MEDIUM-HIGH: Some concerning indicators
        elif score < 60 or coverage < 80:
            return "MEDIUM-HIGH PRIORITY", "#F59E0B", 3
        
        # MEDIUM: Slightly below average
        elif score < 75:
            return "MEDIUM PRIORITY", "#FBBF24", 2
        
        # LOW: Performing well
        elif score < 90:
            return "LOW PRIORITY", "#10B981", 1
        
        # EXEMPLARY: Top performers
        else:
            return "EXEMPLARY", "#059669", 0
    
    # Apply classification
    priority_data = state_agg.apply(
        lambda row: classify_priority(
            row['policy_priority_score'],
            row['coverage_score'],
            row['inequality_score']
        ),
        axis=1,
        result_type='expand'
    )
    
    state_agg['policy_priority'], state_agg['priority_color'], state_agg['priority_level'] = \
        priority_data[0], priority_data[1], priority_data[2]
    
    # -------------------------------
    # Resource allocation recommendations
    # -------------------------------
    def generate_recommendations(row):
        recommendations = []
        
        # Coverage recommendations
        if row['coverage_score'] < 70:
            recommendations.append(f"Expand coverage: {100-row['coverage_score']:.0f}% districts need attention")
        
        # Efficiency recommendations
        if row['efficiency_score'] < 80:
            recommendations.append(f"Improve efficiency: {100-row['efficiency_score']:.0f}% below target")
        
        # Inequality recommendations
        if row['inequality_score'] < 70:
            recommendations.append(f"Reduce inequality: CV = {row['cv_percent']:.1f}% (target < 30%)")
        
        # Risk mitigation
        if row['risk_score'] < 60:
            recommendations.append("High-risk state: Requires immediate monitoring")
        
        if not recommendations:
            recommendations.append("Performing well: Maintain current operations")
        
        return " | ".join(recommendations)
    
    state_agg['recommendations'] = state_agg.apply(generate_recommendations, axis=1)
    
    # -------------------------------
    # Urgency score (0-10)
    # -------------------------------
    state_agg['urgency_score'] = (
        (100 - state_agg['coverage_score']) * 0.3 +
        (100 - state_agg['inequality_score']) * 0.3 +
        (100 - state_agg['risk_score']) * 0.4
    ) / 10
    
    # -------------------------------
    # Resource allocation estimate
    # -------------------------------
    national_budget = 100  # Normalized to 100 units
    total_urgency = state_agg['urgency_score'].sum()
    
    state_agg['resource_allocation_%'] = np.where(
        total_urgency > 0,
        (state_agg['urgency_score'] / total_urgency) * 100,
        100 / len(state_agg)
    ).round(2)
    
    # -------------------------------
    # Performance quadrant classification
    # -------------------------------
    def performance_quadrant(coverage, efficiency):
        if coverage >= 75 and efficiency >= 100:
            return "LEADERS", "High coverage, High efficiency"
        elif coverage >= 75 and efficiency < 100:
            return "COVERAGE STARS", "High coverage, Moderate efficiency"
        elif coverage < 75 and efficiency >= 100:
            return "EFFICIENCY STARS", "Moderate coverage, High efficiency"
        else:
            return "NEEDS SUPPORT", "Low coverage, Low efficiency"
    
    state_agg['performance_quadrant'], state_agg['quadrant_description'] = zip(
        *state_agg.apply(
            lambda row: performance_quadrant(row['coverage_score'], row['efficiency_score']),
            axis=1
        )
    )
    
    # -------------------------------
    # Final formatting and sorting
    # -------------------------------
    # Select and order columns for output
    output_columns = [
        'state',
        'policy_priority_score',
        'policy_priority',
        'priority_level',
        'priority_color',
        'urgency_score',
        'resource_allocation_%',
        'performance_quadrant',
        
        # Dimension scores
        'coverage_score',
        'efficiency_score', 
        'inequality_score',
        'risk_score',
        'growth_score',
        
        # Key metrics
        'total_updates',
        'total_districts',
        'avg_updates_per_district',
        'coverage_rate',
        'cv_percent',
        'zero_update_pct',
        
        # Recommendations
        'recommendations',
        'quadrant_description'
    ]
    
    # Filter to available columns
    available_columns = [col for col in output_columns if col in state_agg.columns]
    
    result_df = state_agg[available_columns].copy()
    
    # Sort by priority (critical first)
    result_df = result_df.sort_values(
        ['priority_level', 'policy_priority_score'],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    return result_df


# ======================================================
# ADDITIONAL POLICY ANALYTICS FUNCTIONS
# ======================================================

def get_policy_insights(priority_df: pd.DataFrame) -> Dict:
    """
    Generate high-level policy insights from priority analysis
    """
    insights = {
        'critical_states': [],
        'recommended_actions': [],
        'resource_distribution': {},
        'performance_summary': {}
    }
    
    # Critical states requiring immediate attention
    critical = priority_df[priority_df['policy_priority'] == 'CRITICAL INTERVENTION']
    if not critical.empty:
        insights['critical_states'] = critical['state'].tolist()
        insights['critical_resource_share'] = critical['resource_allocation_%'].sum()
    
    # Performance distribution
    priority_counts = priority_df['policy_priority'].value_counts()
    insights['performance_summary'] = priority_counts.to_dict()
    
    # Top recommendations
    top_issues = priority_df.head(10)['recommendations'].str.split(' | ', expand=True).stack().value_counts()
    insights['top_recommendations'] = top_issues.head(5).to_dict()
    
    # Resource allocation summary
    insights['resource_distribution'] = {
        'top_5_states_share': priority_df.head(5)['resource_allocation_%'].sum(),
        'bottom_5_states_share': priority_df.tail(5)['resource_allocation_%'].sum(),
        'avg_allocation': priority_df['resource_allocation_%'].mean()
    }
    
    return insights


def policy_heatmap_data(priority_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for policy heatmap visualization
    """
    heatmap_data = priority_df.copy()
    
    # Create scaled values for heatmap (0-1)
    heatmap_columns = ['coverage_score', 'efficiency_score', 
                      'inequality_score', 'risk_score', 'urgency_score']
    
    for col in heatmap_columns:
        if col in heatmap_data.columns:
            min_val = heatmap_data[col].min()
            max_val = heatmap_data[col].max()
            if max_val > min_val:
                heatmap_data[f'{col}_scaled'] = (
                    (heatmap_data[col] - min_val) / (max_val - min_val)
                )
    
    return heatmap_data


def generate_policy_report(priority_df: pd.DataFrame, insights: Dict) -> str:
    """
    Generate a text report for policymakers
    """
    report_lines = []
    
    report_lines.append("=" * 60)
    report_lines.append("POLICY PRIORITY ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append(f"Total States Analyzed: {len(priority_df)}")
    report_lines.append(f"Critical Intervention States: {len(insights.get('critical_states', []))}")
    report_lines.append(f"High Priority States: {(priority_df['policy_priority'] == 'HIGH PRIORITY').sum()}")
    report_lines.append("")
    
    # Critical states section
    if insights['critical_states']:
        report_lines.append("🚨 CRITICAL INTERVENTION REQUIRED:")
        for state in insights['critical_states']:
            state_data = priority_df[priority_df['state'] == state].iloc[0]
            report_lines.append(f"  • {state}: Coverage={state_data['coverage_score']:.0f}%, "
                              f"Inequality={100-state_data['inequality_score']:.0f}%")
        report_lines.append("")
    
    # Resource allocation
    report_lines.append("💰 RESOURCE ALLOCATION SUMMARY:")
    report_lines.append(f"  Top 5 states receive: {insights['resource_distribution']['top_5_states_share']:.1f}% of resources")
    report_lines.append(f"  Average allocation per state: {insights['resource_distribution']['avg_allocation']:.1f}%")
    report_lines.append("")
    
    # Top recommendations
    report_lines.append("🎯 TOP RECOMMENDATIONS:")
    for rec, count in insights.get('top_recommendations', {}).items():
        report_lines.append(f"  • {rec} ({count} states)")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)