import pandas as pd
import numpy as np

def generate_insights(df: pd.DataFrame, col_types: dict, stats_out: dict, cleaning_info: list):
    """
    Analyzes the statistical engine results and generates a list of 10-20 plain-English insights,
    categorized by severity (🔴 critical, 🟡 warning, 🟢 info).
    
    Returns:
        insights (list of dict): Each entry is {'text': str, 'severity': str, 'type': str}
    """
    insights = []
    
    # 1. Missing Value Insights
    for col_info in cleaning_info:
        col = col_info['column']
        missing_pct = col_info['missing_pct']
        if missing_pct > 20:
            insights.append({
                'text': f"Column `{col}` had significant missing data ({missing_pct:.1f}%). Imputed with {col_info['logical_type']} fallback.",
                'severity': 'critical' if missing_pct > 40 else 'warning',
                'type': 'missing'
            })
            
    # 2. Skewness & Normality Insights
    import scipy.stats as stats
    
    # Identify potential categorical grouping columns (cardinality 2 to 10)
    group_cols = []
    for c, ctype in col_types.items():
        if ctype == 'categorical' and c in df.columns:
            try:
                uniq_count = df[c].dropna().nunique()
                if 2 <= uniq_count <= 10:
                    group_cols.append(c)
            except Exception:
                pass

    for col, stat in stats_out['numerical'].items():
        skew = stat.get('skew')
        normality = stat.get('normality')
        
        if pd.notna(skew):
            minimum_val = stat.get('min', 0.0)
            if skew > 1.0:
                if minimum_val <= 0:
                    if minimum_val == 0:
                        recom = "log1p (log(x+1)) or Yeo-Johnson"
                    else:
                        recom = "Yeo-Johnson (as it contains negative values)"
                else:
                    recom = "log or Box-Cox"
                insights.append({
                    'text': f"Column `{col}` is heavily right-skewed (skew={skew:.2f}). Consider a {recom} transformation.",
                    'severity': 'warning',
                    'type': 'skew'
                })
            elif skew < -1.0:
                if minimum_val <= 0:
                    if minimum_val == 0:
                        recom = "reflecting and applying log1p or Yeo-Johnson"
                    else:
                        recom = "reflecting and applying Yeo-Johnson"
                else:
                    recom = "reflecting and applying log or Box-Cox"
                insights.append({
                    'text': f"Column `{col}` is heavily left-skewed (skew={skew:.2f}). Consider a {recom} transformation.",
                    'severity': 'warning',
                    'type': 'skew'
                })
                
        if normality == 'not normal':
            shapiro_p = stat.get('shapiro_p', 0)
            
            # Subgroup normality check
            subgroup_normal = False
            best_group_col = None
            
            for g_col in group_cols:
                try:
                    # Drop NaN values for both columns
                    grouped_data = df[[g_col, col]].dropna()
                    groups = [group[col].values for name, group in grouped_data.groupby(g_col) if len(group[col]) >= 3]
                    
                    if len(groups) >= 2:
                        all_normal = True
                        for g in groups:
                            # Sample up to 5000 observations to avoid scipy limitation and save time
                            sample_data = np.random.choice(g, min(len(g), 5000), replace=False) if len(g) > 3 else g
                            if len(sample_data) >= 3:
                                _, p = stats.shapiro(sample_data)
                                if p < 0.05:
                                    all_normal = False
                                    break
                            else:
                                all_normal = False
                                break
                        if all_normal:
                            subgroup_normal = True
                            best_group_col = g_col
                            break
                except Exception:
                    pass
            
            if subgroup_normal:
                insights.append({
                    'text': f"Column `{col}` is not normally distributed overall (Shapiro-Wilk p={shapiro_p:.4f}), BUT it is normally distributed within individual subgroups of `{best_group_col}`. Therefore, parametric tests (like ANOVA or t-tests) remain valid for group comparisons and non-parametric alternatives are not strictly required.",
                    'severity': 'info',
                    'type': 'normality'
                })
            else:
                insights.append({
                    'text': f"Column `{col}` is not normally distributed overall (Shapiro-Wilk p={shapiro_p:.4f}) or within categorical subgroups. Consider non-parametric tests (e.g. Mann-Whitney U, Kruskal-Wallis) or a Box-Cox/log transformation before applying parametric models.",
                    'severity': 'info',
                    'type': 'normality'
                })

    # 3. Outlier Insights
    for col_info in cleaning_info:
        col = col_info['column']
        outlier_pct = col_info['outlier_pct']
        if outlier_pct > 0.0:
            stat = stats_out['numerical'].get(col, {})
            skew = stat.get('skew', 0.0)
            if outlier_pct > 5.0 and pd.notna(skew) and skew > 1.5:
                insights.append({
                    'text': f"Column `{col}` contains a high outlier rate ({outlier_pct:.1f}%) and heavy right skew. These are likely legitimate heavy-tailed values rather than noise/errors. Avoid blindly dropping them; consider robust scaling (RobustScaler) or tail compression.",
                    'severity': 'warning',
                    'type': 'outliers'
                })
            elif outlier_pct > 10.0:
                insights.append({
                    'text': f"Column `{col}` contains a high percentage of true anomalies / outliers ({outlier_pct:.1f}%). Investigate, clip, or use models like Isolation Forest before training.",
                    'severity': 'warning',
                    'type': 'outliers'
                })
            else:
                insights.append({
                    'text': f"Column `{col}` has {outlier_pct:.1f}% outliers flagged by the IQR method.",
                    'severity': 'info',
                    'type': 'outliers'
                })

    # 4. Dominant Category Imbalance Insights
    for col, stat in stats_out['categorical'].items():
        mode_pct = stat.get('mode_pct', 0.0)
        mode_val = stat.get('mode')
        if mode_pct > 70.0:
            insights.append({
                'text': f"Category `{mode_val}` in column `{col}` dominates with {mode_pct:.1f}% of records. The dataset is highly imbalanced on this feature. If this is a target variable, utilize stratified train-test splitting (stratify=y), class weighting (class_weight='balanced'), and prioritize F1-score or Precision-Recall rather than overall accuracy.",
                'severity': 'warning' if mode_pct > 85.0 else 'info',
                'type': 'imbalance'
            })

    # 5. Strong Correlation & Multicollinearity Insights
    top_corrs = stats_out['dataset'].get('top_correlations', {})
    if top_corrs:
        pos_corrs = top_corrs.get('positive', [])
        neg_corrs = top_corrs.get('negative', [])
        
        for pair, val in pos_corrs:
            if val > 0.8:
                insights.append({
                    'text': f"Strong positive correlation between `{pair[0]}` and `{pair[1]}` (r={val:.2f}) indicates high collinearity. While this does not affect the predictive accuracy of tree-based models (like Random Forests or XGBoost), it can destabilize coefficient estimates in linear regression. Consider regularization (Ridge/Lasso) or dimensionality reduction (PCA) rather than simply dropping a feature.",
                    'severity': 'critical' if val > 0.9 else 'warning',
                    'type': 'correlation'
                })
            elif val > 0.5:
                insights.append({
                    'text': f"Moderate positive correlation detected between `{pair[0]}` and `{pair[1]}` (r={val:.2f}).",
                    'severity': 'info',
                    'type': 'correlation'
                })
                
        for pair, val in neg_corrs:
            if val < -0.8:
                insights.append({
                    'text': f"Strong negative correlation between `{pair[0]}` and `{pair[1]}` (r={val:.2f}) indicates high collinearity. While this does not affect tree-based model performance, it can inflate coefficient variance in linear models. Consider Ridge/Lasso regularization or PCA instead of simply dropping the feature.",
                    'severity': 'warning',
                    'type': 'correlation'
                })
            elif val < -0.5:
                insights.append({
                    'text': f"Moderate negative correlation detected between `{pair[0]}` and `{pair[1]}` (r={val:.2f}).",
                    'severity': 'info',
                    'type': 'correlation'
                })

    # 6. DateTime span & Trend Insights
    for col, stat in stats_out['datetime'].items():
        span_days = stat.get('span_days', 0)
        years = span_days / 365.25
        trend = stat.get('trend', 'flat')
        freq_month = stat.get('frequent_month')
        freq_year = stat.get('frequent_year')
        
        span_text = f"{years:.1f} years" if years >= 1.0 else f"{span_days} days"
        insights.append({
            'text': f"Time series column `{col}` spans {span_text} from {stat.get('min')} to {stat.get('max')}.",
            'severity': 'info',
            'type': 'datetime'
        })
        
        if trend in ['increasing', 'decreasing']:
            insights.append({
                'text': f"Frequency of events in `{col}` shows a clear '{trend}' trend direction over time.",
                'severity': 'info',
                'type': 'datetime'
            })
            
        if pd.notna(freq_month) or pd.notna(freq_year):
            insights.append({
                'text': f"Peak event occurrences in `{col}` are concentrated in {freq_month} / Year: {freq_year}.",
                'severity': 'info',
                'type': 'datetime'
            })

    # 7. Chi-Square & Category Association Insights
    chi_sqs = stats_out['dataset'].get('chi_square', [])
    for test in chi_sqs[:3]: # Limit to top 3 to avoid clutter
        insights.append({
            'text': f"Significant categorical association found between `{test['col1']}` and `{test['col2']}` (Chi-Square p-value = {test['p_value']:.4f}).",
            'severity': 'info',
            'type': 'association'
        })

    # 8. ANOVA & Numerical/Categorical Interactions
    anovas = stats_out['dataset'].get('anova', [])
    for test in anovas[:3]: # Limit to top 3 to avoid clutter
        insights.append({
            'text': f"Significant difference in means of `{test['num_col']}` observed across `{test['cat_col']}` groups (ANOVA p-value = {test['p_value']:.4f}).",
            'severity': 'info',
            'type': 'anova'
        })

    # 9. Fallback Baseline Insights to ensure we hit 10-20 insights
    if len(insights) < 12:
        # Add basic dataset information
        n_rows, n_cols = df.shape
        insights.append({
            'text': f"The dataset consists of {n_rows:,} rows and {n_cols} columns.",
            'severity': 'info',
            'type': 'dataset'
        })
        
        # Count types
        types_counts = {}
        for col, ctype in col_types.items():
            types_counts[ctype] = types_counts.get(ctype, 0) + 1
        for ctype, count in types_counts.items():
            insights.append({
                'text': f"Detected {count} `{ctype}` feature(s) in this dataset.",
                'severity': 'info',
                'type': 'dataset'
            })
            
        # Top unique cardinalities
        for col, stat in stats_out['categorical'].items():
            uniq = stat.get('unique_count', 0)
            if uniq > 0:
                insights.append({
                    'text': f"Column `{col}` has high categorical diversity with {uniq} unique categories.",
                    'severity': 'info',
                    'type': 'cardinality'
                })

    # Prioritize: Critical -> Warning -> Info
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    insights.sort(key=lambda x: severity_order.get(x['severity'], 3))
    
    return insights
