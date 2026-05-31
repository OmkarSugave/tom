import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

def analyze_numerical(series: pd.Series):
    """Computes comprehensive statistics for a single numerical column with high accuracy."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {}
        
    count = len(non_null)
    mean = non_null.mean()
    median = non_null.median()
    
    mode_series = non_null.mode()
    mode = mode_series[0] if len(mode_series) > 0 else np.nan
    
    std = non_null.std()
    var = non_null.var()
    minimum = non_null.min()
    maximum = non_null.max()
    val_range = maximum - minimum
    
    q25 = non_null.quantile(0.25)
    q75 = non_null.quantile(0.75)
    iqr = q75 - q25
    
    skew = non_null.skew()
    kurt = non_null.kurt()
    
    # Skewness interpretation
    if pd.isna(skew):
        skew_desc = "undefined"
    elif skew > 1:
        skew_desc = "heavily right-skewed"
    elif skew < -1:
        skew_desc = "heavily left-skewed"
    elif -0.5 <= skew <= 0.5:
        skew_desc = "approximately symmetric"
    else:
        skew_desc = "moderately skewed"
        
    # Kurtosis interpretation
    if pd.isna(kurt):
        kurt_desc = "undefined"
    elif kurt > 1:
        kurt_desc = "heavy-tailed / leptokurtic"
    elif kurt < -1:
        kurt_desc = "light-tailed / platykurtic"
    else:
        kurt_desc = "normal-tailed / mesokurtic"
        
    # Coefficient of Variation
    cv = (std / mean) if mean != 0 else np.nan
    
    # Shapiro-Wilk Normality Test
    # Sample up to 5000 observations to avoid scipy limitation and save time
    if len(non_null) > 3:
        sample_data = non_null.sample(min(len(non_null), 5000), random_state=42)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            normality = "likely normal" if shapiro_p >= 0.05 else "not normal"
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan
            normality = "unknown"
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        normality = "insufficient data"
        
    return {
        'count': count,
        'mean': mean,
        'median': median,
        'mode': mode,
        'std': std,
        'var': var,
        'min': minimum,
        'max': maximum,
        'range': val_range,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'skew': skew,
        'skew_desc': skew_desc,
        'kurt': kurt,
        'kurt_desc': kurt_desc,
        'cv': cv,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'normality': normality
    }

def analyze_categorical(series: pd.Series):
    """Computes categorical statistics for a single column."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {}
        
    count = len(non_null)
    unique_count = non_null.nunique()
    
    val_counts = non_null.value_counts()
    top_10 = val_counts.head(10).to_dict()
    
    mode_series = non_null.mode()
    mode = mode_series[0] if len(mode_series) > 0 else np.nan
    
    # Mode percentage
    mode_pct = (val_counts.iloc[0] / count * 100) if len(val_counts) > 0 else 0.0
    
    # Entropy calculation (diversity score)
    probs = val_counts / count
    entropy_val = stats.entropy(probs, base=2)
    
    return {
        'count': count,
        'unique_count': unique_count,
        'mode': mode,
        'mode_pct': mode_pct,
        'entropy': entropy_val,
        'top_10': top_10
    }

def analyze_datetime(series: pd.Series):
    """Computes datetime statistics for a single column."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {}
        
    min_date = non_null.min()
    max_date = non_null.max()
    span = max_date - min_date
    
    years = non_null.dt.year
    months = non_null.dt.month_name()
    days_of_week = non_null.dt.day_name()
    
    frequent_year = years.mode()[0] if len(years.mode()) > 0 else np.nan
    frequent_month = months.mode()[0] if len(months.mode()) > 0 else np.nan
    frequent_day_of_week = days_of_week.mode()[0] if len(days_of_week.mode()) > 0 else np.nan
    
    # Trend detection based on record frequency
    try:
        if span.days > 365:
            freq = non_null.dt.to_period("M").value_counts().sort_index()
        else:
            freq = non_null.dt.to_period("D").value_counts().sort_index()
            
        if len(freq) > 3:
            indices = np.arange(len(freq))
            values = freq.values
            # Compute Spearman correlation
            corr, p_val = stats.spearmanr(indices, values)
            if p_val < 0.05:
                if corr > 0.3:
                    trend = "increasing"
                elif corr < -0.3:
                    trend = "decreasing"
                else:
                    trend = "flat"
            else:
                trend = "flat"
        else:
            trend = "flat"
    except Exception:
        trend = "unknown"
        
    return {
        'min': min_date,
        'max': max_date,
        'span_days': span.days,
        'frequent_year': frequent_year,
        'frequent_month': frequent_month,
        'frequent_day_of_week': frequent_day_of_week,
        'trend': trend
    }

def compute_dataset_stats(df: pd.DataFrame, col_types: dict):
    """
    Computes dataset-wide statistics with high precision:
    - Correlation matrix (Pearson)
    - Top 5 positive and negative correlations
    - Chi-Square tests between categorical columns
    - ANOVA tests between categorical & numerical columns
    """
    results = {
        'correlations': {},
        'top_correlations': [],
        'chi_square': [],
        'anova': []
    }
    
    # 1. Pearson Correlation Matrix
    num_cols = [col for col, ctype in col_types.items() if ctype == 'numerical' and col in df.columns]
    cat_cols = [col for col, ctype in col_types.items() if ctype == 'categorical' and col in df.columns]
    
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr(method='pearson')
        results['correlations'] = corr_matrix.to_dict()
        
        # Extract pairs for top 5 positive/negative
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]
                val = corr_matrix.loc[col1, col2]
                if pd.notna(val):
                    pairs.append(((col1, col2), val))
                    
        pairs.sort(key=lambda x: x[1], reverse=True)
        # Highlight top 5 strongest positive and top 5 strongest negative
        top_pos = [p for p in pairs if p[1] > 0][:5]
        top_neg = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])[:5]
        results['top_correlations'] = {
            'positive': top_pos,
            'negative': top_neg
        }
        
    # 2. Chi-Square Test (Associations between Categorical columns)
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col1 = cat_cols[i]
                col2 = cat_cols[j]
                try:
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    # Run chi2 test
                    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                    if p < 0.05:
                        results['chi_square'].append({
                            'col1': col1,
                            'col2': col2,
                            'p_value': p,
                            'chi2': chi2,
                            'dof': dof,
                            'significant': True
                        })
                except Exception:
                    pass
                    
    # 3. ANOVA Test (Associations between Numeric and Categorical columns)
    if len(num_cols) > 0 and len(cat_cols) > 0:
        for num_col in num_cols:
            for cat_col in cat_cols:
                try:
                    # Group numeric by categorical classes
                    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col) if len(group[num_col].dropna()) > 0]
                    # Filter out groups with very small sample size or if only 1 group
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) > 1:
                        f_stat, p_val = stats.f_oneway(*groups)
                        if p_val < 0.05:
                            results['anova'].append({
                                'num_col': num_col,
                                'cat_col': cat_col,
                                'f_stat': f_stat,
                                'p_value': p_val,
                                'significant': True
                            })
                except Exception:
                    pass
                    
    return results

def train_live_baseline_model(df: pd.DataFrame, col_types: dict):
    """
    Autonomously trains a live baseline RandomForest model on the dataset
    using 5-Fold Cross-Validation, returning actual accuracy/R2 metrics and feature importances.
    """
    try:
        import sklearn.model_selection as ms
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return {'error': 'scikit-learn is not installed in this environment.'}
        
    # 1. Target identification
    # We prefer a categorical target column (not ID) with 2 to 10 unique classes
    target_col = None
    target_type = None
    
    for col, ctype in col_types.items():
        if ctype == 'categorical' and col in df.columns and col != df.columns[0]:
            try:
                uniq = df[col].nunique()
                if 2 <= uniq <= 10:
                    target_col = col
                    target_type = 'classification'
                    break
            except Exception:
                pass
                
    # Fallback to the last numerical column (excluding ID)
    if target_col is None:
        num_cols = [c for c, t in col_types.items() if t == 'numerical' and c in df.columns]
        if num_cols:
            target_col = num_cols[-1]
            target_type = 'regression'
            
    if target_col is None:
        return None
        
    # 2. Extract Features
    # Exclude ID, Text_ID, target, and datetime features
    feature_cols = [c for c, t in col_types.items() if t in ['numerical', 'categorical'] and c != target_col and c in df.columns]
    if not feature_cols:
        return None
        
    try:
        # Preprocess features
        X_raw = df[feature_cols].copy()
        y_raw = df[target_col].copy()
        
        # Drop rows with NaN in target
        valid_idx = y_raw.notnull()
        X_raw = X_raw[valid_idx]
        y_raw = y_raw[valid_idx]
        
        if len(y_raw) < 10: # not enough records to build a robust baseline
            return None
            
        # One-hot encode categoricals in feature set
        cat_features = [c for c in feature_cols if col_types.get(c) == 'categorical']
        X = pd.get_dummies(X_raw, columns=cat_features, drop_first=True)
        
        # Impute any leftover NaNs
        X = X.fillna(X.median(numeric_only=True))
        
        # Preprocess target
        if target_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            
            # 5-fold cross validation
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cv_results = ms.cross_validate(model, X, y, cv=min(5, len(X)), scoring=['accuracy', 'f1_weighted'])
            
            # Fit on all data for feature importance
            model.fit(X, y)
            
            acc = float(cv_results['test_accuracy'].mean())
            f1 = float(cv_results['test_f1_weighted'].mean())
            
            results = {
                'target': target_col,
                'type': 'classification',
                'metric_name': 'Accuracy / F1-Score',
                'accuracy': acc,
                'f1_score': f1,
                'classes': [str(c) for c in le.classes_]
            }
        else:
            y = y_raw.values
            
            # 5-fold cross validation
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            cv_results = ms.cross_validate(model, X, y, cv=min(5, len(X)), scoring=['r2', 'neg_root_mean_squared_error'])
            
            # Fit on all data for feature importance
            model.fit(X, y)
            
            r2 = float(cv_results['test_r2'].mean())
            rmse = float(-cv_results['test_neg_root_mean_squared_error'].mean())
            
            results = {
                'target': target_col,
                'type': 'regression',
                'metric_name': 'R² / RMSE',
                'r2': r2,
                'rmse': rmse
            }
            
        # Extract Top 5 Feature Importances
        importances = model.feature_importances_
        feature_names = X.columns
        feat_imp_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
        
        results['feature_importances'] = [{'feature': str(f), 'importance': float(i)} for f, i in feat_imp_pairs]
        return results
        
    except Exception as e:
        return {'error': str(e)}

def compute_all_stats(df: pd.DataFrame, col_types: dict):
    """Runs all individual, global, and baseline machine learning evaluations."""
    stats_out = {
        'numerical': {},
        'categorical': {},
        'datetime': {},
        'dataset': {},
        'baseline_ml': None
    }
    
    for col, ctype in col_types.items():
        if col not in df.columns:
            continue
        if ctype == 'numerical':
            stats_out['numerical'][col] = analyze_numerical(df[col])
        elif ctype == 'categorical':
            stats_out['categorical'][col] = analyze_categorical(df[col])
        elif ctype == 'datetime':
            stats_out['datetime'][col] = analyze_datetime(df[col])
            
    stats_out['dataset'] = compute_dataset_stats(df, col_types)
    stats_out['baseline_ml'] = train_live_baseline_model(df, col_types)
    return stats_out
