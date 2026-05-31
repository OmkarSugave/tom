import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import tom.utils as utils

console = Console()

def clean_data(df: pd.DataFrame):
    """
    Cleans the DataFrame automatically with high data-science accuracy:
    - Duplicate column renaming
    - Trimming whitespaces in strings
    - Datetime parsing for text columns that look like dates
    - Coercion of numerical objects
    - High missing value column dropping (>50%)
    - KNN Imputation for numerical columns (scikit-learn KNNImputer)
    - Mode Imputation for categorical/date features
    - Row-level duplicate removal
    - Outlier identification using Isolation Forest + IQR bounds
    - Classification of low vs high cardinality string columns
    
    Returns:
        df_cleaned (pd.DataFrame): The cleaned DataFrame.
        summary_info (list of dict): Details of each column's cleaning process.
        warnings (list of str): Any warnings generated during cleaning.
        dup_rows_removed (int): Number of duplicate rows dropped.
    """
    warnings = []
    
    # 1. Rename duplicate column names
    df_cleaned = utils.rename_duplicate_columns(df)
    
    # 2. Duplicate Row Removal
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    dup_rows_removed = initial_rows - len(df_cleaned)
    if dup_rows_removed > 0:
        warnings.append(f"Dropped {dup_rows_removed} duplicate rows.")
        
    # Detect initial column types using utils
    col_types = utils.detect_column_types(df_cleaned)
    
    summary_info = []
    dropped_cols = []
    
    # Pre-clean string whitespaces and coerce datetime/numerical objects
    for col in list(df_cleaned.columns):
        series = df_cleaned[col]
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0
        
        if missing_pct > 50:
            warnings.append(f"Column '{col}' has {missing_pct:.1f}% missing values. Dropping column.")
            df_cleaned = df_cleaned.drop(columns=[col])
            dropped_cols.append(col)
            continue
            
        if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
            try:
                series_stripped = series.astype(str).str.strip()
                series_stripped = series_stripped.replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                df_cleaned[col] = series_stripped
                series = df_cleaned[col]
            except Exception:
                pass

        col_logical_type = col_types.get(col, 'categorical')
        
        if col_logical_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(series):
            try:
                df_cleaned[col] = pd.to_datetime(series, errors='coerce')
                cleaned_dtype = "datetime64[ns]"
            except Exception:
                col_types[col] = 'categorical'
                
        if col_logical_type == 'numerical' and not pd.api.types.is_numeric_dtype(series):
            try:
                df_cleaned[col] = pd.to_numeric(series, errors='coerce')
                cleaned_dtype = str(df_cleaned[col].dtype)
            except Exception:
                col_types[col] = 'categorical'

    # Retrieve final columns and types
    col_types = utils.detect_column_types(df_cleaned)
    num_cols = [c for c, t in col_types.items() if t == 'numerical' and c in df_cleaned.columns]
    
    # 3. High-Accuracy Imputation: KNN Imputer for all numerical columns
    imputed_num_vals = {}
    if num_cols:
        cols_with_missing = [c for c in num_cols if df_cleaned[c].isnull().sum() > 0]
        if cols_with_missing:
            try:
                from sklearn.impute import KNNImputer
                # Using 5-neighbors for highly accurate regression-based reconstruction
                imputer = KNNImputer(n_neighbors=min(5, len(df_cleaned)))
                df_cleaned[num_cols] = imputer.fit_transform(df_cleaned[num_cols])
                for c in cols_with_missing:
                    imputed_num_vals[c] = "KNN Imputed"
            except Exception as e:
                # Fallback to median
                warnings.append(f"KNN Imputation failed ({str(e)}), falling back to robust median values.")
                for col in num_cols:
                    median_val = df_cleaned[col].median()
                    if pd.isnull(median_val):
                        median_val = 0.0
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    imputed_num_vals[col] = f"Median ({median_val:.2f})"

    # 4. Mode Imputation for Categorical/Datetime, and Outlier/Anomalies detection
    for col in list(df_cleaned.columns):
        series = df_cleaned[col]
        orig_dtype = str(df[col].dtype) if col in df.columns else str(series.dtype)
        cleaned_dtype = str(series.dtype)
        col_logical_type = col_types.get(col, 'categorical')
        
        missing_count = int(df[col].isnull().sum()) if col in df.columns else 0
        missing_pct = (missing_count / len(df_cleaned)) * 100 if len(df_cleaned) > 0 else 0
        
        imputed_val = None
        missing_filled = 0
        outlier_pct = 0.0
        
        # Continuous variable KNN tracking
        if col_logical_type == 'numerical' and col in imputed_num_vals:
            imputed_val = imputed_num_vals[col]
            missing_filled = missing_count

        # Discrete/Date variable Imputations
        if missing_count > 0 and col_logical_type != 'numerical':
            if col_logical_type in ['categorical', 'text_id', 'id']:
                mode_series = series.mode()
                imputed_val = mode_series[0] if len(mode_series) > 0 else "Unknown"
                df_cleaned[col] = series.fillna(imputed_val)
                missing_filled = missing_count
            elif col_logical_type == 'datetime':
                mode_series = series.mode()
                imputed_val = mode_series[0] if len(mode_series) > 0 else series.min()
                if pd.notnull(imputed_val):
                    df_cleaned[col] = series.fillna(imputed_val)
                    missing_filled = missing_count
            series = df_cleaned[col]

        # Outlier Detection for numerical columns using Isolation Forest
        if col_logical_type == 'numerical':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            iqr_outliers_count = 0
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers_count = len(series[(series < lower_bound) | (series > upper_bound)])
            
            try:
                from sklearn.ensemble import IsolationForest
                # Fit highly accurate Isolation Forest (Contamination auto)
                iso = IsolationForest(contamination='auto', random_state=42)
                preds = iso.fit_predict(series.values.reshape(-1, 1))
                if_outliers_count = int((preds == -1).sum())
                # Use union anomaly metrics to maintain extreme accuracy
                outlier_pct = (max(iqr_outliers_count, if_outliers_count) / len(series)) * 100
                if outlier_pct > 10:
                    warnings.append(f"Column '{col}' has a high percentage of anomalous outliers ({outlier_pct:.1f}%).")
            except Exception:
                outlier_pct = (iqr_outliers_count / len(series)) * 100 if len(series) > 0 else 0.0

        if col_logical_type == 'text_id':
            warnings.append(f"Column '{col}' is high-cardinality string (>20 unique values). Labeled as text/id-like.")
        elif col_logical_type == 'id':
            warnings.append(f"Column '{col}' is recognized as an Identifier (ID/Key) and will be preserved but excluded from statistical summaries and correlation analysis.")

        summary_info.append({
            'column': col,
            'original_type': orig_dtype,
            'cleaned_type': cleaned_dtype,
            'logical_type': col_logical_type,
            'missing_filled': missing_filled,
            'missing_pct': missing_pct,
            'outlier_pct': outlier_pct,
            'imputed_value': str(imputed_val) if imputed_val is not None else 'None'
        })
        
    return df_cleaned, summary_info, warnings, dup_rows_removed

def print_clean_report(summary_info, warnings, dup_rows_removed):
    """Prints a beautiful summary table of the data cleaning using rich."""
    rprint("\n[bold cyan]🧹 Data Preprocessing & Cleaning Summary[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Column Name", style="bold white")
    table.add_column("Original Type", style="cyan")
    table.add_column("Cleaned Type", style="green")
    table.add_column("Logical Category", style="yellow")
    table.add_column("Imputed Counts", justify="right", style="magenta")
    table.add_column("Outlier %", justify="right", style="red")
    
    for info in summary_info:
        missing_str = f"{info['missing_filled']} ({info['missing_pct']:.1f}%)" if info['missing_filled'] > 0 else "0"
        outlier_str = f"{info['outlier_pct']:.1f}%" if info['outlier_pct'] > 0 else "0.0%"
        
        table.add_row(
            info['column'],
            info['original_type'],
            info['cleaned_type'],
            info['logical_type'].upper(),
            missing_str,
            outlier_str
        )
        
    console.print(table)
    
    if dup_rows_removed > 0:
        rprint(f"[bold yellow]⚠️ Removed {dup_rows_removed} duplicate rows from the dataset.[/bold yellow]")
        
    if warnings:
        rprint("\n[bold orange3]⚠️ Cleaning Alerts:[/bold orange3]")
        for w in warnings:
            rprint(f"  • [yellow]{w}[/yellow]")
    else:
        rprint("\n[bold green]✨ No cleaning alerts. Dataset is clean![/bold green]")
