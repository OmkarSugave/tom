import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import tom.utils as utils

console = Console()

def clean_data(df: pd.DataFrame):
    """
    Cleans the DataFrame automatically, performing:
    - Duplicate column renaming
    - Trimming whitespaces in strings
    - Datetime parsing for text columns that look like dates
    - Coercion of numerical objects
    - High missing value column dropping (>50%)
    - Missing value imputation (median for numerical, mode for categorical)
    - Row-level duplicate removal
    - Outlier identification (IQR method)
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
    
    original_shape = df_cleaned.shape
    
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
    
    # 3. Process each column
    for col in list(df_cleaned.columns):
        series = df_cleaned[col]
        orig_dtype = str(series.dtype)
        
        # Calculate missing rate
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0
        
        # Handle high missing rate (>50%)
        if missing_pct > 50:
            warnings.append(f"Column '{col}' has {missing_pct:.1f}% missing values. Dropping column.")
            df_cleaned = df_cleaned.drop(columns=[col])
            dropped_cols.append(col)
            continue
            
        # Clean dtypes and missing values
        cleaned_dtype = orig_dtype
        imputed_val = None
        missing_filled = 0
        outlier_pct = 0.0
        
        # Handle string stripping and auto type conversion
        if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
            try:
                # Strip string whitespace
                series_stripped = series.astype(str).str.strip()
                # If they were all nan strings, map back to real NaN
                series_stripped = series_stripped.replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                df_cleaned[col] = series_stripped
                series = df_cleaned[col]
            except Exception:
                pass

        col_logical_type = col_types.get(col, 'categorical')
        
        # Coerce object to datetime if logical type is datetime
        if col_logical_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(series):
            try:
                df_cleaned[col] = pd.to_datetime(series, errors='coerce')
                series = df_cleaned[col]
                cleaned_dtype = "datetime64[ns]"
            except Exception:
                col_logical_type = 'categorical' # fallback
                
        # Coerce object to numeric if logical type is numerical
        if col_logical_type == 'numerical' and not pd.api.types.is_numeric_dtype(series):
            try:
                df_cleaned[col] = pd.to_numeric(series, errors='coerce')
                series = df_cleaned[col]
                cleaned_dtype = str(series.dtype)
            except Exception:
                col_logical_type = 'categorical' # fallback

        # Impute missing values
        if missing_count > 0:
            if col_logical_type == 'numerical':
                # Numerical -> Fill with median
                imputed_val = series.median()
                if pd.isnull(imputed_val):
                    imputed_val = 0.0 # fallback if median is NaN
                df_cleaned[col] = series.fillna(imputed_val)
                missing_filled = missing_count
            elif col_logical_type in ['categorical', 'text_id', 'id']:
                # Categorical / Text / ID -> Fill with mode
                mode_series = series.mode()
                imputed_val = mode_series[0] if len(mode_series) > 0 else "Unknown"
                df_cleaned[col] = series.fillna(imputed_val)
                missing_filled = missing_count
            elif col_logical_type == 'datetime':
                # Datetime -> Fill with mode or median date
                mode_series = series.mode()
                imputed_val = mode_series[0] if len(mode_series) > 0 else series.min()
                if pd.notnull(imputed_val):
                    df_cleaned[col] = series.fillna(imputed_val)
                    missing_filled = missing_count
            series = df_cleaned[col]
            
        # Outlier Detection for numerical columns using IQR method
        if col_logical_type == 'numerical':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_pct = (len(outliers) / len(series)) * 100
                if outlier_pct > 10:
                    warnings.append(f"Column '{col}' has a high percentage of outliers ({outlier_pct:.1f}%).")
            else:
                outlier_pct = 0.0
                
        # Cardanality check / categorization labels
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
