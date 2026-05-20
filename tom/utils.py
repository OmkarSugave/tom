import os
import pandas as pd
import numpy as np

def rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-renames duplicate column names in a DataFrame by appending _1, _2, etc."""
    cols = []
    counts = {}
    for col in df.columns:
        col_str = str(col)
        if col_str in counts:
            counts[col_str] += 1
            cols.append(f"{col_str}_{counts[col_str]}")
        else:
            counts[col_str] = 0
            cols.append(col_str)
    df_copy = df.copy()
    df_copy.columns = cols
    return df_copy

def detect_column_types(df: pd.DataFrame):
    """
    Detects the logical data types of all columns in the DataFrame.
    Categorizes them into: 'id', 'numerical', 'categorical', 'text_id', 'datetime', 'unsupported'.
    Returns a dictionary of column name -> type category.
    """
    col_types = {}
    for col in df.columns:
        series = df[col]
        # Drop nulls for checking
        non_null = series.dropna()
        if len(non_null) == 0:
            col_types[col] = 'unsupported'
            continue
        
        # 1. Check if it is an ID column
        col_lower = str(col).lower().strip()
        is_id_name = (
            col_lower in ['id', 'rowid', 'row_id', 'index', 'uuid', 'guid', 'pk', 'sk', 'roll_id'] or
            col_lower.endswith('_id') or col_lower.endswith('id') or
            col_lower.endswith('_key') or col_lower.endswith('key') or
            col_lower.endswith('_uuid') or col_lower.endswith('uuid') or
            col_lower.startswith('id_')
        )
        
        is_id_values = False
        if len(non_null) >= 5 and non_null.nunique() == len(non_null):
            # Check for sequential integers (monotonic sequence with step 1)
            if pd.api.types.is_numeric_dtype(series):
                try:
                    sorted_vals = np.sort(non_null.values)
                    if np.all(np.diff(sorted_vals) == 1):
                        is_id_values = True
                except Exception:
                    pass
            # Check if strings are typical hash/uuid/serial values
            elif series.dtype == 'object':
                try:
                    sample = non_null.head(10)
                    is_hash = all(isinstance(x, str) and (len(x) > 5) for x in sample)
                    if is_hash:
                        is_id_values = True
                except Exception:
                    pass
                    
        if is_id_name or is_id_values:
            col_types[col] = 'id'
            continue

        # 2. If already datetime-like
        if pd.api.types.is_datetime64_any_dtype(series):
            col_types[col] = 'datetime'
            continue
            
        # 3. Check if object/string looks like datetime
        if series.dtype == 'object' or isinstance(series.dtype, pd.CategoricalDtype):
            # Try parsing a subset to speed up
            sample_size = min(len(non_null), 200)
            sample = non_null.sample(sample_size, random_state=42)
            try:
                # Convert to string and try to parse
                parsed = pd.to_datetime(sample.astype(str), errors='coerce')
                # If a substantial portion parses and has reasonable year values
                parsed_rate = parsed.notnull().sum() / sample_size
                if parsed_rate > 0.8:
                    # Let's double check if they aren't just single numbers like "1", "2"
                    # which pandas to_datetime converts to 1970 dates
                    is_numeric_like = all(str(x).replace('.','',1).isdigit() for x in sample)
                    if not is_numeric_like:
                        col_types[col] = 'datetime'
                        continue
            except Exception:
                pass

        # 4. Check if numeric
        # First, if it can be fully converted to numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it has float/int types
            # Note: boolean can be treated as categorical
            if series.dtype == 'bool':
                col_types[col] = 'categorical'
            else:
                col_types[col] = 'numerical'
            continue
            
        # Try converting string object to numeric if possible
        try:
            converted = pd.to_numeric(non_null, errors='coerce')
            if converted.notnull().sum() / len(non_null) > 0.9: # >90% convertable to numeric
                col_types[col] = 'numerical'
                continue
        except Exception:
            pass

        # 5. Categorical vs Text/ID
        unique_count = non_null.nunique()
        if unique_count <= 20:
            col_types[col] = 'categorical'
        else:
            col_types[col] = 'text_id'
            
    return col_types

def ensure_dir(path: str):
    """Ensures that a directory exists."""
    os.makedirs(path, exist_ok=True)
