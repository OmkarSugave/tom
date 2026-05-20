import pandas as pd
import numpy as np
import os

def create_mock_dataset():
    """Generates a highly diverse mock dataset to test all tom library modules."""
    np.random.seed(42)
    n_rows = 1000
    
    # 1. Numerical columns
    # A. Normal distribution
    age = np.random.normal(35, 10, n_rows).round(0)
    # B. Heavily skewed column
    income = np.random.exponential(50000, n_rows).round(2)
    # C. Column with significant outliers
    savings = np.random.normal(5000, 1000, n_rows)
    # inject outliers manually
    outlier_idx = np.random.choice(n_rows, 50, replace=False)
    savings[outlier_idx] = savings[outlier_idx] * 5
    savings = savings.round(2)
    
    # 2. Categorical columns
    # A. Low cardinality category (domination check)
    gender = np.random.choice(['Female', 'Male', 'Non-binary'], n_rows, p=[0.75, 0.20, 0.05])
    # B. Moderate cardinality category
    country = np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'India', 'Japan', 'Brazil'], n_rows)
    
    # 3. High cardinality text/id-like column
    user_id = [f"USR_{i:04d}" for i in range(1, n_rows + 1)]
    
    # 4. Datetime column
    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="h")
    
    # Assemble dataframe
    df = pd.DataFrame({
        'user_id': user_id,
        'date_registered': dates,
        'age': age,
        'gender': gender,
        'income': income,
        'savings': savings,
        'country': country
    })
    
    # Inject missing values
    # Income -> 25% missing
    df.loc[df.sample(frac=0.25).index, 'income'] = np.nan
    # Country -> 5% missing
    df.loc[df.sample(frac=0.05).index, 'country'] = np.nan
    # High missing rate column (should be dropped, missing rate > 50%)
    df['unwanted_col'] = pd.Series([np.nan] * len(df), dtype='object')
    df.loc[df.sample(frac=0.3).index, 'unwanted_col'] = 'junk'
    
    # Introduce duplicate rows (say, 5 duplicates)
    dups = df.iloc[:5].copy()
    df = pd.concat([df, dups], ignore_index=True)
    
    # Create duplicate column names to test utils renaming
    df_with_dups = df.copy()
    # Adding a duplicate column name (renames under the hood)
    df_with_dups['age_dup'] = df['age']
    df_with_dups.columns = list(df.columns) + ['age']
    
    output_path = "mock_data.csv"
    df_with_dups.to_csv(output_path, index=False)
    print(f"Mock data successfully written to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_mock_dataset()
