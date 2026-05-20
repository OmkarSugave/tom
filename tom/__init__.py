import os
import sys

# Reconfigure stdout/stderr to UTF-8 to prevent encoding issues on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Global package state
_active_df = None
_active_path = None

console = Console()

def set_active_df(df: pd.DataFrame, path: str = None):
    global _active_df, _active_path
    _active_df = df
    if path:
        _active_path = path

def get_active_df() -> pd.DataFrame:
    global _active_df
    return _active_df

def get_active_path() -> str:
    global _active_path
    return _active_path

# Public API Functions
from tom.loader import file
from tom.reporter import describe
from tom.charts import chart

def clean_report(df: pd.DataFrame = None):
    """Shows only the data cleaning summary for the active or provided dataset."""
    if df is None:
        df = _active_df
    if df is None:
        rprint("[bold red]❌ Error: No dataset loaded. Run mos.file() first.[/bold red]")
        return
        
    from tom.cleaner import clean_data, print_clean_report
    df_cleaned, summary_info, warnings, dup_rows_removed = clean_data(df)
    print_clean_report(summary_info, warnings, dup_rows_removed)

def correlation(df: pd.DataFrame = None):
    """Computes correlation matrix and shows heatmap + Rich table."""
    if df is None:
        df = _active_df
    if df is None:
        rprint("[bold red]❌ Error: No dataset loaded. Run mos.file() first.[/bold red]")
        return
        
    import tom.utils as utils
    col_types = utils.detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == 'numerical']
    
    if len(num_cols) < 2:
        rprint("[bold yellow]⚠️ Not enough numerical columns to compute correlations.[/bold yellow]")
        return
        
    corr_matrix = df[num_cols].corr()
    
    rprint("\n[bold cyan]🔗 Pearson Correlation Matrix[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Feature", style="bold white")
    for col in num_cols:
        table.add_column(col, justify="right")
        
    for index, row in corr_matrix.iterrows():
        row_vals = [index] + [f"{val:.2f}" if pd.notna(val) else "N/A" for val in row]
        table.add_row(*row_vals)
        
    console.print(table)
    
    # Save correlation heatmap plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    
    utils.ensure_dir("./tom_report/charts")
    path = "./tom_report/charts/single_correlation_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    rprint(f"[bold green]✅ Heatmap saved to: {os.path.abspath(path)}[/bold green]")

def outliers(df: pd.DataFrame = None):
    """Generates a detailed outlier report for all numerical columns."""
    if df is None:
        df = _active_df
    if df is None:
        rprint("[bold red]❌ Error: No dataset loaded. Run mos.file() first.[/bold red]")
        return
        
    import tom.utils as utils
    col_types = utils.detect_column_types(df)
    num_cols = [c for c, t in col_types.items() if t == 'numerical']
    
    if not num_cols:
        rprint("[bold yellow]⚠️ No numerical columns found in the dataset.[/bold yellow]")
        return
        
    rprint("\n[bold cyan]🚨 Detailed Outlier Analysis (IQR Method)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Column", style="bold white")
    table.add_column("Lower Limit", justify="right")
    table.add_column("Upper Limit", justify="right")
    table.add_column("Min Outlier", justify="right", style="red")
    table.add_column("Max Outlier", justify="right", style="red")
    table.add_column("Outlier Count", justify="right", style="magenta")
    table.add_column("Outliers %", justify="right", style="bold red")
    
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_series = series[(series < lower) | (series > upper)]
            count = len(outlier_series)
            pct = (count / len(series)) * 100
            
            min_out = f"{outlier_series.min():.2f}" if count > 0 else "-"
            max_out = f"{outlier_series.max():.2f}" if count > 0 else "-"
            
            table.add_row(
                col,
                f"{lower:.2f}",
                f"{upper:.2f}",
                min_out,
                max_out,
                str(count),
                f"{pct:.1f}%"
            )
            
    console.print(table)

def suggest(df: pd.DataFrame = None):
    """Analyzes the loaded dataset characteristics and suggests suitable ML models."""
    if df is None:
        df = _active_df
    if df is None:
        rprint("[bold red]❌ Error: No dataset loaded. Run mos.file() first.[/bold red]")
        return
        
    rprint("\n[bold cyan]🤖 Autonomous Machine Learning Model Suggestions[/bold cyan]")
    n_rows, n_cols = df.shape
    
    import tom.utils as utils
    col_types = utils.detect_column_types(df)
    
    num_c = sum(1 for t in col_types.values() if t == 'numerical')
    cat_c = sum(1 for t in col_types.values() if t == 'categorical')
    
    # Simple target column heuristic (prefer low-cardinality categorical, or last column)
    target_col = df.columns[-1]
    for col, ctype in col_types.items():
        if ctype == 'categorical' and col != df.columns[0]:
            target_col = col
            break
            
    target_type = col_types.get(target_col, 'numerical')
    
    # Construct beautiful suggestion card
    suggestion_text = f"[bold white]Target Column Detected:[/bold white] `{target_col}` (Type: [bold yellow]{target_type.upper()}[/bold yellow])\n\n"
    
    if target_type in ['categorical', 'text_id'] and df[target_col].nunique() <= 20:
        suggestion_text += "[bold green]🎯 Task Classification Suggested[/bold green]\n"
        suggestion_text += "Since your target column is categorical with discrete classes.\n\n"
        suggestion_text += "[bold white]Candidate Models to Try:[/bold white]\n"
        suggestion_text += "  1. [bold cyan]Random Forest Classifier[/bold cyan] (Great baseline, robust to scaling & outliers)\n"
        suggestion_text += "  2. [bold cyan]XGBoost / LightGBM[/bold cyan] (State-of-the-art for tabular data, handles missing values)\n"
        suggestion_text += "  3. [bold cyan]Logistic Regression[/bold cyan] (Good for linear interpretability)\n\n"
        suggestion_text += "[bold white]Pre-processing Advice:[/bold white]\n"
        suggestion_text += f"  - Apply Target Encoding or One-Hot Encoding to the {cat_c} categorical features.\n"
        if num_c > 0:
            suggestion_text += "  - Standardize/Normalize numeric fields if utilizing Logistic Regression."
    else:
        suggestion_text += "[bold green]📈 Task Regression Suggested[/bold green]\n"
        suggestion_text += "Since your target column is continuous / numerical.\n\n"
        suggestion_text += "[bold white]Candidate Models to Try:[/bold white]\n"
        suggestion_text += "  1. [bold cyan]Random Forest Regressor[/bold cyan] (Captures non-linear dependencies cleanly)\n"
        suggestion_text += "  2. [bold cyan]Ridge Regression / Lasso[/bold cyan] (Excellent linear model with regularization)\n"
        suggestion_text += "  3. [bold cyan]Gradient Boosting Regressor[/bold cyan] (High prediction accuracy)\n\n"
        suggestion_text += "[bold white]Pre-processing Advice:[/bold white]\n"
        suggestion_text += "  - Address severe right/left-skewed features with log/Box-Cox transform.\n"
        suggestion_text += "  - Remove outliers or use robust scaling techniques."
        
    console.print(Panel(suggestion_text, title="🧠 ML Model Suggestions", border_style="purple"))

def compare(df2: pd.DataFrame, df1: pd.DataFrame = None):
    """Compares the active dataset side-by-side with another DataFrame."""
    if df1 is None:
        df1 = _active_df
    if df1 is None:
        rprint("[bold red]❌ Error: Active dataset is missing. Run mos.file() first.[/bold red]")
        return
        
    rprint("\n[bold cyan]🔄 Side-by-Side Dataset Comparison[/bold cyan]")
    
    table = Table(box=None, header_style="bold magenta")
    table.add_column("Metric", style="bold white")
    table.add_column("Dataset 1 (Active)", style="cyan")
    table.add_column("Dataset 2", style="green")
    
    table.add_row("Dimensions (Rows, Cols)", f"{df1.shape}", f"{df2.shape}")
    table.add_row("Total Missing Values", f"{df1.isnull().sum().sum()}", f"{df2.isnull().sum().sum()}")
    table.add_row("Duplicate Rows", f"{df1.duplicated().sum()}", f"{df2.duplicated().sum()}")
    
    # Overlapping columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    overlapping = cols1.intersection(cols2)
    unique_to_1 = cols1 - cols2
    unique_to_2 = cols2 - cols1
    
    table.add_row("Overlapping Columns", str(len(overlapping)), str(len(overlapping)))
    table.add_row("Unique Columns", str(len(unique_to_1)), str(len(unique_to_2)))
    
    console.print(table)
    
    if overlapping:
        rprint(f"[bold yellow]Common Columns:[/bold yellow] {', '.join(f'`{c}`' for c in list(overlapping)[:8])}...")
    if unique_to_1:
        rprint(f"[bold yellow]Unique to Dataset 1:[/bold yellow] {', '.join(f'`{c}`' for c in list(unique_to_1)[:8])}...")
    if unique_to_2:
        rprint(f"[bold yellow]Unique to Dataset 2:[/bold yellow] {', '.join(f'`{c}`' for c in list(unique_to_2)[:8])}...")

def export(format_type: str = "pdf"):
    """
    Exports the generated HTML report as PDF.
    Attempts to use pdfkit if available; falls back gracefully.
    """
    format_type = format_type.lower()
    html_path = "./tom_report/report.html"
    
    if not os.path.exists(html_path):
        rprint("[bold red]❌ Error: Generate a report first by running mos.describe().[/bold red]")
        return
        
    if format_type == "pdf":
        try:
            import pdfkit
            pdf_path = "./tom_report/report.pdf"
            rprint("[bold cyan]Generating PDF Report using pdfkit...[/bold cyan]")
            pdfkit.from_file(html_path, pdf_path)
            rprint(f"[bold green]🎉 Success! Report exported to PDF: {os.path.abspath(pdf_path)}[/bold green]")
        except ImportError:
            rprint("[bold red]❌ Error: 'pdfkit' is not installed.[/bold red]")
            rprint("   [yellow]Please run: pip install pdfkit[/yellow]")
        except Exception as e:
            rprint("[bold red]❌ Failed to export PDF.[/bold red]")
            rprint("   [yellow]Note: pdfkit requires wkhtmltopdf binary installed on your system PATH.[/yellow]")
            rprint("   [yellow]You can alternatively print the HTML report in your browser (Ctrl+P) and Save as PDF![/yellow]")
    else:
        rprint(f"[bold red]❌ Unsupported export format: '{format_type}'[/bold red]")
