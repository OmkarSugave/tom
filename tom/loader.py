import os
import sys
import pandas as pd
import difflib
from rich import print as rprint
from rich.console import Console

# We'll import a global reference to access the active dataframe
import tom

console = Console()

def file(path=None):
    """
    Loads a file into the tom active state.
    
    Parameters:
    path (str, optional): The file path to load. If None, prompts interactively.
    """
    if path is None:
        try:
            path = input("📂 Enter file path: ").strip()
        except KeyboardInterrupt:
            rprint("\n[bold red]❌ Load cancelled by user.[/bold red]")
            return None

    if not path:
        rprint("[bold red]❌ Error: No file path provided.[/bold red]")
        return None

    # Handle standard string path sanitization
    path = path.strip('\'"')

    if not os.path.exists(path):
        # Implement friendly check for similar files
        dir_name = os.path.dirname(path) or "."
        base_name = os.path.basename(path)
        
        try:
            all_files = os.listdir(dir_name)
            matches = difflib.get_close_matches(base_name, all_files, n=1, cutoff=0.5)
            if matches:
                suggested = os.path.join(dir_name, matches[0])
                rprint(f"[bold red]❌ File not found: '{path}'[/bold red]")
                rprint(f"   [yellow]Did you mean: '{suggested}'? Check the path and try again.[/yellow]")
            else:
                rprint(f"[bold red]❌ File not found: '{path}'[/bold red]\n   [yellow]Please check the path and try again.[/yellow]")
        except Exception:
            rprint(f"[bold red]❌ File not found: '{path}'[/bold red]")
        return None

    ext = os.path.splitext(path)[1].lower()
    df = None
    
    try:
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif ext == '.json':
            df = pd.read_json(path)
        elif ext == '.parquet':
            df = pd.read_parquet(path)
        elif ext in ['.tsv', '.txt']:
            # Automatically detect separator for txt/tsv using engine='python' and sep=None
            df = pd.read_csv(path, sep=None, engine='python')
        else:
            # Fallback text-based delimiter detection
            try:
                df = pd.read_csv(path, sep=None, engine='python')
            except Exception:
                raise ValueError(f"Unsupported file format '{ext}'")

    except Exception as e:
        rprint(f"[bold red]❌ Failed to load file: {path}[/bold red]")
        rprint(f"   [yellow]Error: {str(e)}[/yellow]")
        return None

    if df is not None:
        # Save to tom package level variables
        tom.set_active_df(df, path)
        rprint(f"[bold green]✅ File loaded: {os.path.basename(path)} | Shape: {df.shape}[/bold green]")
        return df
