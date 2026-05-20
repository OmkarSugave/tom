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

import tom as om
from rich import print as rprint

def run_suite():
    rprint("[bold green]🧪 Starting TOM Package Test Suite[/bold green]")
    
    # 1. Load the mock data
    rprint("\n[bold cyan]Step 1: Loading Data[/bold cyan]")
    om.file("mock_data.csv")
    
    # 2. Show clean report
    rprint("\n[bold cyan]Step 2: Testing Clean Report[/bold cyan]")
    om.clean_report()
    
    # 3. Test detailed outlier report
    rprint("\n[bold cyan]Step 3: Testing Outlier Analysis[/bold cyan]")
    om.outliers()
    
    # 4. Test statistical correlations
    rprint("\n[bold cyan]Step 4: Testing Correlation Analysis[/bold cyan]")
    om.correlation()
    
    # 5. Test ML model suggestions
    rprint("\n[bold cyan]Step 5: Testing ML Suggestions[/bold cyan]")
    om.suggest()
    
    # 6. Test single chart generation
    rprint("\n[bold cyan]Step 6: Testing Single Chart Generation[/bold cyan]")
    om.chart("violin", "savings")
    
    # 7. Run full end-to-end report (describe)
    rprint("\n[bold cyan]Step 7: Testing Full End-to-End Describe Pipeline[/bold cyan]")
    om.describe()
    
    rprint("\n[bold green]🎉 TOM Test Suite Completed Successfully![/bold green]")

if __name__ == "__main__":
    run_suite()
