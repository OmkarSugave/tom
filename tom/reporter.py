import os
import webbrowser
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from jinja2 import Template
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.tree import Tree
from rich import print as rprint
from rich.align import Align

import tom
import tom.cleaner as cleaner
import tom.stats as stats
import tom.charts as charts
import tom.insights as insights
import tom.utils as utils

console = Console()

# Extremely beautiful modern glassmorphism HTML design template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TOM Automated EDA Report - {{ filename }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 29, 49, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #6366f1;
            --accent-sec: #a855f7;
            --green: #10b981;
            --yellow: #f59e0b;
            --red: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-color) 70%);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            padding: 40px 20px;
        }

        .container {
            max-width: 1300px;
            margin: 0 auto;
        }

        /* Glassmorphic elements */
        .glass-panel {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        header {
            text-align: center;
            margin-bottom: 50px;
        }

        header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .meta-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .meta-card h3 {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .meta-card p {
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--accent-primary);
        }

        /* Tabs/collapsible sections */
        .collapsible-card {
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: 16px;
            border: 1px solid var(--card-border);
            background: rgba(17, 24, 39, 0.6);
        }

        .collapsible-header {
            padding: 20px;
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.2rem;
            font-weight: 600;
            user-select: none;
            transition: background 0.3s;
        }

        .collapsible-header:hover {
            background: rgba(255, 255, 255, 0.05);
        }

        .collapsible-content {
            padding: 25px;
            display: none;
            border-top: 1px solid var(--card-border);
        }

        .collapsible-card.active .collapsible-content {
            display: block;
        }

        .collapsible-header::after {
            content: '▼';
            font-size: 0.8rem;
            color: var(--text-secondary);
            transition: transform 0.3s;
        }

        .collapsible-card.active .collapsible-header::after {
            transform: rotate(-180deg);
        }

        /* Insights section styling */
        .insight-list {
            list-style: none;
        }

        .insight-item {
            display: flex;
            align-items: flex-start;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 12px;
            background: rgba(255, 255, 255, 0.02);
            border-left: 5px solid var(--accent-primary);
        }

        .insight-item.critical {
            border-left-color: var(--red);
            background: rgba(239, 68, 68, 0.05);
        }

        .insight-item.warning {
            border-left-color: var(--yellow);
            background: rgba(245, 158, 11, 0.05);
        }

        .insight-item.info {
            border-left-color: var(--green);
            background: rgba(16, 185, 129, 0.05);
        }

        .insight-badge {
            font-size: 1.2rem;
            margin-right: 15px;
        }

        .insight-text {
            font-size: 1rem;
        }

        /* Visual grids */
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }

        .chart-container {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .chart-title {
            margin-bottom: 12px;
            font-size: 1.05rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        /* Tables styling */
        .table-responsive {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.95rem;
        }

        th {
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-secondary);
            font-weight: 600;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.08);
        }

        td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.numeric { background: #3b82f6; color: white; }
        .badge.categorical { background: #10b981; color: white; }
        .badge.datetime { background: #8b5cf6; color: white; }
        .badge.text_id { background: #f59e0b; color: white; }

        .dashboard-btn {
            display: inline-block;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            margin-top: 20px;
            transition: opacity 0.3s;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .dashboard-btn:hover {
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="glass-panel">
            <h1>🚀 TOM Automated EDA Report</h1>
            <p>Single-Line Exploratory Data Analysis & Diagnostic Summary</p>
            
            <div class="meta-grid">
                <div class="meta-card">
                    <h3>Dataset File</h3>
                    <p style="font-size: 1.1rem; padding-top: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{{ filename }}</p>
                </div>
                <div class="meta-card">
                    <h3>Total Rows</h3>
                    <p>{{ rows }}</p>
                </div>
                <div class="meta-card">
                    <h3>Total Columns</h3>
                    <p>{{ cols }}</p>
                </div>
                <div class="meta-card">
                    <h3>Cleaned Alerts</h3>
                    <p style="color: var(--yellow);">{{ alerts_count }}</p>
                </div>
            </div>
            
            {% if interactive_dashboard_path %}
            <a href="charts/plotly_dashboard.html" target="_blank" class="dashboard-btn">🌐 View Interactive Plotly Dashboard</a>
            {% endif %}
        </header>

        <!-- Insights -->
        <div class="glass-panel">
            <h2 style="margin-bottom: 20px;">💡 Natural Language Insights</h2>
            <ul class="insight-list">
                {% for insight in insights %}
                <li class="insight-item {{ insight.severity }}">
                    <span class="insight-badge">
                        {% if insight.severity == 'critical' %}🔴
                        {% elif insight.severity == 'warning' %}🟡
                        {% else %}🟢{% endif %}
                    </span>
                    <span class="insight-text">{{ insight.text }}</span>
                </li>
                {% endfor %}
            </ul>
        </div>

        <!-- Data Cleaning Summary -->
        <div class="collapsible-card">
            <div class="collapsible-header">🧹 Data Preprocessing & Cleaning Details</div>
            <div class="collapsible-content">
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Original Type</th>
                                <th>Cleaned Type</th>
                                <th>Logical Category</th>
                                <th>Imputed Count</th>
                                <th>Outliers Flagged</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col in cleaning_summary %}
                            <tr>
                                <td><strong>{{ col.column }}</strong></td>
                                <td><code>{{ col.original_type }}</code></td>
                                <td><code>{{ col.cleaned_type }}</code></td>
                                <td><span class="badge {{ col.logical_type }}">{{ col.logical_type }}</span></td>
                                <td>{{ col.missing_filled }} ({{ "%.1f"|format(col.missing_pct) }}%)</td>
                                <td>{{ "%.1f"|format(col.outlier_pct) }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Data Statistics Details -->
        <div class="collapsible-card">
            <div class="collapsible-header">📊 Feature Statistics & Engine Computations</div>
            <div class="collapsible-content">
                <h3 style="margin-bottom: 15px; color: var(--accent-primary);">Numerical Variables</h3>
                {% for col, val in stats.numerical.items() %}
                <div class="collapsible-card" style="margin-top: 10px;">
                    <div class="collapsible-header" style="font-size: 1rem; padding: 12px 20px; background: rgba(255,255,255,0.01);">🔍 {{ col }} Metrics</div>
                    <div class="collapsible-content" style="padding: 15px;">
                        <div class="meta-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 0;">
                            <div class="meta-card" style="padding: 10px;">
                                <h3 style="font-size: 0.8rem;">Mean</h3>
                                <p style="font-size: 1.1rem;">{{ "%.2f"|format(val.mean) }}</p>
                            </div>
                            <div class="meta-card" style="padding: 10px;">
                                <h3 style="font-size: 0.8rem;">Median</h3>
                                <p style="font-size: 1.1rem;">{{ "%.2f"|format(val.median) }}</p>
                            </div>
                            <div class="meta-card" style="padding: 10px;">
                                <h3 style="font-size: 0.8rem;">Standard Dev</h3>
                                <p style="font-size: 1.1rem;">{{ "%.2f"|format(val.std) }}</p>
                            </div>
                            <div class="meta-card" style="padding: 10px;">
                                <h3 style="font-size: 0.8rem;">Skewness</h3>
                                <p style="font-size: 1.1rem;">{{ "%.2f"|format(val.skew) }}</p>
                            </div>
                            <div class="meta-card" style="padding: 10px;">
                                <h3 style="font-size: 0.8rem;">Normality</h3>
                                <p style="font-size: 1.1rem; color: var(--green);">{{ val.normality }}</p>
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
                
                <h3 style="margin-top: 30px; margin-bottom: 15px; color: var(--accent-primary);">Categorical Variables</h3>
                {% for col, val in stats.categorical.items() %}
                <div class="collapsible-card" style="margin-top: 10px;">
                    <div class="collapsible-header" style="font-size: 1rem; padding: 12px 20px; background: rgba(255,255,255,0.01);">🔍 {{ col }} Metrics</div>
                    <div class="collapsible-content" style="padding: 15px;">
                        <p><strong>Unique Categories:</strong> {{ val.unique_count }} | <strong>Mode:</strong> {{ val.mode }} ({{ "%.1f"|format(val.mode_pct) }}%) | <strong>Entropy:</strong> {{ "%.3f"|format(val.entropy) }}</p>
                        <h4 style="margin-top: 10px; font-size: 0.9rem; color: var(--text-secondary);">Top 10 Categories:</h4>
                        <div class="table-responsive">
                            <table style="font-size: 0.85rem; margin-top: 5px;">
                                <thead>
                                    <tr>
                                        <th>Category</th>
                                        <th>Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for k, v in val.top_10.items() %}
                                    <tr>
                                        <td>{{ k }}</td>
                                        <td>{{ v }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Static Visualizations (Self-Contained base64 Images) -->
        <div class="glass-panel">
            <h2>📈 Data Visualizations</h2>
            <p style="color: var(--text-secondary); margin-bottom: 25px;">Self-contained static charts exported at high-fidelity.</p>
            
            {% if images.missing_heatmap %}
            <div class="chart-container" style="margin-bottom: 30px;">
                <div class="chart-title">Dataset Missing Value Heatmap</div>
                <img src="data:image/png;base64,{{ images.missing_heatmap }}" alt="Missing Value Heatmap">
            </div>
            {% endif %}
            
            {% if images.distributions_grid %}
            <div class="chart-container" style="margin-bottom: 30px;">
                <div class="chart-title">Numerical Distributions Grid</div>
                <img src="data:image/png;base64,{{ images.distributions_grid }}" alt="Distributions Grid">
            </div>
            {% endif %}
            
            {% if images.outliers_dotplot %}
            <div class="chart-container" style="margin-bottom: 30px;">
                <div class="chart-title">Outliers Standardized Dot Plot</div>
                <img src="data:image/png;base64,{{ images.outliers_dotplot }}" alt="Outlier Dot Plot">
            </div>
            {% endif %}
            
            {% if images.correlation_heatmap %}
            <div class="chart-container" style="margin-bottom: 30px;">
                <div class="chart-title">Numerical Correlation Heatmap</div>
                <img src="data:image/png;base64,{{ images.correlation_heatmap }}" alt="Correlation Heatmap">
            </div>
            {% endif %}
            
            <div class="chart-grid">
                <!-- Single Numerical Charts -->
                {% for col, types in images.numerical.items() %}
                    {% if types.histogram %}
                    <div class="chart-container">
                        <div class="chart-title">{{ col }} Histogram & KDE</div>
                        <img src="data:image/png;base64,{{ types.histogram }}" alt="{{ col }} histogram">
                    </div>
                    {% endif %}
                    {% if types.boxplot %}
                    <div class="chart-container">
                        <div class="chart-title">{{ col }} Box Plot</div>
                        <img src="data:image/png;base64,{{ types.boxplot }}" alt="{{ col }} boxplot">
                    </div>
                    {% endif %}
                {% endfor %}

                <!-- Single Categorical Charts -->
                {% for col, types in images.categorical.items() %}
                    {% if types.barchart %}
                    <div class="chart-container">
                        <div class="chart-title">{{ col }} Category Share Bar Plot</div>
                        <img src="data:image/png;base64,{{ types.barchart }}" alt="{{ col }} barchart">
                    </div>
                    {% endif %}
                {% endfor %}

                <!-- Numerical Categorical Interactions -->
                {% for pair in images.num_cat %}
                <div class="chart-container">
                    <div class="chart-title">{{ pair.num }} grouped by {{ pair.cat }}</div>
                    <img src="data:image/png;base64,{{ pair.base64 }}" alt="grouped boxplot">
                </div>
                {% endfor %}

                <!-- DateTime Trend Charts -->
                {% for pair in images.date_num %}
                <div class="chart-container">
                    <div class="chart-title">{{ pair.num }} trend over time ({{ pair.date }})</div>
                    <img src="data:image/png;base64,{{ pair.base64 }}" alt="trend plot">
                </div>
                {% endfor %}
            </div>
        </div>
    </div>

    <!-- Toggle logic script -->
    <script>
        document.querySelectorAll('.collapsible-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('active');
            });
        });
    </script>
</body>
</html>
"""

def describe(df: pd.DataFrame = None):
    """
    Core function: Auto-cleans data, computes stats, renders terminal panel summary,
    saves high-fidelity PNG plots, generates portable HTML report, and triggers auto-open in browser.
    """
    # 1. Fetch from active state if df is not provided
    if df is None:
        df = tom.get_active_df()
        
    if df is None:
        rprint("[bold red]❌ Error: No dataset is loaded into the TOM state.[/bold red]")
        rprint("   [yellow]Please load a file first using: om.file('filename.csv')[/yellow]")
        return None

    filename = os.path.basename(tom.get_active_path() or "active_dataframe")
    rprint(f"\n[bold cyan]🚀 Starting TOM Automated EDA Diagnostics for: {filename}[/bold cyan]")
    
    # 2. Trigger Cleaner
    df_cleaned, summary_info, warnings, dup_rows_removed = cleaner.clean_data(df)
    
    # Auto-update active dataframe state with the cleaned DataFrame for future requests!
    tom.set_active_df(df_cleaned, tom.get_active_path())
    
    # Print cleaner report
    cleaner.print_clean_report(summary_info, warnings, dup_rows_removed)
    
    # Detect types of cleaned columns
    col_types = utils.detect_column_types(df_cleaned)
    
    # 3. Compute Stats
    stats_out = stats.compute_all_stats(df_cleaned, col_types)
    
    # 4. Generate Visualizations (Static PNG + Interactive Plotly Dashboard)
    rprint("\n[bold cyan]🎨 Rendering and exporting data visualizations...[/bold cyan]")
    chart_paths, base64_images, chart_warnings = charts.generate_all_charts(df_cleaned, col_types)
    for cw in chart_warnings:
        warnings.append(cw)
        
    # 5. Generate Insights
    rprint("[bold cyan]💡 Running statistical engine & extracting NLP insights...[/bold cyan]")
    nlp_insights = insights.generate_insights(df_cleaned, col_types, stats_out, summary_info)
    
    # 6. Render Terminal Report using rich
    render_terminal_dashboard(filename, df_cleaned, summary_info, stats_out, nlp_insights)
    
    # 7. Compile HTML Report using Jinja2
    rprint("\n[bold cyan]📄 Assembling portable HTML report...[/bold cyan]")
    
    # Check if Plotly dashboard exists to display link in header
    has_plotly = 'plotly_dashboard' in chart_paths
    
    html_content = Template(HTML_TEMPLATE).render(
        filename=filename,
        rows=f"{df_cleaned.shape[0]:,}",
        cols=df_cleaned.shape[1],
        alerts_count=len(warnings),
        insights=nlp_insights,
        cleaning_summary=summary_info,
        stats=stats_out,
        images=base64_images,
        interactive_dashboard_path=has_plotly
    )
    
    report_dir = "./tom_report"
    utils.ensure_dir(report_dir)
    report_path = os.path.join(report_dir, "report.html")
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        rprint(f"[bold green]✨ Portable Glassmorphic HTML Report created: {os.path.abspath(report_path)}[/bold green]")
        
        # 8. Auto-open browser
        webbrowser.open(f"file:///{os.path.abspath(report_path)}")
    except Exception as e:
        rprint(f"[bold red]❌ Failed to save or open the HTML report: {str(e)}[/bold red]")
        
    return df_cleaned

def render_terminal_dashboard(filename: str, df: pd.DataFrame, summary_info: list, stats_out: dict, nlp_insights: list):
    rprint("\n" + "="*80)
    console.print(Align.center(f"[bold magenta]📊 TOM EDA DIAGNOSTICS: {filename}[/bold magenta]"))
    rprint("="*80 + "\n")
    
    # A. Dataset overview panel
    overview_table = Table(box=None, padding=(0, 2))
    overview_table.add_column("Rows", style="cyan")
    overview_table.add_column("Columns", style="cyan")
    overview_table.add_column("Numeric Features", style="green")
    overview_table.add_column("Categorical Features", style="yellow")
    overview_table.add_column("Datetime Features", style="purple")
    
    num_c = len(stats_out['numerical'])
    cat_c = len(stats_out['categorical'])
    date_c = len(stats_out['datetime'])
    
    overview_table.add_row(f"{df.shape[0]:,}", str(df.shape[1]), str(num_c), str(cat_c), str(date_c))
    
    console.print(Panel(overview_table, title="[bold white]📂 Dataset Overview[/bold white]", border_style="blue"))
    
    # B. Statistical tree summary
    tree = Tree("[bold cyan]📊 Mathematical Summary Metrics[/bold cyan]")
    
    if stats_out['numerical']:
        num_branch = tree.add("[green]Numerical Features[/green]")
        for col, s in stats_out['numerical'].items():
            num_branch.add(f"[bold white]{col}[/bold white]: Mean={s.get('mean',0):.2f} | Med={s.get('median',0):.2f} | Std={s.get('std',0):.2f} | Normality=[yellow]{s.get('normality')}[/yellow]")
            
    if stats_out['categorical']:
        cat_branch = tree.add("[yellow]Categorical Features[/yellow]")
        for col, s in stats_out['categorical'].items():
            cat_branch.add(f"[bold white]{col}[/bold white]: Unique={s.get('unique_count')} | Mode='{s.get('mode')}' ({s.get('mode_pct',0):.1f}%)")
            
    if stats_out['datetime']:
        date_branch = tree.add("[purple]Datetime Features[/purple]")
        for col, s in stats_out['datetime'].items():
            date_branch.add(f"[bold white]{col}[/bold white]: Range={s.get('min')} to {s.get('max')} | Trend=[magenta]{s.get('trend')}[/magenta]")
            
    console.print(tree)
    rprint("")
    
    # C. Strong Correlations Highlight
    top_corrs = stats_out['dataset'].get('top_correlations', {})
    if top_corrs and (top_corrs.get('positive') or top_corrs.get('negative')):
        corr_table = Table(title="🔗 Strong Correlation Highlights", box=None, header_style="bold magenta")
        corr_table.add_column("Variable A", style="cyan")
        corr_table.add_column("Variable B", style="cyan")
        corr_table.add_column("Pearson Coefficient (r)", justify="right")
        corr_table.add_column("Relationship Strength", style="bold yellow")
        
        for pair, val in top_corrs.get('positive', [])[:3]:
            strength = "Extremely Strong" if val > 0.9 else "Strong" if val > 0.7 else "Moderate"
            corr_table.add_row(pair[0], pair[1], f"+{val:.2f}", f"[green]{strength}[/green]")
            
        for pair, val in top_corrs.get('negative', [])[:3]:
            strength = "Extremely Strong" if val < -0.9 else "Strong" if val < -0.7 else "Moderate"
            corr_table.add_row(pair[0], pair[1], f"{val:.2f}", f"[red]{strength}[/red]")
            
        console.print(corr_table)
        rprint("")
        
    # D. Key Insights list (Top 8 to avoid overwhelming console)
    insight_panel_text = ""
    for idx, insight in enumerate(nlp_insights[:10]):
        badge = "🔴" if insight['severity'] == 'critical' else "🟡" if insight['severity'] == 'warning' else "🟢"
        insight_panel_text += f"{badge} [bold white]{idx+1}.[/bold white] {insight['text']}\n"
        
    console.print(Panel(insight_panel_text.strip(), title="[bold white]💡 NLP Analytics Insights (Top 10)[/bold white]", border_style="purple"))
