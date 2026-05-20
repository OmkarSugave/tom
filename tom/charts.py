import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Headless mode for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tom.utils as utils

# Set standard styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.max_open_warning'] = 50

def get_base64_from_fig(fig):
    """Converts a Matplotlib figure into a base64 encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_all_charts(df: pd.DataFrame, col_types: dict, output_dir: str = "./tom_report/charts"):
    """
    Generates all appropriate Matplotlib/Seaborn and Plotly charts.
    Saves PNGs (300 DPI) to output_dir/ and exports a combined interactive plotly dashboard.
    
    Returns:
        chart_paths (dict): Paths to saved static charts.
        base64_images (dict): Base64 encoded string representation of charts for HTML embedding.
        warnings (list): Any warnings generated during charting.
    """
    utils.ensure_dir(output_dir)
    chart_paths = {}
    base64_images = {}
    warnings = []
    
    # 1. Row sampling for very large datasets
    chart_df = df
    if len(df) > 500000:
        warnings.append(f"Dataset has {len(df):,} rows. Sampling 100,000 rows for faster visual rendering.")
        chart_df = df.sample(n=100000, random_state=42)
        
    num_cols = [c for c, t in col_types.items() if t == 'numerical' and c in chart_df.columns]
    cat_cols = [c for c, t in col_types.items() if t == 'categorical' and c in chart_df.columns]
    date_cols = [c for c, t in col_types.items() if t == 'datetime' and c in chart_df.columns]
    
    # --- Dataset-Level Charts ---
    # A. Missing Value Heatmap
    if df.isnull().sum().sum() > 0:
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(chart_df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title("Dataset Missing Values Heatmap")
            
            path = os.path.join(output_dir, "dataset_missing_heatmap.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['missing_heatmap'] = path
            
            # Save base64
            base64_images['missing_heatmap'] = get_base64_from_fig(fig)
        except Exception as e:
            warnings.append(f"Failed to generate Missing Heatmap: {str(e)}")
            
    # B. Distribution Overview Grid
    if len(num_cols) > 0:
        try:
            n_cols = len(num_cols)
            grid_cols = min(3, n_cols)
            grid_rows = int(np.ceil(n_cols / grid_cols))
            
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows))
            if grid_rows == 1 and grid_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, col in enumerate(num_cols):
                sns.histplot(chart_df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f"{col} Dist")
                axes[i].set_xlabel("")
                
            # Hide empty axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
                
            fig.suptitle("Numerical Distributions Overview Grid", fontsize=14, y=1.02)
            plt.tight_layout()
            
            path = os.path.join(output_dir, "dataset_distributions_grid.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['distributions_grid'] = path
            base64_images['distributions_grid'] = get_base64_from_fig(fig)
        except Exception as e:
            warnings.append(f"Failed to generate Distribution Grid: {str(e)}")
            
    # C. Outlier Summary Dot Plot
    if len(num_cols) > 0:
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            # Standardize numeric data for side-by-side outlier dot comparison
            std_data = []
            for col in num_cols:
                series = chart_df[col].dropna()
                if series.std() > 0:
                    z_scores = (series - series.mean()) / series.std()
                    for z in z_scores:
                        std_data.append({'Column': col, 'Standardized Score (Z)': z})
            if std_data:
                std_df = pd.DataFrame(std_data)
                sns.stripplot(data=std_df, x='Column', y='Standardized Score (Z)', jitter=0.3, size=2, alpha=0.5, palette='Set2', hue='Column', legend=False, ax=ax)
                ax.axhline(3, color='red', linestyle='--', alpha=0.7, label='Z = +3 threshold')
                ax.axhline(-3, color='red', linestyle='--', alpha=0.7, label='Z = -3 threshold')
                ax.set_title("Outlier Summary Dot Plot (Standardized Scores)")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                path = os.path.join(output_dir, "dataset_outliers_dotplot.png")
                fig.savefig(path, dpi=300, bbox_inches='tight')
                chart_paths['outliers_dotplot'] = path
                base64_images['outliers_dotplot'] = get_base64_from_fig(fig)
        except Exception as e:
            warnings.append(f"Failed to generate Outlier Dot Plot: {str(e)}")

    # --- Column-Level Charts ---
    chart_paths['numerical'] = {}
    base64_images['numerical'] = {}
    chart_paths['categorical'] = {}
    base64_images['categorical'] = {}

    # Numerical Columns
    for col in num_cols:
        series = chart_df[col].dropna()
        if len(series) == 0:
            continue
            
        chart_paths['numerical'][col] = {}
        base64_images['numerical'][col] = {}
        
        # 1. Histogram with KDE
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(series, kde=True, color='teal', ax=ax)
            ax.set_title(f"{col} - Histogram with KDE")
            
            path = os.path.join(output_dir, f"num_{col}_histogram.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['numerical'][col]['histogram'] = path
            base64_images['numerical'][col]['histogram'] = get_base64_from_fig(fig)
        except Exception:
            pass

        # 2. Box Plot
        try:
            fig, ax = plt.subplots(figsize=(6, 2))
            sns.boxplot(x=series, color='coral', ax=ax)
            ax.set_title(f"{col} - Box Plot")
            
            path = os.path.join(output_dir, f"num_{col}_boxplot.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['numerical'][col]['boxplot'] = path
            base64_images['numerical'][col]['boxplot'] = get_base64_from_fig(fig)
        except Exception:
            pass

        # 3. Violin Plot
        try:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.violinplot(x=series, color='orchid', ax=ax)
            ax.set_title(f"{col} - Violin Plot")
            
            path = os.path.join(output_dir, f"num_{col}_violin.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['numerical'][col]['violin'] = path
            base64_images['numerical'][col]['violin'] = get_base64_from_fig(fig)
        except Exception:
            pass

        # 4. ECDF Plot
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.ecdfplot(series, color='darkblue', ax=ax)
            ax.set_title(f"{col} - Empirical Cumulative Distribution (ECDF)")
            
            path = os.path.join(output_dir, f"num_{col}_ecdf.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['numerical'][col]['ecdf'] = path
            base64_images['numerical'][col]['ecdf'] = get_base64_from_fig(fig)
        except Exception:
            pass

        # 5. Q-Q Plot
        try:
            fig, ax = plt.subplots(figsize=(5, 5))
            stats.probplot(series, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor('gray')
            ax.get_lines()[0].set_markeredgecolor('gray')
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[1].set_color('red')
            ax.set_title(f"{col} - Q-Q Plot")
            
            path = os.path.join(output_dir, f"num_{col}_qqplot.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['numerical'][col]['qqplot'] = path
            base64_images['numerical'][col]['qqplot'] = get_base64_from_fig(fig)
        except Exception:
            pass

    # Categorical Columns
    for col in cat_cols:
        series = chart_df[col].dropna()
        if len(series) == 0:
            continue
            
        chart_paths['categorical'][col] = {}
        base64_images['categorical'][col] = {}
        val_counts = series.value_counts()
        
        # 1. Bar Chart (Value Counts)
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=val_counts.index[:10], y=val_counts.values[:10], palette='viridis', hue=val_counts.index[:10], legend=False, ax=ax)
            ax.set_title(f"{col} - Top 10 Category Counts")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            path = os.path.join(output_dir, f"cat_{col}_barchart.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['categorical'][col]['barchart'] = path
            base64_images['categorical'][col]['barchart'] = get_base64_from_fig(fig)
        except Exception:
            pass

        # 2. Pie Chart (if <= 8 categories)
        if len(val_counts) <= 8:
            try:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(val_counts.values, labels=val_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel', len(val_counts)))
                ax.set_title(f"{col} - Category Share")
                
                path = os.path.join(output_dir, f"cat_{col}_piechart.png")
                fig.savefig(path, dpi=300, bbox_inches='tight')
                chart_paths['categorical'][col]['piechart'] = path
                base64_images['categorical'][col]['piechart'] = get_base64_from_fig(fig)
            except Exception:
                pass

        # 3. Horizontal Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=val_counts.values[:10], y=val_counts.index[:10], palette='magma', hue=val_counts.index[:10], legend=False, ax=ax)
            ax.set_title(f"{col} - Horizontal Category Counts")
            
            path = os.path.join(output_dir, f"cat_{col}_hbar.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['categorical'][col]['hbar'] = path
            base64_images['categorical'][col]['hbar'] = get_base64_from_fig(fig)
        except Exception:
            pass

    # --- Pairwise and Grouped Charts ---
    # A. Pair Plot
    if len(num_cols) > 1:
        try:
            # Limit columns to 5 for speed
            limit_cols = num_cols[:5]
            pair_grid = sns.pairplot(chart_df[limit_cols], diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
            pair_grid.figure.suptitle("Numerical Scatter Plot Matrix (Pairplot)", y=1.02, fontsize=14)
            
            path = os.path.join(output_dir, "dataset_pairplot.png")
            pair_grid.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['pairplot'] = path
            base64_images['pairplot'] = get_base64_from_fig(pair_grid.figure)
        except Exception as e:
            warnings.append(f"Failed to generate Pairplot: {str(e)}")

    # B. Correlation Heatmap
    if len(num_cols) > 1:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(chart_df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=ax)
            ax.set_title("Numerical Pearson Correlation Heatmap")
            
            path = os.path.join(output_dir, "dataset_correlation_heatmap.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            chart_paths['correlation_heatmap'] = path
            base64_images['correlation_heatmap'] = get_base64_from_fig(fig)
        except Exception as e:
            warnings.append(f"Failed to generate Correlation Heatmap: {str(e)}")

    # C. Numerical + Categorical Groupings
    chart_paths['num_cat'] = []
    base64_images['num_cat'] = []
    if len(num_cols) > 0 and len(cat_cols) > 0:
        # We take the top 2 numeric and top 2 categorical columns to avoid generating 100s of plots
        for num_col in num_cols[:2]:
            for cat_col in cat_cols[:2]:
                try:
                    # 1. Grouped Box Plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(data=chart_df, x=cat_col, y=num_col, palette='Set3', hue=cat_col, legend=False, ax=ax)
                    ax.set_title(f"{num_col} Grouped by {cat_col}")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    path = os.path.join(output_dir, f"group_{num_col}_by_{cat_col}_box.png")
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                    chart_paths['num_cat'].append({'num': num_col, 'cat': cat_col, 'type': 'box', 'path': path})
                    base64_images['num_cat'].append({
                        'num': num_col, 'cat': cat_col, 'type': 'box', 
                        'base64': get_base64_from_fig(fig)
                    })
                except Exception:
                    pass

    # D. DateTime + Numerical Trends
    chart_paths['date_num'] = []
    base64_images['date_num'] = []
    if len(date_cols) > 0 and len(num_cols) > 0:
        for date_col in date_cols[:1]:
            for num_col in num_cols[:2]:
                try:
                    # Aggregate by day/month to plot trend
                    trend_df = chart_df[[date_col, num_col]].dropna()
                    trend_df = trend_df.sort_values(by=date_col)
                    # Group by date value
                    trend_grouped = trend_df.groupby(trend_df[date_col].dt.date).mean()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(trend_grouped.index, trend_grouped[num_col], label='Daily Mean', alpha=0.5, color='dodgerblue')
                    
                    # Rolling average if enough data points
                    if len(trend_grouped) > 7:
                        rolling_avg = trend_grouped[num_col].rolling(window=7, min_periods=1).mean()
                        ax.plot(trend_grouped.index, rolling_avg, label='7-Day Rolling Avg', color='red', linewidth=2)
                        
                    ax.set_title(f"{num_col} Trend Over {date_col}")
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(f"Mean {num_col}")
                    ax.legend()
                    fig.autofmt_xdate()
                    
                    path = os.path.join(output_dir, f"trend_{num_col}_over_{date_col}.png")
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                    chart_paths['date_num'].append({'date': date_col, 'num': num_col, 'path': path})
                    base64_images['date_num'].append({
                        'date': date_col, 'num': num_col, 
                        'base64': get_base64_from_fig(fig)
                    })
                except Exception:
                    pass

    # --- Interactive Plotly Dashboard ---
    try:
        # Create a combined dashboard using subplots
        plotly_path = os.path.join(output_dir, "plotly_dashboard.html")
        
        # Build individual plotly figures and combine them into a single file or a multi-panel visual
        # Let's make an interactive visual layout and save it as an HTML snippet or combined page.
        # Alternatively, we can save a super beautiful plotly dashboard with multi-tabs.
        # To do this cleanly, we'll write a small helper to generate individual plotly figures
        # and assemble them into a beautiful layout inside `./tom_report/charts/plotly_dashboard.html`.
        
        dashboard_figs = []
        
        # Distribution plot for first numerical col
        if num_cols:
            f = px.histogram(chart_df, x=num_cols[0], marginal="box", title=f"Distribution of {num_cols[0]}")
            dashboard_figs.append(f.to_html(full_html=False, include_plotlyjs='cdn'))
            
        # Scatter/Correlation Plot for first pair of numeric cols
        if len(num_cols) > 1:
            f = px.scatter(chart_df, x=num_cols[0], y=num_cols[1], trendline="ols", title=f"{num_cols[0]} vs {num_cols[1]}")
            dashboard_figs.append(f.to_html(full_html=False, include_plotlyjs=False))
            
        # Category share
        if cat_cols:
            f = px.bar(chart_df[cat_cols[0]].value_counts().reset_index(), x=cat_cols[0], y='count', title=f"Category Counts of {cat_cols[0]}")
            dashboard_figs.append(f.to_html(full_html=False, include_plotlyjs=False))
            
        # Trend
        if date_cols and num_cols:
            f = px.line(chart_df.sort_values(date_cols[0]), x=date_cols[0], y=num_cols[0], title=f"{num_cols[0]} Time Series Trend")
            dashboard_figs.append(f.to_html(full_html=False, include_plotlyjs=False))

        # Write to HTML file
        with open(plotly_path, "w", encoding="utf-8") as out:
            out.write("<html><head><title>TOM Interactive Dashboard</title>")
            out.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
            out.write("<style>body { font-family: sans-serif; background: #0f172a; color: white; padding: 20px; } ")
            out.write(".grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; } ")
            out.write(".card { background: #1e293b; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }</style>")
            out.write("</head><body>")
            out.write("<h1>📊 TOM Interactive Plotly Dashboard</h1>")
            out.write("<p>Interactive explorations of key dataset features.</p>")
            out.write("<div class='grid'>")
            for h in dashboard_figs:
                out.write("<div class='card'>")
                out.write(h)
                out.write("</div>")
            out.write("</div></body></html>")
            
        chart_paths['plotly_dashboard'] = plotly_path
    except Exception as e:
        warnings.append(f"Failed to generate Plotly dashboard: {str(e)}")
        
    return chart_paths, base64_images, warnings

def chart(chart_type: str, column_name: str, df: pd.DataFrame = None):
    """
    Generates and displays a single specific chart.
    
    Parameters:
    chart_type (str): Type of chart, e.g. "violin", "histogram", "boxplot", "barchart", "piechart".
    column_name (str): Column to visualize.
    df (pd.DataFrame, optional): Data to use. Uses active dataframe if not provided.
    """
    # If df is not provided, fetch from tom active state
    import tom
    if df is None:
        df = tom.get_active_df()
        
    if df is None:
        print("❌ Error: No dataset loaded. Load a dataset first using mos.file().")
        return
        
    if column_name not in df.columns:
        print(f"❌ Error: Column '{column_name}' not found in dataset.")
        return
        
    fig, ax = plt.subplots(figsize=(6, 4))
    chart_type = chart_type.lower()
    
    try:
        series = df[column_name].dropna()
        if chart_type in ["violin", "violinplot"]:
            sns.violinplot(y=series, color='orchid', ax=ax)
            ax.set_title(f"{column_name} - Violin Plot")
        elif chart_type in ["histogram", "hist", "kde"]:
            sns.histplot(series, kde=True, color='teal', ax=ax)
            ax.set_title(f"{column_name} - Histogram")
        elif chart_type in ["boxplot", "box"]:
            sns.boxplot(x=series, color='coral', ax=ax)
            ax.set_title(f"{column_name} - Box Plot")
        elif chart_type in ["barchart", "bar"]:
            val_counts = series.value_counts()
            sns.barplot(x=val_counts.index[:10], y=val_counts.values[:10], palette='viridis', hue=val_counts.index[:10], ax=ax)
            ax.set_title(f"{column_name} - Bar Chart")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif chart_type in ["piechart", "pie"]:
            val_counts = series.value_counts()
            ax.pie(val_counts.values[:8], labels=val_counts.index[:8], autopct='%1.1f%%')
            ax.set_title(f"{column_name} - Pie Chart")
        else:
            print(f"❌ Unsupported chart type: '{chart_type}'")
            plt.close(fig)
            return
            
        plt.tight_layout()
        # In a real environment, we'd do plt.show(). Since we're headless or running programmatically,
        # we'll save it to a temporary file and display a message.
        os.makedirs("./tom_report/temp", exist_ok=True)
        temp_path = f"./tom_report/temp/single_{column_name}_{chart_type}.png"
        fig.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Chart generated and saved to: {temp_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate chart: {str(e)}")
        plt.close(fig)
