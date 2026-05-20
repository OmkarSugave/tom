# 🚀 tom — One-Line Data Analytics Library

`tom` is a highly autonomous, zero-configuration Python library that delivers end-to-end exploratory data analysis (EDA) in a single line. 

## ✨ Features
- **Smart Loading (`om.file()`)**: Automatic format, encoding, and delimiter detection for `.csv`, `.xlsx`, `.xls`, `.json`, `.parquet`, `.tsv`, and `.txt`.
- **Auto Data Preprocessing (`om.clean_report()`)**: Type coercion (dates, numerical objects), median/mode missing value imputation, and duplicate handling.
- **Statistical Analysis Engine (`om.stats`)**: Normality tests, correlation matrices, Chi-Square associations, and ANOVA tests.
- **Dazzling Data Visualizations (`om.charts`)**: Static high-res Seaborn charts + portable, interactive Plotly dashboard.
- **NLP Insights (`om.insights`)**: Readable plain-English suggestions and red flags.
- **Premium Reporting (`om.describe()`)**: Rich terminal prints and self-contained glassmorphism HTML pages.

## 📦 Installation
```bash
pip install -e .
```

## 🛠️ Usage
```python
import tom as om

# Load data interactively (terminal prompt if no path) or directly
om.file("dataset.csv")

# Generate beautiful, high-quality, comprehensive reports in one go!
om.describe()
```
