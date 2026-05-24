# E-commerce Customer & Revenue Analytics
### RFM Segmentation · Churn Prediction · Tableau Public

> Identifying which customers drive the most revenue, who is at churn risk, and which products and markets should be prioritised — using 396K transactions from a UK-based online retailer.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-SQLite-orange?style=flat-square)](https://sqlite.org)
[![Dashboard](https://img.shields.io/badge/Dashboard-Tableau%20Public-blue?style=flat-square)](https://public.tableau.com/app/profile/chandraditya.enishetty/viz/E-commerceCustomerRevenueAnalyticsRFMSegmentation/Dashboard1)
[![Data](https://img.shields.io/badge/Data-UCI%20Online%20Retail-green?style=flat-square)](https://archive.ics.uci.edu/dataset/352/online+retail)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Contents

- [Project overview](#project-overview)
- [Key findings](#key-findings)
- [Dashboard](#dashboard)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [How to reproduce](#how-to-reproduce)
- [Methodology](#methodology)
- [Recommendations](#recommendations)
- [Contact](#contact)

---

## Project overview

This project analyses 396K transactions from a UK-based online retailer (UCI Online Retail dataset, 2010–2011) to answer three business questions:

| # | Business question | Analytical approach |
|---|---|---|
| 1 | Which customers drive the most revenue and who is at churn risk? | RFM (Recency, Frequency, Monetary) segmentation |
| 2 | Which products follow the Pareto principle? | Cumulative revenue concentration analysis |
| 3 | Which international markets should be prioritised for expansion? | Country-level revenue and order value comparison |

---

## Key findings

**1. 497 Champion customers drive 50% of total revenue**
Champions represent just 11.5% of the customer base but account for £4.4M of £8.77M total revenue. Losing even a fraction of this segment would have an outsized revenue impact — making retention programmes for this group the highest-priority CRM action.

**2. 643 At Risk customers represent £1M in churn exposure**
At Risk customers (high historical spend, declining recency) account for £1.01M in revenue. Without targeted re-engagement, this segment is likely to transition to Lost — a recoverable situation with the right intervention timing.

**3. Top 20% of products drive 78% of revenue**
The revenue concentration is stronger than the classic 80/20 rule. The bottom 80% of SKUs contribute only 22% of revenue — suggesting significant opportunity to rationalise inventory and focus marketing spend on proven performers.

**4. Netherlands is the #1 international market despite its size**
Excluding the UK, Netherlands leads international revenue at £225K — significantly ahead of EIRE and Germany. This likely reflects a concentrated wholesale relationship rather than broad consumer penetration, warranting further investigation.

**5. Churn model achieves AUC 0.969**
A Random Forest classifier trained on Recency, Frequency, and Monetary features achieves AUC 0.969, with Recency (77.5% feature importance) as the dominant predictor. Customers inactive for 90+ days have significantly elevated churn probability.

---

## Dashboard

> **Live dashboard:** [Tableau Public — View here](https://public.tableau.com/app/profile/chandraditya.enishetty/viz/E-commerceCustomerRevenueAnalyticsRFMSegmentation/Dashboard1)

The interactive dashboard includes:

| View | Content |
|---|---|
| Revenue overview | Monthly trend, KPI cards, international market breakdown |
| Customer segments | RFM segment distribution, revenue by segment |
| Product analysis | Top 10 products by revenue, Pareto concentration curve |

Click any segment or country to filter all charts simultaneously.

---

## Dataset

**UCI Online Retail Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- Coverage: December 2010 – December 2011
- Size: 541,909 raw transactions · 8 columns
- Retailer: UK-based online gift store selling to 38 countries
- Access: Free, no signup required

**Raw columns:**

| Column | Description |
|---|---|
| InvoiceNo | Transaction ID (prefix C = cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Transaction timestamp |
| UnitPrice | Price per unit (GBP) |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Project structure

```
ecommerce-rfm-analytics/
│
├── README.md
├── requirements.txt
│
├── ecommerce_analysis.py      ← full analysis pipeline
│
├── outputs/
│   ├── eda_overview.png
│   ├── pareto_curve.png
│   ├── churn_roc.png
│   ├── churn_feature_importance.png
│   ├── shap_churn.png
│   └── model_metrics.csv
│
└── data/
    └── .gitkeep               ← raw data not committed
```

> Raw data files (`ecommerce_clean.csv`, `rfm_segments.csv`) are excluded from version control via `.gitignore`. Download the source data from the UCI link above.

---

## How to reproduce

### 1. Clone the repository

```bash
git clone https://github.com/chandraditya-enishetty/ecommerce-rfm-analytics.git
cd ecommerce-rfm-analytics
```

### 2. Download the dataset

Go to [archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail) → Download → place `Online Retail.xlsx` in the project root folder.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python ecommerce_analysis.py
```

Runtime: approximately 2–3 minutes. Outputs all plots and CSVs to the project folder.

---

## Methodology

### Data cleaning
- Removed 9,288 cancelled orders (InvoiceNo starting with "C")
- Removed 134,697 rows with missing CustomerID
- Removed negative quantities and zero unit prices
- Removed non-product stock codes (POST, DOT, BANK CHARGES etc.)
- Created Revenue = Quantity × UnitPrice
- Clean dataset: **396,470 transactions · 4,334 customers · 3,660 products**

### RFM segmentation

Each customer scored 1–4 on three dimensions using quartile binning:

| Dimension | Definition | Score 4 = best |
|---|---|---|
| Recency (R) | Days since last purchase | Purchased recently |
| Frequency (F) | Number of distinct orders | Ordered many times |
| Monetary (M) | Total spend | Highest spender |

Segment labels assigned by R/F/M score combinations:

| Segment | Criteria | Customers | Revenue share |
|---|---|---|---|
| Champions | R≥4, F≥4, M≥4 | 497 | 50.2% |
| Loyal | R≥3, F≥3 | 1,027 | 23.9% |
| At Risk | R≤2, F≥3 | 643 | 11.5% |
| Needs Attention | Mixed mid scores | 994 | 5.8% |
| Promising | R≥3, F≤2 | 658 | 5.6% |
| Lost | R=1, F=1 | 515 | 2.9% |

### Churn prediction model

- **Target:** At Risk + Lost = churned (1); all others = retained (0)
- **Churn rate:** 26.7%
- **Features:** Recency, Frequency, Monetary (raw values only — no RFM scores to avoid leakage)
- **Models:** Logistic Regression (baseline) vs Random Forest
- **Evaluation:** 5-fold stratified cross-validation + hold-out test set

### Model results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.877 ± 0.007 | 0.882 |
| Random Forest | 0.976 ± 0.005 | **0.969** |

**Top features (Random Forest):**

| Feature | Importance |
|---|---|
| Recency | 77.5% |
| Frequency | 16.2% |
| Monetary | 6.3% |

Recency dominates because it is the most direct behavioural signal of disengagement — a customer inactive for 90+ days is unlikely to return without targeted intervention.

---

## Recommendations

**1. Protect Champions with a dedicated retention programme**
497 customers drive 50% of revenue. A dedicated VIP programme (early access, personalised outreach, loyalty rewards) targeted at this segment would protect the most revenue-critical cohort. Even a 5% improvement in Champions retention adds ~£220K in protected revenue annually.

**2. Re-engage At Risk customers within 30 days**
643 At Risk customers represent £1M in churn exposure. The churn model identifies these customers at AUC 0.969 — meaning the retention team can act on a highly accurate priority list. Email campaigns with personalised product recommendations and time-limited incentives are the recommended first intervention.

**3. Rationalise the bottom 80% of the product catalogue**
Top 20% of SKUs drive 78% of revenue. A structured SKU rationalisation review — removing consistently low-revenue, high-handling-cost products — would reduce operational complexity and allow marketing spend to focus on proven performers.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
shap>=0.43
openpyxl>=3.1
```

---

## Contact

**Chandraditya Enishetty**
[LinkedIn](https://www.linkedin.com/in/chandraditya-enishetty) · [GitHub](https://github.com/chandraditya-enishetty)

---

*Built as a portfolio project demonstrating end-to-end retail analytics: data cleaning, SQL analysis, RFM customer segmentation, churn prediction modelling, and interactive Tableau Public dashboard delivery.*
