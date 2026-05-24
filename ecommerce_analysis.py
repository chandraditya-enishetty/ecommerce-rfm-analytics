"""
E-commerce Customer & Revenue Analytics
========================================
Dataset : UCI Online Retail (541K transactions, 2010-2011)
Run:
    pip install pandas numpy matplotlib seaborn plotly scikit-learn shap openpyxl
    python ecommerce_analysis.py

Outputs:
    ecommerce_clean.csv          → cleaned dataset
    rfm_segments.csv             → RFM scores per customer
    sql_results/                 → SQL query outputs
    eda_overview.png             → EDA charts
    rfm_distribution.png         → RFM segment breakdown
    pareto_curve.png             → product revenue concentration
    churn_roc.png                → model ROC curve
    churn_feature_importance.png → feature importance
    model_metrics.csv            → AUC and classification report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    roc_auc_score, roc_curve,
    average_precision_score,
    classification_report,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

PALETTE = {"teal": "#1D9E75", "purple": "#534AB7", "amber": "#BA7517", "coral": "#D85A30"}


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 60)
print("E-commerce Customer & Revenue Analytics")
print("=" * 60)

df = pd.read_excel(
    "Online Retail.xlsx",
    dtype={"CustomerID": str, "InvoiceNo": str},
)
print(f"\nRaw data loaded     : {len(df):,} rows × {df.shape[1]} columns")
print(f"Date range          : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"Countries           : {df['Country'].nunique()}")
print(f"Customers           : {df['CustomerID'].nunique():,} (including nulls)")


# ── 2. DATA CLEANING ──────────────────────────────────────────────────────────

print("\n── Data cleaning ───────────────────────────────────────────")

# Remove cancelled orders (InvoiceNo starts with C)
cancelled = df["InvoiceNo"].str.startswith("C", na=False)
print(f"Cancelled orders    : {cancelled.sum():,} rows removed")
df = df[~cancelled]

# Drop missing CustomerID (cannot do customer analysis without it)
missing_cust = df["CustomerID"].isna().sum()
print(f"Missing CustomerID  : {missing_cust:,} rows removed")
df = df.dropna(subset=["CustomerID"])

# Remove negative or zero quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
print(f"Negative qty/price  : removed")

# Remove test / bad stock codes
bad_codes = ["POST", "DOT", "M", "BANK CHARGES", "PADS", "AMAZONFEE"]
df = df[~df["StockCode"].isin(bad_codes)]

# Create revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Parse dates
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Year"]        = df["InvoiceDate"].dt.year
df["Month"]       = df["InvoiceDate"].dt.month
df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
df["Hour"]        = df["InvoiceDate"].dt.hour
df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").astype(str)

print(f"\nClean dataset       : {len(df):,} rows")
print(f"Customers           : {df['CustomerID'].nunique():,}")
print(f"Products            : {df['StockCode'].nunique():,}")
print(f"Total revenue       : £{df['Revenue'].sum():,.0f}")

df.to_csv("ecommerce_clean.csv", index=False)
print("Saved: ecommerce_clean.csv")


# ── 3. SQL ANALYSIS LAYER ─────────────────────────────────────────────────────

print("\n── SQL analysis ────────────────────────────────────────────")

conn = sqlite3.connect(":memory:")
df.to_sql("orders", conn, index=False, if_exists="replace")

queries = {

    "revenue_by_country": """
        SELECT
            Country,
            COUNT(DISTINCT InvoiceNo)  AS total_orders,
            COUNT(DISTINCT CustomerID) AS total_customers,
            ROUND(SUM(Revenue), 2)     AS total_revenue,
            ROUND(AVG(Revenue), 2)     AS avg_order_value
        FROM orders
        GROUP BY Country
        ORDER BY total_revenue DESC
    """,

    "monthly_revenue": """
        SELECT
            YearMonth,
            COUNT(DISTINCT InvoiceNo)  AS orders,
            COUNT(DISTINCT CustomerID) AS active_customers,
            ROUND(SUM(Revenue), 2)     AS revenue
        FROM orders
        GROUP BY YearMonth
        ORDER BY YearMonth
    """,

    "top_products": """
        SELECT
            StockCode,
            Description,
            SUM(Quantity)          AS total_qty,
            ROUND(SUM(Revenue), 2) AS total_revenue,
            COUNT(DISTINCT InvoiceNo) AS order_count
        FROM orders
        GROUP BY StockCode, Description
        ORDER BY total_revenue DESC
        LIMIT 20
    """,

    "customer_revenue": """
        SELECT
            CustomerID,
            Country,
            COUNT(DISTINCT InvoiceNo)  AS total_orders,
            SUM(Quantity)              AS total_items,
            ROUND(SUM(Revenue), 2)     AS total_revenue,
            ROUND(AVG(Revenue), 2)     AS avg_order_value,
            MIN(InvoiceDate)           AS first_purchase,
            MAX(InvoiceDate)           AS last_purchase
        FROM orders
        GROUP BY CustomerID, Country
        ORDER BY total_revenue DESC
    """,

    "hourly_orders": """
        SELECT
            DayOfWeek,
            Hour,
            COUNT(DISTINCT InvoiceNo) AS orders
        FROM orders
        GROUP BY DayOfWeek, Hour
        ORDER BY orders DESC
    """,
}

results = {}
for name, query in queries.items():
    results[name] = pd.read_sql(query, conn)
    print(f"  Query '{name}': {len(results[name])} rows")

conn.close()

revenue_by_country = results["revenue_by_country"]
monthly_revenue    = results["monthly_revenue"]
top_products       = results["top_products"]
customer_revenue   = results["customer_revenue"]


# ── 4. RFM SEGMENTATION ───────────────────────────────────────────────────────

print("\n── RFM segmentation ────────────────────────────────────────")

# Snapshot date = 1 day after last invoice
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("CustomerID")
    .agg(
        last_purchase = ("InvoiceDate", "max"),
        frequency     = ("InvoiceNo",   "nunique"),
        monetary      = ("Revenue",     "sum"),
    )
    .reset_index()
)
rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days

# Score 1–4 (4 = best)
rfm["R_score"] = pd.qcut(rfm["recency"],   q=4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"],  q=4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

# Segment labels
def rfm_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal"
    elif r >= 3 and f <= 2:
        return "Promising"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r == 1 and f == 1:
        return "Lost"
    else:
        return "Needs Attention"

rfm["segment"] = rfm.apply(rfm_segment, axis=1)

seg_summary = (
    rfm.groupby("segment")
    .agg(
        customers     = ("CustomerID", "count"),
        total_revenue = ("monetary",   "sum"),
        avg_recency   = ("recency",    "mean"),
        avg_frequency = ("frequency",  "mean"),
        avg_monetary  = ("monetary",   "mean"),
    )
    .reset_index()
)
seg_summary["revenue_pct"] = (seg_summary["total_revenue"] / seg_summary["total_revenue"].sum() * 100).round(1)
seg_summary = seg_summary.sort_values("total_revenue", ascending=False)

print("\nRFM segment summary:")
print(seg_summary[["segment", "customers", "total_revenue", "revenue_pct"]].to_string(index=False))

rfm.to_csv("rfm_segments.csv", index=False)
print("\nSaved: rfm_segments.csv")


# ── 5. EDA PLOTS ──────────────────────────────────────────────────────────────

print("\n── EDA visualisations ──────────────────────────────────────")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("E-commerce Revenue & Customer Overview", fontsize=14, fontweight="bold")

# Plot 1 — Monthly revenue trend
ax = axes[0, 0]
mr = monthly_revenue.copy()
ax.bar(range(len(mr)), mr["revenue"] / 1000, color=PALETTE["teal"], edgecolor="white")
ax.set_xticks(range(0, len(mr), 2))
ax.set_xticklabels(mr["YearMonth"].iloc[::2], rotation=45, ha="right", fontsize=9)
ax.set_title("Monthly revenue trend\nNovember 2011 peak = pre-Christmas spike", fontsize=11)
ax.set_ylabel("Revenue (£ thousands)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}K"))

# Plot 2 — Top 10 countries by revenue (ex UK)
ax = axes[0, 1]
top_countries = revenue_by_country[revenue_by_country["Country"] != "United Kingdom"].head(10)
ax.barh(top_countries["Country"][::-1], top_countries["total_revenue"][::-1] / 1000,
        color=PALETTE["purple"], edgecolor="white")
ax.set_title("Top 10 markets by revenue\n(excl. United Kingdom)", fontsize=11)
ax.set_xlabel("Revenue (£ thousands)")

# Plot 3 — RFM segment customer count
ax = axes[1, 0]
seg_order = seg_summary.sort_values("customers", ascending=False)
colors = [PALETTE["teal"], PALETTE["purple"], PALETTE["amber"],
          PALETTE["coral"], "#B4B2A9", "#378ADD"]
ax.bar(seg_order["segment"], seg_order["customers"],
       color=colors[:len(seg_order)], edgecolor="white")
ax.set_title("Customer count by RFM segment", fontsize=11)
ax.set_ylabel("Customers")
ax.tick_params(axis="x", rotation=20)

# Plot 4 — Revenue share by segment
ax = axes[1, 1]
ax.pie(
    seg_summary["total_revenue"],
    labels=seg_summary["segment"],
    autopct="%1.1f%%",
    colors=colors[:len(seg_summary)],
    startangle=140,
)
ax.set_title("Revenue share by RFM segment", fontsize=11)

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_overview.png")


# ── 6. PARETO CURVE ───────────────────────────────────────────────────────────

prod_rev = (
    df.groupby("StockCode")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
prod_rev["cumulative_pct"] = prod_rev["Revenue"].cumsum() / prod_rev["Revenue"].sum() * 100
prod_rev["product_pct"]    = np.arange(1, len(prod_rev) + 1) / len(prod_rev) * 100

pareto_20 = prod_rev[prod_rev["product_pct"] <= 20]["cumulative_pct"].iloc[-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prod_rev["product_pct"], prod_rev["cumulative_pct"],
        color=PALETTE["teal"], lw=2)
ax.axvline(20, color=PALETTE["coral"], linestyle="--", lw=1.5, label="Top 20% of products")
ax.axhline(pareto_20, color=PALETTE["coral"], linestyle="--", lw=1.5)
ax.fill_between(prod_rev["product_pct"], prod_rev["cumulative_pct"],
                alpha=0.1, color=PALETTE["teal"])
ax.annotate(f"Top 20% of products\ndrive {pareto_20:.0f}% of revenue",
            xy=(20, pareto_20), xytext=(35, pareto_20 - 15),
            fontsize=11, color=PALETTE["coral"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["coral"]))
ax.set_xlabel("Cumulative % of products", fontsize=11)
ax.set_ylabel("Cumulative % of revenue", fontsize=11)
ax.set_title("Pareto curve — revenue concentration by product\n"
             f"Top 20% of SKUs drive {pareto_20:.0f}% of total revenue", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("pareto_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: pareto_curve.png  (top 20% of products = {pareto_20:.0f}% revenue)")


# ── 7. CHURN PREDICTION MODEL ─────────────────────────────────────────────────

print("\n── Churn prediction model ──────────────────────────────────")

# Label: At Risk + Lost = churned (1), others = retained (0)
rfm["churned"] = rfm["segment"].isin(["At Risk", "Lost"]).astype(int)
print(f"Churn rate          : {rfm['churned'].mean():.1%}")

# Features
feature_cols = ["recency", "frequency", "monetary"]
X = rfm[feature_cols].copy()
y = rfm["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=6,
            min_samples_leaf=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nCross-validation (5-fold ROC-AUC):")

metrics_rows = []
for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    print(f"  {name:<25} CV AUC = {scores.mean():.3f} ± {scores.std():.3f}  |  Test AUC = {auc:.3f}")
    metrics_rows.append({"model": name, "cv_auc": scores.mean(), "cv_std": scores.std(), "test_auc": auc, "avg_precision": ap})

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("model_metrics.csv", index=False)
print("Saved: model_metrics.csv")


# ── 8. ROC CURVE PLOT ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], color="#B4B2A9", linestyle="--", lw=1, label="Random baseline")

colors = [PALETTE["purple"], PALETTE["teal"]]
for (name, pipeline), color in zip(models.items(), colors):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC = {auc:.3f})")

ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.set_title("ROC curves — churn prediction model", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("churn_roc.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: churn_roc.png")


# ── 9. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

rf_clf        = models["Random Forest"].named_steps["clf"]
importances   = rf_clf.feature_importances_
feat_imp_df   = (
    pd.DataFrame({"feature": feature_cols, "importance": importances})
    .sort_values("importance", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1],
        color=PALETTE["teal"], edgecolor="white")
ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=11)
ax.set_title("Top features — random forest churn prediction", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("churn_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: churn_feature_importance.png")

print("\nTop features:")
print(feat_imp_df.to_string(index=False))


# ── 10. SHAP ───────────────────────────────────────────────────────

if SHAP_AVAILABLE:
    print("\nComputing SHAP values...")
    scaler     = models["Random Forest"].named_steps["scaler"]
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    explainer  = shap.TreeExplainer(rf_clf)
    shap_vals  = explainer(X_test_sc)
    vals = shap_vals[:, :, 1] if len(shap_vals.shape) == 3 else shap_vals

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(vals, max_display=7, show=False)
    plt.title("SHAP feature importance — churn prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_churn.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: shap_churn.png")


# ── 11. PORTFOLIO SUMMARY ─────────────────────────────────────────────────────

best = metrics_df.sort_values("test_auc", ascending=False).iloc[0]
champions = seg_summary[seg_summary["segment"] == "Champions"]
at_risk   = seg_summary[seg_summary["segment"] == "At Risk"]

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Dataset      : UCI Online Retail (2010–2011)
Transactions : {len(df):,} (after cleaning)
Customers    : {rfm.shape[0]:,}
Countries    : {df['Country'].nunique()}
Total revenue: £{df['Revenue'].sum():,.0f}

RFM segments:
  Champions  : {int(champions['customers'].values[0]) if len(champions) else 'N/A'} customers
  At Risk    : {int(at_risk['customers'].values[0]) if len(at_risk) else 'N/A'} customers
  Churn rate : {rfm['churned'].mean():.1%}

Pareto finding: top 20% of products drive {pareto_20:.0f}% of revenue

Best model   : {best['model']}
  CV AUC     : {best['cv_auc']:.3f} ± {best['cv_std']:.3f}
  Test AUC   : {best['test_auc']:.3f}



Output files:
  ecommerce_clean.csv          → cleaned dataset
  rfm_segments.csv             → RFM scores & segments
  eda_overview.png             → revenue & customer charts
  pareto_curve.png             → product concentration
  churn_roc.png                → model comparison
  churn_feature_importance.png → top predictors
  model_metrics.csv            → AUC numbers


""")
