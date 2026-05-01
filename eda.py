import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# config
DATA_PATH  = "data/Video_Games_Sales_as_at_22_Dec_2016.csv"
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
})
PALETTE = "muted"

# ══════════════════════════════════════════════════════════════════════════════
# Load dataset and initial inspection
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)

# Rename for convenience
df.rename(columns={
    "Name":            "title",
    "Platform":        "platform",
    "Year_of_Release": "year",
    "Genre":           "genre",
    "Publisher":       "publisher",
    "NA_Sales":        "sales_na",
    "EU_Sales":        "sales_eu",
    "JP_Sales":        "sales_jp",
    "Other_Sales":     "sales_other",
    "Global_Sales":    "sales_global",
    "Critic_Score":    "critic_score",
    "Critic_Count":    "critic_count",
    "User_Score":      "user_score",
    "User_Count":      "user_count",
    "Developer":       "developer",
    "Rating":          "esrb",
}, inplace=True)

# Fix dtypes
df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")  
df["year"]       = pd.to_numeric(df["year"],       errors="coerce")

print(f"Dataset shape: {df.shape}")
print(df.dtypes, "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Missing values analysis
# ══════════════════════════════════════════════════════════════════════════════
print("── Missing values ──────────────────────────────────")
missing = (
    df.isnull().sum()
      .rename("count")
      .to_frame()
      .assign(pct=lambda x: 100 * x["count"] / len(df))
      .sort_values("pct", ascending=False)
)
print(missing[missing["count"] > 0].to_string(), "\n")

cols_missing = missing[missing["count"] > 0]

fig, ax = plt.subplots(figsize=(8, 4))
colors = sns.color_palette(PALETTE, len(cols_missing))
bars = ax.barh(cols_missing.index, cols_missing["pct"], color=colors)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing values by column")
ax.set_xlim(0, cols_missing["pct"].max() * 1.18)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_missing_values.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Sales distribution — raw vs log 
# ══════════════════════════════════════════════════════════════════════════════
log_sales = np.log1p(df["sales_global"].dropna())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["sales_global"].dropna(), bins=80, kde=False,
             ax=axes[0], color="#5B8DB8")
axes[0].set_title("Global Sales — raw (millions)")
axes[0].set_xlabel("Sales (M)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

sns.histplot(log_sales, bins=60, kde=True, ax=axes[1], color="#E07B54")
axes[1].set_title("Global Sales — log₁₊ transformed")
axes[1].set_xlabel("log(1 + Sales)")

plt.suptitle("Sales distribution: raw vs log-transformed", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_sales_distribution.png", bbox_inches="tight")
plt.show()

print(f"Skewness  raw : {df['sales_global'].skew():.2f}")
print(f"Skewness  log : {log_sales.skew():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Regional sales 
# ══════════════════════════════════════════════════════════════════════════════
regions      = ["sales_na", "sales_eu", "sales_jp", "sales_other"]
region_totals = df[regions].sum()
region_labels = ["North America", "Europe", "Japan", "Other"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
axes[0].pie(
    region_totals, labels=region_labels,
    autopct="%1.1f%%", startangle=140,
    colors=sns.color_palette(PALETTE, 4)
)
axes[0].set_title("Share of global sales by region")

# Bar chart — mean sales per game per region
mean_regional = df[regions].mean()
bars = axes[1].bar(region_labels, mean_regional,
                   color=sns.color_palette(PALETTE, 4))
axes[1].bar_label(bars, fmt="%.2f M", padding=3, fontsize=8)
axes[1].set_title("Average sales per game by region")
axes[1].set_ylabel("Sales (M)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_regional_sales.png", bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Sales by genre, platform, and ESRB rating
# ══════════════════════════════════════════════════════════════════════════════

def cat_sales_bar(col, label, top_n, filename, figsize=(10, 5)):
    """Bar chart: top N categories by mean global sales + count annotation."""
    grouped = (
        df.groupby(col)["sales_global"]
          .agg(mean_sales="mean", count="count")
          .sort_values("mean_sales", ascending=False)
          .head(top_n)
    )
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(PALETTE, len(grouped))
    bars = ax.bar(grouped.index, grouped["mean_sales"], color=colors)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={int(row['count'])}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Mean global sales by {label} (top {top_n})")
    ax.set_ylabel("Mean Sales (M)")
    ax.set_xlabel(label)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.show()
  

cat_sales_bar("genre",    "Genre",    12, "04_sales_by_genre.png")
cat_sales_bar("platform", "Platform", 15, "05_sales_by_platform.png")
cat_sales_bar("esrb",     "ESRB Rating", 8, "06_sales_by_esrb.png", figsize=(8, 4))

# ══════════════════════════════════════════════════════════════════════════════
# top 10 publishers by total sales
# ══════════════════════════════════════════════════════════════════════════════
top_publishers = (
    df.groupby("publisher")["sales_global"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette(PALETTE, len(top_publishers))
bars = ax.bar(top_publishers.index, top_publishers.values, color=colors)
ax.bar_label(bars, fmt="%.0f M", padding=3, fontsize=8)
ax.set_title("Top 10 publishers by total global sales")
ax.set_ylabel("Total Sales (M)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_top_publishers.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# sales and releases over time
# ══════════════════════════════════════════════════════════════════════════════
yearly = (
    df.dropna(subset=["year"])
      .groupby("year")["sales_global"]
      .agg(total="sum", count="count")
      .loc[1980:2016]  # trim outlier years
)

fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()

ax1.fill_between(yearly.index, yearly["total"],
                 alpha=0.4, color="#5B8DB8", label="Total sales (M)")
ax1.plot(yearly.index, yearly["total"], color="#5B8DB8", linewidth=2)
ax2.plot(yearly.index, yearly["count"], color="#E07B54",
         linewidth=2, linestyle="--", label="# of titles")

ax1.set_xlabel("Year")
ax1.set_ylabel("Total Global Sales (M)", color="#5B8DB8")
ax2.set_ylabel("Number of Titles Released", color="#E07B54")
ax1.set_title("Video game sales and releases over time")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_sales_over_time.png")
plt.show()

