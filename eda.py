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

# ══════════════════════════════════════════════════════════════════════════════
# Review scores vs sales 
# ══════════════════════════════════════════════════════════════════════════════
scored = df.dropna(subset=["critic_score", "user_score", "sales_global"]).copy()
scored["log_sales"] = np.log1p(scored["sales_global"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(scored["critic_score"], scored["log_sales"],
                alpha=0.25, s=12, color="#5B8DB8")
m, b = np.polyfit(scored["critic_score"], scored["log_sales"], 1)
x_line = np.linspace(scored["critic_score"].min(), scored["critic_score"].max(), 100)
axes[0].plot(x_line, m * x_line + b, color="crimson", linewidth=1.5)
axes[0].set_title("Critic score vs log(Global Sales)")
axes[0].set_xlabel("Critic Score (0–100)")
axes[0].set_ylabel("log(1 + Sales)")
corr_c = scored["critic_score"].corr(scored["log_sales"])
axes[0].annotate(f"r = {corr_c:.3f}", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=10, color="crimson")

axes[1].scatter(scored["user_score"], scored["log_sales"],
                alpha=0.25, s=12, color="#E07B54")
m2, b2 = np.polyfit(scored["user_score"], scored["log_sales"], 1)
x_line2 = np.linspace(scored["user_score"].min(), scored["user_score"].max(), 100)
axes[1].plot(x_line2, m2 * x_line2 + b2, color="crimson", linewidth=1.5)
axes[1].set_title("User score vs log(Global Sales)")
axes[1].set_xlabel("User Score (0–10)")
axes[1].set_ylabel("log(1 + Sales)")
corr_u = scored["user_score"].corr(scored["log_sales"])
axes[1].annotate(f"r = {corr_u:.3f}", xy=(0.05, 0.92),
                 xycoords="axes fraction", fontsize=10, color="crimson")

plt.suptitle("Review scores vs sales (complete-case subset, n={:,})".format(len(scored)),
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_scores_vs_sales.png", bbox_inches="tight")
plt.show()
print(f"Critic score correlation with log_sales : {corr_c:.3f}")
print(f"User score  correlation with log_sales  : {corr_u:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Correlation matrix of numeric features
# ══════════════════════════════════════════════════════════════════════════════
num_cols = ["sales_global", "sales_na", "sales_eu", "sales_jp", "sales_other",
            "critic_score", "critic_count", "user_score", "user_count", "year"]
num_cols = [c for c in num_cols if c in df.columns]

corr_matrix = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # upper triangle mask
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 8}
)
ax.set_title("Correlation matrix — numeric features")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_correlation_heatmap.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# high critic score but low sales 
# ══════════════════════════════════════════════════════════════════════════════
# Define: high critic score (≥85) but bottom 25% of sales
high_score_threshold = 85
low_sales_threshold  = df["sales_global"].quantile(0.25)

critically_acclaimed = df[df["critic_score"] >= high_score_threshold].copy()
underperformers = critically_acclaimed[
    critically_acclaimed["sales_global"] <= low_sales_threshold
].sort_values("critic_score", ascending=False)

print("── Critically acclaimed but commercially underperforming games ──")
print(f"High critic score threshold : ≥ {high_score_threshold}")
print(f"Low sales threshold (Q1)    : ≤ {low_sales_threshold:.2f} M")
print(f"Games in this category      : {len(underperformers)}\n")
print(underperformers[["title", "platform", "genre", "publisher",
                        "critic_score", "sales_global"]].head(20).to_string(index=False))

# Scatter highlighting underperformers
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["critic_score"], df["sales_global"],
           alpha=0.2, s=10, color="#AAAAAA", label="All games")
ax.scatter(underperformers["critic_score"], underperformers["sales_global"],
           alpha=0.7, s=20, color="#E07B54",
           label=f"High score / low sales (n={len(underperformers)})")

ax.axvline(high_score_threshold, color="steelblue",
           linewidth=1, linestyle="--", alpha=0.7)
ax.axhline(low_sales_threshold, color="steelblue",
           linewidth=1, linestyle="--", alpha=0.7)
ax.set_xlim(0, 105)
ax.set_ylim(-0.1, df["sales_global"].quantile(0.995))
ax.set_xlabel("Critic Score")
ax.set_ylabel("Global Sales (M)")
ax.set_title("Critically acclaimed games that underperformed commercially")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_acclaimed_underperformers.png")
plt.show()


