import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# config
DATA_PATH  = "data/Video_Games_Sales_as_at_22_Dec_2016.csv"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# Load and clean data
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)

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

print(f"Loaded  : {df.shape[0]:,} rows, {df.shape[1]} columns")


# if user_score contains "tbd" strings coerce to NaN
df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")
df["year"]       = pd.to_numeric(df["year"],       errors="coerce")


# drop rows where target variable is missing
before = len(df)
df.dropna(subset=["sales_global"], inplace=True)
print(f"Dropped : {before - len(df):,} rows missing sales_global")
print(f"Remaining: {len(df):,} rows")


# Sales are heavily right-skewed. log(1+x) brings them to near-normal
# and stabilises variance for regression models.
# Use log1p so that games with 0 sales map to 0 (not -inf).
df["log_sales"] = np.log1p(df["sales_global"])

print(f"\nSales stats (raw):")
print(df["sales_global"].describe().round(3).to_string())
print(f"\nSales stats (log-transformed):")
print(df["log_sales"].describe().round(3).to_string())

# Define AAA publishers as those in the top 20 by total global sales to avoid hard-coding names 
top_publishers = (
    df.groupby("publisher")["sales_global"]
      .sum()
      .sort_values(ascending=False)
      .head(20)
      .index.tolist()
)

df["is_aaa"] = df["publisher"].isin(top_publishers).astype(int)

print(f"\nTop 20 AAA publishers (by total sales):")
for i, pub in enumerate(top_publishers, 1):
    print(f"  {i:2}. {pub}")

print(f"\nAAA games : {df['is_aaa'].sum():,} ({100*df['is_aaa'].mean():.1f}%)")
print(f"Indie games: {(df['is_aaa']==0).sum():,} ({100*(1-df['is_aaa'].mean()):.1f}%)")



# median imputation + a binary flag indicating the score was missing.
# The flag lets the model learn whether "missing score" itself is predictive.
for col in ["critic_score", "user_score", "critic_count", "user_count"]:
    if col in df.columns:
        flag_col = f"{col}_missing"
        df[flag_col] = df[col].isna().astype(int)
        median_val   = df[col].median()
        df[col]      = df[col].fillna(median_val)
        n_imputed    = df[flag_col].sum()
        print(f"\nImputed {col}: {n_imputed:,} values → median ({median_val:.1f})")
        print(f"  Missing flag column added: '{flag_col}'")



# Fill remaining NaNs in categorical columns with "Unknown"
for col in ["genre", "platform", "esrb"]:
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        df[col] = df[col].fillna("Unknown")
        print(f"\nFilled {n_missing} missing values in '{col}' with 'Unknown'")

# Consolidate rare ESRB values (K-A is an old rating, effectively = E)
df["esrb"] = df["esrb"].replace({"K-A": "E", "AO": "M"})
print(f"\nESRB value counts after cleanup:")
print(df["esrb"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# One-hot encode categorical features
# ══════════════════════════════════════════════════════════════════════════════
# drops one category per feature to avoid multicollinearity, the dropped category becomes the reference.
print("\nOne-hot encoding genre, platform, esrb...")

before_cols = df.shape[1]

df = pd.get_dummies(df, columns=["genre", "platform", "esrb"],
                    drop_first=True, dtype=int)

added_cols = df.shape[1] - before_cols
print(f"Added {added_cols} dummy columns")
print(f"New shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# select features and target variable for modeling
# ══════════════════════════════════════════════════════════════════════════════
# Drop columns not used as model features
DROP_COLS = [
    "title",        # free text, not useful for regression
    "publisher",    # replaced by is_aaa flag
    "developer",    # too many unique values, not in scope
    "year",         # optional: include if you want temporal features
    "sales_na", "sales_eu", "sales_jp", "sales_other",  # regional breakdowns
    "sales_global", # raw target, we use log_sales instead
]

# Only drop columns that actually exist
DROP_COLS = [c for c in DROP_COLS if c in df.columns]
df_model = df.drop(columns=DROP_COLS)

TARGET = "log_sales"
FEATURES = [c for c in df_model.columns if c != TARGET]

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"\nFinal feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Target: '{TARGET}'")

# ══════════════════════════════════════════════════════════════════════════════
# train / test split
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\nTrain set : {X_train.shape[0]:,} rows ({100*(1-TEST_SIZE):.0f}%)")
print(f"Test set  : {X_test.shape[0]:,} rows ({100*TEST_SIZE:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Scale numeric features using StandardScaler (mean=0, std=1)
# ══════════════════════════════════════════════════════════════════════════════
# Only scale continuous features,leave binary/dummy columns as-is.
# Fit scaler on train set only to prevent data leakage.
NUMERIC_FEATURES = [
    "critic_score", "critic_count",
    "user_score",   "user_count",
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in X_train.columns]

scaler = StandardScaler()
X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
X_test[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])

print(f"\nScaled {len(NUMERIC_FEATURES)} numeric features: {NUMERIC_FEATURES}")

# ══════════════════════════════════════════════════════════════════════════════
# Saved processed data 
# ══════════════════════════════════════════════════════════════════════════════
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv( OUTPUT_DIR / "X_test.csv",  index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv( OUTPUT_DIR / "y_test.csv",  index=False)
