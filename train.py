import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

# config
DATA_DIR   = Path("data")
FIG_DIR    = Path("figures")
MODEL_DIR  = Path("models")
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

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


# load preprocessed data
print("Loading preprocessed data:")
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print(f"  Train : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")


# ══════════════════════════════════════════════════════════════════════════════
# define models to train
# ══════════════════════════════════════════════════════════════════════════════
# Ridge regression is used instead of plain OLS Linear Regression.
# With many one-hot encoded columns, Ridge's L2 penalty prevents
# overfitting to correlated dummy variables — a better baseline.

models = {
    "Linear Regression (Ridge)": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
                                        n_estimators=200,
                                        max_depth=None,
                                        min_samples_leaf=2,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1,
                                        ),
    "Gradient Boosting": GradientBoostingRegressor(
                                     n_estimators=300,
                                     learning_rate=0.05,
                                     max_depth=4,
                                     subsample=0.8,
                                     random_state=RANDOM_STATE,
                                 ),
}

COLORS = {
    "Linear Regression (Ridge)": "#5B8DB8",
    "Random Forest":"#59A96A",
    "Gradient Boosting":"#E07B54",
}


# ══════════════════════════════════════════════════════════════════════════════
# train models, evaluate on test set, and perform 5-fold cross-validation on training set
# ══════════════════════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}   # model name → dict of metrics


print(f"{'Model':<30} {'MSE':>7} {'RMSE':>7} {'R^2':>7} {'CV R^2':>12}")

for name, model in models.items():
    # train
    model.fit(X_train, y_train)

    # test set predictions
    y_pred = model.predict(X_test)

    # metrics on test set
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=kf, scoring="r2", n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "mse":     mse,
        "rmse":    rmse,
        "r2":      r2,
        "cv_mean": cv_mean,
        "cv_std":  cv_std,
    }

    print(f"{name:<30} {mse:>7.4f} {rmse:>7.4f} {r2:>7.4f}   "
          f"{cv_mean:.4f} ± {cv_std:.4f}")

    # save model
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, MODEL_DIR / f"{safe_name}.pkl")



# ══════════════════════════════════════════════════════════════════════════════
# metrics comparison bar charts (MSE, RMSE, R^2) for all models
# ══════════════════════════════════════════════════════════════════════════════
metrics_df = pd.DataFrame({
    name: {"MSE": r["mse"], "RMSE": r["rmse"], "R^2": r["r2"]}
    for name, r in results.items()
}).T

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, metric in zip(axes, ["MSE", "RMSE", "R^2"]):
    bars = ax.bar(
        metrics_df.index,
        metrics_df[metric],
        color=[COLORS[n] for n in metrics_df.index],
        width=0.5,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(metrics_df.index, rotation=15, ha="right", fontsize=8)
    
    # Lower is better for MSE/RMSE, higher for R^2
    if metric == "R^2":
        ax.set_ylim(0, min(1.0, metrics_df[metric].max() * 1.2))
    else:
        ax.set_ylim(0, metrics_df[metric].max() * 1.2)

plt.suptitle("Model comparison — MSE, RMSE, R^2", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "12_model_comparison_metrics.png", bbox_inches="tight")
plt.show()



# ══════════════════════════════════════════════════════════════════════════════
# cross-validated R^2 comparison bar chart  for all models
# ══════════════════════════════════════════════════════════════════════════════
cv_means = [results[n]["cv_mean"] for n in models]
cv_stds  = [results[n]["cv_std"]  for n in models]
names    = list(models.keys())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(names, cv_means, yerr=cv_stds, capsize=6,
              color=[COLORS[n] for n in names], width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=8, fontsize=8)
ax.set_ylabel("Mean R^2 (5-fold CV)")
ax.set_title("Cross-validated R^2 — train set (5-fold)")
ax.set_ylim(0, max(cv_means) * 1.3)
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_cv_r2_comparison.png", bbox_inches="tight")
plt.show()



# ══════════════════════════════════════════════════════════════════════════════
# predicted vs actual scatter plots for all models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, r) in zip(axes, results.items()):
    y_pred = r["y_pred"]
    ax.scatter(y_test, y_pred, alpha=0.25, s=10, color=COLORS[name])

    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")

    ax.set_title(f"{name}\nR^2 = {r['r2']:.4f}")
    ax.set_xlabel("Actual log(Sales)")
    ax.set_ylabel("Predicted log(Sales)")
    ax.legend(fontsize=8)

plt.suptitle("Predicted vs Actual — log-transformed global sales", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "14_predicted_vs_actual.png", bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# residuals vs predicted scatter plots for all models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, r) in zip(axes, results.items()):
    residuals = y_test - r["y_pred"]
    ax.scatter(r["y_pred"], residuals, alpha=0.25, s=10, color=COLORS[name])
    ax.axhline(0, color="crimson", linewidth=1.2, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Predicted log(Sales)")
    ax.set_ylabel("Residual")

plt.suptitle("Residuals vs Predicted — all models", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "15_residuals.png", bbox_inches="tight")
plt.show()



# ══════════════════════════════════════════════════════════════════════════════
# feature importance bar charts for random forest and gradient boosting models
# ══════════════════════════════════════════════════════════════════════════════
TOP_N = 20

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
    model = results[name]["model"]
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top = importances.sort_values(ascending=False).head(TOP_N)

    bars = ax.barh(top.index[::-1], top.values[::-1], color=COLORS[name])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.set_title(f"Top {TOP_N} features — {name}")
    ax.set_xlabel("Feature Importance")

plt.suptitle("Feature importances — Random Forest vs Gradient Boosting", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "16_feature_importance.png", bbox_inches="tight")
plt.show()


# Also print top 20 for Ridge (coefficients)
ridge = results["Linear Regression (Ridge)"]["model"]
coef  = pd.Series(np.abs(ridge.coef_), index=X_train.columns)
print("\nTop 20 Ridge coefficients (absolute value):")
print(coef.sort_values(ascending=False).head(20).round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# summary
# ══════════════════════════════════════════════════════════════════════════════
print("FINAL RESULTS SUMMARY:")
summary = pd.DataFrame({
    name: {
        "MSE":      round(r["mse"],     4),
        "RMSE":     round(r["rmse"],    4),
        "R^2":       round(r["r2"],      4),
        "CV R^2 (mean)": round(r["cv_mean"], 4),
        "CV R^2 (std)":  round(r["cv_std"],  4),
    }
    for name, r in results.items()
}).T

print(summary.to_string())

best = summary["R^2"].idxmax()
print(f"\nBest model by test R^2: {best} (R^2 = {summary.loc[best, 'R^2']})")
