from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    real_csv = DATA / "netflix_titles.csv"
    demo_csv = DATA / "demo_netflix_titles.csv"
    using_demo = not real_csv.exists()
    csv_path = demo_csv if using_demo else real_csv
    df = pd.read_csv(csv_path)

    if "release_year" not in df.columns:
        raise ValueError("CSV must contain a 'release_year' column.")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"]).copy()
    df["release_year"] = df["release_year"].astype(int)
    df = df[df["release_year"].between(1920, pd.Timestamp.now().year)]
    if df.empty:
        raise ValueError("No valid release years found.")

    counts = df.groupby("release_year").size().sort_index()
    years = pd.RangeIndex(counts.index.min(), counts.index.max() + 1, name="release_year")
    yearly = counts.reindex(years, fill_value=0).rename("title_count").reset_index()
    yearly["year_index"] = yearly["release_year"] - yearly["release_year"].min()
    yearly["year_squared"] = yearly["year_index"] ** 2
    yearly.to_csv(OUT / "yearly_release_counts.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(yearly.release_year, yearly.title_count, marker="o")
    plt.title("Netflix Titles by Release Year")
    plt.xlabel("Release year")
    plt.ylabel("Title count")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / "historical_release_trend.png", dpi=160)
    plt.close()

    if len(yearly) < 8:
        raise ValueError("Need at least 8 distinct release years to evaluate models.")
    X = yearly[["year_index", "year_squared"]].astype(float)
    y = yearly["title_count"].astype(float)
    split = max(5, int(len(yearly) * .8))
    if split >= len(yearly):
        split = len(yearly) - 2
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        rows.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** .5})
    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    test_pred = np.maximum(0, best_model.predict(X_test))

    plt.figure(figsize=(9, 5))
    plt.plot(yearly.release_year.iloc[split:], y_test, marker="o", label="Actual test values")
    plt.plot(yearly.release_year.iloc[split:], test_pred, marker="s", label=f"Predicted ({best_name})")
    plt.title("Model Evaluation: Actual vs Predicted")
    plt.xlabel("Release year")
    plt.ylabel("Title count")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / "model_evaluation.png", dpi=160)
    plt.close()

    best_model.fit(X, y)
    last_year = int(yearly.release_year.max())
    first_year = int(yearly.release_year.min())
    future_years = np.arange(last_year + 1, last_year + 6)
    future_idx = future_years - first_year
    future_X = pd.DataFrame({"year_index": future_idx, "year_squared": future_idx ** 2})
    future_count = np.rint(np.maximum(0, best_model.predict(future_X))).astype(int)
    forecast = pd.DataFrame({"year": future_years, "forecast_title_count": future_count})
    forecast.to_csv(OUT / "future_forecast.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(yearly.release_year, y, marker="o", label="Historical")
    plt.plot(forecast.year, forecast.forecast_title_count, marker="s", linestyle="--", label="Forecast")
    plt.axvline(last_year, linestyle=":", linewidth=1)
    plt.title(f"Five-Year Trend Forecast ({best_name})")
    plt.xlabel("Release year")
    plt.ylabel("Title count")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / "future_forecast.png", dpi=160)
    plt.close()

    with open(OUT / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Chronological holdout metrics\\n")
        f.write(metrics.to_string(index=False))
        f.write(f"\\n\\nSelected model: {best_name}\\n")
        f.write("MAE and RMSE: lower values generally mean smaller errors.\\n")
        if using_demo:
            f.write("\\nWARNING: Demo data used. Replace with real Netflix CSV for valid assignment results.\\n")

    print("Dataset:", csv_path.name)
    print("Rows used:", len(df))
    if using_demo:
        print("NOTE: Demo data used. Add data/netflix_titles.csv for real results.")
    print("\\nModel metrics:\\n", metrics.to_string(index=False))
    print("\\nFive-year forecast:\\n", forecast.to_string(index=False))
    print("\\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
