# regression_model.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import time

# Logging setup
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

log("🚀 Starting regression script...")
log("📥 Loading data...")
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# === Preprocess Genres ===
log("🔧 Preprocessing genres...")
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
genre_dummies = movies['genres'].str.get_dummies(sep=' ')

log("🔗 Merging ratings and movie genre features...")
data = pd.merge(ratings, movies[['movieId']], on='movieId')
data = pd.merge(data, genre_dummies, left_on='movieId', right_index=True)

# === Define X and y ===
X = data[genre_dummies.columns]
y = data['rating']

# === Train/Test Split ===
log("✂ Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Definitions ===
log("🧠 Defining models...")
models = {
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
}

results = []
os.makedirs("plots", exist_ok=True)

# === Train and Evaluate Models ===
for name, model in models.items():
    log(f"🔄 Training model: {name}")
    model.fit(X_train, y_train)

    log(f"📈 Predicting with {name}...")
    y_pred = model.predict(X_test)

    # === Metrics ===
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log(f"✅ {name} - RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
    results.append({
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    })

    # === Plot: Prediction vs Actual ===
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"{name} - Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_predicted_vs_actual.png")
    plt.close()
    log(f"📸 Saved plot: {name}_predicted_vs_actual.png")

    # === Plot: Feature Importance ===
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 5))
        importances = model.feature_importances_
        plt.bar(X.columns, importances)
        plt.xticks(rotation=90)
        plt.title(f"{name} - Feature Importance")
        plt.tight_layout()
        plt.savefig(f"plots/{name}_feature_importance.png")
        plt.close()
        log(f"📸 Saved feature importance plot for {name}")

# === Save Results ===
log("💾 Saving results to CSV...")
pd.DataFrame(results).to_csv("plots/regression_model_results.csv", index=False)

log("🏁 Script complete!")