# classification_model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

print("✅ Loading data...")
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("✅ Preprocessing...")
ratings['liked'] = (ratings['rating'] >= 4).astype(int)
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
genre_dummies = movies['genres'].str.get_dummies(sep=' ')

print("✅ Merging datasets...")
data = pd.merge(ratings, movies[['movieId']], on='movieId')
data = pd.merge(data, genre_dummies, left_on='movieId', right_index=True)

X = data[genre_dummies.columns]
y = data['liked']

print("✅ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Defining models...")
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

results = []

for name, model in models.items():
    print(f"\n🧠 Training model: {name}...")
    model.fit(X_train, y_train)

    print(f"🔍 Predicting with {name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"📊 Generating classification report for {name}...")
    print(classification_report(y_test, y_pred))

    print(f"📉 Plotting confusion matrix for {name}...")
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.ax_.set_title(f"{name} - Confusion Matrix")
    for texts in disp.text_.ravel():
        texts.set_text(f"{int(float(texts.get_text())):,}")
    plt.title(f"{name} - Confusion Matrix")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}_confusion_matrix.png")
    plt.close()

    print(f"📈 Computing ROC & PR for {name}...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(2)
    plt.plot(recall, precision, label=f"{name}")

    results.append({
        "model": name,
        "roc_auc": roc_auc,
        "accuracy": model.score(X_test, y_test)
    })
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{name} - Precision vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_precision_threshold_curve.png")
    plt.close()

print("🖼️ Saving ROC Curve...")
plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/classification_roc_curve.png")
plt.close()

print("🖼️ Saving Precision-Recall Curve...")
plt.figure(2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/classification_pr_curve.png")
plt.close()

print("💾 Saving model results to CSV...")
pd.DataFrame(results).to_csv("plots/classification_model_results.csv", index=False)

print("✅ All done!")
