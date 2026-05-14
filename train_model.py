import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("mudra_landmarks.csv")

# Using all 50 classes — dataset quality is strong across all
# (Smallest class Hamsapaksha still has 177 samples)
print(f"Total samples: {len(df)}, Total classes: {df['label'].nunique()}")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

y_pred = model.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("mudra_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nFine-tuned model saved as mudra_model.pkl")
