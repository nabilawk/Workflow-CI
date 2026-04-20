import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# 1. Load Dataset
df = pd.read_csv("namadataset_preprocessing/diabetes_clean.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 2. MLflow Setup
mlflow.set_experiment("RandomForest Model")

mlflow.sklearn.autolog()

# 3. Training Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluasi
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# Tambahan metric manual
mlflow.log_metric("test_accuracy", acc)
mlflow.log_metric("test_auc", auc)


# 5. Artefak Tambahan
# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

mlflow.log_artifact("confusion_matrix.png")

# Feature Importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance.to_csv("feature_importance.csv", index=False)

mlflow.log_artifact("feature_importance.csv")

# Sample Prediction
sample_pred = X_test.copy()
sample_pred["Actual"] = y_test.values
sample_pred["Prediction"] = y_pred

sample_pred.head(20).to_csv("sample_prediction.csv", index=False)

mlflow.log_artifact("sample_prediction.csv")

# 6. Output Console
print("Training selesai")
print(f"Accuracy : {acc:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print("Artefak tersimpan:")
print("- confusion_matrix.png")
print("- feature_importance.csv")
print("- sample_prediction.csv")
