import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("sensor_data.csv")
X = df.drop("status", axis=1)
y = df["status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\nModel Performance:")
print(classification_report(
    y_test, y_pred,
    target_names=["healthy", "warning", "failure"]
))

# Feature importance
features = X.columns.tolist()
importance = model.feature_importances_
print("\nFeature Importance:")
for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel saved to model.pkl")