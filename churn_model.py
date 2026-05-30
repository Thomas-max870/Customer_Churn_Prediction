from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls")

# Convert Churn to Numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Remove customerID
df = df.drop("customerID", axis=1)

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# Logistic Regression
# =====================================
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_pred))

# =====================================
# Random Forest
# =====================================
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# =====================================
# XGBoost
# =====================================
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_pred))

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# =====================================
# XGBoost Confusion Matrix
# =====================================
cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/xgboost_confusion_matrix.png")

plt.show()