import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

print("="*50)
print("COMPARING RANDOM FOREST AND XGBOOST MODELS")
print("="*50)

#loading test data
X_test = pd.read_csv('results/X_test.csv')
y_test = pd.read_csv('results/y_test.csv').values.ravel()

#loading models
rf_model = joblib.load('results/best_rf_model.pkl')
xgb_model = joblib.load('results/best_xgb_model.pkl')

#predictions
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

#accuracy and ROC AUC
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

#printing Results
print(f"\nRandom Forest - Accuracy: {rf_acc:.4f}, ROC AUC: {rf_auc:.4f}")
print(f"XGBoost       - Accuracy: {xgb_acc:.4f}, ROC AUC: {xgb_auc:.4f}")

#conclusion
print("\nConclusion:")
if rf_auc > xgb_auc:
    print("✅ Random Forest performed better based on ROC AUC.")
elif xgb_auc > rf_auc:
    print("✅ XGBoost performed better based on ROC AUC.")
else:
    print("🔁 Both models performed equally based on ROC AUC.")
