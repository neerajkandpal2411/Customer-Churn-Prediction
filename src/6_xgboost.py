import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

print("="*50)
print("BUILDING XGBOOST MODEL")
print("="*50)

#loading preprocessed data
X_train = pd.read_csv('results/X_train.csv')
X_test = pd.read_csv('results/X_test.csv')
y_train = pd.read_csv('results/y_train.csv').values.ravel()
y_test = pd.read_csv('results/y_test.csv').values.ravel()

#basic XGBoost model
xgb_basic = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_basic.fit(X_train, y_train)

#predicting
xgb_pred = xgb_basic.predict(X_test)
xgb_pred_proba = xgb_basic.predict_proba(X_test)[:, 1]

#evaluating basic model
print("BASIC XGBOOST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, xgb_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

#hyperparameter tuning
print("\n" + "="*30)
print("HYPERPARAMETER TUNING...")
print("="*30)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_search_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid_xgb,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search_xgb.fit(X_train, y_train)

#best XGBoost model
best_xgb = grid_search_xgb.best_estimator_
print(f"Best XGBoost parameters: {grid_search_xgb.best_params_}")

#evaluating tuned model
xgb_best_pred = best_xgb.predict(X_test)
xgb_best_proba = best_xgb.predict_proba(X_test)[:, 1]

print("\nBEST XGBOOST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, xgb_best_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, xgb_best_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_best_pred))

#feature importance
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (XGBoost):")
print(feature_importance_xgb.head(10))

#saving model and feature importance
joblib.dump(best_xgb, 'results/best_xgb_model.pkl')
feature_importance_xgb.to_csv('results/feature_importance_xgb.csv', index=False)
