import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("BUILDING RANDOM FOREST MODEL")
print("="*50)

#loading preprocessed data
X_train = pd.read_csv('results/X_train.csv')
X_test = pd.read_csv('results/X_test.csv')
y_train = pd.read_csv('results/y_train.csv').values.ravel()
y_test = pd.read_csv('results/y_test.csv').values.ravel()

#basic Random Forest model
rf_basic = RandomForestClassifier(random_state=42, n_estimators=100)
rf_basic.fit(X_train, y_train)

#making predictions
rf_pred = rf_basic.predict(X_test)
rf_pred_proba = rf_basic.predict_proba(X_test)[:, 1]

#evaluating basic model
print("BASIC RANDOM FOREST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

#hyperparameter tuning for Random Forest
print("\n" + "="*30)
print("HYPERPARAMETER TUNING...")
print("="*30)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

#best Random Forest model
best_rf = grid_search_rf.best_estimator_
print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")

#evaluating best Random Forest
rf_best_pred = best_rf.predict(X_test)
rf_best_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nBEST RANDOM FOREST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, rf_best_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_best_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_best_pred))

#feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (Random Forest):")
print(feature_importance_rf.head(10))

#saving model and feature importance
joblib.dump(best_rf, 'results/best_rf_model.pkl')
feature_importance_rf.to_csv('results/feature_importance_rf.csv', index=False)
