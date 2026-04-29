import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score
import joblib

def tune_random_forest(X_train, y_train):
    """
    Perform grid search on Random Forest with class balancing.
    Returns the best model and a DataFrame of importance.
    """
    # Parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    # Use stratified k-fold to preserve imbalanced ratio
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Scorer: F1-score (good for imbalanced data)
    scorer = make_scorer(f1_score)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Feature importance
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': [f'col_{i}' for i in range(len(importances))],
        'importance': importances
    }).sort_values('importance', ascending=False)

    return best_model, imp_df, grid.best_params_, grid.best_score_


def tune_xgboost(X_train, y_train):
    """
    Perform grid search on XGBoost with scale_pos_weight balancing.
    Returns the best model and importance DataFrame.
    """
    # Compute scale_pos_weight from class ratio
    neg, pos = np.bincount(y_train)
    scale_weight = neg / pos

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score)
    grid = GridSearchCV(xgb_model, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': [f'col_{i}' for i in range(len(importances))],
        'importance': importances
    }).sort_values('importance', ascending=False)

    return best_model, imp_df, grid.best_params_, grid.best_score_