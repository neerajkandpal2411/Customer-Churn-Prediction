import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Feature importance (columns names are not available after preprocessing, so we use indices)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': [f'col_{i}' for i in range(len(importances))],
                           'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    return model, imp_df

if __name__ == "__main__":
    X_train = pd.read_csv("../data/X_train.csv").values
    y_train = pd.read_csv("../data/y_train.csv").values.ravel()
    model, _ = train_random_forest(X_train, y_train)
    joblib.dump(model, "../data/results/best_rf_model.pkl")