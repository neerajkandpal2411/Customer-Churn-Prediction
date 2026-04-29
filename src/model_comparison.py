import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

def compare_models():
    # This function can be enhanced to return figures; used in UI
    X_test = pd.read_csv("data/X_test.csv").values
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    rf = joblib.load("data/results/best_rf_model.pkl")
    xgb_model = joblib.load("data/results/best_xgb_model.pkl")

    pred_rf = rf.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)
    print("RF Report:\n", classification_report(y_test, pred_rf))
    print("XGB Report:\n", classification_report(y_test, pred_xgb))

    # ROC curves
    plt.figure()
    for name, model in [("RF", rf), ("XGB", xgb_model)]:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig("data/results/roc_curve.png")
    plt.close()

if __name__ == "__main__":
    compare_models()