import pandas as pd
import joblib

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("BUILDING RANDOM FOREST MODEL")
print("="*50)

# Load preprocessed data
X_train = pd.read_csv('results/X_train.csv')
X_test = pd.read_csv('results/X_test.csv')
y_train = pd.read_csv('results/y_train.csv').values.ravel()
y_test = pd.read_csv('results/y_test.csv').values.ravel()

# Basic Random Forest
rf_basic = RandomForestClassifier(random_state=42, n_estimators=100)
rf_basic.fit(X_train, y_train)

# Make predictions
rf_pred = rf_basic.predict(X_test)
rf_pred_proba = rf_basic.predict_proba(X_test)[:, 1]

# Evaluate basic model
print("BASIC RANDOM FOREST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Hyperparameter tuning for Random Forest
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

# Best Random Forest model
best_rf = grid_search_rf.best_estimator_
print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")

# Evaluate best Random Forest
rf_best_pred = best_rf.predict(X_test)
rf_best_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nBEST RANDOM FOREST RESULTS:")
print(f"Accuracy: {accuracy_score(y_test, rf_best_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_best_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_best_pred))

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (Random Forest):")
print(feature_importance_rf.head(10))

# Save model and feature importance (optional)
joblib.dump(best_rf, 'results/best_rf_model.pkl')
feature_importance_rf.to_csv('results/feature_importance_rf.csv', index=False)



# In this step, I trained a Random Forest Classifier to predict churn. 
# I began with a basic model and evaluated it using accuracy, ROC AUC, and classification metrics. 
# Then I used GridSearchCV for hyperparameter tuning to improve performance. 
# The tuned model gave better ROC AUC (0.83) and more balanced precision-recall for churn. 
# Finally, I extracted feature importance, which revealed key drivers of churn like contract type, tenure, and charges.

# We set random_state=42 to ensure reproducibility of results. 
# Since Random Forest uses internal randomness, fixing the random state ensures the model behaves consistently across multiple runs — 
# which is especially useful during debugging, comparison, and deployment.

# I used predict() to generate class labels (0 or 1) for evaluation metrics like accuracy and confusion matrix. 
# Then I used predict_proba() to get class 1 probabilities, which are required for calculating the ROC AUC score — 
# since AUC works with predicted probabilities rather than binary labels.

# We used multiple evaluation metrics to assess model performance. 
# Accuracy gave us a basic measure of correctness, but since our dataset was imbalanced, we relied more on the ROC AUC score, which was 0.81 — 
# indicating that the model was able to distinguish churners and non-churners quite well. 
# We also used a confusion matrix to analyze false positives and false negatives, which helped us understand where the model was making mistakes.

# We used metrics like precision, recall, F1-score, and support to evaluate how well the model performs for both churn and non-churn classes. 
# Since the dataset is imbalanced, F1-score and recall were especially important — 
# they gave us a clearer view of how many churners we were correctly catching and how many we were missing.

# Our model had decent accuracy but relatively low recall, especially for the churn class, which is crucial in customer churn prediction. 
# So, we used GridSearchCV to tune hyperparameters of the Random Forest model, focusing on maximizing recall. 
# This approach helped us systematically explore different parameter combinations and find the one that best improved our recall performance without overfitting.

# Model parameters are internal variables that the model learns from the training data, such as weights in linear regression or split conditions in decision trees. 
# Hyperparameters, on the other hand, are set before training starts and control the learning process—like the number of trees in a random forest. 
# We can tune hyperparameters to improve model performance.

# Overfitting occurs when the model learns not just the patterns but also the noise in the training data, leading to poor generalization. 
# Underfitting happens when the model is too simplistic to capture the true trends in the data. 
# A good model balances both, and techniques like cross-validation help identify that balance.

# We used 5-fold cross-validation to ensure that our model's performance wasn’t dependent on a particular train-test split. 
# By training and validating the model on different subsets, we get a more robust and generalized performance estimate. 
# This helps in avoiding both overfitting and underfitting.