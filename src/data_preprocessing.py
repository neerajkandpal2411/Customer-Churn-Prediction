import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

def preprocess_data(input_path, preprocessor_save_path):
    df = pd.read_csv(input_path)

    # Separate target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to binary

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_save_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    # Save split data
    pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data("../data/engineered_data.csv", "../data/results/preprocessor.pkl")