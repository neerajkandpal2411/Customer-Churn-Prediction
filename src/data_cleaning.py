import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Convert TotalCharges to numeric, coerce errors
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Use assignment instead of inplace=True
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    # Drop customerID if present (not useful)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    # Any other cleaning (e.g., replace 'No internet service' with 'No' for multiple columns)
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    clean_data("../data/telco_data.csv", "../data/cleaned_data.csv")