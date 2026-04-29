import pandas as pd
import numpy as np

def add_engineered_features(df):
    """
    Apply feature engineering to a dataframe.
    Returns the dataframe with new columns added.
    """
    # Avoid SettingWithCopyWarning by making a copy if needed
    df = df.copy()

    # 1. AvgServiceSpend
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['AvgServiceSpend'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # 2. Ratio of Monthly to Total charges
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['Monthly_to_Total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # 3. Customer Lifetime Value per month (CLV)
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['CLV_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # 4. Tenure groups
    if 'tenure' in df.columns:
        bins = [-1, 12, 24, 48, 100]
        labels = ['0-12', '13-24', '25-48', '49+']
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # 5. Monthly charges groups
    if 'MonthlyCharges' in df.columns:
        bins = [0, 30, 60, 90, 200]
        labels = ['Low', 'Medium', 'High', 'Very High']
        df['Monthly_Charges_Group'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels)

    return df


def engineer_features(input_path, output_path):
    """
    Read CSV, apply feature engineering, save to output_path.
    (Used by the pipeline page.)
    """
    df = pd.read_csv(input_path)
    df = add_engineered_features(df)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    engineer_features("../data/cleaned_data.csv", "../data/engineered_data.csv")