import pandas as pd

df = pd.read_csv('data/telco_data.csv')

#basic statistics
print("\nBasic Statistics:")
print(df.describe())

#checking TotalCharges column for issues
print("\nTotalCharges unique values (first 20):")
print(df['TotalCharges'].unique()[:20])

#checking for empty strings or spaces in TotalCharges
print("\nChecking for empty/space values in TotalCharges:")
empty_total_charges = df[df['TotalCharges'] == ' ']
print(f"Empty TotalCharges count: {len(empty_total_charges)}")

#converting TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#cxhecking for missing values again
print("\nMissing values after conversion:")
missing_after_conversion = df.isnull().sum()
print(missing_after_conversion[missing_after_conversion > 0])

#handling missing values in TotalCharges
# For customers with 0 tenure, TotalCharges should be 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

#verifying the fix
print("\nTotalCharges data type after fixing:")
print(df['TotalCharges'].dtype)
print(f"Missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")
