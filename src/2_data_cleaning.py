import pandas as pd

df = pd.read_csv('data/telco_data.csv')

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check TotalCharges column for issues
print("\nTotalCharges unique values (first 20):")
print(df['TotalCharges'].unique()[:20])

# Check for empty strings or spaces in TotalCharges
print("\nChecking for empty/space values in TotalCharges:")
empty_total_charges = df[df['TotalCharges'] == ' ']
print(f"Empty TotalCharges count: {len(empty_total_charges)}")

# Convert TotalCharges to numeric (this will show the real missing values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now check for missing values again
print("\nMissing values after conversion:")
missing_after_conversion = df.isnull().sum()
print(missing_after_conversion[missing_after_conversion > 0])

# Handle missing values in TotalCharges
# For customers with 0 tenure, TotalCharges should be 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Verify the fix
print("\nTotalCharges data type after fixing:")
print(df['TotalCharges'].dtype)
print(f"Missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")