import pandas as pd

# Load the dataset
df = pd.read_csv('data/telco_data.csv')

# Basic dataset information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check all column names
print("\nAll Columns:")
print(df.columns.tolist())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check target variable distribution
print("\nChurn Distribution:")
print(df['Churn'].value_counts())

# Check churn percentage
print("\nChurn Percentage:")
churn_percentage = df['Churn'].value_counts(normalize=True) * 100
print(churn_percentage)

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])  # Only show columns with missing values

