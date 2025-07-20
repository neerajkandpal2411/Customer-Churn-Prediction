import pandas as pd

#dataset
df = pd.read_csv('data/telco_data.csv')

# Basic info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

#all column names
print("\nAll Columns:")
print(df.columns.tolist())

#data types names
print("\nData Types:")
print(df.dtypes)

#target variable distribution
print("\nChurn Distribution:")
print(df['Churn'].value_counts())

#churn percentage
print("\nChurn Percentage:")
churn_percentage = df['Churn'].value_counts(normalize=True) * 100
print(churn_percentage)

#missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

