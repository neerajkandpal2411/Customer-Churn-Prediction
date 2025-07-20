import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/telco_data.csv')

# Convert TotalCharges to numeric, forcing errors to NaN (e.g. blank or invalid entries)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# (Optional) Also convert tenure to numeric if it isn't already
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

# Drop rows where TotalCharges or tenure is NaN (caused by coercion)
df.dropna(subset=['TotalCharges', 'tenure'], inplace=True)

# Create new features
print("Creating new features...")

# Customer lifetime value per month
df['CLV_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero

# Monthly charges to total charges ratio
df['Monthly_to_Total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

# Create tenure group feature
def tenure_group(tenure):
    if tenure <= 12:
        return '0-1 Year'
    elif tenure <= 24:
        return '1-2 Years'
    elif tenure <= 48:
        return '2-4 Years'
    elif tenure <= 60:
        return '4-5 Years'
    else:
        return '5+ Years'

df['Tenure_Group'] = df['tenure'].apply(tenure_group)

# Create monthly charges group
def monthly_charges_group(m):
    if m <= 35:
        return 'Low'
    elif m <= 70:
        return 'Medium'
    else:
        return 'High'

df['Monthly_Charges_Group'] = df['MonthlyCharges'].apply(monthly_charges_group)

print("New features created!")
print(f"Dataset shape after feature engineering: {df.shape}")

# Basic visualization
plt.figure(figsize=(15, 10))

# 1. Churn distribution
plt.subplot(2, 3, 1)
df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')

# 2. Churn by tenure group
plt.subplot(2, 3, 2)
churn_by_tenure = pd.crosstab(df['Tenure_Group'], df['Churn'], normalize='index') * 100
churn_by_tenure.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
plt.title('Churn % by Tenure Group')
plt.xlabel('Tenure Group')
plt.ylabel('Percentage')
plt.legend(['No Churn', 'Churn'])

# 3. Churn by contract type
plt.subplot(2, 3, 3)
churn_by_contract = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
churn_by_contract.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
plt.title('Churn % by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Percentage')
plt.legend(['No Churn', 'Churn'])

# 4. Monthly charges distribution by churn
plt.subplot(2, 3, 4)
df[df['Churn'] == 'No']['MonthlyCharges'].hist(alpha=0.7, label='No Churn', bins=30)
df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(alpha=0.7, label='Churn', bins=30)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.legend()

# 5. Tenure distribution by churn
plt.subplot(2, 3, 5)
df[df['Churn'] == 'No']['tenure'].hist(alpha=0.7, label='No Churn', bins=30)
df[df['Churn'] == 'Yes']['tenure'].hist(alpha=0.7, label='Churn', bins=30)
plt.title('Tenure Distribution')
plt.xlabel('Tenure (months)')
plt.ylabel('Frequency')
plt.legend()

# 6. Internet service vs churn
plt.subplot(2, 3, 6)
churn_by_internet = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
churn_by_internet.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
plt.title('Churn % by Internet Service')
plt.xlabel('Internet Service')
plt.ylabel('Percentage')
plt.legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.show()

# Show correlation with churn
print("\nKey insights:")
print("1. Churn by Contract Type:")
print(churn_by_contract)
print("\n2. Churn by Internet Service:")
print(churn_by_internet)

df.to_csv('data/telco_data.csv', index=False)


from sklearn.model_selection import train_test_split

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert target variable to numeric
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical columns to dummy variables
df = pd.get_dummies(df)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to results folder
X_train.to_csv('results/X_train.csv', index=False)
X_test.to_csv('results/X_test.csv', index=False)
y_train.to_csv('results/y_train.csv', index=False)
y_test.to_csv('results/y_test.csv', index=False)

print("\n✅ Train-test files saved in 'results/' folder.")
