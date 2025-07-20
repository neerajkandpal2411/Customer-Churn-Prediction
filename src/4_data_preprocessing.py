import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/telco_data.csv')

print("Starting data preprocessing...")

#creating a copy for preprocessing
df_ml = df.copy()

#droping customerID as it is not useful for prediction
df_ml = df_ml.drop('customerID', axis=1)

#converting target variable to binary
label_encoder = LabelEncoder()
df_ml['Churn'] = label_encoder.fit_transform(df_ml['Churn'])  # No=0, Yes=1

print(f"Churn mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

#handling categorical variables
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod', 'Tenure_Group', 'Monthly_Charges_Group']

#one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_ml, columns=categorical_features, drop_first=True)

print(f"Dataset shape after encoding: {df_encoded.shape}")
print(f"New features after encoding: {df_encoded.columns.tolist()}")

#separating features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

#spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training set churn distribution: {y_train.value_counts()}")
print(f"Test set churn distribution: {y_test.value_counts()}")

#scalinge numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV_per_month', 'Monthly_to_Total_ratio']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print("Data preprocessing completed!")
print(f"Final features: {X_train.columns.tolist()}")
