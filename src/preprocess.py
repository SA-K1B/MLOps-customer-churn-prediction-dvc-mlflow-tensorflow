import pandas as pd
import joblib  # To save scaler and encoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("data/raw/Churn_Modelling.csv")

# Drop unnecessary columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Identify categorical and numerical features
categorical_features = ["Geography", "Gender"]
numerical_features = ["CreditScore", "Age", "Tenure",
                      "Balance", "NumOfProducts", "EstimatedSalary"]

# Define transformers
# Ensure dense array output
encoder = OneHotEncoder(drop="first", sparse_output=False)
scaler = StandardScaler()

# Apply transformations
preprocessor = ColumnTransformer([
    ("cat", encoder, categorical_features),
    ("num", scaler, numerical_features)
])

# Prepare data
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Fit the preprocessor
preprocessor.fit(X)

# Extract and save fitted encoder and scaler
encoder_fitted = preprocessor.named_transformers_['cat']
scaler_fitted = preprocessor.named_transformers_['num']

joblib.dump(encoder_fitted, "models/encoder.pkl")
joblib.dump(scaler_fitted, "models/scaler.pkl")

print("✅ Encoder and Scaler saved successfully!")

# Transform the data
X_processed = preprocessor.transform(X)

# Get feature names from fitted ColumnTransformer
new_feature_names = encoder_fitted.get_feature_names_out(
    categorical_features).tolist() + numerical_features

# Convert transformed data into a DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=new_feature_names)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42)

# Save processed data in data/processed
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Print columns of X_train
# print("X_train columns:")
# print(X_train.columns)
# # Print first 5 rows of X_train
# print("X_train head:")
# print(X_train.head())

print("✅ Data preprocessing completed and saved!")
