import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib  # For loading the scaler

# Define paths
MODEL_PATH = "models/churn_model.h5"
SCALER_PATH = "models/scaler.pkl"

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Load the trained scaler (must be saved during preprocessing)
scaler = joblib.load(SCALER_PATH)

# Function to process input and predict churn


def preprocess_input(geography, gender, credit_score, age, tenure, balance, num_of_products, estimated_salary):
    """
    Converts user input into the format expected by the trained model.
    """
    # One-hot encoding (Geography)
    geo_germany = 1 if geography.lower() == "germany" else 0
    geo_spain = 1 if geography.lower() == "spain" else 0

    # One-hot encoding (Gender)
    gender_male = 1 if gender.lower() == "male" else 0

    # Create input array (matching the order in X_train)
    input_data = np.array([[geo_germany, geo_spain, gender_male, credit_score,
                          age, tenure, balance, num_of_products, estimated_salary]])

    # Scale numerical values using the same scaler used in preprocessing
    input_data[:, 3:] = scaler.transform(input_data[:, 3:])

    return input_data


def predict_churn(geography, gender, credit_score, age, tenure, balance, num_of_products, estimated_salary):
    """
    Predicts whether a customer will churn based on input features.
    """
    processed_input = preprocess_input(
        geography, gender, credit_score, age, tenure, balance, num_of_products, estimated_salary)
    prediction = (model.predict(processed_input) > 0.5).astype("int32")[0][0]
    return int(prediction)


if __name__ == "__main__":
    # Example input (replace with user-provided values)
    sample_input = ("France", "Male", 600, 35, 3, 125000, 2, 60000)

    # Make prediction
    result = predict_churn(*sample_input)
    print(f"Predicted Churn: {result} (1 = Churn, 0 = No Churn)")
