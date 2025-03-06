from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_churn  # Importing the function from predict.py

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Define request body structure using Pydantic


class CustomerData(BaseModel):
    Geography: str
    Gender: str
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    EstimatedSalary: float


@app.get("/")
def home():
    """
    API Health Check
    """
    return {"message": "Welcome to Customer Churn Prediction"}


@app.post("/predict")
def predict(data: CustomerData):
    """
    Takes customer data as input and returns churn prediction.
    """
    prediction = predict_churn(
        data.Geography, data.Gender, data.CreditScore, data.Age,
        data.Tenure, data.Balance, data.NumOfProducts, data.EstimatedSalary
    )

    result = "Customer will churn" if prediction == 1 else "Customer will stay"

    return {
        "churn_prediction": prediction,
        "message": result
    }
