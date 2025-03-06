# Customer Churn Prediction using DVC, MLflow, and TensorFlow

A machine learning pipeline for predicting customer churn using Deep Learning (ANN). This project incorporates DVC for data versioning, MLflow for experiment tracking, and FastAPI for deployment.

## Key Features
- ✅ Deep Learning Model (ANN) - Built with TensorFlow
- ✅ Data Versioning with DVC - Tracks raw & processed data
- ✅ Experiment Tracking with MLflow - Logs model metrics
- ✅ FastAPI Deployment - REST API for model predictions

## Setup the Project from Scratch

1. **Clone the Repository**
  ```bash
  git clone https://github.com/SA-K1B/customer-churn-prediction-dvc-mlflow-tensorflow
  cd customer-churn-prediction-dvc-mlflow-tensorflow
  ```

2. **Create and Activate Virtual Environment**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On macOS/Linux
  venv\Scripts\activate     # On Windows
  ```

3. **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```


4. **Retrieve Dataset from DVC Storage**
  ```bash
  dvc pull
  ```
  This will download the dataset stored in Google Drive.

5. **Reproduce the Entire ML Pipeline**
  ```bash
  dvc repro
  ```
  This runs:
  - Preprocessing (`preprocess.py`) → Generates processed data.
  - Training (`train.py`) → Trains & saves the model.

## Running the FastAPI Server

To deploy the model as an API:
```bash
uvicorn src.app:app --reload
```
The API will be accessible at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Test endpoint: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)

### Example API Request
Example API Request using Postman:

1. Open Postman.
2. Create a new POST request.
3. Set the URL to [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict).
4. Go to the "Headers" tab and set `Content-Type` to `application/json`.
5. Go to the "Body" tab, select "raw" and paste the following JSON:

  ```json
  {
    "Geography": "France",
    "Gender": "Male",
    "CreditScore": 600,
    "Age": 35,
    "Tenure": 3,
    "Balance": 125000,
    "NumOfProducts": 2,
    "EstimatedSalary": 60000
  }
  ```

Expected response:
```json
{
  "churn_prediction": 0,
  "message": "Customer will stay"
}
```

## Tracking Experiments with MLflow

Launch the MLflow UI to track model metrics:
```bash
mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

## Connect with Me on LinkedIn

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/tarequl-hasan-sakib/)