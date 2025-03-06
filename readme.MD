# Customer Churn Prediction

This project predicts customer churn using deep learning, with tools such as DVC,MLflow, and FastAPI to streamline workflow and ensure reproducibility.

## Tools Used

### DVC (Data Version Control)
DVC is utilized for managing data and models, ensuring reproducibility of experiments. The `dvc.yaml` file outlines the steps for data preparation and model training.

### MLflow
MLflow is employed to track experiments, logging essential information such as parameters and metrics. This enables comparison of different model runs and evaluation of performance.

### TensorFlow
TensorFlow serves as the primary library for building and training the predictive model. It facilitates the creation of neural networks, which are used to predict customer churn.

### FastAPI
FastAPI is used to develop a web API for the model, allowing users to submit customer data and receive predictions efficiently. The API includes endpoints for health checks and making predictions.

## How to Use

1. **Clone the repository**: 
   ```bash
   git clone https://github.com/SA-K1B/MLOps-customer-churn-prediction-dvc-mlflow-tensorflow
   cd customer-churn-prediction-dvc-mlflow-tensorflow
   ```

2. **Install dependencies**: 
   Install the required libraries listed in `requirements.txt`.

3. **Run DVC pipeline**: 
   Prepare the data and train the model using:
   ```bash
   dvc repro
   ```

4. **Start the FastAPI server**: 
   Launch the FastAPI application to access the prediction API:
   ```bash
   uvicorn src.app:app --reload
   ```

5. **Make predictions**: 
   Use the `/predict` endpoint to send customer data and receive churn predictions.
