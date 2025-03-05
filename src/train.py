import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.tensorflow

# Enable MLflow autologging
mlflow.tensorflow.autolog()

# Define paths
PROCESSED_DATA_PATH = "data/processed/"
X_train_path = os.path.join(PROCESSED_DATA_PATH, "X_train.csv")
X_test_path = os.path.join(PROCESSED_DATA_PATH, "X_test.csv")
y_train_path = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
y_test_path = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

# Load preprocessed data
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test = pd.read_csv(y_test_path).values.ravel()

# Define ANN model


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_shape,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Start MLflow tracking
with mlflow.start_run():
    model = build_model(X_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50,
                        batch_size=32, validation_data=(X_test, y_test))

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")

    # Log accuracy manually in MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Save the trained model
    model.save("models/churn_model.h5")

print("Training completed and model saved!")
