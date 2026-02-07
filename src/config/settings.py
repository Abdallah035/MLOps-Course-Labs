
# Project Info

PROJECT_NAME = "Churn Prediction"
MODEL_NAME = "LogisticRegression"

# MLflow Configuration

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "churn-prediction"


# Data

RAW_DATA_PATH = "dataset/Churn_Modelling.csv"


# Model Hyperparameters
LOGISTIC_REGRESSION_MAX_ITER = 1000


# Artifacts
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
