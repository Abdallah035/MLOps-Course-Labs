"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import os
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    with open("col_transformer.pkl", "wb") as f:
        pickle.dump(col_transf, f)
    mlflow.log_artifact("col_transformer.pkl")
    os.remove("col_transformer.pkl")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    logger.info("  -> Fitting logistic regression...")
    log_reg.fit(X_train, y_train)
    logger.info("  -> Fit done. Inferring signature...")

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))
    logger.info("  -> Signature inferred. Logging model to MLflow...")

    # Log model
    mlflow.sklearn.log_model(log_reg, "model", signature=signature)
    logger.info("  -> Model logged. Logging training data...")

    ### Log the data
    mlflow.log_input(mlflow.data.from_pandas(X_train, name="train_data"), context="training")
    logger.info("  -> Training data logged.")

    return log_reg


def main():
    ### Set the tracking URI for MLflow
    logger.info("[1/10] Setting tracking URI...")
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    logger.info("[2/10] Setting experiment name...")
    mlflow.set_experiment("churn-prediction")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():
        logger.info("[3/10] MLflow run started. Loading data...")

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        logger.info("[4/10] Data loaded. Preprocessing...")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        logger.info("[5/10] Preprocessing done. Logging param...")

        ### Log the max_iter parameter
        mlflow.log_param("max_iter", 1000)
        logger.info("[6/10] Training model...")

        model = train(X_train, y_train)
        logger.info("[7/10] Model trained. Computing metrics...")

        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        logger.info(f"[8/10] Metrics logged - accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

        ### Log tag
        mlflow.set_tag("model_type", "LogisticRegression")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()

        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        logger.info("[9/10] Artifacts logged. Showing plot...")

        plt.show()
        logger.info("[10/10] Done!")


if __name__ == "__main__":
    main()
