# src/features/preprocessing.py
import pandas as pd
import pickle
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mlflow

from src.data.rebalance import rebalance


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
