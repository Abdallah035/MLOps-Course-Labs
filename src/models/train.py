# src/models/train.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow
from mlflow.models import infer_signature

def train_model(X_train, y_train, model_type="logistic_regression"):
    if model_type == "logistic_regression":
        
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
             n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            
        )
    elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")        
                
    model.fit(X_train, y_train)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    mlflow.log_param("model_type", model_type)

    mlflow.log_input(
        mlflow.data.from_pandas(X_train, name="train_data"),
        context="training",
    )

    return model