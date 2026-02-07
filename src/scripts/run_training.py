from src.mlflow_utils.tracking import setup_mlflow
from src.features.preprocessing import preprocess
from src.models.train import train_model
from src.models.evaluate import evaluate
from src.utils.logger import setup_logger

import pandas as pd
import mlflow
from mlflow import MlflowClient

logger = setup_logger()


def run_experiment(df, model_type, run_name):
    with mlflow.start_run(run_name=run_name):
        _, X_train, X_test, y_train, y_test = preprocess(df)

        model = train_model(X_train, y_train, model_type=model_type)

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info(f"[{run_name}] Metrics: {metrics}")
        return metrics, mlflow.active_run().info.run_id


def register_model(run_id, model_name, stage):
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.set_registered_model_alias(model_name, stage, model_version.version)
    logger.info(
        f"Model '{model_name}' version {model_version.version} set to alias '{stage}'"
    )
    return model_version


def main():
    setup_mlflow()
    df = pd.read_csv("data/Churn_Modelling.csv")

    # ── Run 1: Logistic Regression (baseline) ──
    metrics_lr, run_id_lr = run_experiment(
        df, "logistic_regression", "LogisticRegression_baseline"
    )

    # ── Run 2: Random Forest ──
    metrics_rf, run_id_rf = run_experiment(
        df, "random_forest", "RandomForest_run"
    )

    # ── Run 3: Gradient Boosting ──
    metrics_gb, run_id_gb = run_experiment(
        df, "gradient_boosting", "GradientBoosting_run"
    )

    # ── Compare & Register ──
    results = {
        "LogisticRegression": (metrics_lr, run_id_lr),
        "RandomForest": (metrics_rf, run_id_rf),
        "GradientBoosting": (metrics_gb, run_id_gb),
    }

    logger.info("\n===== Results Summary =====")
    for name, (m, _) in results.items():
        logger.info(f"{name}: F1={m['f1_score']:.4f}  Acc={m['accuracy']:.4f}  "
                     f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")

    # Sort by F1 score (best overall metric for imbalanced data)
    sorted_models = sorted(results.items(), key=lambda x: x[1][0]["f1_score"], reverse=True)

    best_name, (best_metrics, best_run_id) = sorted_models[0]
    second_name, (second_metrics, second_run_id) = sorted_models[1]

    model_name = "churn-prediction-model"

    # Best F1 → Production (F1 balances precision & recall, critical for churn)
    logger.info(f"\nProduction  → {best_name} (F1={best_metrics['f1_score']:.4f})")
    register_model(best_run_id, model_name, "production")

    # Second best F1 → Staging (candidate for future promotion)
    logger.info(f"Staging     → {second_name} (F1={second_metrics['f1_score']:.4f})")
    register_model(second_run_id, model_name, "staging")


if __name__ == "__main__":
    main()
