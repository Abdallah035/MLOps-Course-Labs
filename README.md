# MLOps Course Labs — Bank Customer Churn Prediction API

**Author:** Abdallah Mohamed
Lab repository for the [MLOps Course](https://github.com/Heba-Atef99/MLOps-Course).

This project serves a **Bank Customer Churn Prediction** model (a GradientBoosting
champion selected via MLflow) through a **Litestar** REST API with logging, tests,
and optional HyperDX live observability.

## Setup

```bash
uv sync
uv run pre-commit install
# the trained model lives in data/model.joblib
uv run litestar --app main:app run --reload
```

Swagger UI: http://localhost:8000/schema/swagger

## Tests

```bash
uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
```

Current coverage: **>90%** (9 tests).

## Endpoints

| Method | Path       | Description                          |
| ------ | ---------- | ------------------------------------ |
| GET    | `/`        | Welcome message + endpoint list      |
| GET    | `/health`  | Health check                         |
| POST   | `/predict` | Returns churn prediction (0/1)       |

### Example `/predict` request body

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

## HyperDX live logs (bonus)

Set the API key before starting the server to ship logs/traces to HyperDX:

```bash
export HYPERDX_API_KEY="your-key"
export OTEL_SERVICE_NAME="churn-api"
uv run litestar --app main:app run
```
