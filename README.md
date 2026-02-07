# Churn Prediction MLOps Pipeline

An end-to-end MLOps pipeline for predicting customer churn using **MLflow** for experiment tracking, model comparison, and model registry.

## Project Structure

```
MLops_V2/
├── data/
│   └── Churn_Modelling.csv          # Raw dataset
├── src/
│   ├── config/
│   │   └── settings.py              # Project-wide configuration
│   ├── data/
│   │   ├── loader.py                # Data loading utilities
│   │   └── rebalance.py             # Class rebalancing (downsampling)
│   ├── features/
│   │   └── preprocessing.py         # Feature engineering & transformations
│   ├── mlflow_utils/
│   │   └── tracking.py              # MLflow setup & configuration
│   ├── models/
│   │   ├── train.py                 # Model training (multi-model support)
│   │   └── evaluate.py              # Metrics computation
│   ├── scripts/
│   │   └── run_training.py          # Main entry point
│   └── utils/
│       └── logger.py                # Logging configuration
├── requirements.txt
└── README.md
```

## Models

The pipeline trains and compares **three** models:

| Model                | Description                                           |
|----------------------|-------------------------------------------------------|
| Logistic Regression  | Baseline linear model (`max_iter=1000`)               |
| Random Forest        | Ensemble of 200 trees (`max_depth=10`)                |
| Gradient Boosting    | Sequential boosting with 150 estimators (`lr=0.1`)    |

## Model Selection & Registration

After all runs complete, models are ranked by **F1 score** and automatically registered in MLflow:

| Alias        | Criteria        | Justification                                                                                              |
|--------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| `production` | Best F1 score   | F1 balances precision and recall — critical for churn where both false positives and false negatives are costly |
| `staging`    | 2nd best F1     | Candidate for future promotion after further validation                                                     |

> **Why F1?** In churn prediction, missing a churner (false negative) means lost revenue, while falsely flagging a loyal customer (false positive) wastes retention budget. F1 score balances both, making it the most suitable single metric.

## Setup & Usage

### 1. Create and activate a virtual environment

```bash
cd MLops_V2
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the MLflow tracking server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 4. Run the training pipeline (in a separate terminal)

```bash
venv\Scripts\activate
python -m src.scripts.run_training
```

### 5. View results

Open [http://localhost:5000](http://localhost:5000) to explore experiments, compare runs, and view registered models.

## Pipeline Steps

1. **Load** raw data from `data/Churn_Modelling.csv`
2. **Rebalance** classes via downsampling the majority class
3. **Preprocess** features — `StandardScaler` for numerical, `OneHotEncoder` for categorical
4. **Train** three models (Logistic Regression, Random Forest, Gradient Boosting)
5. **Evaluate** each model on accuracy, precision, recall, and F1 score
6. **Log** all parameters, metrics, and artifacts to MLflow
7. **Register** the top two models with `production` and `staging` aliases

## Tech Stack

- **Python 3.x**
- **scikit-learn** 1.6.1 — model training & preprocessing
- **MLflow** 2.22.0 — experiment tracking & model registry
- **pandas** 2.2.3 — data manipulation
- **NumPy** 2.2.5 — numerical operations
