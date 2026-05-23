"""
Locust load test for the Churn Prediction API.

Run (UI mode — what you screenshot for the deliverable):
    uv run locust --host http://localhost:8000

Run (headless — for a reproducible CI-style number):
    uv run locust --host http://localhost:8000 \
        --users 200 --spawn-rate 20 --run-time 2m --headless

Then open http://localhost:8089 (the Locust UI, not the app).
"""

import random

from locust import HttpUser, between, task


# Pool of valid payloads. We pre-build these so the test measures the API,
# not Python's random.choice overhead. Geography and Gender are varied so
# the pipeline's OneHotEncoder branches all get exercised.
PAYLOADS = [
    {
        "CreditScore": 650, "Geography": "France",  "Gender": "Female",
        "Age": 40, "Tenure": 3, "Balance": 60000,
        "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
        "EstimatedSalary": 50000,
    },
    {
        "CreditScore": 720, "Geography": "Germany", "Gender": "Male",
        "Age": 55, "Tenure": 8, "Balance": 125000,
        "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0,
        "EstimatedSalary": 90000,
    },
    {
        "CreditScore": 480, "Geography": "Spain",   "Gender": "Female",
        "Age": 28, "Tenure": 1, "Balance": 0,
        "NumOfProducts": 3, "HasCrCard": 0, "IsActiveMember": 1,
        "EstimatedSalary": 35000,
    },
]


class ChurnUser(HttpUser):
    # Each simulated user waits 0.5–1.5s between requests. With 100 users that
    # gives ~67–200 RPS of *offered* load — the actual served RPS is what we
    # report as peak.
    wait_time = between(0.5, 1.5)

    @task(8)  # weight 8 → ~80% of requests
    def predict(self):
        # name= groups all /predict calls under one row in the UI even if the
        # body differs. Without it Locust would split by URL only (still one
        # row here, but the habit matters once you have path params).
        self.client.post("/predict", json=random.choice(PAYLOADS), name="/predict")

    @task(1)  # ~10%
    def health(self):
        self.client.get("/health")

    @task(1)  # ~10%
    def home(self):
        self.client.get("/")
