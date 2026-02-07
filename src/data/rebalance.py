from sklearn.utils import resample
import pandas as pd


def rebalance(data: pd.DataFrame) -> pd.DataFrame:
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]

    churn_maj, churn_min = (
        (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
    )

    churn_maj_down = resample(
        churn_maj,
        n_samples=len(churn_min),
        replace=False,
        random_state=1234,
    )

    return pd.concat([churn_maj_down, churn_min])