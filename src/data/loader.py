import pandas as pd
from src.config.settings import RAW_DATA_PATH


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw churn dataset.

    Args:
        path (str): Path to raw data file

    Returns:
        pd.DataFrame: Raw dataset
    """
    df = pd.read_csv(path)
    return df