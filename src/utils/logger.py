# src/utils/logger.py
import logging

def setup_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("train.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)
