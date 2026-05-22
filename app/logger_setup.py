"""
Logging configuration for the Churn Prediction API.

Centralizes logging so every module shares one consistent format.
Author: Abdallah Mohamed
"""

import logging


def setup_logging(name: str = "churn-api", level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return a named logger.

    Format: timestamp | LEVEL | logger-name | message
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(name)
    logger.info("Logger initialized for service '%s'", name)
    return logger
