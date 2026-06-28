import pandas as pd
from src.config.config import TRAIN_DATA

def load_data():
    """
    Load the raw training dataset.
    """

    df = pd.read_csv(TRAIN_DATA)

    return df