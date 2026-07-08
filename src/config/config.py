from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
TRAIN_DATA = RAW_DATA_DIR / "train.csv"

TARGET_COLUMN = "Exited"

DROP_COLUMNS = [
    "id",
    "CustomerId",
    "Surname"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model Directory
MODEL_DIR = PROJECT_ROOT / "models"

# Current Model
MODEL_NAME = "xgboost"
