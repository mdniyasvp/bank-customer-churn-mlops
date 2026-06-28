from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
TRAIN_DATA = RAW_DATA_DIR / "train.csv"

# Target
TARGET_COLUMN = "Exited"

# Columns to Drop
DROP_COLUMNS = ["id","CustomerId","Surname"]

# Random State
RANDOM_STATE = 42

# Test Size
TEST_SIZE = 0.2

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"