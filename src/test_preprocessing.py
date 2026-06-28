from src.data.loader import load_data
from src.data.preprocessing import preprocess_data

df = load_data()

X, y = preprocess_data(df)

print(X.head())

print()

print(y.head())