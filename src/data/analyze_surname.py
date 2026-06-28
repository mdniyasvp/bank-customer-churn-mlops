from src.data.loader import load_data

df = load_data()

print("Unique surnames:", df["Surname"].nunique())

print()

print(df["Surname"].value_counts().head(20))