import pandas as pd

df = pd.read_csv("data/raw/BATADAL_test_dataset.csv")
raw_dates = df["DATETIME"].astype(str).str.strip()

print("Raw sample dates:")
print(raw_dates.head(10))

# Try explicit format
dt_explicit = pd.to_datetime(raw_dates, format="%d/%m/%y %H")
print("\nExplicit format '%d/%m/%y %H' Monotonic:", dt_explicit.is_monotonic_increasing)
print("Explicit min date:", dt_explicit.min())
print("Explicit max date:", dt_explicit.max())

print("\nSample parsed explicit:")
print(dt_explicit.head(5))
print(dt_explicit.tail(5))
