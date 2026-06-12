import pandas as pd

try:
    df = pd.read_csv("clean_df.csv")
    print(f"data loaded successfully.")
    print(f"shape: {df.shape}")
    print(df.info())
except FileNotFoundError:
    print(f"Data doesn't exist! Try running extract_data.py.")