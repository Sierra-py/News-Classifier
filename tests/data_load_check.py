import pandas as pd
from configs.config import PROCESSED_DATA_DIR
try:
    df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
    print(f"data loaded successfully.")
    print(f"shape: {df.shape}")
    print(df.info())
except FileNotFoundError:
    print(f"Data doesn't exist! Try running extract_data.py.")