import pandas as pd
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def extract_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = Path(RAW_DATA_DIR / "News_Category_Dataset_v3.json")
    if not path.exists():
        print("Downloading Dataset")
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            "rmisra/news-category-dataset",
            path=RAW_DATA_DIR,
            unzip=True
        )
        print("Dataset Downloaded")
    try:
        df = pd.read_json(path, lines=True)
    except FileNotFoundError:
        sys.exit("JSON file doesn't exist!")

        
    df['text'] = df['headline'] + ". " + df['short_description']
    clean_df = df[['category', 'text']]

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_DIR / "clean_df.csv", index=False)

if __name__ == "__main__":
    extract_data()