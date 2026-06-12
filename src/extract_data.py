import pandas as pd
from pathlib import Path
import sys
from kaggle.api.kaggle_api_extended import KaggleApi


path = Path(r".\data\raw\News_Category_Dataset_v3.json")
if not path.exists():
    print("Downloading Dataset")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "rmisra/news-category-dataset",
        path="./data/raw",
        unzip=True
    )
    print("Dataset Downloaded")
try:
    df = pd.read_json(path, lines=True)
except FileNotFoundError:
    sys.exit("JSON file doesn't exist!")

    
df['text'] = df['headline'] + ". " + df['short_description']
clean_df = df[['category', 'text']]

clean_df.to_csv("clean_df.csv", index=False)