import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from configs.config import SPLIT_DIR, ENCODER_DIR, PROCESSED_DATA_DIR
from src.extract_data import extract_data

def process_data():
    # If data is not processed yet
    if not PROCESSED_DATA_DIR.exists():
        extract_data()

    # load csv
    df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

    # Encodeing the label strings to numeric values

    label_encoder = LabelEncoder()
    label_encoder.fit(df['category']) 
    class_names = label_encoder.classes_.tolist()  
    df['encoded_label'] = label_encoder.transform(df['category'])


    # Creating test set
    X, X_test, y, y_test = train_test_split(df['text'], df['encoded_label'], test_size=0.2, stratify=df['category'], random_state=9)

    # Creating train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Save splits and label_encoder

    # Create directory if it doesn't exist
    splits_dir = SPLIT_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = ENCODER_DIR
    encoder_dir.mkdir(exist_ok=True)


    # Save features
    X_train.to_csv(splits_dir / "X_train.csv", index=False)
    X_val.to_csv(splits_dir / "X_val.csv", index=False)
    X_test.to_csv(splits_dir / "X_test.csv", index=False)

    # Save targets
    y_train.to_csv(splits_dir / "y_train.csv", index=False)
    y_val.to_csv(splits_dir / "y_val.csv", index=False)
    y_test.to_csv(splits_dir / "y_test.csv", index=False)


    # Save encoder
    joblib.dump(label_encoder, encoder_dir / "label_encoder.pkl")

    # save class names for debugging
    class_names = label_encoder.classes_.tolist()
    joblib.dump(class_names, encoder_dir / "class_names.pkl")

    print("Saved all dataset splits to data/splits/ and label encoder and class names in artifacts/encoder")

if __name__ == "__main__":
    process_data()