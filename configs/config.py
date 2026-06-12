from pathlib import Path

# model parameters
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 2e-5
NUM_WORKERS = 0 # not used till now
ROOT_DIR = Path(__file__).parent.parent  # goes up from configs/ to root
RAW_DATA_DIR = ROOT_DIR / "data/raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data/processed"
SPLIT_DIR = ROOT_DIR / "data/splits"
ENCODER_DIR = ROOT_DIR / "artifacts/encoder"
TOKENIZER_DIR = ROOT_DIR / "artifacts/tokenizer"
HF_MODEL_DIR = ROOT_DIR / "artifacts/models/hf_pretrained"
MODEL_DIR_V1 = ROOT_DIR / "artifacts/models/model_v1"