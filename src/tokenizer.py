"""Download pretrained tokenizer from Hugging face transformers"""

from transformers import AutoTokenizer
from configs.config import MODEL_NAME, TOKENIZER_DIR

def download_pretrained_tokenizer():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_DIR)

if __name__ == "__main__":
    download_pretrained_tokenizer()