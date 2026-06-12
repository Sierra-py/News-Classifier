"""Download pretrained tokenizer from Hugging face transformers"""

from transformers import AutoTokenizer
from configs.config import MODEL_NAME

bert = MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(bert)
tokenizer.save_pretrained("../artifacts/tokenizer")