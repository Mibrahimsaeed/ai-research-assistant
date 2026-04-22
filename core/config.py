import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")

FAISS_METADATA_FILE = os.path.join(DATA_DIR, "faiss_metadata.pkl")