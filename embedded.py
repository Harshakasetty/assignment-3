# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_text_list(texts):
    model = get_embedding_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # ensure python list floats (json-serializable)
    return [emb.tolist() for emb in embs]

def embed_text(text):
    model = get_embedding_model()
    emb = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
    return emb.tolist()
