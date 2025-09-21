import numpy as np
from pymongo import MongoClient
from config import MONGODB_URI, DB_NAME, VECTOR_COLLECTION
from sentence_transformers import SentenceTransformer

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[VECTOR_COLLECTION]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=5):
    query_vector = embed_model.encode(query)
    results = []

    for doc in collection.find():
        # Skip if 'vector' field is missing
        if 'vector' not in doc:
            continue
        sim = cosine_similarity(query_vector, np.array(doc['vector']))
        results.append((sim, doc['text']))

    results.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in results[:top_k]]
