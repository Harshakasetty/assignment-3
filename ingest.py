from datetime import datetime
from pymongo import MongoClient, ASCENDING
from config import MONGODB_URI, DB_NAME, VECTOR_COLLECTION, TTL_SECONDS
from sentence_transformers import SentenceTransformer
import uuid
from PyPDF2 import PdfReader

# CPU-friendly embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[VECTOR_COLLECTION]

# Ensure TTL index
collection.create_index([("createdAt", ASCENDING)], expireAfterSeconds=TTL_SECONDS)

def read_pdf(file_path):
    text = ""
    pdf = PdfReader(file_path)
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def ingest_file(file_path):
    text = read_pdf(file_path)
    chunks = chunk_text(text)
    for chunk in chunks:
        vector = embed_model.encode(chunk)
        collection.insert_one({
            "_id": str(uuid.uuid4()),
            "text": chunk,
            "vector": vector.tolist(),
            "createdAt": datetime.utcnow()  # <-- Use UTC time instead of server_info()
        })
    return len(chunks)
