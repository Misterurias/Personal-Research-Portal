import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "../../data/processed/chunks.json"
INDEX_PATH = "../../data/processed/faiss.index"
META_PATH = "../../data/processed/chunk_metadata.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "w") as f:
    json.dump(chunks, f)

print("Index built successfully.")