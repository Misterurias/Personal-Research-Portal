import os
import json
import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm
import pandas as pd

RAW_DIR = "../../data/raw"
OUT_PATH = "../../data/processed/chunks.json"
MANIFEST_PATH = "../../data/data_manifest.csv"

CHUNK_SIZE = 700
OVERLAP = 150

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, source_id):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    chunk_idx = 1

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunk_id = f"{source_id}_chunk_{str(chunk_idx).zfill(3)}"

        chunks.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "text": chunk_text
        })

        start += CHUNK_SIZE - OVERLAP
        chunk_idx += 1

    return chunks

def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    all_chunks = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        source_id = row["source_id"]
        pdf_path = os.path.join(RAW_DIR, os.path.basename(row["raw_path"]))

        print(f"Processing {source_id}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text, source_id)
        all_chunks.extend(chunks)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved {len(all_chunks)} chunks.")

if __name__ == "__main__":
    main()