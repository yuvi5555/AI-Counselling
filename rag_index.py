import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "CAP_MASTER_STRUCTURED.csv"
INDEX_FILE = "rag_csv_index.index"
CHUNK_FILE = "rag_csv_chunks.pkl"
METADATA_FILE = "rag_csv_metadata.pkl"

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading CSV dataset...")
df = pd.read_csv(CSV_FILE)

print("Original Shape:", df.shape)

# Remove rows without essential info
df = df.dropna(subset=["college_name", "branch_name", "final_rank", "final_percentile"])

print("After Cleaning:", df.shape)

# ==========================================================
# CREATE STRUCTURED TEXT CHUNKS
# ==========================================================

def create_structured_chunk(row):

    return (
        f"College: {row['college_name']}. "
        f"Branch: {row['branch_name']}. "
        f"Year: {int(row['year'])}. "
        f"Round: {row['round_number']}. "
        f"Category: {row['normalized_category']}. "
        f"Gender: {row['gender']}. "
        f"Quota: {row['quota_type']}. "
        f"Stage: {row['stage']}. "
        f"University Type: {row['university_type']}. "
        f"Cutoff Rank: {int(row['final_rank'])}. "
        f"Cutoff Percentile: {round(row['final_percentile'],2)}."
    )

print("Creating structured chunks...")
chunks = df.apply(create_structured_chunk, axis=1).tolist()

# Metadata for advanced filtering later
metadata = df[[
    "college_code",
    "college_name",
    "branch_code",
    "branch_name",
    "year",
    "round_number",
    "normalized_category"
]].to_dict(orient="records")

print("Total Chunks Created:", len(chunks))

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# GENERATE EMBEDDINGS
# ==========================================================

print("Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, INDEX_FILE)

# Save chunks
with open(CHUNK_FILE, "wb") as f:
    pickle.dump(chunks, f)

# Save metadata
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print("\n✅ CSV-Based RAG Index Created Successfully")
print("Index Size:", index.ntotal)
