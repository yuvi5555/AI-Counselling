import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

INDEX_FILE = "rag_csv_index.index"
CHUNK_FILE = "rag_csv_chunks.pkl"
METADATA_FILE = "rag_csv_metadata.pkl"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 👇 IMPORTANT: Change base_url
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading chunks...")
with open(CHUNK_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# RETRIEVE FUNCTION
# ==========================================================

def retrieve(query, top_k=5):

    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    return [chunks[i] for i in indices[0]]

# ==========================================================
# GENERATE ANSWER (OPENROUTER)
# ==========================================================

def generate_answer(query):

    context_chunks = retrieve(query)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI-based Engineering Counselling Assistant.

Answer ONLY using the context below.
If answer not present say:
"Information not available in dataset."

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",  # OpenRouter model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# ==========================================================
# MAIN LOOP
# ==========================================================

if __name__ == "__main__":

    print("\n===== OpenRouter RAG Counselling Assistant =====\n")

    while True:

        query = input("Ask question (type exit to quit): ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "="*60)
