"""
RAG Assessment Skeleton
----------------------
This is a technical interview assessment for a Solutions Engineer role at Glean.

Your task: Complete the functions and logic below to build a simple Retrieval-Augmented Generation (RAG) demo.

Requirements:
- Read and chunk markdown files from the 'faqs/' directory.
- Embed the chunks using OpenAI's embedding API.
- Retrieve the top-k most relevant chunks for a user query.
- Generate an answer using OpenAI's chat completion API, citing at least two source files.
- Output the answer and sources as a JSON object to stdout.

"""

import os
import json
from openai import OpenAI
import numpy as np
from tqdm import tqdm

# --- Config ---
FAQ_DIR = "faqs"
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 200  # characters per chunk
TOP_K = 4

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, size=CHUNK_SIZE):
    """
    Split the input text into chunks of approximately 'size' characters.
    Return a list of text chunks.
    """
    return [text[i:i+size] for i in range(0, len(text), size)]

def cosine_sim(a, b):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_and_chunk_faqs(faq_dir):
    """
    Read all .md files in faq_dir, chunk their contents, and return:
    - chunks: List of text chunks
    - sources: List of corresponding source filenames
    """
    chunks = []
    sources = []
    for fname in os.listdir(faq_dir):
        if fname.endswith(".md"):
            with open(os.path.join(faq_dir, fname)) as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append(chunk)
                sources.append(fname)
    return chunks, sources

def embed_texts(texts):
    """
    Given a list of texts, return a list of their embeddings as numpy arrays.
    Use OpenAI's embedding API.
    """
    embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        # TODO1: Call the OpenAI embedding API and append the result as a numpy array

        # Call OpenAI to get the embedding
        response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL
        )

        # Extract the vector from the returned object and turn add to embeddings as numpy array
        embedding = response.data[0].embedding
        embeddings.append(np.array(embedding))

    return embeddings

def main():
    # --- 1. Load & Chunk ---
    chunks, sources = load_and_chunk_faqs(FAQ_DIR)

    # --- 2. Embed Chunks ---
    # TODO2: Embed the chunks using embed_texts
    chunk_embeddings = embed_texts(chunks)
    #test faq embeddings were created correctly
    print(f"DEBUG: Created {len(chunk_embeddings)} embeddings. First one has length: {len(chunk_embeddings[0])}")

    # --- 3. Query Loop ---
    query = input("Enter your question: ")
    # TODO3: Embed the query using client.embeddings.create
    #send user's query to OpenAI to retrieve its vector
    response = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    )
    
    #extract vector from response
    query_emb = np.array(response.data[0].embedding)
    #test uery embedding was created correctly
    print(f"DEBUG: Query embedding created. Length: {len(query_emb)}")

    # --- 4. Retrieve Top-k ---
    # TODO4: Compute similarities and get top-k indices
    #create array to hold similarity scores
    sims = []
    #loop through each chunk embedding to compare it to the query embedding
    for chunk_embedding in chunk_embeddings:
        sim = cosine_sim(query_emb, chunk_embedding)
        sims.append(sim)

    #convert sims list to numpy array for argsort function below
    sims = np.array(sims)

    top_indices = np.argsort(sims)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    top_files = [sources[i] for i in top_indices]

    #check cosine sim process to confirm chunks matching query are found
    print("DEBUG: Top found files:", top_files)

    # --- 5. Generate Answer ---
    context = "\n\n".join([f"From {sources[i]}:\n{chunks[i]}" for i in top_indices])
    prompt = (
        f"Answer the following question using the provided context. "
        f"Cite at least two of the file names in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources):"
    )
    # TODO5: Generate answer using client.chat.completions.create
    answer = ""  # Replace with actual answer

    # --- 6. Output JSON ---
    output = {
        "answer": answer,
        "sources": list(sorted(set(top_files)))[:2]  # at least two
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main() 