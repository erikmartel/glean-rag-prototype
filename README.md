A lightweight Retrieval-Augmented Generation (RAG) prototype designed to answer employee questions based on internal FAQ documentation.

# Overview 
This tool ingests markdown FAQ files, indexes them using OpenAI embeddings, and uses a semantic search loop to retrieve relevant context for the LLM. It is optimized for accuracy in policy-heavy domains (e.g., Equity, PTO).

# Key Features
Semantic Search: Uses Cosine Similarity to match user intent
Optimized Indexing: Custom chunking strategy (300 chars) to preserve policy context specific to our tiny, low-noise database.
Low TOP_K Parameter: With a dataset of 4 items, we only compare the top 2 chunks to user's query, to ensure a focused context-window relative to database size.
Fact-Grounded: Low-temperature (0.1) to minimize hallucination.
Interactive Mode: Continuous query loop for seamless user testing, and a one-time FAQ indexing process.

# Setup & Usage

1. Set OpenAI API Key:

export OPENAI_API_KEY="your-key-here"

2. Run the script

python rag_assessment.py

# Configuration 
1. Model: gpt-3.5-turbo
2. Chunk Size: 800 characters (chosen to capture entire file per embedding)
3. TOP_K: 1 chunk (chosen to provide only the single relevant file to LLM before answer generation)
4. Temperature: 0-0.1 to prevent hallucination

