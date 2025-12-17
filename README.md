A lightweight Retrieval-Augmented Generation (RAG) prototype designed to answer employee questions based on internal FAQ documentation.

# Overview 
This tool ingests markdown FAQ files, indexes them using OpenAI embeddings, and uses a semantic search loop to retrieve relevant context for the LLM. It is optimized for accuracy with the provided small FAQ content library.

# Key Features
1. Semantic Search: Uses Cosine Similarity to match user intent
2. Optimized Indexing: Custom chunking strategy (800 chars) to preserve entire semantic unite (1 file per embedding)
3. Low TOP_K Parameter: With a tiny dataset of 3 files (1 per embedding), we compare the top 2 chunks to user's query, and let the LLM reason what information is most important to answer user query.
4. Fact-Grounded: Low-temperature (0) to minimize hallucination.
5. Interactive Mode: Continuous query loop for seamless user testing, and a one-time FAQ indexing process.

# Setup & Usage

1. Set OpenAI API Key:

export OPENAI_API_KEY="your-key-here"

2. Run the script

python rag_assessment.py

3. Ask questions about the FAQ content.

# Configuration 
1. Model: gpt-3.5-turbo
2. Chunk Size: 800 characters (chosen to capture entire file per embedding)
3. TOP_K: 2 chunks (chosen because broad answers may be spread across multiple files)
4. Temperature: 0 to prevent hallucination

