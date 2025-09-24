# Mini RAG System - Movie Plots

A minimal **Retrieval-Augmented Generation (RAG)** pipeline that answers questions about movie plots using a subset of Wikipedia movie plot data.  
This implementation is **CLI/notebook-friendly** and does not require a UI or API.

---

## Description

This project demonstrates a lightweight RAG system with the following components:

1. **Dataset Loading**  
   Loads a subset of movie plots (`wiki_movie_plots_deduped.csv`) for faster experimentation.

2. **Text Chunking**  
   Splits each movie plot into smaller chunks using `RecursiveCharacterTextSplitter` for better retrieval granularity.

3. **Embeddings**  
   Generates embeddings for each text chunk using `sentence-transformers/all-MiniLM-L6-v2`.

4. **FAISS Indexing**  
   Builds a FAISS vector index for similarity search and quick retrieval of relevant text chunks.

5. **Retrieval & Answering**  
   - Retrieves the top-k most relevant chunks for a user query.  
   - Passes the retrieved context to **Groq LLM (ChatGroq)** to generate an answer.  
   - Provides reasoning about which chunks were used to answer the question.


---

## Requirements

- Python 3.9+  
- Dependencies:

```bash
pip install pandas faiss-cpu sentence-transformers langchain langchain-groq python-dotenv
```bash

## How to Run

Clone or download the repository.

Ensure .env exists and contains your GROQ_API_KEY.

Place the dataset wiki_movie_plots_deduped.csv in the project folder.

Run the script from CLI

```bash
streamlit run app.py
```bash

