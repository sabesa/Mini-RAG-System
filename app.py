import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 300
TOP_K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


st.title("Movie Plot Q&A (RAG)")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("wiki_movie_plots_deduped.csv")
        df = df[["Title", "Plot"]].sample(300, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def create_embeddings(documents):
    try:
        embedder = SentenceTransformer(EMBED_MODEL)
        embeddings = embedder.encode(documents, show_progress_bar=False)
        return embedder, embeddings
    except Exception as e:
        st.error(f"Failed to create embeddings: {e}")
        return None, None

@st.cache_data
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

#Process Data
df = load_data()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)

documents, metadata = [], []
for _, row in df.iterrows():
    chunks = text_splitter.split_text(row["Plot"])
    for chunk in chunks:
        documents.append(chunk)
        metadata.append({"title": row["Title"]})

embedder, embeddings = create_embeddings(documents)
index = build_faiss_index(embeddings)

#Initialize LLM
llm = ChatGroq(
    temperature=0.2,
    max_tokens=500,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

#Retrieval Function
def retrieve(query, top_k=TOP_K):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [(documents[i], metadata[i]) for i in I[0]]


def generate_answer(query):
    retrieved = retrieve(query)
    context_texts = [c[0] for c in retrieved]
    titles = list({m["title"] for _, m in retrieved})

    system_prompt = """You are a helpful assistant answering questions about movie plots.
Use the context below to answer the question. 
If you don't know, say so. Don't make up answers."""
    
    user_prompt = f"Question: {query}\n\nContexts:\n" + "\n".join(context_texts)
    
    response = llm.invoke(system_prompt + "\n\n" + user_prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    
    reasoning = (
        f"The query was '{query}'. Found {len(context_texts)} chunks "
        f"from {', '.join(titles)} and used them to form the answer."
    )
    return answer, context_texts, reasoning


query = st.text_input("Ask a movie question:")
if st.button("Get Answer") and query:
    with st.spinner("Generating answer..."):
        result = generate_answer(query)
        st.json(result)
