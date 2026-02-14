import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load and chunk documents
def build_rag_collection(doc_path="docs/bfsi_policies.txt"):
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="bfsi_rag")

    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk.page_content).tolist()
        collection.add(
            documents=[chunk.page_content],
            ids=[str(i)],
            embeddings=[emb]
        )
    return collection, model

def rag_retrieve(query, collection, model, k=2):
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return results["documents"][0]
