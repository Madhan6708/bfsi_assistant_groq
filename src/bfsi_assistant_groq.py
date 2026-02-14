import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from rag_layer import build_rag_collection, rag_retrieve

# Init Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load dataset
with open("data/v1_dataset_150.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# Build RAG collection
rag_collection, rag_model = build_rag_collection("bfsi_policies.txt")

questions = [item["input"] for item in data]
answers = [item["output"] for item in data]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB (Chroma)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="bfsi_dataset")

for i, q in enumerate(questions):
    emb = model.encode(q).tolist()
    collection.add(
        documents=[q],
        metadatas=[{"answer": answers[i]}],
        ids=[str(i)],
        embeddings=[emb]
    )

def search_dataset(query, threshold=0.7):
    query_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=1
    )

    if len(results["documents"][0]) == 0:
        return None, 0.0

    matched_answer = results["metadatas"][0][0]["answer"]
    distance = results["distances"][0][0]
    similarity = 1 / (1 + distance)

    if similarity >= threshold:
        return matched_answer, similarity
    else:
        return None, similarity

def ask_groq_with_context(query, context_chunks):
    context_text = "\n".join(context_chunks)
    prompt = f"""
You are a BFSI assistant. Answer using ONLY the context below.
If the answer is not present, say you cannot provide exact details.

Context:
{context_text}

Question:
{query}
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def ask_groq_llm(query):
    system_prompt = """You are a BFSI customer support assistant.
Follow compliance:
- Do not guess interest rates or EMI values.
- Do not invent bank policies.
- If exact data is unavailable, say you cannot provide exact figures.
- Maintain a professional tone.
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        q = input("Ask BFSI question: ")

        ans, sim = search_dataset(q)
        if ans:
            print(f"\n✅ Dataset Answer (similarity={sim:.2f}):\n{ans}\n")
        else:
            # Simple heuristic: use RAG for complex policy queries
            if any(x in q.lower() for x in ["penalty", "formula", "prepayment", "breakdown"]):
                print("\n📚 Using RAG (document retrieval)...\n")
                ctx = rag_retrieve(q, rag_collection, rag_model)
                print(ask_groq_with_context(q, ctx), "\n")
            else:
                print(f"\n⚠️ No dataset match (similarity={sim:.2f})")
                print("🌐 Using Groq LLM...\n")
                print(ask_groq_llm(q), "\n")
