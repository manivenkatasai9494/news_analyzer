import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

# -------- ENV --------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

# -------- LLM --------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# -------- EMBEDDINGS --------
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- PINECONE --------
pc = Pinecone(api_key=PINECONE_API_KEY)

# 🔥 FIXED SCRAPER (Moneycontrol-safe)
def fetch_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9"
    }

    r = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")

    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return text


@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/process-urls", methods=["POST"])
def process_urls():
    urls = request.json["urls"]

    urls = list(set(u for u in urls if u.strip()))

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # ✅ SAFE DELETE (FIX)
    index = pc.Index(INDEX_NAME)
    try:
        index.delete(delete_all=True)
    except Exception:
        print("No previous data found. Fresh start.")

    documents = []
    for url in urls:
        text = fetch_text(url)

        if len(text) < 500:
            print("Skipped (no content):", url)
            continue

        documents.append(
            Document(page_content=text, metadata={"source": url})
        )

    if not documents:
        return jsonify({"error": "No readable content found from URLs"}), 400

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    documents = splitter.split_documents(documents)

    PineconeVectorStore.from_documents(
        documents,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return jsonify({"status": "Old data cleared (if any). New URLs indexed."})


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return jsonify({
            "answer": "No relevant information found in the indexed articles.",
            "sources": []
        })

    context = "\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata["source"] for d in docs})

    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say "Not mentioned in the article".

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke([HumanMessage(content=prompt)]).content

    return jsonify({
        "answer": answer,
        "sources": sources
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
