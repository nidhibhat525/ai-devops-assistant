
import chromadb
import requests
import uuid
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()

OLLAMA_EMBED_URL = os.getenv(
    "OLLAMA_EMBED_URL",
    "http://localhost:11434/api/embed"
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL",
    "gemini-embedding-001"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Separate collections for different embedding models
if AI_PROVIDER == "gemini":
    collection_name = "devops_documents_gemini_001"
else:
    collection_name = "devops_documents"

collection = client.get_or_create_collection(
    name=collection_name
)


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):

    if AI_PROVIDER == "gemini":

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing")

        embeddings = []

        for text in texts:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent",
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "model": f"models/{GEMINI_EMBED_MODEL}",
                    "content": {
                        "parts": [{"text": text}]
                    },
                    "taskType": task_type
                },
                timeout=180
            )

            response.raise_for_status()

            embeddings.append(
                response.json()["embedding"]["values"]
            )

        return embeddings

    elif AI_PROVIDER == "ollama":

        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "input": texts
            },
            timeout=180
        )

        response.raise_for_status()

        return response.json()["embeddings"]

    else:
        raise ValueError(
            f"Unsupported AI provider: {AI_PROVIDER}"
        )


def store_document(text, filename):
    chunks = split_text(text)

    embeddings = get_embeddings(
        chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )

    document_id = str(uuid.uuid4())
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                "filename": filename,
                "document_id": document_id
            }
            for _ in chunks
        ]
    )

    return {
        "document_id": document_id,
        "chunks_stored": len(chunks)
    }


def retrieve_context(question, document_id, n_results=3):

    results = collection.get(
        where={"document_id": document_id},
        include=["documents"]
    )

    document_count = len(results["ids"])

    if document_count == 0:
        return ""

    question_embedding = get_embeddings(
        [question],
        task_type="RETRIEVAL_QUERY"
    )[0]

    search_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, document_count),
        where={"document_id": document_id}
    )

    return "\n\n".join(
        search_results["documents"][0]
    )
