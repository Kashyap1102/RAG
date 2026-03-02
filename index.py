import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from documentLoader import load_docs
from ollama_embeddings import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

documents = load_docs()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)

def _normalize_doc_metadata(doc: Document) -> Document:
    source = doc.metadata.get("source", "")
    doc.metadata["lecture"] = os.path.basename(source) if source else "unknown"
    raw_page = doc.metadata.get("page", 0)
    try:
        doc.metadata["page_number"] = int(raw_page) + 1
    except (TypeError, ValueError):
        doc.metadata["page_number"] = 1
    return doc


for i, chunk in enumerate(chunks):
    chunk.metadata["doc_id"] = i
    _normalize_doc_metadata(chunk)


# Split documents into smaller chunks
def split_docs_create_embeddings():
    embeddings = OllamaEmbeddings(os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"))

    try:
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="mlpdfs",
            host="localhost",
            port=6333,
            force_recreate=True,
            timeout=120,
        )
        print(f"Successfully created vectorstore with {len(chunks)} chunks")

    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


def vector_search(query, top_k=3) -> List[Tuple[Document, float]]:
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"))
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="mlpdfs",
        host="localhost",
        port=6333,
        embedding=embeddings,
        validate_collection_config=False,
    )

    search_results = vector_store.similarity_search_with_score(query, k=top_k)
    print(f"Found {len(search_results)} similar documents")
    return [(_normalize_doc_metadata(doc), float(score)) for doc, score in search_results]


def bm25_search(query, top_k=3):
    tokenized_docs = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(chunks[idx], float(score)) for idx, score in ranked[:top_k]]


def rerank(query, candidate_docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    if not pairs:
        return []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidate_docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_k]

def hybrid_retrieve(query, top_k=5):
    bm25_results = bm25_search(query, top_k)
    vector_results = vector_search(query, top_k)

    candidates = []
    seen = set()

    for doc, _ in bm25_results + vector_results:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.metadata.get("start_index", ""),
            hash(doc.page_content),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(doc)

    return candidates


if __name__ == "__main__":
    split_docs_create_embeddings()
