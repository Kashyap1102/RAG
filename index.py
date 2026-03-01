import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from documentLoader import load_docs
from ollama_embeddings import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

documents = load_docs()


# Split documents into smaller chunks
def split_docs_create_embeddings():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = i

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


def vector_search(query, top_k=3) -> List[Tuple[int, float]]:
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"))
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="mlpdfs",
        host="localhost",
        port=6333,
        embedding=embeddings,
        validate_collection_config=False,
    )

    embedded_query = embeddings.embed_query(query)
    search_results = vector_store.similarity_search_with_score_by_vector(embedded_query, top_k)
    print(f"Found {len(search_results)} similar documents")

    scored_results: List[Tuple[int, float]] = []
    for rank, (doc, score) in enumerate(search_results):
        raw_doc_id = doc.metadata.get("doc_id", rank)
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError):
            doc_id = rank
        scored_results.append((doc_id, float(score)))

    return scored_results


def bm25_search(query, top_k=3):
    tokenized_docs = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rerank(query, candidate_ids, top_k=3):
    pairs = [(query, documents[i].page_content) for i in candidate_ids]
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_k]



split_docs_create_embeddings()
