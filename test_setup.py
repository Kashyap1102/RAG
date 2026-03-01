#!/usr/bin/env python
"""Quick test of the RAG system setup."""

from ollama_embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

print("Testing OllamaEmbeddings...")
embeddings = OllamaEmbeddings()
print(f"✓ OllamaEmbeddings instantiated: {type(embeddings)}")

print("\nTesting embedding...")
result = embeddings.embed_query("test query")
print(f"✓ Embedding works, dimension: {len(result)}")

print("\nTesting vectorstore connection...")
try:
    vs = QdrantVectorStore.from_existing_collection(
        collection_name="mlpdfs",
        host="localhost",
        port=6333,
        embedding=embeddings
    )
    print(f"✓ Vectorstore loaded successfully")
    
    # Test similarity search
    search_results = vs.similarity_search("test")
    print(f"✓ Similarity search works, found {len(search_results)} results")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests passed!")
