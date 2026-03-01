#!/usr/bin/env python
"""Simple RAG chat test."""

import os
from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient
from ollama_embeddings import OllamaEmbeddings

load_dotenv()

# Create raw Qdrant client for queries
client = QdrantClient("localhost", port=6333)

# Use Ollama embeddings for encoding queries
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Test query
user_query = "What is the main topic?"
print(f"Query: {user_query}")

# Search using embeddings directly through vector similarity
from qdrant_client.models import PointStruct

query_embedding = embeddings.embed_query(user_query)
search_results = client.search(
    collection_name="mlpdfs",
    query_vector=query_embedding,
    limit=3
)
print(f"✓ Found {len(search_results)} similar documents")
