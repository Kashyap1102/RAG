import os
import ollama
from qdrant_client import QdrantClient
import requests
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from ollama_embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"))
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="mlpdfs",
    host="localhost",
    port=6333,
    embedding=embeddings,
    validate_collection_config=False  # Skip dimension validation
)


user_query = input("Enter your query: ")

embedded_query = embeddings.embed_query(user_query)  # Ensure embedding works before searching

search_results = vector_store.similarity_search_by_vector(embedded_query, k=3)
print(f"✓ Found {len(search_results)} similar documents")

SYSTEM_PROMPT = """You are a helpful assistant that provides information based on the
provided documents along with the page contents and page numbers.

You should only answer questions based on the provided documents. 
Always cite the source documents and page numbers in your response.

Context:{context}

"""

context = "\n".join([f"Page {result.metadata['page']}: {result.page_content} : {result.metadata['source']}" 
                    for result in search_results])


messages = [
    {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
    {"role": "user", "content": user_query}
]
# List all models installed locally
models = ollama.list()
for model in models['models']:
    print(model['model'])

# Use Ollama to generate response
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
response = ollama.chat(
    model=ollama_model,
    messages=messages,
    stream=False
)

print(f"Response: {response['message']['content']}")