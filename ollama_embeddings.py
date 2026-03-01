import os
from typing import List
import ollama
from langchain_core.embeddings import Embeddings


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str | None = None, host: str | None = None):
        super().__init__()
        self.model = model or os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Initialize ollama client with the host
        self.client = ollama.Client(host=self.host)

    def _call(self, inputs: List[str]) -> List[List[float]]:
        """Call Ollama embeddings using the Python client."""
        embeddings = []
        for text in inputs:
            try:
                response = self.client.embeddings(model=self.model, prompt=text)
                # Handle EmbeddingsResponse object
                if hasattr(response, 'embedding'):
                    embeddings.append(response.embedding)
                elif isinstance(response, dict) and "embedding" in response:
                    embeddings.append(response["embedding"])
                elif isinstance(response, list):
                    embeddings.append(response)
                else:
                    print(f"DEBUG: Response type: {type(response)}, value: {response}")
                    embeddings.append(response)
            except Exception as e:
                raise ValueError(f"Error getting embedding for text: {e}")
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        res = self._call([text])
        return res[0] if res else []
