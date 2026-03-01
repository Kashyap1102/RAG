import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from ollama_embeddings import OllamaEmbeddings


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Define the path to the MLPDFs folder
mlpdfs_folder = "MLPDFs"

# Read all file paths in the MLPDFs folder
file_paths = []
if os.path.exists(mlpdfs_folder):
    for filename in os.listdir(mlpdfs_folder):
        file_path = os.path.join(mlpdfs_folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.pdf'):
            file_paths.append(file_path)
else:
    print(f"Warning: {mlpdfs_folder} folder not found.")
            
# Load documents from all PDF files
documents = []
for file_path in file_paths:
    try:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if not documents:
    print("Error: No documents loaded. Please ensure PDF files exist in the MLPDFs folder.")
    exit(1)

# Split documents into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Create OllamaEmbeddings instance
embeddings = OllamaEmbeddings(os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"))

try:
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="mlpdfs",
        host="localhost",
        port=6333,
        force_recreate=True)
    print(f"Successfully created vectorstore with {len(chunks)} chunks")
except Exception as e:
    print(f"Error creating vectorstore: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
    