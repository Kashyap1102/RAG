# Define the path to the MLPDFs folder
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def load_docs(folder = "MLPDFs"):
    

# Read all file paths in the MLPDFs folder
    file_paths = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.pdf'):
                file_paths.append(file_path)
    else:
        print(f"Warning: {folder} folder not found.")

    file_paths.sort()
                
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

    return documents
