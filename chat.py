import os
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

embeddings = CohereEmbeddings(model="embed-english-v3.0")

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="mlpdfs",
    host="localhost",
    port=6333,
    embedding=embeddings)

user_query = input("Enter your query: ")

search_results = vector_store.similarity_search(user_query)

context = "\n".join([f"Page {result.metadata['page']}: {result.page_content} : {result.metadata['source']}" 
                     for result in search_results])

SYSTEM_PROMPT = """You are a helpful assistant that provides information based on the
 provided documents along with the page contents and page numbers.
 
You should only answer questions based on the provided documents. 
Always cite the source documents and page numbers in your response.

If the answer is not found in the documents, say you don't know.

Context:{{context}}"""

# llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
#     {"role": "user", "content": user_query}
# ]
# response = llm.invoke(messages)

llm = ChatCohere(model="command-r-plus", temperature=0.7)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
    {"role": "user", "content": user_query}
]

response = llm.invoke(messages)
print(f"Response: {response.content}")