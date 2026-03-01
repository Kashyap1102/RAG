import os
import ollama
from dotenv import load_dotenv
from index import bm25_search, hybrid_retrieve, rerank, vector_search

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def hybrid_retrieve(query, top_k=5):
    bm25_results = bm25_search(query, top_k)
    vector_results = vector_search(query, top_k)

    candidate_ids = set()

    for idx, _ in bm25_results:
        candidate_ids.add(idx)

    for idx, _ in vector_results:
        candidate_ids.add(idx)

    return list(candidate_ids)

def search(query):
    candidates = hybrid_retrieve(query, top_k=5)
    final_results = rerank(query, candidates, top_k=3)



    context = "\n".join(
        [f"Page {result.metadata['page']}: {result.page_content} : {result.metadata['source']}" 
         for result in final_results])


    SYSTEM_PROMPT = """You are a helpful assistant that provides information based on the
                        provided documents along with the page contents and page numbers.

                        You should only answer questions based on the provided documents. 
                        Always cite the source documents and page numbers in your response.

                        Context:{context}

            """


    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query}
    ]
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    response = ollama.chat(
        model=ollama_model,
        messages=messages,
        stream=False
    )

    print(f"Response: {response['message']['content']}")

search(input("Enter your query: "))











# def rrf_fusion(bm25_results, vector_results, k=60):
#     scores = {}

#     for rank, (doc_id, _) in enumerate(bm25_results):
#         scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

#     for rank, (doc_id, _) in enumerate(vector_results):
#         scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

#     return sorted(scores.items(), key=lambda x: x[1], reverse=True)



# def hybrid_retrieve(query, top_k=5):
#     bm25_results = bm25_search(query, top_k)
#     vector_results = vector_search(query, top_k)

#     candidate_ids = set()

#     for idx, _ in bm25_results:
#         candidate_ids.add(idx)

#     for idx, _ in vector_results:
#         candidate_ids.add(idx)

#     return list(candidate_ids)

# # List all models installed locally
# models = ollama.list()
# for model in models['models']:
#     print(model['model'])

# Use Ollama to generate response
