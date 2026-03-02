import os
import ollama
from dotenv import load_dotenv
from index import hybrid_retrieve, rerank

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))



def search(query):
    retrieve_k = int(os.getenv("RETRIEVE_TOP_K", "8"))
    rerank_k = int(os.getenv("RERANK_TOP_K", "5"))
    candidates = hybrid_retrieve(query, top_k=retrieve_k)
    final_results = rerank(query, candidates, top_k=rerank_k)

    citation_rows = []
    for idx, (doc, _) in enumerate(final_results, start=1):
        page_number = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        lecture = doc.metadata.get("lecture", doc.metadata.get("source", "unknown"))
        citation_rows.append((f"C{idx}", page_number, lecture, doc.page_content))

    context = "\n\n".join(
        [f"[{cid}] Lecture={lecture} Page={page}\n{content}" for cid, page, lecture, content in citation_rows]
    )


    SYSTEM_PROMPT = """You are a helpful assistant that provides information based on the
                        provided documents along with the page contents and page numbers.

                        You should only answer questions based on the provided documents. 
                        Use citation tags like [C1], [C2] in your answer.
                        Include all relevant citations from the provided context.

                        Context:{context}

                        If you can't find the answer in the provided documents, say you don't know. 
                        Do not make up an answer.

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
    if citation_rows:
        print("\nCitations used:")
        for cid, page, lecture, _ in citation_rows:
            print(f"[{cid}] {lecture} - Page {page}")
    else:
        print("\nCitations used: none")

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
