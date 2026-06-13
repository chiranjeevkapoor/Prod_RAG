from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-2-preview"
    )


documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.',
        metadata={'type':'product'}
    ),
    Document(
        page_content='For network connectivity issue, first check the ethernet cables and router status lights.',
        metadata={'type':'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.',
        metadata={'type':'error'}
    ),
    Document(
        page_content='The authentication process requires valid credentials, use OAuth2 for secure API access.',
        metadata={'type':'auth'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.',
        metadata={'type':'config'}
    ),
    Document(
        page_content='WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.',
        metadata={'type':'compliance'}
    )

]


print(f"Loaded documents: {len(documents)}")


vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name='hybrid_test'
)


vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)

print("Vector retriever ready")


bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

print('BM25 retriver ready')



def hybrid_retriever(query, retrievers, weights, k=3, rrf_k = 6):
    """Combine multiple retrievers using weighted Reciprocal Ranked Fusion."""
    doc_scores = {}

    for retriver, weight in zip(retrievers, weights):
        results = retriver.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight*(1.0/(rank+rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)
            
    sorted_docs = sorted(doc_scores.values(), key=lambda x:x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]


print("Hybrid retriever ready.")


def test_query(query, name, retriever):
    """Test a query and show results."""
    if name=="HYBRID":
        results = retriever(query, retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])
    else:
        results = retriever.invoke(query)

    print(f"\\n {name} - Query: \"{query}\"")
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f"  {i+1}. {preview}")

    return results


test_queries = [
    "SKU-7742X specifications",
    "E_CONN_REFUSED error",
    "How do i authenticate?",
    "WCAG compliance",
    "router configuration"

]


for query in test_queries:
    print('=' * 60)

    vector_results = test_query(query, 'VECTOR', vector_retriever)

    bm25_results = test_query(query, 'BM25', bm25_retriever)

    hybrid_results = test_query(query, 'HYBRID', hybrid_retriever)

