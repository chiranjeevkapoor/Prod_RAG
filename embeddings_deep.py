from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import tempfile
import os
import numpy as np

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-2-preview"
    )

def basic_embeddings():
    """Create basic Gemini Embeddings.""" 

    text = "What is machine learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Single embedding for: '{text}' : \n{len(single_embedding)}\n")
    print(f"First 5 values: {single_embedding[:5]}\n")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}\n")
    


def batch_embeddings():
    text = [
        "What is machine learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?"
    ]

    batch_embeddings = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - vector norm: {np.linalg.norm(emb):.4f}")



def similarity_search():
    docs = [
        "Python is a programming language.",
        "Javascript is used for web development.",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks.",
        "Cats are popular pets."
    ]

    query = "What programming languages exit?"

    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    

    similarities =[cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x:x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"{score:.4f}:{doc}")



if __name__=="__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()