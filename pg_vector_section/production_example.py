import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from typing import Optional
load_dotenv()


# Configuration
@dataclass
class Config:
# Database - use pooler URL in production
    database_url: str = os.getenv(
    "SUPABASE_DATABASE_URL" , 
    os. getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgress"),
    )
    collection_name: str = "production_documents"
# Model settings
    embedding_model: str = "gemini-embedding-2-preview"
    chat_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
# Search settings
    default_k: int = 5
    min_similarity: float = 0.5
class RAGService:
    """Production-ready RAG service with pgvector"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chain = None

    @property
    def vectorstore (self) -> PGVector:
        """Lazy initialization of vectorstore""" 
        if self._vectorstore is None:
            embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            self._vectorstore = PGVector(
            embeddings=embeddings, 
            collection_name=self.config.collection_name, 
            connection=self.config.database_url, 
            use_jsonb=True,
            )
        return self._vectorstore
    

    @property
    def chain(self):
        """Lazy initialization of RAG chain""" 
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain
    def _create_chain(self):
        """Create the RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k}
        )
        llm = ChatGoogleGenerativeAI(model = self.config.chat_model, temperature=self.config.temperature)
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Answer the question based on provided context.

Context:
{context}
Question:{question}

Answer concisely and accurately, If the context doesn't contain relevant information say "I don't have enough information to answer that question."
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        return (
            {"context":retriever | format_docs, "question":RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vectorstore"""
        return self.vectorstore.add_documents (documents)
    
    
    def search(self, query: str, k: Optional[int] = None, filter_dict: Optional[dict]=None) -> list[tuple[Document, float]]:
        """Search with optional filtering"""
        search_kwargs = {"k": k or self.config.default_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)
    
    def ask(self, question:str)->str:
        """Ask a question using RAG"""
        return self.chain.invoke(question)
    
    def ask_with_sources(self, question:str)->dict:
        """Ask a question and return sources."""
        docs_with_scores = self.search(question)

        answer = self.ask(question)

        return {
            "answer":answer,
            "sources":[
                {
                    "content":doc.page_content,
                    "metadata":doc.metadata,
                    "similarity":score
                }
                for doc, score in docs_with_scores
            ]
             }
        



def main():

    print("PRODUCTION RAG SERVICE DEMO")
    print("="*60)

    print(f"\n INITIALIZING RAG SERVICE...")
    service = RAGService()
    print(f"Service ready")

    print("\n ADDING SAMPLE DOCUMENTS...")

    sample_docs = [
        Document(
            page_content="pgvector is an open-source extension for PostgreSQL that enables the storage and querying of vector embeddings. Crucial for AI applications, it allows you to perform fast, exact, and approximate nearest neighbor (ANN) searches directly within your database, seamlessly combining relational data with machine learning capabilities like semantic search.",
            metadata={"source":"docs","topic":"pgvector"}
        ),
        Document(
            page_content="The Hierarchical Navigable Small World (HNSW) index is a state-of-the-art graph-based algorithm for fast approximate nearest neighbor search in high-dimensional vector spaces. It structures data into a multi-layered graph, where top layers enable fast, long-range skipping and bottom layers provide precise local routing. This architecture ensures logarithmic search time scalability.",
            metadata={"source":"docs","topic":"indexing"}
        ),
        Document(
            page_content="For production RAG (Retrieval-Augmented Generation), prioritize data quality via clean parsing and optimal chunk sizing. Implement hybrid search (combining keyword and vector) alongside a reranker to maximize retrieval precision. Guardrails are non-negotiable—continually evaluate your system's outputs for hallucination, and monitor latency to ensure a reliable user experience.",
            metadata={"source":"best-practices","topic":"production"}
        )
    ]


    ids = service.add_documents(sample_docs)
    print(f"Added {len(ids)} documents.")

    print(f"\n Testing search...")

    results = service.search("How do i make pgvector faster?", k=2)

    for doc, score in results:
        print(f"   Score:{score:.4f}-{doc.page_content[:50]}...")

    
    print("\n TESTING RAG...")
    question = "What is RAG and how do i use it?"
    response = service.ask_with_sources(question)
    print(f"\n Question:{question}")
    print(f"\n Answer:{response['answer']}")
    print(f"Sources used : {len(response['sources'])}")

    print("\n"+"="*60)

    print("DONE")

if __name__ == "__main__":
    main()

