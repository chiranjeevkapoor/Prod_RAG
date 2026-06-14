import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


SUPABASE_URL= os.getenv("SUPABASE_DATABASE_URL")

DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgress"
)


def connect_to_supabase():
    """Connect to Supabase PGvector"""

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True
    )

    return vector_store

def verify_connection(vectorstore):
    """Verify the connection works."""
    from langchain_core.documents import Document


    test_doc = Document(
        page_content="This is a test document to test supabase",
        metadata={"test":True}
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Added test document:{ids[0]}")

        results = vectorstore.similarity_search("test document")
        if results:
            print(f"Search works: {results[0].page_content}") 

        # vectorstore.delete(ids)
        # print("cleanup complete")

        return True
    except Exception as e:
        print(f"Error:{e}")
        return False
    

def main():
    print("=" * 60)
    print("SUPABASE pgvector connection test")
    print("=" * 60)

    if SUPABASE_URL:
        print("\nConnecting to SUPABASE...")
        try:
            # Safely extract host details without throwing IndexError
            host_part = SUPABASE_URL.split('@')[1].split('/')[0]
            print(f"HOST: {host_part}")
        except IndexError:
            print("HOST: [Hidden or complex connection string format]")
    else:
        print("⚠️ SUPABASE_DATABASE_URL string is not setup in .env. Using fallback.")

    vs = connect_to_supabase()
    
    print("\nRunning verification tests...")
    success = verify_connection(vs)
    
    print("\n" + "=" * 60)
    print(f"Connection Verification Result: {'PASSED' if success else 'FAILED'}")
    print("=" * 60)

# CRITICAL FIX: The entry point block required to actually run the script
if __name__ == "__main__":
    main()