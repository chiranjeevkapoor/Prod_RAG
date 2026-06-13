import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-2-preview"
    )


document = """
#Authentication

1. Authentication & Authorization (OAuth 2.1)
OAuth 2.1 modernizes API security by eliminating legacy vulnerabilities. It removes the implicit and password grant types entirely. For user-facing applications, the Authorization Code Flow with PKCE is mandatory, requiring a cryptographic challenge to prevent interception. For service-to-service communication, the Client Credentials Flow is utilized. Access tokens must reside exclusively in the Authorization: Bearer header, never in query strings. Additionally, implement token rotation to invalidate compromised refresh tokens instantly.

2. Secure Webhooks
Webhooks reverse the API dynamic, pushing real-time data to clients. To guarantee authenticity, implement HMAC-SHA256 signatures. The server hashes the payload combined with a timestamp using a shared secret key, sending the result in a custom header. Clients verify this signature and check the timestamp to prevent replay attacks. For delivery resilience, process events using background queues with an exponential backoff retry strategy, and include unique delivery IDs so clients can deduplicate requests.

3. Standardized Error Handling
API failures must be structured and predictable. Adopt the RFC 7807 (Problem Details) standard to return uniform JSON payloads detailing the error type, title, status code, and specific validation failures. Use accurate HTTP status codes: 401 for invalid credentials, 403 for insufficient permissions, 422 for business logic or validation failures, and 429 for rate limits. Mask internal 500 server stack traces from public view, exposing only a tracking correlation ID.

4. Rate Limiting
Protect your infrastructure from abuse using algorithms like Token Bucket or Sliding Window Counters managed via a fast cache layer like Redis. Inform clients of their status using response headers indicating their total limit, remaining requests, and reset windows. When a threshold is breached, return a 429 Too Many Requests status with a Retry-After header. Layer your limits by applying strict IP-based caps on unauthenticated routes and tiered quotas for authenticated users.

"""


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap = 50,
    separators=['\n\n','\n', '.', ' ']
)

recursive_chunks = recursive_splitter.split_text(document)

# print(f"Recursive chunks: {len(recursive_chunks)}")
# for i, chunk in enumerate(recursive_chunks):
#     print(f"\\n--- Chunk {i+1} ({len(chunk)}) chars---")
#     print(chunk[:100] + "..." if len(chunk)>100 else chunk)



semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)


semantic_chunks =semantic_chunker.split_text(document)

print(f"Semantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\\n--- Chunk {i+1} ({len(chunk)}) chars---")
    print(chunk[:100] + "..." if len(chunk)>100 else chunk)
