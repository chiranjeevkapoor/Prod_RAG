from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI

print(f"LangChain version : {core_version}")
print(f"LangGraph version : {lg_version}")





def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
