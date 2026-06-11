import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (TextLoader, WebBaseLoader, DirectoryLoader, PyPDFLoader)

from dotenv import load_dotenv
load_dotenv()

def pdf_loader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents(s) from PDF")

    for i , doc in enumerate(documents):
        print(f"Document {i+1} Content preview: {doc.page_content[:100]}")
        print(f"Metadate: {doc.metadata}")
        
