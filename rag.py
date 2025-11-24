import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data/realestate.txt"
VECTOR_PATH = "vector_store"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def load_and_split():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.create_documents([text])
    return documents

def build_vector_store():
    print("Building vector store...")
    documents = load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(VECTOR_PATH)

    return vectordb

def get_vector_store():
    if os.path.exists(VECTOR_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectordb
    return build_vector_store()

def get_retriever():
    vectordb = get_vector_store()
    # Updated retriever configuration for newer LangChain versions
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever