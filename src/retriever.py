from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTOR_DB_DIR = "data/vector_db"

def retrieve(query: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "What is LangChain?"
    docs = retrieve(query)

    print("\nRetrieved chunks:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.page_content}\n")
