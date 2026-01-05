import os
from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

VECTOR_DB_DIR = "data/vector_db"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        {
            "role": "system",
            "content": "Answer the question using ONLY the provided context.other wise please say I can't answer"
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "" 
    
    while question.lower() != 'false':
        question = input("Please enter your question (or type 'False' to quit): ")
        
        if question.lower() != 'false':
            answer = ask(question)  # Ensure you have a defined ask() function
            print("\nAnswer:\n")
            print(answer)
