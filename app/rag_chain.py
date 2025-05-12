from langchain.llms import HuggingFaceHub  
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  

vectordb = Chroma(persist_directory="./db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vectordb.as_retriever()

# Utilisation d'un modèle LLM gratuit de Hugging Face
rag_qa = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Modèle gratuit
        model_kwargs={"temperature": 0.5, "max_length": 512}
    ),
    retriever=retriever
)