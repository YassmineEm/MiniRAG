import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store():
    all_docs = []
    
    # Parcourir tous les fichiers PDF dans le dossier ./data/
    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("data", filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    # Découper les documents en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(all_docs)

    # Créer les embeddings et stocker dans Chroma
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs_split, embedding, persist_directory="./db")
    vectordb.persist()


