import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store():
    all_docs = []
    
    for filename in os.listdir("data"):
        if filename.endswith(".docx"):
            docx_path = os.path.join("data", filename)
            try:
                loader = Docx2txtLoader(docx_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"Erreur lors du chargement du fichier {filename}: {e}")

    if not all_docs:
        raise ValueError("Aucun document valide n'a été chargé depuis le dossier data")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(all_docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs_split, embedding, persist_directory="./db")
    print("Vector store créé avec succès!")

