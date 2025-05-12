from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vectordb = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
retriever = vectordb.as_retriever()

rag_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever
)
