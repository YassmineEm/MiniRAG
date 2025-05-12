from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import rag_qa

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(item: Question):
    result = rag_qa.run(item.question)
    return {"answer": result}
