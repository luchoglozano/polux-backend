from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

class AskRequest(BaseModel):
    question: str

app = FastAPI()

@app.on_event("startup")
def load_retriever():
    global qa_chain
    persist_dir = "./chroma_db"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=retriever
    )

@app.post("/ask")
async def ask_question(data: AskRequest):
    response = qa_chain.run(data.question)
    return {"answer": response}
