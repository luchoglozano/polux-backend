from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "POLUX backend operativo"}

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_qa_chain():
    global qa_chain
    persist_directory = "./chroma_db"
    
    # Verifica si ya existe la base persistida
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=retriever
    )

@app.post("/ask")
async def ask_question(data: AskRequest):
    response = qa_chain.run(data.question)
    return {"answer": response}
