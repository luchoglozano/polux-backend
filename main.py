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

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_qa_chain():
    global qa_chain
    persist_directory = "./chroma_db"
    
    # Verifica si ya existe la base persistida
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        # Carga todos los PDF desde la carpeta /docs
        all_docs = []
        for file in os.listdir("./docs"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join("./docs", file))
                all_docs.extend(loader.load())

        # Divide el texto en chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)

        # Crea el vectorstore y guarda
        vectordb = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
        vectordb.persist()
    else:
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
