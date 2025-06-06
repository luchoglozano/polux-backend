from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Definición del modelo de solicitud
class AskRequest(BaseModel):
    question: str

# Inicialización de la aplicación FastAPI
app = FastAPI()

# Carga del vectorstore y configuración del chain en el arranque
@app.on_event("startup")
def load_qa_chain():
    global qa_chain
    persist_directory = "./chroma_db"
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=retriever
    )

# Endpoint POST para recibir preguntas
@app.post("/ask")
async def ask_question(data: AskRequest):
    answer = qa_chain.run(data.question)
    return {"answer": answer}
