# import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from function.ask_gemini import ask_gemini
# from function.kb_create import update_knowledge_vector

app = FastAPI()

# Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# @app.post("/update_vector")
# def update_vector():
#     try:
#         subprocess.run(["python", "function/vector_create.py"], check=True)
#         return {"status": "✅ Vector creation completed."}
#     except subprocess.CalledProcessError as e:
#         raise HTTPException(status_code=500, detail=f"⚠️ Error updating vector: {e}")

# @app.post("/update_knowledge")
# def update_knowledge():
#     try:
#         VECTORSTORE_DIR = "database/vectorstore_db"
#         OUTPUT_CSV = "database/datalake_db/knowledge_base.csv"
#         update_knowledge_vector(VECTORSTORE_DIR, OUTPUT_CSV)
#         return {"status": "✅ Knowledge creation completed."}
#     except subprocess.CalledProcessError as e:
#         raise HTTPException(status_code=500, detail=f"⚠️Error creating knowledge base: {e}")

class AgentQuery(BaseModel):
    chat_history: str
    question: str

@app.post("/agent")
def query_agent(data: AgentQuery):
    try:
        answer = ask_gemini(data.chat_history, data.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️Error gemini agent API: {e}")
