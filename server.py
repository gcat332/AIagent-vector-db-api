# import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from function.ask_gemini import ask_gemini
from function.ask_mfecgpt import ask_mfecgpt
from function.sum_mfecgpt import sum_mfecgpt
# from function.kb_create import update_knowledge_vector

app = FastAPI()

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
    
class RecordQuery(BaseModel):
    table: str
    number: str
    state: str
    short_desc: str
    description: str
    assignment_group: str
    assigned_to: str
    resolution_code: str
    close_notes : str
    work_note : str
    ai_kb_answer : str

# @app.post("/agent_gemini")
# def query_agent_gemini(data: AgentQuery):
#     try:
#         answer = ask_gemini(data.chat_history, data.question)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"⚠️Error gemini agent API: {e}")

@app.post("/agent_mfecgpt")
def query_agent_mfecgpt(data: AgentQuery):
    try:
        answer = ask_mfecgpt(data.chat_history, data.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️Error mfecgpt agent API: {e}")

@app.post("/sum_mfecgpt")
def summary_agent_mfecgpt(data: RecordQuery):
    try:
        answer = sum_mfecgpt(data)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️Error mfecgpt agent API: {e}")
