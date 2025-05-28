import os
from openai import OpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# ตั้งค่า MFECGPT API
# os.environ["MFEC_API_KEY"] = "sk-"
client = OpenAI(
    base_url="https://gpt.mfec.co.th/litellm",
    api_key=os.environ["MFEC_API_KEY"],
)

# โหลดเวกเตอร์ฐานความรู้
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
VECTOR_PATH = "./database/vectorstore_db"
vectorstore = FAISS.load_local(VECTOR_PATH, embedding_model, allow_dangerous_deserialization=True)

def ask_mfecgpt(chat_history: str, question: str) -> str:
    relevant_docs = vectorstore.similarity_search(question, k=5)

    if not relevant_docs:
        return ("❌ ไม่พบข้อมูลในคลังความรู้ที่เกี่ยวข้องกับคำถามของคุณ\n\n"
                "📌 กรุณาตรวจสอบคำถาม หรือปรับปรุงให้ชัดเจนมากขึ้น\n"
                "🧠 ตัวอย่าง: ลองเพิ่มคำสำคัญ เช่น หมวดหมู่ หรือคำเฉพาะทางที่เกี่ยวข้อง")

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    agentRole = f"""
    คุณคือผู้ช่วยอัจฉริยะที่มีบุคลิกใจดี เป็นมิตร และพร้อมช่วยเหลือเสมอ

    กรุณาตอบคำถามของผู้ใช้ด้วยภาษาที่เข้าใจง่าย อธิบายอย่างเป็นกันเอง และสุภาพ
    หากมีข้อมูลอ้างอิงจากบริบท ให้ตอบโดยสรุปเนื้อหา พร้อมแจ้งชื่อไฟล์ข้อมูลอ้างอิง
    หากไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในข้อมูลอ้างอิง" และให้คำแนะนำเบื้องต้นจากความรู้ของคุณอย่างเป็นมิตร พร้อมแจ้งว่านี่เป็นคำตอบจาก AI

    โปรดแสดงผลเป็น HTML เพื่อใช้ในระบบแชท โดยคงขนาดตัวอักษรให้เหมาะสม และพูดเหมือนคุณกำลังคุยกับคนจริง ๆ
    """

    prompt = f"""
    <ข้อมูลอ้างอิง>
    {context}
    </ข้อมูลอ้างอิง>

    <บทสนทนาก่อนหน้า>
    {chat_history}
    </บทสนทนาก่อนหน้า>

    <คำถาม>
    {question}
    </คำถาม>
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": agentRole},
                {"role": "user", "content": prompt}
            ]
        )
        return str(response.choices[0].message.content).strip();
    except Exception as e:
        return f"⚠️ เกิดข้อผิดพลาดขณะเรียก Gemini API: {e}"
