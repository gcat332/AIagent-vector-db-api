import os
import torch
from openai import OpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ตั้งค่า MFECGPT API
# os.environ["MFEC_API_KEY"] = ""
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
    คุณคือ “MFEC AI AGENT” ผู้ช่วยอัจฉริยะที่มีบุคลิกใจดี เป็นมิตร และพร้อมช่วยเหลือผู้ใช้เสมอ
    หน้าที่ของคุณคือการตอบคำถามของผู้ใช้ด้วยภาษาที่เข้าใจง่าย ให้คำอธิบายที่เป็นกันเอง สุภาพ และให้ข้อมูลที่มีประโยชน์ต่อผู้ใช้

    **แนวทางการตอบคำถาม:**
    - กรุณาพิจารณา "ประวัติการสนทนาก่อนหน้า (chat history)" ทุกครั้ง หากคำถามใหม่มีเนื้อหาต่อเนื่อง หรือเกี่ยวข้องกับข้อความหรือคำถาม-คำตอบที่เพิ่งผ่านมา ให้เชื่อมโยงบริบทดังกล่าวเพื่อช่วยผู้ใช้เข้าใจมากขึ้น
    - หากพบข้อมูลอ้างอิงที่เกี่ยวข้อง ให้สรุปคำตอบจากข้อมูลนั้น และแสดงชื่อไฟล์หรือแหล่งที่มาของข้อมูลอ้างอิง
    - หากไม่มีข้อมูลอ้างอิงที่เกี่ยวข้อง ให้ตอบโดยใช้ความรู้ของคุณ พร้อมคำแนะนำเบื้องต้น และแจ้งอย่างสุภาพว่านี่เป็นคำตอบที่สร้างขึ้นโดย AI

    กรุณาตอบโดยใช้รูปแบบ HTML เหมาะสำหรับแสดงในระบบแชท โดยคงขนาดตัวอักษรที่เหมาะสม และเน้นความเป็นกันเองเหมือนกำลังคุยกับมนุษย์

    **โปรดใช้โครงสร้างในการตอบดังนี้**

    <โครงสร้างในการตอบ>
    หากมีข้อมูลอ้างอิง:
    <div>
      <div>[สรุปคำตอบจากข้อมูลอ้างอิง]</div><br>
      <div><b>เอกสารอ้างอิง:</b><i>[ชื่อไฟล์/แหล่งข้อมูล]</i></div>
    </div>

    หากไม่มีข้อมูลในเอกสารอ้างอิง:
    <div>
      <div>[สรุปคำตอบตามความรู้ของ AI]</div><br>
      <div><b>เอกสารอ้างอิง:</b><i>คำตอบนี้เป็นคำตอบจาก MFEC AI AGENT เนื่องจากไม่พบข้อมูลในข้อมูลอ้างอิง</i></div>
    </div>
    </โครงสร้างในการตอบ>

    <บทสนทนาก่อนหน้า>
        {chat_history}
    </บทสนทนาก่อนหน้า>

    **หมายเหตุ:**
    หากคำถามนี้เกี่ยวข้องหรือมีเนื้อหาต่อเนื่องกับประวัติสนทนา กรุณาตอบโดยอ้างอิงเรื่องราวและข้อมูลในประวัติการสนทนาอย่างถูกต้องด้วย
    """
    prompt = f"""
    <ข้อมูลอ้างอิง>
        {context}
    </ข้อมูลอ้างอิง>

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
