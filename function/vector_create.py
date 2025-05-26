import os
import csv
import asyncio
import torch
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loader import load_all_supported_files

# ปิด tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Directory setup
ROOT_DIR = "./database/datalake_db"
OUTPUT_DIR = "./database/vectorstore_db"
LOG_DIR = "./logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Log setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"vector_process_{timestamp}.log")
csv_file = os.path.join(LOG_DIR, f"vector_summary_{timestamp}.csv")
logger.add(log_file, rotation="1 MB", retention="7 days", level="INFO")

# Embedding model (HuggingFace multilingual)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Log summary records
summary_records = []

async def process_category(category):
    category_path = os.path.join(ROOT_DIR, category)
    if not os.path.isdir(category_path):
        return

    logger.info(f"📂 Processing category: {category}")
    all_docs = []
    file_success = 0
    file_failed = 0

    # 🔁 ใช้ os.walk เพื่อไล่ทุกไฟล์ในทุก subfolder
    for root, dirs, files in os.walk(category_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                docs = await load_all_supported_files(file_path, category)
                if docs:
                    all_docs.extend(docs)
                    file_success += 1
                    logger.info(f"✅ {category} :: Loaded: {filename} | {len(docs)} docs")
                else:
                    logger.warning(f"⚠️ {category} :: Empty: {filename}")
            except Exception as e:
                file_failed += 1
                logger.error(f"❌ {category} :: Failed: {filename} | Error: {str(e)}")

    if not all_docs:
        logger.warning(f"⚠️ No documents found in category: {category}")
        return

    chunks = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_path = os.path.join(OUTPUT_DIR, category)
    vectorstore.save_local(save_path)
    logger.success(f"✅ Saved vectorstore for category: {category} | Chunks: {len(chunks)}")

    # Save summary
    summary_records.append({
        "category": category,
        "total_files": file_success + file_failed,
        "files_success": file_success,
        "files_failed": file_failed,
        "total_documents": len(all_docs),
        "total_chunks": len(chunks)
    })

async def main():
    categories = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    await tqdm_asyncio.gather(*(process_category(cat) for cat in categories))

    # Save CSV summary
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "total_files", "files_success", "files_failed", "total_documents", "total_chunks"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    logger.info(f"📝 Saved summary to CSV: {csv_file}")

if __name__ == "__main__":
    asyncio.run(main())
