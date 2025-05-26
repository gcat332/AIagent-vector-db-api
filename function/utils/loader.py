import os
import io
import pytesseract
import tempfile
import pypandoc
from PIL import Image
from pdf2image import convert_from_path
from loguru import logger
from langchain.schema import Document
from langchain.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    TextLoader
)

# OCR จากรูปใน PDF
async def extract_images_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        ocr_texts = [pytesseract.image_to_string(image, lang="tha+eng") for image in images]
        return "\n".join(ocr_texts)
    except Exception as e:
        logger.error(f"OCR error in {pdf_path}: {e}")
        return ""

# โหลดไฟล์ทุกประเภทที่รองรับ
async def load_all_supported_files(filepath, category):
    ext = filepath.lower().split('.')[-1]
    docs = []

    try:
        if ext == "pdf":
            docs += PyMuPDFLoader(filepath).load()
            ocr_text = await extract_images_from_pdf(filepath)
            if ocr_text.strip():
                docs.append(Document(page_content=ocr_text, metadata={"source": filepath, "category": category}))

        elif ext == "docx":
            docs += UnstructuredWordDocumentLoader(filepath).load()

        elif ext == "doc":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_docx = tmp_file.name
            try:
                pypandoc.convert_file(filepath, 'docx', outputfile=tmp_docx)
                docs += UnstructuredWordDocumentLoader(tmp_docx).load()
                os.remove(tmp_docx)
            except Exception as e:
                logger.error(f"❌ Failed to convert .doc to .docx: {filepath} | Error: {e}")

        elif ext == "pptx":
            docs += UnstructuredPowerPointLoader(filepath).load()

        elif ext == "xlsx":
            docs += UnstructuredExcelLoader(filepath).load()

        elif ext in ["html", "htm"]:
            docs += UnstructuredHTMLLoader(filepath).load()

        elif ext == "txt":
            docs += TextLoader(filepath).load()

        else:
            logger.warning(f"⚠️ Unsupported file format: {filepath}")

    except Exception as e:
        logger.error(f"❌ Error loading {filepath}: {e}")

    for doc in docs:
        doc.metadata["category"] = category
        doc.metadata["source"] = filepath

    return docs
