# ✅ Base image ขนาดเล็ก
FROM python:3.10.14

# ✅ ติดตั้ง system dependencies ที่จำเป็นสำหรับบางไลบรารี (เช่น pdf2image, pytesseract)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ สร้าง working directory
WORKDIR /app

# ✅ ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ คัดลอกโปรเจกต์ทั้งหมดเข้า container
COPY . .

# ✅ เปิดพอร์ตสำหรับ FastAPI
EXPOSE 8080

# ✅ คำสั่งเริ่มต้น
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
