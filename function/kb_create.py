import os
import uuid
import json
import torch
import pandas as pd
import google.generativeai as genai
from langchain.vectorstores import FAISS # Assuming this is from langchain_community.vectorstores
from langchain_community.embeddings import HuggingFaceEmbeddings # Corrected import path for newer Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ตั้งค่า API key ของ Gemini
# For security, it's better to load API keys from environment variables or a secure config
# However, using the provided setup for this example.
os.environ["GOOGLE_API_KEY"] = "AIzaSyDoeFXNAZ40C6j2Pwn_He5ZkU_pCIdzQ0k" # Replace with your actual key if needed
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Ensure the model name is valid for your access level.
# Common models: "gemini-1.5-flash-latest", "gemini-1.0-pro"
# Using "gemini-pro" as a generally available and capable model.
# If "gemini-2.0-flash" is specific to your setup and works, you can keep it.
try:
    model = genai.GenerativeModel("gemini-2.0-flash") # Or your specific "gemini-2.0-flash"
except Exception as e:
    print(f"Error initializing GenerativeModel. Ensure model name is correct and API key is valid: {e}")
    print("Falling back to 'gemini-pro' if 'gemini-1.5-flash-latest' or your specified model failed.")
    try:
        model = genai.GenerativeModel("gemini-pro")
    except Exception as e_fallback:
        print(f"Fallback to 'gemini-pro' also failed: {e_fallback}")
        exit("Could not initialize a Gemini model. Please check your API key and model name.")


# โหลด embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

def load_vectorstore_by_category(path):
    # allow_dangerous_deserialization is necessary for FAISS with custom embeddings
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def summarize_with_gemini(text, category):
    prompt = f"""คุณเป็นผู้ช่วยสร้างคลังความรู้
จากข้อความด้านล่างในหมวดหมู่ "{category}" ให้คุณสรุปข้อมูลออกมาเป็นลิสต์ของ JSON objects โดยแต่ละ object มีรูปแบบดังนี้:
{{
  "KnowledgeName": "ชื่อความรู้โดยย่อ",
  "KnowledgeDetail": "สรุปความรู้โดยรวมอย่างกระชับ"
}}

หากมีหลายองค์ความรู้ที่สามารถสรุปได้จากข้อความ ให้สร้าง JSON object แยกสำหรับแต่ละองค์ความรู้
โปรดตอบเป็น JSON array เท่านั้น ตามรูปแบบตัวอย่างนี้ และไม่มีข้อความอื่นนอกเหนือจาก JSON array:
[
  {{
    "KnowledgeName": "ตัวอย่างชื่อความรู้ 1",
    "KnowledgeDetail": "ตัวอย่างสรุปความรู้ 1"
  }},
  {{
    "KnowledgeName": "ตัวอย่างชื่อความรู้ 2",
    "KnowledgeDetail": "ตัวอย่างสรุปความรู้ 2"
  }}
]

ข้อความ:
{text}

โปรดตอบเป็นเนื้อหาโดยสรุปภาษาไทยเป็นหลัก ยกเว้นพวกชื่อเมนู หรือ ชื่อเฉพาะ ให้เป็นภาษาอังกฤษได้ และอยู่ใน Format JSON array เท่านั้น และไม่มีข้อความอื่นนอกเหนือจาก JSON array นี้"""
    try:
        response = model.generate_content(prompt)
        # Clean the response text: Gemini might sometimes wrap output in markdown
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[len("```json"):]
        if cleaned_text.startswith("```"):
             cleaned_text = cleaned_text[len("```"):]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")]
        cleaned_text = cleaned_text.strip()
        return cleaned_text
    except Exception as e:
        print(f"Gemini API error for category {category}: {e}")
        return None

def extract_individual_json_objects(text_data):
    """
    Extracts individual JSON objects from a string that might contain multiple such objects,
    possibly mixed with other text, if the primary JSON array parsing fails.
    Uses json.JSONDecoder.raw_decode for robust sequential parsing.
    """
    json_objects = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text_data):
        # Find the next opening curly brace
        current_pos = text_data.find('{', idx)
        if current_pos == -1:
            break # No more opening braces

        try:
            # Attempt to decode a JSON object starting from this position
            obj, end_pos_relative = decoder.raw_decode(text_data[current_pos:])
            json_objects.append(obj)
            idx = current_pos + end_pos_relative # Move scanner past the decoded object
        except json.JSONDecodeError:
            # If decoding fails, advance past the current opening brace to avoid infinite loops
            # on malformed JSON.
            idx = current_pos + 1
    return json_objects

def generate_knowledge_base(vector_dir, output_csv_file):
    records = []
    if not os.path.exists(vector_dir):
        print(f"❌ Error: Vector directory '{vector_dir}' not found.")
        return

    for category in os.listdir(vector_dir):
        category_path = os.path.join(vector_dir, category)
        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")
        try:
            vs = load_vectorstore_by_category(category_path)
        except Exception as e:
            print(f"⚠️ Error loading vector store for category {category}: {e}")
            continue

        # Try to get all documents directly from docstore
        combined_text = ""
        reference_files = []
        if hasattr(vs, 'docstore') and hasattr(vs.docstore, '_dict') and vs.docstore._dict:
            all_doc_items = list(vs.docstore._dict.values())
            if all_doc_items:
                combined_text = "\n\n".join([doc.page_content for doc in all_doc_items if doc.page_content])
                reference_files = sorted(list(set(doc.metadata.get("source", "N/A") for doc in all_doc_items)))
            else:
                print(f"No documents found in docstore for category {category}.")

        if not combined_text: # Fallback or if docstore was empty
            print(f"Docstore for '{category}' was empty or inaccessible, trying similarity_search.")
            try:
                num_docs_to_fetch = vs.index.ntotal if hasattr(vs.index, 'ntotal') else 100 # Fetch many
                docs_sim = vs.similarity_search(" ", k=num_docs_to_fetch) # Generic query
                if docs_sim:
                    combined_text = "\n\n".join([doc.page_content for doc in docs_sim if doc.page_content])
                    reference_files = sorted(list(set(doc.metadata.get("source", "N/A") for doc in docs_sim)))
                else:
                    print(f"No docs found via similarity_search in category {category}.")
            except Exception as e_sim:
                print(f"Error during similarity_search for {category}: {e_sim}")


        if not combined_text.strip():
            print(f"No content to summarize for category {category}, skipping.")
            continue

        summary_text_from_gemini = summarize_with_gemini(combined_text, category)
        if not summary_text_from_gemini:
            print(f"Failed to get summary from Gemini for category {category}. Skipping.")
            continue

        parsed_summaries = []
        try:
            # Primary attempt: Parse as a JSON array
            parsed_data = json.loads(summary_text_from_gemini)
            if isinstance(parsed_data, list):
                parsed_summaries = parsed_data
            elif isinstance(parsed_data, dict): # Handle if Gemini returns a single object instead of array
                parsed_summaries = [parsed_data]
                print(f"ℹ️ Gemini returned a single JSON object for {category}, processed as a list of one.")
            else:
                # This case means it parsed to something else (e.g. string, number), which is unexpected.
                print(f"⚠️ Gemini response for {category} parsed but was not a JSON array or object: {type(parsed_data)}. Attempting fallback.")
                raise json.JSONDecodeError("Unexpected data type after initial parse", summary_text_from_gemini, 0)

        except json.JSONDecodeError as jde:
            error_snippet = summary_text_from_gemini[:200] + "..." if len(summary_text_from_gemini) > 200 else summary_text_from_gemini
            print(f"⚠️ Failed to parse Gemini response as JSON array for {category}. Error: {jde}. Response snippet: '{error_snippet}'. Attempting fallback extraction.")
            parsed_summaries = extract_individual_json_objects(summary_text_from_gemini)
            if not parsed_summaries:
                print(f"Fallback extraction also found no JSON objects in Gemini response for {category}.")
                continue # Skip if fallback also fails
            else:
                print(f"ℹ️ Fallback extraction successful: Found {len(parsed_summaries)} JSON object(s) for {category}.")

        # Process the list of summary dictionaries (from primary or fallback)
        for summary_item in parsed_summaries:
            if not isinstance(summary_item, dict):
                print(f"Skipping non-dictionary item in parsed summaries for {category}: {summary_item}")
                continue

            records.append({
                "KnowledgeID": str(uuid.uuid4()),
                "Category": category,
                "KnowledgeName": summary_item.get("KnowledgeName", "N/A"),
                "KnowledgeDetail": summary_item.get("KnowledgeDetail", "N/A"),
                "ReferenceFile": "; ".join(reference_files) if reference_files else "N/A"
            })

        print(f"Successfully processed {len(parsed_summaries)} knowledge items for category: {category}")


    if not records:
        print("No records were generated overall.")
        return

    df = pd.DataFrame(records)
    try:
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig') # Save to CSV
        print(f"✅ Knowledge base saved to CSV: {output_csv_file}")
    except Exception as e:
        print(f"❌ Error saving DataFrame to CSV: {e}")

def update_knowledge_vector(vector_dir,output_csv_file):
    try:
        df = pd.read_csv(output_csv_file)
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for _, row in df.iterrows():
            content = f"{row['KnowledgeDetail']}\n\n[Reference: {row['ReferenceFile']}]"
            metadata = {
                "KnowledgeID": row['KnowledgeID'],
                "Category": row['Category'],
                "KnowledgeName": row['KnowledgeName'],
                "ReferenceFile": row['ReferenceFile']
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vector_dir)

        print(f"✅ Knowledge base saved to VECTOR: {vector_dir}")
        return {"status": "Knowledge vector updated."}
    except Exception as e:
        print(f"❌ Error saving DataFrame to CSV: {e}")

if __name__ == "__main__":
    VECTORSTORE_DIR = "./database/vectorstore_db"
    OUTPUT_CSV = "./database/datalake_db/knowledge_base.csv"
    if not os.path.exists(VECTORSTORE_DIR):
        print(f"Creating dummy vectorstore directory: {VECTORSTORE_DIR}")
        os.makedirs(VECTORSTORE_DIR)
        dummy_category_path = os.path.join(VECTORSTORE_DIR, "dummy_category")
        if not os.path.exists(dummy_category_path):
            os.makedirs(dummy_category_path)
            print(f"Created dummy category folder: {dummy_category_path}")
            print("Note: This dummy category likely won't have a valid FAISS index and will be skipped.")
            print("Please ensure your actual vectorstore folders with FAISS indexes are present.")

    generate_knowledge_base(VECTORSTORE_DIR, OUTPUT_CSV)
    update_knowledge_vector(VECTORSTORE_DIR, OUTPUT_CSV)
