from fastapi import FastAPI, UploadFile, File
import io
import json
import re
import os

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

def clean_text(t: str) -> str:
    # Fix some weird repeated letters caused by certain PDF fonts
    t = re.sub(r'([A-Za-z])\1{2,}', r'\1', t)  # "CCCOmpany" -> "Company"
    t = re.sub(r'\s+', ' ', t)                 # collapse whitespace
    return t.strip()

def extract_json(text: str) -> str:
    """
    Removes ```json fences if present and tries to keep only the JSON object.
    """
    text = text.strip()

    # Remove starting fence like ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove ending fence ```
    text = re.sub(r"\s*```$", "", text)

    # Keep only the JSON object part if extra text exists
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return text

@app.get("/")
def home():
    return {"status": "ok", "message": "server running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    # 1) Extract text from PDF
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    if len(text.strip()) == 0:
        return {
            "filename": file.filename,
            "size_bytes": len(pdf_bytes),
            "error": "No text found. This PDF is likely scanned (image). Needs OCR."
        }

    text = clean_text(text)

    # 2) Ask OpenAI to return strict JSON
    schema_hint = {
        "vendor_name": "",
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "currency": "",
        "subtotal": "",
        "tax": "",
        "total": "",
        "line_items": [
            {"description": "", "quantity": "", "unit_price": "", "amount": ""}
        ]
    }

    prompt = f"""
Extract invoice fields from the text below.
Return ONLY valid JSON. No commentary. No markdown.

If a field is missing, use "".
Dates as YYYY-MM-DD if possible.
Numbers as strings.

Expected JSON shape example:
{json.dumps(schema_hint, indent=2)}

INVOICE TEXT:
{text}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    output_text = resp.output_text.strip()

    # 3) Remove ```json fences if the model still adds them
    json_str = extract_json(output_text)

    # 4) Parse JSON safely
    try:
        data = json.loads(json_str)
    except Exception:
        return {
            "filename": file.filename,
            "chars": len(text),
            "preview": text[:700],
            "raw_model_output": output_text[:2000],
            "cleaned_json_attempt": json_str[:2000],
            "error": "Model did not return valid JSON"
        }

    return {
        "filename": file.filename,
        "chars": len(text),
        "extracted": data
    }
