from fastapi import FastAPI, UploadFile, File
import io
import json
import re
import os

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

import gspread
from google.oauth2.service_account import Credentials

# Loads .env locally (Render env vars also work automatically)
load_dotenv()

app = FastAPI()

# -------- ENV VARS --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_WORKSHEET_NAME = os.getenv("GOOGLE_WORKSHEET_NAME", "Sheet1")

# On Render: /etc/secrets/service_account.json
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")

# -------- CLIENTS --------
client = OpenAI(api_key=OPENAI_API_KEY)


def clean_text(t: str) -> str:
    # Fix weird repeated letters caused by some PDF fonts
    t = re.sub(r'([A-Za-z])\1{2,}', r'\1', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def extract_json(text: str) -> str:
    """
    Removes ```json fences and returns only the JSON object if possible.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def get_sheet():
    """
    Connect to Google Sheets using service account file.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    ws = sh.worksheet(GOOGLE_WORKSHEET_NAME)
    return ws


@app.get("/")
def home():
    return {"status": "ok", "message": "server running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # -------- sanity checks --------
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY (set it in Render env vars or .env)"}
    if not GOOGLE_SHEET_ID:
        return {"error": "Missing GOOGLE_SHEET_ID (set it in Render env vars or .env)"}

    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        return {"error": f"Please upload a PDF. Got content_type={file.content_type}"}

    pdf_bytes = await file.read()

    # -------- 1) PDF text extraction --------
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    if len(text.strip()) == 0:
        return {
            "filename": file.filename,
            "size_bytes": len(pdf_bytes),
            "error": "No text found. Likely scanned PDF (image). Needs OCR."
        }

    text = clean_text(text)

    # -------- 2) OpenAI -> JSON --------
    schema_hint = {
        "vendor_name": "",
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "currency": "",
        "subtotal": "",
        "tax": "",
        "total": "",
        "line_items": []
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
    json_str = extract_json(output_text)

    try:
        data = json.loads(json_str)
    except Exception:
        return {
            "filename": file.filename,
            "chars": len(text),
            "preview": text[:700],
            "raw_model_output": output_text[:2000],
            "error": "Model did not return valid JSON"
        }

    # -------- 3) Write to Google Sheets --------
    try:
        ws = get_sheet()

        row = [
            data.get("vendor_name", ""),
            data.get("invoice_number", ""),
            data.get("invoice_date", ""),
            data.get("due_date", ""),
            data.get("currency", ""),
            data.get("subtotal", ""),
            data.get("tax", ""),
            data.get("total", ""),
        ]

        ws.append_row(row, value_input_option="USER_ENTERED")

    except Exception as e:
        return {
            "filename": file.filename,
            "extracted": data,
            "service_account_file_used": SERVICE_ACCOUNT_FILE,
            "sheet_id_used": GOOGLE_SHEET_ID,
            "worksheet_used": GOOGLE_WORKSHEET_NAME,
            "error": f"Google Sheets write failed: {str(e)}"
        }

    return {
        "filename": file.filename,
        "chars": len(text),
        "extracted": data,
        "sheet_status": "row_appended",
        "service_account_file_used": SERVICE_ACCOUNT_FILE,
        "worksheet_used": GOOGLE_WORKSHEET_NAME
    }
