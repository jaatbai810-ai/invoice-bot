import os
import re
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

import pdfplumber
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import gspread
from google.oauth2.service_account import Credentials

from openai import OpenAI


# =========================
# CONFIG (Environment Vars)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SHEET_ID = os.getenv("SHEET_ID", "")  # Google Sheet ID (from URL)
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "invoices")

# Render Secret File path is usually /etc/secrets/<filename>
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "/etc/secrets/service_account.json")

# If running locally and you keep the file in project folder:
if not os.path.exists(SERVICE_ACCOUNT_FILE) and os.path.exists("service_account.json"):
    SERVICE_ACCOUNT_FILE = "service_account.json"


# =========================
# APP
# =========================
app = FastAPI()


@app.get("/")
def home():
    return {"status": "ok", "message": "server running"}


# =========================
# HELPERS
# =========================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_parts: List[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n\n".join(text_parts).strip()


def clean_model_json(raw: str) -> str:
    """
    If the model returns ```json ... ``` or extra text, extract JSON object.
    """
    raw = raw.strip()

    # Remove code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    # Try to find first {...} block
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return raw


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # remove currency symbols/commas
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except:
        return None


def normalize_extracted(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure keys exist and totals are consistent.
    """
    vendor_name = (data.get("vendor_name") or "").strip()
    invoice_number = (data.get("invoice_number") or "").strip()
    invoice_date = (data.get("invoice_date") or "").strip()
    due_date = (data.get("due_date") or "").strip()
    currency = (data.get("currency") or "").strip().upper()

    subtotal = safe_float(data.get("subtotal"))
    tax = safe_float(data.get("tax"))
    total = safe_float(data.get("total"))

    # If total missing but subtotal + tax present, compute
    if total is None and subtotal is not None and tax is not None:
        total = subtotal + tax

    # Default numeric blanks to empty strings (Sheets friendly)
    def num_out(v: Optional[float]) -> str:
        return "" if v is None else str(v)

    line_items = data.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []

    return {
        "vendor_name": vendor_name,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "currency": currency,
        "subtotal": num_out(subtotal),
        "tax": num_out(tax),
        "total": num_out(total),
        "line_items": line_items,
    }


def openai_extract(invoice_text: str) -> Dict[str, Any]:
    """
    Call OpenAI to extract structured invoice JSON.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Keep prompt simple + strict
    system = (
        "You extract invoice fields from raw text. "
        "Return ONLY valid JSON object with keys:\n"
        "vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, line_items.\n"
        "line_items is an array of objects: description, quantity, unit_price, amount.\n"
        "If unknown, use empty string or empty array."
    )

    user = f"INVOICE TEXT:\n{invoice_text}"

    # response_format json_object makes it output strict JSON (when supported)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    content = clean_model_json(content)
    parsed = json.loads(content)
    return parsed


def get_gspread_client() -> gspread.Client:
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise RuntimeError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return gspread.authorize(creds)


def append_row_by_headers(extracted: dict) -> dict:
    """
    Appends by matching your Sheet headers EXACTLY (row 1).
    Fixes the column shift problem permanently.
    """
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID missing")

    gc = get_gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)

    # Read header row
    headers = ws.row_values(1)
    # Normalize (handles extra spaces/case)
    norm_headers = [h.strip().lower() for h in headers]

    # Build a map that matches YOUR headers
    row_map = {
        "invoice_number": extracted.get("invoice_number", ""),
        "invoice_data": extracted.get("invoice_date", ""),   # your header uses invoice_data
        "vendor_name": extracted.get("vendor_name", ""),
        "invoice_amount": extracted.get("total", ""),        # your header uses invoice_amount
        "due_date": extracted.get("due_date", ""),
        "currency": extracted.get("currency", ""),
        "subtotal": extracted.get("subtotal", ""),
        "tax": extracted.get("tax", ""),
        "total": extracted.get("total", ""),
    }

    # Build row in the same order as the sheet headers
    row = [row_map.get(h, "") for h in norm_headers]

    ws.append_row(row, value_input_option="USER_ENTERED")
    last_row = len(ws.get_all_values())

    return {
        "sheet_status": "row_appended",
        "sheet_row": last_row,
        "worksheet_used": WORKSHEET_NAME,
        "service_account_file_used": os.path.basename(SERVICE_ACCOUNT_FILE),
    }


# =========================
# ROUTES
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Zapier sends multipart/form-data with field name: file
    """
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)

        # Keep preview short
        preview = (text[:800] + "…") if len(text) > 800 else text

        # If PDF has almost no text, extraction will be weak (OCR is next upgrade)
        # Still send it to OpenAI so you can see behavior.
        model_out = openai_extract(text if text else preview)
        extracted = normalize_extracted(model_out)

        sheet_info = append_row_by_headers(extracted)

        return {
            "filename": file.filename,
            "chars": len(text),
            "preview": preview,
            "extracted": extracted,
            **sheet_info,
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,  # keep 200 so Zapier doesn't hard-fail; you can change to 500 later
            content={
                "filename": getattr(file, "filename", ""),
                "error": str(e),
            },
        )
