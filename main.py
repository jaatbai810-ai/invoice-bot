# main.py
import os
import json
import re
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import pdfplumber

import gspread
from google.oauth2.service_account import Credentials

from openai import OpenAI


app = FastAPI(title="Invoice Bot", version="1.0.0")


# ---------------------------
# Config
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SHEET_ID = os.getenv("SHEET_ID", "")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "invoices")

# Render secret file location (you said this exists)
SERVICE_ACCOUNT_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/etc/secrets/service_account.json"
)

# Your sheet headers (row 1) — we will write in exactly this order
SHEET_COLUMNS = [
    "invoice_number",
    "invoice_data",     # mapped from invoice_date
    "vendor_name",
    "invoice_amount",   # mapped from total
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
]


# ---------------------------
# Helpers
# ---------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


def safe_float(val: Any) -> Optional[float]:
    """Convert values like '1,245.00', '$124.50', 'AUD 99' to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # remove currency symbols/letters and keep digits, dot, comma, minus
    s = re.sub(r"[^0-9\.,\-]", "", s)
    # handle thousands commas
    if s.count(",") > 0 and s.count(".") <= 1:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def normalize_extracted(obj: Any) -> Dict[str, Any]:
    """
    Make sure keys exist and types are reasonable.
    Expected keys:
    vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, line_items(optional)
    """
    if not isinstance(obj, dict):
        obj = {}

    def gs(key: str) -> str:
        v = obj.get(key)
        if v is None:
            return ""
        return str(v).strip()

    normalized = {
        "vendor_name": gs("vendor_name"),
        "invoice_number": gs("invoice_number"),
        "invoice_date": gs("invoice_date"),
        "due_date": gs("due_date"),
        "currency": gs("currency") or "AUD",
        "subtotal": safe_float(obj.get("subtotal")) or 0.0,
        "tax": safe_float(obj.get("tax")) or 0.0,
        "total": safe_float(obj.get("total")) or 0.0,
    }

    # Keep line_items if present (not used for sheet but useful in response)
    li = obj.get("line_items")
    if isinstance(li, list):
        normalized["line_items"] = li
    else:
        normalized["line_items"] = []

    return normalized


def looks_like_template_pdf(
    filename: str,
    extracted: Dict[str, Any],
    text: str
) -> bool:
    """
    Safety gate: block obvious placeholder/template PDFs so they don't pollute the sheet.
    """
    fname = (filename or "").lower()
    inv_no = (extracted.get("invoice_number") or "").strip().lower()
    vendor = (extracted.get("vendor_name") or "").strip().lower()
    preview_text = (text or "").lower()

    bad_tokens = [
        "[", "]",
        "enter date",
        "payment due date",
        "invoicesimple",
        "template",
        "placeholder",
    ]

    return (
        "invoicesimple-pdf-template" in fname
        or "template" in fname
        or any(tok in preview_text for tok in bad_tokens)
        or ("[" in inv_no or "]" in inv_no)
        or ("[" in vendor or "]" in vendor)
        or inv_no in ["", "0", "0000", "na", "n/a"]
        or vendor in ["", "na", "n/a"]
    )


def get_gspread_client() -> gspread.Client:
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise RuntimeError(f"service_account.json not found at {SERVICE_ACCOUNT_PATH}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH, scopes=scopes)
    return gspread.authorize(creds)


def append_to_sheet(extracted: Dict[str, Any]) -> str:
    """
    Append one row to Google Sheets.
    Uses your mapping:
      invoice_data <- invoice_date
      invoice_amount <- total
    """
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID missing (Render env var not set).")

    gc = get_gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.sheet1  # fallback

    row_payload = {
        "invoice_number": extracted.get("invoice_number", ""),
        "invoice_data": extracted.get("invoice_date", ""),     # mapping
        "vendor_name": extracted.get("vendor_name", ""),
        "invoice_amount": extracted.get("total", 0.0),         # mapping
        "due_date": extracted.get("due_date", ""),
        "currency": extracted.get("currency", ""),
        "subtotal": extracted.get("subtotal", 0.0),
        "tax": extracted.get("tax", 0.0),
        "total": extracted.get("total", 0.0),
    }

    row = [row_payload.get(col, "") for col in SHEET_COLUMNS]
    ws.append_row(row, value_input_option="USER_ENTERED")
    return "row_appended"


def extract_with_openai(text: str) -> Dict[str, Any]:
    """
    Use OpenAI to extract invoice fields as JSON.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (Render env var not set).")

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You extract structured invoice data from text. "
        "Return ONLY valid JSON (no markdown). "
        "If a field is missing, return an empty string for text fields and 0 for numbers.\n\n"
        "Required keys:\n"
        "vendor_name (string)\n"
        "invoice_number (string)\n"
        "invoice_date (string, YYYY-MM-DD if possible)\n"
        "due_date (string, YYYY-MM-DD if possible)\n"
        "currency (string like AUD/USD)\n"
        "subtotal (number)\n"
        "tax (number)\n"
        "total (number)\n"
        "line_items (array of {description, quantity, unit_price, amount})\n"
    )

    user = f"Invoice text:\n\n{text}"

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
    try:
        data = json.loads(content)
    except Exception:
        # In case model returns something weird, try to salvage JSON
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    return data


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "server running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Zapier sends form-data:
      key: file
      value: Gmail attachment
    """
    try:
        file_bytes = await file.read()
        text = extract_pdf_text(file_bytes)

        if not text:
            return JSONResponse(
                status_code=200,
                content={
                    "filename": file.filename,
                    "chars": 0,
                    "extracted": {},
                    "sheet_status": "skipped_no_text",
                    "reason": "No extractable text found (might be a scanned PDF). Add OCR later.",
                },
            )

        model_out = extract_with_openai(text)
        extracted = normalize_extracted(model_out)

        # --- Safety: skip templates/placeholders ---
        if looks_like_template_pdf(file.filename or "", extracted, text):
            return {
                "filename": file.filename,
                "chars": len(text),
                "extracted": extracted,
                "sheet_status": "skipped_template",
                "reason": "Looks like a blank/template invoice (placeholders detected).",
            }

        # Append to sheet
        sheet_status = append_to_sheet(extracted)

        return {
            "filename": file.filename,
            "chars": len(text),
            "extracted": extracted,
            "sheet_status": sheet_status,
        }

    except Exception as e:
        # Keep errors readable for Zapier debugging
        return JSONResponse(
            status_code=200,
            content={
                "filename": getattr(file, "filename", ""),
                "error": True,
                "message": str(e),
            },
        )
