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
# ENV / CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SHEET_ID = os.getenv("SHEET_ID", "")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "invoices")

# Render secret file path: /etc/secrets/service_account.json
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "/etc/secrets/service_account.json")
if not os.path.exists(SERVICE_ACCOUNT_FILE) and os.path.exists("service_account.json"):
    SERVICE_ACCOUNT_FILE = "service_account.json"


# =========================
# APP
# =========================
app = FastAPI()


@app.get("/")
def home():
    return {"status": "ok", "message": "server running"}


@app.get("/debug-env")
def debug_env():
    """Quick debug to confirm Render env vars are actually loaded."""
    sheet_id = os.getenv("SHEET_ID") or ""
    sa_path = os.getenv("SERVICE_ACCOUNT_FILE", "/etc/secrets/service_account.json")
    return {
        "SHEET_ID_present": bool(sheet_id),
        "SHEET_ID_value_preview": sheet_id[:6],
        "WORKSHEET_NAME": os.getenv("WORKSHEET_NAME"),
        "SERVICE_ACCOUNT_FILE": sa_path,
        "service_account_file_exists": os.path.exists(sa_path),
    }


# =========================
# HELPERS
# =========================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            t = (p.extract_text() or "").strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts).strip()


def clean_to_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return m.group(0).strip() if m else raw


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except:
        return None


def normalize_extracted(d: Dict[str, Any]) -> Dict[str, Any]:
    vendor_name = (d.get("vendor_name") or "").strip()
    invoice_number = (d.get("invoice_number") or "").strip()
    invoice_date = (d.get("invoice_date") or "").strip()
    due_date = (d.get("due_date") or "").strip()
    currency = (d.get("currency") or "").strip().upper()

    subtotal = safe_float(d.get("subtotal"))
    tax = safe_float(d.get("tax"))
    total = safe_float(d.get("total"))

    if total is None and subtotal is not None and tax is not None:
        total = subtotal + tax

    def out_num(v: Optional[float]) -> str:
        return "" if v is None else str(v)

    line_items = d.get("line_items") or []
    if not isinstance(line_items, list):
        line_items = []

    return {
        "vendor_name": vendor_name,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "currency": currency,
        "subtotal": out_num(subtotal),
        "tax": out_num(tax),
        "total": out_num(total),
        "line_items": line_items,
    }


def openai_extract(text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "Extract invoice fields from text. Return ONLY a valid JSON object with keys:\n"
        "vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, line_items.\n"
        "line_items is an array of objects: description, quantity, unit_price, amount.\n"
        "If unknown, use empty string or empty array."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"INVOICE TEXT:\n{text}"},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    content = clean_to_json_object(content)
    return json.loads(content)


def get_gspread_client() -> gspread.Client:
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise RuntimeError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return gspread.authorize(creds)


def append_row_by_headers(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes to sheet by matching your header names (Row 1) EXACTLY.
    Your headers:
    invoice_number, invoice_data, vendor_name, invoice_amount, due_date, currency, subtotal, tax, total
    """
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID missing")

    gc = get_gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)

    headers = ws.row_values(1)
    norm_headers = [h.strip().lower() for h in headers]

    row_map = {
        "invoice_number": extracted.get("invoice_number", ""),
        "invoice_data": extracted.get("invoice_date", ""),   # your sheet uses invoice_data
        "vendor_name": extracted.get("vendor_name", ""),
        "invoice_amount": extracted.get("total", ""),        # your sheet uses invoice_amount
        "due_date": extracted.get("due_date", ""),
        "currency": extracted.get("currency", ""),
        "subtotal": extracted.get("subtotal", ""),
        "tax": extracted.get("tax", ""),
        "total": extracted.get("total", ""),
    }

    row = [row_map.get(h, "") for h in norm_headers]
    ws.append_row(row, value_input_option="USER_ENTERED")

    # last row number (simple approach)
    last_row = len(ws.get_all_values())
    return {
        "sheet_status": "row_appended",
        "sheet_row": last_row,
        "worksheet_used": WORKSHEET_NAME,
        "service_account_file_used": os.path.basename(SERVICE_ACCOUNT_FILE),
    }


# =========================
# ROUTE
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)

        # If empty text, OCR is the next upgrade. For now still try with what we have.
        model_out = openai_extract(text if text else "")
        extracted = normalize_extracted(model_out)

        sheet_info = append_row_by_headers(extracted)

        return {
            "filename": file.filename,
            "chars": len(text),
            "extracted": extracted,
            **sheet_info,
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "filename": getattr(file, "filename", ""),
                "error": str(e),
            },
        )
