import os
import io
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI

import gspread
from google.oauth2.service_account import Credentials


# ======================
# ENV
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SHEET_ID = os.getenv("SHEET_ID")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "invoices")

# Render secret file path
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/service_account.json")

app = FastAPI()


# ======================
# ROUTES
# ======================
@app.get("/")
def health():
    return {"status": "ok", "message": "server running", "version": "dedupe-v1"}


# ======================
# HELPERS
# ======================
def extract_text(pdf_bytes: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n\n".join(parts).strip()


def coerce_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if not s:
        return None

    s = s.replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum())


def looks_like_template(extracted: dict, raw_text: str) -> bool:
    text = (raw_text or "").lower()

    markers = ["template", "sample", "lorem", "enter date", "enter invoice", "placeholder"]
    if any(m in text for m in markers):
        return True

    vendor = (extracted.get("vendor_name") or "").strip().lower()
    inv = (extracted.get("invoice_number") or "").strip().lower()

    if "[" in vendor or "]" in vendor or "[" in inv or "]" in inv:
        return True

    if not vendor or vendor in {"n/a", "na", "none", "null"}:
        return True
    if not inv or inv in {"n/a", "na", "none", "null"}:
        return True

    if "xxxx" in inv or inv in {"0000", "1234", "12345"}:
        return True

    return False


def make_dedupe_key(extracted: dict, raw_text: str) -> str:
    vendor = normalize(extracted.get("vendor_name", ""))
    inv = normalize(extracted.get("invoice_number", ""))
    total = extracted.get("total")

    total_str = ""
    try:
        if total not in (None, "", "null"):
            total_str = f"{float(total):.2f}"
    except Exception:
        total_str = ""

    if vendor and inv and total_str:
        return f"{vendor}|{inv}|{total_str}"

    preview = (raw_text or "")[:2000].encode("utf-8", errors="ignore")
    h = hashlib.sha256(preview).hexdigest()[:16]
    return f"{vendor or 'unknown'}|{inv or h}|{total_str or '0.00'}"


def openai_extract_invoice(text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
Return ONLY JSON with keys:
vendor_name, invoice_number, invoice_date (YYYY-MM-DD), due_date (YYYY-MM-DD),
currency, subtotal, tax, total.

Use null when missing. Do not add extra text.

TEXT:
{text}
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(resp.choices[0].message.content or "{}")

    # normalize number fields
    data["subtotal"] = coerce_float(data.get("subtotal"))
    data["tax"] = coerce_float(data.get("tax"))
    data["total"] = coerce_float(data.get("total"))

    # doc_type optional (default invoice)
    data["doc_type"] = (data.get("doc_type") or "invoice").strip().lower()

    return data


def open_sheet():
    if not SHEET_ID:
        raise HTTPException(status_code=500, detail="SHEET_ID not set")

    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise HTTPException(status_code=500, detail=f"service_account.json not found at {SERVICE_ACCOUNT_PATH}")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(WORKSHEET_NAME)
    return ws


def clean_header(h: str) -> str:
    # remove normal + non-breaking spaces
    return (h or "").replace("\u00A0", " ").strip()


def get_headers(ws) -> List[str]:
    headers_raw = ws.row_values(1)
    if not headers_raw:
        raise HTTPException(status_code=500, detail="Row 1 headers empty")
    return [clean_header(h) for h in headers_raw]


def col_to_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def dedupe_exists(ws, headers: List[str], dedupe_key: str, lookback: int = 2000) -> bool:
    if "dedupe_key" not in headers:
        return False

    idx = headers.index("dedupe_key") + 1
    col = col_to_letter(idx)

    vals = ws.get(f"{col}2:{col}")
    flat = [(r[0] or "").strip() for r in vals if r and len(r) > 0]

    if not flat:
        return False

    last = flat[-lookback:] if len(flat) > lookback else flat
    return dedupe_key in last


def build_row(headers: List[str], extracted: Dict[str, Any], dedupe_key: str) -> List[str]:
    now_iso = datetime.now(timezone.utc).isoformat()

    # Your sheet mapping (IMPORTANT)
    values_map = {
        "invoice_number": extracted.get("invoice_number"),
        "invoice_data": extracted.get("invoice_date"),   # your special mapping
        "vendor_name": extracted.get("vendor_name"),
        "invoice_amount": extracted.get("total"),        # your special mapping
        "due_date": extracted.get("due_date"),
        "currency": extracted.get("currency"),
        "subtotal": extracted.get("subtotal"),
        "tax": extracted.get("tax"),
        "total": extracted.get("total"),

        # extra columns
        "dedupe_key": dedupe_key,
        "status": "processed",
        "pdf_url": "",  # fill later
        "doc_type": extracted.get("doc_type", "invoice"),
        "reason": "",
        "processed_at": now_iso,
    }

    row: List[str] = []
    for h in headers:
        v = values_map.get(h)
        row.append("" if v is None else str(v))
    return row


# ======================
# ENDPOINT
# ======================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) extract pdf text
    raw_text = extract_text(pdf_bytes)

    # 2) openai extraction
    extracted = openai_extract_invoice(raw_text)

    # 3) template skip
    if looks_like_template(extracted, raw_text):
        return {
            **extracted,
            "sheet_status": "skipped_template",
            "status": "skipped_template",
            "reason": "Looks like a blank/template invoice (placeholders detected).",
        }

    # 4) dedupe key
    dedupe_key = make_dedupe_key(extracted, raw_text)

    # 5) open sheet + headers
    ws = open_sheet()
    headers = get_headers(ws)

    # 6) dedupe check
    if dedupe_exists(ws, headers, dedupe_key):
        return {
            **extracted,
            "dedupe_key": dedupe_key,
            "sheet_status": "skipped_duplicate",
            "status": "skipped_duplicate",
            "reason": "duplicate invoice detected",
        }

    # 7) append header-mapped row (fills extra columns)
    row = build_row(headers, extracted, dedupe_key)
    ws.append_row(row, value_input_option="USER_ENTERED")

    # 8) return (so Zapier step 4 shows dedupe_key)
    return {
        **extracted,
        "dedupe_key": dedupe_key,
        "sheet_status": "row_appended",
        "status": "processed",
    }
