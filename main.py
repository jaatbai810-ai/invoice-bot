# main.py
import os
import io
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# PDF text extraction
import pdfplumber

# Google APIs
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# OpenAI (new SDK)
# pip install openai
from openai import OpenAI


APP_VERSION = "filehash-dedupe-needsreview-v2-strict-template"

# ----------------------------
# Strict template detection
# ----------------------------

TEMPLATE_STRICT_KEYWORDS = [
    "invoice template",
    "sample invoice",
    "example invoice",
    "this is a sample",
    "lorem ipsum",
    "placeholder",
    "type here",
    "enter your",
    "enter the",
    "insert your",
    "your company name",
    "your business name",
    "your address",
    "your logo",
]

TEMPLATE_STRICT_PATTERNS = [
    r"\blorem\s+ipsum\b",
    r"\binvoice\s+template\b",
    r"\b(sample|example)\s+invoice\b",
    r"\bplaceholder\b",
    r"\btype\s+here\b",
    r"\benter\s+(your|the)\s+(date|amount|invoice\s+number|invoice\s+no\.?|total|subtotal|tax|gst|vat|address|name)\b",
    r"\binsert\s+(your|the)\s+(logo|name|address)\b",
    r"\byour\s+(company|business)\s+name\b",
    r"\byour\s+address\b",
    r"\[.*?(enter|type|placeholder|your).{0,60}.*?\]",  # [Enter Amount]
    r"\{.*?(enter|type|placeholder|your).{0,60}.*?\}",  # {Enter Date}
]

def is_explicit_template_strict(raw_text: str) -> bool:
    """
    ONLY skip if we see explicit template/placeholder signals.
    Never skip because fields are missing or invoice is messy.
    """
    if not raw_text:
        return False  # scanned/empty -> needs_review, NOT skipped

    t = raw_text.lower()

    for kw in TEMPLATE_STRICT_KEYWORDS:
        if kw in t:
            return True

    for pat in TEMPLATE_STRICT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE | re.DOTALL):
            return True

    return False


# ----------------------------
# Helpers
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def short_sha256(data: bytes, n: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n]

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    return str(x).strip()

def build_dedupe_key(vendor: str, invoice_number: str, total: str) -> str:
    v = safe_str(vendor).lower()
    i = safe_str(invoice_number).lower()
    t = safe_str(total).lower()
    return f"{v}|{i}|{t}"

def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """
    Extract text from first N pages. If scanned, this will be empty/short.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            texts = []
            for idx, page in enumerate(pdf.pages[:max_pages]):
                txt = page.extract_text() or ""
                if txt:
                    texts.append(txt)
            return "\n".join(texts).strip()
    except Exception:
        return ""


# ----------------------------
# OpenAI extraction
# ----------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EXTRACTION_SCHEMA = {
    "vendor_name": "",
    "invoice_number": "",
    "invoice_date": "",
    "due_date": "",
    "currency": "",
    "subtotal": "",
    "tax": "",
    "total": "",
    "doc_type": "invoice",
}

def extract_invoice_fields_with_openai(client: OpenAI, raw_text: str) -> Dict[str, Any]:
    """
    Uses the PDF text layer. If text is empty, we still return blanks (needs_review).
    """
    # Keep prompt strict: return JSON only
    sys = (
        "You extract invoice fields from text. "
        "Return ONLY valid JSON with these keys:\n"
        "vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, doc_type\n"
        "Rules:\n"
        "- Use ISO date format YYYY-MM-DD when possible.\n"
        "- If unknown, return empty string.\n"
        "- doc_type should be 'invoice' unless clearly 'credit_note' or 'statement'.\n"
        "- No extra keys."
    )

    user = f"Invoice text:\n\n{raw_text[:20000]}"  # cap

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)

        # Normalize: ensure all keys exist
        out = dict(EXTRACTION_SCHEMA)
        for k in out.keys():
            if k in data:
                out[k] = safe_str(data.get(k))
        return out
    except Exception:
        # If OpenAI fails, return blanks -> needs_review
        return dict(EXTRACTION_SCHEMA)


def missing_fields(extracted: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Core missing -> triggers needs_review.
    Optional missing -> included in reason.
    """
    core = ["vendor_name", "invoice_date", "total"]
    optional = ["invoice_number", "currency", "due_date", "subtotal", "tax"]

    missing_core = [k for k in core if not safe_str(extracted.get(k))]
    missing_optional = [k for k in optional if not safe_str(extracted.get(k))]
    return missing_core, missing_optional


# ----------------------------
# Google Sheets
# ----------------------------

SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1").strip()

REQUIRED_HEADERS = [
    "vendor_name","invoice_number","invoice_date","due_date","currency","subtotal","tax","total",
    "file_hash","dedupe_key","status","pdf_url","doc_type","reason","processed_at"
]

def get_google_creds() -> Credentials:
    """
    Env var options:
    - GOOGLE_SERVICE_ACCOUNT_JSON: full JSON string
    OR
    - GOOGLE_SERVICE_ACCOUNT_FILE: path to json file
    """
    js = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    jf = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        # Drive scope not needed if webhook isn't uploading to Drive
    ]

    if js:
        info = json.loads(js)
        return Credentials.from_service_account_info(info, scopes=scopes)

    if jf:
        return Credentials.from_service_account_file(jf, scopes=scopes)

    raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE")

def sheets_service():
    creds = get_google_creds()
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def read_headers(svc) -> List[str]:
    rng = f"{SHEET_NAME}!1:1"
    res = svc.spreadsheets().values().get(spreadsheetId=SHEET_ID, range=rng).execute()
    values = res.get("values", [])
    return values[0] if values else []

def ensure_headers_ok(headers: List[str]) -> Optional[str]:
    # strict match by name (order matters for your mapping)
    if headers != REQUIRED_HEADERS:
        return (
            "Sheet headers do not match required Row 1 exactly.\n"
            f"Expected: {','.join(REQUIRED_HEADERS)}\n"
            f"Got: {','.join(headers)}"
        )
    return None

def find_existing_by_file_hash(svc, headers: List[str], file_hash: str) -> bool:
    """
    Simple dedupe: scan the file_hash column.
    For V1 scale this is fine. For huge sheets, optimize later.
    """
    try:
        idx = headers.index("file_hash")  # 0-based
    except ValueError:
        return False

    # Column letter: A=1 -> convert idx to letter
    col = idx + 1
    def col_to_letter(n: int) -> str:
        s = ""
        while n:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    letter = col_to_letter(col)
    rng = f"{SHEET_NAME}!{letter}:{letter}"
    res = svc.spreadsheets().values().get(spreadsheetId=SHEET_ID, range=rng).execute()
    col_vals = [row[0] for row in res.get("values", [])[1:] if row]  # skip header row
    return file_hash in col_vals

def append_row(svc, headers: List[str], row_dict: Dict[str, Any]) -> None:
    values = [safe_str(row_dict.get(h)) for h in headers]
    body = {"values": [values]}
    svc.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=f"{SHEET_NAME}!A:A",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI()

@app.get("/status")
def status():
    return {"status": "ok", "message": "server running", "version": APP_VERSION}

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    processed_at = now_iso()

    # Basic validation
    filename = file.filename or "uploaded.pdf"
    content_type = (file.content_type or "").lower()

    pdf_bytes = await file.read()
    if not pdf_bytes:
        return JSONResponse(
            status_code=400,
            content={"detail": "Empty file received. Make sure Zapier sends real file blob in multipart/form-data key 'file'."},
        )

    file_hash = short_sha256(pdf_bytes, n=16)

    # Extract text
    raw_text = extract_pdf_text(pdf_bytes)

    # Strict template skip (ONLY explicit template signals)
    if is_explicit_template_strict(raw_text):
        dedupe_key = build_dedupe_key("", "", "")
        return {
            "sheet_status": "skipped_template",
            "status": "skipped_template",
            "reason": "Explicit template/placeholder text detected (strict rule).",
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "processed_at": processed_at,
            "doc_type": "invoice",
            "filename": filename,
            "content_type": content_type,
        }

    # OpenAI extraction (even if raw_text empty -> blanks -> needs_review)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        extracted = dict(EXTRACTION_SCHEMA)
    else:
        client = OpenAI(api_key=api_key)
        extracted = extract_invoice_fields_with_openai(client, raw_text)

    # Determine needs_review vs processed
    missing_core, missing_optional = missing_fields(extracted)

    if not raw_text or len(raw_text.strip()) < 30:
        status_val = "needs_review"
        reason = "No/low text detected in PDF (likely scanned). Needs OCR/manual review."
    elif missing_core:
        status_val = "needs_review"
        reason = f"Missing key fields: {', '.join(missing_core)}. Missing optional: {', '.join(missing_optional)}."
    else:
        status_val = "processed"
        reason = "Processed successfully."

    dedupe_key = build_dedupe_key(
        extracted.get("vendor_name", ""),
        extracted.get("invoice_number", ""),
        extracted.get("total", ""),
    )

    # Google Sheets
    if not SHEET_ID:
        # still return extraction for debugging
        return {
            "sheet_status": "needs_review",
            "status": "needs_review",
            "reason": "Missing SHEET_ID env var on server.",
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "processed_at": processed_at,
            "doc_type": extracted.get("doc_type", "invoice"),
            "extracted": extracted,
        }

    try:
        svc = sheets_service()
        headers = read_headers(svc)

        hdr_err = ensure_headers_ok(headers)
        if hdr_err:
            return {
                "sheet_status": "needs_review",
                "status": "needs_review",
                "reason": hdr_err,
                "file_hash": file_hash,
                "dedupe_key": dedupe_key,
                "processed_at": processed_at,
                "doc_type": extracted.get("doc_type", "invoice"),
                "extracted": extracted,
            }

        # Primary dedupe: file_hash
        if find_existing_by_file_hash(svc, headers, file_hash):
            return {
                "sheet_status": "skipped_duplicate",
                "status": "skipped_duplicate",
                "reason": "Duplicate detected by file_hash (same PDF already processed).",
                "file_hash": file_hash,
                "dedupe_key": dedupe_key,
                "processed_at": processed_at,
                "doc_type": extracted.get("doc_type", "invoice"),
            }

        # Append row (ALWAYS for messy/scanned)
        row = {
            "vendor_name": extracted.get("vendor_name", ""),
            "invoice_number": extracted.get("invoice_number", ""),
            "invoice_date": extracted.get("invoice_date", ""),
            "due_date": extracted.get("due_date", ""),
            "currency": extracted.get("currency", ""),
            "subtotal": extracted.get("subtotal", ""),
            "tax": extracted.get("tax", ""),
            "total": extracted.get("total", ""),
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "status": status_val,
            "pdf_url": "",  # Zapier fills after Drive upload
            "doc_type": extracted.get("doc_type", "invoice"),
            "reason": reason,
            "processed_at": processed_at,
        }

        append_row(svc, headers, row)

        return {
            "sheet_status": "row_appended",
            "status": status_val,
            "reason": reason,
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "processed_at": processed_at,
            "doc_type": extracted.get("doc_type", "invoice"),
        }

    except Exception as e:
        # Don't break Zapier; return needs_review with error info
        return {
            "sheet_status": "needs_review",
            "status": "needs_review",
            "reason": f"Server error while writing to Google Sheets: {type(e).__name__}: {str(e)}",
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "processed_at": processed_at,
            "doc_type": extracted.get("doc_type", "invoice"),
        }
