# main.py
import os
import io
import re
import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

import pdfplumber
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# Config
# -----------------------------
APP_VERSION = "v4-reviewpolicy+melbourne-time"
SHEET_ID = os.getenv("SHEET_ID", "").strip()

# Accept either name (you used GOOGLE_WORKSHEET_NAME)
SHEET_NAME = (os.getenv("GOOGLE_WORKSHEET_NAME") or os.getenv("SHEET_NAME") or "invoices").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

API_KEY = os.getenv("API_KEY", "").strip()

# Your fixed header order (Row 1 must match)
HEADERS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "file_hash",
    "dedupe_key",
    "status",
    "pdf_url",
    "doc_type",
    "reason",
    "processed_at",
]

# Minimum text to even try OpenAI extraction
MIN_TEXT_CHARS = 60

# Stricter "low-text/scanned" detection for review policy
LOW_TEXT_REVIEW_COMPACT_CHARS = 250

# Timezone for timestamps
TZ = ZoneInfo("Australia/Melbourne")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()


# -----------------------------
# Security (API key)
# -----------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing API_KEY env var")
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -----------------------------
# Google auth helpers
# -----------------------------
def load_service_account_info() -> Dict[str, Any]:
    """
    Supports:
      - GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON string)
      - GOOGLE_SERVICE_ACCOUNT_FILE (path)
      - GOOGLE_APPLICATION_CREDENTIALS (path)  <-- most common on Render
      - SECRET_ACCOUNT_FILE (path)             <-- some older setups
    """
    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        return json.loads(json_str)

    file_path = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("SECRET_ACCOUNT_FILE")
        or ""
    ).strip()

    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        "Missing GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE / "
        "GOOGLE_APPLICATION_CREDENTIALS / SECRET_ACCOUNT_FILE"
    )


def get_gspread_client() -> gspread.Client:
    info = load_service_account_info()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def get_worksheet():
    if not SHEET_ID:
        raise RuntimeError("Missing SHEET_ID env var")
    gc = get_gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    return sh.worksheet(SHEET_NAME)


def ensure_headers_ok(ws) -> Optional[str]:
    row1 = ws.row_values(1)
    if not row1:
        return "Row 1 headers are empty. Add headers exactly and freeze row 1."
    if row1 != HEADERS:
        return "Sheet headers mismatch. Row 1 must match exactly:\n" + ",".join(HEADERS)
    return None


# -----------------------------
# PDF helpers
# -----------------------------
def sha256_short(data: bytes, n: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_chunks.append(t)
    return "\n".join(text_chunks).strip()


def is_low_text_pdf(pdf_text: str, min_compact_chars: int = LOW_TEXT_REVIEW_COMPACT_CHARS) -> bool:
    """
    More robust "scanned/low-text" check:
    - strips whitespace so random newlines don't inflate char count
    """
    t = (pdf_text or "").strip()
    compact = re.sub(r"\s+", "", t)
    return len(compact) < min_compact_chars


# -----------------------------
# Template detection (ULTRA strict)
# Only skip if explicit template placeholders/words exist
# -----------------------------
TEMPLATE_PATTERNS = [
    r"\binvoice template\b",
    r"\btemplate invoice\b",
    r"\blorem ipsum\b",
    r"\bplaceholder\b",
    r"\btype here\b",
    r"\benter (date|amount|invoice|invoice number|total|subtotal)\b",
    r"\bfill (in|out)\b",
    r"\bsample invoice\b",
    r"\bdemo invoice\b",
    r"\bexample invoice\b",
    r"\bstart typing\b",
    r"\bdelete this\b",
    r"\breplace with\b",
    r"\bcompany name here\b",
    r"\baddress here\b",
    r"\bphone here\b",
    r"\bemail here\b",
    # bracket placeholders like [enter ...] or <enter ...>
    r"\[[^\]]{0,40}(enter|placeholder|type)[^\]]{0,40}\]",
    r"\<[^\>]{0,40}(enter|placeholder|type)[^\>]{0,40}\>",
]


def looks_like_template(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    for pat in TEMPLATE_PATTERNS:
        if re.search(pat, low):
            return True
    return False


# -----------------------------
# Review policy (V4)
# Mark needs_review if ANY:
# - invoice_number missing
# - currency missing OR currency has no evidence in text
# - total missing OR total not explicitly labeled (Total/Amount Due/Balance Due/Grand Total/Invoice Total)
# - scanned/low-text PDF
# -----------------------------
TOTAL_LABEL_RE = re.compile(r"\b(total|amount\s+due|balance\s+due|grand\s+total|invoice\s+total)\b", re.IGNORECASE)

CURRENCY_EVIDENCE_RE = re.compile(
    r"\b(AUD|USD|NZD|CAD|EUR|GBP|INR|SGD|HKD)\b|(?:^|\s)[$€£]\s?\d",
    re.IGNORECASE,
)


def has_total_label(pdf_text: str) -> bool:
    return bool(TOTAL_LABEL_RE.search(pdf_text or ""))


def has_currency_evidence(pdf_text: str) -> bool:
    return bool(CURRENCY_EVIDENCE_RE.search(pdf_text or ""))


# -----------------------------
# OpenAI extraction
# -----------------------------
def openai_extract(text: str) -> Dict[str, Any]:
    """
    Returns dict with keys:
      vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, doc_type
    """
    if not OPENAI_API_KEY:
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system = (
            "You extract invoice fields from text. "
            "Return ONLY valid JSON with these keys:\n"
            "vendor_name, invoice_number, invoice_date, due_date, currency, subtotal, tax, total, doc_type.\n"
            "Dates must be YYYY-MM-DD if possible. Use empty string if unknown.\n"
            "IMPORTANT: Do not guess. If a field is not explicitly present, return empty string."
        )

        user = f"Invoice text:\n{text}\n\nReturn JSON only."

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").strip()

        # Try to parse JSON even if model wrapped it
        json_match = re.search(r"\{.*\}", content, re.S)
        if json_match:
            content = json_match.group(0)

        data = json.loads(content)
        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_currency(x: Any) -> str:
    s = clean_str(x).upper()
    if re.fullmatch(r"[A-Z]{3}", s):
        return s
    if s in ["$", "AUD$"]:
        return "AUD"
    return s


def normalize_amount(x: Any) -> str:
    s = clean_str(x)
    if not s:
        return ""
    s = re.sub(r"[^\d\.\-]", "", s)
    if not re.search(r"\d", s):
        return ""
    return s


def normalize_date(x: Any) -> str:
    s = clean_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.search(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})", s)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""


def review_policy(extracted: Dict[str, Any], pdf_text: str) -> Tuple[str, str]:
    """
    Returns (status, reason)
    status: processed | needs_review
    reason: semicolon-separated reasons (only when needs_review), otherwise "Processed successfully."
    """
    reasons: List[str] = []

    invoice_number = clean_str(extracted.get("invoice_number"))
    currency = normalize_currency(extracted.get("currency"))
    total = normalize_amount(extracted.get("total"))

    if not invoice_number:
        reasons.append("invoice_number missing")

    if not currency:
        reasons.append("currency missing")
    else:
        if not has_currency_evidence(pdf_text):
            reasons.append("currency uncertain (no currency symbol/code detected in PDF text)")

    if total:
        if not has_total_label(pdf_text):
            reasons.append("total not explicitly labeled (no Total/Amount Due/Balance Due/Grand Total in PDF text; may be inferred)")
    else:
        reasons.append("total missing")

    if is_low_text_pdf(pdf_text):
        reasons.append("scanned/low-text PDF (weak text extraction; OCR/manual review needed)")

    if reasons:
        return "needs_review", "; ".join(reasons)

    return "processed", "Processed successfully."


# -----------------------------
# Sheet helpers
# -----------------------------
def find_row_by_file_hash(ws, file_hash: str) -> Optional[int]:
    """
    Returns sheet row number (1-based). Data starts at row 2.
    """
    idx = HEADERS.index("file_hash") + 1  # 1-based
    col_vals = ws.col_values(idx)  # includes header at row 1
    target = (file_hash or "").strip()
    for i in range(2, len(col_vals) + 1):
        if (col_vals[i - 1] or "").strip() == target:
            return i
    return None


def append_invoice_row(ws, row_dict: Dict[str, Any]) -> int:
    """
    Appends a row and returns row_number.
    """
    values = [clean_str(row_dict.get(h, "")) for h in HEADERS]
    ws.append_row(values, value_input_option="USER_ENTERED")

    # Return last row number reliably by searching hash
    fh = clean_str(row_dict.get("file_hash", ""))
    rn = find_row_by_file_hash(ws, fh)
    if rn:
        return rn

    # fallback (rare)
    return len(ws.get_all_values())


def update_cell_by_header(ws, row_number: int, header: str, value: str):
    col = HEADERS.index(header) + 1
    ws.update_cell(row_number, col, value)


# -----------------------------
# API models
# -----------------------------
class PdfUrlUpdate(BaseModel):
    file_hash: str
    pdf_url: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/status")
def status():
    return {"status": "ok", "message": "server running", "version": APP_VERSION}


@app.post("/upload")
def upload(file: UploadFile = File(...), _auth: None = Depends(require_api_key)):
    # Melbourne time (with timezone offset)
    processed_at = datetime.now(TZ).isoformat(timespec="seconds")

    try:
        pdf_bytes = file.file.read()
        if not pdf_bytes:
            return JSONResponse(status_code=400, content={"detail": "Empty file"})

        file_hash = sha256_short(pdf_bytes, 16)

        # Extract text
        text = extract_pdf_text(pdf_bytes)

        # Strict template skip
        if looks_like_template(text):
            return {
                "sheet_status": "skipped_template",
                "status": "skipped_template",
                "reason": "Looks like a template invoice (explicit template/placeholder words detected).",
                "file_hash": file_hash,
                "dedupe_key": "",
                "processed_at": processed_at,
                "doc_type": "invoice",
            }

        # Open sheet
        ws = get_worksheet()
        hdr_err = ensure_headers_ok(ws)
        if hdr_err:
            return {
                "sheet_status": "needs_review",
                "status": "needs_review",
                "reason": f"Server error: {hdr_err}",
                "file_hash": file_hash,
                "dedupe_key": "",
                "processed_at": processed_at,
                "doc_type": "invoice",
            }

        # Dedupe by file_hash
        existing = find_row_by_file_hash(ws, file_hash)
        if existing:
            return {
                "sheet_status": "skipped_duplicate",
                "status": "skipped_duplicate",
                "reason": "Duplicate PDF (file_hash already exists).",
                "file_hash": file_hash,
                "dedupe_key": "",
                "processed_at": processed_at,
                "doc_type": "invoice",
            }

        # Extract fields (smart), but status decided by stricter policy
        extracted: Dict[str, Any] = {}
        if len(text) >= MIN_TEXT_CHARS:
            extracted = openai_extract(text)
        else:
            extracted = {}

        vendor_name = clean_str(extracted.get("vendor_name"))
        invoice_number = clean_str(extracted.get("invoice_number"))
        invoice_date = normalize_date(extracted.get("invoice_date"))
        due_date = normalize_date(extracted.get("due_date"))
        currency = normalize_currency(extracted.get("currency"))
        subtotal = normalize_amount(extracted.get("subtotal"))
        tax = normalize_amount(extracted.get("tax"))
        total = normalize_amount(extracted.get("total"))
        doc_type = clean_str(extracted.get("doc_type")) or "invoice"

        # Dedupe key (secondary, informational)
        dedupe_key = f"{vendor_name.lower()}|{invoice_number.lower()}|{total}".strip()

        # Apply review policy (based on your desired rules)
        policy_status, policy_reason = review_policy(
            {"invoice_number": invoice_number, "currency": currency, "total": total},
            text,
        )

        # Optional extra notes (won't flip processed -> needs_review; just notes)
        extra_notes: List[str] = []
        if not vendor_name:
            extra_notes.append("vendor_name missing")
        if not invoice_date:
            extra_notes.append("invoice_date missing")

        if policy_status == "needs_review":
            parts = [policy_reason] if policy_reason else []
            if extra_notes:
                parts.append("extra: " + ", ".join(extra_notes))
            reason = "; ".join([p for p in parts if p]).strip()
        else:
            reason = "Processed successfully."
            if extra_notes:
                reason = reason + " Note: " + ", ".join(extra_notes)

        row = {
            "vendor_name": vendor_name,
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "currency": currency,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "status": policy_status,
            "pdf_url": "",  # filled later by /update_pdf_url
            "doc_type": doc_type,
            "reason": reason,
            "processed_at": processed_at,
        }

        row_number = append_invoice_row(ws, row)

        return {
            "sheet_status": "row_appended",
            "status": policy_status,
            "reason": reason,
            "file_hash": file_hash,
            "dedupe_key": dedupe_key,
            "processed_at": processed_at,
            "doc_type": doc_type,
            "row_number": row_number,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error: {type(e).__name__}: {str(e)}"},
        )


@app.post("/update_pdf_url")
def update_pdf_url(payload: PdfUrlUpdate, _auth: None = Depends(require_api_key)):
    file_hash = (payload.file_hash or "").strip()
    pdf_url = (payload.pdf_url or "").strip()

    if not file_hash or not pdf_url:
        return JSONResponse(status_code=400, content={"detail": "file_hash and pdf_url are required"})

    try:
        ws = get_worksheet()
        hdr_err = ensure_headers_ok(ws)
        if hdr_err:
            return JSONResponse(status_code=500, content={"detail": hdr_err})

        row_number = find_row_by_file_hash(ws, file_hash)
        if not row_number:
            return JSONResponse(status_code=404, content={"detail": f"Row not found for file_hash={file_hash}"})

        update_cell_by_header(ws, row_number, "pdf_url", pdf_url)

        return {"status": "ok", "file_hash": file_hash, "row_number": row_number, "pdf_url": pdf_url}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error updating pdf_url: {type(e).__name__}: {str(e)}"},
        )
