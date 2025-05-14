# backend/app/main.py

import os
import re
import PyPDF2
import unicodedata
from io import BytesIO
from typing import List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ─── CORS ────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health Check ───────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Backend is running and Agent-1 is loaded."}

# ─── Evaluation Endpoint ────────────────────────────────────────────────────────
@app.post("/predict/evaluate")
def evaluate(payload: dict):
    """
    Expects JSON { "input": "Question: ... Student Answer: ..." }
    Returns  JSON { "evaluation": "<model response>" }
    """
    prompt = payload.get("input")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    # TODO: implement evaluation logic
    return {"evaluation": None}

# ─── PDF Assessment Helpers ─────────────────────────────────────────────────────
def assess_pdf_bytes(data: bytes) -> Dict:
    """
    Extracts text from PDF bytes and finds all arithmetic equations
    of any length on the LHS (e.g. 5+8-4=9), preserving original text.
    Returns a dict with:
      - "assessments": list of { text, expression, result, is_correct }
      - "formatted":   newline-separated display string
    """
    # 1) extract all text
    reader = PyPDF2.PdfReader(BytesIO(data))
    full_text = "".join(page.extract_text() or "" for page in reader.pages)

    # 2) normalize to ASCII (operators, dashes, compatibility forms)
    full_text = unicodedata.normalize("NFKC", full_text)
    for dash in ["−", "–", "—", "‒"]:
        full_text = full_text.replace(dash, "-")
    full_text = full_text.replace("×", "*").replace("÷", "/")

    # 3) regex: group1 = entire LHS expr, group2 = RHS integer
    pattern = re.compile(r'([\d+\-*/()\s]+)\s*=\s*(-?\d+)')

    assessments: List[Dict] = []
    lines: List[str]     = []

    for m in pattern.finditer(full_text):
        orig_eq = m.group(0).strip()  # e.g. "(5+8-4)=9"
        lhs_str = m.group(1)          # e.g. "5+8-4"
        c_str   = m.group(2)          # e.g. "9"

        # parse RHS
        c = int(c_str)

        # safely evaluate LHS (only +,-,*,/ expected)
        try:
            expected = eval(lhs_str)
        except Exception:
            continue

        # correctness check (float tolerance for division)
        if isinstance(expected, float):
            is_correct = abs(expected - c) < 1e-9
        else:
            is_correct = (expected == c)

        # collect structured result
        assessments.append({
            "text":       orig_eq,
            "expression": lhs_str,
            "result":     c,
            "is_correct": is_correct
        })

        # build formatted line
        mark = "✅" if is_correct else "❌"
        lines.append(f"{orig_eq} → {mark} {is_correct}")

    return {
        "assessments": assessments,
        "formatted":   "\n".join(lines)
    }

# ─── File Upload & Assessment Endpoint ─────────────────────────────────────────
@app.post("/upload/assessment")
async def upload_assessment(file: UploadFile = File(...)):
    # enforce PDF only
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")

    # read all bytes once
    data = await file.read()

    # save PDF for future processing
    upload_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "uploads")
    )
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(data)

    # assess equations directly from bytes
    result = assess_pdf_bytes(data)
    return result
