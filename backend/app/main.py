# backend/app/main.py

import os
import re
import PyPDF2
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
def assess_pdf_bytes(data: bytes) -> List[Dict]:
    """
    Extracts text from PDF bytes and finds all simple equations
    of the form "<int> plus|minus <int> equals <int>" or using '+'/'-' and '='.
    Returns a list of dicts with assessment results.
    """
    reader = PyPDF2.PdfReader(BytesIO(data))
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    # ── normalize unicode operators to ASCII ────────────────────────────────
    full_text = (
        full_text
        .replace('−', '-')   # unicode minus → hyphen-minus
        .replace('×', '*')   # multiplication sign → asterisk
        .replace('÷', '/')   # division sign → slash
    )

    # match +, -, *, /, ×, ÷  (we escape * in the class)
    pattern = re.compile(
        r'\(?\s*(-?\d+)\s*\)?'  # optional parens/spaces around first number
        r'\s*([+\-*/])\s*'      # operator
        r'\(?\s*(-?\d+)\s*\)?'  # optional parens/spaces around second number
        r'\s*=\s*'
        r'\(?\s*(-?\d+)\s*\)?'  # optional parens/spaces around result
    )


    assessments = []
    for a_str, op_str, b_str, c_str in pattern.findall(full_text):
        a, b = int(a_str), int(b_str)
        # always parse the RHS; override for division if needed
        c = int(c_str)
        if op_str == '+':
            expected = a + b
            op_key = 'plus'
        elif op_str == '-':
            expected = a - b
            op_key = 'minus'
        elif op_str == '*':
            expected = a * b
            op_key = 'multiply'
        elif op_str == '/':
            expected = None if b == 0 else a / b
            c = float(c_str)       # re-parse as float for division checks
            op_key = 'divide'
        else:
            expected = None
            op_key = 'unknown'
        
        # ── determine correctness ───────────────────────────────────
        if isinstance(expected, float):
            is_correct = abs(expected - c) < 1e-9
        else:
            is_correct = (expected == c)

        assessments.append({
            "text": f"{a} {op_str} {b} = {c_str}",
            "left": a,
            "op": op_key,
            "right": b,
            "result": c,
            "is_correct": is_correct
        })
    return assessments

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
    assessments = assess_pdf_bytes(data)
    return {"assessments": assessments}
