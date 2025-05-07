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

    pattern = re.compile(
        r'(\d+)\s*(plus|\+|minus|\-)\s*(\d+)\s*(?:equals|=)\s*(\d+)',
        flags=re.IGNORECASE
    )

    assessments = []
    for a_str, op_str, b_str, c_str in pattern.findall(full_text):
        a, b, c = int(a_str), int(b_str), int(c_str)
        op = 'plus' if op_str.lower() in ('plus', '+') else 'minus'
        expected = a + b if op == 'plus' else a - b
        is_correct = (expected == c)
        text = f"{a} {op_str} {b} = {c}"
        assessments.append({
            "text": text,
            "left": a,
            "op": op,
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
