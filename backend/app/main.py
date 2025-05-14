# backend/app/main.py

import os
import re
import PyPDF2
import unicodedata
from io import BytesIO
from typing import List, Dict
from sympy import symbols, Eq, solve
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
    Parses PDF text for:
      1) simple arithmetic lines (any-length LHS = RHS)
      2) single-variable linear equations with provided answer (e.g. "2x+7=15, x=4")
    Returns a dict with:
      - assessments: list of detail dicts
      - formatted: newline-separated feedback strings
    """
    # ── 1) Extract & normalize text ───────────────────────────────
    reader = PyPDF2.PdfReader(BytesIO(data))
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    full_text = unicodedata.normalize("NFKC", full_text)
    for dash in ["−","–","—","‒"]:
        full_text = full_text.replace(dash, "-")
    full_text = full_text.replace("×","*").replace("÷","/")

    assessments: List[Dict] = []
    lines:       List[str]  = []

    # ── 2) Linear-with-solution pattern ───────────────────────────
    # matches "2x+7=15, x=4" capturing [coeff, var, const, rhs, provided]
    pat_lin = re.compile(
        r'(-?\d*)\s*([a-zA-Z])\s*([+\-]\s*\d+)\s*=\s*(-?\d+)'
        r'\s*,\s*\2\s*=\s*(-?\d+)'
    )
    for coeff_str, var, const_str, rhs_str, sol_str in pat_lin.findall(full_text):
        # normalize pieces
        coeff = int(coeff_str) if coeff_str not in ("", "+", "-") else int(f"{coeff_str}1")
        const = int(const_str.replace(" ", ""))
        rhs   = int(rhs_str)
        provided = int(sol_str)

        # solve via sympy
        x = symbols(var)
        equation = Eq(coeff * x + const, rhs)
        true_sol = solve(equation, x)[0]
        is_correct = (true_sol == provided)

        orig = f"{coeff_str}{var}{const_str} = {rhs_str}, {var} = {sol_str}"
        assessments.append({
            "text":       orig,
            "type":       "linear",
            "equation":   f"{coeff_str}{var}{const_str} = {rhs_str}",
            "provided":   provided,
            "expected":   int(true_sol),
            "is_correct": is_correct
        })
        mark = "✅" if is_correct else "❌"
        lines.append(f"{orig} → {mark} {is_correct}")

    # ── 3) Generic arithmetic pattern ─────────────────────────────
    # matches any-length LHS expression = integer RHS
    pat_num = re.compile(r'([\d+\-*/()\s]+)\s*=\s*(-?\d+)')
    for m in pat_num.finditer(full_text):
        orig_eq = m.group(0).strip()
        lhs_str = m.group(1)
        c_str   = m.group(2)

        # skip if it was already handled by pat_lin
        if pat_lin.match(orig_eq):
            continue

        # evaluate LHS
        try:
            expected = eval(lhs_str)
        except Exception:
            continue

        # parse RHS
        c = int(c_str)
        # compare (allow float tolerance)
        if isinstance(expected, float):
            is_correct = abs(expected - c) < 1e-9
        else:
            is_correct = (expected == c)

        assessments.append({
            "text":       orig_eq,
            "type":       "arithmetic",
            "expression": lhs_str,
            "result":     c,
            "is_correct": is_correct
        })
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
    return assess_pdf_bytes(data)
