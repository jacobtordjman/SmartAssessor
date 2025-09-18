import os
from io import BytesIO
from typing import Iterable

import PyPDF2
import unicodedata
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from .firebase_queue import create_job_and_wait


def _debug_preview(label: str, data: str | bytes, limit: int = 200) -> None:
    """Print a short preview for debugging without flooding the console."""
    if isinstance(data, bytes):
        snippet = data[:limit]
        print(f"[Debug] {label} (len={len(data)} bytes) -> {snippet!r}{'…' if len(data) > limit else ''}")
    else:
        snippet = data[:limit]
        print(f"[Debug] {label} (len={len(data)} chars) -> {snippet!r}{'…' if len(data) > limit else ''}")


def _debug_iterable(label: str, values: Iterable[str], limit: int = 3) -> None:
    preview = []
    for idx, val in enumerate(values, start=1):
        if idx > limit:
            preview.append("…")
            break
        preview.append(f"[{idx}] len={len(val)} -> {val[:120]!r}{'…' if len(val) > 120 else ''}")
    print(f"[Debug] {label}:\n  " + "\n  ".join(preview) if preview else f"[Debug] {label}: <empty>")


class GradeTextPayload(BaseModel):
    text: str


app = FastAPI()

# Configure CORS to work locally and on Pages
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() in ("", "*"):
    _allow_origins = ["*"]
    _allow_credentials = False  # star origin requires credentials disabled
else:
    _allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=_allow_credentials,
)
print(f"[Startup] CORS allow_origins={_allow_origins} allow_credentials={_allow_credentials}")


def extract_text_from_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(data))
    raw_txt = "".join(page.extract_text() or "" for page in reader.pages)
    return unicodedata.normalize("NFKC", raw_txt)


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    pieces: list[str] = []
    cur = ""
    for block in text.split("\n\n"):
        if not cur:
            cur = block
        elif len(cur) + len(block) + 2 <= max_chars:
            cur += "\n\n" + block
        else:
            pieces.append(cur.strip())
            cur = block
    if cur:
        pieces.append(cur.strip())
    return pieces


@app.post("/upload/assessment")
async def upload_assessment(file: UploadFile = File(...)):
    try:
        print("=== upload_assessment called ===")
        print(f"[Backend] Incoming file metadata: name={getattr(file, 'filename', 'unknown')} content_type={file.content_type}")
        if file.content_type != "application/pdf":
            raise HTTPException(400, "Only PDF files allowed.")

        data = await file.read()
        _debug_preview("Raw PDF data", data, limit=80)

        student_text = extract_text_from_pdf(data)
        _debug_preview("Extracted text", student_text, limit=400)

        chunks = chunk_text(student_text)
        print(f"[Backend] Split into {len(chunks)} chunk(s).")
        _debug_iterable("Chunk previews", chunks, limit=5)

        payload = {"chunks": chunks}
        print(f"[Backend] Dispatching job with payload keys={list(payload.keys())} total_chunk_chars={sum(len(c) for c in chunks)}")
        job = create_job_and_wait(payload, req_type="grade_pdf", timeout_s=180)

        if job.get("status") == "done":
            combined_feedback = ((job.get("result") or {}).get("text") or "")
            _debug_preview("Combined feedback", combined_feedback, limit=400)
            response = {"evaluation": combined_feedback}
            print(f"[Backend] Returning JSON: {response.keys()}")
            return response

        err = job.get("error", "timeout")
        print(f"[Backend] Worker returned non-success status: {job.get('status')} error={err}")
        raise HTTPException(502, f"Worker error or timeout: {err}")

    except Exception as e:
        print(f"[Backend] UPLOAD ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.post("/grade-text")
async def grade_text(p: GradeTextPayload):
    print("=== grade_text called ===")
    _debug_preview("Incoming text payload", p.text, limit=400)
    job = create_job_and_wait({"student_text": p.text}, req_type="grade_pdf", timeout_s=90)
    if job.get("status") == "done":
        result_text = (job.get("result") or {}).get("text", "")
        _debug_preview("grade_text feedback", result_text, limit=400)
        return {"evaluation": result_text}
    err = job.get("error", "timeout")
    print(f"[Backend] grade_text worker error: {err}")
    raise HTTPException(502, f"Worker error or timeout: {err}")