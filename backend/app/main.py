from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2, unicodedata
from io import BytesIO
import os

from backend.app.llm_client import llm

app = FastAPI()

# Configure CORS to work locally and on Pages
# Use env var ALLOWED_ORIGINS=comma,separated,origins for strict mode.
# Default to wildcard when not set (and disable credentials in that case).
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
    pieces, cur = [], ""
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
        if file.content_type != "application/pdf":
            raise HTTPException(400, "Only PDF files allowed.")
        data = await file.read()
        try:
            print(f"[Backend] Received file: name={getattr(file, 'filename', 'unknown')} size={len(data)} bytes content_type={file.content_type}")
        except Exception:
            pass

        student_text = extract_text_from_pdf(data)
        print(f"[Backend] Extracted text (first 200 chars):\n{student_text[:200]!r}")

        chunks = chunk_text(student_text)
        print(f"[Backend] Split into {len(chunks)} chunk(s).")

        feedbacks = []
        for idx, piece in enumerate(chunks, start=1):
            print(f"[Backend] Grading chunk {idx}/{len(chunks)}:")
            print(piece[:200])
            try:
                fb = llm.chat(piece)
                print(f"[Backend] Feedback for chunk {idx}:\n{fb[:200]!r}")
                feedbacks.append(fb)
            except Exception as chat_error:
                print(f"[Backend] Error in llm.chat for chunk {idx}: {chat_error.__class__.__name__}: {str(chat_error)}")
                feedbacks.append(f"Error processing chunk {idx}: {str(chat_error)}")

        combined_feedback = "\n\n".join(feedbacks)
        print(f"[Backend] Combined feedback (first 200 chars):\n{combined_feedback[:200]!r}")

        response = {"evaluation": combined_feedback}
        print(f"[Backend] Returning JSON:\n{response!r}")
        return response
    
    except Exception as e:
        print(f"[Backend] UPLOAD ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")
