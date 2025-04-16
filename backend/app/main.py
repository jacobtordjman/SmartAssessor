# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from tempfile import NamedTemporaryFile
from PyPDF2 import PdfReader

app = FastAPI()

# Specify the allowed origins for cross-origin requests
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

@app.post("/upload/assessment")
async def upload_assessment(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        try:
            reader = PdfReader(temp_file.name)
            extracted_text = " ".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            return {"error": f"Failed to extract text: {e}"}
    return {"extracted_text": extracted_text}
