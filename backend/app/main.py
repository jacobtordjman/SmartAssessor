# backend/app/main.py

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Text2TextGenerationPipeline
)

app = FastAPI()

# ─── CORS ────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Your Fine-Tuned Agent-1 Model & Tokenizer ─────────────────────────────
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training", "agent1_model")
)

# Load tokenizer & model from local directory only
tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, local_files_only=True)

# Create a pipeline for text2text-generation
device = 0 if torch.cuda.is_available() else -1
agent1 = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device
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

    # Run the pipeline
    outputs = agent1(prompt, max_length=64, num_return_sequences=1)
    return {"evaluation": outputs[0]["generated_text"]}
