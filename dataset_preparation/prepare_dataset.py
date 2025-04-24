#!/usr/bin/env python3
"""
prepare_dataset.py

1. Extracts text from each PDF in ./pdfs
2. Saves raw text into ./extracted_texts
3. Parses “Question X:” / “Answer X:” pairs
4. Emits a single dataset.csv of input/output training pairs
"""

import os
import re
import pdfplumber
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Base dir is this script's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER       = os.path.join(BASE_DIR, 'pdfs')
EXTRACT_FOLDER   = os.path.join(BASE_DIR, 'extracted_texts')
OUTPUT_CSV_PATH  = os.path.join(BASE_DIR, 'dataset.csv')

# Regex patterns—adjust if your PDFs use different labels
QUESTION_PATTERN = re.compile(r'(Question\s*\d+:.*?)(?=Answer\s*\d+:|$)', re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN   = re.compile(r'(Answer\s*\d+:.*?)(?=Question\s*\d+:|$)', re.DOTALL | re.IGNORECASE)

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create extraction folder if it doesn’t exist."""
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)

def extract_text_from_pdf(path):
    """Return the full textual content of a PDF file."""
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)

def parse_questions_answers(full_text):
    """
    Find pairs of (question, answer) in the extracted text.
    Returns a list of dicts: [{"question": str, "answer": str}, ...]
    """
    # find all question/answer blocks
    questions = QUESTION_PATTERN.findall(full_text)
    answers   = ANSWER_PATTERN.findall(full_text)

    # clean up whitespace/newlines
    questions = [q.strip().replace('\n', ' ') for q in questions]
    answers   = [a.strip().replace('\n', ' ') for a in answers]

    # pair them one-to-one (up to min length)
    pairs = []
    for q, a in zip(questions, answers):
        pairs.append({"question": q, "answer": a})
    return pairs

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    training_rows = []

    for fname in sorted(os.listdir(PDF_FOLDER)):
        if not fname.lower().endswith('.pdf'):
            continue

        pdf_path = os.path.join(PDF_FOLDER, fname)
        print(f"Processing {fname}...")

        # 1) Extract raw text
        text = extract_text_from_pdf(pdf_path)

        # 2) Save raw text for inspection
        txt_name = fname.replace('.pdf', '.txt')
        txt_path = os.path.join(EXTRACT_FOLDER, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # 3) Parse Q&A pairs
        pairs = parse_questions_answers(text)
        if not pairs:
            print(f"  ⚠️ No Q&A pairs found in {fname}. Check your PDF labels.")
            continue
        print(f"  → Found {len(pairs)} pairs.")

        # 4) Build training rows
        for pair in pairs:
            input_text  = f"Question: {pair['question']} Student Answer: {pair['answer']}"
            output_text = "Correct."  # modify manually or expand later
            training_rows.append({"input": input_text, "output": output_text})

    # 5) Dump to CSV
    df = pd.DataFrame(training_rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"\n✅ Dataset complete: {len(df)} rows saved to {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()
