# SmartAssessor

## New Architecture (Firestore Queue)

Frontend → Backend (PC) → Firestore “jobs” → Colab worker (GPU LLM) → Firestore → Backend → Frontend.

The backend no longer runs a local LLM. Instead, it creates a job document in Firestore and waits for a Colab worker (running the GPU model) to process it and write back results.

### Setup

- Firebase: create a project, enable Firestore (Native), and create a Service Account (Admin SDK JSON). Keep the JSON private.
- Env vars (PC + Colab):
  - `FIREBASE_PROJECT_ID=smart-assessor-fdeaa`
  - `GOOGLE_APPLICATION_CREDENTIALS=<path to service-account.json>`
- Python deps (PC + Colab):
  - `pip install google-cloud-firestore google-auth`

### Backend (PC)

- Endpoints:
  - `POST /upload/assessment` — Accepts a PDF, extracts text, splits into chunks, enqueues a Firestore job, waits for result, and returns `{ evaluation: string }`.
  - `POST /grade-text` — Smoke test; sends a single text as one job.
- CORS is configurable via `ALLOWED_ORIGINS`.

Run locally:

```
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Colab Worker (GPU LLM)

- Notebook has two cells:
  1) Load model and define `_llm.chat(text)` returning a JSON-first line.
  2) Worker loop: polls Firestore for `status="queued"`, processes `chunks` or `student_text`, then writes back `status="done"` + `result.text`.

### Frontend

- Still calls the backend at `POST /upload/assessment` and renders `evaluation`.
- Configure API base via any of:
  - `VITE_API_BASE` (build time)
  - `window.API_BASE` (runtime; `frontend/public/config.js`)
  - `localStorage.setItem('API_BASE', 'http(s)://...')`

### Notes

- Keep Service Account JSON out of git (e.g., store under a private path and point `GOOGLE_APPLICATION_CREDENTIALS` there).
- Firestore rules should be open only during dev; lock down for production.
