import os
import time
from pathlib import Path
from typing import Any, Dict

from google.cloud import firestore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
try:
    PROJECT_ROOT = BASE_DIR.parents[1]
except IndexError:
    PROJECT_ROOT = BASE_DIR.parent

if load_dotenv:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    else:
        load_dotenv()


def _resolve_credentials_path(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    search_roots = (BASE_DIR, BASE_DIR.parent, PROJECT_ROOT)
    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((BASE_DIR / path).resolve())


def _ensure_credentials_env() -> None:
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _resolve_credentials_path(raw)


def _summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        sample = value[0]
        preview = f"{sample[:80]!r}{'…' if isinstance(sample, str) and len(sample) > 80 else ''}" if isinstance(sample, str) else type(sample).__name__
        return f"list(len={len(value)}, sample={preview})"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes(len={len(value)})"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    return repr(value)


_db: firestore.Client | None = None


def get_firestore_client() -> firestore.Client:
    global _db
    if _db is None:
        _ensure_credentials_env()
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        _db = firestore.Client(project=project_id) if project_id else firestore.Client()
        print(f"[Firestore] Initialized client for project={project_id or '<auto>'}")
    return _db


def create_job_and_wait(payload: Dict[str, Any], req_type: str = "grade_pdf", timeout_s: int = 90) -> Dict[str, Any]:
    db = get_firestore_client()
    doc_ref = db.collection("jobs").document()

    payload_summary = {k: _summarize_value(v) for k, v in payload.items()}
    print(
        f"[Firestore] Creating job id={doc_ref.id} type={req_type} timeout={timeout_s}s payload={payload_summary}"
    )

    now = firestore.SERVER_TIMESTAMP
    doc_ref.set(
        {
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "request": {"type": req_type, "payload": payload},
        }
    )
    print(f"[Firestore] Job {doc_ref.id} set to queued")

    deadline = time.time() + timeout_s
    last_status = "queued"
    while time.time() < deadline:
        snap = doc_ref.get()
        d = snap.to_dict() or {}
        status = d.get("status")
        if status != last_status:
            print(f"[Firestore] Job {doc_ref.id} status transition: {last_status} -> {status}")
            last_status = status
        if status in ("done", "error"):
            print(f"[Firestore] Job {doc_ref.id} completed with status={status}")
            return d
        time.sleep(1.0)

    print(f"[Firestore] Job {doc_ref.id} timed out after {timeout_s}s")
    return {"status": "error", "error": "Timeout waiting for worker", "result": {"text": ""}}