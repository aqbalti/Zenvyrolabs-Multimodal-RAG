"""
main.py
=======
FastAPI application for the Zenvyrolabs Multimodal RAG System.

Endpoints
---------
POST /api/upload   – Ingest a PDF or image document
POST /api/chat     – Ask a question (with conversation history)
GET  /api/health   – Health check
"""

import logging
import os
import shutil

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from rag_engine import process_and_store_document, query_rag_system

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zenvyrolabs Multimodal RAG API",
    description="Production-ready RAG system supporting PDFs, novels, and manga.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_BOOK_TYPES = {"coding", "novel", "manga"}
MAX_FILE_MB = 100


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    """A single turn in the conversation history."""
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for /api/chat."""
    message: str
    book_type: str = "coding"
    chat_history: list[ChatTurn] = []

    @field_validator("book_type")
    @classmethod
    def validate_book_type(cls, v: str) -> str:
        if v not in ALLOWED_BOOK_TYPES:
            raise ValueError(f"book_type must be one of {ALLOWED_BOOK_TYPES}")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


class UploadResponse(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Simple liveness probe."""
    return HealthResponse(status="ok", version=app.version)


@app.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    book_type: str = Form("coding"),
) -> UploadResponse:
    """
    Upload a PDF document and ingest it into the vector database.

    - **file**: PDF file (max 100 MB)
    - **book_type**: 'coding', 'novel', or 'manga'
    """
    if book_type not in ALLOWED_BOOK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"book_type must be one of {ALLOWED_BOOK_TYPES}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Guard against excessively large uploads
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_MB} MB.",
        )

    # Persist to temp directory
    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(TEMP_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    logger.info("Received upload: %s (%.1f MB, type=%s)", safe_name, size_mb, book_type)

    try:
        result_msg = process_and_store_document(file_path, book_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during document processing")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        ) from exc

    return UploadResponse(message=result_msg)


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Ask a question about the uploaded documents.

    Accepts an optional **chat_history** array to maintain conversation memory
    across turns (Bug 3 fix).
    """
    logger.info(
        "Chat request – mode=%s, history_turns=%d, query=%s",
        request.book_type,
        len(request.chat_history),
        request.message[:80],
    )

    try:
        history = [t.model_dump() for t in request.chat_history]
        answer = query_rag_system(
            user_message=request.message,
            book_type=request.book_type,
            chat_history=history,
        )
    except Exception as exc:
        logger.exception("Unexpected error during chat query")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {exc}",
        ) from exc

    return ChatResponse(answer=answer)


# ---------------------------------------------------------------------------
# Serve frontend static files (same origin as API)
# ---------------------------------------------------------------------------
_frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_path):
    app.mount("/", StaticFiles(directory=_frontend_path, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
