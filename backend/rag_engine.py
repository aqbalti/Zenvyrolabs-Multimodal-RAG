"""
rag_engine.py
=============
Core RAG engine for Zenvyrolabs Multimodal RAG System.

Solves:
  Bug 1 – Vector Blindness      : physical page/line metadata injected into chunk text
  Bug 2 – Myopic Context        : global-query detection with full-book retrieval
  Bug 3 – Amnesia               : conversation history injected into prompt
  Bug 4 – Manga OCR             : EasyOCR + OpenCV pipeline for image-heavy PDFs

Rate-limiting strategy:
  • tenacity RetryError with exponential backoff on every LLM call
  • Chunk batching during ingestion (EMBED_BATCH_SIZE)
  • In-process response cache (lru_cache on query key)
"""

import os
import re
import logging
import time
import functools
from typing import Optional

import fitz                      # PyMuPDF
import cv2
import numpy as np
import easyocr
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 64          # how many chunks to embed per batch
DEFAULT_K = 8                  # normal retrieval top-k
GLOBAL_K = 60                  # top-k for global / summarise queries
MAX_HISTORY_PAIRS = 6          # how many user/assistant turns to keep in prompt

# Keywords that signal the user wants a global answer (Bug 2 fix)
GLOBAL_KEYWORDS = {
    "summarize", "summary", "summarise", "entire book", "whole book",
    "all chapters", "overview", "throughout", "across the book",
    "best problem", "main theme", "overall", "in general",
    "what does the book", "tell me about the book",
}

# ---------------------------------------------------------------------------
# Singletons – initialised once at import time
# ---------------------------------------------------------------------------
logger.info("Loading embedding model (all-MiniLM-L6-v2)…")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

logger.info("Initialising EasyOCR reader (en)…")
_ocr_reader: Optional[easyocr.Reader] = None   # lazy init to save startup time

def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Lazy-loading EasyOCR model…")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _build_llm() -> ChatOpenAI:
    """
    Build an OpenAI-compatible LLM client using OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Check your .env file."
        )

    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    model = os.getenv(
        "LLM_MODEL",
        "openai/gpt-4o-mini"
    )

    return ChatOpenAI(
        api_key=api_key.strip(),
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )

llm = _build_llm()

# ---------------------------------------------------------------------------
# Simple in-memory response cache (Bug 4 / rate-limit mitigation)
# ---------------------------------------------------------------------------
_response_cache: dict[str, str] = {}

def _cache_key(message: str, book_type: str) -> str:
    return f"{book_type}::{message.strip().lower()}"


# ---------------------------------------------------------------------------
# Bug 4 – Manga OCR helpers
# ---------------------------------------------------------------------------

def _preprocess_image_for_ocr(img_array: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw image for better OCR accuracy:
      1. Convert to greyscale
      2. Apply CLAHE for contrast enhancement
      3. Denoise with fastNlMeansDenoising
      4. Adaptive threshold → binary image (ideal for comic text bubbles)
    """
    grey = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return binary


def _extract_manga_text(file_path: str) -> list[Document]:
    """
    OCR pipeline for Manga / Comic PDFs.
    Each PDF page is rasterised to an image, preprocessed, then fed to EasyOCR.
    Returns a list of LangChain Documents with rich metadata.
    """
    reader = _get_ocr_reader()
    doc = fitz.open(file_path)
    documents: list[Document] = []
    book_name = os.path.splitext(os.path.basename(file_path))[0]

    for page_num, page in enumerate(doc, start=1):
        logger.info("OCR processing manga page %d / %d", page_num, len(doc))

        # Rasterise at 2× resolution for better OCR accuracy
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # Convert to numpy array for OpenCV processing
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        processed = _preprocess_image_for_ocr(img_cv)

        # EasyOCR expects a numpy array
        results = reader.readtext(processed, detail=1, paragraph=False)

        if not results:
            logger.debug("No text found on manga page %d", page_num)
            continue

        # Build a structured dialogue string with line numbers
        lines: list[str] = []
        for line_num, (bbox, text, confidence) in enumerate(results, start=1):
            if confidence >= 0.3 and text.strip():
                lines.append(f"[Line {line_num}] {text.strip()}")

        if not lines:
            continue

        page_text = "\n".join(lines)
        # ── Bug 1 fix for Manga: inject physical page reference ──────────
        page_header = f"[Source: {book_name} | Manga Page {page_num}]\n"
        full_content = page_header + page_text

        documents.append(
            Document(
                page_content=full_content,
                metadata={
                    "source": book_name,
                    "book_type": "manga",
                    "page": page_num,
                    "total_pages": len(doc),
                },
            )
        )

    doc.close()
    logger.info("Manga OCR complete – extracted text from %d pages", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Bug 1 – Vector Blindness fix: PDF text extraction with injected metadata
# ---------------------------------------------------------------------------

def _extract_pdf_text_with_metadata(file_path: str, book_type: str) -> list[Document]:
    """
    Load a PDF and inject physical location metadata into every chunk's
    page_content so that semantic search can locate exact pages and lines.

    Metadata injected per chunk:
      [Source: <BookName> | Page <N> | Para <P> | Lines <start>–<end>]
    """
    book_name = os.path.splitext(os.path.basename(file_path))[0]
    raw_docs = PyMuPDFLoader(file_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    enriched: list[Document] = []
    para_counter: dict[int, int] = {}   # page_num → running paragraph index

    for raw_doc in raw_docs:
        page_num: int = raw_doc.metadata.get("page", 0) + 1  # PyMuPDF is 0-indexed
        para_counter.setdefault(page_num, 0)

        # Split each page into sub-chunks
        page_chunks = splitter.split_text(raw_doc.page_content)

        for chunk_text in page_chunks:
            para_counter[page_num] += 1
            para_num = para_counter[page_num]

            # Count lines within this chunk
            lines_in_chunk = [l for l in chunk_text.splitlines() if l.strip()]
            line_count = len(lines_in_chunk)

            # ── Key Bug 1 fix: prepend location header into the chunk text ──
            location_header = (
                f"[Source: {book_name} | Page {page_num} | "
                f"Para {para_num} | Lines 1–{line_count}]\n"
            )
            enriched_text = location_header + chunk_text

            enriched.append(
                Document(
                    page_content=enriched_text,
                    metadata={
                        "source": book_name,
                        "book_type": book_type,
                        "page": page_num,
                        "paragraph": para_num,
                        "line_count": line_count,
                    },
                )
            )

    logger.info(
        "Extracted %d enriched chunks from '%s' (%s)",
        len(enriched), book_name, book_type,
    )
    return enriched


# ---------------------------------------------------------------------------
# Embedding with batching & retry  (rate-limit / Bug 4 mitigation)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((RateLimitError, Exception)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def _embed_batch_with_retry(batch: list[Document], db_dir: str) -> Chroma:
    """Embed a single batch of documents into ChromaDB with automatic retry."""
    return Chroma.from_documents(
        documents=batch,
        embedding=embeddings,
        persist_directory=db_dir,
    )


def _batch_embed(documents: list[Document], db_dir: str) -> None:
    """
    Embed documents in batches of EMBED_BATCH_SIZE.
    Between batches a short sleep prevents hammering any rate-limited API
    (though HuggingFace embeddings are local, this guard is good practice).
    """
    total = len(documents)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d chunks…",
            start + 1, min(start + EMBED_BATCH_SIZE, total), total,
        )
        _embed_batch_with_retry(batch, db_dir)
        if start + EMBED_BATCH_SIZE < total:
            time.sleep(0.5)   # small pause between batches


# ---------------------------------------------------------------------------
# Public API: process_and_store_document
# ---------------------------------------------------------------------------

def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Ingest a document into ChromaDB.

    Parameters
    ----------
    file_path : str
        Absolute path to the uploaded file (PDF expected).
    book_type : str
        One of 'coding', 'novel', 'manga'.

    Returns
    -------
    str
        Human-readable status message.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")

    logger.info("Processing document: %s (type=%s)", file_path, book_type)

    # ── Bug 4: route manga through OCR pipeline ──────────────────────────
    if book_type == "manga":
        documents = _extract_manga_text(file_path)
    else:
        documents = _extract_pdf_text_with_metadata(file_path, book_type)

    if not documents:
        raise ValueError("No text could be extracted from the uploaded document.")

    _batch_embed(documents, DB_DIR)

    logger.info("Successfully stored %d chunks into ChromaDB.", len(documents))
    return f"Successfully processed {len(documents)} chunks from '{os.path.basename(file_path)}'."


# ---------------------------------------------------------------------------
# Bug 2 – Global query detection
# ---------------------------------------------------------------------------

def _is_global_query(message: str) -> bool:
    """Return True when the user is asking a broad / whole-book question."""
    lower = message.lower()
    return any(kw in lower for kw in GLOBAL_KEYWORDS)

def _extract_page_number(message: str) -> Optional[int]:
    """
    Extract an explicit page number from queries such as:
    'tell me about page 16'
    'what is on page 25?'
    'explain page 10'
    """
    match = re.search(r"\bpage\s+(\d+)\b", message.lower())
    return int(match.group(1)) if match else None

# ---------------------------------------------------------------------------
# Bug 2 + 3: LLM call with retry and exponential backoff
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(6),
)
def _invoke_llm_with_retry(chain, inputs: dict) -> str:
    """Invoke a LangChain chain with automatic exponential-backoff retry."""
    return chain.invoke(inputs)


# ---------------------------------------------------------------------------
# Public API: query_rag_system
# ---------------------------------------------------------------------------

def query_rag_system(
    user_message: str,
    book_type: str = "coding",
    chat_history: Optional[list[dict]] = None,
) -> str:
    """
    Answer a user question using hybrid RAG.

    Parameters
    ----------
    user_message : str
        The current user question.
    book_type : str
        Filter retrieval to this book type.
    chat_history : list[dict], optional
        Previous turns: [{"role": "user"|"assistant", "content": "…"}, …]

    Returns
    -------
    str
        The LLM's answer, with source citations when available.
    """
    if not user_message.strip():
        return "Please enter a question."

    # ── In-memory cache check ─────────────────────────────────────────────
    cache_k = _cache_key(user_message, book_type)
    # Only cache non-conversational queries (no history context)
    if not chat_history and cache_k in _response_cache:
        logger.info("Cache hit for query: %s", user_message[:60])
        return _response_cache[cache_k]

    # ── Load vector DB ────────────────────────────────────────────────────
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # ── Bug 2: dynamic k based on query type ─────────────────────────────
    # Page-specific or normal retrieval
    page_number = _extract_page_number(user_message)

    if page_number is not None:
        logger.info(
            "Page-specific retrieval: page=%d, book_type=%s",
            page_number,
            book_type,
        )

        retrieved_docs = vector_db.similarity_search(
            user_message,
            k=DEFAULT_K,
            filter={
                "$and": [
                    {"book_type": {"$eq": book_type}},
                    {"page": {"$eq": page_number}},
                ]
            },
        )

    else:
        k = GLOBAL_K if _is_global_query(user_message) else DEFAULT_K

        logger.info(
            "Retrieval k=%d (global=%s) for query: %s",
            k,
            _is_global_query(user_message),
            user_message[:80],
        )

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"book_type": book_type},
            },
        )

        retrieved_docs = retriever.invoke(user_message)

    # Build context string with explicit citations
    context_parts: list[str] = []
    for doc in retrieved_docs:
        meta = doc.metadata
        citation = (
            f"[Book: {meta.get('source', 'Unknown')} | "
            f"Page: {meta.get('page', '?')} | "
            f"Para: {meta.get('paragraph', '?')}]"
        )
        context_parts.append(f"{citation}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Bug 3: format conversation history ───────────────────────────────
    history_text = ""
    if chat_history:
        recent = chat_history[-(MAX_HISTORY_PAIRS * 2):]   # keep last N pairs
        lines: list[str] = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '').strip()}")
        history_text = "\n".join(lines)

    # ── Build prompt ──────────────────────────────────────────────────────
    system_prompt = """You are an expert Interactive Study Tutor powered by a RAG system.

Rules:
1. Answer ONLY based on the context provided below. Never hallucinate.
2. If the context does not contain enough information, say so clearly.
3. When answering page-specific questions, cite the exact [Source] tag from the context.
4. For summaries, synthesise all provided context into a coherent, structured answer.
5. If the user is asking a follow-up (based on conversation history), resolve pronouns and
   references before answering.

Conversation History (for context):
{history}

Retrieved Context (your knowledge base):
{context}

User Question: {question}

Answer (be precise, structured, and cite sources):"""

    prompt = ChatPromptTemplate.from_template(system_prompt)

    chain = prompt | llm | StrOutputParser()

    try:
        answer = _invoke_llm_with_retry(
            chain,
            {"history": history_text, "context": context, "question": user_message},
        )
    except Exception as exc:
        logger.error("LLM call failed after retries: %s", exc)
        return (
            "⚠️ The AI service is temporarily unavailable (rate limit or network error). "
            "Please try again in a few seconds."
        )

    # Cache only deterministic (no history) answers
    if not chat_history:
        _response_cache[cache_k] = answer

    return answer
