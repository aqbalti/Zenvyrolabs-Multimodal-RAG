# Zenvyrolabs – Multimodal RAG AI System

A production-ready Retrieval-Augmented Generation (RAG) assistant that reads coding textbooks, novels, and manga/comics and answers precise questions about them.

---

## Features

| # | Feature | Status |
|---|---------|--------|
| 1 | **Vector Blindness Fix** – physical page/line metadata injected into every chunk | ✅ |
| 2 | **Global Summarisation** – dynamic k-expansion for whole-book queries | ✅ |
| 3 | **Conversation Memory** – full chat history sent with every request | ✅ |
| 4 | **Manga OCR** – EasyOCR + OpenCV preprocessing pipeline | ✅ |
| 5 | **Rate Limiting** – tenacity exponential backoff + batch embedding + response cache | ✅ |

---

## Project Structure

```
zenvyrolabs/
├── backend/
│   ├── main.py            ← FastAPI application & endpoints
│   ├── rag_engine.py      ← Core RAG logic (all bug fixes live here)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js             ← Chat history (Bug 3 fix) lives here
│   └── assets/
│       └── LOGO.png
├── chroma_db/             ← Auto-created; persistent vector store
├── temp/                  ← Temporary upload staging area
├── .env
└── README.md
```

---

## Installation & Setup

### 1. Clone / unzip

```bash
cd zenvyrolabs
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure the LLM

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (or OPENAI_API_KEY)
```

### 5. Start the server

```bash
cd backend
python main.py
```

### 6. Open in browser

```
http://localhost:8000
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                │
│                                                                          │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Sidebar (Upload Panel) │    │       Chat Window                │   │
│  │  ─ Mode selector         │    │  ─ Message history (per mode)    │   │
│  │  ─ PDF drag-and-drop     │    │  ─ Markdown + code highlighting  │   │
│  │  ─ Upload status         │    │  ─ Loading spinner               │   │
│  └────────────┬─────────────┘    └──────────────┬───────────────────┘   │
│               │ POST /api/upload                 │ POST /api/chat        │
│               │ (FormData + book_type)           │ (JSON + chat_history) │
└───────────────┼──────────────────────────────────┼───────────────────────┘
                │                                  │
                ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FastAPI  (main.py)                               │
│                                                                          │
│   /api/upload ──────────────────────────────────┐                       │
│   /api/chat   ──────────────────────────────┐   │                       │
│   /api/health                               │   │                       │
└─────────────────────────────────────────────┼───┼───────────────────────┘
                                              │   │
                         ┌────────────────────┘   │
                         │                        │
                         ▼                        ▼
┌────────────────────────────────────┐  ┌─────────────────────────────────┐
│        RAG Engine (rag_engine.py)  │  │     Document Ingestion          │
│                                    │  │                                 │
│  1. Query type detection           │  │  book_type == "manga"?          │
│     (global vs. precise)           │  │  ┌──────────────────────────┐  │
│                                    │  │  │ OCR Pipeline             │  │
│  2. Hybrid Retrieval               │  │  │ fitz → rasterise page    │  │
│     Chroma similarity +            │  │  │ OpenCV preprocess        │  │
│     metadata filter (book_type,    │  │  │ EasyOCR readtext()       │  │
│     page, paragraph)               │  │  └──────────────────────────┘  │
│                                    │  │           ↓                     │
│  3. Citation building              │  │  book_type in {coding,novel}?   │
│     [Book | Page | Para] tags      │  │  ┌──────────────────────────┐  │
│                                    │  │  │ PDF Text Pipeline        │  │
│  4. History injection              │  │  │ PyMuPDFLoader            │  │
│     Last N turns in prompt         │  │  │ RecursiveTextSplitter    │  │
│                                    │  │  │ Inject [Source: Page X]  │  │
│  5. LLM call (with retry)          │  │  └──────────────────────────┘  │
│     tenacity exponential backoff   │  │           ↓                     │
│                                    │  │  Batch embed (EMBED_BATCH_SIZE) │
│  6. Response cache                 │  │  HuggingFace all-MiniLM-L6-v2  │
│     In-memory dict for identical   │  │           ↓                     │
│     queries (no history)           │  │  ChromaDB persist_directory     │
└────────────────────────────────────┘  └─────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  OpenRouter / OpenAI Compatible LLM                      │
│           (configurable via .env – GPT-4o-mini, Llama3, etc.)           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a PDF; form fields: `file`, `book_type` |
| `POST` | `/api/chat` | JSON body: `{ message, book_type, chat_history }` |
| `GET`  | `/api/health` | Liveness probe – returns `{ status, version }` |

---

## Interview Preparation

### Why ChromaDB?
ChromaDB is a lightweight, embedded vector database that requires zero infrastructure setup. It persists vectors to disk automatically, supports metadata filtering natively, and integrates seamlessly with LangChain. For a demo/intern project it avoids the operational complexity of Pinecone or Weaviate while demonstrating the same RAG concepts.

### Why Sentence Transformers (all-MiniLM-L6-v2)?
It is a free, CPU-friendly model that runs locally with no API key or cost. It produces 384-dimensional embeddings that are fast to compute and accurate enough for document retrieval. Choosing a local model also eliminates a rate-limit dependency during ingestion, which matters when processing large textbooks.

### Why metadata?
Pure semantic search is "blind" to structural information like page numbers, paragraph positions, and document names. By storing and injecting this metadata, we can answer questions such as *"What does page 34 line 5 say?"* with exact precision, and we can filter search results to a specific book or chapter without scanning the entire vector store.

### Why OCR?
Manga and comic books are image-based PDFs. PyMuPDF's text layer extractor only works on documents with embedded text. For manga, every panel is a rasterised image. OCR (EasyOCR) reads pixels and converts them to text. Without OCR the system would return empty chunks and the AI would have nothing to answer from.

### How does RAG work?
1. **Ingestion**: The document is split into overlapping text chunks. Each chunk is converted to a dense embedding vector and stored in ChromaDB alongside its metadata.
2. **Retrieval**: When a user asks a question, the question is also embedded. ChromaDB finds the `k` most similar chunk vectors using cosine similarity.
3. **Generation**: The retrieved chunks are injected into a prompt as "context". The LLM (GPT-4o-mini etc.) reads only the provided context and generates a grounded answer. It cannot hallucinate facts not in the context.

### How does Hybrid Retrieval work?
Our retrieval combines two strategies:
- **Semantic search**: finds chunks whose *meaning* is closest to the query.
- **Metadata filtering**: restricts results to a specific `book_type` (or page range), acting like a SQL WHERE clause on top of the vector search.
This is "hybrid" because it is not just keyword matching (sparse) or just semantic matching (dense) — it is semantic search scoped by structured metadata.

### What is vector blindness?
When page numbers live only in chunk *metadata* and not in the chunk *text*, the embedding model never "sees" them. A query like *"What is on page 34?"* produces an embedding that has no similarity to any chunk, because no chunk's text contains "page 34". The fix is to prepend a `[Source: Book | Page 34 | Para 2]` header **directly into the chunk text** before embedding, so semantic search can match on it.

### How is conversation memory implemented?
The frontend maintains a `chatHistory` array per mode. Every time the user sends a message, the full history (all prior user and assistant turns) is serialised to JSON and sent to the backend inside the request body. The backend injects the last `MAX_HISTORY_PAIRS` turns into the LLM prompt as a "Conversation History" section. The LLM can then resolve follow-up references ("what about the previous chapter?") because it sees the prior context.

### Why map-reduce summarisation?
A 500-page textbook produces hundreds of chunks that exceed any LLM's context window. Map-reduce handles this by summarising each chunk (or group of chunks) independently ("map"), then combining those summaries into a final summary ("reduce"). Our implementation uses a large `k` (60 chunks) for global queries, which approximates the map step within a single call. A full map-reduce pipeline would chain multiple LLM calls.

### How is rate limiting handled?
Three complementary strategies:
1. **Exponential backoff** (`tenacity`): If the LLM API returns a `RateLimitError`, the request is retried automatically after 4 s, 8 s, 16 s… up to 6 attempts.
2. **Batch embedding with sleep**: Documents are embedded in batches of 64 chunks with a 0.5 s pause between batches, preventing burst requests to any external embedding API.
3. **Response cache**: Identical queries (no conversation history) hit an in-memory dict instead of calling the LLM again, reducing redundant API calls during a demo where the same question may be asked multiple times.

---

## Future Improvements

- **Persistent conversation store** (SQLite or Redis) so history survives server restarts
- **Full map-reduce summarisation** using LangChain's `MapReduceDocumentsChain`
- **Re-ranking** with a cross-encoder model (e.g. `ms-marco-MiniLM`) to improve retrieval precision
- **Authentication** and per-user document namespacing in ChromaDB
- **Async ingestion queue** (Celery + Redis) for large file uploads
- **Multi-language OCR** (EasyOCR supports 80+ languages)
- **Docker Compose** for one-command deployment
