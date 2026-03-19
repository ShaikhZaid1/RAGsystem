# 🔐 CyberSec RAG — Retrieval-Augmented Generation Learning Platform

A production-ready RAG system built for cybersecurity education. Upload your notes, PDFs, and syllabi — then ask questions, generate viva questions, and get exam-focused answers grounded strictly in your own materials.

Powered by **Claude AI** (Anthropic), **ChromaDB**, and **SentenceTransformers**.

---

## ✨ Features

- **Multi-format ingestion** — PDF, plain text, Markdown, and live web pages
- **Three chunking strategies** — fixed-size, sliding window, semantic
- **Local embeddings** — `all-MiniLM-L6-v2` runs entirely on your machine (no API cost)
- **Hybrid search** — vector similarity + keyword overlap for best recall
- **Cross-encoder re-ranking** — dramatically improves result precision
- **Four answer modes** — Q&A, summarize, viva questions, multi-hop reasoning
- **Conversation memory** — sliding window keeps context across turns
- **Anti-hallucination prompting** — answers grounded strictly in your documents
- **Source citation** — every answer references which document it came from
- **Streaming chat** — real-time token-by-token responses via WebSocket
- **Clean chat UI** — dark-themed, single-page web interface

---

## 📁 Project Structure

```
rag_system/
├── main.py                        # FastAPI app — all HTTP + WebSocket endpoints
├── config.py                      # Central settings (loaded from .env)
├── requirements.txt
│
├── ingestion/
│   ├── pdf_parser.py              # PyMuPDF (primary) + pdfplumber (fallback)
│   ├── web_scraper.py             # BeautifulSoup page scraper
│   └── preprocessor.py           # All 3 chunking strategies + text cleaning
│
├── embeddings/
│   └── embedder.py                # SentenceTransformers + OpenAI, with disk cache
│
├── vectorstore/
│   └── chroma_store.py            # ChromaDB (primary) + FAISS (alternative)
│
├── retrieval/
│   └── retriever.py               # Query rewriting, hybrid search, re-ranking
│
├── llm/
│   └── claude_client.py           # Claude API with 4 prompt templates
│
├── memory/
│   └── conversation_memory.py     # Sliding window session memory
│
├── pipeline/
│   └── rag_pipeline.py            # Orchestrator — ties everything together
│
└── static/
    └── index.html                 # Chat UI (vanilla HTML/CSS/JS, dark theme)
```

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone https://github.com/your-username/rag_system.git
cd rag_system
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** if you see `ERROR: Could not find a version...` for `faiss-cpu`, run:
> ```bash
> pip install faiss-cpu --index-url https://pypi.org/simple/
> ```

### 4. Create your `.env` file

```bash
# Copy the template
cp .env.example .env
```

Then edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional — only needed if using OpenAI embeddings
OPENAI_API_KEY=sk-your-openai-key

# Optional — for persistent session memory across restarts
REDIS_URL=redis://localhost:6379
```

Get your Anthropic API key at [console.anthropic.com](https://console.anthropic.com).

### 5. Fix the module path (Windows only)

Add these lines to the very top of `main.py`, before any other imports:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### 6. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 🖥️ Using the Chat UI

1. **Upload documents** — click the upload area in the sidebar or drag and drop PDF/TXT files
2. **Ingest a URL** — paste any web page URL and click "Ingest URL"
3. **Choose a mode** from the dropdown:
   - 📚 **Q&A** — direct answers to questions
   - 📝 **Summarize** — converts a topic into structured study notes
   - 🎯 **Viva** — generates exam questions with model answers
   - 🔗 **Multi-hop** — step-by-step reasoning for complex questions
4. **Ask your question** — type or click a hint chip, press Enter or click ➤

---

## 🐍 Using the Pipeline Directly (Python)

```python
from pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Ingest documents
rag.ingest("notes/network_security.pdf", chunk_strategy="semantic")
rag.ingest("notes/cryptography.pdf")
rag.ingest("https://owasp.org/www-project-top-ten/")

# Ask a question
result = rag.query("Explain Diffie-Hellman key exchange with a step-by-step example")
print(result["answer"])
print("Sources:", result["sources"])
print("Confidence:", result["confidence"])

# Generate viva questions
result = rag.query("OSINT tools and techniques", mode="viva")
print(result["answer"])

# Summarize a topic
result = rag.query("AES encryption", mode="summarize")

# Multi-turn conversation (pass the same session_id)
r1 = rag.query("What is AES?", session_id="student_001")
r2 = rag.query("How does it compare to DES?", session_id="student_001")  # remembers context
r3 = rag.query("Which one should I use today?", session_id="student_001")

# Check what's been ingested
print(rag.get_stats())
```

---

## 🌐 API Reference

All endpoints are served at `http://localhost:8000`. Interactive docs available at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/upload` | Upload a PDF or text file |
| `POST` | `/ingest-url` | Ingest a web page by URL |
| `POST` | `/query` | Ask a question |
| `GET` | `/search?q=...` | Raw similarity search (debug) |
| `GET` | `/stats` | System stats (chunk count, models) |
| `GET` | `/sources` | List all ingested documents |
| `DELETE` | `/source?source=...` | Remove a document |
| `DELETE` | `/session/{id}` | Clear a conversation session |
| `WS` | `/ws/chat` | Streaming chat (WebSocket) |

### POST `/query` — request body

```json
{
  "question": "Explain SQL injection with an example",
  "session_id": "student_001",
  "mode": "qa",
  "top_k": 5,
  "source_filter": "network_security.pdf"
}
```

### POST `/query` — response

```json
{
  "answer": "SQL injection is an attack where...",
  "sources": ["network_security.pdf"],
  "confidence": "high",
  "session_id": "student_001",
  "chunks_used": 3,
  "input_tokens": 1842,
  "output_tokens": 312
}
```

---

## ⚙️ Configuration

All settings live in `config.py` and are overridable via `.env`.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model version |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `5` | Candidates retrieved from vector DB |
| `RERANK_TOP_N` | `3` | Kept after cross-encoder re-ranking |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score to include |
| `HYBRID_ALPHA` | `0.5` | 0 = keyword only, 1 = vector only |
| `MAX_CONVERSATION_TURNS` | `10` | Turns kept in session memory |
| `CACHE_EMBEDDINGS` | `True` | Disk-cache computed embeddings |

---

## 🧠 Chunking Strategy Guide

Choose your strategy when calling `ingest()`:

```python
rag.ingest("file.pdf", chunk_strategy="sliding_window")  # default
rag.ingest("file.pdf", chunk_strategy="semantic")        # highest quality
rag.ingest("file.pdf", chunk_strategy="fixed")           # fastest
```

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `fixed` | ⚡ Fast | ★★☆ | Short notes, FAQs, homogeneous text |
| `sliding_window` | ⚡ Fast | ★★★ | General purpose — recommended default |
| `semantic` | 🐢 Slow | ★★★★ | Textbooks, research papers, complex PDFs |

---

## 🔍 Embedding Model Options

Swap the model in `config.py` or `.env`:

| Model | Dim | Speed | Quality | Cost |
|-------|-----|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ★★★ | Free (local) |
| `BAAI/bge-large-en-v1.5` | 1024 | ⚡⚡ | ★★★★ | Free (local) |
| `text-embedding-3-small` | 1536 | ⚡⚡ | ★★★★ | OpenAI API |
| `text-embedding-3-large` | 3072 | ⚡ | ★★★★★ | OpenAI API |

> ⚠️ Never mix models — if you change the embedding model, you must re-ingest all documents. The vector dimensions will be incompatible.

---

## 🗄️ Vector Store Options

| Store | Best For | Persistence | Metadata Filtering |
|-------|----------|-------------|-------------------|
| **ChromaDB** (default) | Local dev, up to ~500k chunks | ✅ | ✅ |
| **FAISS** | High-speed in-memory search | Manual (serialize) | ❌ (handle separately) |
| **Pinecone** | Production cloud, millions of vectors | ✅ | ✅ |

To switch to FAISS, set `VECTOR_STORE_TYPE=faiss` in `.env` and update the pipeline instantiation in `rag_pipeline.py`.

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'pipeline'`**
Add to the top of `main.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

**`pydantic_settings` import error**
```bash
pip install pydantic-settings
```

**`ANTHROPIC_API_KEY` not found**
Make sure your `.env` file is in the project root (same folder as `main.py`), not inside a subdirectory.

**ChromaDB `sqlite3` error on Python 3.13**
```bash
pip install chromadb --upgrade
```

**Embedding model download fails (no internet / corporate proxy)**
Pre-download the model and set the local path:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
```

**Empty answers / "No documents ingested yet"**
Call `/stats` to check `total_chunks`. If it's 0, your upload didn't succeed — check the `/upload` response for error details.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Query Rewriter ──► expands vague queries, extracts keywords
    │
    ▼
Embedder ──► converts query to vector (384-dim)
    │
    ▼
ChromaDB ──► hybrid search: vector similarity + keyword overlap
    │         returns top-k candidates with metadata
    ▼
Cross-Encoder ──► re-ranks candidates (query × document)
    │              keeps top-n highest relevance
    ▼
Context Builder ──► formats chunks with [Document N] Source: ... headers
    │
    ▼
Claude API ──► strict grounding prompt + conversation history
    │           generates cited, structured answer
    ▼
Response ──► answer + sources + confidence + token usage
```

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | ≥0.28 | Claude API client |
| `fastapi` | ≥0.111 | Web framework |
| `chromadb` | ≥0.5 | Vector database |
| `sentence-transformers` | ≥3.0 | Local embedding models |
| `pymupdf` | ≥1.24 | PDF parsing |
| `faiss-cpu` | ≥1.8 | FAISS vector index |
| `tiktoken` | ≥0.7 | Token counting |
| `tenacity` | ≥8.3 | Retry logic for API calls |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Anthropic](https://anthropic.com) for the Claude API
- [ChromaDB](https://trychroma.com) for the vector database
- [SentenceTransformers](https://sbert.net) for the embedding models
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
