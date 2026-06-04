# 🤖 Upwork API Support Bot

A Retrieval-Augmented Generation (RAG) technical-support bot that answers
developer questions about the **Upwork API** using **only** a local PDF
(`API_Documentation_Partial.pdf`) as its knowledge source.

It is built to **never hallucinate**: if the answer is not in the retrieved
documentation, it replies with exactly:

> I'm sorry, but the provided documentation does not contain that information.

---

## ✨ How it works

```
PDF ──pypdf──> raw text ──RecursiveCharacterTextSplitter──> chunks
      ──all-MiniLM-L6-v2 (local, CPU)──> embeddings ──> ChromaDB (./chroma_db)

question ──> top-K similar chunks ──> strict system prompt + context
         ──> Llama-3.1-8B-Instruct-Turbo (via DeepInfra) ──> grounded answer
```

| Layer        | Technology                                                        |
|--------------|-------------------------------------------------------------------|
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` (runs locally on CPU)    |
| Vector DB    | ChromaDB, persisted to `./chroma_db`                              |
| LLM          | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` via DeepInfra       |
| Framework    | LangChain (`langchain-chroma`, `langchain-huggingface`, splitters)|
| UI           | Streamlit                                                         |
| Secrets      | `python-dotenv` (`.env`, never committed)                         |

### Retrieval tuning (why the defaults were changed)

The starting parameters (chunk_size 500 / overlap 50 / top-3) gave poor recall on
this particular PDF: it scatters key facts across the page and mixes prose with
large JSON sample payloads, and the mandated embedding model
(`all-MiniLM-L6-v2`) ranks code/JSON-heavy passages weakly. As a result many
answerable questions (refresh-token TTL, token endpoint, GraphQL queries, HTTP
status-code rules) wrongly returned the "not in documentation" fallback. The
following evidence-based changes fixed them while keeping the anti-hallucination
guarantee (questions whose answer is genuinely absent — rate limit, pricing —
still fall back):

- **Ingestion cleaning** ([ingest.py](ingest.py) `clean_text`): strip the repeated
  page-footer chrome and JSON sample payloads that were dominating embeddings.
- **chunk_size 1000 / overlap 200**: keep each fact together with its context.
- **top-18 retrieval** ([rag.py](rag.py) `RETRIEVAL_K`): with a weak mandated
  embedder, a higher k is the legitimate way to recover recall on a ~42-chunk
  corpus. With a stronger embedder, k=3–5 would suffice.
- **Extraction-friendly prompt with an anti-conflation rule**: answer from code
  samples / GraphQL queries when present, and don't confuse adjacent cases (e.g.
  missing-scopes → HTTP 200 vs GraphQL-layer failure → 5XX).

---

## 📁 Project structure

```
upwork-rag-bot/
├── ingest.py          # Part A: extract → chunk → embed → store
├── rag.py             # Retrieval + LLM call logic (importable, testable)
├── app.py             # Streamlit UI
├── evaluate.py        # Runs the 3 ground-truth questions from the CLI
├── requirements.txt   # Pinned versions
├── .env.example       # Template for your DeepInfra key
├── .gitignore         # Ignores .env, venv/, chroma_db/, __pycache__/
└── README.md          # This file
```

---

## 🚀 Setup & run

> Requires **Python 3.10+**.

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your DeepInfra API key

Copy the template and paste your real key into the new `.env` file:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```
**Mac / Linux:**
```bash
cp .env.example .env
```

Then edit `.env`:
```
DEEPINFRA_API_KEY=sk-your-real-key-here
```

Get a key at <https://deepinfra.com/dash/api_keys>.
**Never** commit `.env` or share the key — `.gitignore` already excludes it.

### 4. Place the source PDF

Put `API_Documentation_Partial.pdf` in the **project root** (next to `ingest.py`).

### 5. Build the knowledge base
```bash
python ingest.py
```
This prints a sanity check (page count, character count, first 500 chars),
the number of chunks, and a success summary. It creates `./chroma_db`.
Re-running is safe — it rebuilds from scratch (no duplicate vectors).

### 6. Launch the app
```bash
streamlit run app.py
```
Your browser opens at <http://localhost:8501>.

### 7. (Optional) Run the ground-truth evaluation
```bash
python evaluate.py
```
This runs the three assignment questions and prints answers, latency, and
sources. **Q1 should return the exact fallback sentence** (the rate limit is
intentionally absent from the partial docs — the anti-hallucination test).

---

## 🧪 The three evaluation questions

| #  | Question                                                                 | Expected behaviour                                  |
|----|--------------------------------------------------------------------------|-----------------------------------------------------|
| Q1 | Request-per-second rate limit, per Key or per IP?                        | **Fallback sentence** — not in the docs.            |
| Q2 | How long is an OAuth access token valid for?                             | Answered: **24 hours**.                             |
| Q3 | Can a Client Credentials Grant access a user's private contract details? | Answered from the Client Credentials / Service Account passages. |

---

## 🛠️ Troubleshooting

**First run is slow / appears to hang.**
The first time you run `ingest.py` or `app.py`, `sentence-transformers`
downloads the `all-MiniLM-L6-v2` model (~90 MB). This is a one-time download;
it is cached under your home directory afterwards.

**`HTTP 401` / "Authentication failed".**
Your `DEEPINFRA_API_KEY` is missing, mistyped, or revoked. Open `.env`, confirm
the key has no extra spaces/quotes, and that you saved the file. Restart the app
so the new value is picked up.

**`HTTP 429` / rate limited.**
You're sending requests faster than your DeepInfra plan allows. The bot already
retries twice with exponential backoff; wait a few seconds and try again.

**"Vector store not found. Run ingest.py first."**
You launched `app.py`/`evaluate.py` before building the index. Run
`python ingest.py` and refresh.

**ChromaDB / SQLite version error** (e.g. *"unsupported version of sqlite3"*).
ChromaDB needs SQLite ≥ 3.35. On older Linux systems, either upgrade SQLite or
install the bundled binary:
```bash
pip install pysqlite3-binary
```
and add this at the very top of `ingest.py` / `rag.py`:
```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules["pysqlite3"]
```
(Not needed on Windows or recent macOS, which ship a modern SQLite.)

**"No text could be extracted from the PDF."**
The PDF is likely a scanned image. OCR it first (e.g. with `ocrmypdf`) and
re-run `python ingest.py`.

---

## 🔐 Security notes

- The API key is read at runtime from `.env` via `python-dotenv` — it is
  **never** hardcoded in any source file.
- `.env`, `venv/`, `chroma_db/`, and `__pycache__/` are all git-ignored.
- If a key is ever exposed (committed, pasted in chat, screenshotted), **rotate
  it immediately** in the DeepInfra dashboard.
