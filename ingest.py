"""
ingest.py — Part A: Knowledge Engineering pipeline.

Pipeline:  load PDF  ->  extract text  ->  chunk  ->  embed  ->  store in Chroma.

Run this ONCE (or whenever the source PDF changes) before starting the app:

    python ingest.py

Design notes
------------
* Everything that touches the filesystem or prints to stdout lives behind
  `if __name__ == "__main__"` so that importing this module has NO side effects.
* The Chroma directory is fully deleted and rebuilt on every run. That makes
  re-running idempotent: you can never end up with duplicate vectors from a
  previous ingest.
* We deliberately keep configuration constants in one place at the top so a
  junior dev can see every "knob" at a glance.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Windows consoles often default to a legacy code page (cp1252) that cannot
# encode the box-drawing characters / em-dashes used in our status output, which
# would crash print() with a UnicodeEncodeError. Forcing UTF-8 here makes the
# script run identically on Windows, macOS, and Linux. errors="replace" is a
# belt-and-suspenders fallback so output is never fatal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

# Silence ChromaDB's noisy (and harmless) telemetry failures: request telemetry
# off via env var (before chromadb is imported) AND silence its logger directly,
# which is the guaranteed fix. See rag.py for the full explanation.
import logging
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (kept identical to rag.py so retrieval reads what ingest wrote)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# The source PDF may live next to this script OR one folder up, and its real
# filename uses spaces ("API Documentation Partial.pdf") even though the spec
# wrote it with underscores. We therefore search a list of candidate locations
# and accept either spelling, so the project "just works" without renaming.
PDF_CANDIDATES = [
    PROJECT_ROOT / "API_Documentation_Partial.pdf",
    PROJECT_ROOT / "API Documentation Partial.pdf",
    PROJECT_ROOT.parent / "API_Documentation_Partial.pdf",
    PROJECT_ROOT.parent / "API Documentation Partial.pdf",
]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "upwork_api_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters. We use 1000/200 rather than a smaller window because this
# particular PDF scatters key facts (e.g. "TTL for a refresh token is 2 weeks",
# GraphQL query definitions, HTTP status-code rules) across the page. Larger
# chunks keep each fact together with its surrounding context, which dramatically
# improves retrieval recall — small 500-char chunks isolated facts into chunks
# that did not rank for the natural question, causing false "not in docs" answers.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def resolve_pdf_path(cli_arg: Optional[str] = None) -> Path:
    """Find the source PDF.

    Resolution order:
      1. An explicit path passed on the command line (`python ingest.py <path>`).
      2. The first existing file in PDF_CANDIDATES (handles spaces-vs-underscores
         and "project root" vs "one folder up").

    Args:
        cli_arg: Optional path string from sys.argv.

    Returns:
        The resolved Path. (Existence of the CLI path is validated later in
        load_pdf_text so the user gets one consistent error message.)
    """
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()

    for candidate in PDF_CANDIDATES:
        if candidate.exists():
            return candidate

    # None found: return the canonical expected location so the error message
    # in load_pdf_text points the user at the obvious place to drop the file.
    return PDF_CANDIDATES[0]


def load_pdf_text(pdf_path: Path) -> Tuple[str, int]:
    """Extract and concatenate text from every page of the PDF.

    Handles three edge cases required by the assignment:
      1. Missing file        -> exit with a clear "place it here" message.
      2. Empty/None pages    -> skip safely, then count & report them.
      3. Totally empty text  -> abort (the PDF is almost certainly scanned).

    Args:
        pdf_path: Absolute path to the source PDF.

    Returns:
        A tuple of (full extracted text, total page count). Returning the page
        count here means we open and parse the (large) PDF exactly once.
    """
    # ── Edge case 1: file missing ────────────────────────────────────────────
    if not pdf_path.exists():
        searched = "\n".join(f"          - {c}" for c in PDF_CANDIDATES)
        sys.exit(
            "\n[ERROR] Could not find the knowledge source PDF.\n"
            f"        Looked for: {pdf_path}\n"
            "        Searched these locations:\n"
            f"{searched}\n"
            "        Either place the PDF in one of those spots, or pass an\n"
            "        explicit path:  python ingest.py \"D:\\path\\to\\file.pdf\"\n"
        )

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    page_texts: List[str] = []
    empty_pages = 0

    for page in reader.pages:
        # ── Edge case 2: extract_text() can return None or "" for some pages ──
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(text)
        else:
            empty_pages += 1

    full_text = "\n".join(page_texts)

    print(f"[INFO] Pages in PDF:            {total_pages}")
    print(f"[INFO] Pages with no text:      {empty_pages}")

    # ── Edge case 3: nothing extractable at all ──────────────────────────────
    if not full_text.strip():
        sys.exit(
            "\n[ERROR] No text could be extracted from the PDF.\n"
            "        This usually means it is a scanned/image-only document.\n"
            "        OCR the PDF first (e.g. with `ocrmypdf`) and try again.\n"
        )

    return full_text, total_pages


def print_sanity_check(full_text: str, total_pages: int) -> None:
    """Print the sanity check required by the assignment.

    Shows total character count, total page count, and the first 500 chars so a
    human can eyeball that extraction actually worked.

    Args:
        full_text:   The concatenated extracted text.
        total_pages: Number of pages in the PDF (computed once in load_pdf_text).
    """
    print("\n──────────────── SANITY CHECK ────────────────")
    print(f"Total pages:        {total_pages}")
    print(f"Total characters:   {len(full_text)}")
    print("First 500 characters:")
    print("-----------------------------------------------")
    print(full_text[:500])
    print("───────────────────────────────────────────────\n")


def clean_text(text: str) -> str:
    """Strip repeated page-chrome that pollutes embeddings.

    The exported PDF repeats a navigation footer ("Stack Overflow Getting Started
    TOS FAQ Changelog") and scattered video timestamps ("00:59") and info glyphs
    on almost every page. These carry no semantic value but, because they recur
    ~20+ times, they create near-duplicate chunks that crowd out the real answers
    during retrieval. Removing them measurably improves recall.

    Args:
        text: Raw extracted PDF text.

    Returns:
        The cleaned text.
    """
    noise = "Stack Overflow Getting Started TOS FAQ Changelog"
    text = text.replace(noise, " ")
    text = text.replace("ℹ", " ")                 # stray info glyphs
    text = re.sub(r"\b\d{2}:\d{2}\b", " ", text)   # video timestamps like 00:59

    # Strip JSON sample payloads (request/response/error examples). They are pure
    # retrieval noise: a chunk's embedding gets dominated by "{...}" token salad
    # instead of the prose answer next to it (this is exactly why the 5XX and
    # "missing-scopes -> 200" explanations ranked so low). We remove brace blocks
    # that contain a double-quote (JSON keys/strings), repeatedly to handle
    # nesting. GraphQL *query* blocks use UNQUOTED field names, so they contain no
    # double-quotes and are deliberately preserved.
    json_block = re.compile(r'\{[^{}]*"[^{}]*\}')
    prev = None
    while prev != text:
        prev = text
        text = json_block.sub(" ", text)

    text = re.sub(r"[ \t]+", " ", text)            # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank-line runs
    return text


def chunk_text(full_text: str) -> List[Document]:
    """Split the raw text into overlapping chunks for embedding.

    RecursiveCharacterTextSplitter tries to split on natural boundaries
    (paragraph -> line -> sentence -> word) before falling back to a hard cut,
    which keeps chunks semantically coherent. We also drop whitespace-only
    chunks so we never embed empty vectors.

    Args:
        full_text: The concatenated extracted text.

    Returns:
        A list of LangChain Document objects, one per non-empty chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    raw_chunks = splitter.split_text(full_text)

    # Filter out whitespace-only chunks (belt-and-suspenders; rare but cheap).
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]

    print(f"[INFO] Chunks created:          {len(documents)} "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return documents


def build_vectorstore(documents: List[Document]) -> None:
    """Embed all chunks locally and persist them to ChromaDB.

    The existing ./chroma_db directory is deleted first so re-runs are fully
    idempotent (no duplicate vectors accumulate across runs).

    Args:
        documents: The non-empty chunks produced by `chunk_text`.
    """
    # ── Rebuild from scratch for idempotency ─────────────────────────────────
    if CHROMA_DIR.exists():
        print(f"[INFO] Removing existing vector store at {CHROMA_DIR} ...")
        shutil.rmtree(CHROMA_DIR)

    print(f"[INFO] Loading embedding model:  {EMBEDDING_MODEL}")
    print("       (first run downloads ~90 MB; subsequent runs are cached)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},          # explicit: CPU-only, no GPU needed
        encode_kwargs={"normalize_embeddings": True},  # cosine-friendly vectors
    )

    print("[INFO] Embedding chunks and writing to Chroma ...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    # langchain-chroma persists automatically when persist_directory is set.


def main() -> None:
    """Run the full ingest pipeline end to end."""
    print("=== Upwork API Support Bot — Ingestion ===\n")

    # Allow an optional explicit path: `python ingest.py "C:\path\to\doc.pdf"`.
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pdf_path = resolve_pdf_path(cli_arg)
    print(f"[INFO] Using PDF:               {pdf_path}")

    full_text, total_pages = load_pdf_text(pdf_path)
    print_sanity_check(full_text, total_pages)

    cleaned = clean_text(full_text)
    removed = len(full_text) - len(cleaned)
    print(f"[INFO] Cleaned page-chrome:     removed {removed} chars of noise")

    documents = chunk_text(cleaned)
    build_vectorstore(documents)

    print("\n[SUCCESS] Ingestion complete.")
    print(f"          Collection : {COLLECTION_NAME}")
    print(f"          Vectors    : {len(documents)}")
    print(f"          Location   : {CHROMA_DIR}")
    print("          You can now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
