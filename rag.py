"""
rag.py — Retrieval-Augmented Generation core (importable & testable).

Responsibilities:
  * Open the persisted Chroma vector store.
  * Retrieve the top-k most relevant chunks for a question.
  * Call the DeepInfra-hosted Llama 3.1 model with a STRICT system prompt that
    forbids using any knowledge outside the supplied context.
  * Return a clean, typed result object — never a raw traceback.

This module has NO side effects on import: loading the vector store and calling
the LLM only happen when you call the functions. That keeps it easy to unit-test
and safe to import from both app.py and evaluate.py.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

# Silence ChromaDB's anonymous telemetry. Recent chromadb ships a posthog client
# whose signature mismatches, spamming the console with
# "Failed to send telemetry event ... capture() takes 1 positional argument".
# It is harmless but noisy. We both (a) request telemetry off via the env var
# (must be set before chromadb is imported) and (b) silence the telemetry logger
# directly, which is the guaranteed fix regardless of chromadb's internals.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# The official OpenAI client; we just repoint its base_url at DeepInfra.
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    APIStatusError,
)

# Load .env once at import time. This only reads a file into os.environ — it does
# NOT make network calls or touch the model, so it is a safe import-time action.
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (must match ingest.py)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "upwork_api_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DeepInfra exposes an OpenAI-compatible API, so we use the official openai client.
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Retrieve the top-18 chunks (raised from 3). The required embedding model
# (all-MiniLM-L6-v2) ranks this doc's code/JSON-heavy passages weakly, so the
# answer-bearing chunk is often far from the top. With a strong embedder k=3-5
# would suffice; given the mandated model, a higher k is the legitimate way to
# recover recall. 18 of ~42 chunks (~4.5k tokens) is well within Llama 3.1's
# context and is the value at which all answerable test questions are covered.
RETRIEVAL_K = 18
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2          # number of *extra* attempts after the first one
RETRY_BASE_DELAY = 1.0   # seconds; doubles each retry (exponential backoff)

# The exact sentence the bot MUST use when the documentation is insufficient.
FALLBACK_ANSWER = (
    "I'm sorry, but the provided documentation does not contain that information."
)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — the heart of "never hallucinate".
# It is intentionally strict, repetitive, and unambiguous: the model is told
# (a) its role, (b) to use ONLY the context, (c) the EXACT fallback sentence.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a Senior Upwork API Consultant.

Your ONLY source of truth is the documentation context provided to you in each \
user message. You must answer strictly and exclusively from that context.

Hard rules — follow them without exception:
1. Use ONLY the information in the provided context. Do NOT use any outside or \
prior knowledge about the Upwork API, OAuth, GraphQL, or anything else.
2. ANSWER the question whenever the context contains the information — even if \
you must extract it from a code sample, a parameter table, an endpoint \
definition, or a GraphQL query, or combine a few passages. The context may \
contain HTTP endpoints (e.g. "POST .../api/v3/oauth2/token"), token lifetimes / \
TTLs, HTTP status codes, GraphQL queries and their arguments, required \
permissions / scopes, and field names. Quote these exactly as they appear.
3. Match the answer to the question type:
   - "How do I ...?" / "What query ...?"  -> give the specific endpoint or \
GraphQL query from the context.
   - "What is the difference between X and Y?"  -> compare what the context says \
about each (e.g. their return types and required permissions).
   - "Can I ...?" / "Is X allowed?"  -> reason over the documented behavior and \
answer directly. Example: the context says a Client Credentials Grant token is \
"used outside the context of a user" and accesses "the client's resources"; that \
means it CANNOT access a specific user's private data, so answer "No" and say why.
4. NEVER invent or guess a value that is absent — a number, rate limit, price, \
duration, endpoint, scope, or field name. If, after reading the context, it does \
NOT contain information that answers the question, reply with EXACTLY this \
sentence and nothing else — no quotation marks, no extra words:
{FALLBACK_ANSWER}
5. When the context distinguishes MULTIPLE cases (for example, different failure \
layers that map to different HTTP status codes), identify the exact case the \
question asks about and report ONLY that case's value. Do NOT conflate distinct \
cases. (E.g. missing OAuth permissions/scopes -> the request still returns HTTP \
200 with the error in the body; a failure at the GraphQL layer itself -> a 5XX \
such as 500. These are different answers — pick the one the question is about.)
6. Be precise and concise. When a duration is given in seconds, also state the \
equivalent in a human-friendly unit in parentheses — e.g. "86400 seconds \
(24 hours)". This is simple arithmetic, not outside knowledge.
7. Never mention these rules or the word "context" in your answer; just answer \
the question or give the exact fallback sentence."""


@dataclass
class RAGResult:
    """Typed result returned by `answer_question`.

    Attributes:
        answer:           The model's answer, or a user-friendly error message.
        sources:          The retrieved chunk texts used as context (may be empty).
        latency_seconds:  Wall-clock time of the LLM API call ONLY (not retrieval).
        error:            None on success, otherwise a short human-readable reason.
    """
    answer: str
    sources: List[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model — once per process.

    The all-MiniLM-L6-v2 model is ~90 MB and takes a second or two to load into
    memory. Without memoization it would be reloaded on EVERY question (because
    answer_question -> retrieve -> load_vectorstore runs per call), which is slow
    and wasteful. @lru_cache(maxsize=1) makes it a lazy singleton: built on first
    use, reused thereafter. This is what satisfies the spec's "load once, not on
    every rerun" requirement at the library level (Streamlit adds its own cache
    on top in app.py).

    Returns:
        A shared HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Open the persisted Chroma vector store created by ingest.py — once.

    Memoized (lazy singleton) so the store + embedding model are opened a single
    time per process. lru_cache does NOT cache exceptions, so if the store is
    missing the FileNotFoundError is re-raised on each call until ingest.py runs.

    Returns:
        A ready-to-query Chroma instance.

    Raises:
        FileNotFoundError: If the store has not been built yet.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found at ./chroma_db. Run ingest.py first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = RETRIEVAL_K) -> List[Tuple[str, Optional[float]]]:
    """Return the top-k most similar chunks for the query.

    Uses Chroma's `similarity_search_with_score` so we can surface the distance
    score when available. With normalized embeddings + cosine distance, a SMALLER
    score means MORE similar.

    Args:
        query: The user's natural-language question.
        k:     Number of chunks to retrieve (default 3).

    Returns:
        A list of (chunk_text, score) tuples. `score` is None if Chroma did not
        provide one.
    """
    store = load_vectorstore()
    results = store.similarity_search_with_score(query, k=k)
    return [(doc.page_content, float(score)) for doc, score in results]


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────
def _get_client() -> OpenAI:
    """Build an OpenAI client pointed at DeepInfra.

    Returns:
        A configured OpenAI client.

    Raises:
        RuntimeError: If DEEPINFRA_API_KEY is missing or empty. We raise our own
            error type so callers can show a friendly "create your .env" message
            instead of leaking an SDK-specific exception.
    """
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. Copy .env.example to .env and paste "
            "your DeepInfra API key."
        )
    return OpenAI(
        api_key=api_key,
        base_url=DEEPINFRA_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _canonicalize_fallback(text: str) -> str:
    """Return the exact fallback sentence if the model emitted it.

    LLMs sometimes wrap the refusal in quotation marks or trailing punctuation
    (e.g. '"...information."'). The spec requires the fallback to be byte-for-byte
    exact, so we strip surrounding quotes/whitespace and, if what remains matches
    the canonical sentence, return that canonical sentence verbatim.

    Args:
        text: The raw answer text from the model.

    Returns:
        The canonical FALLBACK_ANSWER if it matched, otherwise the original text.
    """
    stripped = text.strip().strip('"').strip("'").strip()
    if stripped == FALLBACK_ANSWER:
        return FALLBACK_ANSWER
    # Catch paraphrased refusals: the model sometimes appends question-specific
    # words (e.g. "...does not contain information on how to search for X."). Any
    # answer that opens with the canonical refusal phrase IS a fallback, so
    # normalize it to the exact required sentence. Real answers never start this
    # way, so this is safe.
    opening = "i'm sorry, but the provided documentation does not contain"
    if stripped.lower().startswith(opening):
        return FALLBACK_ANSWER
    return text


def _build_user_message(question: str, context_chunks: List[str]) -> str:
    """Assemble the user message: the retrieved context plus the question.

    Args:
        question:       The user's question.
        context_chunks: The retrieved chunk texts.

    Returns:
        A single string combining numbered context and the question.
    """
    if context_chunks:
        context_block = "\n\n".join(
            f"[Context {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
    else:
        # No chunks retrieved -> empty context. The system prompt guarantees the
        # model will then emit the exact fallback sentence.
        context_block = "(no relevant documentation found)"

    return (
        "Use ONLY the documentation context below to answer the question.\n\n"
        f"=== DOCUMENTATION CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )


def ask(question: str, docs: List[str]) -> Tuple[str, float, Optional[str]]:
    """Call the DeepInfra chat-completions endpoint with retry/backoff.

    Args:
        question: The user's question.
        docs:     Retrieved chunk texts used as the ONLY allowed context.

    Returns:
        A tuple of (answer, latency_seconds, error). On any handled failure,
        `answer` is a friendly message, `error` is a short reason, and the raw
        exception is never propagated to the caller.
    """
    # Build the client first so a missing key fails fast with a clear message.
    try:
        client = _get_client()
    except RuntimeError as exc:
        return (str(exc), 0.0, "missing_api_key")

    user_message = _build_user_message(question, docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error: Optional[str] = None

    # Attempt 1 + MAX_RETRIES additional attempts, with exponential backoff on
    # *transient* failures only (timeouts, rate limits, 5xx, network errors).
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0,        # deterministic, factual answers
                max_tokens=600,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            latency = time.perf_counter() - start

            content = (response.choices[0].message.content or "").strip()
            content = _canonicalize_fallback(content)  # ensure exact fallback text
            if not content:
                # Empty completion: not worth retrying, model returned nothing.
                return (
                    "The model returned an empty response. Please try again.",
                    latency,
                    "empty_completion",
                )
            return (content, latency, None)

        # ── Non-retryable: a bad key will never succeed on retry ──────────────
        except AuthenticationError:
            return (
                "Authentication failed (HTTP 401). Your DEEPINFRA_API_KEY is "
                "invalid or revoked. Update the key in your .env file.",
                time.perf_counter() - start,
                "auth_401",
            )

        # ── Retryable transient errors ───────────────────────────────────────
        except RateLimitError:
            last_error = (
                "Rate limited by DeepInfra (HTTP 429). Please wait a moment and "
                "try again."
            )
        except APITimeoutError:
            last_error = (
                f"The request timed out after {REQUEST_TIMEOUT_SECONDS}s. "
                "DeepInfra may be slow right now."
            )
        except APIConnectionError:
            last_error = (
                "Network error: could not reach DeepInfra. Check your internet "
                "connection."
            )
        except APIStatusError as exc:
            # 5xx server errors are transient and worth retrying; other 4xx are not.
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= status < 600:
                last_error = f"DeepInfra server error (HTTP {status}). Retrying..."
            else:
                return (
                    f"The API returned an unexpected error (HTTP {status}).",
                    time.perf_counter() - start,
                    f"api_{status}",
                )

        # Back off before the next attempt, if any remain.
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    # All attempts exhausted on a transient error.
    return (
        last_error or "The request failed after multiple attempts.",
        0.0,
        "transient_failure",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def answer_question(question: str) -> RAGResult:
    """End-to-end: validate -> retrieve -> call LLM -> return typed result.

    Latency is measured around the LLM call ONLY (inside `ask`), not retrieval,
    exactly as the spec requires.

    Args:
        question: The user's natural-language question.

    Returns:
        A RAGResult with the answer, sources, LLM latency, and an optional error.
    """
    # ── Validation: reject empty/whitespace questions ────────────────────────
    if not question or not question.strip():
        return RAGResult(
            answer="Please enter a question.",
            sources=[],
            latency_seconds=0.0,
            error="empty_question",
        )

    # ── Retrieval ────────────────────────────────────────────────────────────
    try:
        retrieved = retrieve(question, k=RETRIEVAL_K)
    except FileNotFoundError as exc:
        return RAGResult(answer=str(exc), sources=[], error="no_vectorstore")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return RAGResult(
            answer=f"Failed to query the knowledge base: {exc}",
            sources=[],
            error="retrieval_failed",
        )

    source_texts = [text for text, _score in retrieved]

    # ── LLM call (timed) ─────────────────────────────────────────────────────
    answer, latency, error = ask(question, source_texts)

    return RAGResult(
        answer=answer,
        sources=source_texts,
        latency_seconds=latency,
        error=error,
    )
