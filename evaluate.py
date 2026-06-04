"""
evaluate.py — Ground-truth evaluation harness.

Runs the three assignment questions through the RAG pipeline and prints the
question, answer, LLM latency, and the retrieved sources for each.

    python evaluate.py

Expected behaviour (this is the grading rubric):

  Q1  Rate limit (per Key vs per IP)
        -> This fact is NOT in the partial documentation, so the CORRECT answer
           is the EXACT fallback sentence:
           "I'm sorry, but the provided documentation does not contain that
            information."
        This is the key anti-hallucination test: the model must NOT invent a
        plausible-sounding rate limit.

  Q2  OAuth access-token validity
        -> Present in the docs. Correct answer: 24 hours.

  Q3  Client Credentials Grant for private contract details
        -> Answerable from the Client Credentials / Service Account passages
           (these grants act as the app itself, not on behalf of a user, so they
           cannot reach a specific user's private contract data).
"""

from __future__ import annotations

import sys

# Force UTF-8 console output so the box-drawing characters below don't crash on
# Windows terminals using the legacy cp1252 code page. (See ingest.py for detail.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import rag

# The three ground-truth questions, in order.
EVAL_QUESTIONS = [
    "What is the specific request-per-second rate limit for the Upwork API, "
    "and is it enforced per Key or per IP?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract "
    "details?",
]


def run_evaluation() -> None:
    """Execute each evaluation question and print a formatted report."""
    # Fail early with a friendly message if ingest hasn't been run yet.
    if not rag.CHROMA_DIR.exists():
        sys.exit(
            "[ERROR] Vector store not found at ./chroma_db.\n"
            "        Run `python ingest.py` before evaluating.\n"
        )

    print("=" * 78)
    print("UPWORK API SUPPORT BOT — GROUND-TRUTH EVALUATION")
    print("=" * 78)

    for i, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n{'─' * 78}")
        print(f"Q{i}: {question}")
        print("─" * 78)

        result = rag.answer_question(question)

        if result.error:
            # Surface configuration/network problems clearly rather than pretending
            # we got a valid answer.
            print(f"[!] Error ({result.error}): {result.answer}")
            continue

        print(f"Answer:\n{result.answer}\n")
        print(f"LLM latency: {result.latency_seconds:.2f} s")
        print(f"Sources ({len(result.sources)} snippets):")
        for j, snippet in enumerate(result.sources, start=1):
            # Keep the console readable: show a one-line preview per snippet.
            preview = " ".join(snippet.split())[:160]
            print(f"  [{j}] {preview}...")

    print(f"\n{'=' * 78}")
    print("Evaluation complete.")
    print("Reminder: Q1 SHOULD return the exact fallback sentence (not in docs).")
    print("=" * 78)


if __name__ == "__main__":
    run_evaluation()
