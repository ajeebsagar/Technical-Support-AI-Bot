"""
app.py — Streamlit chat UI for the Upwork API Support Bot.

Run with:

    streamlit run app.py

Key Streamlit-specific design choices:
  * @st.cache_resource warms the embedding model + vector store ONCE per process,
    not on every rerun (Streamlit re-executes this whole script on each
    interaction, so without caching we'd reload the model every keystroke).
  * Chat history lives in st.session_state so it survives reruns.
  * Every user-facing failure is shown as st.error / a friendly message — the
    app never crashes with a raw traceback.
"""

from __future__ import annotations

import os

import streamlit as st

import rag  # our importable RAG core (no side effects on import)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the first Streamlit call.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Secrets bridge — make the DeepInfra key available to rag.py on Streamlit Cloud.
# Locally we read the key from .env (via python-dotenv). When deployed to
# Streamlit Community Cloud there is no .env; the key is provided through the
# app's "Secrets" manager and exposed via st.secrets. We copy it into os.environ
# so rag._get_client()'s os.getenv("DEEPINFRA_API_KEY") finds it either way.
# st.secrets raises if no secrets file exists (the normal local case), so we
# guard it and fall back silently to the .env value.
try:
    if "DEEPINFRA_API_KEY" in st.secrets:
        os.environ["DEEPINFRA_API_KEY"] = str(st.secrets["DEEPINFRA_API_KEY"])
except Exception:
    pass

# The 3 ground-truth evaluation questions, surfaced as clickable buttons.
SAMPLE_QUESTIONS = [
    "What is the specific request-per-second rate limit for the Upwork API, "
    "and is it enforced per Key or per IP?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract "
    "details?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Light custom CSS — subtle, professional spacing/colors (not flashy).
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject a small amount of CSS for visual polish."""
    st.markdown(
        """
        <style>
            .main-title { font-size: 2.0rem; font-weight: 700; margin-bottom: 0; }
            .subtitle   { color: #6b7280; font-size: 1.0rem; margin-top: 0.2rem; }
            .latency-badge {
                display: inline-block; padding: 2px 10px; border-radius: 12px;
                background: #eef2ff; color: #4338ca; font-size: 0.8rem;
                font-weight: 600;
            }
            .stChatMessage { padding-top: 0.3rem; }
            section[data-testid="stSidebar"] { background: #fafafa; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources — load the heavy model + vector store only once.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing the knowledge base (first run builds the index)...")
def get_vectorstore() -> rag.Chroma:
    """Load and CACHE the vector store, BUILDING it first if necessary.

    On Streamlit Community Cloud the prebuilt ./chroma_db is not in the repo (it
    is git-ignored), so on first boot we build it from the PDF that ships in the
    repo — extract -> clean -> chunk -> embed -> persist — exactly what
    `python ingest.py` does locally. This makes deployment self-healing: no manual
    ingest step is required. The result is cached by @st.cache_resource so the
    build happens once per app instance, not on every rerun.

    Returns:
        A ready-to-query Chroma instance.

    Raises:
        FileNotFoundError: If neither the index nor the source PDF is available.
    """
    if not rag.CHROMA_DIR.exists():
        # Build the index from the bundled PDF (import here to avoid a hard
        # dependency on ingest.py for users who only ever run a prebuilt store).
        import ingest

        pdf_path = ingest.resolve_pdf_path()
        if not pdf_path.exists():
            raise FileNotFoundError(
                "No vector store and no source PDF found. Commit "
                "'API Documentation Partial.pdf' to the repo or run "
                "`python ingest.py` locally."
            )
        text, _pages = ingest.load_pdf_text(pdf_path)
        documents = ingest.chunk_text(ingest.clean_text(text))
        ingest.build_vectorstore(documents)
        rag.load_vectorstore.cache_clear()  # drop any cached miss, open fresh

    return rag.load_vectorstore()


def get_kb_status() -> tuple[bool, str]:
    """Determine the knowledge-base status for the sidebar indicator.

    Triggers the (cached) build/load so the first question is fast and the sidebar
    can show an accurate green/red badge.

    Returns:
        (ok, message). `ok` is True if the store is ready.
    """
    try:
        get_vectorstore()  # builds on first run if needed, then caches
        return (True, "Knowledge base loaded and ready.")
    except Exception as exc:  # pragma: no cover - defensive
        return (False, f"Knowledge base unavailable: {exc}")


def render_sidebar(kb_ok: bool, kb_msg: str) -> None:
    """Render the sidebar: app info, KB status, sample questions, clear button.

    Args:
        kb_ok:  Whether the knowledge base loaded successfully.
        kb_msg: Human-readable status message.
    """
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "Answers developer questions about the **Upwork API** using ONLY a "
            "local PDF as its knowledge source. It will not guess: if the answer "
            "isn't in the docs, it says so."
        )

        st.subheader("⚙️ Configuration")
        st.markdown(f"- **LLM:** `{rag.LLM_MODEL}`")
        st.markdown(f"- **Embeddings:** `{rag.EMBEDDING_MODEL}`")
        st.markdown(f"- **Vector DB:** ChromaDB (`./chroma_db`)")
        st.markdown(f"- **Retrieved chunks:** top-{rag.RETRIEVAL_K}")

        st.subheader("📚 Knowledge base status")
        if kb_ok:
            st.success(kb_msg)
        else:
            st.error(kb_msg)

        st.subheader("💡 Sample questions")
        st.caption("Click to ask one of the evaluation questions.")
        for i, q in enumerate(SAMPLE_QUESTIONS):
            # A short label keeps buttons tidy; the full question is sent on click.
            if st.button(f"Q{i + 1}. {q[:42]}…", key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()


def render_history() -> None:
    """Replay the stored conversation so it persists across reruns."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Error turns render as st.error with no (misleading) latency badge.
                if msg.get("error"):
                    st.error(msg["content"])
                else:
                    st.markdown(msg["content"])
                    if msg.get("latency") is not None:
                        st.markdown(
                            f"<span class='latency-badge'>⏱ API latency: "
                            f"{msg['latency']:.2f} s</span>",
                            unsafe_allow_html=True,
                        )
                sources = msg.get("sources") or []
                if sources:
                    with st.expander(f"📄 Sources ({len(sources)} snippets)"):
                        for i, snippet in enumerate(sources, start=1):
                            st.markdown(f"**Snippet {i}**")
                            st.code(snippet, language="text")
            else:
                st.markdown(msg["content"])


def handle_question(question: str) -> None:
    """Process one question: echo it, call the RAG core, render the answer.

    Args:
        question: The user's question (already validated as non-empty).
    """
    # Show & store the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and render the assistant's reply.
    with st.chat_message("assistant"):
        with st.spinner("Consulting documentation..."):
            result = rag.answer_question(question)

        # Any error (missing key, no vector store, 401/429/timeout, empty reply)
        # is shown prominently as st.error; the message text tells the user how to
        # fix it. The app never crashes with a raw traceback.
        if result.error:
            st.error(result.answer)
        else:
            st.markdown(result.answer)
            st.markdown(
                f"<span class='latency-badge'>⏱ API latency: "
                f"{result.latency_seconds:.2f} s</span>",
                unsafe_allow_html=True,
            )

        # Show retrieved sources whenever retrieval succeeded (even if the LLM
        # call later failed) — they are still useful context for the user.
        if result.sources:
            with st.expander(f"📄 Sources ({len(result.sources)} snippets)"):
                for i, snippet in enumerate(result.sources, start=1):
                    st.markdown(f"**Snippet {i}**")
                    st.code(snippet, language="text")

    # Persist the assistant turn (with metadata) so it survives reruns.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "latency": result.latency_seconds,
            "sources": result.sources,
            "error": result.error,
        }
    )


def main() -> None:
    """Compose the full page."""
    inject_css()

    # ── Session state init ───────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown('<div class="main-title">🤖 Upwork API Support Bot</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask developer questions about the Upwork API — '
        'answered strictly from the official documentation, with sources.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    kb_ok, kb_msg = get_kb_status()
    render_sidebar(kb_ok, kb_msg)

    # If the KB is missing, guide the user instead of letting them ask questions.
    if not kb_ok:
        st.warning(
            "The knowledge base is not ready. Open a terminal in the project "
            "folder and run `python ingest.py`, then refresh this page."
        )

    render_history()

    # ── Input handling ───────────────────────────────────────────────────────
    # A sidebar button may have queued a question; otherwise read the chat input.
    typed = st.chat_input("Ask about rate limits, OAuth tokens, scopes...")
    pending = st.session_state.pending_question
    st.session_state.pending_question = None  # consume it

    question = (typed or pending or "").strip()

    if question:
        if not kb_ok:
            st.error("Cannot answer until the knowledge base is built. "
                     "Run `python ingest.py` first.")
        else:
            handle_question(question)


if __name__ == "__main__":
    main()
