"""Streamlit front end for the NLP-Based Plagiarism Checker.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from plagiarism import (
    HIGH_LABEL,
    LOW_LABEL,
    MODERATE_LABEL,
    SimilarityError,
    classify_similarity,
    compare_documents,
)
from preprocessing import ensure_nltk_resources, preprocess_text

logger = logging.getLogger(__name__)

APP_TITLE = "NLP Plagiarism Checker"
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB is far more than a text comparison needs
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLE_PAIRS: Dict[str, Tuple[str, str]] = {
    "Identical documents": ("original.txt", "original.txt"),
    "Highly similar": ("original.txt", "similar.txt"),
    "Paraphrased": ("original.txt", "paraphrased.txt"),
    "Completely different": ("original.txt", "different.txt"),
}

# label -> (text colour, background colour)
LABEL_STYLES = {
    LOW_LABEL: ("#15803D", "#DCFCE7"),
    MODERATE_LABEL: ("#B45309", "#FEF3C7"),
    HIGH_LABEL: ("#B91C1C", "#FEE2E2"),
}

DISCLAIMER = (
    "**Textual similarity is not definitive proof of plagiarism.** This tool only "
    "compares the two documents you supply; it does not search the internet or any "
    "database. Quotations, shared terminology and common phrases can also produce "
    "high scores, so always review the texts yourself."
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
.hero { background: linear-gradient(135deg, #312E81 0%, #4F46E5 55%, #6366F1 100%); border-radius: 18px; padding: 2rem 2.25rem; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25); }
.hero-title { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 0.35rem; }
.hero-text { font-family: 'Inter', 'Segoe UI', sans-serif; color: #E0E7FF; font-size: 1.02rem; line-height: 1.55; max-width: 46rem; }
.panel-heading { font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 600; font-size: 1.05rem; color: #0F172A; margin: 0.25rem 0 0.5rem 0; }
.card { font-family: 'Inter', 'Segoe UI', sans-serif; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.1rem 1.25rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05); min-height: 8.5rem; }
.card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color: #64748B; font-weight: 600; }
.card-value { font-size: 2.2rem; font-weight: 700; color: #0F172A; line-height: 1.2; margin: 0.3rem 0; }
.card-caption { font-size: 0.85rem; color: #64748B; }
.badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; font-weight: 600; font-size: 1rem; margin: 0.55rem 0 0.4rem 0; }
.meter { width: 100%; height: 14px; background: #E2E8F0; border-radius: 999px; overflow: hidden; margin: 1.1rem 0 0.35rem 0; }
.meter-fill { height: 100%; border-radius: 999px; }
.meter-scale { display: flex; justify-content: space-between; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 0.78rem; color: #64748B; margin-bottom: 1rem; }
.chip { display: inline-block; background: #EEF2FF; color: #3730A3; border: 1px solid #C7D2FE; border-radius: 999px; padding: 0.2rem 0.75rem; margin: 0 0.4rem 0.45rem 0; font-size: 0.88rem; font-weight: 500; }
.textbox { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem 1rem; max-height: 220px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem; color: #1E293B; line-height: 1.55; }
.stButton > button { width: 100%; height: 3.1rem; border-radius: 12px; font-weight: 600; font-size: 1.05rem; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3); }
.footer-note { font-family: 'Inter', 'Segoe UI', sans-serif; text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: 2.5rem; }
</style>
"""


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
@dataclass
class InputDocument:
    """One side of the comparison after reading the text area / uploaded file."""

    title: str
    source_name: str
    text: str = ""
    error: Optional[str] = None


def decode_uploaded_file(uploaded_file) -> Tuple[str, Optional[str]]:
    """Validate and decode an uploaded file. Returns (text, error_message)."""
    name = uploaded_file.name
    if not name.lower().endswith(".txt"):
        return "", f"'{name}' is not supported. Please upload a .txt file."

    raw = uploaded_file.getvalue()
    if len(raw) > MAX_FILE_SIZE_BYTES:
        return "", f"'{name}' is larger than {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
    if not raw.strip():
        return "", f"'{name}' is empty."

    try:
        return raw.decode("utf-8-sig"), None  # utf-8-sig also strips a Windows BOM
    except UnicodeDecodeError:
        return "", (
            f"'{name}' could not be read as UTF-8 text. Re-save it with UTF-8 "
            "encoding (in Notepad: File > Save As > Encoding: UTF-8) and try again."
        )


def render_input_panel(title: str, key: str) -> InputDocument:
    """Draw a text area plus a file uploader and return the resulting document."""
    st.markdown(f'<div class="panel-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    pasted_text = st.text_area(
        f"{title} text",
        key=f"{key}_text",
        height=240,
        placeholder="Paste text here, or upload a .txt file below...",
        label_visibility="collapsed",
    )
    uploaded_file = st.file_uploader(
        f"Or upload a .txt file for {title}", type=["txt"], key=f"{key}_file"
    )

    if uploaded_file is not None:
        text, error = decode_uploaded_file(uploaded_file)
        if error:
            st.error(error)
            return InputDocument(title, uploaded_file.name, error=error)
        st.success(
            f"Loaded **{uploaded_file.name}** ({len(text.split()):,} words). "
            "The uploaded file is used instead of the pasted text."
        )
        return InputDocument(title, uploaded_file.name, text)

    return InputDocument(title, "Pasted text", pasted_text)


def load_sample_pair() -> None:
    """Sidebar callback: fill both text areas with a sample pair from samples/."""
    file_a, file_b = SAMPLE_PAIRS[st.session_state["sample_choice"]]
    try:
        st.session_state["doc1_text"] = (SAMPLES_DIR / file_a).read_text(encoding="utf-8")
        st.session_state["doc2_text"] = (SAMPLES_DIR / file_b).read_text(encoding="utf-8")
        st.session_state["sample_error"] = None
        st.session_state.pop("analysis", None)
    except OSError as exc:
        st.session_state["sample_error"] = f"Could not read the sample files: {exc}"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Preparing language resources...")
def load_nlp_resources() -> Tuple[bool, str]:
    """Download/verify NLTK data once per server process."""
    return ensure_nltk_resources()


def run_analysis(doc_a: InputDocument, doc_b: InputDocument) -> Optional[dict]:
    """Validate inputs, run the NLP pipeline and return a dict for display (or None)."""
    problems = []
    for doc in (doc_a, doc_b):
        if doc.error:
            problems.append(doc.error)
        elif not doc.text.strip():
            problems.append(f"{doc.title} is empty. Paste some text or upload a .txt file.")
    if problems:
        for message in problems:
            st.error(message)
        return None

    try:
        with st.spinner("Analyzing documents..."):
            clean_a = preprocess_text(doc_a.text)
            clean_b = preprocess_text(doc_b.text)
            result = compare_documents(clean_a, clean_b)
    except SimilarityError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:  # last line of defence: never show a raw traceback
        logger.exception("Unexpected error during analysis")
        st.error(f"Something unexpected went wrong while analyzing the documents: {exc}")
        return None

    return {
        "result": result,
        "clean_a": clean_a,
        "clean_b": clean_b,
        "name_a": doc_a.source_name,
        "name_b": doc_b.source_name,
    }


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def metric_card(title: str, value: str, caption: str) -> str:
    # Single-line HTML: indented lines would be treated as a code block by Markdown.
    return (
        '<div class="card">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-caption">{html.escape(caption)}</div>'
        "</div>"
    )


def render_header() -> None:
    st.markdown(
        '<div class="hero">'
        f'<div class="hero-title">{APP_TITLE}</div>'
        '<div class="hero-text">Compare two documents using TF-IDF and cosine similarity. '
        "Paste text or upload .txt files, then review a similarity score, shared keywords "
        "and the preprocessed text behind it.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    with st.expander("How this works"):
        st.markdown(
            """
1. **Preprocess** - lowercase, remove punctuation, tokenize, drop stopwords and lemmatize (NLTK).
2. **Vectorize** - turn each document into a TF-IDF vector (scikit-learn).
3. **Compare** - cosine similarity measures the angle between the two vectors: 100% means the same word weighting, 0% means no shared vocabulary.
4. **Classify** - the percentage is labelled using the thresholds in the sidebar.

**Why TF-IDF?** Plain word counts over-reward common words. TF-IDF multiplies how often a word appears
in a document (term frequency) by how rare it is across the documents (inverse document frequency),
so distinctive words weigh more than filler. With only two documents the IDF effect is modest:
words in both documents get the lowest weight, words in one get a slightly higher one.
            """
        )


def render_sidebar() -> Tuple[int, int]:
    """Draw the sidebar and return (low_threshold, high_threshold)."""
    with st.sidebar:
        st.markdown("### Settings")
        low, high = st.slider(
            "Similarity thresholds (%)",
            min_value=0,
            max_value=100,
            value=(40, 70),
            help="Drag the handles to decide where Moderate and High Similarity begin.",
        )
        st.caption(f"Low below {low}%  |  Moderate {low}-{high}%  |  High from {high}%")

        st.divider()
        st.markdown("### Try sample documents")
        st.selectbox("Sample pair", list(SAMPLE_PAIRS), key="sample_choice")
        st.button("Load sample", on_click=load_sample_pair)
        if st.session_state.get("sample_error"):
            st.error(st.session_state["sample_error"])
        st.caption("Clear any uploaded files first; uploads take priority over pasted text.")

        st.divider()
        st.markdown("### Good to know")
        st.caption(
            "Only the two supplied documents are compared. English text only. "
            "The score reflects shared wording, not meaning, so paraphrased copying "
            "can score low."
        )
    return low, high


def render_results(analysis: dict, low: int, high: int) -> None:
    result = analysis["result"]
    similarity = result.similarity_percentage
    label = classify_similarity(similarity, low, high)
    text_color, background = LABEL_STYLES[label]

    st.subheader("Results")
    col_sim, col_diff, col_label = st.columns(3)
    col_sim.markdown(
        metric_card("Similarity", f"{similarity:.1f}%", "Cosine similarity of TF-IDF vectors"),
        unsafe_allow_html=True,
    )
    col_diff.markdown(
        metric_card("Difference", f"{result.difference_percentage:.1f}%", "100% minus similarity"),
        unsafe_allow_html=True,
    )
    col_label.markdown(
        '<div class="card">'
        '<div class="card-title">Classification</div>'
        f'<div class="badge" style="color:{text_color};background:{background}">{label}</div>'
        f'<div class="card-caption">Thresholds: {low}% / {high}%</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="meter" role="progressbar" aria-valuenow="{similarity:.0f}" '
        'aria-valuemin="0" aria-valuemax="100">'
        f'<div class="meter-fill" style="width:{similarity:.1f}%;background:{text_color}"></div>'
        "</div>"
        '<div class="meter-scale"><span>0%</span><span>50%</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )
    st.info(DISCLAIMER)

    st.subheader("Common keywords")
    if result.common_keywords:
        chips = "".join(
            f'<span class="chip">{html.escape(word)}</span>'
            for word, _ in result.common_keywords[:25]
        )
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(
            f"{result.shared_term_count} shared terms out of {result.vocabulary_size} "
            "unique terms across both documents (after preprocessing)."
        )
        with st.expander("See all shared keywords with weights"):
            table = pd.DataFrame(
                list(result.common_keywords), columns=["Keyword", "Avg. TF-IDF weight"]
            )
            table.index = range(1, len(table) + 1)
            st.dataframe(table)
    else:
        st.write("The documents share no keywords after preprocessing.")

    with st.expander("Preprocessed text"):
        left, right = st.columns(2)
        for column, name, text in (
            (left, analysis["name_a"], analysis["clean_a"]),
            (right, analysis["name_b"], analysis["clean_b"]),
        ):
            column.markdown(f"**{html.escape(name)}** ({len(text.split()):,} tokens)")
            column.markdown(f'<div class="textbox">{html.escape(text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="\U0001F50D", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_header()
    low, high = render_sidebar()
    render_how_it_works()

    resources_ready, message = load_nlp_resources()
    if not resources_ready:
        load_nlp_resources.clear()  # so the next interaction retries the download
        st.error(message)
        st.stop()

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        doc_a = render_input_panel("Document 1", "doc1")
    with col_b:
        doc_b = render_input_panel("Document 2", "doc2")

    st.write("")
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        analyze_clicked = st.button("Analyze similarity", type="primary", key="analyze")

    if analyze_clicked:
        analysis = run_analysis(doc_a, doc_b)
        if analysis is None:
            st.session_state.pop("analysis", None)
        else:
            st.session_state["analysis"] = analysis
            st.success("Analysis complete.")

    # Results live in session_state so moving the threshold sliders re-labels
    # them without re-running the whole analysis.
    if "analysis" in st.session_state:
        render_results(st.session_state["analysis"], low, high)

    st.markdown(
        '<div class="footer-note">Compares only the documents you provide. '
        "It does not check the internet.</div>",
        unsafe_allow_html=True,
    )


main()
