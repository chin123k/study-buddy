"""Streamlit app for AI Slide Explainer."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Dict, List

import streamlit as st

from ai_engine import generate_explanation_package, generate_quiz, summarize_slide_images
from document_parser import parse_document
from redis_cache import build_cache_key, create_redis_client, get_json, set_json, touch_key
from retrieval_engine import build_vector_store, retrieve_relevant_chunks

st.set_page_config(page_title="AI Slide Explainer", layout="wide")


def show_ai_error(error: Exception) -> None:
    """Display friendly AI error messages in the UI."""
    message = str(error)
    lowered = message.lower()

    if "could not connect to ollama" in lowered:
        st.error("Cannot connect to Ollama server.")
        st.info("Start Ollama and make sure the server URL is correct in sidebar settings.")
        return

    st.error(f"AI request failed: {message}")


@st.cache_data(show_spinner=False)
def cached_parse_document(file_name: str, file_bytes: bytes) -> List[Dict[str, object]]:
    """Cache extracted pages/slides so parsing happens only once per file."""
    return parse_document(file_name=file_name, file_bytes=file_bytes)


@st.cache_data(show_spinner=False)
def cached_build_vector_store(pages: List[str]):
    """Cache in-memory vectors built from page/slide chunks."""
    return build_vector_store(pages)


@st.cache_resource(show_spinner=False)
def cached_redis_client():
    """Create a shared Redis client (or None when unavailable)."""
    return create_redis_client()


def _get_cache_key(file_bytes: bytes, index: int) -> str:
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{file_hash}:{index}"


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "explanation_cache" not in st.session_state:
        st.session_state.explanation_cache = {}
    if "quiz_cache" not in st.session_state:
        st.session_state.quiz_cache = {}
    if "image_summary_cache" not in st.session_state:
        st.session_state.image_summary_cache = {}
    if "last_model_fingerprint" not in st.session_state:
        st.session_state.last_model_fingerprint = ""


_init_session_state()
redis_client = cached_redis_client()
cache_ttl_seconds = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))

st.title("AI Slide Explainer")
st.caption("Upload a PDF or PPTX file, then get simple explanations and questions for each slide/page.")

with st.sidebar:
    st.header("Settings")
    ollama_base_url = st.text_input(
        "Ollama Server URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Default local Ollama API URL.",
    )
    ollama_model = st.text_input(
        "Ollama Model",
        value=os.getenv("OLLAMA_MODEL", "gemma:2b"),
        help="Example: llama3.1:8b, mistral, phi3",
    )
    enable_image_analysis = st.checkbox(
        "Enable Image Analysis",
        value=True,
        help="Uses a vision model in Ollama to summarize images in the current slide/page.",
    )
    vision_model = st.text_input(
        "Ollama Vision Model",
        value=os.getenv("OLLAMA_VISION_MODEL", "llava:7b"),
        help="Example: llava:7b, llava:13b, minicpm-v",
    )
    st.caption(
        f"Redis cache: {'connected' if redis_client else 'not connected'} | "
        f"TTL: {cache_ttl_seconds}s"
    )

    model_fingerprint_src = (
        f"{ollama_base_url.strip()}|{ollama_model.strip()}|"
        f"{vision_model.strip()}|{int(enable_image_analysis)}"
    )
    if model_fingerprint_src.strip("|"):
        fingerprint = hashlib.md5(model_fingerprint_src.encode("utf-8")).hexdigest()
        if st.session_state.last_model_fingerprint != fingerprint:
            st.session_state.explanation_cache = {}
            st.session_state.quiz_cache = {}
            st.session_state.image_summary_cache = {}
            st.session_state.last_model_fingerprint = fingerprint

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "pptx"],
    help="Supported formats: PDF, PPTX",
)

if not uploaded_file:
    st.info("Upload a PDF or PPTX file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    records = cached_parse_document(uploaded_file.name, file_bytes)
except Exception as error:
    st.error(f"Could not parse the document: {error}")
    st.stop()

if not records:
    st.warning("No text could be extracted from this file.")
    st.stop()

page_texts = [str(item.get("text", "")) for item in records]
vector_store = cached_build_vector_store(page_texts)

with st.sidebar:
    st.header("Navigation")
    labels = [f"Page/Slide {i + 1}" for i in range(len(records))]
    selected_label = st.radio("Choose page/slide", labels)

selected_index = labels.index(selected_label)
selected_record = records[selected_index]
selected_text = str(selected_record.get("text", ""))
selected_images = selected_record.get("images", [])
if not isinstance(selected_images, list):
    selected_images = []
selected_preview_image = selected_record.get("preview_image")
selected_cache_key = _get_cache_key(file_bytes, selected_index)
session_id = str(st.session_state.session_id)
explain_redis_key = build_cache_key(
    session_id=session_id,
    namespace="explain",
    item_key=f"{selected_cache_key}:{ollama_model.strip()}:{vision_model.strip()}:{int(enable_image_analysis)}",
)
vision_redis_key = build_cache_key(
    session_id=session_id,
    namespace="vision",
    item_key=f"{selected_cache_key}:{vision_model.strip()}",
)
quiz_redis_key = build_cache_key(
    session_id=session_id,
    namespace="quiz",
    item_key=f"{selected_cache_key}:{ollama_model.strip()}",
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Original Slide Text")
    st.text_area("Extracted text", value=selected_text, height=320)

    st.markdown("### Current Slide Preview")
    if selected_preview_image:
        st.image(selected_preview_image, caption=selected_label, width=280)
    else:
        st.info("Preview not available for this slide.")

with right_col:
    st.subheader("AI Explanation + Q&A")

    if not ollama_model.strip():
        st.warning("Enter an Ollama model name in the sidebar.")
        st.stop()

    explanation_cache: Dict[str, Dict[str, object]] = st.session_state.explanation_cache

    if selected_cache_key not in explanation_cache:
        redis_hit = get_json(redis_client, explain_redis_key)
        if isinstance(redis_hit, dict):
            explanation_cache[selected_cache_key] = redis_hit
            touch_key(redis_client, explain_redis_key, cache_ttl_seconds)

    if selected_cache_key not in explanation_cache:
        relevant_chunks = retrieve_relevant_chunks(selected_text, vector_store, top_k=3)
        context_parts = [item[0].text for item in relevant_chunks if item[1] > 0]

        if enable_image_analysis and selected_images:
            image_summary_cache: Dict[str, str] = st.session_state.image_summary_cache
            image_summary_key = f"{selected_cache_key}:vision"

            if image_summary_key not in image_summary_cache:
                redis_vision = get_json(redis_client, vision_redis_key)
                if isinstance(redis_vision, str):
                    image_summary_cache[image_summary_key] = redis_vision
                    touch_key(redis_client, vision_redis_key, cache_ttl_seconds)

            if image_summary_key not in image_summary_cache:
                try:
                    with st.spinner("Analyzing slide images..."):
                        image_summary_cache[image_summary_key] = summarize_slide_images(
                            images=selected_images,
                            vision_model=vision_model.strip(),
                            base_url=ollama_base_url.strip(),
                        )
                except Exception as error:
                    image_summary_cache[image_summary_key] = ""
                    st.warning(f"Image analysis skipped: {error}")

                if image_summary_cache[image_summary_key]:
                    set_json(
                        redis_client,
                        vision_redis_key,
                        image_summary_cache[image_summary_key],
                        ttl_seconds=cache_ttl_seconds,
                    )

            image_summary = image_summary_cache.get(image_summary_key, "").strip()
            if image_summary:
                context_parts.append(f"Image Insights:\n{image_summary}")

        context_text = "\n\n".join(context_parts)

        with st.spinner("Generating explanation and questions..."):
            try:
                explanation_cache[selected_cache_key] = generate_explanation_package(
                    slide_text=selected_text,
                    api_key="",
                    context_text=context_text,
                    model=ollama_model.strip(),
                    base_url=ollama_base_url.strip(),
                )
                set_json(
                    redis_client,
                    explain_redis_key,
                    explanation_cache[selected_cache_key],
                    ttl_seconds=cache_ttl_seconds,
                )
            except Exception as error:
                show_ai_error(error)
                st.stop()

    result = explanation_cache[selected_cache_key]

    st.markdown("### Explanation")
    st.write(result.get("explanation", "No explanation generated."))

    key_points = result.get("key_points", [])
    if isinstance(key_points, list) and key_points:
        st.markdown("### Key Points")
        for point in key_points:
            st.markdown(f"- {point}")

    st.markdown("### Questions")
    questions = result.get("questions", [])
    if isinstance(questions, list) and questions:
        for idx, question in enumerate(questions, start=1):
            st.markdown(f"{idx}. {question}")
    else:
        st.write("No questions generated.")

    st.markdown("### Answers")
    answers = result.get("answers", [])
    if isinstance(answers, list) and answers:
        for idx, answer in enumerate(answers, start=1):
            st.markdown(f"{idx}. {answer}")
    else:
        st.write("No answers generated.")

st.divider()
st.subheader("Generate Quiz (Optional)")

if st.button("Generate Quiz", use_container_width=True):
    if not ollama_model.strip():
        st.warning("Enter an Ollama model name in the sidebar to generate a quiz.")
    else:
        quiz_cache: Dict[str, str] = st.session_state.quiz_cache
        if selected_cache_key not in quiz_cache:
            redis_quiz = get_json(redis_client, quiz_redis_key)
            if isinstance(redis_quiz, str):
                quiz_cache[selected_cache_key] = redis_quiz
                touch_key(redis_client, quiz_redis_key, cache_ttl_seconds)

        if selected_cache_key not in quiz_cache:
            with st.spinner("Generating 5 MCQs for this slide..."):
                try:
                    quiz_cache[selected_cache_key] = generate_quiz(
                        slide_text=selected_text,
                        api_key="",
                        model=ollama_model.strip(),
                        base_url=ollama_base_url.strip(),
                    )
                    set_json(
                        redis_client,
                        quiz_redis_key,
                        quiz_cache[selected_cache_key],
                        ttl_seconds=cache_ttl_seconds,
                    )
                except Exception as error:
                    show_ai_error(error)
                    st.stop()

        st.markdown(quiz_cache[selected_cache_key])
