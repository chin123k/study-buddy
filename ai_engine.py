"""Ollama integration and response parsing for AI Slide Explainer."""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "gemma:2b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "llava:7b"
DEFAULT_TEXT_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "240"))
DEFAULT_VISION_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "300"))
DEFAULT_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))


def _is_timeout_exception(error: Exception) -> bool:
    if isinstance(error, (socket.timeout, TimeoutError)):
        return True

    if isinstance(error, URLError) and isinstance(error.reason, (socket.timeout, TimeoutError)):
        return True

    return "timed out" in str(error).lower() or "timeout" in str(error).lower()


def _is_runner_crash_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "runner process has terminated" in lowered
        or "exit status 2" in lowered
        or "unable to allocate cpu buffer" in lowered
        or "error loading model" in lowered
    )


def _clean_markdown(text: str) -> str:
    """Remove common markdown wrappers that break simple parsing."""
    cleaned = text.replace("**", "").replace("__", "").strip()
    return cleaned


def _build_explainer_prompt(slide_text: str, context_text: str = "") -> str:
    context_block = ""
    if context_text.strip():
        context_block = (
            "Related Context from document (for better understanding):\n"
            f"{context_text.strip()}\n\n"
        )

    return (
        "Explain the following slide content in very simple language so a student can "
        "understand it easily.\n"
        "Then generate 3 important questions from the content and provide clear answers.\n"
        "If the slide already contains direct questions, answer them naturally inside the explanation.\n"
        "Do not create a separate label like 'Questions found in the slide'.\n"
        "The 3 generated questions in the Questions section should be separate study questions.\n"
        "Do not copy slide questions verbatim into the generated Questions section.\n"
        "Do not use markdown symbols like ** or bullet points.\n\n"
        f"{context_block}"
        "Slide Content:\n"
        f"{slide_text}\n\n"
        "Format the response exactly as:\n\n"
        "Explanation:\n"
        "...\n\n"
        "Questions:\n"
        "1.\n"
        "2.\n"
        "3.\n\n"
        "Answers:\n"
        "1.\n"
        "2.\n"
        "3."
    )


def _build_qa_repair_prompt(slide_text: str, existing_output: str) -> str:
    return (
        "The previous response did not include complete Questions and Answers.\n"
        "Regenerate only the missing parts and return full final output in this exact format:\n\n"
        "Explanation:\n"
        "...\n\n"
        "Questions:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n\n"
        "Answers:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n\n"
        "No markdown symbols.\n\n"
        "Slide Content:\n"
        f"{slide_text}\n\n"
        "Previous Incomplete Output:\n"
        f"{existing_output}\n"
    )


def _build_answer_only_prompt(
    slide_text: str,
    explanation: str,
    questions: List[str],
) -> str:
    numbered_questions = "\n".join(
        f"{idx}. {question}" for idx, question in enumerate(questions, start=1)
    )

    return (
        "Provide clear and short answers for the following questions.\n"
        "Return exactly this format:\n\n"
        "Answers:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n\n"
        "No markdown symbols.\n\n"
        "Slide Content:\n"
        f"{slide_text}\n\n"
        "Current Explanation:\n"
        f"{explanation}\n\n"
        "Questions:\n"
        f"{numbered_questions}\n"
    )


def _build_quiz_prompt(slide_text: str) -> str:
    return (
        "Create exactly 5 multiple-choice questions from the following slide content.\n"
        "Each question must have exactly 4 options (A, B, C, D).\n"
        "For every question, include one line: Correct Answer: <option letter> - <short reason>.\n"
        "Do not use markdown symbols like **.\n\n"
        "Format exactly like this:\n"
        "1. <question>\n"
        "A. ...\n"
        "B. ...\n"
        "C. ...\n"
        "D. ...\n"
        "Correct Answer: A - ...\n\n"
        "2. <question>\n"
        "A. ...\n"
        "B. ...\n"
        "C. ...\n"
        "D. ...\n"
        "Correct Answer: B - ...\n\n"
        "... up to 5 questions.\n\n"
        "Slide Content:\n"
        f"{slide_text}\n"
    )


def _build_quiz_repair_prompt(slide_text: str, existing_output: str) -> str:
    return (
        "The previous quiz output is incomplete.\n"
        "Regenerate exactly 5 MCQs with options A-D and a Correct Answer line for EACH question.\n"
        "No markdown symbols.\n\n"
        "Slide Content:\n"
        f"{slide_text}\n\n"
        "Previous Incomplete Quiz:\n"
        f"{existing_output}\n"
    )


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Call local Ollama server and return generated text."""
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Keep text model warm for a while to avoid repeated load/unload crashes.
        "keep_alive": "10m",
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    parsed: Dict[str, object] = {}
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=DEFAULT_TEXT_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
            break
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if _is_runner_crash_error(body):
                raise RuntimeError(
                    "Ollama model crashed while loading/running (likely low RAM/VRAM). "
                    "Use a smaller model, close other heavy apps, or disable image analysis."
                ) from error
            raise RuntimeError(f"Ollama HTTP {error.code}: {body}") from error
        except URLError as error:
            if _is_timeout_exception(error):
                if attempt < DEFAULT_MAX_RETRIES:
                    time.sleep(1.0 + attempt)
                    continue
                raise RuntimeError(
                    "Ollama request timed out after "
                    f"{DEFAULT_TEXT_TIMEOUT_SECONDS}s. Increase OLLAMA_REQUEST_TIMEOUT_SECONDS, "
                    "or use a smaller/faster model."
                ) from error
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running on "
                f"{base_url}."
            ) from error
        except Exception as error:
            message = str(error)
            if _is_runner_crash_error(message):
                raise RuntimeError(
                    "Ollama model crashed while loading/running (likely low RAM/VRAM). "
                    "Use a smaller model, close other heavy apps, or disable image analysis."
                ) from error
            if _is_timeout_exception(error):
                if attempt < DEFAULT_MAX_RETRIES:
                    time.sleep(1.0 + attempt)
                    continue
                raise RuntimeError(
                    "Ollama request timed out after "
                    f"{DEFAULT_TEXT_TIMEOUT_SECONDS}s. Increase OLLAMA_REQUEST_TIMEOUT_SECONDS, "
                    "or use a smaller/faster model."
                ) from error
            raise RuntimeError(f"Ollama request failed: {error}") from error

    output = str(parsed.get("response", "")).strip()
    if not output:
        raise RuntimeError("Ollama returned an empty response. Please try again.")

    return output


def _call_ollama_vision(
    prompt: str,
    images: List[bytes],
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Call Ollama chat endpoint with images using a vision-capable model."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    encoded_images = [base64.b64encode(blob).decode("utf-8") for blob in images]

    payload = {
        "model": model,
        "stream": False,
        # Free model from memory after response to reduce RAM pressure.
        "keep_alive": "0s",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": encoded_images,
            }
        ],
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    parsed: Dict[str, object] = {}
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=DEFAULT_VISION_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
            break
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama vision HTTP {error.code}: {body}") from error
        except URLError as error:
            if _is_timeout_exception(error):
                if attempt < DEFAULT_MAX_RETRIES:
                    time.sleep(1.0 + attempt)
                    continue
                raise RuntimeError(
                    "Ollama vision request timed out after "
                    f"{DEFAULT_VISION_TIMEOUT_SECONDS}s. Increase OLLAMA_VISION_TIMEOUT_SECONDS, "
                    "or use a smaller vision model."
                ) from error
            raise RuntimeError(
                "Could not connect to Ollama vision endpoint. Make sure Ollama is running on "
                f"{base_url}."
            ) from error
        except Exception as error:
            if _is_timeout_exception(error):
                if attempt < DEFAULT_MAX_RETRIES:
                    time.sleep(1.0 + attempt)
                    continue
                raise RuntimeError(
                    "Ollama vision request timed out after "
                    f"{DEFAULT_VISION_TIMEOUT_SECONDS}s. Increase OLLAMA_VISION_TIMEOUT_SECONDS, "
                    "or use a smaller vision model."
                ) from error
            raise RuntimeError(f"Ollama vision request failed: {error}") from error

    message = parsed.get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError("Ollama vision returned an empty response.")
    return _clean_markdown(content)


def summarize_slide_images(
    images: List[bytes],
    vision_model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Summarize visual information found in slide images."""
    if not images:
        return ""

    # One image is enough for summary and keeps memory usage lower.
    selected_images = images[:1]
    prompt = (
        "Analyze these slide images and summarize key visual information in simple language. "
        "Include any visible text, charts, diagrams, code snippets, or important objects. "
        "Return 4 to 6 short bullet points."
    )

    try:
        return _call_ollama_vision(
            prompt=prompt,
            images=selected_images,
            model=vision_model,
            base_url=base_url,
        )
    except RuntimeError as error:
        message = str(error)
        lowered = message.lower()
        if "unable to allocate cpu buffer" in lowered or "error loading model" in lowered:
            raise RuntimeError(
                "Vision model could not load due to low system memory. "
                "Use a smaller vision model in sidebar (for example moondream), "
                "or disable image analysis."
            ) from error
        raise


def _parse_numbered_lines(section_text: str) -> List[str]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    cleaned: List[str] = []

    for line in lines:
        cleaned_line = re.sub(r"^\d+[\.)]\s*", "", line).strip()
        if cleaned_line:
            cleaned.append(cleaned_line)

    return cleaned


def _extract_fallback_questions(raw_response: str) -> List[str]:
    """Best-effort extraction when model misses the Questions section label."""
    questions: List[str] = []
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if "found in the slide" in lowered:
            continue
        plain = re.sub(r"^\d+[\.)]\s*", "", stripped)
        if "?" in plain and plain not in questions:
            questions.append(plain)
        if len(questions) >= 5:
            break
    return questions


def _extract_answers_only(raw_response: str) -> List[str]:
    raw_response = _clean_markdown(raw_response)
    answers_match = re.search(
        r"Answers:\s*(.*)$",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    target = answers_match.group(1) if answers_match else raw_response
    return _parse_numbered_lines(target)


def _has_missing_answers(answers: List[str]) -> bool:
    if not answers:
        return True

    for answer in answers:
        if not answer.strip() or "answer not provided" in answer.lower():
            return True

    return False


def _is_quiz_complete(quiz_text: str, expected_questions: int = 5) -> bool:
    numbered_questions = re.findall(r"(?m)^\s*\d+[\.)]\s+", quiz_text)
    answer_lines = re.findall(r"(?mi)^\s*Correct\s*Answer\s*:\s*", quiz_text)
    return len(numbered_questions) >= expected_questions and len(answer_lines) >= expected_questions


def parse_explainer_response(raw_response: str) -> Dict[str, object]:
    """Parse model output into explanation, questions, and answers."""
    raw_response = _clean_markdown(raw_response)
    explanation = ""
    questions: List[str] = []
    answers: List[str] = []

    explanation_match = re.search(
        r"Explanation:\s*(.*?)\s*Questions:",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    questions_match = re.search(
        r"Questions:\s*(.*?)\s*Answers:",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    answers_match = re.search(
        r"Answers:\s*(.*)$",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if explanation_match:
        explanation = explanation_match.group(1).strip()
    if questions_match:
        questions = _parse_numbered_lines(questions_match.group(1))
    if answers_match:
        answers = _parse_numbered_lines(answers_match.group(1))

    if not questions:
        questions = _extract_fallback_questions(raw_response)[:3]

    if questions and len(answers) < len(questions):
        missing = len(questions) - len(answers)
        answers.extend(["Answer not provided in model output."] * missing)

    if not explanation:
        explanation = raw_response.strip()

    key_points = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", explanation)
        if sentence.strip()
    ][:4]

    return {
        "raw": raw_response,
        "explanation": explanation,
        "questions": questions,
        "answers": answers,
        "key_points": key_points,
    }


def generate_explanation_package(
    slide_text: str,
    api_key: str = "",
    context_text: str = "",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, object]:
    """Generate explanation + Q&A for a single slide/page."""
    prompt = _build_explainer_prompt(slide_text, context_text=context_text)
    raw_output = _call_ollama(prompt=prompt, model=model, base_url=base_url)
    parsed = parse_explainer_response(raw_output)

    questions = parsed.get("questions", [])
    answers = parsed.get("answers", [])
    needs_repair = not questions or not answers or len(answers) < len(questions)

    if needs_repair:
        repair_prompt = _build_qa_repair_prompt(slide_text=slide_text, existing_output=raw_output)
        repaired_output = _call_ollama(prompt=repair_prompt, model=model, base_url=base_url)
        repaired_parsed = parse_explainer_response(repaired_output)

        repaired_questions = repaired_parsed.get("questions", [])
        repaired_answers = repaired_parsed.get("answers", [])
        if repaired_questions and repaired_answers:
            parsed = repaired_parsed

    questions = parsed.get("questions", [])
    answers = parsed.get("answers", [])
    explanation = str(parsed.get("explanation", "")).strip()

    if isinstance(questions, list) and isinstance(answers, list):
        if questions and _has_missing_answers(answers):
            answer_prompt = _build_answer_only_prompt(
                slide_text=slide_text,
                explanation=explanation,
                questions=questions,
            )
            answer_output = _call_ollama(prompt=answer_prompt, model=model, base_url=base_url)
            recovered_answers = _extract_answers_only(answer_output)

            if recovered_answers:
                parsed["answers"] = recovered_answers[: len(questions)]
                if len(parsed["answers"]) < len(questions):
                    missing = len(questions) - len(parsed["answers"])
                    parsed["answers"].extend(["Answer unavailable."] * missing)

    return parsed


def generate_quiz(
    slide_text: str,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Generate 5 MCQs from a slide/page."""
    prompt = _build_quiz_prompt(slide_text)
    quiz_output = _clean_markdown(_call_ollama(prompt=prompt, model=model, base_url=base_url))

    if not _is_quiz_complete(quiz_output, expected_questions=5):
        repair_prompt = _build_quiz_repair_prompt(slide_text=slide_text, existing_output=quiz_output)
        quiz_output = _clean_markdown(
            _call_ollama(prompt=repair_prompt, model=model, base_url=base_url)
        )

    return quiz_output
