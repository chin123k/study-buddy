"""Simple in-memory retrieval pipeline for slide/page chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ChunkRecord:
    chunk_id: str
    source_index: int
    text: str
    vector: Dict[str, float]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _embed_text(text: str) -> Dict[str, float]:
    """Create a normalized bag-of-words vector."""
    tokens = _tokenize(text)
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = float(sum(counts.values()))
    return {token: count / total for token, count in counts.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(value * vec_b.get(token, 0.0) for token, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    """Split text into overlapping character chunks."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0

    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = max(0, end - overlap)

    return chunks


def build_vector_store(pages: List[str]) -> List[ChunkRecord]:
    """Create chunk records and vectors for all pages/slides."""
    store: List[ChunkRecord] = []

    for page_index, page_text in enumerate(pages):
        chunks = chunk_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            store.append(
                ChunkRecord(
                    chunk_id=f"{page_index}-{chunk_idx}",
                    source_index=page_index,
                    text=chunk,
                    vector=_embed_text(chunk),
                )
            )

    return store


def retrieve_relevant_chunks(
    query_text: str,
    store: List[ChunkRecord],
    top_k: int = 3,
) -> List[Tuple[ChunkRecord, float]]:
    """Retrieve the top-k most relevant chunks for a query."""
    query_vector = _embed_text(query_text)
    scored: List[Tuple[ChunkRecord, float]] = []

    for record in store:
        score = _cosine_similarity(query_vector, record.vector)
        scored.append((record, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
