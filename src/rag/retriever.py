"""Dependency-free lexical retrieval over local UTF-8 text documents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    index: int
    text: str
    score: float


class TextRetriever:
    """Chunk documents by words and rank chunks with normalized term overlap."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        if not chunks:
            raise ValueError("at least one non-empty chunk is required")
        self._chunks = chunks
        self._term_counts = [Counter(_terms(chunk.text)) for chunk in chunks]

    @classmethod
    def from_files(cls, paths: list[str | Path], *, chunk_words: int = 120, overlap_words: int = 24) -> "TextRetriever":
        if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
            raise ValueError("chunk_words must be positive and overlap smaller than it")
        chunks: list[RetrievedChunk] = []
        stride = chunk_words - overlap_words
        for path in paths:
            words = Path(path).read_text(encoding="utf-8").split()
            for start in range(0, len(words), stride):
                text = " ".join(words[start : start + chunk_words])
                if text:
                    chunks.append(RetrievedChunk(str(path), len(chunks), text, 0.0))
        return cls(chunks)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = Counter(_terms(query))
        scored = []
        for chunk, counts in zip(self._chunks, self._term_counts):
            overlap = sum(min(query_terms[term], counts[term]) for term in query_terms)
            score = overlap / max(1, sum(query_terms.values()))
            scored.append(RetrievedChunk(chunk.source, chunk.index, chunk.text, score))
        return sorted(scored, key=lambda item: (-item.score, item.source, item.index))[:top_k]

    def prompt(self, question: str, *, top_k: int = 3, max_context_chars: int = 2_000) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(f"[Source: {item.source}]\n{item.text}" for item in retrieved)
        return f"Use only the supplied context to answer. If it is insufficient, say so.\n\nContext:\n{context[:max_context_chars]}\n\nQuestion: {question}\nAnswer:", retrieved
