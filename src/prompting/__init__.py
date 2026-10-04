"""Prompting strategies for Project Ascent."""

from .prompts import (
    COT_CUE,
    EXEMPLARS,
    STYLES,
    Exemplar,
    PromptTask,
    build_prompt,
    clip_generation,
    extract_answer,
    load_tasks,
    normalize,
    score_answer,
)

__all__ = [
    "COT_CUE", "EXEMPLARS", "STYLES", "Exemplar", "PromptTask", "build_prompt",
    "clip_generation", "extract_answer", "load_tasks", "normalize", "score_answer",
]
