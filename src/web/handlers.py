"""Request handlers for the web interface: plain functions over JSON-like dicts.

Kept separate from the HTTP layer so they can be tested directly. Every input is
validated; failures raise :class:`ApiError`, which the server turns into a JSON
error response.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import torch

from src.agent import Agent, ConversationMemory
from src.agent.chat import make_generate_fn
from src.eval.attention import attention_weights
from src.eval.metrics import repetition_rate
from src.generation import beam_search, generate
from src.rag import TextRetriever

from .registry import ModelRegistry

MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 300
MAX_BEAM_NEW_TOKENS = 100
MAX_SESSIONS = 20
MAX_ATTENTION_TOKENS = 64
CHAT_CONTEXT_CHARS = 400
# \Z (not $) so a trailing newline cannot slip through the allow-list patterns.
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")
# One optional sub-directory level; no segment may start with a dot (no "..").
FIGURE_PATTERN = re.compile(r"^(?:[A-Za-z0-9_-][A-Za-z0-9_.-]*/)?[A-Za-z0-9_-][A-Za-z0-9_.-]*\.png\Z")
DOCUMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.txt\Z")


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- validation

def _string(payload: dict, key: str, *, max_len: int, required: bool = True) -> str | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise ApiError(400, f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise ApiError(400, f"'{key}' must be a string")
    if required and not value.strip():
        raise ApiError(400, f"'{key}' must not be empty")
    if len(value) > max_len:
        raise ApiError(400, f"'{key}' is longer than {max_len} characters")
    return value


def _integer(payload: dict, key: str, *, low: int, high: int, default: int | None = None) -> int | None:
    value = payload.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(400, f"'{key}' must be an integer")
    if not low <= value <= high:
        raise ApiError(400, f"'{key}' must be between {low} and {high}")
    return value


def _number(payload: dict, key: str, *, low: float, high: float, default: float | None = None) -> float | None:
    value = payload.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"'{key}' must be a number")
    if not low <= value <= high:
        raise ApiError(400, f"'{key}' must be between {low} and {high}")
    return float(value)


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Api:
    """All endpoint logic. One instance is shared by every request thread."""

    def __init__(self, root: str | Path, *, device: str = "cpu", cache_size: int = 2) -> None:
        self.root = Path(root).resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.data_dir = self.root / "data"
        self.registry = ModelRegistry(self.artifacts_dir, cache_size=cache_size, device=device)
        self._sessions: OrderedDict[str, Agent] = OrderedDict()
        self._retrievers: dict[tuple[str, ...], TextRetriever] = {}
        self._session_lock = threading.Lock()

    # ------------------------------------------------------------ helpers

    def _model(self, payload: dict):
        model_id = _string(payload, "model", max_len=200)
        try:
            return self.registry.get(model_id)
        except KeyError:
            raise ApiError(404, f"unknown model '{model_id}'") from None
        except Exception as error:  # noqa: BLE001
            raise ApiError(500, f"could not load model: {type(error).__name__}") from error

    @staticmethod
    def _unknown_characters(tokenizer, text: str) -> list[str]:
        known = tokenizer.token_to_id
        return sorted({character for character in text if character not in known})[:20]

    # ----------------------------------------------------------- endpoints

    def health(self) -> dict:
        return {"status": "ok", "models": len(self.registry.discover())}

    def models(self) -> dict:
        return {"models": self.registry.list_models()}

    def documents(self) -> dict:
        names = sorted(p.name for p in self.data_dir.glob("*.txt") if p.is_file()) if self.data_dir.is_dir() else []
        return {"documents": names}

    def generate(self, payload: dict) -> dict:
        prompt = _string(payload, "prompt", max_len=MAX_PROMPT_CHARS)
        mode = payload.get("mode", "greedy")
        if mode not in {"greedy", "sample", "beam"}:
            raise ApiError(400, "'mode' must be greedy, sample or beam")
        cap = MAX_BEAM_NEW_TOKENS if mode == "beam" else MAX_NEW_TOKENS
        max_new = _integer(payload, "max_new_tokens", low=1, high=cap, default=80)
        temperature = _number(payload, "temperature", low=0.05, high=5.0, default=1.0)
        top_k = _integer(payload, "top_k", low=1, high=100_000)
        top_p = _number(payload, "top_p", low=0.01, high=1.0)
        seed = _integer(payload, "seed", low=0, high=2**31 - 1)
        beam_width = _integer(payload, "beam_width", low=1, high=5, default=3)

        with self.registry.lock:
            loaded = self._model(payload)
            model, tokenizer = loaded.model, loaded.tokenizer
            context = model.config.context_length
            ids = tokenizer.encode(prompt)
            if not ids:
                raise ApiError(400, "'prompt' encodes to zero tokens")
            device = next(model.parameters()).device
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            if seed is not None:
                torch.manual_seed(seed)
            started = time.perf_counter()
            if mode == "beam":
                output = beam_search(model, tensor, context_length=context, max_new_tokens=max_new, beam_width=beam_width)
            else:
                output = generate(
                    model, tensor, context_length=context, max_new_tokens=max_new,
                    temperature=temperature, top_k=top_k, top_p=top_p, do_sample=(mode == "sample"),
                )
            seconds = time.perf_counter() - started
            new_ids = output[0, len(ids):].tolist()
            text = tokenizer.decode(new_ids)

        return {
            "model": loaded.id,
            "mode": mode,
            "text": text,
            "prompt_tokens": len(ids),
            "new_tokens": len(new_ids),
            "prompt_unknown_tokens": sum(1 for i in ids if i == 0),
            "unknown_characters": self._unknown_characters(tokenizer, prompt),
            "prompt_cropped": len(ids) > context,
            "context_length": context,
            "seconds": round(seconds, 4),
            "tokens_per_second": round(len(new_ids) / seconds, 1) if seconds > 0 else None,
            "repetition_3gram": round(repetition_rate(text), 4),
        }

    def chat(self, payload: dict) -> dict:
        message = _string(payload, "message", max_len=500)
        max_new = _integer(payload, "max_new_tokens", low=1, high=200, default=60)
        session_id = payload.get("session")
        if session_id is None:
            session_id = uuid.uuid4().hex
        elif not isinstance(session_id, str) or not SESSION_PATTERN.match(session_id):
            raise ApiError(400, "'session' must match [A-Za-z0-9_-]{1,64}")

        with self.registry.lock:
            loaded = self._model(payload)
            with self._session_lock:
                agent = self._sessions.get(session_id)
                if agent is None:
                    agent = Agent(lambda prompt: "", ConversationMemory(), max_context_chars=CHAT_CONTEXT_CHARS)
                    self._sessions[session_id] = agent
                    while len(self._sessions) > MAX_SESSIONS:
                        self._sessions.popitem(last=False)
                self._sessions.move_to_end(session_id)
            # Memory persists across turns; the model behind it may be switched.
            agent.generate = make_generate_fn(loaded.model, loaded.tokenizer, max_new_tokens=max_new)
            response = agent.respond(message)
            context = loaded.model.config.context_length
            last_prompt = response.prompts[-1] if response.prompts else None
            prompt_tokens = len(loaded.tokenizer.encode(last_prompt)) if last_prompt else None
            unknown = self._unknown_characters(loaded.tokenizer, message) if last_prompt else []

        return {
            "session": session_id,
            "model": loaded.id,
            "reply": response.text,
            "tool_calls": response.tool_calls,
            "tool_limit_reached": response.tool_limit_reached,
            "memory_turns": len(agent.memory),
            "memory": [{"role": turn.role, "text": turn.text} for turn in agent.memory.turns],
            "prompt": last_prompt,
            "prompt_tokens": prompt_tokens,
            "prompt_cropped": prompt_tokens is not None and prompt_tokens > context,
            "context_length": context,
            "unknown_characters": unknown,
        }

    def chat_reset(self, payload: dict) -> dict:
        session_id = payload.get("session")
        if not isinstance(session_id, str) or not SESSION_PATTERN.match(session_id):
            raise ApiError(400, "'session' must match [A-Za-z0-9_-]{1,64}")
        with self._session_lock:
            existed = self._sessions.pop(session_id, None) is not None
        return {"session": session_id, "reset": existed}

    def _retriever(self, names: list[str]) -> TextRetriever:
        key = tuple(names)
        if key not in self._retrievers:
            try:
                self._retrievers[key] = TextRetriever.from_files([self.data_dir / name for name in names])
            except (ValueError, OSError):  # empty/non-UTF-8/unreadable documents (UnicodeDecodeError is a ValueError)
                raise ApiError(400, "the selected documents are empty or not readable UTF-8 text") from None
        return self._retrievers[key]

    def rag(self, payload: dict) -> dict:
        question = _string(payload, "question", max_len=300)
        names = payload.get("documents", ["knowledge.txt"])
        if not isinstance(names, list) or not names or len(names) > 5 or not all(isinstance(n, str) for n in names):
            raise ApiError(400, "'documents' must be a list of 1 to 5 document names")
        available = set(self.documents()["documents"])
        for name in names:
            if not DOCUMENT_PATTERN.match(name) or name not in available:
                raise ApiError(400, f"unknown document '{name[:60]}'")
        top_k = _integer(payload, "top_k", low=1, high=5, default=3)
        max_new = _integer(payload, "max_new_tokens", low=1, high=150, default=60)

        with self.registry.lock:
            loaded = self._model(payload)
            model, tokenizer = loaded.model, loaded.tokenizer
            context = model.config.context_length
            default_chars = max(16, context // 2)  # the v1 behaviour of src.rag.query
            context_chars = _integer(payload, "context_chars", low=16, high=4000, default=default_chars)
            retriever = self._retriever(sorted(set(names)))
            prompt, sources = retriever.prompt(question, top_k=top_k, max_context_chars=context_chars)
            device = next(model.parameters()).device

            def answer(text: str) -> tuple[str, int]:
                ids = tokenizer.encode(text)
                if not ids:
                    raise ApiError(400, "prompt encodes to zero tokens")
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                output = generate(model, tensor, context_length=context, max_new_tokens=max_new)
                return tokenizer.decode(output[0, len(ids):].tolist()), len(ids)

            with_context, with_tokens = answer(prompt)
            closed_prompt = f"Question: {question}\nAnswer:"
            closed_book, closed_tokens = answer(closed_prompt)

        return {
            "model": loaded.id,
            "sources": [
                {"source": Path(s.source).name, "chunk": s.index, "score": round(s.score, 3), "text": s.text[:300]}
                for s in sources
            ],
            "with_context": {"prompt": prompt, "prompt_tokens": with_tokens, "answer": with_context, "prompt_cropped": with_tokens > context},
            "closed_book": {"prompt": closed_prompt, "prompt_tokens": closed_tokens, "answer": closed_book},
            "context_length": context,
            "context_chars": context_chars,
            "unknown_characters": self._unknown_characters(tokenizer, prompt),
        }

    def attention(self, payload: dict) -> dict:
        prompt = _string(payload, "prompt", max_len=MAX_PROMPT_CHARS)
        with self.registry.lock:
            loaded = self._model(payload)
            model, tokenizer = loaded.model, loaded.tokenizer
            ids = tokenizer.encode(prompt)
            if not ids:
                raise ApiError(400, "'prompt' encodes to zero tokens")
            limit = min(MAX_ATTENTION_TOKENS, model.config.context_length)
            cropped = len(ids) > limit
            ids = ids[-limit:]
            device = next(model.parameters()).device
            weights = attention_weights(model, torch.tensor([ids], dtype=torch.long, device=device)).cpu()
        return {
            "model": loaded.id,
            "tokens": [tokenizer.decode([i]) for i in ids],
            "cropped": cropped,
            "layers": int(weights.shape[0]),
            "heads": int(weights.shape[1]),
            "weights": weights.round(decimals=3).tolist(),
        }

    def results(self) -> dict:
        comparison = _read_json(self.artifacts_dir / "tick-204" / "comparison.json")
        v1 = []
        if isinstance(comparison, list):
            for run in comparison:
                greedy = next((s for s in run.get("samples", []) if str(s.get("mode", "")).startswith("greedy")), {})
                v1.append({
                    "tokenizer": run.get("tokenizer"),
                    "vocabulary_size": run.get("vocabulary_size"),
                    "parameters": run.get("parameters"),
                    "characters_per_token": run.get("characters_per_token"),
                    "validation_bits_per_character": run.get("validation_bits_per_character"),
                    "validation_perplexity_per_character": run.get("validation_perplexity_per_character"),
                    "greedy_repetition": greedy.get("repetition_3gram"),
                    "greedy_characters_per_second": greedy.get("characters_per_second"),
                })
        figures: set[str] = set()
        for root in self._figure_roots():
            if root.is_dir():
                for path in root.rglob("*.png"):
                    relative = path.relative_to(root)
                    name = relative.as_posix()
                    # Only list what /figures/<name> will actually serve (pattern, depth and no symlink escape).
                    if len(relative.parts) <= 2 and self.figure_path(name) is not None:
                        figures.add(name)
        return {
            "v1_comparison": v1,
            "figures": sorted(figures),
            "v2_summary": _read_json(self.artifacts_dir / "v2" / "summary.json"),
        }

    def _figure_roots(self) -> list[Path]:
        return [self.artifacts_dir / "tick-403", self.artifacts_dir / "v2" / "figures"]

    def figure_path(self, name: str) -> Path | None:
        """Resolve a figure (``file.png`` or ``subdir/file.png``) inside the allow-listed roots only."""
        if not FIGURE_PATTERN.match(name):
            return None
        for root in self._figure_roots():
            candidate = root / name
            if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
                return candidate
        return None
