"""Tests for the localhost web interface (TICK-701 backend, TICK-702 static checks).

A real server is started once for the module on an ephemeral port (``port=0``),
against a temporary project root that holds two tiny random-weight checkpoints
(character and BPE), one corrupt checkpoint, small documents and fake figures.
No trained artifacts from the real project are read or modified.
"""

from __future__ import annotations

import http.client
import json
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the project requires PyTorch
    torch = None

if torch is not None:
    from src.generation import generate
    from src.model import GPTConfig, MiniGPT
    from src.tokenizer import BPETokenizer, CharacterTokenizer
    from src.web.registry import ModelRegistry
    from src.web.server import _hostname, create_server

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / "src" / "web" / "static" / "index.html"

# Lower-case only on purpose: "Z" (and every capital letter) is an unknown character.
CORPUS = "abcdefghijklmnopqrstuvwxyz .:\n(),?'"
KNOWLEDGE = (
    "Mars is known as the red planet because iron minerals in its soil oxidize. "
    "Jupiter is the largest planet in the Solar System. A triangle has three sides.\n"
)
PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"
CONTEXT = 32


def write_checkpoint(directory: Path, tokenizer, *, seed: int):
    torch.manual_seed(seed)
    model = MiniGPT(GPTConfig(tokenizer.vocabulary_size, context_length=CONTEXT, embedding_dim=16,
                              num_heads=2, num_layers=2, dropout=0.0))
    model.eval()
    if isinstance(tokenizer, CharacterTokenizer):
        payload = {"type": "character", "vocabulary": list(tokenizer.vocabulary)}
    else:
        payload = {"type": "bpe", "vocabulary": list(tokenizer.vocabulary),
                   "merges": [list(pair) for pair in tokenizer.merges]}
    directory.mkdir(parents=True)
    torch.save({"model_config": model.config.to_dict(), "model_state": model.state_dict(), "completed_steps": 5,
                "tokenizer": payload, "vocabulary": tokenizer.vocabulary}, directory / "checkpoint.pt")
    return model


class Fixture:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        artifacts = self.root / "artifacts"
        data = self.root / "data"
        self.char_tokenizer = CharacterTokenizer.from_text(CORPUS)
        self.bpe_tokenizer = BPETokenizer.train(CORPUS * 5, max_merges=10)
        self.char_model = write_checkpoint(artifacts / "tiny" / "char", self.char_tokenizer, seed=1)
        self.bpe_model = write_checkpoint(artifacts / "tiny" / "bpe", self.bpe_tokenizer, seed=2)
        (artifacts / "broken").mkdir(parents=True)
        (artifacts / "broken" / "checkpoint.pt").write_bytes(b"this is not a checkpoint")

        comparison = [
            {"tokenizer": name, "vocabulary_size": 66, "parameters": 1000, "characters_per_token": 1.0,
             "validation_bits_per_character": bits, "validation_perplexity_per_character": 2 ** bits,
             "samples": [{"mode": "greedy", "repetition_3gram": 0.97, "characters_per_second": 3429.0},
                         {"mode": "top-k sampling (k=20, T=0.8)", "repetition_3gram": 0.12, "characters_per_second": 3144.0}]}
            for name, bits in (("character", 3.45), ("bpe", 3.09))
        ]
        (artifacts / "tick-204").mkdir(parents=True)
        self.comparison_path = artifacts / "tick-204" / "comparison.json"
        self.comparison_text = json.dumps(comparison)
        self.comparison_path.write_text(self.comparison_text, encoding="utf-8")
        (artifacts / "tick-403" / "sub").mkdir(parents=True)
        (artifacts / "tick-403" / "loss.png").write_bytes(PNG)
        (artifacts / "tick-403" / "sub" / "attn.png").write_bytes(PNG)
        (self.root / "secret.png").write_bytes(b"SECRET")
        try:
            (artifacts / "tick-403" / "leak.png").symlink_to(self.root / "secret.png")
            self.symlinks = True
        except (OSError, NotImplementedError):
            self.symlinks = False

        data.mkdir()
        (data / "knowledge.txt").write_text(KNOWLEDGE, encoding="utf-8")
        (data / "other.txt").write_text("Water boils at one hundred degrees Celsius at sea level.\n", encoding="utf-8")
        (data / "empty.txt").write_text("", encoding="utf-8")
        (data / "secret.json").write_text("{}", encoding="utf-8")

        self.server = create_server(self.root, port=0, quiet=True)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()


FIXTURE: Fixture | None = None


def setUpModule() -> None:
    global FIXTURE
    if torch is None:
        raise unittest.SkipTest("PyTorch unavailable")
    FIXTURE = Fixture()


def tearDownModule() -> None:
    if FIXTURE is not None:
        FIXTURE.close()


class WebCase(unittest.TestCase):
    """HTTP helpers shared by the test classes."""

    def request(self, method: str, path: str, payload=None, *, headers=None, raw: bytes | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", FIXTURE.port, timeout=30)
        try:
            send = {"Content-Type": "application/json"} if method == "POST" else {}
            send.update(headers or {})
            body = raw if raw is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
            connection.request(method, path, body=body, headers=send)
            response = connection.getresponse()
            return response.status, response.msg, response.read()
        finally:
            connection.close()

    def get(self, path: str, **kwargs):
        status, _, body = self.request("GET", path, **kwargs)
        return status, json.loads(body)

    def post(self, path: str, payload=None, **kwargs):
        status, _, body = self.request("POST", path, payload, **kwargs)
        return status, json.loads(body)


class ReadOnlyEndpointTests(WebCase):
    def test_health_models_and_documents(self) -> None:
        status, health = self.get("/api/health")
        self.assertEqual((status, health), (200, {"status": "ok", "models": 3}))

        status, data = self.get("/api/models")
        self.assertEqual(status, 200)
        by_id = {model["id"]: model for model in data["models"]}
        self.assertEqual(set(by_id), {"tiny/char", "tiny/bpe", "broken"})
        char, bpe = by_id["tiny/char"], by_id["tiny/bpe"]
        self.assertEqual(char["tokenizer"], "character")
        self.assertEqual(char["vocabulary_size"], FIXTURE.char_tokenizer.vocabulary_size)
        self.assertEqual(char["parameters"], FIXTURE.char_model.parameter_count())
        self.assertEqual((char["context_length"], char["num_layers"], char["num_heads"], char["completed_steps"]), (CONTEXT, 2, 2, 5))
        self.assertEqual(bpe["tokenizer"], "bpe")
        self.assertEqual(bpe["vocabulary_size"], FIXTURE.bpe_tokenizer.vocabulary_size)
        # A corrupt checkpoint is reported in the list instead of failing the whole request.
        self.assertIn("error", by_id["broken"])
        self.assertNotIn("parameters", by_id["broken"])

        status, documents = self.get("/api/documents")
        self.assertEqual(documents, {"documents": ["empty.txt", "knowledge.txt", "other.txt"]})

    def test_unknown_routes_are_json_404(self) -> None:
        self.assertEqual(self.get("/nope"), (404, {"error": "not found"}))
        self.assertEqual(self.post("/nope", {})[0], 404)
        self.assertEqual(self.post("/api/health", {})[0], 404)  # GET-only route

    def test_index_page_and_headers(self) -> None:
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn(b"Project Ascent", body)
        status, headers, _ = self.request("GET", "/api/health")
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_server_binds_to_loopback_only(self) -> None:
        self.assertEqual(FIXTURE.server.server_address[0], "127.0.0.1")
        for host in ("0.0.0.0", "::1", "example.com", ""):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    create_server(FIXTURE.root, host=host, port=0, quiet=True)


class SecurityTests(WebCase):
    def test_foreign_host_header_is_rejected(self) -> None:
        for host in ("evil.example", "evil.example:8000", "127.0.0.1.evil.com", "localhost.evil.com:80", ""):
            with self.subTest(host=host):
                status, _ = self.get("/api/health", headers={"Host": host})
                self.assertEqual(status, 403)
        self.assertEqual(self.get("/", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.post("/api/chat/reset", {"session": "x"}, headers={"Host": "evil.example"})[0], 403)

    def test_loopback_host_names_are_accepted(self) -> None:
        for host in ("localhost", "localhost:1234", "127.0.0.1:8000", "LOCALHOST", "[::1]:8000"):
            with self.subTest(host=host):
                self.assertEqual(self.get("/api/health", headers={"Host": host})[0], 200)

    def test_hostname_parser(self) -> None:
        cases = {"127.0.0.1:8000": "127.0.0.1", "[::1]:8000": "::1", "LocalHost": "localhost",
                 "evil.com:80": "evil.com", " localhost ": "localhost", "[broken": ""}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(_hostname(value), expected)

    def test_cross_site_origin_is_rejected_on_post(self) -> None:
        body = {"session": "origin-test"}
        for origin in ("http://evil.example", "https://evil.example:443", "null", "http://localhost.evil.com"):
            with self.subTest(origin=origin):
                self.assertEqual(self.post("/api/chat/reset", body, headers={"Origin": origin})[0], 403)
        for origin in ("http://localhost:3000", "http://127.0.0.1:8000"):
            with self.subTest(origin=origin):
                self.assertEqual(self.post("/api/chat/reset", body, headers={"Origin": origin})[0], 200)

    def test_post_body_requirements(self) -> None:
        self.assertEqual(self.post("/api/chat/reset", {"session": "x"}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post("/api/chat/reset", raw=b"")[0], 400)
        for raw in (b"{not json", b"[1, 2]", b"\"text\"", b"\xff\xfe\x00"):
            with self.subTest(raw=raw):
                self.assertEqual(self.post("/api/chat/reset", raw=raw)[0], 400)

    def test_oversized_and_malformed_content_length(self) -> None:
        for length, expected in ((str(64 * 1024 + 1), 413), ("abc", 400), ("-5", 400)):
            with self.subTest(length=length):
                connection = http.client.HTTPConnection("127.0.0.1", FIXTURE.port, timeout=10)
                try:
                    connection.putrequest("POST", "/api/generate")
                    connection.putheader("Content-Type", "application/json")
                    connection.putheader("Content-Length", length)
                    connection.endheaders()
                    self.assertEqual(connection.getresponse().status, expected)
                finally:
                    connection.close()

    def test_client_cannot_name_files_as_models(self) -> None:
        checkpoint = str(FIXTURE.root / "artifacts" / "tiny" / "char" / "checkpoint.pt")
        for model in ("../../etc/passwd", "tiny/char/checkpoint.pt", checkpoint, "tiny/../tiny/char", "missing"):
            with self.subTest(model=model):
                status, data = self.post("/api/generate", {"model": model, "prompt": "abc"})
                self.assertEqual(status, 404)
                self.assertNotIn("Traceback", json.dumps(data))
        self.assertEqual(self.post("/api/generate", {"prompt": "abc"})[0], 400)
        self.assertEqual(self.post("/api/generate", {"model": 5, "prompt": "abc"})[0], 400)

    def test_corrupt_checkpoint_gives_clean_error(self) -> None:
        status, headers, raw = self.request("POST", "/api/generate", {"model": "broken", "prompt": "abc"})
        self.assertEqual(status, 500)
        text = raw.decode("utf-8")
        self.assertIn("could not load model", text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn(str(FIXTURE.root), text)


class FigureTests(WebCase):
    def test_allow_listed_figures_are_served(self) -> None:
        for name in ("loss.png", "sub/attn.png"):
            with self.subTest(name=name):
                status, headers, body = self.request("GET", "/figures/" + name)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], "image/png")
                self.assertEqual(body, PNG)

    def test_traversal_and_other_paths_are_not_found(self) -> None:
        paths = ["/figures/../data/knowledge.txt", "/figures/..%2f..%2fsecret.png", "/figures/%2e%2e/secret.png",
                 "/figures/sub/../loss.png", "/figures/loss.PNG", "/figures/missing.png", "/figures/",
                 "/figures/a/b/c.png", "/figures//loss.png", "/figures/.hidden.png", "/figures/loss.png%0a"]
        if FIXTURE.symlinks:
            paths.append("/figures/leak.png")  # symlink pointing outside the allow-listed root
        for path in paths:
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"SECRET", body)

    def test_results_lists_only_servable_figures(self) -> None:
        status, data = self.get("/api/results")
        self.assertEqual(status, 200)
        self.assertEqual(data["figures"], ["loss.png", "sub/attn.png"])  # leak.png (symlink) is not listed


class ResultsTests(WebCase):
    def test_v1_comparison_rows_and_missing_v2(self) -> None:
        status, data = self.get("/api/results")
        self.assertEqual(status, 200)
        rows = {row["tokenizer"]: row for row in data["v1_comparison"]}
        self.assertEqual(set(rows), {"character", "bpe"})
        self.assertEqual(rows["bpe"]["validation_bits_per_character"], 3.09)
        self.assertEqual(rows["character"]["greedy_repetition"], 0.97)
        self.assertEqual(rows["character"]["greedy_characters_per_second"], 3429.0)
        self.assertIsNone(data["v2_summary"])

    def test_v2_summary_and_figures_appear_when_present(self) -> None:
        v2 = FIXTURE.root / "artifacts" / "v2"
        try:
            (v2 / "figures").mkdir(parents=True)
            (v2 / "summary.json").write_text(json.dumps({"gate": "G1", "bpc": [3.1, 3.2]}), encoding="utf-8")
            (v2 / "figures" / "f.png").write_bytes(PNG)
            status, data = self.get("/api/results")
            self.assertEqual(data["v2_summary"], {"gate": "G1", "bpc": [3.1, 3.2]})
            self.assertIn("f.png", data["figures"])
            self.assertEqual(self.request("GET", "/figures/f.png")[0], 200)
        finally:
            shutil.rmtree(v2, ignore_errors=True)
        self.assertIsNone(self.get("/api/results")[1]["v2_summary"])

    def test_corrupt_result_files_degrade_cleanly(self) -> None:
        v2 = FIXTURE.root / "artifacts" / "v2"
        try:
            FIXTURE.comparison_path.write_text("{not json", encoding="utf-8")
            v2.mkdir(parents=True)
            (v2 / "summary.json").write_text("also not json", encoding="utf-8")
            status, data = self.get("/api/results")
            self.assertEqual(status, 200)
            self.assertEqual(data["v1_comparison"], [])
            self.assertIsNone(data["v2_summary"])
        finally:
            FIXTURE.comparison_path.write_text(FIXTURE.comparison_text, encoding="utf-8")
            shutil.rmtree(v2, ignore_errors=True)


class GenerateTests(WebCase):
    def generate(self, **overrides):
        body = {"model": "tiny/char", "prompt": "the cat", "mode": "greedy", "max_new_tokens": 8}
        body.update(overrides)
        return self.post("/api/generate", body)

    def test_greedy_output_matches_the_library_call(self) -> None:
        for model_id, tokenizer, model in (("tiny/char", FIXTURE.char_tokenizer, FIXTURE.char_model),
                                           ("tiny/bpe", FIXTURE.bpe_tokenizer, FIXTURE.bpe_model)):
            with self.subTest(model=model_id):
                status, data = self.generate(model=model_id)
                self.assertEqual(status, 200)
                ids = tokenizer.encode("the cat")
                expected = generate(model, torch.tensor([ids], dtype=torch.long), context_length=CONTEXT, max_new_tokens=8)
                self.assertEqual(data["text"], tokenizer.decode(expected[0, len(ids):].tolist()))
                self.assertEqual((data["model"], data["mode"], data["new_tokens"], data["prompt_tokens"]), (model_id, "greedy", 8, len(ids)))
                self.assertEqual((data["prompt_unknown_tokens"], data["unknown_characters"], data["prompt_cropped"], data["context_length"]),
                                 (0, [], False, CONTEXT))
                self.assertGreaterEqual(data["seconds"], 0)
                self.assertTrue(0.0 <= data["repetition_3gram"] <= 1.0)
                self.assertEqual(self.generate(model=model_id)[1]["text"], data["text"])  # deterministic

    def test_seeded_sampling_is_reproducible_and_beam_works(self) -> None:
        options = {"mode": "sample", "temperature": 1.0, "top_k": 10, "top_p": 0.9, "seed": 7, "max_new_tokens": 12}
        first, second = self.generate(**options)[1], self.generate(**options)[1]
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["new_tokens"], 12)
        status, beam = self.generate(mode="beam", beam_width=2, max_new_tokens=4)
        self.assertEqual((status, beam["new_tokens"]), (200, 4))

    def test_unknown_characters_and_cropping_are_reported(self) -> None:
        for model_id in ("tiny/char", "tiny/bpe"):
            with self.subTest(model=model_id):
                data = self.generate(model=model_id, prompt="Zebra", max_new_tokens=2)[1]
                self.assertEqual(data["unknown_characters"], ["Z"])
                self.assertGreaterEqual(data["prompt_unknown_tokens"], 1)
        long = self.generate(prompt="a" * 40, max_new_tokens=3)[1]
        self.assertTrue(long["prompt_cropped"])
        self.assertEqual(long["new_tokens"], 3)

    def test_input_validation(self) -> None:
        bad = [
            {"prompt": ""}, {"prompt": "   "}, {"prompt": 5}, {"prompt": "x" * 2001}, {"mode": "magic"},
            {"max_new_tokens": 0}, {"max_new_tokens": 301}, {"max_new_tokens": True}, {"max_new_tokens": "5"},
            {"max_new_tokens": 2.5}, {"mode": "beam", "max_new_tokens": 101}, {"temperature": 0}, {"temperature": 6},
            {"temperature": "hot"}, {"top_k": 0}, {"top_p": 0}, {"top_p": 1.5}, {"seed": -1}, {"beam_width": 6},
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides):
                status, data = self.generate(**overrides)
                self.assertEqual(status, 400)
                self.assertIn("error", data)
        self.assertEqual(self.generate(model="nope")[0], 404)


class ChatTests(WebCase):
    def chat(self, message, session=None, model="tiny/char", **extra):
        body = {"model": model, "message": message, **extra}
        if session is not None:
            body["session"] = session
        return self.post("/api/chat", body)

    def test_calculator_memory_and_reset(self) -> None:
        status, first = self.chat("/calc (2+3)*4", "calc-1")
        self.assertEqual(status, 200)
        self.assertEqual(first["reply"], "(2+3)*4 = 20")
        self.assertEqual(first["tool_calls"], [{"expression": "(2+3)*4", "result": "20"}])
        self.assertEqual((first["session"], first["memory_turns"], first["prompt"], first["prompt_cropped"]), ("calc-1", 2, None, False))
        self.assertEqual(first["unknown_characters"], [])  # the model never saw a /calc message

        second = self.chat("hello there", "calc-1")[1]
        self.assertEqual(second["memory_turns"], 4)
        self.assertEqual([turn["role"] for turn in second["memory"]], ["user", "assistant", "user", "assistant"])
        self.assertIn("User: /calc (2+3)*4", second["prompt"])
        self.assertIn("User: hello there", second["prompt"])
        self.assertTrue(second["prompt"].endswith("\nAssistant:"))
        self.assertIsInstance(second["reply"], str)
        self.assertTrue(second["prompt_cropped"])  # ~130 characters against a 32-token context
        self.assertGreater(second["prompt_tokens"], CONTEXT)

        self.assertEqual(self.post("/api/chat/reset", {"session": "calc-1"})[1], {"session": "calc-1", "reset": True})
        self.assertEqual(self.post("/api/chat/reset", {"session": "calc-1"})[1]["reset"], False)
        self.assertEqual(self.chat("/calc 1+1", "calc-1")[1]["memory_turns"], 2)

    def test_unsafe_expression_is_contained_and_new_session_is_issued(self) -> None:
        data = self.chat("/calc __import__('os').system('echo hi')")[1]
        self.assertTrue(data["reply"].startswith("Calculator error:"))
        self.assertIn("error", data["tool_calls"][0])
        self.assertRegex(data["session"], r"^[0-9a-f]{32}$")
        self.assertEqual(self.chat("/calc 2**1000")[1]["reply"], "Calculator error: exponent magnitude is limited to 100")

    def test_works_with_a_bpe_model(self) -> None:
        data = self.chat("hello", "bpe-chat", model="tiny/bpe")[1]
        self.assertEqual(data["model"], "tiny/bpe")
        self.assertEqual(data["memory_turns"], 2)

    def test_sessions_are_bounded(self) -> None:
        for index in range(21):
            self.chat(f"/calc 1+{index}", f"bound-{index}")
        self.assertEqual(self.chat("/calc 2+2", "bound-0")[1]["memory_turns"], 2)  # oldest was evicted
        self.assertEqual(self.chat("/calc 2+2", "bound-20")[1]["memory_turns"], 4)  # newest survived

    def test_input_validation(self) -> None:
        for session in ("bad session", "x\n", "a" * 65, 5, ""):
            with self.subTest(session=session):
                self.assertEqual(self.chat("hi", session)[0], 400)
        self.assertEqual(self.chat("")[0], 400)
        self.assertEqual(self.chat("x" * 501)[0], 400)
        self.assertEqual(self.chat("hi", max_new_tokens=201)[0], 400)
        self.assertEqual(self.post("/api/chat/reset", {})[0], 400)
        self.assertEqual(self.post("/api/chat/reset", {"session": "no good"})[0], 400)
        self.assertEqual(self.chat("hi", model="nope")[0], 404)


class RagTests(WebCase):
    def rag(self, **overrides):
        body = {"model": "tiny/char", "question": "Which planet is red?", "documents": ["knowledge.txt"], "max_new_tokens": 4}
        body.update(overrides)
        return self.post("/api/rag", body)

    def test_default_budget_reproduces_the_v1_truncation(self) -> None:
        status, data = self.rag()
        self.assertEqual(status, 200)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual((data["sources"][0]["source"], data["sources"][0]["chunk"], data["sources"][0]["score"]), ("knowledge.txt", 0, 0.75))
        self.assertEqual(data["context_chars"], max(16, CONTEXT // 2))
        prompt = data["with_context"]["prompt"]
        self.assertTrue(prompt.startswith("Use only the supplied context"))
        self.assertIn("Question: Which planet is red?", prompt)
        self.assertNotIn("Mars", prompt)  # the retrieved text was cut off (defect D1 in the design record)
        self.assertTrue(data["with_context"]["prompt_cropped"])
        self.assertEqual(data["closed_book"]["prompt"], "Question: Which planet is red?\nAnswer:")
        self.assertIsInstance(data["with_context"]["answer"], str)
        self.assertIsInstance(data["closed_book"]["answer"], str)
        self.assertIsInstance(data["unknown_characters"], list)
        self.assertEqual(data["context_length"], CONTEXT)

    def test_larger_budget_includes_the_retrieved_text(self) -> None:
        data = self.rag(context_chars=400)[1]
        self.assertEqual(data["context_chars"], 400)
        self.assertIn("Mars is known as the red planet", data["with_context"]["prompt"])
        self.assertIn("Mars is known", data["sources"][0]["text"])

    def test_multiple_documents_rank_by_overlap(self) -> None:
        data = self.rag(question="boil water", documents=["knowledge.txt", "other.txt"])[1]
        self.assertEqual([s["source"] for s in data["sources"]], ["other.txt", "knowledge.txt"])
        self.assertEqual(data["sources"][0]["score"], 0.5)

    def test_input_validation(self) -> None:
        bad = [
            {"documents": ["nope.txt"]}, {"documents": ["../data/knowledge.txt"]}, {"documents": ["secret.json"]},
            {"documents": ["knowledge.txt\n"]}, {"documents": []}, {"documents": "knowledge.txt"}, {"documents": [5]},
            {"documents": ["knowledge.txt"] * 6}, {"documents": ["empty.txt"]}, {"context_chars": 5},
            {"context_chars": 4001}, {"top_k": 6}, {"top_k": 0}, {"question": ""}, {"question": "q" * 301},
            {"max_new_tokens": 151},
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides):
                status, data = self.rag(**overrides)
                self.assertEqual(status, 400)
                self.assertIn("error", data)
        self.assertEqual(self.rag(model="nope")[0], 404)


class AttentionTests(WebCase):
    def test_shape_normalisation_and_causality(self) -> None:
        status, data = self.post("/api/attention", {"model": "tiny/char", "prompt": "abc def"})
        self.assertEqual(status, 200)
        self.assertEqual(data["tokens"], list("abc def"))
        self.assertEqual((data["layers"], data["heads"], data["cropped"]), (2, 2, False))
        weights = data["weights"]
        self.assertEqual((len(weights), len(weights[0]), len(weights[0][0]), len(weights[0][0][0])), (2, 2, 7, 7))
        for layer in weights:
            for head in layer:
                for row_index, row in enumerate(head):
                    self.assertAlmostEqual(sum(row), 1.0, delta=0.01)  # values are rounded to 3 decimals
                    self.assertTrue(all(value == 0.0 for value in row[row_index + 1:]))

    def test_unknown_tokens_bpe_tokens_and_cropping(self) -> None:
        self.assertEqual(self.post("/api/attention", {"model": "tiny/char", "prompt": "Ab"})[1]["tokens"], ["<unk>", "b"])
        ids = FIXTURE.bpe_tokenizer.encode("the cat")
        tokens = self.post("/api/attention", {"model": "tiny/bpe", "prompt": "the cat"})[1]["tokens"]
        self.assertEqual(tokens, [FIXTURE.bpe_tokenizer.decode([i]) for i in ids])
        long = self.post("/api/attention", {"model": "tiny/char", "prompt": "a" * 80})[1]
        self.assertEqual((len(long["tokens"]), long["cropped"]), (CONTEXT, True))

    def test_input_validation(self) -> None:
        self.assertEqual(self.post("/api/attention", {"model": "tiny/char", "prompt": ""})[0], 400)
        self.assertEqual(self.post("/api/attention", {"model": "tiny/char", "prompt": "x" * 2001})[0], 400)
        self.assertEqual(self.post("/api/attention", {"model": "nope", "prompt": "abc"})[0], 404)


class RegistryTests(unittest.TestCase):
    def test_discovery_cache_and_errors(self) -> None:
        registry = ModelRegistry(FIXTURE.root / "artifacts", cache_size=1)
        self.assertEqual(set(registry.discover()), {"tiny/char", "tiny/bpe", "broken"})
        registry.get("tiny/char")
        registry.get("tiny/bpe")
        self.assertEqual(list(registry._cache), ["tiny/bpe"])  # least recently used model was evicted
        with self.assertRaises(KeyError):
            registry.get("missing")
        with self.assertRaises(Exception) as caught:
            registry.get("broken")
        self.assertNotIsInstance(caught.exception, KeyError)
        with self.assertRaises(ValueError):
            ModelRegistry(FIXTURE.root, cache_size=0)

    def test_root_level_and_symlinked_checkpoints_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            write_checkpoint(artifacts / "ok", FIXTURE.char_tokenizer, seed=3)
            (artifacts / "checkpoint.pt").write_bytes(b"root level")
            outside = root / "outside.pt"
            outside.write_bytes(b"outside")
            (artifacts / "evil").mkdir()
            try:
                (artifacts / "evil" / "checkpoint.pt").symlink_to(outside)
            except (OSError, NotImplementedError):
                pass
            self.assertEqual(set(ModelRegistry(artifacts).discover()), {"ok"})
            self.assertEqual(ModelRegistry(root / "does-not-exist").discover(), {})


class FrontendStaticTests(unittest.TestCase):
    """Checks that need no browser: wiring between the page and the API, and basic safety."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = INDEX_HTML.read_text(encoding="utf-8")

    def test_every_referenced_element_id_exists(self) -> None:
        defined = set(re.findall(r'\bid="([^"]+)"', self.html))
        referenced = set(re.findall(r"""["']#([A-Za-z0-9_-]+)["']""", self.html))
        self.assertTrue(referenced)
        self.assertEqual(referenced - defined, set())

    def test_tabs_map_to_panels(self) -> None:
        tabs = re.findall(r'data-tab="([a-z]+)"', self.html)
        self.assertEqual(tabs, ["play", "chat", "rag", "attn", "results"])
        for name in tabs:
            self.assertIn(f'id="tab-{name}"', self.html)

    def test_every_api_endpoint_is_used_by_the_page(self) -> None:
        for endpoint in ("/api/models", "/api/documents", "/api/generate", "/api/chat", "/api/chat/reset",
                         "/api/rag", "/api/attention", "/api/results", "/figures/"):
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, self.html)

    def test_page_is_offline_and_does_not_render_html_from_the_model(self) -> None:
        self.assertEqual(re.findall(r"""["'(]\s*(?:https?:)?//""", self.html), [])
        self.assertNotRegex(self.html, r"@import|<link\b[^>]*href=|<script\b[^>]*\bsrc=")
        for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "localStorage"):
            with self.subTest(token=dangerous):
                self.assertNotIn(dangerous, self.html)
        self.assertIn("tiny research models", self.html)  # the honesty banner

    @unittest.skipUnless(shutil.which("node"), "Node.js not installed")
    def test_script_has_valid_syntax(self) -> None:
        script = re.search(r"<script>(.*)</script>", self.html, re.S).group(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.js"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
