"""Localhost HTTP server for the Project Ascent web interface.

Run from the repository root::

    .venv/bin/python -m src.web.server --port 8000

then open http://127.0.0.1:8000. The server binds to loopback only, rejects
requests whose Host/Origin is not a loopback name (DNS-rebinding and cross-site
guard), requires JSON for POST, and caps request size.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .handlers import Api, ApiError

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}  # accepted in the Host/Origin headers
BIND_HOSTS = {"localhost", "127.0.0.1"}  # the server socket is IPv4 only
MAX_BODY_BYTES = 64 * 1024
STATIC_DIR = Path(__file__).parent / "static"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def _hostname(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else ""
    return value.rsplit(":", 1)[0] if ":" in value else value


class AscentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], api: Api, *, quiet: bool = False) -> None:
        super().__init__(address, RequestHandler)
        self.api = api
        self.quiet = quiet


class RequestHandler(BaseHTTPRequestHandler):
    server: AscentServer
    server_version = "AscentWeb/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - signature fixed by the base class
        if not self.server.quiet:
            super().log_message(format, *args)

    # ------------------------------------------------------------ responses

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send_bytes(self, status: int, body: bytes, content_type: str, *, html: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        if html:
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    # ---------------------------------------------------------------- guards

    def _host_allowed(self) -> bool:
        return _hostname(self.headers.get("Host", "")) in LOOPBACK_HOSTS

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        hostname = urlsplit(origin).hostname
        return hostname is not None and hostname.lower() in LOOPBACK_HOSTS

    # --------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._host_allowed():
            return self._send_error_json(403, "forbidden host")
        path = urlsplit(self.path).path
        api = self.server.api
        routes = {
            "/api/health": api.health,
            "/api/models": api.models,
            "/api/documents": api.documents,
            "/api/results": api.results,
        }
        try:
            if path in ("/", "/index.html"):
                page = (STATIC_DIR / "index.html").read_bytes()
                return self._send_bytes(200, page, "text/html; charset=utf-8", html=True)
            if path in routes:
                return self._send_json(200, routes[path]())
            if path.startswith("/figures/"):
                figure = api.figure_path(path[len("/figures/"):])
                if figure is None:
                    return self._send_error_json(404, "figure not found")
                return self._send_bytes(200, figure.read_bytes(), "image/png")
            return self._send_error_json(404, "not found")
        except ApiError as error:
            return self._send_error_json(error.status, error.message)
        except Exception as error:  # noqa: BLE001 - never leak a traceback to the client
            print(f"internal error on GET {path}: {type(error).__name__}: {error}", file=sys.stderr)
            return self._send_error_json(500, "internal error")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed() or not self._origin_allowed():
            return self._send_error_json(403, "forbidden origin")
        path = urlsplit(self.path).path
        api = self.server.api
        routes = {
            "/api/generate": api.generate,
            "/api/chat": api.chat,
            "/api/chat/reset": api.chat_reset,
            "/api/rag": api.rag,
            "/api/attention": api.attention,
        }
        handler = routes.get(path)
        if handler is None:
            return self._send_error_json(404, "not found")
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return self._send_error_json(415, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_error_json(400, "invalid Content-Length")
        if length <= 0:
            return self._send_error_json(400, "empty request body")
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            return self._send_error_json(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return self._send_error_json(400, "request body is not valid JSON")
        if not isinstance(payload, dict):
            return self._send_error_json(400, "request body must be a JSON object")
        try:
            return self._send_json(200, handler(payload))
        except ApiError as error:
            return self._send_error_json(error.status, error.message)
        except Exception as error:  # noqa: BLE001
            print(f"internal error on POST {path}: {type(error).__name__}: {error}", file=sys.stderr)
            return self._send_error_json(500, "internal error")


def create_server(root: str | Path = ".", *, host: str = "127.0.0.1", port: int = 8000,
                  device: str = "cpu", quiet: bool = False) -> AscentServer:
    if host not in BIND_HOSTS:
        raise ValueError("the web interface only binds to a loopback IPv4 address (127.0.0.1 or localhost)")
    return AscentServer((host, port), Api(root, device=device), quiet=quiet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", default=".", help="Project root containing artifacts/ and data/")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    server = create_server(args.root, host=args.host, port=args.port, device=args.device)
    found = len(server.api.registry.discover())
    print(f"Project Ascent web interface: http://{args.host}:{server.server_address[1]}  ({found} checkpoint(s) found)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
