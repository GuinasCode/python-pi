"""A local HTTP fixture server for browser integration tests — spec
section 38: "testar com um servidor HTTP local controlado" rather than
depending on real internet sites (flaky, slow, and untrustworthy for
CI). Serves small hand-written pages covering the interaction surface
the browser harness needs to prove against: a form, a button, dynamic
JS, a select, a file input, a downloadable file, and a cookie-setting
route (used to prove real session/context persistence across
navigations without needing browser_evaluate yet).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PAGES: dict[str, bytes] = {
    "/": b"""
<!doctype html><html><body>
<h1 id="heading">Fixture Home</h1>
<a href="/form">form</a>
</body></html>
""",
    "/form": b"""
<!doctype html><html><body>
<form id="the-form">
  <input type="text" name="username" id="username" />
  <select id="color"><option value="red">Red</option><option value="blue">Blue</option></select>
  <input type="file" id="upload" />
  <button type="button" id="submit-btn" onclick="
    document.getElementById('result').textContent = 'clicked: ' + document.getElementById('username').value
  ">Submit</button>
</form>
<div id="result"></div>
<a id="download-link" href="/download" download="fixture.txt">Download</a>
</body></html>
""",
    "/dynamic": b"""
<!doctype html><html><body>
<div id="content">loading...</div>
<script>
setTimeout(function() {
  document.getElementById('content').textContent = 'loaded';
}, 200);
</script>
</body></html>
""",
    "/download": b"file contents for download test",
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # keep test output quiet

    def do_GET(self) -> None:
        if self.path == "/set-cookie":
            self.send_response(200)
            self.send_header("Set-Cookie", "fixture_session=abc123; Path=/")
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>cookie set</body></html>")
            return

        if self.path == "/echo-cookie":
            cookie_header = self.headers.get("Cookie", "")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body>{cookie_header}</body></html>".encode())
            return

        if self.path == "/download":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", "attachment; filename=fixture.txt")
            self.end_headers()
            self.wfile.write(_PAGES["/download"])
            return

        body = _PAGES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def fixture_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
