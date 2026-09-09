"""Serve a portable tour through a local-only HTTP endpoint."""

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def view_tour(output, port=8083):
    root = Path(output).resolve()
    if not (root / "tour.json").is_file():
        raise ValueError("Tour directory must contain tour.json")
    viewer = Path(__file__).with_name("tour_viewer.html").read_bytes()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(viewer)))
                self.end_headers()
                self.wfile.write(viewer)
            else:
                super().do_GET()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", port), functools.partial(Handler, directory=str(root))
    )
    print(f"tour: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
