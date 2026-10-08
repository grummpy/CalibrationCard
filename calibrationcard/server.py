"""Bind the UI to 127.0.0.1 and open it in the local browser."""

from __future__ import annotations

import socket
import threading
import webbrowser

from calibrationcard import DEFAULT_PORT, HOST
from calibrationcard.app import create_app


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free TCP port on {HOST} starting at {preferred}")


def make_bound_server(app, port: int = 0):
    from werkzeug.serving import make_server

    return make_server(HOST, port, app, threaded=True)


def serve(*, port: int | None = None, open_browser: bool = True, data_dir=None) -> None:
    chosen = pick_port(DEFAULT_PORT if port is None else port)
    app = create_app(data_dir=data_dir)
    url = f"http://{HOST}:{chosen}/"
    print(f"CalibrationCard is running at {url}", flush=True)
    print("Your predictions stay on this computer. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: _open(url)).start()
    server = make_bound_server(app, chosen)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def _open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        print(f"Open this URL in a browser: {url}")
