"""FL DoS Swarm IDS console as a desktop window (no browser).

Starts this folder's Streamlit dashboard on a local port (if it is not already running), opens it in a native
macOS window (pywebview / WKWebView), and stops the server it started when the window is closed.

  .venv/bin/python desktop_app.py          (or double-click "FL DoS Swarm v14.app")
"""
import os
import subprocess
import sys
import time
import urllib.request

import webview

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("FLDOS_PORT", 8502))


def server_up(port):
    try:
        urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
        return True
    except Exception:
        return False


def main():
    proc = None
    if not server_up(PORT):
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        log = open(os.path.join(ROOT, "runs", "app_server.log"), "a")
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py"),
                                 "--server.port", str(PORT), "--server.headless", "true",
                                 "--browser.gatherUsageStats", "false"],
                                cwd=ROOT, env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1"),
                                stdout=log, stderr=subprocess.STDOUT)
        for _ in range(160):
            if server_up(PORT):
                break
            time.sleep(0.25)
    webview.create_window("FL DoS Swarm · IDS console", f"http://localhost:{PORT}", width=1440, height=900,
                          min_size=(1100, 700))
    try:
        webview.start()
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
