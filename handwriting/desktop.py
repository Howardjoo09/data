"""손글씨 노트 — desktop app.

Double-click HandNote.exe (or run ``python desktop.py``): a small window opens and the
converter appears in the web browser.  Everything runs on this computer; closing the window
stops it.  Data (my handwriting, recent results) lives in %LOCALAPPDATA%\\HandNote, and every
finished notebook is also saved to Documents\\손글씨노트.

    python desktop.py --self-test     # start, convert text and a PDF, exit 0 if all worked
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser

APP_NAME = "손글씨 노트"
PORT = 5173  # fixed, so a second double-click finds the running app


def data_dir():
    if os.name == "nt":
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "HandNote")
    return os.path.join(os.path.expanduser("~"), ".handnote")


def save_dir():
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(docs if os.path.isdir(docs) else os.path.expanduser("~"), "손글씨노트")


def _quiet_streams(folder):
    """A windowed program has no console; send stray output to a log file instead."""
    if sys.stdout is None or sys.stderr is None:
        os.makedirs(folder, exist_ok=True)
        log = open(os.path.join(folder, "handnote.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def _running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.5) as r:
            return json.load(r).get("ok") is True
    except Exception:
        return False


def _free_port(preferred):
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("빈 포트를 찾지 못했습니다.")


def start_server(port):
    import server  # reads HWNOTE_DATA / HWNOTE_SAVE_DIR when imported
    from werkzeug.serving import make_server
    httpd = make_server("127.0.0.1", port, server.app, threaded=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _open_folder(path):
    os.makedirs(path, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 (a folder chosen by the program)
    else:
        import subprocess
        subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", path])


def window(url, httpd):
    import tkinter as tk
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("400x180")
    root.resizable(False, False)
    tk.Label(root, text="손글씨 노트가 실행 중입니다", font=("Malgun Gothic", 13, "bold")).pack(pady=(18, 4))
    tk.Label(root, text="브라우저에서 변환하세요. 이 창을 닫으면 프로그램이 끝납니다.").pack()
    buttons = tk.Frame(root)
    buttons.pack(pady=16)
    tk.Button(buttons, text="브라우저 열기", width=12, command=lambda: webbrowser.open(url)).pack(side="left", padx=4)
    tk.Button(buttons, text="결과 폴더 열기", width=12, command=lambda: _open_folder(save_dir())).pack(side="left", padx=4)
    tk.Button(buttons, text="종료", width=8, command=root.destroy).pack(side="left", padx=4)
    root.after(400, lambda: webbrowser.open(url))
    root.mainloop()
    httpd.shutdown()


def main():
    os.environ.setdefault("HWNOTE_DATA", data_dir())
    os.environ.setdefault("HWNOTE_SAVE_DIR", save_dir())
    _quiet_streams(os.environ["HWNOTE_DATA"])
    if _running(PORT):  # already open: just show it again
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
        return 0
    port = _free_port(PORT)
    httpd = start_server(port)
    url = f"http://127.0.0.1:{port}/"
    try:
        window(url, httpd)
    except Exception:  # no window system: keep serving until interrupted
        print(f"{APP_NAME}: {url}  (끝내려면 Ctrl+C)")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            httpd.shutdown()
    return 0


# --------------------------------------------------------------------------- self-test

def _request(url, data=None, files=None):
    """Tiny multipart/JSON client (urllib only, so the packaged app needs nothing more)."""
    headers = {}
    body = None
    if files is not None or data is not None:
        boundary = "----handnote" + str(time.time_ns())
        parts = []
        for key, value in (data or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, (name, content) in (files or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\n'
                         f'Content-Type: application/octet-stream\r\n\r\n'.encode() + content + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw[:1] in (b"{", b"[") else raw


def _wait(base, job, limit=300):
    deadline = time.time() + limit
    while time.time() < deadline:
        state = _request(f"{base}/api/jobs/{job}")
        if state["state"] in ("done", "error"):
            return state
        time.sleep(0.5)
    raise RuntimeError("변환이 끝나지 않았습니다")


def self_test():
    """Start the app on a scratch folder, learn a handwriting from the bundled sample pages,
    convert typed text and a PDF page, and round-trip the handwriting file."""
    import pymupdf
    scratch = tempfile.mkdtemp(prefix="handnote-test-")
    os.environ["HWNOTE_DATA"] = os.path.join(scratch, "data")
    os.environ["HWNOTE_SAVE_DIR"] = os.path.join(scratch, "saved")
    _quiet_streams(scratch)
    import server
    from hwnote import extract
    port = _free_port(0)
    httpd = start_server(port)
    base = f"http://127.0.0.1:{port}"
    try:
        assert _request(f"{base}/api/health")["ok"]
        assert b"<title>" in _request(f"{base}/")
        samples = os.path.join(server.BASE, "samples")
        uploads = []
        for name in sorted(os.listdir(samples)):
            if name.startswith("demo_page") and name.endswith(".jpg"):
                with open(os.path.join(samples, name), "rb") as f:
                    uploads.append((f.read(), name))
        profile, _, _ = extract.build_profile(server.DATA, uploads)
        _request(f"{base}/api/me/use/{profile['id']}", data={})
        me = _request(f"{base}/api/me")["profile"]
        assert me and me["glyphs"] > 50, me

        job = _request(f"{base}/api/jobs", data={"mode": "text", "text": "손글씨 노트 시험입니다.\n가나다 abc 123",
                                                  "paper": "grid"})["job"]
        state = _wait(base, job)
        assert state["state"] == "done" and state["pages"], state
        assert state["saved"] and os.path.exists(state["saved"]), state

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
        page.insert_text((72, 100), "합성함수의 미분법: y = f(x) 이면 dy/dx", fontname="kr", fontsize=12)
        pdf = doc.tobytes()
        doc.close()
        assert _request(f"{base}/api/pdfinfo", files={"document": ("t.pdf", pdf)})["pages"] == 1
        job = _request(f"{base}/api/jobs", data={"mode": "layout", "pages": "1", "paper": "grid"},
                       files={"document": ("t.pdf", pdf)})["job"]
        state = _wait(base, job)
        assert state["state"] == "done" and len(state["pages"]) == 1, state
        assert _request(f"{base}{state['pdf']}")[:5] == b"%PDF-"

        archive = _request(f"{base}/api/me/export")
        again = _request(f"{base}/api/me/import", files={"file": ("me.zip", archive)})["profile"]
        assert again["glyphs"] == me["glyphs"], again
    finally:
        httpd.shutdown()
    print("self-test ok")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        try:
            sys.exit(self_test())
        except Exception as exc:  # the exit code is what the build checks
            import traceback
            traceback.print_exc()
            print(f"self-test failed: {exc}")
            sys.exit(1)
    sys.exit(main())
