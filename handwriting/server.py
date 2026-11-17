"""손글씨 노트 변환기 — prototype web server.

    python server.py            # then open http://127.0.0.1:5000
"""
import glob
import json
import os
import re
import secrets
import shutil
import time

from flask import Flask, abort, jsonify, request, send_file

from hwnote import charsets, extract, render, template as T, textio

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("HWNOTE_DATA", os.path.join(BASE, "data"))
MAX_TEXT = 20000
OUTPUT_TTL = 24 * 3600

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
PROFILE_RE = re.compile(r"^[0-9a-f]{10}$")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")


def _install_samples():
    """Make the bundled demo templates known so the demo photos can be uploaded."""
    os.makedirs(os.path.join(DATA, "templates"), exist_ok=True)
    for path in glob.glob(os.path.join(BASE, "samples", "template_*.json")):
        target = os.path.join(DATA, "templates", os.path.basename(path).replace("template_", ""))
        if not os.path.exists(target):
            shutil.copyfile(path, target)


def _error(message, status=400):
    return jsonify({"error": message}), status


def _profile_path(profile_id):
    if not PROFILE_RE.match(profile_id or ""):
        abort(404)
    path = extract.profile_dir(DATA, profile_id)
    if not os.path.exists(os.path.join(path, "profile.json")):
        abort(404)
    return path


def _profile_summary(profile):
    chars = {g["char"] for g in profile["glyphs"]}
    return {
        "id": profile["id"],
        "glyphs": len(profile["glyphs"]),
        "chars": len(chars),
        "hangul": sum(1 for c in chars if charsets.is_syllable(c)),
        "missing_jamo": [j for j in charsets.JAMO if j not in chars],
        "sheet": f"/api/profile/{profile['id']}/sheet.png?v={len(profile['glyphs'])}",
    }


@app.get("/")
def index():
    return send_file(os.path.join(BASE, "static", "index.html"))


@app.get("/api/charsets")
def list_charsets():
    return jsonify([{"key": k, "label": label, "count": len(chars)} for k, (label, chars) in charsets.CHARSETS.items()])


@app.post("/api/template")
def make_template():
    body = request.get_json(force=True, silent=True) or {}
    chars = charsets.build_charlist(body.get("sets", []), body.get("text", "")[:MAX_TEXT], body.get("repeat", 1))
    if not chars:
        return _error("템플릿에 넣을 글자를 하나 이상 고르세요.")
    if len(chars) > T.CELLS_PER_PAGE * 20:
        return _error(f"글자가 너무 많습니다 ({len(chars)}칸). 20쪽 이하로 줄여 주세요.")
    info = T.new_template(DATA, chars)
    with open(os.path.join(DATA, "templates", f"{info['id']}.pdf"), "wb") as f:
        f.write(T.build_pdf(info))
    return jsonify({"id": info["id"], "pages": len(info["pages"]), "cells": len(chars),
                    "pdf": f"/api/template/{info['id']}.pdf"})


@app.get("/api/template/<int:template_id>.pdf")
def template_pdf(template_id):
    path = os.path.join(DATA, "templates", f"{template_id}.pdf")
    if not os.path.exists(path):
        info = T.load_template(DATA, template_id)
        if info is None:
            abort(404)
        with open(path, "wb") as f:
            f.write(T.build_pdf(info))
    return send_file(path, mimetype="application/pdf", download_name=f"handwriting_template_{template_id}.pdf")


@app.post("/api/profile")
def make_profile():
    files = [f for f in request.files.getlist("files") if f.filename]
    if not files:
        return _error("작성한 템플릿 사진이나 스캔 파일을 올려 주세요.")
    hint = request.form.get("template_id", "").strip()
    extend = request.form.get("extend_profile", "").strip()
    if hint and not hint.isdigit():
        return _error("템플릿 ID는 숫자입니다.")
    if extend and not PROFILE_RE.match(extend):
        return _error("프로필 ID 형식이 올바르지 않습니다.")
    try:
        profile, empty, warnings = extract.build_profile(
            DATA, [(f.read(), f.filename) for f in files], hint or None, extend or None)
    except extract.ExtractError as exc:
        return _error(str(exc))
    summary = _profile_summary(profile)
    summary.update(empty=empty, warnings=warnings)
    return jsonify(summary)


@app.get("/api/profile/<profile_id>")
def profile_info(profile_id):
    with open(os.path.join(_profile_path(profile_id), "profile.json"), encoding="utf-8") as f:
        return jsonify(_profile_summary(json.load(f)))


@app.get("/api/profile/<profile_id>/sheet.png")
def profile_sheet(profile_id):
    path = _profile_path(profile_id)
    sheet = os.path.join(path, "sheet.png")
    stamp = os.path.getmtime(os.path.join(path, "profile.json"))
    if not os.path.exists(sheet) or os.path.getmtime(sheet) < stamp:
        render.contact_sheet(render.Profile(path)).save(sheet)
    return send_file(sheet, mimetype="image/png", max_age=0)


@app.post("/api/text")
def upload_text():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return _error("PDF 또는 TXT 파일을 골라 주세요.")
    try:
        text = textio.extract_text(upload.read(), upload.filename, request.form.get("keep_lines") == "1")
    except textio.TextError as exc:
        return _error(str(exc))
    return jsonify({"text": text[:MAX_TEXT], "truncated": len(text) > MAX_TEXT})


def _float(body, key, default, lo, hi):
    try:
        return min(hi, max(lo, float(body.get(key, default))))
    except (TypeError, ValueError):
        return default


def _prune_outputs():
    now = time.time()
    for path in glob.glob(os.path.join(DATA, "outputs", "*")):
        if now - os.path.getmtime(path) > OUTPUT_TTL:
            shutil.rmtree(path, ignore_errors=True)


@app.post("/api/render")
def render_notes():
    body = request.get_json(force=True, silent=True) or {}
    path = _profile_path(body.get("profile_id", ""))
    text = (body.get("text") or "").strip("\n")
    if not text.strip():
        return _error("변환할 글을 입력하거나 파일을 올려 주세요.")
    if len(text) > MAX_TEXT:
        return _error(f"글이 너무 깁니다 (최대 {MAX_TEXT}자).")
    opts = render.Options(
        paper=body.get("paper") if body.get("paper") in ("lined", "grid", "blank") else "lined",
        ink=body.get("ink") if body.get("ink") in ("auto", *render.INK_COLORS) else "auto",
        line_pitch=_float(body, "line_pitch", 8.0, 6.0, 12.0),
        size=_float(body, "size", 0.6, 0.35, 0.9),
        letter_spacing=_float(body, "letter_spacing", 1.0, 0.0, 3.0),
        word_spacing=_float(body, "word_spacing", 1.0, 0.3, 3.0),
        messiness=_float(body, "messiness", 0.5, 0.0, 1.0),
        margin_line=bool(body.get("margin_line", True)),
    )
    _prune_outputs()
    renderer = render.Renderer(render.Profile(path), opts)
    pages = renderer.render(text)
    job = secrets.token_hex(6)
    out = os.path.join(DATA, "outputs", job)
    os.makedirs(out)
    urls = []
    for n, page in enumerate(pages, 1):
        page.save(os.path.join(out, f"page-{n}.jpg"), quality=88)
        urls.append(f"/api/output/{job}/page-{n}.jpg")
    with open(os.path.join(out, "notes.pdf"), "wb") as f:
        f.write(render.pages_to_pdf(pages))
    return jsonify({"pages": urls, "pdf": f"/api/output/{job}/notes.pdf",
                    "composed": "".join(sorted(renderer.factory.composed)),
                    "fallback": "".join(sorted(renderer.factory.fallback))})


@app.get("/api/output/<job>/<name>")
def output_file(job, name):
    if not JOB_RE.match(job) or not re.match(r"^(page-\d+\.jpg|notes\.pdf)$", name):
        abort(404)
    path = os.path.join(DATA, "outputs", job, name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, download_name="handwritten_notes.pdf" if name.endswith(".pdf") else name)


_install_samples()

if __name__ == "__main__":
    import sys
    import threading
    import webbrowser

    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}"
    print(f"손글씨 노트 변환기: {url}  (끝내려면 Ctrl+C)")
    if "--open" in sys.argv:
        threading.Timer(1.2, webbrowser.open, (url,)).start()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, threaded=True)
