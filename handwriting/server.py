"""손글씨 노트 변환기 — prototype web server.

    python server.py            # then open http://127.0.0.1:5000
"""
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import time

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from hwnote import charsets, extract, render, style, template as T, textio

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("HWNOTE_DATA", os.path.join(BASE, "data"))
MAX_TEXT = 20000
OUTPUT_TTL = 24 * 3600

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
PROFILE_RE = re.compile(r"^[0-9a-f]{10}$")
JOB_RE = re.compile(r"^[0-9a-f]{12}$")


def _install_samples():
    """Make the official templates and the demo templates known, so filled-in copies of them
    are recognised even on a fresh server."""
    os.makedirs(os.path.join(DATA, "templates"), exist_ok=True)
    sources = glob.glob(os.path.join(BASE, "templates", "*.json"))
    sources += glob.glob(os.path.join(BASE, "samples", "template_*.json"))
    for path in sources:
        target = os.path.join(DATA, "templates", os.path.basename(path).replace("template_", ""))
        if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(path):
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
        "freeform": profile.get("report"),
        "sheet": f"/api/profile/{profile['id']}/sheet.png?v={len(profile['glyphs'])}",
    }


@app.get("/")
def index():
    return send_file(os.path.join(BASE, "static", "convert.html"))


@app.get("/advanced")
def advanced():
    return send_file(os.path.join(BASE, "static", "advanced.html"))


@app.get("/samples/<name>")
def sample_file(name):
    if not re.match(r"^[\w.-]+\.(jpg|png|txt|pdf)$", name):
        abort(404)
    return send_from_directory(os.path.join(BASE, "samples"), name)


@app.get("/api/demo")
def demo():
    """Example inputs for the main page: a free-form handwriting photo, its text, and a text to convert."""
    def read(name):
        with open(os.path.join(BASE, "samples", name), encoding="utf-8") as f:
            return f.read().strip()
    return jsonify({"handwriting": ["freeform_demo.jpg"], "transcript": read("freeform_demo.txt"),
                    "text": read("demo_text.txt")})


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
            DATA, [(f.read(), f.filename) for f in files], hint or None, extend or None,
            transcript=request.form.get("transcript", "")[:MAX_TEXT])
    except extract.ExtractError as exc:
        return _error(str(exc))
    summary = _profile_summary(profile)
    summary.update(empty=empty, warnings=warnings)
    return jsonify(summary)


@app.get("/api/profile/<profile_id>")
def profile_info(profile_id):
    with open(os.path.join(_profile_path(profile_id), "profile.json"), encoding="utf-8") as f:
        return jsonify(_profile_summary(json.load(f)))


@app.post("/api/profile/<profile_id>/calibrate")
def calibrate_profile(profile_id):
    """Tune a handwriting profile to a sample of the writer's own sentences (photo + text)."""
    path = _profile_path(profile_id)
    files = [f for f in request.files.getlist("sample") if f.filename]
    transcript = request.form.get("transcript", "")[:MAX_TEXT]
    if not files or not transcript.strip():
        return _error("손글씨 문장 사진과, 그 사진에 적힌 내용을 함께 올려 주세요.")
    try:
        images = [img for f in files for img in extract.load_images(f.read(), f.filename)]
    except extract.ExtractError as exc:
        return _error(str(exc))
    sample, rendered, corrections = style.calibrate(path, images, transcript)
    if sample["syllables"] < 10:
        return _error("사진에서 글자를 충분히 찾지 못했습니다. 손글씨만 반듯하게 찍고, 내용을 사진과 똑같이 입력해 주세요.")
    keys = ["width", "letter_gap", "word_gap", "middle_spread", "height_spread"]
    return jsonify({"syllables": sample["syllables"], "corrected": len(corrections),
                    "sample": {k: round(sample[k], 3) for k in keys},
                    "program": {k: round(rendered[k], 3) for k in keys}})


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


def _options(source):
    """Rendering options from a JSON body or a form."""
    def number(key, default, lo, hi):
        try:
            return min(hi, max(lo, float(source.get(key, default))))
        except (TypeError, ValueError):
            return default

    margin = source.get("margin_line", True)
    if isinstance(margin, str):
        margin = margin.lower() not in ("0", "false", "off", "")
    return render.Options(
        paper=source.get("paper") if source.get("paper") in ("lined", "grid", "blank") else "lined",
        ink=source.get("ink") if source.get("ink") in ("auto", *render.INK_COLORS) else "auto",
        line_pitch=number("line_pitch", 8.0, 6.0, 12.0),
        size=number("size", 0.6, 0.35, 0.9),
        letter_spacing=number("letter_spacing", 1.0, 0.0, 3.0),
        word_spacing=number("word_spacing", 1.0, 0.3, 3.0),
        messiness=number("messiness", 0.5, 0.0, 1.0),
        margin_line=bool(margin),
    )


def _prune_outputs():
    now = time.time()
    for path in glob.glob(os.path.join(DATA, "outputs", "*")):
        if now - os.path.getmtime(path) > OUTPUT_TTL:
            shutil.rmtree(path, ignore_errors=True)


def _render_job(profile_path, text, opts):
    _prune_outputs()
    renderer = render.Renderer(render.Profile(profile_path), opts)
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
    return {"pages": urls, "pdf": f"/api/output/{job}/notes.pdf",
            "composed": "".join(sorted(renderer.factory.composed)),
            "fallback": "".join(sorted(renderer.factory.fallback))}


def _check_text(text):
    if not text.strip():
        return "변환할 글이 없습니다. PDF를 올리거나 글을 입력해 주세요."
    if len(text) > MAX_TEXT:
        return f"글이 너무 깁니다 (최대 {MAX_TEXT}자)."
    return None


@app.post("/api/render")
def render_notes():
    body = request.get_json(force=True, silent=True) or {}
    path = _profile_path(body.get("profile_id", ""))
    text = (body.get("text") or "").strip("\n")
    problem = _check_text(text)
    if problem:
        return _error(problem)
    return jsonify(_render_job(path, text, _options(body)))


@app.post("/api/convert")
def convert():
    """One step: document + handwriting (template or free-form) -> handwritten notes."""
    form = request.form
    text = form.get("text", "").strip("\n")
    document = request.files.get("document")
    source = "text"
    if not text.strip() and document and document.filename:
        try:
            text = textio.extract_text(document.read(), document.filename, form.get("keep_lines") == "1")
        except textio.TextError as exc:
            return _error(str(exc))
        source = "document"
    problem = _check_text(text)
    if problem:
        return _error(problem)

    files = [f for f in request.files.getlist("handwriting") if f.filename]
    transcript = form.get("transcript", "")[:MAX_TEXT]
    warnings, empty = [], []
    if files:
        uploads = [(f.read(), f.filename) for f in files]
        digest = hashlib.sha256()
        for data, _ in uploads:
            digest.update(hashlib.sha256(data).digest())
        digest.update(charsets.normalize_text(transcript).strip().encode())
        profile_id = digest.hexdigest()[:10]  # same files again -> reuse what was learned
        path = extract.profile_dir(DATA, profile_id)
        if os.path.exists(os.path.join(path, "profile.json")):
            with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
                profile = json.load(f)
        else:
            try:
                profile, empty, warnings = extract.build_profile(DATA, uploads, transcript=transcript,
                                                                 new_id=profile_id)
            except extract.ExtractError as exc:
                return _error(str(exc))
    elif form.get("profile_id"):
        path = _profile_path(form.get("profile_id"))
        with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
            profile = json.load(f)
    else:
        return _error("내 글씨체 파일(템플릿 사진 또는 손글씨 사진)을 올려 주세요.")
    if not profile["glyphs"]:
        return _error("글씨체 파일에서 글자를 찾지 못했습니다.\n" + "\n".join(warnings))

    result = _render_job(extract.profile_dir(DATA, profile["id"]), text, _options(form))
    result.update(profile=_profile_summary(profile), warnings=warnings, empty=empty,
                  text_source=source, text=text if source == "document" else None)
    return jsonify(result)


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
