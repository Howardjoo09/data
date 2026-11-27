"""손글씨 노트 변환기 — prototype web server.

    python server.py            # then open http://127.0.0.1:5000
"""
import glob
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import zipfile

import pymupdf
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from hwnote import charsets, extract, pageworker, pdfnotes, render, simple, style, template as T, textio

# Inside the packaged app the program files are unpacked to a temporary folder.
BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
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
    return send_file(os.path.join(BASE, "static", "app.html"))


@app.get("/quick")
def quick():
    """The one-step page that learns the handwriting from the uploaded files each time."""
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


def _known_characters(profile_id=""):
    """Characters a handwriting (the given profile, else my handwriting) already has."""
    path = None
    if profile_id and PROFILE_RE.match(profile_id):
        path = extract.profile_dir(DATA, profile_id)
    else:
        _, path = _my_profile()
    if not path or not os.path.exists(os.path.join(path, "profile.json")):
        return ""
    with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
        return "".join({g["char"] for g in json.load(f)["glyphs"]})


@app.post("/api/template/scan")
def scan_document_for_template():
    """The characters of a PDF (or text file), most frequent first, for a template that has
    exactly what that document needs.  ``extra`` repeats the frequent ones (2 more copies at
    60+ uses, 1 more at 20+), so they get written in several versions."""
    upload = request.files.get("document")
    if not upload or not upload.filename:
        return _error("PDF 또는 TXT 파일을 골라 주세요.")
    data = upload.read()
    if data[:5] == b"%PDF-" or upload.filename.lower().endswith(".pdf"):
        try:
            counts, pages = pdfnotes.document_characters(data)
        except Exception:
            return _error("PDF를 열 수 없습니다.")
    else:
        try:
            text = textio.extract_text(data, upload.filename)
        except textio.TextError as exc:
            return _error(str(exc))
        from collections import Counter
        counts = Counter(c for c in charsets.normalize_text(text) if not c.isspace() and c.isprintable())
        pages = None
    hanja = [c for c in counts if charsets.is_hanja(c)]
    if request.form.get("exclude_hanja") == "1":
        for c in hanja:
            del counts[c]
    known = set()
    if request.form.get("skip_known") == "1":
        known = set(_known_characters(request.form.get("profile_id", ""))) & set(counts)
        for c in known:
            del counts[c]
    known_all = _known_characters(request.form.get("profile_id", "")) if request.form.get("skip_known") == "1" else ""
    preview = {}
    for n in (1, 2, 3):  # what a short template of n pages would give
        chosen = simple.choose(counts, n * T.CELLS_PER_PAGE, known_all)
        preview[n] = {k: round(v, 3) for k, v in simple.coverage(counts, chosen, known_all).items()}
    ordered = [c for c, _ in counts.most_common()]
    # Further copies go round the list (x n ) ( ... then x n ...), never two of a kind in a row:
    # a character written twice in succession comes out the same twice.
    extra = "".join(c for c, n in counts.most_common() if n >= 20) + \
        "".join(c for c, n in counts.most_common() if n >= 60)
    return jsonify({"chars": "".join(ordered), "extra": extra, "unique": len(ordered), "pages": pages,
                    "counts": dict(counts), "simple": preview,
                    "hanja": len(hanja), "hanja_removed": request.form.get("exclude_hanja") == "1",
                    "known": len(known), "uses": sum(counts.values())})


@app.post("/api/template")
def make_template():
    """A template from character sets and text (every character), or a short one ("simple":
    1-3 pages) for a document given as character counts or text (see hwnote/simple.py)."""
    from collections import Counter
    body = request.get_json(force=True, silent=True) or {}
    skip = _known_characters(str(body.get("profile_id", ""))) if body.get("skip_known") else ""
    coverage = None
    if body.get("simple"):
        pages = int(body.get("simple")) if str(body.get("simple")).isdigit() else 1
        pages = min(3, max(1, pages))
        raw = body.get("counts") or {}
        if raw:
            counts = Counter({str(c)[:1]: int(n) for c, n in raw.items() if str(c) and int(n) > 0})
        else:
            counts = Counter(c for c in charsets.normalize_text(body.get("text", "")[:MAX_TEXT])
                             if not c.isspace() and c.isprintable())
        if body.get("exclude_hanja"):
            counts = Counter({c: n for c, n in counts.items() if not charsets.is_hanja(c)})
        chars = simple.choose(counts, pages * T.CELLS_PER_PAGE, skip)
        coverage = {k: round(v, 3) for k, v in simple.coverage(counts, chars, skip).items()}
    else:
        chars = charsets.build_charlist(body.get("sets", []), body.get("text", "")[:MAX_TEXT], body.get("repeat", 1),
                                        exclude_hanja=bool(body.get("exclude_hanja")),
                                        extra=str(body.get("extra", ""))[:MAX_TEXT], skip=skip)
    if not chars:
        return _error("템플릿에 넣을 글자를 하나 이상 고르세요.")
    if len(chars) > T.CELLS_PER_PAGE * 20:
        return _error(f"글자가 너무 많습니다 ({len(chars)}칸). 20쪽 이하로 줄여 주세요.")
    info = T.new_template(DATA, chars)
    with open(os.path.join(DATA, "templates", f"{info['id']}.pdf"), "wb") as f:
        f.write(T.build_pdf(info))
    return jsonify({"id": info["id"], "pages": len(info["pages"]), "cells": len(chars),
                    "pdf": f"/api/template/{info['id']}.pdf", "coverage": coverage})


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


@app.get("/api/profile/<profile_id>/glyphs")
def profile_glyphs(profile_id):
    """Every learned copy of every character, for choosing which ones to use."""
    with open(os.path.join(_profile_path(profile_id), "profile.json"), encoding="utf-8") as f:
        profile = json.load(f)
    excluded = set(profile.get("excluded", []))
    return jsonify([{"char": g["char"], "file": g["file"], "excluded": g["file"] in excluded}
                    for g in profile["glyphs"]])


@app.get("/api/profile/<profile_id>/glyph/<name>")
def profile_glyph(profile_id, name):
    if not re.match(r"^g\d{4,6}\.png$", name):
        abort(404)
    return send_from_directory(_profile_path(profile_id), name, max_age=3600)


@app.post("/api/profile/<profile_id>/exclude")
def exclude_glyphs(profile_id):
    body = request.get_json(force=True, silent=True) or {}
    files = [f for f in body.get("files", []) if isinstance(f, str)]
    try:
        excluded = extract.set_excluded(_profile_path(profile_id), files, not body.get("restore"))
    except extract.ExtractError as exc:
        return _error(str(exc))
    return jsonify({"excluded": excluded})


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


# --------------------------------------------------------------------------- my handwriting
# The app remembers one handwriting ("내 글씨체") so it need not be uploaded every time.  It is
# exchanged as a zip of the profile folder (profile.json + glyph images).

PROFILE_FILE = re.compile(r"^(profile\.json|g\d{4,6}\.png)$")


def _settings():
    path = os.path.join(DATA, "settings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_settings(settings):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False)


def _my_profile():
    profile_id = _settings().get("profile_id")
    if profile_id and PROFILE_RE.match(profile_id):
        path = extract.profile_dir(DATA, profile_id)
        if os.path.exists(os.path.join(path, "profile.json")):
            return profile_id, path
    return None, None


@app.get("/api/me")
def my_handwriting():
    profile_id, path = _my_profile()
    if not profile_id:
        return jsonify({"profile": None})
    with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
        return jsonify({"profile": _profile_summary(json.load(f))})


@app.post("/api/me/import")
def import_handwriting():
    """Take a handwriting file (zip exported from this app) and make it my handwriting."""
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return _error("글씨체 파일(.zip)을 골라 주세요.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(upload.read()))
    except zipfile.BadZipFile:
        return _error("글씨체 파일이 아닙니다 (zip 파일이어야 합니다).")
    members = {}
    for info in archive.infolist():
        name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if info.is_dir() or not PROFILE_FILE.match(name):
            continue
        if info.file_size > 20 * 1024 * 1024:
            return _error("글씨체 파일 안에 너무 큰 파일이 있습니다.")
        members[name] = archive.read(info)
    if "profile.json" not in members:
        return _error("글씨체 파일에 profile.json이 없습니다.")
    try:
        profile = json.loads(members["profile.json"].decode("utf-8"))
        names = [g["file"] for g in profile["glyphs"]]
    except (ValueError, KeyError, TypeError):
        return _error("글씨체 파일의 profile.json을 읽을 수 없습니다.")
    missing = [n for n in names if n not in members]
    if missing:
        return _error(f"글씨체 파일에 글자 이미지 {len(missing)}개가 빠져 있습니다.")
    profile_id = profile.get("id") if PROFILE_RE.match(str(profile.get("id", ""))) else secrets.token_hex(5)
    profile["id"] = profile_id
    path = extract.profile_dir(DATA, profile_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    for name, data in members.items():
        if name != "profile.json":
            with open(os.path.join(path, name), "wb") as f:
                f.write(data)
    with open(os.path.join(path, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    settings = _settings()
    settings["profile_id"] = profile_id
    _save_settings(settings)
    return jsonify({"profile": _profile_summary(profile)})


@app.post("/api/me/use/<profile_id>")
def use_handwriting(profile_id):
    """Make a profile built on the advanced page my handwriting."""
    path = _profile_path(profile_id)
    settings = _settings()
    settings["profile_id"] = profile_id
    _save_settings(settings)
    with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
        return jsonify({"profile": _profile_summary(json.load(f))})


@app.get("/api/me/export")
def export_handwriting():
    profile_id, path = _my_profile()
    if not profile_id:
        abort(404)
    return send_file(io.BytesIO(profile_zip(path)), mimetype="application/zip",
                     download_name=f"my_handwriting_{profile_id}.zip")


def profile_zip(path):
    """The handwriting file: profile.json and the glyph images it names."""
    with open(os.path.join(path, "profile.json"), encoding="utf-8") as f:
        profile = json.load(f)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("profile.json", json.dumps(profile, ensure_ascii=False))
        for g in profile["glyphs"]:
            archive.write(os.path.join(path, g["file"]), g["file"])
    return buf.getvalue()


# --------------------------------------------------------------------------- conversion jobs
# A PDF page takes several seconds, so conversions run in the background, one at a time, and
# the page asks how far they have got.

JOBS = {}
JOB_LOCK = threading.Lock()
WORK_LOCK = threading.Lock()
MAX_PAGES = 200


def parse_pages(spec, count):
    """'29-31, 35' -> [28, 29, 30, 34] (0-based); empty -> every page."""
    spec = (spec or "").strip()
    if not spec:
        return list(range(count))
    pages = []
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-~]\s*(\d+))?", part)
        if not m:
            raise ValueError(f"쪽 번호를 읽을 수 없습니다: {part}")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if a < 1 or b < a or b > count:
            raise ValueError(f"이 PDF는 1~{count}쪽입니다: {part}")
        pages.extend(range(a - 1, b))
    return list(dict.fromkeys(pages))


def _job_update(job, **values):
    with JOB_LOCK:
        JOBS[job].update(values)


def _save_copy(pdf_path, name):
    """The desktop app keeps every result in a folder of the user's documents as well."""
    folder = os.environ.get("HWNOTE_SAVE_DIR")
    if not folder:
        return None
    os.makedirs(folder, exist_ok=True)
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", os.path.splitext(name or "")[0]).strip() or "글"
    target = os.path.join(folder, f"{stem}_손글씨_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
    shutil.copyfile(pdf_path, target)
    return target


_POOL = None


def _pool():
    """Worker processes for writing PDF pages, started once and kept (None: write here)."""
    global _POOL
    if _POOL is None and pageworker.worker_count() > 1 and os.environ.get("HWNOTE_WORKERS") != "1":
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            _POOL = ProcessPoolExecutor(pageworker.worker_count(), mp_context=multiprocessing.get_context("spawn"))
        except Exception:
            _POOL = False
    return _POOL or None


def _layout_pages(path, pdf_path, pages, opts, seed):
    """JPEG bytes of the written pages, in order, as they become ready.  If the worker
    processes fail (cannot start, or one is killed), the remaining pages are written here."""
    global _POOL
    work = [(i, seed + i) for i in pages]
    done = 0
    pool = _pool()
    if pool is not None:
        try:
            futures = [pool.submit(pageworker.write_page, path, pdf_path, i, opts, s) for i, s in work]
            for future in futures:
                yield future.result()
                done += 1
        except Exception:
            from concurrent.futures.process import BrokenProcessPool
            _POOL = False
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            if not isinstance(sys.exc_info()[1], BrokenProcessPool):
                raise
    for i, s in work[done:]:
        yield pageworker.write_page(path, pdf_path, i, opts, s)


def _run_job(job, path, kind, payload, opts, pages, name=""):
    out = os.path.join(DATA, "outputs", job)
    doc = pymupdf.open()
    urls = []

    def add(jpeg):
        n = len(urls) + 1
        name = f"page-{n}.jpg"
        with open(os.path.join(out, name), "wb") as f:
            f.write(jpeg)
        page = doc.new_page(width=595.28, height=841.89)
        page.insert_image(page.rect, filename=os.path.join(out, name))
        urls.append(f"/api/output/{job}/{name}")
        _job_update(job, done=n, pages=list(urls))

    try:
        with WORK_LOCK:
            _job_update(job, state="running", started=time.time())
            if kind == "layout":
                source = os.path.join(out, "source.pdf")
                with open(source, "wb") as f:
                    f.write(payload)
                for jpeg in _layout_pages(path, source, pages, opts, int(job[:6], 16)):
                    add(jpeg)
                try:
                    os.remove(source)
                except OSError:
                    pass  # still held somewhere; old outputs are cleared after a day
            else:
                renderer = render.Renderer(render.Profile(path), opts)
                images = renderer.render(payload)
                _job_update(job, total=len(images))
                for image in images:
                    buf = io.BytesIO()
                    image.save(buf, "JPEG", quality=90)
                    add(buf.getvalue())
            doc.save(os.path.join(out, "notes.pdf"), garbage=3, deflate=True)
        saved = _save_copy(os.path.join(out, "notes.pdf"), name)
        _job_update(job, state="done", pdf=f"/api/output/{job}/notes.pdf", saved=saved, finished=time.time())
    except Exception as exc:  # reported to the page instead of a silent stop
        _job_update(job, state="error", error=f"변환 중 문제가 생겼습니다: {exc}")
    finally:
        doc.close()


@app.post("/api/jobs")
def start_job():
    """Convert a PDF (keeping its layout, or rewriting its text) or typed text with my
    handwriting.  Returns a job id; GET /api/jobs/<id> tells how far it has got."""
    profile_id, path = _my_profile()
    if request.form.get("profile_id"):
        path = _profile_path(request.form.get("profile_id"))
    elif not profile_id:
        return _error("먼저 내 글씨체 파일을 불러와 주세요.")
    form = request.form
    paper = form.get("paper") if form.get("paper") in ("grid", "lined", "blank") else "grid"
    kind = form.get("mode", "layout")
    document = request.files.get("document")
    if kind == "layout":
        if not document or not document.filename:
            return _error("변환할 PDF를 골라 주세요.")
        payload = document.read()
        try:
            source = pymupdf.open(stream=payload, filetype="pdf")
            count = source.page_count
            source.close()
        except Exception:
            return _error("PDF를 열 수 없습니다.")
        try:
            pages = parse_pages(form.get("pages"), count)
        except ValueError as exc:
            return _error(str(exc))
        if len(pages) > MAX_PAGES:
            return _error(f"한 번에 {MAX_PAGES}쪽까지 변환할 수 있습니다.")
        opts = render.Options(paper=paper, margin_line=False)
        total = len(pages)
    else:
        if document and document.filename:
            try:
                payload = textio.extract_text(document.read(), document.filename, form.get("keep_lines") == "1")
            except textio.TextError as exc:
                return _error(str(exc))
        else:
            payload = form.get("text", "")
        payload = payload.strip("\n")
        problem = _check_text(payload)
        if problem:
            return _error(problem)
        size = {"small": 0.45, "medium": 0.55, "large": 0.65}.get(form.get("size"), 0.55)
        opts = render.Options(paper=paper, margin_line=paper == "lined", line_pitch=2 * render.GRID_STEP,
                              size=size)
        pages, total, kind = None, None, "text"
    _prune_outputs()
    job = secrets.token_hex(6)
    os.makedirs(os.path.join(DATA, "outputs", job))
    with JOB_LOCK:
        JOBS[job] = {"id": job, "state": "waiting", "done": 0, "total": total, "pages": [], "pdf": None,
                     "error": None, "created": time.time()}
    name = document.filename if document and document.filename else "입력한 글"
    threading.Thread(target=_run_job, args=(job, path, kind, payload, opts, pages, name), daemon=True).start()
    return jsonify({"job": job})


@app.get("/api/jobs/<job>")
def job_status(job):
    if not JOB_RE.match(job):
        abort(404)
    with JOB_LOCK:
        if job not in JOBS:
            abort(404)
        return jsonify(dict(JOBS[job]))


@app.post("/api/pdfinfo")
def pdf_info():
    upload = request.files.get("document")
    if not upload or not upload.filename:
        return _error("PDF를 골라 주세요.")
    try:
        doc = pymupdf.open(stream=upload.read(), filetype="pdf")
        count = doc.page_count
        doc.close()
    except Exception:
        return _error("PDF를 열 수 없습니다.")
    return jsonify({"pages": count})


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


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
