"""Write the pages of a PDF by hand in several processes at once.

A page takes a few seconds of pure computation, so a long document is written several pages
at a time, one per processor core.  Each worker keeps the handwriting profile and the open
PDF between pages; every page gets its own seed, so the result does not depend on which
worker wrote it.
"""
import io
import os

_cache = {}


def _profile(path):
    from . import render
    stamp = os.path.getmtime(os.path.join(path, "profile.json"))
    if _cache.get("profile_key") != (path, stamp):
        _cache["profile"] = render.Profile(path)
        _cache["profile_key"] = (path, stamp)
    return _cache["profile"]


def _document(pdf_path):
    import pymupdf
    if _cache.get("pdf_path") != pdf_path:
        if _cache.get("pdf") is not None:
            _cache["pdf"].close()
        _cache["pdf"] = pymupdf.open(pdf_path)
        _cache["pdf_path"] = pdf_path
    return _cache["pdf"]


def write_page(profile_path, pdf_path, index, opts, seed):
    """JPEG bytes of page ``index`` (0-based) of the PDF at ``pdf_path``, written by hand."""
    from . import pdfnotes
    writer = pdfnotes.PageWriter(_profile(profile_path), opts, seed)
    image = writer.write(_document(pdf_path)[index])
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def worker_count():
    """Leave one core for the rest of the computer."""
    return max(1, min(6, (os.cpu_count() or 2) - 1))
