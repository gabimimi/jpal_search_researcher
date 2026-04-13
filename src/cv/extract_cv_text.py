from __future__ import annotations

import io
import re
from typing import Optional
from bs4 import BeautifulSoup

from pypdf import PdfReader
from docx import Document


def sniff_filetype(url: str, content_type: str | None, first_bytes: bytes) -> str:
    """
    Returns: 'pdf' | 'docx' | 'html' | 'unknown'
    """
    u = (url or "").lower()
    ct = (content_type or "").lower()

    if u.endswith(".pdf") or "pdf" in ct or first_bytes.startswith(b"%PDF"):
        return "pdf"
    if u.endswith(".docx") or "officedocument.wordprocessingml.document" in ct or first_bytes.startswith(b"PK"):
        # DOCX is a zip, starts with PK
        return "docx"
    if "text/html" in ct or b"<html" in first_bytes.lower() or b"<!doctype html" in first_bytes.lower():
        return "html"
    return "unknown"


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def extract_docx_text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text and para.text.strip():
            parts.append(para.text.strip())
    return "\n".join(parts).strip()


def extract_html_text(data: bytes) -> str:
    html = data.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # remove noisy tags
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
