from __future__ import annotations
import re
from bs4 import BeautifulSoup

# Tags that are almost always noise for “main text”
DROP_TAGS = {
    "script", "style", "noscript", "svg", "canvas", "iframe",
    "header", "footer", "nav", "aside", "form"
}

def html_to_text(html: str) -> str:
    """
    Very simple, robust HTML -> clean text extractor.
    Not perfect, but good enough for an MVP.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noisy tags
    for tag in soup.find_all(list(DROP_TAGS)):
        tag.decompose()

    # Prefer main-ish content if present
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile("content|main", re.I)) or soup.body
    if main is None:
        main = soup

    text = main.get_text(separator="\n")

    # Clean up whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty
    text = "\n".join(lines)

    # Collapse too many newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
