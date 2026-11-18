"""Turn raw HTML/PDF into clean text.

Design choices:
- HTML -> trafilatura with markdown output. It removes navigation, footers and
  cookie banners, and keeps headings as "# ..." lines. We need those headings later
  for structure-aware chunking.
- PDF -> PyMuPDF page text, pages separated by a blank line.
- normalise_text() is deterministic. The test set stores gold passages as character
  ranges into this text, so the same input must always give the same output.
"""

import re
import unicodedata
from pathlib import Path

import pymupdf
import trafilatura
from lxml import etree
from lxml import html as LH


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphens break BM25 tokens
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line between blocks
    return text.strip()


_DROP_XPATH = (
    '//nav | //*[@role="navigation" or @role="dialog" or @role="alertdialog"'
    ' or contains(@id, "nav")'
    ' or contains(concat(" ", normalize-space(@class), " "), " modal ")'
    ' or contains(translate(@id, "COKIE", "cokie"), "cookie")'
    ' or contains(translate(@class, "COKIE", "cokie"), "cookie")]'
)
_UNHIDE_XPATH = '//*[@aria-hidden="true" or @hidden or contains(@class, "collapse")]'


def _preprocess_html(html: str) -> str:
    """Fix two patterns that mislead trafilatura on official sites.

    1. Modal dialogs (cookie/privacy popups) can be the largest text block on the
       page, so the extractor picks them as the "main content". Drop them, along
       with navigation menus (some sites put a <nav> inside <main>).
    2. FAQ answers often sit in collapsed accordion panels. The extractor treats
       hidden elements as non-content, so un-hide them.
    """
    try:
        tree = LH.fromstring(html)
    except (etree.ParserError, ValueError):
        return html
    for el in tree.xpath(_DROP_XPATH):
        if el.getparent() is not None:
            el.drop_tree()
    for el in tree.xpath(_UNHIDE_XPATH):
        el.attrib.pop("aria-hidden", None)
        el.attrib.pop("hidden", None)
        if "class" in el.attrib:
            el.set("class", " ".join(c for c in el.get("class").split()
                                     if c not in {"collapse", "hidden"}))
    return LH.tostring(tree, encoding="unicode")


def _trafilatura(html: str, url: str) -> str:
    return trafilatura.extract(
        html,
        url=url or None,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        include_comments=False,
        favor_recall=True,  # official pages are short; we would rather keep a bit too much
    ) or ""


def _content_region(html: str) -> str | None:
    """HTML of the explicit content region: the accordion blocks (FAQ pages) or <main>."""
    try:
        tree = LH.fromstring(html)
    except (etree.ParserError, ValueError):
        return None
    accordions = tree.xpath('//*[contains(concat(" ", normalize-space(@class), " "),'
                            ' " accordion ")][not(ancestor::*[contains(concat(" ",'
                            ' normalize-space(@class), " "), " accordion ")])]')
    region = accordions or tree.xpath('//main | //*[@role="main"]')[:1]
    if not region:
        return None
    body = "".join(LH.tostring(el, encoding="unicode") for el in region)
    return f"<html><body><main>{body}</main></body></html>"


def html_to_text(html: str, url: str = "") -> tuple[str, str]:
    """Return (title, text). Empty text means extraction failed."""
    meta = trafilatura.extract_metadata(html)  # from the untouched page
    html = _preprocess_html(html)
    text = _trafilatura(html, url)

    # Fallback: on FAQ/accordion-heavy pages trafilatura's heuristics sometimes pick
    # a menu or teaser. If the page's explicit content region has far more text,
    # extract from that region instead.
    region = _content_region(html)
    if region:
        region_len = len(LH.fromstring(region).text_content().strip())
        if region_len > 3 * len(text):
            text = _trafilatura(region, url) or text

    title = (meta.title if meta and meta.title else "").strip()
    return title, normalise_text(text)


def pdf_to_text(path: Path) -> tuple[str, str]:
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title", "") or ""
        pages = [page.get_text("text") for page in doc]
    return title.strip(), normalise_text("\n\n".join(pages))


def remove_repeated_blocks(texts: list[str], min_docs: int = 3,
                           min_share: float = 0.5) -> list[str]:
    """Remove blocks (paragraphs split on blank lines) that repeat across one site's pages.

    trafilatura works on one page at a time, so it can't know that a cookie popup or a
    menu appears on every page of a site. Across pages it's obvious. `texts` must all
    come from the same domain. If a block appears in >= min_share of them (and there
    are at least min_docs pages), it's boilerplate.
    """
    if len(texts) < min_docs:
        return texts
    split = [t.split("\n\n") for t in texts]
    doc_freq: dict[str, int] = {}
    for blocks in split:
        for b in set(blocks):
            doc_freq[b] = doc_freq.get(b, 0) + 1
    threshold = max(2, min_share * len(texts))
    boiler = {b for b, n in doc_freq.items() if n >= threshold}
    return [normalise_text("\n\n".join(b for b in blocks if b not in boiler)) for blocks in split]


def extract(raw_path: Path, url: str = "") -> tuple[str, str]:
    if raw_path.suffix == ".pdf":
        return pdf_to_text(raw_path)
    html = raw_path.read_bytes().decode("utf-8", errors="replace")
    return html_to_text(html, url)
