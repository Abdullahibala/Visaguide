"""Locate a copied quote inside a document and return exact character offsets.

The annotator copies a passage from the processed text or from the live web page.
Copies differ from the stored text in small ways: line breaks, double spaces,
markdown ** and # markers, bullet symbols, letter case. So we compare normalised
versions while keeping a map from each normalised character back to its position
in the original text. The offsets we return are always into the original text.
"""

import unicodedata

IGNORED = set("*#•")


def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, drop markdown/bullet symbols, collapse whitespace.

    Returns (normalised_text, index_map), where index_map[j] is the position in
    `text` of normalised character j.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if ch in IGNORED:
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        # .lower() can return 2 characters: Turkish "İ" -> "i" + combining dot.
        # Drop the combining mark so that "ikamet" matches "İkamet".
        for low in ch.lower():
            if unicodedata.combining(low):
                continue
            out.append(low)
            idx.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalise(text: str) -> str:
    return normalise_with_map(text)[0]


def find_all_quotes(doc_text: str, quote: str) -> list[tuple[int, int]]:
    """(start, end) of every occurrence of the quote, in document order."""
    q = normalise(quote)
    if not q:
        return []
    norm, idx = normalise_with_map(doc_text)
    spans, pos = [], norm.find(q)
    while pos >= 0:
        spans.append((idx[pos], idx[pos + len(q) - 1] + 1))
        pos = norm.find(q, pos + 1)
    return spans


def find_quote(doc_text: str, quote: str) -> tuple[int, int, int] | None:
    """Return (start, end, n_occurrences) of the first match, or None if absent."""
    spans = find_all_quotes(doc_text, quote)
    if not spans:
        return None
    return spans[0][0], spans[0][1], len(spans)
