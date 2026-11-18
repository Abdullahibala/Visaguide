"""Two chunking strategies with the same output type, so they can be compared directly.

A) fixed     - sliding window of N words with overlap. Ignores structure.
               Simple and predictable, but may cut a list of requirements in half
               or separate a heading from its content.
B) structure - split at headings, then pack whole paragraphs up to a size limit.
               Keeps "Documents required:" together with its list, and records the
               heading path, which we prepend to the text we embed.

Invariant for both: doc_text[chunk.start:chunk.end] == chunk.text.
The evaluation relies on this. Gold passages are character ranges in the document,
so "did we retrieve the right passage?" means "does a retrieved chunk's range overlap
the gold range?". That works for any chunking strategy.
"""

import re
from dataclasses import asdict, dataclass

WORD = re.compile(r"\S+")
# A heading is a markdown heading line or a bold-only line (trafilatura turns many
# HTML headings and FAQ questions into **bold** lines). Bold headings rank below
# markdown headings in the heading path.
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
BOLD_HEADING = re.compile(r"^\*\*([^*\n]{3,150})\*\*:?$")
BLOCK = re.compile(r"[^\n]+(?:\n[^\n]+)*")  # consecutive non-empty lines = one block


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str        # exact substring of the document
    heading: str     # heading path at the chunk's start ("" for fixed chunks)
    title: str
    url: str
    country: str

    def embed_text(self) -> str:
        """What we embed and index: document title (+ heading) + body.

        A chunk such as "You need 3 copies" means little alone. Adding "Student
        visa - German Missions in India > Required documents" gives both BM25 and
        the embedding the context that a reader would get from the page.
        """
        heading = self.heading
        # Pages often repeat their title as the H1; don't say it twice.
        first, _, rest = heading.partition(" > ")
        if first and first.lower() in self.title.lower():
            heading = rest
        parts = [self.title, heading, self.text]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return asdict(self)


def _make(doc: dict, strategy: str, i: int, start: int, end: int, heading: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc['doc_id']}#{strategy[0]}{i:04d}",
        doc_id=doc["doc_id"], start=start, end=end, text=doc["text"][start:end],
        heading=heading, title=doc.get("title", ""), url=doc.get("url", ""),
        country=doc.get("country", ""),
    )


def _word_windows(spans: list[tuple[int, int]], size: int, overlap: int):
    """Yield (start_char, end_char) for word windows over the given word spans."""
    if not spans:
        return
    step = max(1, size - overlap)
    i = 0
    while True:
        window = spans[i:i + size]
        yield window[0][0], window[-1][1]
        if i + size >= len(spans):
            break
        i += step


def chunk_fixed(doc: dict, size_words: int, overlap_words: int) -> list[Chunk]:
    spans = [m.span() for m in WORD.finditer(doc["text"])]
    return [_make(doc, "fixed", i, s, e, "")
            for i, (s, e) in enumerate(_word_windows(spans, size_words, overlap_words))]


def chunk_structure(doc: dict, max_words: int, min_words: int) -> list[Chunk]:
    text = doc["text"]
    chunks: list[Chunk] = []
    path: dict[int, str] = {}           # heading level -> heading text
    cur_start = cur_end = None          # char span of the chunk being built
    cur_words = 0
    cur_heading = ""

    def heading_path() -> str:
        return " > ".join(path[lvl] for lvl in sorted(path))

    def flush():
        nonlocal cur_start, cur_end, cur_words
        if cur_start is not None:
            chunks.append(_make(doc, "structure", len(chunks), cur_start, cur_end, cur_heading))
        cur_start = cur_end = None
        cur_words = 0

    for m in BLOCK.finditer(text):
        b_start, b_end = m.span()
        block = m.group()
        n_words = len(WORD.findall(block))

        md, bold = MD_HEADING.match(block), BOLD_HEADING.match(block)
        if md or bold:
            # A new section starts. Close the current chunk unless it's too small
            # to stand alone (then the small piece is merged forward).
            if cur_words >= min_words:
                flush()
            level = len(md.group(1)) if md else 7
            path = {lvl: t for lvl, t in path.items() if lvl < level}
            path[level] = (md.group(2) if md else bold.group(1)).strip()

        if n_words > max_words:
            # One very long paragraph (common in PDFs and legal text): fall back
            # to word windows inside it, keeping the current heading.
            flush()
            spans = [(b_start + s, b_start + e)
                     for s, e in (w.span() for w in WORD.finditer(block))]
            for s, e in _word_windows(spans, max_words, max_words // 5):
                cur_heading = heading_path()
                cur_start, cur_end, cur_words = s, e, max_words
                flush()
            continue

        if cur_start is not None and cur_words + n_words > max_words:
            flush()
        if cur_start is None:
            cur_start, cur_heading = b_start, heading_path()
        cur_end = b_end
        cur_words += n_words

    flush()
    return chunks


def chunk_document(doc: dict, strategy: str, cfg) -> list[Chunk]:
    """cfg is the ChunkingCfg section of the config."""
    if strategy == "fixed":
        return chunk_fixed(doc, cfg.fixed.size_words, cfg.fixed.overlap_words)
    if strategy == "structure":
        return chunk_structure(doc, cfg.structure.max_words, cfg.structure.min_words)
    raise ValueError(f"unknown chunking strategy: {strategy}")


STRATEGIES = ("fixed", "structure")
