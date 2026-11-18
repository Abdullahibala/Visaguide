import pytest

from visaguide.chunking import WORD, chunk_fixed, chunk_structure

DOC_TEXT = """# Student visa

You need a national visa before you travel. Apply at the German mission responsible
for your place of residence.

## Required documents

**Proof of finances**

A blocked account with the amount set for the current year. Bank statements are not accepted.

**Health insurance**

Travel health insurance valid until your enrolment.

## Processing time

""" + " ".join(f"word{i}" for i in range(450))  # one very long paragraph


def doc(text=DOC_TEXT):
    return {"doc_id": "de_test", "text": text, "title": "Test", "url": "u", "country": "germany"}


@pytest.mark.parametrize("make", [
    lambda d: chunk_fixed(d, size_words=50, overlap_words=10),
    lambda d: chunk_structure(d, max_words=60, min_words=15),
])
def test_offsets_are_exact_substrings(make):
    d = doc()
    for c in make(d):
        assert d["text"][c.start:c.end] == c.text


@pytest.mark.parametrize("make", [
    lambda d: chunk_fixed(d, size_words=50, overlap_words=10),
    lambda d: chunk_structure(d, max_words=60, min_words=15),
])
def test_every_word_is_covered(make):
    # If a word falls between chunks, it can never be retrieved.
    d = doc()
    chunks = make(d)
    for m in WORD.finditer(d["text"]):
        assert any(c.start <= m.start() and m.end() <= c.end for c in chunks), m.group()


def test_fixed_respects_size_and_overlap():
    chunks = chunk_fixed(doc(), size_words=50, overlap_words=10)
    words = [WORD.findall(c.text) for c in chunks]
    assert all(len(w) <= 50 for w in words)
    assert words[0][-10:] == words[1][:10]  # consecutive windows share 10 words


def test_structure_respects_max_words():
    chunks = chunk_structure(doc(), max_words=60, min_words=15)
    assert all(len(WORD.findall(c.text)) <= 60 for c in chunks)


def test_structure_keeps_heading_with_its_content_and_records_path():
    chunks = chunk_structure(doc(), max_words=60, min_words=15)
    finance = next(c for c in chunks if "blocked account" in c.text)
    assert "Required documents" in finance.heading
    # the long paragraph is split, but every piece keeps its section heading
    long_parts = [c for c in chunks if "word300" in c.text]
    assert long_parts and all("Processing time" in c.heading for c in long_parts)


def test_structure_merges_tiny_sections_forward():
    # "**Health insurance**" + one sentence is under min_words, so it must not be
    # a chunk of its own when a larger limit lets it merge.
    chunks = chunk_structure(doc(), max_words=200, min_words=40)
    assert not any(c.text.strip() == "**Health insurance**" for c in chunks)


def test_tiny_document_gives_one_chunk():
    for chunks in (chunk_fixed(doc("Just one line."), 50, 10),
                   chunk_structure(doc("Just one line."), 60, 15)):
        assert len(chunks) == 1 and chunks[0].text == "Just one line."


def test_chunk_ids_unique_and_embed_text_has_context():
    chunks = chunk_structure(doc(), max_words=60, min_words=15)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    finance = next(c for c in chunks if "blocked account" in c.text)
    assert finance.embed_text().startswith("Test\n")
    assert "Required documents" in finance.embed_text()


def test_embed_text_does_not_repeat_title_heading():
    d = {"doc_id": "x", "text": "# Student visa\n\n" + "Some words here. " * 20,
         "title": "Student visa - German Missions", "url": "u", "country": "germany"}
    c = chunk_structure(d, max_words=200, min_words=10)[0]
    assert c.embed_text().count("Student visa") == 2  # title line + the body's "# Student visa"
