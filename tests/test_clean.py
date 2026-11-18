from visaguide.ingest.clean import html_to_text, normalise_text, remove_repeated_blocks
from visaguide.ingest.fetch import doc_id_for


def test_repeated_blocks_removed_unique_content_kept():
    cookie = "We use cookies on this website."
    pages = [f"{cookie}\n\nUnique content {i} about permits." for i in range(4)]
    cleaned = remove_repeated_blocks(pages)
    assert all(cookie not in t for t in cleaned)
    assert cleaned[2] == "Unique content 2 about permits."


def test_repeated_blocks_untouched_for_small_sites():
    pages = ["Menu\n\nA", "Menu\n\nB"]  # only 2 pages: too little evidence
    assert remove_repeated_blocks(pages) == pages


def test_cookie_modal_dropped_and_accordion_content_kept():
    cookie = " ".join(["We process personal data and use cookies on this website."] * 30)
    html = f"""<html><head><title>FAQ</title></head><body>
      <div class="modal fade" role="dialog"><div class="modal-body"><p>{cookie}</p></div></div>
      <main><h1>Work permit FAQ</h1>
      <div class="accordion"><h5>Can students work?</h5>
        <div class="collapse"><p>Associate and undergraduate students may work after their
        first year. Graduate students can work without this restriction, subject to a
        work permit from the Ministry of Labour and Social Security.</p></div></div>
      </main></body></html>"""
    _, text = html_to_text(html)
    assert "cookies" not in text
    assert "Graduate students can work" in text


def test_normalise_collapses_whitespace_and_blank_lines():
    raw = "Visa  \t rules\r\n\r\n\r\n\r\nBlocked­account  "
    assert normalise_text(raw) == "Visa rules\n\nBlockedaccount"


def test_normalise_is_idempotent():
    # Gold passages are char offsets into cleaned text, so cleaning twice must change nothing.
    text = "# Heading\n\n\n\nSome  text\n- item"
    once = normalise_text(text)
    assert normalise_text(once) == once


def test_html_extraction_drops_navigation_keeps_heading():
    html = """<html><head><title>Student visa</title></head><body>
      <nav><a href="/">Home</a> | <a href="/contact">Contact</a></nav>
      <main><h1>Student visa</h1>
      <p>Applicants must provide proof of financial resources for the first year of study.
      This is usually done with a blocked account at a bank in Germany.</p>
      <p>The visa application must be submitted at the German mission responsible
      for your place of residence, well before the start of your studies.</p></main>
      <footer>Imprint | Privacy</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "Student visa"
    assert "blocked account" in text
    assert "Imprint" not in text


def test_doc_id_is_stable():
    url = "https://example.org/page"
    assert doc_id_for(url, "germany") == doc_id_for(url, "germany")
    assert doc_id_for(url, "germany").startswith("ge_")
