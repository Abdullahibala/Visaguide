"""Polite fetching of source pages: robots.txt check, per-host delay, raw snapshot on disk.

We keep the *raw* bytes exactly as served. Cleaning is a separate step, so if we
change the cleaner we can re-run it without hitting the websites again.
"""

import hashlib
import ssl
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import truststore


def doc_id_for(url: str, country: str) -> str:
    """Stable ID derived from the URL, so re-fetching gives the same ID."""
    return f"{country[:2]}_{hashlib.sha1(url.encode()).hexdigest()[:10]}"


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    raw_path: Path | None
    sha256: str
    retrieved_at: str
    error: str = ""


class PoliteFetcher:
    def __init__(self, user_agent: str, timeout_s: float, delay_s: float):
        self.client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            follow_redirects=True,
            # Use the OS certificate store: it can complete chains for servers that
            # omit intermediate certificates. We never disable verification instead.
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )
        self.user_agent = user_agent
        self.delay_s = delay_s
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _wait_for_host(self, host: str) -> None:
        elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_hit[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._wait_for_host(parts.netloc)
                r = self.client.get(base + "/robots.txt")
                # No robots.txt (4xx) means everything is allowed, by convention.
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(self.user_agent, url)

    def fetch(self, url: str, out_dir: Path, doc_id: str) -> FetchResult:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        if not self.allowed(url):
            return FetchResult(url, url, 0, "", None, "", now, error="disallowed by robots.txt")
        r, last_err = None, ""
        for attempt in range(2):  # one retry for transient failures (timeouts, 5xx)
            self._wait_for_host(urlparse(url).netloc)
            try:
                r = self.client.get(url)
                if r.status_code < 500:
                    break
                last_err = f"HTTP {r.status_code}"
            except httpx.TimeoutException as e:
                last_err, r = f"{type(e).__name__}: {e}", None
            except httpx.HTTPError as e:  # e.g. TLS errors: retrying will not help
                return FetchResult(url, url, 0, "", None, "", now,
                                   error=f"{type(e).__name__}: {e}")
            time.sleep(5 * (attempt + 1))
        if r is None:
            return FetchResult(url, url, 0, "", None, "", now, error=last_err)

        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if r.status_code != 200:
            return FetchResult(url, str(r.url), r.status_code, ctype, None, "", now,
                               error=f"HTTP {r.status_code}")

        is_pdf = ctype == "application/pdf" or r.content[:5] == b"%PDF-"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"{doc_id}.{'pdf' if is_pdf else 'html'}"
        raw_path.write_bytes(r.content)
        return FetchResult(
            url=url,
            final_url=str(r.url),
            status=200,
            content_type="application/pdf" if is_pdf else ctype,
            raw_path=raw_path,
            sha256=hashlib.sha256(r.content).hexdigest(),
            retrieved_at=now,
        )
