"""Structured query log: one JSON object per /ask request.

Written to stdout (Cloud Run ships stdout to Cloud Logging automatically, since the
container's disk is wiped on restart) and appended to a local JSONL file for
analysis during development.

Privacy: client IPs are stored only as a salted hash, enough to spot one client
abusing the API, not enough to identify a person. Questions are logged because
evaluating real usage needs them; the UI says so.
"""

import hashlib
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path


def hash_ip(ip: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:16]


class QueryLogger:
    def __init__(self, path: Path | None, stream=sys.stdout):
        self.path = path
        self.stream = stream
        self.lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **record}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.lock:
            print(line, file=self.stream, flush=True)
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
