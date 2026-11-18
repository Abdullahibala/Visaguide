"""Abuse protection for a public demo that spends real money per request.

Two layers, because each covers the other's weakness:

1. RateLimiter, a per-client sliding window (N per minute, M per day). Stops one user
   or script from hammering the API. Weakness: an attacker can rotate IP addresses.
2. DailyBudget, a global cap on today's generation cost in USD. However many IPs are
   used, spending stops at the cap until midnight UTC. This is what actually protects
   the API key.

Both are in memory: simple, no extra service. Trade-off: the state resets on restart
and is per instance, so the deployment runs a single instance. At larger scale they
would move to Redis or a similar shared store.

Clocks are injectable so the tests can move time forward instead of sleeping.
"""

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int,
                 clock: Callable[[], float] = time.monotonic):
        self.windows = ((60.0, per_minute), (86_400.0, per_day))
        self.clock = clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request from `key` if allowed. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self.lock:
            q = self.hits[key]
            longest = max(w for w, _ in self.windows)
            while q and now - q[0] >= longest:  # forget hits older than the longest window
                q.popleft()
            for window, limit in self.windows:
                in_window = [t for t in q if now - t < window]
                if len(in_window) >= limit:
                    return False, int(window - (now - in_window[0])) + 1
            q.append(now)
            return True, 0


class DailyBudget:
    def __init__(self, limit_usd: float,
                 today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat()):
        self.limit = limit_usd
        self.today = today
        self.day = today()
        self.spent = 0.0
        self.lock = threading.Lock()

    def _roll(self) -> None:
        if self.today() != self.day:
            self.day, self.spent = self.today(), 0.0

    def available(self) -> bool:
        with self.lock:
            self._roll()
            return self.spent < self.limit

    def add(self, usd: float | None) -> None:
        # Unknown cost (unpriced model) is counted as zero but logged as such, so a
        # pricing gap shows up in the logs instead of silently disabling the cap.
        with self.lock:
            self._roll()
            self.spent += usd or 0.0

    def status(self) -> dict:
        with self.lock:
            self._roll()
            return {"day": self.day, "spent_usd": round(self.spent, 4), "limit_usd": self.limit}
