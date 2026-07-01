"""API-key authentication and a simple in-memory sliding-window rate limiter.

For a single instance this is sufficient; in a multi-instance deployment you would
back the rate limiter with Redis. The interface stays the same.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from .config import settings

# key -> timestamps of recent requests
_HITS: dict[str, deque[float]] = defaultdict(deque)


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Validate the ``x-api-key`` header against the configured allow-list."""
    if not x_api_key or x_api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


def enforce_rate_limit(request: Request, api_key: str) -> None:
    """Sliding 60-second window per API key."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    window_start = now - 60.0
    hits = _HITS[api_key]
    while hits and hits[0] < window_start:
        hits.popleft()
    if len(hits) >= limit:
        retry_after = max(1, int(60 - (now - hits[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )
    hits.append(now)


def reset_rate_limits() -> None:
    """Test helper."""
    _HITS.clear()
