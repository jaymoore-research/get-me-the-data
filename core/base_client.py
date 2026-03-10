"""Async HTTP client with caching and rate-limiting. Adapted from ClawBio."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

DEFAULT_CACHE_TTL = 86400  # 24 hours
DEFAULT_TIMEOUT = 30


class BaseClient:
    """Rate-limited, caching async HTTP client."""

    def __init__(
        self,
        base_url: str,
        user_agent: str = "GetMeTheData/0.1.0",
        rate_interval: float = 0.25,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_interval = rate_interval
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._last_request_time = 0.0

        self.client = httpx.AsyncClient(
            headers={
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
            timeout=timeout,
            follow_redirects=True,
        )
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def close(self):
        await self.client.aclose()

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_interval:
            import asyncio
            # For sync throttle tracking only — actual sleep handled in async methods
        self._last_request_time = time.time()

    def _cache_key(self, method: str, url: str, params: dict, body: Any) -> str:
        raw = f"{method}|{url}|{json.dumps(params, sort_keys=True)}|{json.dumps(body, sort_keys=True) if body else ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[Any]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if time.time() - data.get("_cached_at", 0) < self.cache_ttl:
                return data.get("response")
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _set_cached(self, key: str, response_data: Any):
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "_cached_at": time.time(),
            "response": response_data,
        }, indent=2, default=str))

    async def get(self, endpoint: str, params: dict | None = None) -> Any:
        params = params or {}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        cache_key = self._cache_key("GET", url, params, None)

        if self.use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        import asyncio
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_interval:
            await asyncio.sleep(self.rate_interval - elapsed)
        self._last_request_time = time.time()

        resp = await self.client.get(url, params=params)

        if resp.status_code == 429:
            await asyncio.sleep(2.0)
            resp = await self.client.get(url, params=params)

        resp.raise_for_status()
        data = resp.json()

        if self.use_cache:
            self._set_cached(cache_key, data)
        return data

    async def get_raw(self, url: str, params: dict | None = None) -> httpx.Response:
        """GET returning raw response (for downloading PDFs, HTML, etc.)."""
        import asyncio
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_interval:
            await asyncio.sleep(self.rate_interval - elapsed)
        self._last_request_time = time.time()

        resp = await self.client.get(url, params=params)
        resp.raise_for_status()
        return resp
