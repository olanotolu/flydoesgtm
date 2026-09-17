"""Server-side Clay Public API adapter.

The browser never receives the Clay API key. Configure CLAY_PUBLIC_API_KEY
only in the server environment. Endpoint paths follow Clay's current Public API:
https://developers.clay.com/llms.txt
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("CLAY_API_BASE_URL", "https://api.clay.com/public/v0").rstrip("/")


class ClayAPIError(RuntimeError):
    pass


class ClayClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20) -> None:
        self.api_key = api_key or os.environ.get("CLAY_PUBLIC_API_KEY")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if not self.configured:
            raise ClayAPIError("CLAY_PUBLIC_API_KEY is not configured on the server")
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{BASE_URL}/{path.lstrip('/')}",
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "clay-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ClayAPIError(f"Clay API {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ClayAPIError(f"Clay API unavailable: {exc.reason}") from exc

    def authenticated_user(self) -> Any:
        return self._request("GET", "/me")

    def create_search(self, query: str) -> Any:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        return self._request("POST", "/search/query-mode", {"query": query})

    def run_routine(self, routine_id: str, items: list[dict[str, Any]], webhook_id: str | None = None) -> Any:
        if not routine_id.strip():
            raise ValueError("routine_id is required")
        if not 1 <= len(items) <= 100:
            raise ValueError("routine items must contain 1-100 items")
        body: dict[str, Any] = {"items": items}
        if webhook_id:
            body["webhook_id"] = webhook_id
        return self._request("POST", f"/routines/{routine_id.strip()}/run", body)

    def routine_results(self, routine_run_id: str) -> Any:
        return self._request("GET", f"/routines/run/{routine_run_id.strip()}/results")


def verify_webhook_signature(body: bytes, signature: str, secret: str | None = None) -> bool:
    secret = secret or os.environ.get("CLAY_WEBHOOK_SIGNING_SECRET", "")
    if not secret or not signature:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
