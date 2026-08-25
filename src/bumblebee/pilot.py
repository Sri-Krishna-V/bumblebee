"""Privacy-first pilot controls for the hosted Bumblebee API.

The OCR worker deliberately does not retain customer documents or OCR output.
This module adds the small amount of control-plane state needed by a design
partner: tenant-scoped API keys, page-metered usage, idempotency guards, and a
short operational audit trail.  It only stores request metadata; never PDF
bytes, markdown, layout JSON, or chunks.

``BUMBLEBEE_API_KEYS_JSON`` is intended to live in a Modal Secret.  Its shape
is a mapping from a tenant id to either a token string or an object::

    {
      "acme-research": {"key": "bb_live_...", "monthly_page_limit": 50000},
      "demo": "bb_demo_..."
    }

The legacy ``BUMBLEBEE_API_KEY`` remains a compatible single-tenant fallback.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_IDEMPOTENCY_KEY_LENGTH = 200


class PilotConfigError(ValueError):
    """Raised when the pilot key configuration is malformed."""


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    """A customer identity derived from a valid bearer token."""

    tenant_id: str
    token_digest: str
    monthly_page_limit: int | None


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Usage visible to the authenticated tenant for the current UTC month."""

    tenant_id: str
    month: str
    requests: int
    pages: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    monthly_page_limit: int | None

    @property
    def pages_remaining(self) -> int | None:
        """Return the remaining monthly allowance, when a tenant has one."""
        if self.monthly_page_limit is None:
            return None
        return max(0, self.monthly_page_limit - self.pages)

    def as_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-ready, tenant-safe representation."""
        return {
            "tenant_id": self.tenant_id,
            "month": self.month,
            "requests": self.requests,
            "pages": self.pages,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "monthly_page_limit": self.monthly_page_limit,
            "pages_remaining": self.pages_remaining,
        }


def utc_now() -> datetime:
    """Return an aware UTC timestamp in one place for testability."""
    return datetime.now(UTC)


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotConfigError(f"{name} must be a positive integer")
    return value


class ApiKeyRegistry:
    """Resolve bearer tokens into design-partner identities without storing tokens."""

    def __init__(self, principals: list[ApiPrincipal]) -> None:
        """Create a registry from already-hashed configured principals."""
        if len({principal.token_digest for principal in principals}) != len(principals):
            raise PilotConfigError("each configured tenant must have a distinct API key")
        self._principals = principals

    @classmethod
    def from_env(cls) -> ApiKeyRegistry:
        """Load configured tenants, falling back to the original single key."""
        raw = os.environ.get("BUMBLEBEE_API_KEYS_JSON")
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PilotConfigError("BUMBLEBEE_API_KEYS_JSON is not valid JSON") from exc
            if not isinstance(parsed, dict) or not parsed:
                raise PilotConfigError("BUMBLEBEE_API_KEYS_JSON must be a non-empty tenant mapping")
            entries = cast("dict[str, Any]", parsed)
            principals = [cls._principal_from_config(str(tenant), config) for tenant, config in entries.items()]
            return cls(principals)

        legacy_key = os.environ.get("BUMBLEBEE_API_KEY")
        if not legacy_key:
            return cls([])
        default_tenant = os.environ.get("BUMBLEBEE_DEFAULT_TENANT", "default")
        if not _TENANT_ID.fullmatch(default_tenant):
            raise PilotConfigError(f"invalid tenant id {default_tenant!r}")
        return cls(
            [
                ApiPrincipal(
                    tenant_id=default_tenant,
                    token_digest=_token_digest(legacy_key),
                    monthly_page_limit=None,
                )
            ]
        )

    @staticmethod
    def _principal_from_config(tenant_id: str, config: Any) -> ApiPrincipal:
        if not _TENANT_ID.fullmatch(tenant_id):
            raise PilotConfigError(f"invalid tenant id {tenant_id!r}")
        if isinstance(config, str):
            token = config
            page_limit = None
        elif isinstance(config, dict):
            entry = cast("dict[str, Any]", config)
            raw_token = entry.get("key")
            token = raw_token if isinstance(raw_token, str) else ""
            page_limit = _positive_int(entry.get("monthly_page_limit"), name=f"{tenant_id}.monthly_page_limit")
        else:
            raise PilotConfigError(f"{tenant_id!r} must be a token string or object")
        if not token.strip():
            raise PilotConfigError(f"{tenant_id!r} is missing a non-empty key")
        return ApiPrincipal(tenant_id=tenant_id, token_digest=_token_digest(token), monthly_page_limit=page_limit)

    @property
    def configured(self) -> bool:
        """Whether at least one valid API key was configured."""
        return bool(self._principals)

    def authenticate(self, authorization: str | None) -> ApiPrincipal | None:
        """Return the matching principal, using constant-time digest comparison."""
        if not authorization or not authorization.startswith("Bearer "):
            return None
        digest = _token_digest(authorization.removeprefix("Bearer ").strip())
        for principal in self._principals:
            if hmac.compare_digest(digest, principal.token_digest):
                return principal
        return None


class PilotUsageStore:
    """Small SQLite metadata ledger suitable for a one-container pilot.

    The store is intentionally single-writer and protected by a thread lock.
    Modal's pilot endpoint is capped at one container, while local development
    uses the same object in a single FastAPI process.  This is not a substitute
    for a multi-region billing system.
    """

    def __init__(self, path: str | Path = ":memory:", *, retention_days: int = 30) -> None:
        """Open or initialize the metadata ledger at ``path``."""
        if retention_days < 1:
            raise ValueError("retention_days must be at least one day")
        self.retention_days = retention_days
        self._lock = threading.Lock()
        location = str(path)
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(location, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                request_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                pages INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, request_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                tenant_id TEXT NOT NULL,
                key_digest TEXT NOT NULL,
                request_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, key_digest)
            )
            """
        )
        self._connection.commit()

    @classmethod
    def from_env(cls) -> PilotUsageStore:
        """Build a ledger from the deployment's non-document environment settings."""
        path = os.environ.get("BUMBLEBEE_USAGE_DB", ":memory:")
        raw_days = os.environ.get("BUMBLEBEE_AUDIT_RETENTION_DAYS", "30")
        try:
            retention_days = int(raw_days)
        except ValueError as exc:
            raise PilotConfigError("BUMBLEBEE_AUDIT_RETENTION_DAYS must be an integer") from exc
        if retention_days < 1:
            raise PilotConfigError("BUMBLEBEE_AUDIT_RETENTION_DAYS must be at least one day")
        return cls(path, retention_days=retention_days)

    def close(self) -> None:
        """Close the backing connection (primarily useful in tests)."""
        with self._lock:
            self._connection.close()

    def claim_idempotency_key(self, principal: ApiPrincipal, value: str | None, request_id: str) -> bool:
        """Reserve an optional key; return false when the request was seen before."""
        if value is None:
            return True
        key = value.strip()
        if not key or len(key) > _MAX_IDEMPOTENCY_KEY_LENGTH:
            raise ValueError(f"Idempotency-Key must be 1-{_MAX_IDEMPOTENCY_KEY_LENGTH} characters")
        now = utc_now().isoformat()
        with self._lock:
            self._prune_locked(now)
            try:
                self._connection.execute(
                    "INSERT INTO idempotency_keys (tenant_id, key_digest, request_id, created_at) VALUES (?, ?, ?, ?)",
                    (principal.tenant_id, _token_digest(key), request_id, now),
                )
            except sqlite3.IntegrityError:
                return False
            self._connection.commit()
            return True

    def release_idempotency_key(self, principal: ApiPrincipal, value: str | None) -> None:
        """Release a reservation after a failed parse so a caller may retry it."""
        if value is None:
            return
        with self._lock:
            self._connection.execute(
                "DELETE FROM idempotency_keys WHERE tenant_id = ? AND key_digest = ?",
                (principal.tenant_id, _token_digest(value.strip())),
            )
            self._connection.commit()

    def usage(self, principal: ApiPrincipal) -> UsageSnapshot:
        """Return the authenticated tenant's current UTC-month usage."""
        month = utc_now().strftime("%Y-%m")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS requests, COALESCE(SUM(pages), 0) AS pages,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM usage_events
                WHERE tenant_id = ? AND created_at >= ? AND created_at < ?
                """,
                (principal.tenant_id, f"{month}-01T00:00:00+00:00", _next_month(month)),
            ).fetchone()
        assert row is not None
        return UsageSnapshot(
            tenant_id=principal.tenant_id,
            month=month,
            requests=int(row["requests"]),
            pages=int(row["pages"]),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            total_tokens=int(row["total_tokens"]),
            monthly_page_limit=principal.monthly_page_limit,
        )

    def record(
        self,
        principal: ApiPrincipal,
        *,
        request_id: str,
        pages: int,
        duration_ms: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> UsageSnapshot:
        """Write metadata for a successful parse and return updated monthly usage."""
        now = utc_now().isoformat()
        with self._lock:
            self._prune_locked(now)
            self._connection.execute(
                """
                INSERT INTO usage_events
                    (request_id, tenant_id, created_at, pages, duration_ms, input_tokens, output_tokens, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, principal.tenant_id, now, pages, duration_ms, input_tokens, output_tokens, total_tokens),
            )
            self._connection.commit()
        return self.usage(principal)

    def audit(self, principal: ApiPrincipal, *, limit: int = 20) -> list[dict[str, int | str]]:
        """Return recent operational metadata only; documents and OCR outputs are absent."""
        safe_limit = max(1, min(limit, 100))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT request_id, created_at, pages, duration_ms, input_tokens, output_tokens, total_tokens
                FROM usage_events
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (principal.tenant_id, safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def _prune_locked(self, now: str) -> None:
        cutoff = (datetime.fromisoformat(now) - timedelta(days=self.retention_days)).isoformat()
        self._connection.execute("DELETE FROM usage_events WHERE created_at < ?", (cutoff,))
        self._connection.execute("DELETE FROM idempotency_keys WHERE created_at < ?", (cutoff,))


def _next_month(month: str) -> str:
    year, month_number = (int(part) for part in month.split("-", maxsplit=1))
    if month_number == 12:
        return f"{year + 1}-01-01T00:00:00+00:00"
    return f"{year}-{month_number + 1:02d}-01T00:00:00+00:00"
