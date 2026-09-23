"""Persistent keep-alive client for TypeSafe System One.

Every production ``_post_systemone`` call (no injected ``http_client``) shares
one ``http.client.HTTPSConnection``. A dead socket is closed and retried once;
the second failure raises so existing fail-safe paths stay unchanged.

Exact-repeat decisions are cached by content hash plus the serialized question
spec. TTL and the on/off switch live in ``jev.decision_cache`` (default on,
3600s, 512 LRU entries). No new environment variables.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import re
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

DEFAULT_CACHE_TTL_SECONDS = 3600.0
DEFAULT_CACHE_MAX_ENTRIES = 512
_HASH_RE = re.compile(r"hash=([0-9a-fA-F]{8,64})")

# Connection errors that mean the kept-alive socket is dead. Timeouts are not
# in this set: retrying them would double the caller's budget.
_STALE_TYPES = (
    ConnectionError,
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
)

ConnectionFactory = Callable[[float], Any]


@dataclass(frozen=True)
class DecisionCacheConfig:
    enabled: bool = True
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    max_entries: int = DEFAULT_CACHE_MAX_ENTRIES


_override: Optional[DecisionCacheConfig] = None
_CLIENT: Optional["PersistentJevClient"] = None
_CLIENT_LOCK = threading.Lock()


def decision_cache_key(body: Mapping[str, Any]) -> str:
    """Key of (content hash + serialized question spec).

    Hygiene ``hash=`` tokens are the content identity. A digest of the full
    state is appended so lexicon or prior-turn metadata cannot collide two
    different payloads that share one short hash. The question spec is the
    canonical JSON of model + questions.
    """
    state_blob = json.dumps(
        body.get("state"),
        sort_keys=True,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    hashes = _HASH_RE.findall(state_blob)
    content_hash = (
        "|".join(hashes)
        if hashes
        else hashlib.sha256(state_blob.encode("utf-8")).hexdigest()[:16]
    )
    state_digest = hashlib.sha256(state_blob.encode("utf-8")).hexdigest()
    question_spec = json.dumps(
        {"model": body.get("model"), "questions": body.get("questions")},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{content_hash}|{state_digest}\n{question_spec}"


def configure_decision_cache(
    *,
    enabled: bool = True,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
) -> None:
    """Override file config. Tests use this; production leaves it unset."""
    global _override
    _override = DecisionCacheConfig(
        enabled=bool(enabled),
        ttl_seconds=float(ttl_seconds),
        max_entries=max(1, int(max_entries)),
    )


def _settings() -> DecisionCacheConfig:
    if _override is not None:
        return _override
    try:
        from hermes_cli.config import load_config_readonly

        raw = load_config_readonly()
        block = raw.get("jev") if isinstance(raw, dict) else None
        cache = block.get("decision_cache") if isinstance(block, dict) else None
    except Exception:
        cache = None
    return _parse_cache_config(cache)


def _parse_cache_config(raw: Any) -> DecisionCacheConfig:
    if not isinstance(raw, dict):
        return DecisionCacheConfig()
    enabled = raw.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "off", "no", "disabled"}
    else:
        enabled = bool(enabled)
    try:
        ttl = float(raw.get("ttl_seconds", DEFAULT_CACHE_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = DEFAULT_CACHE_TTL_SECONDS
    try:
        max_entries = int(raw.get("max_entries", DEFAULT_CACHE_MAX_ENTRIES))
    except (TypeError, ValueError):
        max_entries = DEFAULT_CACHE_MAX_ENTRIES
    if ttl <= 0:
        enabled = False
    return DecisionCacheConfig(
        enabled=enabled,
        ttl_seconds=max(0.0, ttl),
        max_entries=max(1, max_entries),
    )


def _is_stale_connection(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return False
    return isinstance(exc, _STALE_TYPES)


def _env_proxy_set() -> bool:
    return any(
        os.environ.get(name)
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )


def _target() -> Tuple[str, int, str]:
    from agent.context_compressor_jev import TYPESAFE_SYSTEMONE_URL

    parts = urlsplit(TYPESAFE_SYSTEMONE_URL)
    host = parts.hostname or "api.typesafe.ai"
    port = parts.port or 443
    path = parts.path or "/v1/systemone"
    if parts.query:
        path = f"{path}?{parts.query}"
    return host, port, path


class _DecisionCache:
    def __init__(self) -> None:
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()

    def clear(self) -> None:
        self._data.clear()

    def get(self, key: str, now: float) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if now >= expires:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return copy.deepcopy(value)

    def put(self, key: str, value: Any, expires: float, max_entries: int) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (expires, copy.deepcopy(value))
        while len(self._data) > max_entries:
            self._data.popitem(last=False)


class PersistentJevClient:
    """One keep-alive HTTPS connection, shared by every System One call site."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: Any = None
        self._cache = _DecisionCache()
        self._factory: Optional[ConnectionFactory] = None
        self.opens = 0

    def set_connection_factory(self, factory: Optional[ConnectionFactory]) -> None:
        self._factory = factory

    def close(self) -> None:
        with self._lock:
            self._discard(self._conn)
            self._cache.clear()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _discard(self, conn: Any) -> None:
        if conn is None:
            return
        if self._conn is conn:
            self._conn = None
        close = getattr(conn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _open(self, timeout: float) -> Any:
        self.opens += 1
        if self._factory is not None:
            conn = self._factory(timeout)
        else:
            host, port, _path = _target()
            conn = http.client.HTTPSConnection(host, port, timeout=timeout)
        try:
            conn.timeout = timeout
        except Exception:
            pass
        return conn

    def _exchange(self, conn: Any, body: bytes, headers: Dict[str, str], timeout: float):
        try:
            conn.timeout = timeout
        except Exception:
            pass
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(timeout)
            except Exception:
                pass
        _host, _port, path = _target()
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        status = getattr(resp, "status", None)
        if status is None:
            status = getattr(resp, "status_code", 0)
        getheader = getattr(resp, "getheader", None)
        connection = ""
        if callable(getheader):
            connection = str(getheader("connection") or "")
        return int(status or 0), raw, connection.lower()

    def _exchange_bounded(self, conn: Any, body: bytes, headers: Dict[str, str], timeout: float):
        box: Dict[str, Any] = {}

        def runner() -> None:
            try:
                box["result"] = self._exchange(conn, body, headers, timeout)
            except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
                box["error"] = exc

        worker = threading.Thread(target=runner, name="jev-http", daemon=True)
        worker.start()
        worker.join(timeout if timeout and timeout > 0 else 0.001)
        if worker.is_alive():
            self._discard(conn)
            raise TimeoutError("jev timeout")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def post(
        self,
        body: Mapping[str, Any],
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> Tuple[Any, float, float]:
        cfg = _settings()
        key = decision_cache_key(body)
        key_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        cache_key = f"{key_id}\n{key}"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        with self._lock:
            now = time.monotonic()
            if cfg.enabled:
                hit = self._cache.get(cache_key, now)
                if hit is not None:
                    elapsed = (time.monotonic() - now) * 1000.0
                    return hit, elapsed, elapsed
            last: Optional[BaseException] = None
            for attempt in (0, 1):
                conn = self._conn
                if conn is None or attempt == 1:
                    conn = self._open(timeout_seconds)
                    self._conn = conn
                t0 = time.perf_counter()
                try:
                    status, raw, connection = self._exchange_bounded(
                        conn, payload, headers, timeout_seconds,
                    )
                    if "close" in connection:
                        self._discard(conn)
                    data = _interpret(status, raw)
                    ready = (time.perf_counter() - t0) * 1000.0
                    if cfg.enabled and cfg.ttl_seconds > 0:
                        self._cache.put(
                            cache_key,
                            data,
                            time.monotonic() + cfg.ttl_seconds,
                            cfg.max_entries,
                        )
                    return copy.deepcopy(data), ready, ready
                except Exception as exc:
                    self._discard(conn)
                    last = exc
                    if attempt == 0 and _is_stale_connection(exc):
                        continue
                    raise
            assert last is not None
            raise last


def _interpret(status: int, raw: bytes) -> Any:
    if status == 429:
        raise RuntimeError("jev rate limited (429)")
    if status >= 400:
        raise RuntimeError(f"jev http {status}")
    if not raw:
        raise RuntimeError("jev empty response")
    return json.loads(raw.decode("utf-8"))


def get_persistent_client() -> PersistentJevClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = PersistentJevClient()
        return _CLIENT


def set_connection_factory_for_tests(factory: Optional[ConnectionFactory]) -> None:
    get_persistent_client().set_connection_factory(factory)


def reset_jev_transport_for_tests() -> None:
    global _override, _CLIENT
    _override = None
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
        _CLIENT = None


def post_systemone(
    body: Mapping[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
) -> Tuple[Any, float, float]:
    """POST one System One request on the shared keep-alive connection."""
    return get_persistent_client().post(
        body, api_key=api_key, timeout_seconds=timeout_seconds,
    )
