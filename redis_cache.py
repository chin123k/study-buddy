"""Redis-backed cache helpers for Streamlit session data."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis


def create_redis_client() -> Optional[redis.Redis]:
    """Create Redis client from environment, return None when disabled/unavailable."""
    enabled = os.getenv("REDIS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return None

    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def build_cache_key(session_id: str, namespace: str, item_key: str) -> str:
    return f"slide-explainer:{session_id}:{namespace}:{item_key}"


def get_json(client: Optional[redis.Redis], key: str) -> Optional[Any]:
    if client is None:
        return None

    try:
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def set_json(client: Optional[redis.Redis], key: str, value: Any, ttl_seconds: int) -> bool:
    if client is None:
        return False

    try:
        payload = json.dumps(value)
        client.setex(key, max(1, ttl_seconds), payload)
        return True
    except Exception:
        return False


def touch_key(client: Optional[redis.Redis], key: str, ttl_seconds: int) -> None:
    if client is None:
        return

    try:
        client.expire(key, max(1, ttl_seconds))
    except Exception:
        return
