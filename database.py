"""
DermoScan - Supabase-backed helpers for custom user management.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in test environments
    def load_dotenv() -> bool:
        return False

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_ROLE", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
    or os.getenv("SUPABASE_ANON_KEY", "").strip()
)

try:
    from supabase import client as supabase_client
except ImportError:  # pragma: no cover - dependency may not be installed in some environments
    supabase_client = None


if SUPABASE_URL and SUPABASE_KEY and supabase_client:
    supabase = supabase_client.create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


class SupabaseDB:
    """Compatibility shim for the old SQLAlchemy-style initialization."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def init_app(self, app: Any) -> None:  # noqa: ARG002
        return None

    @property
    def session(self) -> Any:
        raise RuntimeError("Supabase persistence is not wired yet; use supabase.client directly.")


# Replace the old SQLAlchemy instance with a Supabase-backed wrapper.
db = SupabaseDB(supabase)


def _get_client() -> Any | None:
    return supabase


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _hash_password(password: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), b"dermoscan", 100_000).hex()


def _verify_password(password: str, password_hash: str) -> bool:
    return _hash_password(password) == password_hash


def _safe_supabase_single(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, dict) else None
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else None


def _safe_supabase_data(response: Any) -> list[dict[str, Any]]:
    if response is None:
        return []
    if isinstance(response, dict):
        data = response.get("data")
        return data if isinstance(data, list) else []
    data = getattr(response, "data", None)
    return data if isinstance(data, list) else []


def _coerce_user(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if not isinstance(row, dict):
        row = {
            "id": getattr(row, "id", None),
            "first_name": getattr(row, "first_name", ""),
            "last_name": getattr(row, "last_name", ""),
            "username": getattr(row, "username", ""),
            "password_hash": getattr(row, "password_hash", ""),
            "is_premium": getattr(row, "is_premium", False),
            "is_admin": getattr(row, "is_admin", False),
            "created_at": getattr(row, "created_at", None),
        }
    return {
        "id": row.get("id"),
        "first_name": row.get("first_name", ""),
        "last_name": row.get("last_name", ""),
        "username": row.get("username", ""),
        "password_hash": row.get("password_hash", ""),
        "is_premium": bool(row.get("is_premium", False)),
        "is_admin": bool(row.get("is_admin", False)),
        "created_at": row.get("created_at") or datetime.utcnow().isoformat(),
    }


def init_db() -> None:
    return None


def create_user(
    first_name: str,
    last_name: str,
    username: str,
    password: str,
    is_admin: bool = False,
    is_premium: bool = False,
) -> dict[str, Any] | None:
    normalized_username = _normalize_username(username)
    if not normalized_username or not password:
        return None

    client = _get_client()
    if client is None:
        raise RuntimeError("Supabase is not configured for user registration.")

    try:
        existing_rows = _safe_supabase_data(
            client.table("users").select("id").eq("username", normalized_username).execute()
        )
        if existing_rows:
            return None
    except Exception:
        existing_rows = []

    payload = {
        "id": str(uuid.uuid4()),
        "first_name": (first_name or "").strip(),
        "last_name": (last_name or "").strip(),
        "username": normalized_username,
        "password_hash": _hash_password(password),
        "is_premium": 1 if is_premium else 0,
        "is_admin": 1 if is_admin else 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        response = client.table("users").insert(payload).execute()
    except Exception as exc:
        try:
            duplicate_rows = _safe_supabase_data(
                client.table("users").select("id").eq("username", normalized_username).execute()
            )
            if duplicate_rows:
                return None
        except Exception:
            pass
        raise RuntimeError("Unable to create your account right now. Please try again later.") from exc

    data = _safe_supabase_single(response)
    if data is None:
        data = _safe_supabase_data(response)
        if isinstance(data, list) and data:
            data = data[0]
    if data is None:
        raise RuntimeError("Unable to create your account right now. Please try again later.")
    return _coerce_user(data)


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    normalized_username = _normalize_username(username)
    if not normalized_username or not password:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        row = _safe_supabase_single(
            client.table("users").select("*").eq("username", normalized_username).maybe_single().execute()
        )
    except Exception:
        return None

    if row is None:
        return None
    if not _verify_password(password, row.get("password_hash", "")):
        return None
    return _coerce_user(row)


def get_user_by_id(user_id: str | None) -> dict[str, Any] | None:
    if not user_id:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        row = _safe_supabase_single(client.table("users").select("*").eq("id", user_id).maybe_single().execute())
    except Exception:
        return None
    return _coerce_user(row)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    normalized_username = _normalize_username(username)
    if not normalized_username:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        row = _safe_supabase_single(
            client.table("users").select("*").eq("username", normalized_username).maybe_single().execute()
        )
    except Exception:
        return None
    return _coerce_user(row)


def update_user(user_id: str | None, updates: dict[str, Any]) -> bool:
    if not user_id or not updates:
        return False

    safe_updates: dict[str, Any] = {}
    for key, value in updates.items():
        if key == "password":
            safe_updates["password_hash"] = _hash_password(str(value))
        elif key == "username":
            safe_updates[key] = _normalize_username(str(value))
        elif key in {"first_name", "last_name"}:
            safe_updates[key] = str(value).strip()
        elif key in {"is_premium", "is_admin"}:
            safe_updates[key] = 1 if bool(value) else 0

    if not safe_updates:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        client.table("users").update(safe_updates).eq("id", user_id).execute()
    except Exception:
        return False
    return True


def list_users() -> list[dict[str, Any]]:
    client = _get_client()
    if client is None:
        return []

    try:
        rows = _safe_supabase_data(client.table("users").select("*").order("created_at", desc=True).execute())
    except Exception:
        return []
    return [_coerce_user(row) for row in rows if _coerce_user(row)]


def insert_scan_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    client = _get_client()
    if client is None:
        raise RuntimeError("Supabase is not configured for scan storage.")

    normalized_payload = dict(payload)
    normalized_payload.setdefault("created_at", datetime.utcnow().isoformat())

    if isinstance(normalized_payload.get("all_scores"), dict):
        normalized_payload["all_scores"] = json.dumps(normalized_payload["all_scores"])

    candidates = []
    candidates.append(normalized_payload)

    if "image_bytes" in normalized_payload:
        candidates.append({**normalized_payload, "image_data": normalized_payload["image_bytes"]})
    if "image_filename" in normalized_payload:
        candidates.append({**normalized_payload, "image_name": normalized_payload["image_filename"]})

    table_names = ["scan_history", "scans", "scan_histories", "skin_scans"]
    last_error: Exception | None = None

    for table_name in table_names:
        for attempt in candidates:
            try:
                response = client.table(table_name).insert(attempt).execute()
                data = _safe_supabase_single(response)
                if data is None:
                    data = _safe_supabase_data(response)
                    if isinstance(data, list) and data:
                        data = data[0]
                if data is not None:
                    return data
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        raise RuntimeError(str(last_error)) from last_error
    return None


@dataclass
class User:
    id: str | None = None
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    password_hash: str = ""
    is_premium: bool = False
    is_admin: bool = False
    created_at: Any = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def initials(self) -> str:
        return f"{self.first_name[0]}{self.last_name[0]}".upper() if self.first_name and self.last_name else ""


@dataclass
class ScanHistory:
    id: int | None = None
    user_id: int | None = None
    image_filename: str = ""
    predicted_class: str = ""
    confidence: float = 0.0
    all_scores: str | None = None
    created_at: Any = None

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 1)

    @property
    def badge_color(self) -> str:
        mapping = {
            "normal": "success",
            "acne": "warning",
            "eczema": "info",
            "psoriasis": "danger",
            "ringworm": "secondary",
        }
        return mapping.get(self.predicted_class.lower(), "primary")


@dataclass
class Feedback:
    id: int | None = None
    user_id: int | None = None
    message: str = ""
    rating: int = 5
    created_at: Any = None
